import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from src.config import load_yaml
from src.evaluation import evaluate_rankings, paired_bootstrap_ci
from src.models import RagConfig
from src.reranking import rerank
from src.retrieval import RetrievalIndex, retrieve


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def select_queries(path: Path, split: str) -> list[dict]:
    if split not in {"dev", "test"}:
        raise ValueError("Evaluation split must be dev or test")
    queries = _read_jsonl(path)
    if any(row.get("split") not in {"train", "dev", "test"} for row in queries):
        raise ValueError("Every query must have a train/dev/test split")
    partitions = {}
    for row in queries:
        keys = [("text", " ".join(row["text"].casefold().split()))]
        if row.get("group_id"):
            keys.append(("group", row["group_id"]))
        for key in keys:
            if key in partitions and partitions[key] != row["split"]:
                raise ValueError(f"Query/group leaks across splits: {key}")
            partitions[key] = row["split"]
    selected = [row for row in queries if row["split"] == split]
    if not selected:
        raise ValueError(f"No queries for split={split}")
    return selected


def validate_config(config: dict, require_index: bool = False) -> None:
    required = ("queries", "qrels", "corpus_version", "experiments")
    missing = [key for key in required if not config.get(key)]
    if missing:
        raise ValueError(f"Missing required config values: {', '.join(missing)}")
    for key in ("queries", "qrels"):
        if not Path(config[key]).is_file():
            raise FileNotFoundError(f"{key} file not found: {config[key]}")
    if not isinstance(config["experiments"], dict) or not config["experiments"]:
        raise ValueError("experiments must be a non-empty mapping")
    allowed = {f"E{number}" for number in range(8)}
    unknown = set(config["experiments"]) - allowed
    if unknown:
        raise ValueError(f"Unknown experiment IDs: {sorted(unknown)}")
    if require_index:
        index_dir = Path(config.get("index_dir", ""))
        if not index_dir.is_dir() or not (index_dir / "chunks.jsonl").is_file():
            raise FileNotFoundError(f"index_dir not found or incomplete: {index_dir}")
        if index_dir.name != config["corpus_version"]:
            raise ValueError("index_dir does not match corpus_version")
        chunk_ids = {row["chunk_id"] for row in _read_jsonl(index_dir / "chunks.jsonl")}
        queries = _read_jsonl(Path(config["queries"]))
        query_ids = {row["query_id"] for row in queries}
        if len(query_ids) != len(queries):
            raise ValueError("queries contain duplicate query_id")
        positive_query_ids = set()
        for row in _read_jsonl(Path(config["qrels"])):
            if row["query_id"] not in query_ids:
                raise ValueError(f"qrels contain unknown query: {row['query_id']}")
            if row["chunk_id"] not in chunk_ids:
                raise ValueError(f"qrels contain unknown chunk: {row['chunk_id']}")
            if int(row["relevance"]) > 0:
                positive_query_ids.add(row["query_id"])
        if query_ids - positive_query_ids:
            raise ValueError(f"queries missing positive qrels: {sorted(query_ids - positive_query_ids)}")


@lru_cache(maxsize=2)
def _load_index(path: str) -> RetrievalIndex:
    return RetrievalIndex.load(path)


def experiment_rag_config(experiment_id: str, config: dict) -> RagConfig:
    experiment = config["experiments"][experiment_id]
    retrieval_config = config.get("retrieval", {})
    return RagConfig(
        method=experiment["method"],
        top_l=int(retrieval_config.get("top_l", 100)),
        rerank_n=int(retrieval_config.get("rerank_n", 20)),
        context_k=int(retrieval_config.get("context_k", 5)),
        context_parent_words=int(retrieval_config.get("context_parent_words", 900)),
        alpha=float(experiment.get("alpha", 0.5)),
        adaptive_beta=float(experiment.get("adaptive_beta", 0.3)),
        rrf_k=int(experiment.get("rrf_k", 60)),
        use_reranker=bool(experiment.get("use_reranker", False)),
        refusal_threshold=float(retrieval_config.get("refusal_threshold", 0)),
    )


def execute_query(experiment_id: str, query: dict, config: dict) -> dict:
    rag_config = experiment_rag_config(experiment_id, config)
    started = time.perf_counter()
    results = retrieve(query["text"], _load_index(str(config["index_dir"])), rag_config)
    rerank_ms = 0.0
    if rag_config.use_reranker:
        results, rerank_ms = rerank(
            query["text"],
            results,
            rag_config.rerank_n,
            config.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
        )
    return {
        "ranked_chunk_ids": [result.chunk.chunk_id for result in results],
        "latency_ms": (time.perf_counter() - started) * 1000,
        "rerank_ms": rerank_ms,
    }


def _run_id(config: dict) -> str:
    digest = hashlib.sha256(
        json.dumps(config, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:8]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{timestamp}-{digest}"


def _checkpoint_rows(path: Path) -> list[dict]:
    return _read_jsonl(path) if path.exists() else []


def _write_outputs(run_dir: Path, rows: list[dict], qrels_rows: list[dict], *, complete: bool, queries: list[dict]) -> None:
    fieldnames = ["experiment_id", "query_id", "ranked_chunk_ids", "latency_ms", "rerank_ms"]
    with (run_dir / "per_query.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row.get(key, "") for key in fieldnames},
                    "ranked_chunk_ids": json.dumps(row["ranked_chunk_ids"]),
                }
            )
    if not complete:
        return
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_rows:
        qrels.setdefault(row["query_id"], {})[row["chunk_id"]] = int(row["relevance"])
    metrics = {}
    for experiment_id in sorted({row["experiment_id"] for row in rows}):
        rankings = {
            row["query_id"]: row["ranked_chunk_ids"]
            for row in rows
            if row["experiment_id"] == experiment_id
        }
        metrics[experiment_id] = evaluate_rankings(rankings, qrels) if rankings else {}
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if "E1" in metrics:
        rankings_by_method = {method: {row["query_id"]: row["ranked_chunk_ids"] for row in rows if row["experiment_id"] == method} for method in metrics}
        identifiers = [query["query_id"] for query in queries]
        grouped = all(query.get("group_id") for query in queries)
        groups = [query["group_id"] for query in queries] if grouped else None
        scores = {method: [evaluate_rankings({identifier: rankings[identifier]}, qrels)["ndcg@10"] for identifier in identifiers] for method, rankings in rankings_by_method.items()}
        comparisons = {method: {"baseline": "E1", "delta_ndcg@10": metrics[method]["ndcg@10"] - metrics["E1"]["ndcg@10"], "paired_ci95": paired_bootstrap_ci(scores[method], scores["E1"], groups=groups), "bootstrap_unit": "group_id" if grouped else "query", "groups": len(set(groups)) if groups else len(identifiers), "interpretation": "Exploratory comparisons; not corrected for multiple testing"} for method in metrics if method != "E1"}
        (run_dir / "paired_comparisons.json").write_text(json.dumps(comparisons, ensure_ascii=False, indent=2), encoding="utf-8")


def run_experiment(config_path: Path | str, resume_run_id: str | None = None, split: str = "dev") -> Path:
    config = load_yaml(config_path)
    validate_config(config, require_index=True)
    queries = select_queries(Path(config["queries"]), split)
    config["evaluated_split"] = split
    runs_dir = Path(config.get("runs_dir", "runs"))
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = resume_run_id or _run_id(config)
    run_dir = runs_dir / run_id
    fingerprint = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [Path(config["queries"]), Path(config["qrels"]), *sorted(Path(config["index_dir"]).glob("*"))]
        if path.is_file()
    }
    if resume_run_id:
        frozen = run_dir / "config.json"
        hashes = run_dir / "input_hashes.json"
        if not frozen.is_file() or json.loads(frozen.read_text()) != config:
            raise ValueError("Cannot resume: frozen config changed or missing")
        if not hashes.is_file() or json.loads(hashes.read_text()) != fingerprint:
            raise ValueError("Cannot resume: frozen benchmark/index changed or missing")
    run_dir.mkdir(exist_ok=bool(resume_run_id))
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "input_hashes.json").write_text(json.dumps(fingerprint, sort_keys=True), encoding="utf-8")
    status_path = run_dir / "status.json"
    status_path.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    checkpoint_path = run_dir / "checkpoint.jsonl"
    rows = _checkpoint_rows(checkpoint_path)
    completed = {(row["experiment_id"], row["query_id"]) for row in rows}
    qrels_rows = _read_jsonl(Path(config["qrels"]))
    errors: list[dict] = []
    try:
        with checkpoint_path.open("a", encoding="utf-8") as checkpoint:
            for experiment_id in config["experiments"]:
                for query in queries:
                    key = (experiment_id, query["query_id"])
                    if key in completed:
                        continue
                    try:
                        result = execute_query(experiment_id, query, config)
                    except Exception as error:
                        errors.append(
                            {
                                "experiment_id": experiment_id,
                                "query_id": query["query_id"],
                                "error": str(error),
                            }
                        )
                        continue
                    row = {"experiment_id": experiment_id, "query_id": query["query_id"], **result}
                    checkpoint.write(json.dumps(row, ensure_ascii=False) + "\n")
                    checkpoint.flush()
                    rows.append(row)
        expected = {(experiment_id, query["query_id"]) for experiment_id in config["experiments"] for query in queries}
        observed = {(row["experiment_id"], row["query_id"]) for row in rows}
        complete = not errors and observed == expected
        _write_outputs(run_dir, rows, qrels_rows, complete=complete, queries=queries)
        if errors:
            with (run_dir / "errors.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["experiment_id", "query_id", "error"])
                writer.writeheader()
                writer.writerows(errors)
        status_path.write_text(json.dumps({"status": "completed" if complete else "partial", "completed": len(observed), "expected": len(expected)}), encoding="utf-8")
    except BaseException:
        status_path.write_text(json.dumps({"status": "interrupted"}), encoding="utf-8")
        raise
    return run_dir
