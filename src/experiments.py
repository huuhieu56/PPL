import csv
import hashlib
import json
import time
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

from src.config import load_yaml
from src.evaluation import evaluate_rankings
from src.models import RagConfig
from src.reranking import rerank
from src.retrieval import RetrievalIndex, retrieve


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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
    if require_index and not Path(config.get("index_dir", "")).is_dir():
        raise FileNotFoundError(f"index_dir not found: {config.get('index_dir', '')}")


@lru_cache(maxsize=2)
def _load_index(path: str) -> RetrievalIndex:
    return RetrievalIndex.load(path)


def execute_query(experiment_id: str, query: dict, config: dict) -> dict:
    experiment = config["experiments"][experiment_id]
    retrieval_config = config.get("retrieval", {})
    rag_config = RagConfig(
        method=experiment["method"],
        top_l=int(retrieval_config.get("top_l", 100)),
        rerank_n=int(retrieval_config.get("rerank_n", 20)),
        context_k=int(retrieval_config.get("context_k", 5)),
        alpha=float(experiment.get("alpha", 0.5)),
        rrf_k=int(experiment.get("rrf_k", 60)),
        use_reranker=bool(experiment.get("use_reranker", False)),
    )
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


def _write_outputs(run_dir: Path, rows: list[dict], qrels_rows: list[dict]) -> None:
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


def run_experiment(config_path: Path | str, resume_run_id: str | None = None) -> Path:
    config = load_yaml(config_path)
    validate_config(config)
    runs_dir = Path(config.get("runs_dir", "runs"))
    runs_dir.mkdir(parents=True, exist_ok=True)
    run_id = resume_run_id or _run_id(config)
    run_dir = runs_dir / run_id
    run_dir.mkdir(exist_ok=bool(resume_run_id))
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    status_path = run_dir / "status.json"
    status_path.write_text(json.dumps({"status": "running"}), encoding="utf-8")
    checkpoint_path = run_dir / "checkpoint.jsonl"
    rows = _checkpoint_rows(checkpoint_path)
    completed = {(row["experiment_id"], row["query_id"]) for row in rows}
    queries = _read_jsonl(Path(config["queries"]))
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
        _write_outputs(run_dir, rows, qrels_rows)
        if errors:
            with (run_dir / "errors.csv").open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=["experiment_id", "query_id", "error"])
                writer.writeheader()
                writer.writerows(errors)
        status_path.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    except BaseException:
        status_path.write_text(json.dumps({"status": "interrupted"}), encoding="utf-8")
        raise
    return run_dir
