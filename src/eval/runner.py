import hashlib
import json
import platform
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.bench.manifest import check_test_lock
from src.bench.remap import remap_qrels
from src.cache import RetrievalCache
from src.eval.metrics import query_metrics
from src.eval.spec import resolve_configs
from src.index import RetrievalIndex
from src.io_utils import read_jsonl, write_csv
from src.pipeline import RetrievalPipeline

RESULT_FIELDS = ("chunk_id", "sparse_score", "sparse_rank", "dense_score", "dense_rank", "fusion_score", "rerank_score")
STAGES = ("sparse", "dense", "fusion", "rerank", "total")


def load_benchmark(bench_dir, split: str):
    directory = Path(bench_dir)
    queries = read_jsonl(directory / "queries.jsonl")
    if split != "all":
        queries = [query for query in queries if query.get("split") == split]
    if not queries:
        raise ValueError(f"No queries with split={split} in {directory}; run `python -m src.cli bench split` first")
    qrels: dict[str, dict[str, int]] = {}
    for row in read_jsonl(directory / "qrels.jsonl"):
        qrels.setdefault(row["query_id"], {})[row["chunk_id"]] = int(row["relevance"])
    evidence_path = directory / "evidence.jsonl"
    return queries, qrels, read_jsonl(evidence_path) if evidence_path.exists() else []


def default_pipeline_factory(settings):
    cache = RetrievalCache(settings.cache_path)
    return lambda index_dir: RetrievalPipeline(RetrievalIndex.load(index_dir), cache=cache)


def hardware_info() -> dict:
    info = {"platform": platform.platform(), "processor": platform.processor(), "python": platform.python_version()}
    try:
        import torch
    except ImportError:
        return info
    info["torch"] = torch.__version__
    info["cuda_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return info


def _as_qrels(rows: list[dict]) -> dict[str, dict[str, int]]:
    qrels: dict[str, dict[str, int]] = {}
    for row in rows:
        qrels.setdefault(row["query_id"], {})[row["chunk_id"]] = int(row["relevance"])
    return qrels


def _group_means(per_query: list[dict], names: list[str], key: str | None) -> dict:
    groups = defaultdict(list)
    for row in per_query:
        groups[(row["config"], row[key] if key else None)].append(row)
    summary: dict = {}
    for (config, group), rows in sorted(groups.items(), key=lambda item: (item[0][0], str(item[0][1]))):
        values = {name: float(np.mean([row[name] for row in rows])) for name in names}
        if key:
            summary.setdefault(config, {})[group] = {**values, "n": len(rows)}
        else:
            summary[config] = values
    return summary


def write_metrics(run_dir, rows, queries, qrels_by_config, ks) -> dict:
    info = {query["query_id"]: query for query in queries}
    per_query = []
    for row in rows:
        if row["config"] not in qrels_by_config or row["query_id"] not in info:
            continue
        query = info[row["query_id"]]
        ranking = [item[0] for item in row["results"]]
        per_query.append(
            {
                "config": row["config"],
                "query_id": row["query_id"],
                "category": query["category"],
                "origin": query.get("origin", "llm"),
                **query_metrics(ranking, qrels_by_config[row["config"]].get(row["query_id"], {}), ks),
            }
        )
    names = [name for name in (per_query[0] if per_query else {}) if name not in ("config", "query_id", "category", "origin")]
    summary = {
        "overall": _group_means(per_query, names, None),
        "by_category": _group_means(per_query, names, "category"),
        "by_origin": _group_means(per_query, names, "origin"),
        "n_queries": len(queries),
    }
    target = Path(run_dir)
    (target / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    write_csv(target / "metrics_per_query.csv", per_query)
    return summary


def measure_latency(entries, pipeline_for, queries, warmup: int) -> dict:
    report = {"hardware": hardware_info(), "warmup": warmup, "configs": {}}
    for name, entry in entries.items():
        pipeline = pipeline_for(entry)
        for query in queries[:warmup]:
            pipeline.run(query["text"], entry.pipeline, use_cache=False)
        samples = {stage: [] for stage in STAGES}
        for query in queries:
            timings = pipeline.run(query["text"], entry.pipeline, use_cache=False).timings_ms
            for stage in STAGES:
                samples[stage].append(float(timings.get(stage, 0.0)))
        report["configs"][name] = {
            stage: {
                "mean": float(np.mean(values)),
                "p50": float(np.percentile(values, 50)),
                "p95": float(np.percentile(values, 95)),
            }
            for stage, values in samples.items()
        }
    return report


def _write_status(run_dir: Path, status: str) -> None:
    (run_dir / "status.json").write_text(json.dumps({"status": status}), encoding="utf-8")


def run_evaluation(
    spec,
    split: str,
    *,
    index_dir: Path,
    pipeline_factory,
    frozen: dict | None,
    resume_run_id: str | None = None,
    only: list[str] | None = None,
    latency: bool = True,
) -> Path:
    queries, base_qrels, evidence = load_benchmark(spec.bench_dir, split)
    lock = check_test_lock(spec.bench_dir, spec.frozen_path) if split == "test" else None
    if lock and lock["lock_violation"]:
        print("CẢNH BÁO: frozen_params.yaml đã thay đổi sau lần chạy test đầu tiên (lock_violation).", file=sys.stderr)
    entries = resolve_configs(spec, frozen)
    if only:
        unknown = [name for name in only if name not in entries]
        if unknown:
            raise ValueError(f"Unknown configs in --only: {unknown}")
        entries = {name: entry for name, entry in entries.items() if name in only}

    pipelines: dict[str, object] = {}

    def pipeline_for(entry):
        key = str(entry.index_dir or index_dir)
        if key not in pipelines:
            pipelines[key] = pipeline_factory(Path(key))
        return pipelines[key]

    qrels_by_config = {
        name: _as_qrels(remap_qrels(evidence, pipeline_for(entry).index.chunk_list)) if entry.remap_qrels else base_qrels
        for name, entry in entries.items()
    }
    payload = json.dumps({"spec": spec.raw, "split": split, "frozen": frozen}, sort_keys=True, default=str)
    run_id = resume_run_id or (
        f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{split}-{hashlib.sha256(payload.encode()).hexdigest()[:8]}"
    )
    run_dir = spec.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=bool(resume_run_id))
    config_record = {
        "split": split,
        "spec_path": str(spec.path),
        "spec": spec.raw,
        "frozen": frozen,
        "lock": lock,
        "configs": {
            name: {
                **entry.pipeline.to_dict(),
                "index_dir": str(entry.index_dir or index_dir),
                "index_version": pipeline_for(entry).index.version,
                "remap_qrels": entry.remap_qrels,
            }
            for name, entry in entries.items()
        },
        "indexes": {key: pipeline.index.meta for key, pipeline in pipelines.items()},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(config_record, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    qrels_used = {"default": base_qrels, **{name: qrels_by_config[name] for name, entry in entries.items() if entry.remap_qrels}}
    (run_dir / "qrels_used.json").write_text(json.dumps(qrels_used, ensure_ascii=False), encoding="utf-8")
    _write_status(run_dir, "running")

    checkpoint = run_dir / "per_query.jsonl"
    rows = read_jsonl(checkpoint) if checkpoint.exists() else []
    done = {(row["config"], row["query_id"]) for row in rows}
    errors = []
    try:
        with checkpoint.open("a", encoding="utf-8") as stream:
            for name, entry in entries.items():
                pipeline = pipeline_for(entry)
                for query in queries:
                    if (name, query["query_id"]) in done:
                        continue
                    try:
                        result = pipeline.run(query["text"], entry.pipeline)
                    except Exception as error:
                        errors.append({"config": name, "query_id": query["query_id"], "error": str(error)})
                        continue
                    row = {
                        "config": name,
                        "query_id": query["query_id"],
                        "alpha_used": result.alpha_used,
                        "timings_ms": result.timings_ms,
                        "results": [
                            [item.chunk.chunk_id, *[getattr(item.scores, field) for field in RESULT_FIELDS[1:]]]
                            for item in result.results
                        ],
                    }
                    stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                    stream.flush()
                    rows.append(row)
        write_metrics(run_dir, rows, queries, qrels_by_config, spec.ks)
        if errors:
            write_csv(run_dir / "errors.csv", errors, ["config", "query_id", "error"])
        if latency:
            report = measure_latency(entries, pipeline_for, queries, spec.warmup)
            (run_dir / "latency.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        _write_status(run_dir, "completed")
    except BaseException:
        _write_status(run_dir, "interrupted")
        raise
    return run_dir
