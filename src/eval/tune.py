import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from src.eval.metrics import query_metrics
from src.eval.runner import load_benchmark
from src.eval.spec import DEFAULT_FROZEN, resolve_configs
from src.io_utils import write_csv

REQUIRED_CONFIGS = ("C1", "C2", "C3-WS", "C3-RRF", "X2", "C4-WS")


def _choose(scores: dict, default):
    best = max(scores.values())
    tied = [value for value, score in scores.items() if math.isclose(score, best)]
    return min(tied, key=lambda value: abs(value - default))


def tune(spec, *, index_dir: Path, pipeline_factory, force: bool = False) -> Path:
    if spec.frozen_path.exists() and not force:
        raise FileExistsError(
            f"{spec.frozen_path} already exists; pass --force to re-tune (recorded as lock violation after a test run)"
        )
    queries, qrels, _ = load_benchmark(spec.bench_dir, "dev")
    base = resolve_configs(spec, DEFAULT_FROZEN)
    missing = [name for name in REQUIRED_CONFIGS if name not in base]
    if missing:
        raise ValueError(f"eval tune requires configs {missing} in {spec.path}")
    pipeline = pipeline_factory(index_dir)
    metric = spec.primary_metric
    rows: list[dict] = []

    def record(param: str, value, config) -> float:
        by_category = defaultdict(list)
        overall = []
        for query in queries:
            ranking = [item.chunk.chunk_id for item in pipeline.run(query["text"], config).results]
            score = query_metrics(ranking, qrels.get(query["query_id"], {}), spec.ks)[metric]
            overall.append(score)
            by_category[query["category"]].append(score)
        rows.append({"param": param, "value": value, "category": "all", "metric": metric, "score": float(np.mean(overall))})
        for category, scores in sorted(by_category.items()):
            rows.append({"param": param, "value": value, "category": category, "metric": metric, "score": float(np.mean(scores))})
        return float(np.mean(overall))

    grid = spec.tune_grid
    single = {name: record("single", name, base[name].pipeline) for name in ("C1", "C2")}
    best_single = max(single, key=lambda name: (single[name], name))
    alpha = _choose({value: record("alpha", value, base["C3-WS"].pipeline.replace(alpha=value)) for value in grid["alpha"]}, 0.5)
    rrf_k = _choose({value: record("rrf_k", value, base["C3-RRF"].pipeline.replace(rrf_k=value)) for value in grid["rrf_k"]}, 60)
    beta = _choose(
        {value: record("adaptive_beta", value, base["X2"].pipeline.replace(alpha=alpha, adaptive_beta=value)) for value in grid["adaptive_beta"]},
        0.3,
    )
    rerank_base = base["C4-WS"].pipeline
    rerank_n = _choose(
        {
            value: record("rerank_n", value, rerank_base.replace(alpha=alpha, rerank_n=value, context_k=min(rerank_base.context_k, value)))
            for value in grid["rerank_n"]
        },
        30,
    )
    frozen = {
        "alpha": float(alpha),
        "rrf_k": int(rrf_k),
        "adaptive_beta": float(beta),
        "rerank_n": int(rerank_n),
        "best_single": best_single,
        "primary_metric": metric,
        "tuned_on": "dev",
        "index_version": pipeline.index.version,
        "dev_queries": len(queries),
        "tuned_at": datetime.now(timezone.utc).isoformat(),
    }
    write_csv(spec.bench_dir / "tune_results.csv", rows, ["param", "value", "category", "metric", "score"])
    spec.frozen_path.write_text(yaml.safe_dump(frozen, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return spec.frozen_path
