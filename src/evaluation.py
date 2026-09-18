import csv
import json
import math
from pathlib import Path

import numpy as np


def evaluate_rankings(
    rankings: dict[str, list[str]],
    qrels: dict[str, dict[str, int]],
    ks: tuple[int, ...] = (1, 3, 5, 10),
) -> dict[str, float]:
    if not rankings:
        raise ValueError("rankings must not be empty")
    metrics: dict[str, float] = {}
    query_ids = list(rankings)
    reciprocal_ranks = []
    for query_id in query_ids:
        relevant = {key for key, value in qrels.get(query_id, {}).items() if value > 0}
        first = next(
            (rank for rank, chunk_id in enumerate(rankings[query_id], start=1) if chunk_id in relevant),
            None,
        )
        reciprocal_ranks.append(0.0 if first is None or first > 10 else 1 / first)
    metrics["mrr@10"] = float(np.mean(reciprocal_ranks))
    for k in ks:
        hits, precisions, recalls, ndcgs = [], [], [], []
        for query_id in query_ids:
            retrieved = rankings[query_id][:k]
            grades = qrels.get(query_id, {})
            relevant = {key for key, value in grades.items() if value > 0}
            matched = sum(chunk_id in relevant for chunk_id in retrieved)
            hits.append(float(matched > 0))
            precisions.append(matched / k)
            recalls.append(matched / len(relevant) if relevant else 0.0)
            dcg = sum((2 ** grades.get(chunk_id, 0) - 1) / math.log2(rank + 1) for rank, chunk_id in enumerate(retrieved, start=1))
            ideal = sorted(grades.values(), reverse=True)[:k]
            idcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal, start=1))
            ndcgs.append(dcg / idcg if idcg else 0.0)
        metrics[f"hit_rate@{k}"] = float(np.mean(hits))
        metrics[f"precision@{k}"] = float(np.mean(precisions))
        metrics[f"recall@{k}"] = float(np.mean(recalls))
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs))
    return metrics


def paired_bootstrap_ci(
    left: list[float], right: list[float], seed: int = 42, samples: int = 2000
) -> tuple[float, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired samples must have the same non-zero length")
    differences = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)
    rng = np.random.default_rng(seed)
    means = [float(np.mean(rng.choice(differences, size=len(differences), replace=True))) for _ in range(samples)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_evaluation(
    output_dir: Path | str, metrics: dict[str, float], per_query: list[dict]
) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    (target / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    if per_query:
        with (target / "per_query.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(per_query[0]))
            writer.writeheader()
            writer.writerows(per_query)
