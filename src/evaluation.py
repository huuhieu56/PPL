import math
import random
import statistics


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
    metrics["mrr@10"] = statistics.mean(reciprocal_ranks)
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
        metrics[f"hit_rate@{k}"] = statistics.mean(hits)
        metrics[f"precision@{k}"] = statistics.mean(precisions)
        metrics[f"recall@{k}"] = statistics.mean(recalls)
        metrics[f"ndcg@{k}"] = statistics.mean(ndcgs)
    return metrics


def paired_bootstrap_ci(
    left: list[float], right: list[float], seed: int = 42, samples: int = 2000, groups: list[str] | None = None
) -> tuple[float, float]:
    if len(left) != len(right) or not left:
        raise ValueError("paired samples must have the same non-zero length")
    if samples <= 0:
        raise ValueError("samples must be positive")
    differences = [float(a) - float(b) for a, b in zip(left, right)]
    rng = random.Random(seed)
    if groups is None:
        means = sorted(statistics.mean(rng.choices(differences, k=len(differences))) for _ in range(samples))
    else:
        if len(groups) != len(differences):
            raise ValueError("groups must match paired samples")
        clustered: dict[str, list[float]] = {}
        for group, difference in zip(groups, differences):
            clustered.setdefault(group, []).append(difference)
        clusters = list(clustered.values())
        means = sorted(statistics.mean(value for cluster in rng.choices(clusters, k=len(clusters)) for value in cluster) for _ in range(samples))

    def percentile(p: float) -> float:
        position = (len(means) - 1) * p
        low = math.floor(position)
        return means[low] + (means[math.ceil(position)] - means[low]) * (position - low)

    return percentile(0.025), percentile(0.975)
