import math

import numpy as np

DEFAULT_KS = (1, 3, 5, 10)


def metric_names(ks=DEFAULT_KS) -> list[str]:
    names = ["mrr@10"]
    for k in ks:
        names += [f"hit_rate@{k}", f"precision@{k}", f"recall@{k}", f"ndcg@{k}"]
    return names


def query_metrics(ranking: list[str], grades: dict[str, int], ks=DEFAULT_KS) -> dict[str, float]:
    relevant = {chunk_id for chunk_id, grade in grades.items() if grade > 0}
    first = next((rank for rank, chunk_id in enumerate(ranking, start=1) if chunk_id in relevant), None)
    values = {"mrr@10": 0.0 if first is None or first > 10 else 1.0 / first}
    ideal_grades = sorted(grades.values(), reverse=True)
    for k in ks:
        retrieved = ranking[:k]
        matched = sum(chunk_id in relevant for chunk_id in retrieved)
        dcg = sum(
            (2 ** grades.get(chunk_id, 0) - 1) / math.log2(rank + 1)
            for rank, chunk_id in enumerate(retrieved, start=1)
        )
        idcg = sum((2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(ideal_grades[:k], start=1))
        values[f"hit_rate@{k}"] = float(matched > 0)
        values[f"precision@{k}"] = matched / k
        values[f"recall@{k}"] = matched / len(relevant) if relevant else 0.0
        values[f"ndcg@{k}"] = dcg / idcg if idcg else 0.0
    return values


def evaluate_rankings(rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]], ks=DEFAULT_KS) -> dict[str, float]:
    if not rankings:
        raise ValueError("rankings must not be empty")
    rows = [query_metrics(ranking, qrels.get(query_id, {}), ks) for query_id, ranking in rankings.items()]
    return {name: float(np.mean([row[name] for row in rows])) for name in rows[0]}
