import random

import pytest
from ranx import Qrels, Run, evaluate

from src.eval.metrics import evaluate_rankings, metric_names, query_metrics


def test_query_metrics_on_hand_example():
    values = query_metrics(["x", "b", "a", "z"], {"a": 2, "b": 1, "z": 0}, ks=(1, 3))
    assert values["mrr@10"] == 0.5
    assert values["hit_rate@1"] == 0.0 and values["hit_rate@3"] == 1.0
    assert values["precision@3"] == pytest.approx(2 / 3)
    assert values["recall@3"] == 1.0
    assert list(values) == metric_names((1, 3))


def test_query_without_relevant_chunks_scores_zero():
    values = query_metrics(["a"], {}, ks=(1,))
    assert values == {"mrr@10": 0.0, "hit_rate@1": 0.0, "precision@1": 0.0, "recall@1": 0.0, "ndcg@1": 0.0}


def test_evaluate_rankings_rejects_empty_input():
    with pytest.raises(ValueError):
        evaluate_rankings({}, {})


def test_metrics_match_ranx_on_random_runs():
    rng = random.Random(7)
    documents = [f"d{i}" for i in range(30)]
    qrels, rankings = {}, {}
    for number in range(25):
        query = f"q{number}"
        qrels[query] = {doc: rng.choice([0, 1, 2]) for doc in rng.sample(documents, 6)}
        if not any(qrels[query].values()):
            qrels[query][documents[0]] = 1
        rankings[query] = rng.sample(documents, 15)
    ours = evaluate_rankings(rankings, qrels, ks=(1, 3, 5, 10))
    reference = evaluate(
        Qrels({q: {d: g for d, g in grades.items() if g > 0} for q, grades in qrels.items()}),
        Run({q: {d: float(len(ids) - i) for i, d in enumerate(ids)} for q, ids in rankings.items()}),
        ["mrr@10", "hit_rate@5", "precision@5", "recall@5", "ndcg_burges@10", "ndcg_burges@3"],
    )
    assert ours["mrr@10"] == pytest.approx(reference["mrr@10"])
    assert ours["hit_rate@5"] == pytest.approx(reference["hit_rate@5"])
    assert ours["precision@5"] == pytest.approx(reference["precision@5"])
    assert ours["recall@5"] == pytest.approx(reference["recall@5"])
    assert ours["ndcg@10"] == pytest.approx(reference["ndcg_burges@10"])
    assert ours["ndcg@3"] == pytest.approx(reference["ndcg_burges@3"])
