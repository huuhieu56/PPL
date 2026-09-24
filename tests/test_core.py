import csv
import json

import pytest
import yaml

from src.evaluation import evaluate_rankings
from src.experiments import run_experiment


def test_ranking_metrics():
    metrics = evaluate_rankings(
        rankings={"q1": ["c1", "c2"]},
        qrels={"q1": {"c1": 2}},
        ks=(1, 10),
    )
    assert metrics["hit_rate@1"] == 1.0
    assert metrics["mrr@10"] == 1.0


def test_experiment_run_resumes_without_repeating_completed_queries(tmp_path, monkeypatch):
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        "\n".join(json.dumps({"query_id": query, "text": query}) for query in ["q1", "q2"]),
        encoding="utf-8",
    )
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(
        "\n".join(
            json.dumps({"query_id": query, "chunk_id": "c1", "relevance": 2})
            for query in ["q1", "q2"]
        ),
        encoding="utf-8",
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "runs_dir": str(tmp_path / "runs"),
                "queries": str(queries),
                "qrels": str(qrels),
                "corpus_version": "fixture",
                "experiments": {"E0": {"method": "bm25", "use_reranker": False}},
            }
        ),
        encoding="utf-8",
    )

    calls = []
    interrupted = {"value": False}

    def fake_execute_query(experiment_id, query, config):
        calls.append(query["query_id"])
        if query["query_id"] == "q2" and not interrupted["value"]:
            interrupted["value"] = True
            raise KeyboardInterrupt
        return {"ranked_chunk_ids": ["c1"], "latency_ms": 1.0}

    monkeypatch.setattr("src.experiments.execute_query", fake_execute_query)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(config_path)

    run_dir = next((tmp_path / "runs").iterdir())
    run_experiment(config_path, resume_run_id=run_dir.name)
    with (run_dir / "per_query.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    assert calls.count("q1") == 1
    assert {row["query_id"] for row in rows} == {"q1", "q2"}
