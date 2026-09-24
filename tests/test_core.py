import csv
import json
from types import SimpleNamespace

import pytest
import yaml

from src.config import load_settings
from src.evaluation import evaluate_rankings
from src.experiments import run_experiment
from src.models import Chunk, RagConfig
from src.rag import answer_question
from src.retrieval import adaptive_alpha, fuse_weighted, minmax_scores
from src.storage import Database


def test_foundation_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-persisted")
    settings = load_settings(tmp_path)
    db = Database(settings.db_path)
    db.initialize()
    db.save_rag_config("demo", RagConfig(method="rrf", use_reranker=False))

    saved = db.list_rag_configs()
    assert saved[0]["name"] == "demo"
    assert saved[0]["config"]["method"] == "rrf"
    assert "must-not-be-persisted" not in json.dumps(saved)


def test_retrieval_fusion_and_metrics(monkeypatch):
    chunks = {
        "c1": Chunk("c1", "d1", "Slide", "AI101", "slide", 1, ("Mã môn",), "Mã môn AI101", "Mã môn AI101"),
        "c2": Chunk("c2", "d1", "Slide", "AI101", "slide", 2, ("Khái niệm",), "Giải thích học máy", "Giải thích học máy"),
    }
    assert minmax_scores({"c1": 5.0, "c2": 5.0}) == {"c1": 1.0, "c2": 1.0}

    exact_alpha, exact_signals = adaptive_alpha(
        "Mã môn AI101 là gì?", alpha0=0.5, beta=0.3, idf={"ai101": 1.0}
    )
    semantic_alpha, _ = adaptive_alpha(
        "Giải thích học máy", alpha0=0.5, beta=0.3, idf={"học": 0.1, "máy": 0.1}
    )
    assert exact_signals["code"] == 1.0
    assert exact_alpha > semantic_alpha

    fused = fuse_weighted(
        bm25_scores={"c1": 8.0, "c2": 1.0},
        dense_scores={"c1": 0.6, "c2": 0.5},
        chunks=chunks,
        alpha=0.7,
    )
    assert fused[0].chunk.chunk_id == "c1"

    metrics = evaluate_rankings(
        rankings={"q1": ["c1", "c2"]},
        qrels={"q1": {"c1": 2}},
        ks=(1, 10),
    )
    assert metrics["hit_rate@1"] == 1.0
    assert metrics["mrr@10"] == 1.0

    monkeypatch.setattr("src.rag.retrieve", lambda query, index, config: fused)

    class FakeCompletions:
        def __init__(self):
            self.calls = 0

        def create(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="Theo tài liệu [1] và [99]."))],
                usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
            )

    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    answer = answer_question(
        "Mã môn AI101 là gì?",
        index=object(),
        config=RagConfig(method="weighted", use_reranker=False),
        client=client,
        model="test-model",
    )
    assert "[1]" in answer.text and "[99]" not in answer.text
    assert answer.citations[0]["chunk_id"] == "c1"

    refused = answer_question(
        "Câu hỏi ngoài tài liệu",
        index=object(),
        config=RagConfig(method="weighted", refusal_threshold=2.0, use_reranker=False),
        client=client,
        model="test-model",
    )
    assert refused.refused is True
    assert completions.calls == 1


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
