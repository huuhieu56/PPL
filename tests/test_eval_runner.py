import json

import pytest
import yaml

from src.config import load_settings
from src.eval.runner import load_benchmark, run_evaluation
from src.eval.spec import DEFAULT_FROZEN, load_spec
from src.io_utils import read_csv, write_jsonl
from src.models import RetrievedChunk, StageScores
from src.pipeline import PipelineResult
from tests.fakes import make_chunk

CHUNKS = {chunk_id: make_chunk(chunk_id, chunk_id) for chunk_id in ("c1", "c2", "c3")}


class FakeIndex:
    version = "fake-v1"
    meta = {"embedding_model": "fake", "chunk_count": 3}
    chunk_list = list(CHUNKS.values())


class FakePipeline:
    def __init__(self, fail_once_on=None):
        self.index = FakeIndex()
        self.calls = []
        self.fail_once_on = fail_once_on

    def run(self, query, config, use_cache=True):
        self.calls.append((query, config.fusion, use_cache))
        if query == self.fail_once_on:
            self.fail_once_on = None
            raise KeyboardInterrupt
        order = ["c1", "c2", "c3"] if config.sparse and not config.dense else ["c2", "c1", "c3"]
        results = [
            RetrievedChunk(CHUNKS[chunk_id], StageScores(rank=rank, sparse_score=1.0 / rank if config.sparse else None))
            for rank, chunk_id in enumerate(order, start=1)
        ]
        return PipelineResult(results, {"sparse": 1.0, "dense": 2.0, "fusion": 0.1, "rerank": 0.0, "total": 3.1})


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setenv("PPL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PPL_RUNS_DIR", str(tmp_path / "runs"))
    settings = load_settings(tmp_path)
    bench = tmp_path / "bench"
    write_jsonl(bench / "queries.jsonl", [
        {"query_id": "q1", "text": "q1", "category": "exact", "origin": "llm", "split": "test"},
        {"query_id": "q2", "text": "q2", "category": "concept", "origin": "human", "split": "test"},
        {"query_id": "q3", "text": "q3", "category": "concept", "origin": "llm", "split": "dev"},
    ])
    write_jsonl(bench / "qrels.jsonl", [
        {"query_id": "q1", "chunk_id": "c1", "relevance": 2},
        {"query_id": "q2", "chunk_id": "c2", "relevance": 1},
        {"query_id": "q3", "chunk_id": "c3", "relevance": 1},
    ])
    config = {
        "bench_dir": str(bench), "runs_dir": "${RUNS_DIR}", "latency": {"warmup": 1},
        "configs": {
            "C1": {"sparse": True, "dense": False, "fusion": "none", "rerank": False},
            "C3-WS": {"fusion": "weighted", "alpha": "frozen", "rerank": False},
        },
        "comparisons": {}, "error_analysis": {"target": "C1"},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return load_spec(path, settings), bench


def _run(spec, split, pipeline, **kwargs):
    return run_evaluation(spec, split, index_dir=spec.bench_dir, pipeline_factory=lambda _: pipeline, **kwargs)


def test_load_benchmark_filters_split(setup):
    spec, bench = setup
    queries, qrels, evidence = load_benchmark(bench, "test")
    assert [query["query_id"] for query in queries] == ["q1", "q2"]
    assert qrels["q2"] == {"c2": 1} and evidence == []
    with pytest.raises(ValueError, match="bench split"):
        load_benchmark(bench, "holdout")


def test_dev_run_writes_metrics_by_group_and_latency(setup):
    spec, _ = setup
    pipeline = FakePipeline()
    run_dir = _run(spec, "dev", pipeline, frozen=DEFAULT_FROZEN)
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_queries"] == 1
    assert metrics["overall"]["C1"]["mrr@10"] == pytest.approx(1 / 3)
    assert metrics["by_category"]["C1"]["concept"]["n"] == 1
    rows = read_csv(run_dir / "metrics_per_query.csv")
    assert {row["config"] for row in rows} == {"C1", "C3-WS"}
    latency = json.loads((run_dir / "latency.json").read_text(encoding="utf-8"))
    assert latency["configs"]["C1"]["total"]["p95"] == pytest.approx(3.1)
    assert "python" in latency["hardware"]
    assert any(use_cache is False for _, _, use_cache in pipeline.calls)
    first = json.loads((run_dir / "per_query.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert first["results"][0][:3] == ["c1", 1.0, None]
    assert json.loads((run_dir / "status.json").read_text())["status"] == "completed"


def test_test_split_requires_frozen_params(setup):
    spec, _ = setup
    with pytest.raises(FileNotFoundError, match="eval tune"):
        _run(spec, "test", FakePipeline(), frozen=None)
    assert not spec.runs_dir.exists() or not any(spec.runs_dir.iterdir())


def test_test_run_records_lock_violation(setup):
    spec, bench = setup
    from src.bench.manifest import write_manifest

    write_manifest(bench, "fake-v1", 42, 0.3)
    spec.frozen_path.write_text("alpha: 0.5\n", encoding="utf-8")
    first = _run(spec, "test", FakePipeline(), frozen={"alpha": 0.5}, latency=False)
    assert json.loads((first / "config.json").read_text(encoding="utf-8"))["lock"]["lock_violation"] is False
    spec.frozen_path.write_text("alpha: 0.9\n", encoding="utf-8")
    second = _run(spec, "test", FakePipeline(), frozen={"alpha": 0.9}, latency=False)
    assert json.loads((second / "config.json").read_text(encoding="utf-8"))["lock"]["lock_violation"] is True


def test_run_resumes_without_repeating_completed_queries(setup):
    spec, _ = setup
    pipeline = FakePipeline(fail_once_on="q2")
    with pytest.raises(KeyboardInterrupt):
        _run(spec, "all", pipeline, frozen=DEFAULT_FROZEN, only=["C1"], latency=False)
    run_dir = next(spec.runs_dir.iterdir())
    assert json.loads((run_dir / "status.json").read_text())["status"] == "interrupted"
    _run(spec, "all", pipeline, frozen=DEFAULT_FROZEN, only=["C1"], latency=False, resume_run_id=run_dir.name)
    queries_run = [query for query, _, _ in pipeline.calls]
    assert queries_run.count("q1") == 1
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["n_queries"] == 3


def test_only_rejects_unknown_config(setup):
    spec, _ = setup
    with pytest.raises(ValueError, match="C9"):
        _run(spec, "dev", FakePipeline(), frozen=DEFAULT_FROZEN, only=["C9"])
