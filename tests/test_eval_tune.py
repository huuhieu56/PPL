import pytest
import yaml

from src.config import load_settings
from src.eval.spec import load_spec
from src.eval.tune import tune
from src.io_utils import read_csv, write_jsonl
from src.models import RetrievedChunk, StageScores
from src.pipeline import PipelineResult
from tests.fakes import make_chunk

CHUNKS = {chunk_id: make_chunk(chunk_id, chunk_id) for chunk_id in ("good", "other")}


class AlphaSensitivePipeline:
    class index:
        version = "fake-v1"

    def run(self, query, config, use_cache=True):
        good_first = (config.fusion == "weighted" and config.alpha >= 0.7) or (config.sparse and not config.dense)
        order = ["good", "other"] if good_first else ["other", "good"]
        return PipelineResult(
            [RetrievedChunk(CHUNKS[c], StageScores(rank=r)) for r, c in enumerate(order, start=1)],
            {"total": 1.0},
        )


@pytest.fixture
def spec(tmp_path, monkeypatch):
    monkeypatch.setenv("PPL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PPL_RUNS_DIR", str(tmp_path / "runs"))
    bench = tmp_path / "bench"
    write_jsonl(bench / "queries.jsonl", [
        {"query_id": "q1", "text": "a", "category": "exact", "split": "dev"},
        {"query_id": "q2", "text": "b", "category": "concept", "split": "dev"},
        {"query_id": "q3", "text": "c", "category": "concept", "split": "test"},
    ])
    write_jsonl(bench / "qrels.jsonl", [
        {"query_id": q, "chunk_id": "good", "relevance": 2} for q in ("q1", "q2", "q3")
    ])
    raw = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
    raw.update(bench_dir=str(bench), tune={"alpha": [0.5, 0.7, 0.9], "rrf_k": [60], "adaptive_beta": [0.3], "rerank_n": [10, 30]})
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return load_spec(path, load_settings(tmp_path))


def test_tune_selects_best_values_and_writes_curves(spec):
    frozen_path = tune(spec, index_dir=spec.bench_dir, pipeline_factory=lambda _: AlphaSensitivePipeline())
    frozen = yaml.safe_load(frozen_path.read_text(encoding="utf-8"))
    assert frozen["alpha"] == 0.7
    assert frozen["rrf_k"] == 60 and frozen["adaptive_beta"] == 0.3
    assert frozen["rerank_n"] == 30
    assert frozen["best_single"] == "C1"
    assert frozen["tuned_on"] == "dev" and frozen["dev_queries"] == 2
    rows = read_csv(spec.bench_dir / "tune_results.csv")
    alpha_all = [row for row in rows if row["param"] == "alpha" and row["category"] == "all"]
    assert [float(row["value"]) for row in alpha_all] == [0.5, 0.7, 0.9]
    assert any(row["category"] == "concept" for row in rows)


def test_tune_refuses_to_overwrite_without_force(spec):
    tune(spec, index_dir=spec.bench_dir, pipeline_factory=lambda _: AlphaSensitivePipeline())
    with pytest.raises(FileExistsError, match="--force"):
        tune(spec, index_dir=spec.bench_dir, pipeline_factory=lambda _: AlphaSensitivePipeline())
    tune(spec, index_dir=spec.bench_dir, pipeline_factory=lambda _: AlphaSensitivePipeline(), force=True)
