import pytest
import yaml

from src.config import load_settings
from src.eval.compare import compare_run
from src.eval.spec import load_spec
from src.io_utils import read_csv, write_csv


@pytest.fixture
def spec(tmp_path, monkeypatch):
    monkeypatch.setenv("PPL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PPL_RUNS_DIR", str(tmp_path / "runs"))
    raw = {
        "bench_dir": str(tmp_path / "bench"),
        "configs": {
            "C1": {"sparse": True, "dense": False, "fusion": "none", "rerank": False},
            "C2": {"sparse": False, "dense": True, "fusion": "none", "rerank": False},
            "C3-WS": {"fusion": "weighted", "rerank": False},
        },
        "comparisons": {"RQ2": [["C3-WS", "best_single"], ["C3-WS", "C1"]]},
        "error_analysis": {"target": "C3-WS"},
    }
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return load_spec(path, load_settings(tmp_path))


def _per_query(run_dir, configs):
    rows = []
    for config, scores in configs.items():
        for number, score in enumerate(scores):
            rows.append({"config": config, "query_id": f"q{number}", "category": "exact" if number % 2 else "concept",
                         "origin": "llm", "mrr@10": score, "ndcg@10": score, "recall@5": score})
    write_csv(run_dir / "metrics_per_query.csv", rows)


def test_compare_run_uses_best_single_and_holm(tmp_path, spec):
    run_dir = tmp_path / "run"
    _per_query(run_dir, {"C1": [0.0] * 20, "C2": [0.5] * 20, "C3-WS": [1.0] * 20})
    rows = compare_run(run_dir, spec, {"best_single": "C2"}, samples=500)
    overall = [row for row in rows if row["category"] == "all" and row["metric"] == "mrr@10"]
    assert [(row["system"], row["baseline"]) for row in overall] == [("C3-WS", "C2"), ("C3-WS", "C1")]
    assert overall[0]["diff"] == pytest.approx(0.5)
    assert overall[0]["ci_low"] == pytest.approx(0.5)
    assert all(row["p_holm"] != "" and row["p_holm"] >= row["p_value"] for row in overall)
    by_category = [row for row in rows if row["category"] == "exact" and row["metric"] == "mrr@10"]
    assert by_category[0]["n"] == 10 and by_category[0]["p_holm"] == ""
    saved = read_csv(run_dir / "comparisons.csv")
    assert len(saved) == len(rows) == 2 * 3 * 3


def test_compare_reports_missing_configs(tmp_path, spec):
    run_dir = tmp_path / "run"
    _per_query(run_dir, {"C1": [0.0] * 4, "C3-WS": [1.0] * 4})
    with pytest.raises(ValueError, match="C2"):
        compare_run(run_dir, spec, {"best_single": "C2"}, samples=100)
    with pytest.raises(ValueError, match="best_single"):
        compare_run(run_dir, spec, None, samples=100)
