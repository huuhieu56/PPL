from pathlib import Path

import pytest
import yaml

from src.config import load_settings
from src.eval.spec import DEFAULT_FROZEN, load_spec, read_frozen, resolve_configs


@pytest.fixture
def settings(tmp_path, monkeypatch):
    monkeypatch.setenv("PPL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PPL_RUNS_DIR", str(tmp_path / "runs"))
    return load_settings(tmp_path)


def _write(tmp_path, **overrides):
    raw = yaml.safe_load(Path("configs/experiment.yaml").read_text(encoding="utf-8"))
    raw.update(overrides)
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw, allow_unicode=True), encoding="utf-8")
    return path


def test_repository_config_loads_and_expands_placeholders(tmp_path, settings):
    spec = load_spec(_write(tmp_path), settings)
    assert spec.bench_dir == settings.data_dir / "benchmark"
    assert spec.runs_dir == settings.runs_dir
    assert spec.primary_metric == "mrr@10"
    assert spec.frozen_path == settings.data_dir / "benchmark" / "frozen_params.yaml"
    assert ("C4-WS", "C3-WS") in spec.comparisons["RQ3"]
    assert spec.tune_grid["alpha"][:3] == [0.0, 0.1, 0.2]


def test_resolve_requires_frozen_values(tmp_path, settings):
    spec = load_spec(_write(tmp_path), settings)
    with pytest.raises(FileNotFoundError, match="eval tune"):
        resolve_configs(spec, None)
    entries = resolve_configs(spec, {**DEFAULT_FROZEN, "alpha": 0.7, "rerank_n": 20})
    assert entries["C3-WS"].pipeline.alpha == 0.7
    assert entries["C4-WS"].pipeline.rerank_n == 20
    assert entries["X5-N50"].pipeline.rerank_n == 50
    assert entries["X3-pyvi-C1"].pipeline.tokenizer == "pyvi"
    assert entries["C1"].pipeline.top_l == 100 and entries["C1"].index_dir is None


def test_config_without_frozen_values_resolves_without_tuning(tmp_path, settings):
    path = _write(tmp_path, configs={"C1": {"sparse": True, "dense": False, "fusion": "none", "rerank": False}},
                  comparisons={}, error_analysis={"target": "C1"})
    assert resolve_configs(load_spec(path, settings), None)["C1"].pipeline.fusion == "none"


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"configs": {"C1": {"method": "bm25"}}}, "unknown keys"),
        ({"primary_metric": "map@10"}, "Unknown metric"),
        ({"comparisons": {"RQ9": [["C1", "C7"]]}}, "C7"),
        ({"error_analysis": {"target": "C9"}}, "C9"),
        ({"configs": {"C1": {"fusion": "none", "sparse": True, "dense": False, "top_l": "frozen"}}}, "top_l"),
    ],
)
def test_invalid_specs_are_rejected(tmp_path, settings, overrides, message):
    base = {"comparisons": {}, "error_analysis": {"target": "C1"}}
    if "configs" not in overrides:
        base["configs"] = {"C1": {"sparse": True, "dense": False, "fusion": "none", "rerank": False}}
    with pytest.raises(ValueError, match=message):
        spec = load_spec(_write(tmp_path, **{**base, **overrides}), settings)
        resolve_configs(spec, DEFAULT_FROZEN)


def test_per_config_index_dir_and_remap(tmp_path, settings):
    path = _write(
        tmp_path,
        configs={"X4": {"fusion": "weighted", "rerank": False, "index_dir": "${DATA_DIR}/indexes/v2", "qrels": "remap"}},
        comparisons={},
        error_analysis={"target": "X4"},
    )
    entry = resolve_configs(load_spec(path, settings), None)["X4"]
    assert entry.index_dir == settings.data_dir / "indexes" / "v2"
    assert entry.remap_qrels is True


def test_read_frozen(tmp_path, settings):
    spec = load_spec(_write(tmp_path), settings)
    assert read_frozen(spec) is None
    spec.frozen_path.parent.mkdir(parents=True)
    spec.frozen_path.write_text("alpha: 0.6\nbest_single: C2\n", encoding="utf-8")
    assert read_frozen(spec) == {"alpha": 0.6, "best_single": "C2"}
