import json
import sqlite3

from src.config import load_settings
from src.models import PipelineConfig
from src.storage import Database


def _db(tmp_path, monkeypatch):
    monkeypatch.delenv("PPL_DATA_DIR", raising=False)
    monkeypatch.delenv("PPL_RUNS_DIR", raising=False)
    settings = load_settings(tmp_path)
    db = Database(settings.db_path)
    db.initialize()
    return settings, db


def test_pipeline_config_round_trip_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-persisted")
    _, db = _db(tmp_path, monkeypatch)
    db.save_rag_config("hybrid", PipelineConfig(fusion="rrf", rerank=False))
    configs, invalid = db.load_pipeline_configs()
    assert configs["hybrid"].fusion == "rrf" and invalid == []
    assert "must-not-be-persisted" not in json.dumps(db.list_rag_configs())


def test_legacy_rag_config_is_reported_invalid(tmp_path, monkeypatch):
    settings, db = _db(tmp_path, monkeypatch)
    with sqlite3.connect(settings.db_path) as connection:
        connection.execute(
            "INSERT INTO rag_configs(name, config_json) VALUES (?, ?)",
            ("old", json.dumps({"method": "adaptive", "use_reranker": True})),
        )
    configs, invalid = db.load_pipeline_configs()
    assert configs == {} and invalid == ["old"]


def test_settings_honor_data_dir_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PPL_DATA_DIR", str(tmp_path / "drive" / "data"))
    monkeypatch.setenv("PPL_RUNS_DIR", str(tmp_path / "drive" / "runs"))
    settings = load_settings(tmp_path / "repo")
    data_dir = (tmp_path / "drive" / "data").resolve()
    assert settings.db_path == data_dir / "app.db"
    assert settings.indexes_dir == data_dir / "indexes"
    assert settings.cache_path == data_dir / "cache" / "retrieval.sqlite"
    assert settings.runs_dir.is_dir()


def test_saving_same_corpus_version_twice_is_idempotent(tmp_path, monkeypatch):
    _, db = _db(tmp_path, monkeypatch)
    record = {"version_id": "v1", "manifest_hash": "h", "chunks_path": "p", "chunk_count": 1}
    db.save_corpus_version(record)
    db.save_corpus_version(record)
    db.set_active_corpus("v1")
    assert db.get_active_corpus()["version_id"] == "v1"
