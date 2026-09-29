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


def test_chat_sessions_are_split_by_corpus_and_scope(tmp_path, monkeypatch):
    _, db = _db(tmp_path, monkeypatch)
    db.start_chat_session("s1", "alice", "v1", ())
    db.start_chat_session("s2", "alice", "v1", ("d2", "d1"))
    db.start_chat_session("s3", "alice", "v2", ())
    db.save_message({"session_id": "s2", "username": "alice", "query": "q1", "answer": "a1"})
    db.save_message({"session_id": "s2", "username": "alice", "query": "q2", "answer": "a2"})
    assert [item["session_id"] for item in db.list_chat_sessions("alice", "v1", ())] == ["s1"]
    assert [item["session_id"] for item in db.list_chat_sessions("alice", "v1", ("d1", "d2"))] == ["s2"]
    assert [turn["query"] for turn in db.chat_turns("s2", "alice")] == ["q1", "q2"]
    assert db.chat_turns("s2", "mallory") == []


def test_initialize_migrates_old_chat_sessions_table(tmp_path, monkeypatch):
    settings = load_settings(tmp_path)
    with sqlite3.connect(settings.db_path) as connection:
        connection.execute("CREATE TABLE chat_sessions (session_id TEXT PRIMARY KEY, username TEXT NOT NULL, created_at TEXT)")
    db = Database(settings.db_path)
    db.initialize()
    db.start_chat_session("s1", "alice", "v1", ("d1",))
    assert db.list_chat_sessions("alice", "v1", ("d1",))[0]["session_id"] == "s1"


def test_document_source_path_survives_metadata(tmp_path, monkeypatch):
    _, db = _db(tmp_path, monkeypatch)
    db.save_document({"doc_id": "d1", "filename": "a.pdf", "course": "AI101", "status": "processed"})
    db.set_document_source("d1", "/data/raw/uploads/d1/a.pdf")
    assert db.list_documents()[0]["source_path"] == "/data/raw/uploads/d1/a.pdf"
    assert db.list_documents()[0]["course"] == "AI101"
