import json
import sqlite3
import uuid
from pathlib import Path

from src.models import RagConfig


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    username TEXT PRIMARY KEY,
                    password_hash TEXT NOT NULL,
                    salt TEXT NOT NULL,
                    role TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS documents (
                    doc_id TEXT PRIMARY KEY,
                    metadata_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS corpus_versions (
                    version_id TEXT PRIMARY KEY,
                    metadata_json TEXT NOT NULL,
                    active INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS rag_configs (
                    name TEXT PRIMARY KEY,
                    config_json TEXT NOT NULL,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS chat_sessions (
                    session_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY,
                    message_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS experiment_runs (
                    run_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    result_path TEXT,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                """
            )

    def save_rag_config(self, name: str, config: RagConfig) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO rag_configs(name, config_json) VALUES (?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    config_json = excluded.config_json,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (name, json.dumps(config.to_dict(), ensure_ascii=False)),
            )

    def list_rag_configs(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT name, config_json, updated_at FROM rag_configs ORDER BY name"
            ).fetchall()
        return [
            {"name": row["name"], "config": json.loads(row["config_json"]), "updated_at": row["updated_at"]}
            for row in rows
        ]

    def upsert_user(self, username: str, password_hash: str, salt: str, role: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users(username, password_hash, salt, role) VALUES (?, ?, ?, ?)
                ON CONFLICT(username) DO UPDATE SET
                    password_hash = excluded.password_hash,
                    salt = excluded.salt,
                    role = excluded.role
                """,
                (username, password_hash, salt, role),
            )

    def get_user(self, username: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT username, password_hash, salt, role FROM users WHERE username = ?",
                (username,),
            ).fetchone()
        return dict(row) if row else None

    def save_document(self, record: dict) -> None:
        payload = {key: value for key, value in record.items() if key not in {"doc_id", "status"}}
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents(doc_id, metadata_json, status) VALUES (?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    metadata_json = excluded.metadata_json,
                    status = excluded.status
                """,
                (record["doc_id"], json.dumps(payload, ensure_ascii=False), record["status"]),
            )

    def list_documents(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT doc_id, metadata_json, status, created_at FROM documents ORDER BY created_at"
            ).fetchall()
        return [
            {
                "doc_id": row["doc_id"],
                **json.loads(row["metadata_json"]),
                "status": row["status"],
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def save_corpus_version(self, record: dict) -> None:
        version_id = record["version_id"]
        payload = {key: value for key, value in record.items() if key != "version_id"}
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO corpus_versions(version_id, metadata_json) VALUES (?, ?)",
                (version_id, json.dumps(payload, ensure_ascii=False)),
            )

    def get_active_corpus(self) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT version_id, metadata_json FROM corpus_versions WHERE active = 1 LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return {"version_id": row["version_id"], **json.loads(row["metadata_json"])}

    def set_active_corpus(self, version_id: str) -> None:
        with self._connect() as connection:
            exists = connection.execute(
                "SELECT 1 FROM corpus_versions WHERE version_id = ?", (version_id,)
            ).fetchone()
            if exists is None:
                raise KeyError(f"Unknown corpus version: {version_id}")
            connection.execute("UPDATE corpus_versions SET active = 0")
            connection.execute(
                "UPDATE corpus_versions SET active = 1 WHERE version_id = ?", (version_id,)
            )

    def save_message(self, record: dict) -> str:
        message_id = record.get("message_id", uuid.uuid4().hex)
        payload = {key: value for key, value in record.items() if key not in {"message_id", "session_id"}}
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO messages(message_id, session_id, payload_json) VALUES (?, ?, ?)",
                (message_id, record["session_id"], json.dumps(payload, ensure_ascii=False)),
            )
        return message_id

    def save_feedback(self, record: dict) -> str:
        feedback_id = record.get("feedback_id", uuid.uuid4().hex)
        payload = {key: value for key, value in record.items() if key not in {"feedback_id", "message_id"}}
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO feedback(feedback_id, message_id, payload_json) VALUES (?, ?, ?)",
                (feedback_id, record["message_id"], json.dumps(payload, ensure_ascii=False)),
            )
        return feedback_id
