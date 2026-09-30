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
                    corpus_version TEXT NOT NULL DEFAULT '',
                    scope_json TEXT NOT NULL DEFAULT '[]',
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
                """
            )

    def start_chat_session(self, session_id: str, username: str, corpus_version: str, doc_ids: tuple[str, ...] = ()) -> None:
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO chat_sessions(session_id, username, corpus_version, scope_json) VALUES (?, ?, ?, ?)",
                (session_id, username, corpus_version, json.dumps(sorted(set(doc_ids)))),
            )

    def list_chat_sessions(self, username: str, corpus_version: str, doc_ids: tuple[str, ...] = ()) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT session_id, created_at FROM chat_sessions WHERE username = ? AND corpus_version = ? AND scope_json = ? ORDER BY created_at DESC, rowid DESC",
                (username, corpus_version, json.dumps(sorted(set(doc_ids)))),
            ).fetchall()
        return [dict(row) for row in rows]

    def chat_turns(self, session_id: str, username: str) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                """SELECT m.message_id, m.payload_json FROM messages AS m
                   JOIN chat_sessions AS s ON s.session_id = m.session_id
                   WHERE s.session_id = ? AND s.username = ?
                   ORDER BY m.created_at, m.rowid""",
                (session_id, username),
            ).fetchall()
        return [{"message_id": row["message_id"], **json.loads(row["payload_json"])} for row in rows]

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

    def publish_corpus(self, documents: list[dict], record: dict) -> None:
        """Commit document registry, corpus metadata, and active pointer together."""
        version_id = record["version_id"]
        payload = {key: value for key, value in record.items() if key != "version_id"}
        with self._connect() as connection:
            existing = connection.execute(
                "SELECT metadata_json FROM corpus_versions WHERE version_id = ?", (version_id,)
            ).fetchone()
            if existing and json.loads(existing[0]) != payload:
                raise ValueError(f"Corpus version metadata differs: {version_id}")
            for document in documents:
                document_payload = {key: value for key, value in document.items() if key != "doc_id"}
                connection.execute(
                    """INSERT INTO documents(doc_id, metadata_json, status) VALUES (?, ?, 'processed')
                       ON CONFLICT(doc_id) DO UPDATE SET metadata_json = excluded.metadata_json, status = 'processed'""",
                    (document["doc_id"], json.dumps(document_payload, ensure_ascii=False)),
                )
            if not existing:
                connection.execute(
                    "INSERT INTO corpus_versions(version_id, metadata_json) VALUES (?, ?)",
                    (version_id, json.dumps(payload, ensure_ascii=False)),
                )
            connection.execute("UPDATE corpus_versions SET active = 0")
            connection.execute("UPDATE corpus_versions SET active = 1 WHERE version_id = ?", (version_id,))

    def get_active_corpus(self) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT version_id, metadata_json FROM corpus_versions WHERE active = 1 LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        return {"version_id": row["version_id"], **json.loads(row["metadata_json"])}

    def save_message(self, record: dict) -> str:
        message_id = record.get("message_id", uuid.uuid4().hex)
        payload = {key: value for key, value in record.items() if key not in {"message_id", "session_id", "username"}}
        with self._connect() as connection:
            cursor = connection.execute(
                """INSERT INTO messages(message_id, session_id, payload_json)
                   SELECT ?, session_id, ? FROM chat_sessions
                   WHERE session_id = ? AND username = ?""",
                (message_id, json.dumps(payload, ensure_ascii=False), record["session_id"], record["username"]),
            )
            if cursor.rowcount != 1:
                raise PermissionError("Chat session does not belong to user")
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
