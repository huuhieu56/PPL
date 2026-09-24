import hashlib
import json
import sqlite3
from pathlib import Path


def query_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RetrievalCache:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS first_stage (
                key TEXT PRIMARY KEY,
                results_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rerank (
                model TEXT NOT NULL,
                query_key TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                score REAL NOT NULL,
                PRIMARY KEY (model, query_key, chunk_id)
            );
            """
        )
        self._connection.commit()

    @staticmethod
    def first_stage_key(index_version: str, branch: str, variant: str, query: str, top_l: int) -> str:
        return "|".join([index_version, branch, variant, query_key(query), str(top_l)])

    def get_first_stage(self, key: str) -> list[tuple[str, float]] | None:
        row = self._connection.execute(
            "SELECT results_json FROM first_stage WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return [(chunk_id, float(score)) for chunk_id, score in json.loads(row[0])]

    def put_first_stage(self, key: str, results: list[tuple[str, float]]) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO first_stage(key, results_json) VALUES (?, ?)",
            (key, json.dumps([[chunk_id, score] for chunk_id, score in results])),
        )
        self._connection.commit()

    def get_rerank(self, model: str, query: str, chunk_ids: list[str]) -> dict[str, float]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self._connection.execute(
            f"SELECT chunk_id, score FROM rerank WHERE model = ? AND query_key = ? "
            f"AND chunk_id IN ({placeholders})",
            (model, query_key(query), *chunk_ids),
        ).fetchall()
        return {chunk_id: float(score) for chunk_id, score in rows}

    def put_rerank(self, model: str, query: str, scores: dict[str, float]) -> None:
        key = query_key(query)
        self._connection.executemany(
            "INSERT OR REPLACE INTO rerank(model, query_key, chunk_id, score) VALUES (?, ?, ?, ?)",
            [(model, key, chunk_id, float(score)) for chunk_id, score in scores.items()],
        )
        self._connection.commit()
