import hashlib

import numpy as np

from src.bench.generate import CATEGORIES
from src.io_utils import read_csv, write_csv
from src.text import normalize_text

REVIEW_COLUMNS = ["query_id", "category", "text", "source_text", "action", "new_text", "new_category"]


def export_review(drafts: list[dict], index, path) -> None:
    rows = [
        {
            "query_id": draft["query_id"],
            "category": draft["category"],
            "text": draft["text"],
            "source_text": "\n---\n".join(index.chunks[chunk_id].body for chunk_id in draft["source_chunk_ids"]),
            "action": "keep",
            "new_text": "",
            "new_category": "",
        }
        for draft in drafts
    ]
    write_csv(path, rows, REVIEW_COLUMNS)


def _checked_category(value: str, where: str) -> str:
    if value not in CATEGORIES:
        raise ValueError(f"Unknown category '{value}' for {where}; expected one of {CATEGORIES}")
    return value


def _human_queries(path) -> list[dict]:
    queries = []
    for row in read_csv(path):
        text = normalize_text(row["text"])
        if not text:
            continue
        category = _checked_category(row["category"].strip(), f"human question '{text}'")
        query_id = row.get("query_id", "").strip() or "h-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
        queries.append(
            {"query_id": query_id, "text": text, "category": category, "origin": "human", "split": None,
             "source_chunk_ids": [], "evidence": [], "generator": "human"}
        )
    return queries


def import_review(review_path, drafts: list[dict], encoder, human_path=None, duplicate_threshold: float = 0.92):
    by_id = {draft["query_id"]: draft for draft in drafts}
    reviewed = []
    for row in read_csv(review_path):
        query_id = row["query_id"]
        action = row["action"].strip().lower() or "keep"
        if action == "drop":
            continue
        if action not in ("keep", "edit"):
            raise ValueError(f"Unknown action '{action}' for {query_id}; use keep, edit or drop")
        query = dict(by_id[query_id])
        if action == "edit":
            new_text = normalize_text(row["new_text"])
            if not new_text:
                raise ValueError(f"Edited question {query_id} has empty new_text")
            query["text"] = new_text
            query["category"] = _checked_category(row["new_category"].strip() or query["category"], query_id)
        reviewed.append(query)
    if human_path:
        reviewed.extend(_human_queries(human_path))
    if not reviewed:
        return [], []
    vectors = np.asarray(encoder.encode([query["text"] for query in reviewed], normalize_embeddings=True))
    kept_rows: list[int] = []
    queries, duplicates = [], []
    for row, query in enumerate(reviewed):
        if kept_rows:
            similarities = vectors[kept_rows] @ vectors[row]
            best = int(np.argmax(similarities))
            if similarities[best] >= duplicate_threshold:
                duplicates.append({**query, "duplicate_of": reviewed[kept_rows[best]]["query_id"]})
                continue
        kept_rows.append(row)
        queries.append(query)
    return queries, duplicates
