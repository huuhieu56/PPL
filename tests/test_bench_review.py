import pytest

from src.bench.review import export_review, import_review
from src.io_utils import read_csv, write_csv
from tests.fakes import FakeEncoder, make_chunk


class TinyIndex:
    chunks = {"c1": make_chunk("c1", "Khóa chính"), "c2": make_chunk("c2", "Khóa ngoại")}


DRAFTS = [
    {"query_id": "q1", "text": "Khóa chính là gì?", "category": "concept", "origin": "llm", "split": None,
     "source_chunk_ids": ["c1"], "evidence": [], "generator": "m"},
    {"query_id": "q2", "text": "Khóa ngoại dùng làm gì?", "category": "concept", "origin": "llm", "split": None,
     "source_chunk_ids": ["c2"], "evidence": [], "generator": "m"},
    {"query_id": "q3", "text": "Câu hỏi tệ", "category": "exact", "origin": "llm", "split": None,
     "source_chunk_ids": ["c1"], "evidence": [], "generator": "m"},
]


def test_export_review_lists_sources_and_default_action(tmp_path):
    export_review(DRAFTS, TinyIndex(), tmp_path / "review.csv")
    rows = read_csv(tmp_path / "review.csv")
    assert [row["action"] for row in rows] == ["keep", "keep", "keep"]
    assert rows[1]["source_text"] == "Khóa ngoại"


def test_import_review_applies_actions_adds_humans_and_dedups(tmp_path):
    export_review(DRAFTS, TinyIndex(), tmp_path / "review.csv")
    rows = read_csv(tmp_path / "review.csv")
    rows[1].update(action="edit", new_text="Vai trò của khóa ngoại trong lược đồ?", new_category="paraphrase")
    rows[2].update(action="drop")
    write_csv(tmp_path / "review.csv", rows)
    write_csv(
        tmp_path / "human.csv",
        [
            {"text": "Khóa chính là gì?", "category": "concept"},
            {"text": "Làm sao chuẩn hóa bảng dữ liệu?", "category": "concept"},
        ],
    )

    queries, duplicates = import_review(tmp_path / "review.csv", DRAFTS, FakeEncoder(), tmp_path / "human.csv")

    assert [query["query_id"] for query in queries][:2] == ["q1", "q2"]
    assert queries[1]["text"] == "Vai trò của khóa ngoại trong lược đồ?"
    assert queries[1]["category"] == "paraphrase"
    assert queries[2]["origin"] == "human" and queries[2]["query_id"].startswith("h-")
    assert len(queries) == 3
    assert duplicates[0]["duplicate_of"] == "q1"


def test_import_review_rejects_unknown_action_and_category(tmp_path):
    export_review(DRAFTS[:1], TinyIndex(), tmp_path / "review.csv")
    rows = read_csv(tmp_path / "review.csv")
    rows[0]["action"] = "maybe"
    write_csv(tmp_path / "review.csv", rows)
    with pytest.raises(ValueError, match="q1"):
        import_review(tmp_path / "review.csv", DRAFTS, FakeEncoder())
    rows[0].update(action="edit", new_text="x", new_category="other")
    write_csv(tmp_path / "review.csv", rows)
    with pytest.raises(ValueError, match="category"):
        import_review(tmp_path / "review.csv", DRAFTS, FakeEncoder())
