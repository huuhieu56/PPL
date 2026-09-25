import pytest

from src.bench.agreement import (
    agreement_report,
    merge_labels,
    parse_relevance,
    read_annotations,
    weighted_kappa,
    write_disagreements,
)
from src.io_utils import read_csv, write_csv
from tests.fakes import make_chunk


class TinyIndex:
    chunks = {
        "c1": make_chunk("c1", "Khóa chính xác định duy nhất bản ghi", page=4),
        "c2": make_chunk("c2", "Khóa ngoại tham chiếu bảng khác", page=5),
        "c3": make_chunk("c3", "Chuẩn hóa loại bỏ dư thừa", page=6),
    }


POOL = [
    {"pool_id": "p1", "query_id": "q1", "chunk_id": "c1", "doc_id": "doc-1", "page": 4, "systems": ["C1"]},
    {"pool_id": "p2", "query_id": "q1", "chunk_id": "c2", "doc_id": "doc-1", "page": 5, "systems": ["C2"]},
    {"pool_id": "p3", "query_id": "q1", "chunk_id": "c3", "doc_id": "doc-1", "page": 6, "systems": ["C2"]},
]
QUERIES = [{"query_id": "q1", "text": "Khóa chính là gì?"}]


def _annotation(path, labels, quotes=None):
    quotes = quotes or {}
    write_csv(
        path,
        [{"pool_id": pid, "relevance": label, "evidence_quote": quotes.get(pid, "")} for pid, label in labels.items()],
    )
    return read_annotations(path)


def test_weighted_kappa_matches_hand_computation():
    assert weighted_kappa([0, 1, 2, 2], [0, 1, 2, 1]) == pytest.approx(0.8)
    assert weighted_kappa([1, 1], [1, 1]) == 1.0
    assert weighted_kappa([0, 2], [2, 0]) < 0


def test_parse_relevance_accepts_excel_numbers():
    assert parse_relevance("2.0", "p1") == 2
    assert parse_relevance(" ", "p1") is None


def test_read_annotations_rejects_invalid_label(tmp_path):
    write_csv(tmp_path / "a.csv", [{"pool_id": "p9", "relevance": "3", "evidence_quote": ""}])
    with pytest.raises(ValueError, match="p9"):
        read_annotations(tmp_path / "a.csv")


def test_agreement_report_uses_only_double_labeled_items(tmp_path):
    first = _annotation(tmp_path / "a.csv", {"p1": "2", "p2": "0", "p3": "1"})
    second = _annotation(tmp_path / "b.csv", {"p1": "2", "p2": "1", "p3": ""})
    report = agreement_report(first, second)
    assert report["overlap"] == 2
    assert report["raw_agreement"] == 0.5
    assert report["disagreements"] == ["p2"]
    write_disagreements(tmp_path / "dis.csv", report, POOL, first, second, QUERIES, TinyIndex())
    rows = read_csv(tmp_path / "dis.csv")
    assert rows == [{"pool_id": "p2", "query": "Khóa chính là gì?", "chunk_text": "Khóa ngoại tham chiếu bảng khác",
                     "label_a": "0", "label_b": "1", "final": ""}]


def test_merge_labels_resolves_and_builds_evidence(tmp_path):
    first = _annotation(
        tmp_path / "a.csv",
        {"p1": "2", "p2": "0", "p3": "1"},
        {"p1": "xác định duy nhất bản ghi", "p3": "câu không có trong chunk"},
    )
    second = _annotation(tmp_path / "b.csv", {"p1": "2", "p2": "1", "p3": ""})
    with pytest.raises(ValueError, match="p2"):
        merge_labels(POOL, [first, second], TinyIndex())

    write_csv(tmp_path / "dis.csv", [{"pool_id": "p2", "final": "0"}])
    qrels, evidence, warnings = merge_labels(POOL, [first, second], TinyIndex(), tmp_path / "dis.csv")

    assert qrels == [
        {"query_id": "q1", "chunk_id": "c1", "relevance": 2},
        {"query_id": "q1", "chunk_id": "c2", "relevance": 0},
        {"query_id": "q1", "chunk_id": "c3", "relevance": 1},
    ]
    assert evidence == [
        {"query_id": "q1", "doc_id": "doc-1", "page": 4, "quote": "xác định duy nhất bản ghi", "relevance": 2}
    ]
    assert len(warnings) == 1 and "p3" in warnings[0]
