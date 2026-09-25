import json

import pytest

from src.bench.pool import build_pool, write_pool
from src.index import RetrievalIndex
from src.io_utils import read_csv
from src.models import PipelineConfig
from src.pipeline import RetrievalPipeline
from tests.fakes import FakeEncoder, make_chunk

TEXTS = {
    "c1": "Mã môn AI101 là học phần nhập môn trí tuệ nhân tạo",
    "c2": "Học máy là lĩnh vực nghiên cứu thuật toán học từ dữ liệu",
    "c3": "Cơ sở dữ liệu quan hệ lưu trữ bảng và khóa chính",
    "c4": "Mạng nơ ron sâu gồm nhiều tầng biến đổi phi tuyến",
}
SYSTEMS = {
    "C1": PipelineConfig(dense=False, fusion="none", rerank=False, top_l=10),
    "C2": PipelineConfig(sparse=False, fusion="none", rerank=False, top_l=10),
}
QUERIES = [{"query_id": "q1", "text": "mã môn AI101", "source_chunk_ids": ["c3"]}]


@pytest.fixture
def index(tmp_path):
    chunks = [make_chunk(cid, text, heading_path=("Bài 1",), page=2) for cid, text in TEXTS.items()]
    return RetrievalIndex.build(chunks, tmp_path / "idx", embedding_model="fake", encoder=FakeEncoder())


def test_build_pool_merges_systems_source_and_manual(index):
    pipeline = RetrievalPipeline(index, synchronize=lambda: None)
    pool = build_pool(QUERIES, pipeline, SYSTEMS, depth=2, manual_additions=[{"query_id": "q1", "chunk_id": "c4"}])
    by_chunk = {entry["chunk_id"]: entry for entry in pool}
    assert "C1" in by_chunk["c1"]["systems"] and "C2" in by_chunk["c1"]["systems"]
    assert "source" in by_chunk["c3"]["systems"]
    assert "manual" in by_chunk["c4"]["systems"]
    assert by_chunk["c1"]["page"] == 2 and by_chunk["c1"]["doc_id"] == "doc-1"
    assert len({entry["pool_id"] for entry in pool}) == len(pool)


def test_build_pool_rejects_unknown_manual_chunk(index):
    pipeline = RetrievalPipeline(index, synchronize=lambda: None)
    with pytest.raises(ValueError, match="c99"):
        build_pool(QUERIES, pipeline, SYSTEMS, depth=2, manual_additions=[{"query_id": "q1", "chunk_id": "c99"}])


def test_write_pool_creates_blind_shuffled_annotation_files(tmp_path, index):
    pipeline = RetrievalPipeline(index, synchronize=lambda: None)
    pool = build_pool(QUERIES, pipeline, SYSTEMS, depth=4)
    paths = write_pool(pool, QUERIES, index, tmp_path / "bench", annotators=("A", "B"), seed=1)
    assert [path.name for path in paths] == ["annotation_A.csv", "annotation_B.csv"]
    rows = read_csv(tmp_path / "bench" / "annotation_A.csv")
    assert len(rows) == len(pool)
    assert "systems" not in rows[0] and rows[0]["relevance"] == ""
    assert rows[0]["heading_path"] == "Tài liệu > Bài 1"
    saved = json.loads((tmp_path / "bench" / "pool_map.json").read_text(encoding="utf-8"))
    assert {entry["pool_id"] for entry in saved} == {row["pool_id"] for row in rows}
    assert [row["pool_id"] for row in rows] == [
        row["pool_id"] for row in read_csv(tmp_path / "bench" / "annotation_B.csv")
    ]


def test_write_pool_refuses_to_overwrite_annotations(tmp_path, index):
    pipeline = RetrievalPipeline(index, synchronize=lambda: None)
    pool = build_pool(QUERIES, pipeline, SYSTEMS, depth=2)
    write_pool(pool, QUERIES, index, tmp_path / "bench")
    with pytest.raises(ValueError, match="already exists"):
        write_pool(pool, QUERIES, index, tmp_path / "bench")
