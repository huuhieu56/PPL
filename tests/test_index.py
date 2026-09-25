import unicodedata

import pytest

from src.index import RetrievalIndex
from tests.fakes import FakeEncoder, make_chunk

TEXTS = {
    "c1": "Mã môn AI101 là học phần nhập môn trí tuệ nhân tạo",
    "c2": "Học máy là lĩnh vực nghiên cứu thuật toán học từ dữ liệu",
    "c3": "Cơ sở dữ liệu quan hệ lưu trữ bảng và khóa chính",
    "c4": "Mạng nơ ron sâu gồm nhiều tầng biến đổi phi tuyến",
}


@pytest.fixture
def chunks():
    return [make_chunk(chunk_id, text) for chunk_id, text in TEXTS.items()]


@pytest.fixture
def index(tmp_path, chunks):
    return RetrievalIndex.build(
        chunks, tmp_path / "idx-v1", embedding_model="fake", tokenizers=("whitespace",), encoder=FakeEncoder()
    )


def test_build_and_load_round_trip(tmp_path, index, chunks):
    loaded = RetrievalIndex.load(tmp_path / "idx-v1", encoder=FakeEncoder())
    assert loaded.version == "idx-v1"
    assert loaded.chunk_list == chunks
    assert loaded.tokenizers == ["whitespace"]
    assert loaded.meta["bm25"] == {"k1": 1.5, "b": 0.75}
    assert loaded.sparse_search("AI101", "whitespace", 10) == index.sparse_search("AI101", "whitespace", 10)


def test_sparse_search_returns_only_matching_chunks(index):
    results = index.sparse_search("mã môn AI101", "whitespace", 10)
    assert [chunk_id for chunk_id, _ in results] == ["c1"]
    assert results[0][1] > 0
    assert index.sparse_search("???", "whitespace", 10) == []


def test_sparse_search_matches_nfd_query(index):
    nfd = unicodedata.normalize("NFD", "Học máy thuật toán")
    assert index.sparse_search(nfd, "whitespace", 10) == index.sparse_search("Học máy thuật toán", "whitespace", 10)


def test_dense_search_ranks_by_cosine(index):
    results = index.dense_search("cơ sở dữ liệu quan hệ", 2)
    assert results[0][0] == "c3"
    assert len(results) == 2
    assert results[0][1] >= results[1][1]


def test_missing_tokenizer_error_names_fix(index):
    with pytest.raises(ValueError, match="add-tokenizer"):
        index.sparse_search("học máy", "pyvi", 10)


def test_add_tokenizer_persists(tmp_path, index):
    index.add_tokenizer("pyvi")
    loaded = RetrievalIndex.load(tmp_path / "idx-v1", encoder=FakeEncoder())
    assert loaded.tokenizers == ["whitespace", "pyvi"]
    assert loaded.sparse_search("nghiên cứu", "pyvi", 10)[0][0] == "c2"
    assert "nghiên_cứu" in loaded.idf("pyvi")


def test_faiss_backend_matches_numpy(tmp_path, chunks, index):
    pytest.importorskip("faiss")
    faiss_index = RetrievalIndex.build(
        chunks, tmp_path / "idx-faiss", embedding_model="fake", dense_backend="faiss", encoder=FakeEncoder()
    )
    assert (tmp_path / "idx-faiss" / "faiss.index").exists()
    reloaded = RetrievalIndex.load(tmp_path / "idx-faiss", encoder=FakeEncoder())
    expected = index.dense_search("học máy dữ liệu", 3)
    actual = reloaded.dense_search("học máy dữ liệu", 3)
    assert [chunk_id for chunk_id, _ in actual] == [chunk_id for chunk_id, _ in expected]
    assert [score for _, score in actual] == pytest.approx([score for _, score in expected], abs=1e-5)


def test_build_rejects_empty_corpus(tmp_path):
    with pytest.raises(ValueError, match="without chunks"):
        RetrievalIndex.build([], tmp_path / "empty", embedding_model="fake", encoder=FakeEncoder())
