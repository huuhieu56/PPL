from src.cache import RetrievalCache
from src.reranking import Reranker
from tests.fakes import FakeCrossEncoder, make_chunk


def test_first_stage_cache_round_trip(tmp_path):
    cache = RetrievalCache(tmp_path / "cache" / "retrieval.sqlite")
    key = RetrievalCache.first_stage_key("v1", "sparse", "whitespace", "câu hỏi", 100)
    assert cache.get_first_stage(key) is None
    cache.put_first_stage(key, [("a", 2.5), ("b", 1.0)])
    reopened = RetrievalCache(tmp_path / "cache" / "retrieval.sqlite")
    assert reopened.get_first_stage(key) == [("a", 2.5), ("b", 1.0)]
    assert key != RetrievalCache.first_stage_key("v1", "sparse", "pyvi", "câu hỏi", 100)


def test_reranker_scores_only_uncached_chunks(tmp_path):
    cache = RetrievalCache(tmp_path / "retrieval.sqlite")
    model = FakeCrossEncoder()
    reranker = Reranker("fake-model", model=model, cache=cache)
    chunks = [make_chunk("a", "học máy là gì"), make_chunk("b", "cơ sở dữ liệu")]

    first = reranker.score("học máy", chunks)
    assert first == [2.0, 0.0]
    assert model.pairs_seen == 2

    again = reranker.score("học máy", [*chunks, make_chunk("c", "máy tính")])
    assert again == [2.0, 0.0, 1.0]
    assert model.pairs_seen == 3


def test_reranker_bypasses_cache_when_requested(tmp_path):
    cache = RetrievalCache(tmp_path / "retrieval.sqlite")
    model = FakeCrossEncoder()
    reranker = Reranker("fake-model", model=model, cache=cache)
    chunks = [make_chunk("a", "học máy")]
    reranker.score("học máy", chunks, use_cache=False)
    reranker.score("học máy", chunks, use_cache=False)
    assert model.pairs_seen == 2
    assert cache.get_rerank("fake-model", "học máy", ["a"]) == {}
    assert Reranker("fake-model", model=model).score("x", []) == []
