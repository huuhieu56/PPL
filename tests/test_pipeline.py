import pytest

from src.cache import RetrievalCache
from src.index import RetrievalIndex
from src.models import PipelineConfig
from src.pipeline import RetrievalPipeline
from src.reranking import Reranker
from tests.fakes import FakeCrossEncoder, FakeEncoder, make_chunk

TEXTS = {
    "c1": "Mã môn AI101 là học phần nhập môn trí tuệ nhân tạo",
    "c2": "Học máy là lĩnh vực nghiên cứu thuật toán học từ dữ liệu",
    "c3": "Cơ sở dữ liệu quan hệ lưu trữ bảng và khóa chính",
    "c4": "Mạng nơ ron sâu gồm nhiều tầng biến đổi phi tuyến",
}
BM25 = PipelineConfig(dense=False, fusion="none", rerank=False, top_l=10)
DENSE = PipelineConfig(sparse=False, fusion="none", rerank=False, top_l=10)
HYBRID = PipelineConfig(fusion="weighted", rerank=False, top_l=10)


@pytest.fixture
def pipeline(tmp_path):
    index = RetrievalIndex.build(
        [make_chunk(chunk_id, text) for chunk_id, text in TEXTS.items()],
        tmp_path / "idx-v1",
        embedding_model="fake",
        encoder=FakeEncoder(),
    )
    cross = FakeCrossEncoder()
    cache = RetrievalCache(tmp_path / "cache.sqlite")
    built = RetrievalPipeline(
        index,
        cache=cache,
        reranker_factory=lambda name: Reranker(name, model=cross, cache=cache),
        synchronize=lambda: None,
    )
    built.cross = cross
    return built


def test_bm25_only_records_sparse_stage(pipeline):
    result = pipeline.run("mã môn AI101", BM25)
    top = result.results[0]
    assert top.chunk.chunk_id == "c1"
    assert top.scores.rank == 1 and top.scores.sparse_rank == 1
    assert top.scores.dense_score is None and top.scores.fusion_score is None
    assert set(result.timings_ms) == {"sparse", "dense", "fusion", "rerank", "total"}
    assert result.timings_ms["dense"] == 0.0


def test_dense_only_ranks_all_chunks(pipeline):
    result = pipeline.run("cơ sở dữ liệu quan hệ", DENSE)
    assert result.results[0].chunk.chunk_id == "c3"
    assert len(result.results) == 4
    assert result.results[0].scores.sparse_score is None


def test_hybrid_logs_every_stage_score(pipeline):
    result = pipeline.run("học máy dữ liệu", HYBRID)
    by_id = {item.chunk.chunk_id: item.scores for item in result.results}
    assert by_id["c2"].sparse_rank is not None and by_id["c2"].dense_rank is not None
    assert by_id["c4"].sparse_score is None and by_id["c4"].dense_score is not None
    assert [item.scores.rank for item in result.results] == list(range(1, len(result.results) + 1))
    fusion_scores = [item.scores.fusion_score for item in result.results]
    assert fusion_scores == sorted(fusion_scores, reverse=True)
    assert result.alpha_used == 0.5


def test_hybrid_with_no_sparse_hits_falls_back_to_dense(pipeline):
    result = pipeline.run("zzz qqq", HYBRID)
    assert len(result.results) == 4
    assert all(item.scores.sparse_score is None for item in result.results)


def test_rrf_and_adaptive_fusion(pipeline):
    rrf = pipeline.run("học máy dữ liệu", HYBRID.replace(fusion="rrf", rrf_k=60))
    assert rrf.results[0].scores.fusion_score == pytest.approx(
        sum(1 / (60 + rank) for rank in (rrf.results[0].scores.sparse_rank, rrf.results[0].scores.dense_rank) if rank)
    )
    adaptive = pipeline.run("Mã môn AI101", HYBRID.replace(fusion="adaptive", adaptive_beta=0.3))
    assert adaptive.alpha_used > 0.5


def test_rerank_reorders_head_and_keeps_tail(pipeline):
    config = HYBRID.replace(rerank=True, rerank_n=2, context_k=2)
    result = pipeline.run("học máy thuật toán dữ liệu", config)
    head = result.results[:2]
    assert all(item.scores.rerank_score is not None for item in head)
    assert head[0].scores.rerank_score >= head[1].scores.rerank_score
    assert all(item.scores.rerank_score is None for item in result.results[2:])
    assert result.timings_ms["rerank"] >= 0.0


def test_cache_skips_recomputation_and_can_be_bypassed(pipeline):
    config = HYBRID.replace(rerank=True, rerank_n=2, context_k=2)
    pipeline.run("học máy", config)
    seen = pipeline.cross.pairs_seen
    encoder_calls = pipeline.index._encoder.calls
    pipeline.run("học máy", config)
    assert pipeline.cross.pairs_seen == seen
    assert pipeline.index._encoder.calls == encoder_calls
    pipeline.run("học máy", config, use_cache=False)
    assert pipeline.cross.pairs_seen == seen + 2
    assert pipeline.index._encoder.calls == encoder_calls + 1
