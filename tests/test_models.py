import pytest

from src.models import PipelineConfig, RetrievedChunk, StageScores


def test_pipeline_config_defaults_follow_report():
    config = PipelineConfig()
    assert (config.sparse, config.dense, config.fusion, config.rerank) == (True, True, "weighted", True)
    assert (config.top_l, config.rerank_n, config.context_k) == (100, 30, 5)
    assert PipelineConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"fusion": "sum"}, "Unsupported fusion"),
        ({"fusion": "none"}, "exactly one"),
        ({"dense": False}, "requires both"),
        ({"alpha": 1.5}, "alpha"),
        ({"rerank_n": 200}, "context_k <= rerank_n <= top_l"),
        ({"rerank": False, "context_k": 101}, "context_k <= top_l"),
        ({"tokenizer": "spacy"}, "Unsupported tokenizer"),
    ],
)
def test_pipeline_config_rejects_inconsistent_values(changes, message):
    with pytest.raises(ValueError, match=message):
        PipelineConfig(**changes)


def test_pipeline_config_from_dict_rejects_legacy_keys():
    with pytest.raises(ValueError, match="Unknown config keys"):
        PipelineConfig.from_dict({"method": "rrf", "use_reranker": False})


def test_sparse_only_config_is_valid():
    config = PipelineConfig(dense=False, fusion="none", rerank=False)
    assert config.replace(tokenizer="pyvi").tokenizer == "pyvi"


def test_retrieved_chunk_score_prefers_latest_stage():
    assert RetrievedChunk(None, StageScores(rank=1, sparse_score=3.0)).score == 3.0
    assert RetrievedChunk(None, StageScores(rank=1, dense_score=0.4)).score == 0.4
    fused = StageScores(rank=1, sparse_score=3.0, dense_score=0.4, fusion_score=0.7)
    assert RetrievedChunk(None, fused).score == 0.7
    reranked = StageScores(rank=1, fusion_score=0.7, rerank_score=-1.2)
    assert RetrievedChunk(None, reranked).score == -1.2
    assert StageScores(rank=2).to_dict()["rerank_score"] is None
