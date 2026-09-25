import pytest

from src.fusion import adaptive_alpha, fuse_rrf, fuse_weighted, minmax_scores, rank_ids


def test_minmax_handles_constant_and_empty_scores():
    assert minmax_scores({}) == {}
    assert minmax_scores({"a": 5.0, "b": 5.0}) == {"a": 1.0, "b": 1.0}
    assert minmax_scores({"a": 2.0, "b": 4.0, "c": 3.0}) == {"a": 0.0, "b": 1.0, "c": 0.5}


def test_rank_ids_breaks_ties_by_chunk_id():
    assert rank_ids({"b": 1.0, "a": 1.0, "c": 2.0}) == ["c", "a", "b"]


def test_rrf_sums_reciprocal_ranks_across_lists():
    fused = fuse_rrf([["a", "b"], ["b", "c"]], k=60)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert rank_ids(fused) == ["b", "a", "c"]


def test_weighted_fusion_uses_union_and_zero_for_missing_branch():
    fused = fuse_weighted({"a": 8.0, "b": 1.0}, {"b": 0.9, "c": 0.5}, alpha=0.7)
    assert fused["a"] == pytest.approx(0.7 * 1.0 + 0.3 * 0.0)
    assert fused["b"] == pytest.approx(0.7 * 0.0 + 0.3 * 1.0)
    assert fused["c"] == pytest.approx(0.0)
    assert fuse_weighted({"a": 1.0}, {}, alpha=0.5) == {"a": 0.5}


def test_adaptive_alpha_raises_for_codes_and_lowers_for_concepts():
    idf = {"ai101": 2.0, "học": 0.1, "máy": 0.1, "giải": 0.2, "thích": 0.2}
    exact, signals = adaptive_alpha(
        "Mã môn AI101 là gì?", ["mã", "môn", "ai101", "là", "gì"], 0.5, 0.3, idf
    )
    concept, _ = adaptive_alpha(
        "Giải thích học máy", ["giải", "thích", "học", "máy"], 0.5, 0.3, idf
    )
    assert signals["code"] == 1.0
    assert signals["max_idf"] == pytest.approx(1.0)
    assert exact > 0.5 > concept
    assert 0.1 <= concept and exact <= 0.9


def test_adaptive_alpha_with_empty_idf_stays_in_bounds():
    alpha, signals = adaptive_alpha("???", [], 0.5, 0.3, {})
    assert signals["max_idf"] == 0.0
    assert 0.1 <= alpha <= 0.9
