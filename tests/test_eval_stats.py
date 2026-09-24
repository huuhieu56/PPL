import pytest

from src.eval.stats import holm_adjust, paired_bootstrap_ci, paired_randomization_test


def test_identical_systems_have_zero_interval_and_p_one():
    values = [0.2, 0.5, 1.0, 0.0, 0.3]
    assert paired_bootstrap_ci(values, values, samples=500) == (0.0, 0.0)
    assert paired_randomization_test(values, values, permutations=500) == 1.0


def test_clear_improvement_is_significant_and_deterministic():
    better = [1.0] * 30
    worse = [0.0] * 25 + [1.0] * 5
    low, high = paired_bootstrap_ci(better, worse, seed=1, samples=2000)
    assert 0 < low <= high <= 1
    p = paired_randomization_test(better, worse, seed=1, permutations=2000)
    assert p < 0.01
    assert p == paired_randomization_test(better, worse, seed=1, permutations=2000)


def test_stats_reject_mismatched_inputs():
    with pytest.raises(ValueError):
        paired_bootstrap_ci([1.0], [1.0, 2.0])
    with pytest.raises(ValueError):
        paired_randomization_test([], [])


def test_holm_adjustment_matches_hand_computation():
    adjusted = holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03})
    assert adjusted == pytest.approx({"a": 0.03, "c": 0.06, "b": 0.06})
    assert holm_adjust({"x": 0.6, "y": 0.9}) == pytest.approx({"x": 1.0, "y": 1.0})
    assert holm_adjust({}) == {}
