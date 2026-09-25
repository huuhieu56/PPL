import numpy as np


def _differences(left, right) -> np.ndarray:
    if len(left) != len(right) or not len(left):
        raise ValueError("paired samples must have the same non-zero length")
    return np.asarray(left, dtype=float) - np.asarray(right, dtype=float)


def paired_bootstrap_ci(left, right, seed: int = 42, samples: int = 10000, confidence: float = 0.95) -> tuple[float, float]:
    differences = _differences(left, right)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(differences), size=(samples, len(differences)))
    means = differences[indices].mean(axis=1)
    tail = (1 - confidence) / 2 * 100
    return float(np.percentile(means, tail)), float(np.percentile(means, 100 - tail))


def paired_randomization_test(left, right, seed: int = 42, permutations: int = 10000) -> float:
    differences = _differences(left, right)
    observed = abs(differences.mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(permutations, len(differences)))
    permuted = np.abs((signs * differences).mean(axis=1))
    extreme = int(np.sum(permuted >= observed - 1e-12))
    return (extreme + 1) / (permutations + 1)


def holm_adjust(pvalues: dict) -> dict:
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    total = len(ordered)
    adjusted = {}
    running = 0.0
    for position, (key, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - position) * value))
        adjusted[key] = running
    return adjusted
