import math
import re

_CODE = re.compile(r"\b[A-Z]{2,}[A-Z0-9-]*\d+[A-Z0-9-]*\b")


def minmax_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if math.isclose(low, high):
        return {key: 1.0 for key in scores}
    scale = high - low
    return {key: (value - low) / scale for key, value in scores.items()}


def rank_ids(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))


def fuse_rrf(rankings: list[list[str]], k: int) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


def fuse_weighted(
    sparse: dict[str, float], dense: dict[str, float], alpha: float
) -> dict[str, float]:
    sparse_norm, dense_norm = minmax_scores(sparse), minmax_scores(dense)
    return {
        chunk_id: alpha * sparse_norm.get(chunk_id, 0.0) + (1 - alpha) * dense_norm.get(chunk_id, 0.0)
        for chunk_id in set(sparse_norm) | set(dense_norm)
    }


def adaptive_alpha(
    query: str,
    query_tokens: list[str],
    alpha0: float,
    beta: float,
    idf: dict[str, float],
) -> tuple[float, dict[str, float]]:
    compact = query.strip()
    length = max(len(compact), 1)
    corpus_max = max(idf.values(), default=0.0)
    query_max = max((idf.get(token, 0.0) for token in query_tokens), default=0.0)
    signals = {
        "digit": sum(character.isdigit() for character in compact) / length,
        "symbol": sum(not character.isalnum() and not character.isspace() for character in compact)
        / length,
        "code": float(bool(_CODE.search(compact))),
        "max_idf": query_max / corpus_max if corpus_max > 0 else 0.0,
    }
    lexical = min(1.0, sum(signals.values()))
    signals["lexical_score"] = lexical
    return min(0.9, max(0.1, alpha0 + beta * (2 * lexical - 1))), signals
