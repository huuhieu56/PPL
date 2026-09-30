import math
import re

from src.index import RetrievalIndex
from src.models import Chunk, RagConfig, SearchResult


def minmax_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if math.isclose(low, high):
        return {key: float(high > 0) for key in scores}
    scale = high - low
    return {key: (value - low) / scale for key, value in scores.items()}


def adaptive_alpha(
    query: str,
    alpha0: float,
    beta: float,
    idf: dict[str, float],
) -> tuple[float, dict[str, float]]:
    compact = query.strip()
    length = max(len(compact), 1)
    tokens = re.findall(r"\w+", compact.lower(), flags=re.UNICODE)
    idf_scale = max(1.0, max(idf.values(), default=0.0))
    signals = {
        "digit": sum(character.isdigit() for character in compact) / length,
        "symbol": sum(not character.isalnum() and not character.isspace() for character in compact)
        / length,
        "code": float(bool(re.search(r"\b[A-Z]{2,}[A-Z0-9-]*\d+[A-Z0-9-]*\b", compact))),
        "max_idf": max((idf.get(token, 0.0) for token in tokens), default=0.0) / idf_scale,
    }
    lexical_score = min(
        1.0,
        signals["digit"] + signals["symbol"] + signals["code"] + signals["max_idf"],
    )
    signals["lexical_score"] = lexical_score
    return min(0.9, max(0.1, alpha0 + beta * lexical_score)), signals


def fuse_weighted(
    bm25_scores: dict[str, float],
    dense_scores: dict[str, float],
    chunks: dict[str, Chunk],
    alpha: float,
) -> list[SearchResult]:
    sparse = minmax_scores(bm25_scores)
    dense = minmax_scores(dense_scores)
    identifiers = set(sparse) | set(dense)
    ranked = sorted(
        identifiers,
        key=lambda chunk_id: (
            -(alpha * sparse.get(chunk_id, 0.0) + (1 - alpha) * dense.get(chunk_id, 0.0)),
            chunk_id,
        ),
    )
    return [
        SearchResult(
            chunks[chunk_id],
            alpha * sparse.get(chunk_id, 0.0) + (1 - alpha) * dense.get(chunk_id, 0.0),
            rank,
            "weighted",
        )
        for rank, chunk_id in enumerate(ranked, start=1)
    ]


def fuse_rrf(
    result_lists: list[list[SearchResult]], rrf_k: int = 60
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for results in result_lists:
        for result in results:
            chunk_id = result.chunk.chunk_id
            chunks[chunk_id] = result.chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rrf_k + result.rank)
    ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [
        SearchResult(chunks[chunk_id], scores[chunk_id], rank, "rrf")
        for rank, chunk_id in enumerate(ranked, start=1)
    ]


def retrieve(query: str, index: RetrievalIndex, config: RagConfig) -> list[SearchResult]:
    sparse = index.sparse_scores(query, config.top_l) if config.method != "dense" else {}
    dense = index.dense_scores(query, config.top_l) if config.method != "bm25" else {}
    if config.method == "bm25":
        ranked = sorted(sparse, key=lambda key: (-sparse[key], key))
        return [SearchResult(index.chunks[key], sparse[key], rank, "bm25") for rank, key in enumerate(ranked, 1)]
    if config.method == "dense":
        ranked = sorted(dense, key=lambda key: (-dense[key], key))
        return [SearchResult(index.chunks[key], dense[key], rank, "dense") for rank, key in enumerate(ranked, 1)]
    if config.method == "rrf":
        sparse_results = [SearchResult(index.chunks[key], value, rank, "bm25") for rank, (key, value) in enumerate(sorted(sparse.items(), key=lambda item: (-item[1], item[0])), 1)]
        dense_results = [SearchResult(index.chunks[key], value, rank, "dense") for rank, (key, value) in enumerate(sorted(dense.items(), key=lambda item: (-item[1], item[0])), 1)]
        return fuse_rrf([sparse_results, dense_results], config.rrf_k)
    alpha = config.alpha
    if config.method == "adaptive":
        alpha, _ = adaptive_alpha(query, config.alpha, config.adaptive_beta, self_idf(index))
    if config.method not in {"weighted", "adaptive"}:
        raise ValueError(f"Unsupported retrieval method: {config.method}")
    return fuse_weighted(sparse, dense, index.chunks, alpha)


def self_idf(index: RetrievalIndex) -> dict[str, float]:
    return {token: float(value) for token, value in index.bm25.idf.items()}
