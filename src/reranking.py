import time

from src.models import SearchResult


def rerank(
    query: str,
    results: list[SearchResult],
    limit: int = 20,
    model_name: str = "BAAI/bge-reranker-v2-m3",
    model=None,
) -> tuple[list[SearchResult], float]:
    if not results:
        return [], 0.0
    started = time.perf_counter()
    if model is None:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name)
    candidates = results[:limit]
    scores = model.predict(
        [(query, result.chunk.text) for result in candidates],
        batch_size=min(16, len(candidates)),
        show_progress_bar=False,
    )
    ordered = sorted(zip(candidates, scores), key=lambda item: (-float(item[1]), item[0].chunk.chunk_id))
    reranked = [
        SearchResult(result.chunk, float(score), rank, "reranker")
        for rank, (result, score) in enumerate(ordered, start=1)
    ]
    return reranked, (time.perf_counter() - started) * 1000
