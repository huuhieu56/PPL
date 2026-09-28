import time
from functools import lru_cache

from src.models import SearchResult


@lru_cache(maxsize=2)
def _load_model(model_name: str):
    import torch
    from sentence_transformers import CrossEncoder

    # ponytail: keep the second large model on CPU below 6 GiB; revisit if both fit after quantization.
    device = "cpu" if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 6 * 1024**3 else None
    revision = "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e" if model_name == "BAAI/bge-reranker-v2-m3" else None
    return CrossEncoder(model_name, device=device, revision=revision)


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
        model = _load_model(model_name)
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
