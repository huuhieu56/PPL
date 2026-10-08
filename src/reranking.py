import json
import math
import os
import time
from urllib.request import Request, urlopen

from src.models import SearchResult


def rerank(
    query: str,
    results: list[SearchResult],
    limit: int = 20,
) -> tuple[list[SearchResult], float]:
    if not results:
        return [], 0.0
    if limit < 1:
        raise ValueError("Rerank limit must be positive")
    api_key = os.environ.get("EMBEDDING_API_KEY")
    if not api_key:
        raise ValueError("EMBEDDING_API_KEY is required for OpenRouter reranking")
    started = time.perf_counter()
    candidates = results[:limit]
    request = Request(
        "https://openrouter.ai/api/v1/rerank",
        data=json.dumps({
            "model": "voyageai/rerank-2.5-lite",
            "query": query,
            "documents": [result.chunk.text for result in candidates],
            "top_n": len(candidates),
        }).encode(),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=30) as response:
        items = json.load(response)["results"]
    indices = [item["index"] for item in items]
    if any(type(index) is not int for index in indices) or sorted(indices) != list(range(len(candidates))):
        raise ValueError("Rerank API returned missing, duplicate or invalid document indices")
    if any(not math.isfinite(float(item["relevance_score"])) for item in items):
        raise ValueError("Rerank API returned non-finite scores")
    ordered = sorted(items, key=lambda item: (-float(item["relevance_score"]), candidates[item["index"]].chunk.chunk_id))
    reranked = [
        SearchResult(candidates[item["index"]].chunk, float(item["relevance_score"]), rank, "reranker")
        for rank, item in enumerate(ordered, start=1)
    ]
    return reranked, (time.perf_counter() - started) * 1000
