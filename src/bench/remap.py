from collections import defaultdict

from src.text import normalize_text


def _words(text: str) -> list[str]:
    return normalize_text(text).lower().split()


def _longest_common_run(needle: list[str], haystack: list[str]) -> int:
    best = 0
    previous = [0] * (len(haystack) + 1)
    for left in needle:
        current = [0] * (len(haystack) + 1)
        for position, right in enumerate(haystack, start=1):
            if left == right:
                current[position] = previous[position - 1] + 1
                best = max(best, current[position])
        previous = current
    return best


def remap_qrels(evidence: list[dict], chunks, min_fraction: float = 0.6) -> list[dict]:
    by_doc = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append((chunk.chunk_id, _words(chunk.body)))
    grades: dict[tuple[str, str], int] = {}
    for item in evidence:
        quote = _words(item["quote"])
        if not quote:
            continue
        needed = min_fraction * len(quote)
        joined_quote = " ".join(quote)
        for chunk_id, words in by_doc[item["doc_id"]]:
            if joined_quote in " ".join(words) or _longest_common_run(quote, words) >= needed:
                key = (item["query_id"], chunk_id)
                grades[key] = max(grades.get(key, 0), int(item["relevance"]))
    return [
        {"query_id": query_id, "chunk_id": chunk_id, "relevance": grade}
        for (query_id, chunk_id), grade in sorted(grades.items())
    ]
