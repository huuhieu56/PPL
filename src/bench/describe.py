import statistics
from collections import Counter, defaultdict


def describe_benchmark(queries, qrels, pool: list[dict] | None = None, agreement: dict | None = None) -> dict:
    counts = Counter((query["category"], query.get("origin", "llm"), query.get("split")) for query in queries)
    relevant = defaultdict(set)
    for row in qrels:
        if int(row["relevance"]) >= 1:
            relevant[row["query_id"]].add(row["chunk_id"])
    per_category = defaultdict(list)
    for query in queries:
        per_category[query["category"]].append(len(relevant[query["query_id"]]))
    contribution = []
    if pool:
        found = defaultdict(int)
        unique = defaultdict(int)
        for entry in pool:
            if entry["chunk_id"] not in relevant[entry["query_id"]]:
                continue
            for system in entry["systems"]:
                found[system] += 1
            if len(entry["systems"]) == 1:
                unique[entry["systems"][0]] += 1
        contribution = [
            {"system": system, "relevant_found": found[system], "unique_relevant": unique[system]}
            for system in sorted(found)
        ]
    return {
        "counts": [
            {"category": category, "origin": origin, "split": split, "queries": number}
            for (category, origin, split), number in sorted(counts.items(), key=lambda item: tuple(str(part) for part in item[0]))
        ],
        "relevant_per_query": [
            {"category": category, "mean": statistics.mean(values), "median": statistics.median(values)}
            for category, values in sorted(per_category.items())
        ],
        "pool_contribution": contribution,
        "agreement": agreement,
        "totals": {
            "queries": len(queries),
            "qrels": len(qrels),
            "relevant": sum(len(chunk_ids) for chunk_ids in relevant.values()),
        },
    }
