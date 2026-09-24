import random
from collections import defaultdict


def split_queries(queries, qrels, dev_ratio: float = 0.3, seed: int = 42) -> tuple[list[dict], list[str]]:
    answerable = {row["query_id"] for row in qrels if int(row["relevance"]) >= 1}
    dropped = [query["query_id"] for query in queries if query["query_id"] not in answerable]
    groups = defaultdict(list)
    for query in queries:
        if query["query_id"] in answerable:
            groups[query["category"]].append(query["query_id"])
    rng = random.Random(seed)
    dev_ids = set()
    for category in sorted(groups):
        members = sorted(groups[category])
        rng.shuffle(members)
        dev_ids.update(members[: round(len(members) * dev_ratio)])
    kept = [
        {**query, "split": "dev" if query["query_id"] in dev_ids else "test"}
        for query in queries
        if query["query_id"] in answerable
    ]
    return kept, dropped
