from src.bench.describe import describe_benchmark

QUERIES = [
    {"query_id": "q1", "category": "exact", "origin": "llm", "split": "dev"},
    {"query_id": "q2", "category": "exact", "origin": "human", "split": "test"},
    {"query_id": "q3", "category": "concept", "origin": "llm", "split": "test"},
]
QRELS = [
    {"query_id": "q1", "chunk_id": "a", "relevance": 2},
    {"query_id": "q1", "chunk_id": "b", "relevance": 1},
    {"query_id": "q2", "chunk_id": "a", "relevance": 1},
    {"query_id": "q3", "chunk_id": "c", "relevance": 2},
    {"query_id": "q3", "chunk_id": "d", "relevance": 0},
]
POOL = [
    {"query_id": "q1", "chunk_id": "a", "systems": ["C1", "C2"]},
    {"query_id": "q1", "chunk_id": "b", "systems": ["C1"]},
    {"query_id": "q2", "chunk_id": "a", "systems": ["C2"]},
    {"query_id": "q3", "chunk_id": "c", "systems": ["manual"]},
    {"query_id": "q3", "chunk_id": "d", "systems": ["C1"]},
]


def test_describe_counts_relevance_and_pool_contribution():
    report = describe_benchmark(QUERIES, QRELS, POOL, {"kappa": 0.8})
    assert {"category": "exact", "origin": "human", "split": "test", "queries": 1} in report["counts"]
    exact = next(row for row in report["relevant_per_query"] if row["category"] == "exact")
    assert exact == {"category": "exact", "mean": 1.5, "median": 1.5}
    contribution = {row["system"]: row for row in report["pool_contribution"]}
    assert contribution["C1"] == {"system": "C1", "relevant_found": 2, "unique_relevant": 1}
    assert contribution["C2"] == {"system": "C2", "relevant_found": 2, "unique_relevant": 1}
    assert contribution["manual"]["unique_relevant"] == 1
    assert report["totals"] == {"queries": 3, "qrels": 5, "relevant": 4}
    assert report["agreement"] == {"kappa": 0.8}


def test_describe_without_pool():
    report = describe_benchmark(QUERIES, QRELS)
    assert report["pool_contribution"] == []
