import numpy as np

from src.bench.checks import contains_quote
from src.io_utils import read_csv, write_csv
from src.text import normalize_text


def weighted_kappa(first: list[int], second: list[int], labels=(0, 1, 2)) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("kappa needs two non-empty label lists of equal length")
    position = {label: row for row, label in enumerate(labels)}
    size = len(labels)
    observed = np.zeros((size, size))
    for left, right in zip(first, second):
        observed[position[left], position[right]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    grid = np.arange(size)
    weights = (grid[:, None] - grid[None, :]) ** 2 / (size - 1) ** 2
    expected_disagreement = float((weights * expected).sum())
    observed_disagreement = float((weights * observed).sum())
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0
    return 1.0 - observed_disagreement / expected_disagreement


def parse_relevance(value: str, pool_id: str) -> int | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        number = -1.0
    if number not in (0.0, 1.0, 2.0):
        raise ValueError(f"Invalid relevance '{value}' for pool_id {pool_id}; use 0, 1 or 2")
    return int(number)


def read_annotations(path) -> dict[str, dict]:
    return {
        row["pool_id"]: {
            "relevance": parse_relevance(row.get("relevance", ""), row["pool_id"]),
            "evidence_quote": normalize_text(row.get("evidence_quote", "")),
        }
        for row in read_csv(path)
    }


def agreement_report(first: dict, second: dict) -> dict:
    shared = sorted(
        pool_id
        for pool_id in set(first) & set(second)
        if first[pool_id]["relevance"] is not None and second[pool_id]["relevance"] is not None
    )
    if not shared:
        return {"overlap": 0, "kappa": None, "raw_agreement": None, "disagreements": []}
    left = [first[pool_id]["relevance"] for pool_id in shared]
    right = [second[pool_id]["relevance"] for pool_id in shared]
    return {
        "overlap": len(shared),
        "kappa": weighted_kappa(left, right),
        "raw_agreement": sum(a == b for a, b in zip(left, right)) / len(shared),
        "disagreements": [pool_id for pool_id, a, b in zip(shared, left, right) if a != b],
    }


def write_disagreements(path, report, pool, first, second, queries, index) -> None:
    entries = {entry["pool_id"]: entry for entry in pool}
    texts = {query["query_id"]: query["text"] for query in queries}
    rows = [
        {
            "pool_id": pool_id,
            "query": texts[entries[pool_id]["query_id"]],
            "chunk_text": index.chunks[entries[pool_id]["chunk_id"]].body,
            "label_a": first[pool_id]["relevance"],
            "label_b": second[pool_id]["relevance"],
            "final": "",
        }
        for pool_id in report["disagreements"]
    ]
    write_csv(path, rows, ["pool_id", "query", "chunk_text", "label_a", "label_b", "final"])


def merge_labels(pool: list[dict], annotations: list[dict], index, resolved_path=None):
    resolved = {}
    if resolved_path:
        for row in read_csv(resolved_path):
            final = parse_relevance(row.get("final", ""), row["pool_id"])
            if final is not None:
                resolved[row["pool_id"]] = final
    qrels, evidence, warnings, unresolved = [], [], [], []
    for entry in pool:
        pool_id = entry["pool_id"]
        labels = [item[pool_id]["relevance"] for item in annotations if pool_id in item and item[pool_id]["relevance"] is not None]
        if pool_id in resolved:
            final = resolved[pool_id]
        elif labels and len(set(labels)) == 1:
            final = labels[0]
        elif labels:
            unresolved.append(pool_id)
            continue
        else:
            continue
        qrels.append({"query_id": entry["query_id"], "chunk_id": entry["chunk_id"], "relevance": final})
        if final < 1:
            continue
        quote = next((item[pool_id]["evidence_quote"] for item in annotations if pool_id in item and item[pool_id]["evidence_quote"]), "")
        if not quote:
            continue
        if not contains_quote(quote, index.chunks[entry["chunk_id"]].body):
            warnings.append(f"Evidence quote for pool_id {pool_id} is not found in chunk {entry['chunk_id']}")
            continue
        evidence.append(
            {"query_id": entry["query_id"], "doc_id": entry["doc_id"], "page": entry["page"], "quote": quote, "relevance": final}
        )
    if unresolved:
        raise ValueError(f"Unresolved disagreements (fill 'final' in disagreements.csv): {', '.join(unresolved)}")
    return qrels, evidence, warnings
