import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from src.io_utils import read_csv, read_jsonl, write_csv

STAGES = ("first_stage_miss", "fusion_demoted", "rerank_demoted", "ranked_below_k")
CAUSES = ("extraction", "chunk_boundary", "tokenization", "vocabulary_mismatch", "multi_hop", "label_error", "other")
SAMPLE_COLUMNS = [
    "query_id", "category", "query", "stage", "final_rank", "sparse_rank", "dense_rank",
    "relevant_chunk", "top1_chunk", "cause",
]


def classify_failures(run_dir, target: str, k: int = 10) -> list[dict]:
    directory = Path(run_dir)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))["configs"][target]
    qrels_used = json.loads((directory / "qrels_used.json").read_text(encoding="utf-8"))
    qrels = qrels_used.get(target, qrels_used["default"])
    failures = []
    for row in read_jsonl(directory / "per_query.jsonl"):
        if row["config"] != target:
            continue
        relevant = {chunk_id: grade for chunk_id, grade in qrels.get(row["query_id"], {}).items() if grade > 0}
        ids = [item[0] for item in row["results"]]
        if not relevant or any(chunk_id in relevant for chunk_id in ids[:k]):
            continue
        positions = {item[0]: (rank, item) for rank, item in enumerate(row["results"], start=1)}
        found = [chunk_id for chunk_id in relevant if chunk_id in positions]
        top1 = ids[0] if ids else None
        if not found:
            best = max(relevant, key=lambda chunk_id: (relevant[chunk_id], chunk_id))
            failures.append({"query_id": row["query_id"], "stage": "first_stage_miss", "chunk_id": best,
                             "grade": relevant[best], "final_rank": None, "sparse_rank": None, "dense_rank": None,
                             "rerank_score": None, "top1_chunk_id": top1})
            continue
        best = min(found, key=lambda chunk_id: (-relevant[chunk_id], positions[chunk_id][0]))
        rank, item = positions[best]
        if config["rerank"]:
            stage = "rerank_demoted" if item[6] is not None else "fusion_demoted"
        else:
            stage = "fusion_demoted" if config["fusion"] != "none" else "ranked_below_k"
        failures.append({"query_id": row["query_id"], "stage": stage, "chunk_id": best, "grade": relevant[best],
                         "final_rank": rank, "sparse_rank": item[2], "dense_rank": item[4],
                         "rerank_score": item[6], "top1_chunk_id": top1})
    return failures


def export_error_sample(failures, queries, chunks: dict, path, sample: int = 50, seed: int = 42) -> list[dict]:
    info = {query["query_id"]: query for query in queries}
    groups = defaultdict(list)
    for failure in sorted(failures, key=lambda item: item["query_id"]):
        groups[info[failure["query_id"]]["category"]].append(failure)
    rng = random.Random(seed)
    for members in groups.values():
        rng.shuffle(members)
    picked = []
    while len(picked) < sample and any(groups.values()):
        for category in sorted(groups):
            if groups[category] and len(picked) < sample:
                picked.append(groups[category].pop(0))
    rows = [
        {
            "query_id": failure["query_id"],
            "category": info[failure["query_id"]]["category"],
            "query": info[failure["query_id"]]["text"],
            "stage": failure["stage"],
            "final_rank": failure["final_rank"],
            "sparse_rank": failure["sparse_rank"],
            "dense_rank": failure["dense_rank"],
            "relevant_chunk": chunks[failure["chunk_id"]].body if failure["chunk_id"] in chunks else "",
            "top1_chunk": chunks[failure["top1_chunk_id"]].body if failure["top1_chunk_id"] in chunks else "",
            "cause": "",
        }
        for failure in picked
    ]
    write_csv(path, rows, SAMPLE_COLUMNS)
    return rows


def summarize_causes(path) -> dict:
    by_stage, by_cause, pairs = Counter(), Counter(), Counter()
    for row in read_csv(path):
        cause = row.get("cause", "").strip() or "unlabeled"
        if cause != "unlabeled" and cause not in CAUSES:
            raise ValueError(f"Unknown cause '{cause}' for {row['query_id']}; use one of {CAUSES}")
        by_stage[row["stage"]] += 1
        by_cause[cause] += 1
        pairs[(row["stage"], cause)] += 1
    return {
        "by_stage": dict(by_stage),
        "by_cause": dict(by_cause),
        "rows": [{"stage": stage, "cause": cause, "count": count} for (stage, cause), count in sorted(pairs.items())],
    }
