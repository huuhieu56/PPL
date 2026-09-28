"""Pool real retrieval candidates for blinded human relevance judgments."""

import argparse
import csv
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from src.config import load_yaml
from src.experiments import execute_query, select_queries
from src.retrieval import RetrievalIndex


def export_qrels(source: Path) -> Path:
    with source.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = []
    seen = set()
    for row in rows:
        key = (row["query_id"], row["chunk_id"])
        if key in seen or row.get("relevance") not in {"0", "1", "2"} or not row.get("reviewer", "").strip():
            raise ValueError(f"Missing/invalid judgment or reviewer, or duplicate pair: {key}")
        seen.add(key)
        labels.append({"query_id": key[0], "chunk_id": key[1], "relevance": int(row["relevance"])})
    if not labels:
        raise ValueError("No judgments")
    target = source.with_suffix(".qrels.jsonl")
    if target.exists():
        raise FileExistsError(target)
    target.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in labels), encoding="utf-8")
    return target


def build_pool(config_path: Path, split: str, depth: int) -> Path:
    if depth < 1:
        raise ValueError("pool depth must be positive")
    config = load_yaml(config_path)
    queries = select_queries(Path(config["queries"]), split)
    index = RetrievalIndex.load(config["index_dir"])
    if Path(config["index_dir"]).name != config["corpus_version"]:
        raise ValueError("Index does not match corpus version")
    directory = Path(config.get("runs_dir", "runs")) / ("judgments-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    directory.mkdir(parents=True)
    rows, provenance = [], {}
    for query in queries:
        candidates: dict[str, list[str]] = {}
        for method in config["experiments"]:
            result = execute_query(method, query, config)
            for chunk_id in result["ranked_chunk_ids"][:depth]:
                candidates.setdefault(chunk_id, []).append(method)
        identifiers = sorted(candidates)
        if not identifiers:
            raise ValueError(f"No candidates for {query['query_id']}; cannot create judgments")
        random.Random(f"{config.get('seed', 42)}:{query['query_id']}").shuffle(identifiers)
        for chunk_id in identifiers:
            chunk = index.chunks[chunk_id]
            rows.append({"query_id": query["query_id"], "chunk_id": chunk_id, "query": query["text"], "source": chunk.doc_id, "course": chunk.course, "section": chunk.section, "page": chunk.page, "page_end": chunk.page_end, "text": chunk.text, "relevance": "", "reviewer": "", "notes": ""})
        provenance[query["query_id"]] = candidates
        print(f"Pooled {query['query_id']}: {len(identifiers)} candidates", flush=True)
    with (directory / "judgments.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Keep this key away from raters: method identities must stay blinded.
    (directory / "pool_key.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    (directory / "protocol.json").write_text(json.dumps({"config": config, "split": split, "depth_per_method": depth, "query_sha256": hashlib.sha256(Path(config["queries"]).read_bytes()).hexdigest(), "index_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in sorted(Path(config["index_dir"]).glob("*")) if path.is_file()}, "limits": "Pooled judgments can miss relevant chunks. Use independently authored questions; do not tune on test judgments."}, ensure_ascii=False, indent=2), encoding="utf-8")
    return directory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path)
    group.add_argument("--judgments", type=Path)
    parser.add_argument("--split", choices=("dev", "test"))
    parser.add_argument("--depth", type=int, default=10)
    args = parser.parse_args()
    if args.config and not args.split:
        parser.error("--split is required with --config")
    print(build_pool(args.config, args.split, args.depth) if args.config else export_qrels(args.judgments))


if __name__ == "__main__":
    main()
