"""Reproducible out-of-domain retrieval check on VieQuADRetrieval."""

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as parquet
import yaml

from src.evaluation import evaluate_rankings, paired_bootstrap_ci
from src.models import Chunk, SearchResult
from src.retrieval import RetrievalIndex, adaptive_alpha, fuse_rrf, fuse_weighted, self_idf


REVISION = "f956535"
DATA_DIR = Path("data/external/VieQuADRetrieval")
INDEX_DIR = Path("data/indexes") / f"viequad_{REVISION}"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def rows(part: str) -> list[dict]:
    source = DATA_DIR / part / "validation-00000-of-00001.parquet"
    if not source.is_file():
        raise FileNotFoundError(f"Download the pinned VieQuAD files first: {source}")
    return parquet.read_table(source).to_pylist()


def ranked(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda key: (-scores[key], key))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", type=int, default=100)
    parser.add_argument("--test", type=int, default=200)
    args = parser.parse_args()
    corpus = rows("corpus")
    queries = rows("queries")
    judgments = rows("qrels")
    if args.dev < 1 or args.test < 1 or args.dev + args.test > len(queries):
        raise ValueError("Invalid dev/test sizes")
    chunks = [Chunk(row["_id"], row["_id"], "VieQuAD", "benchmark", 1, row["title"], f"{row['title']}\n{row['text']}") for row in corpus]
    if (INDEX_DIR / "index_meta.json").exists():
        index = RetrievalIndex.load(INDEX_DIR)
    else:
        index = RetrievalIndex.build(chunks, INDEX_DIR, "whitespace", "BAAI/bge-m3")
    if set(index.chunks) != {chunk.chunk_id for chunk in chunks}:
        raise ValueError("VieQuAD index does not match pinned corpus")
    qrels: dict[str, dict[str, int]] = {}
    for row in judgments:
        qrels.setdefault(row["query-id"], {})[row["corpus-id"]] = row["score"]
    title_by_id = {row["_id"]: row["title"] for row in corpus}
    by_title: dict[str, list[dict]] = defaultdict(list)
    for query in queries:
        titles = {title_by_id[chunk_id] for chunk_id in qrels.get(query["_id"], {})}
        if len(titles) == 1:
            by_title[titles.pop()].append(query)
    ordered_titles = sorted(by_title, key=lambda title: hashlib.sha256(f"42:{title}".encode()).digest())
    dev_titles: list[str] = []
    for title in ordered_titles:
        if len(dev_titles) >= max(1, len(ordered_titles) // 3) and sum(len(by_title[name]) for name in dev_titles) >= args.dev:
            break
        dev_titles.append(title)
    test_titles = [title for title in ordered_titles if title not in dev_titles]
    def sample(titles: list[str], count: int) -> list[dict]:
        pool = [query for title in titles for query in by_title[title]]
        pool.sort(key=lambda query: hashlib.sha256(f"42:{query['_id']}".encode()).digest())
        if len(pool) < count:
            raise ValueError("Too few article-disjoint queries for the requested split")
        return pool[:count]

    dev_queries = sample(dev_titles, args.dev)
    test_queries = sample(test_titles, args.test)
    selected = dev_queries + test_queries
    rankings = {name: {} for name in ["bm25", "dense", "rrf", "adaptive", *(f"alpha_{alpha:.2f}" for alpha in ALPHAS)]}
    idf = self_idf(index)
    for number, query in enumerate(selected, 1):
        query_id = query["_id"]
        sparse = index.sparse_scores(query["text"], 100)
        dense = index.dense_scores(query["text"], 100)
        rankings["bm25"][query_id] = ranked(sparse)
        rankings["dense"][query_id] = ranked(dense)
        result_lists = [
            [SearchResult(index.chunks[key], scores[key], rank, source) for rank, key in enumerate(ranked(scores), 1)]
            for source, scores in (("bm25", sparse), ("dense", dense))
        ]
        rankings["rrf"][query_id] = [item.chunk.chunk_id for item in fuse_rrf(result_lists)]
        for alpha in ALPHAS:
            rankings[f"alpha_{alpha:.2f}"][query_id] = [item.chunk.chunk_id for item in fuse_weighted(sparse, dense, index.chunks, alpha)]
        alpha, _ = adaptive_alpha(query["text"], 0.5, 0.3, idf)
        rankings["adaptive"][query_id] = [item.chunk.chunk_id for item in fuse_weighted(sparse, dense, index.chunks, alpha)]
        if number % 25 == 0:
            print(f"Retrieved {number}/{len(selected)}", flush=True)

    dev_ids = [query["_id"] for query in dev_queries]
    test_ids = [query["_id"] for query in test_queries]

    def metrics(name: str, ids: list[str]) -> dict[str, float]:
        return evaluate_rankings({query_id: rankings[name][query_id] for query_id in ids}, qrels)

    dev_metrics = {name: metrics(name, dev_ids) for name in rankings}
    chosen = max((f"alpha_{alpha:.2f}" for alpha in ALPHAS), key=lambda name: (dev_metrics[name]["ndcg@10"], -float(name.split("_")[1])))
    test_metrics = {name: metrics(name, test_ids) for name in ("bm25", "dense", "rrf", "adaptive", chosen)}
    paired = {}
    test_groups = [next(iter({title_by_id[chunk_id] for chunk_id in qrels[query_id]})) for query_id in test_ids]
    for baseline in ("bm25", "dense", "rrf"):
        selected_scores = [metrics(chosen, [query_id])["ndcg@10"] for query_id in test_ids]
        baseline_scores = [metrics(baseline, [query_id])["ndcg@10"] for query_id in test_ids]
        paired[baseline] = {"delta_ndcg@10": test_metrics[chosen]["ndcg@10"] - test_metrics[baseline]["ndcg@10"], "ci95_clustered_by_source_title": paired_bootstrap_ci(selected_scores, baseline_scores, groups=test_groups)}
    output = {
        "source": "mteb/VieQuADRetrieval",
        "revision": REVISION,
        "corpus_size": len(corpus),
        "all_queries": len(queries),
        "dev_source_titles": dev_titles,
        "test_source_titles": test_titles,
        "excluded_multi_title_queries": len(queries) - sum(map(len, by_title.values())),
        "dev_query_ids": dev_ids,
        "test_query_ids": test_ids,
        "selection_metric": "dev ndcg@10",
        "index_dir": str(INDEX_DIR),
        "embedding_model": index.embedding_model,
        "embedding_revision": index.embedding_revision,
        "tokenizer": index.tokenizer_mode,
        "candidate_limit_per_retriever": 100,
        "seed": 42,
        "chosen": chosen,
        "dev": dev_metrics,
        "test": test_metrics,
        "paired_test": paired,
        "limits": "Wikipedia passages, not learning documents; qrels may omit other relevant passages.",
    }
    run_dir = Path("runs") / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-viequad")
    run_dir.mkdir(parents=True)
    (run_dir / "results.json").write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    (run_dir / "queries.jsonl").write_text("".join(json.dumps({"query_id": query["_id"], "text": query["text"], "split": "dev" if query["_id"] in dev_ids else "test", "group_id": next(iter({title_by_id[chunk_id] for chunk_id in qrels[query['_id']]}))}, ensure_ascii=False) + "\n" for query in selected), encoding="utf-8")
    (run_dir / "qrels.jsonl").write_text("".join(json.dumps({"query_id": query["_id"], "chunk_id": chunk_id, "relevance": grade}) + "\n" for query in selected for chunk_id, grade in qrels[query["_id"]].items()), encoding="utf-8")
    (run_dir / "retrieval.yaml").write_text(yaml.safe_dump({"corpus_version": INDEX_DIR.name, "index_dir": str(INDEX_DIR), "queries": str(run_dir / "queries.jsonl"), "qrels": str(run_dir / "qrels.jsonl"), "runs_dir": "runs", "seed": 42, "retrieval": {"top_l": 100}, "experiments": {"E0": {"method": "bm25"}, "E1": {"method": "dense"}, "E2": {"method": "rrf", "rrf_k": 60}, "E3": {"method": "weighted", "alpha": float(chosen.split('_')[1])}, "E6": {"method": "adaptive", "alpha": 0.5, "adaptive_beta": 0.3}}}, sort_keys=False), encoding="utf-8")
    with (run_dir / "per_query.jsonl").open("w", encoding="utf-8") as stream:
        for query in selected:
            query_id = query["_id"]
            for method, method_rankings in rankings.items():
                stream.write(json.dumps({"query_id": query_id, "query": query["text"], "split": "dev" if query_id in dev_ids else "test", "source_title": next(iter({title_by_id[chunk_id] for chunk_id in qrels[query_id]})), "method": method, "relevant_ids": qrels[query_id], "top10": method_rankings[query_id][:10]}, ensure_ascii=False) + "\n")
    print(json.dumps({"run_dir": str(run_dir), "chosen": chosen, "test": test_metrics, "paired_test": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
