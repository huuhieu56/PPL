"""Publish one frozen retrieval split to LangSmith and evaluate a selected method."""

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from src.config import load_yaml
from src.evaluation import evaluate_rankings
from src.experiments import execute_query, validate_config, select_queries


def prepare_examples(config: dict, split: str) -> list[dict]:
    queries = select_queries(Path(config["queries"]), split)
    qrels: dict[str, dict[str, int]] = {}
    for line in Path(config["qrels"]).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            qrels.setdefault(row["query_id"], {})[row["chunk_id"]] = int(row["relevance"])
    return [
        {"inputs": {"query_id": row["query_id"], "text": row["text"]}, "outputs": {"qrels": qrels[row["query_id"]]}}
        for row in queries
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval in LangSmith")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_yaml(args.config)
    validate_config(config, require_index=True)
    if args.experiment not in config["experiments"]:
        parser.error(f"Không có experiment {args.experiment} trong config")
    examples = prepare_examples(config, args.split)
    hashes = {key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest() for key in ("queries", "qrels")}
    summary = {"corpus_version": config["corpus_version"], "experiment": args.experiment, "split": args.split, "questions": len(examples), "sha256": hashes}
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    load_dotenv()
    if not os.getenv("LANGSMITH_API_KEY"):
        parser.error("Cần LANGSMITH_API_KEY trong môi trường hoặc .env")
    from langsmith import Client

    client = Client()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    dataset = client.create_dataset(dataset_name=f"ppl-{config['corpus_version']}-{args.split}-{stamp}", description="Frozen retrieval questions and qrels; not answer-level gold labels")
    client.create_examples(dataset_id=dataset.id, examples=examples)

    def target(inputs: dict) -> dict:
        return execute_query(args.experiment, inputs, config)

    def score(metric: str):
        def evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> float:
            query_id = inputs["query_id"]
            return evaluate_rankings({query_id: outputs["ranked_chunk_ids"]}, {query_id: reference_outputs["qrels"]})[metric]
        evaluator.__name__ = metric.replace("@", "_at_")
        return evaluator

    client.evaluate(target, data=dataset.name, evaluators=[score("ndcg@10"), score("mrr@10"), score("hit_rate@1")], experiment_prefix=args.experiment, max_concurrency=1, metadata=summary)
    print(json.dumps({"dataset": dataset.name, **summary}, ensure_ascii=False))


if __name__ == "__main__":
    main()
