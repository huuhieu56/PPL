import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from openai import OpenAI

from src.config import load_settings
from src.eval.runner import load_benchmark
from src.eval.spec import load_spec
from src.index import RetrievalIndex
from src.indexing import resolve_index_dir
from src.models import PipelineConfig
from src.pipeline import RetrievalPipeline
from src.rag import answer_question
from src.storage import Database


SYSTEMS = {
    "C2": PipelineConfig(sparse=False, dense=True, fusion="none", rerank=False),
    "C3-WS": PipelineConfig(fusion="weighted", alpha=0.5, rerank=False),
    "X2-R": PipelineConfig(fusion="adaptive", alpha=0.5, rerank=True),
}
SCORE_COLUMNS = ("correctness_1_5", "faithfulness_1_5", "citation_correct_0_1")


def generate(config_path: Path) -> Path:
    settings = load_settings()
    if not settings.openai_api_key or not settings.openai_model:
        raise ValueError("OPENAI_API_KEY and OPENAI_MODEL are required")
    spec = load_spec(config_path, settings)
    database = Database(settings.db_path)
    database.initialize()
    pipeline = RetrievalPipeline(RetrievalIndex.load(spec.index_dir or resolve_index_dir(None, settings, database)))
    queries, _, _ = load_benchmark(spec.bench_dir, "test")
    seed = spec.seed
    rng = random.Random(seed)
    output_dir = spec.runs_dir / (
        "rag-eval-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir.mkdir(parents=True)
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    rows, key = [], {}
    for query in queries:
        system_names = list(SYSTEMS)
        rng.shuffle(system_names)
        for position, system_name in enumerate(system_names):
            item_id = hashlib.sha256(
                f"{seed}|{query['query_id']}|{position}".encode()
            ).hexdigest()[:16]
            answer = answer_question(
                query["text"], pipeline, SYSTEMS[system_name], client, settings.openai_model
            )
            key[item_id] = {"query_id": query["query_id"], "system": system_name}
            rows.append(
                {
                    "item_id": item_id,
                    "query": query["text"],
                    "answer": answer.text,
                    "citations": json.dumps(answer.citations, ensure_ascii=False),
                    "latency_ms": round(answer.retrieval_ms + answer.generation_ms, 3),
                    "input_tokens": answer.prompt_tokens,
                    "output_tokens": answer.completion_tokens,
                    "correctness_1_5": "",
                    "faithfulness_1_5": "",
                    "citation_correct_0_1": "",
                    "notes": "",
                }
            )
    with (output_dir / "rag_answers_blinded.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "rag_answers_key.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return output_dir


def summarize(completed_csv: Path) -> Path:
    key_path = completed_csv.with_name("rag_answers_key.json")
    key = json.loads(key_path.read_text(encoding="utf-8"))
    values = defaultdict(lambda: defaultdict(list))
    with completed_csv.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            system = key[row["item_id"]]["system"]
            for column in SCORE_COLUMNS:
                if row[column] == "":
                    raise ValueError(f"Missing {column} for item {row['item_id']}")
                values[system][column].append(float(row[column]))
    summary = {
        system: {
            column: sum(scores) / len(scores) for column, scores in columns.items()
        }
        for system, columns in values.items()
    }
    output = completed_csv.with_name("rag_evaluation_summary.json")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate or summarize blinded RAG evaluation")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config", type=Path)
    group.add_argument("--summarize", type=Path)
    arguments = parser.parse_args()
    print(generate(arguments.config) if arguments.config else summarize(arguments.summarize))


if __name__ == "__main__":
    main()
