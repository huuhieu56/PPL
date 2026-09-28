import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


from src.config import load_settings, load_yaml
from src.experiments import experiment_rag_config, select_queries, validate_config
from src.rag import answer_question, chat_model
from src.retrieval import RetrievalIndex


SCORE_COLUMNS = ("correctness_1_5", "faithfulness_1_5", "citation_correct_0_1")


def generate(config_path: Path, split: str) -> Path:
    config = load_yaml(config_path)
    validate_config(config, require_index=True)
    queries = select_queries(Path(config["queries"]), split)
    settings = load_settings()
    if not settings.openai_api_key or not settings.openai_model:
        raise ValueError("OPENAI_API_KEY and OPENAI_MODEL are required")
    index = RetrievalIndex.load(config["index_dir"])
    seed = int(config.get("seed", 42))
    rng = random.Random(seed)
    output_dir = Path(config.get("runs_dir", settings.runs_dir)) / (
        "rag-eval-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    output_dir.mkdir(parents=True)
    (output_dir / "config.json").write_text(json.dumps({**config, "evaluated_split": split}, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "input_hashes.json").write_text(json.dumps({key: hashlib.sha256(Path(config[key]).read_bytes()).hexdigest() for key in ("queries", "qrels")}), encoding="utf-8")
    client = chat_model(settings)
    rows, key = [], {}
    for query in queries:
        system_names = list(config["experiments"])
        rng.shuffle(system_names)
        for position, system_name in enumerate(system_names):
            item_id = hashlib.sha256(
                f"{seed}|{query['query_id']}|{position}".encode()
            ).hexdigest()[:16]
            answer = answer_question(
                query["text"], index, experiment_rag_config(system_name, config), client, settings.openai_model
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
        rows = list(csv.DictReader(stream))
        ids = [row["item_id"] for row in rows]
        if len(ids) != len(set(ids)) or set(ids) != set(key):
            raise ValueError("Evaluation is incomplete: missing, duplicate or unknown items")
        for row in rows:
            system = key[row["item_id"]]["system"]
            for column in SCORE_COLUMNS:
                if row[column] == "":
                    raise ValueError(f"Missing {column} for item {row['item_id']}")
                value = float(row[column])
                allowed = {0, 1} if column == "citation_correct_0_1" else {1, 2, 3, 4, 5}
                if value not in allowed:
                    raise ValueError(f"Invalid {column} for item {row['item_id']}")
                values[system][column].append(value)
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
    parser.add_argument("--split", choices=("dev", "test"))
    arguments = parser.parse_args()
    if arguments.config and not arguments.split:
        parser.error("--split dev/test is required with --config")
    print(generate(arguments.config, arguments.split) if arguments.config else summarize(arguments.summarize))


if __name__ == "__main__":
    main()
