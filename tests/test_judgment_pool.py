import csv
import json

import pytest

from build_judgment_pool import export_qrels
from run_rag_evaluation import summarize
from tune_retrieval import tune
import yaml


def test_unjudged_pool_cannot_become_gold_labels(tmp_path):
    source = tmp_path / "pool.csv"
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query_id", "chunk_id", "relevance", "reviewer"])
        writer.writeheader()
        writer.writerow({"query_id": "q1", "chunk_id": "c1", "relevance": "", "reviewer": ""})
    with pytest.raises(ValueError, match="judgment|reviewer"):
        export_qrels(source)
    assert not source.with_suffix(".qrels.jsonl").exists()
    with source.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["query_id", "chunk_id", "relevance", "reviewer"])
        writer.writeheader()
        writer.writerow({"query_id": "q1", "chunk_id": "c1", "relevance": "2", "reviewer": "human-a"})
    target = export_qrels(source)
    assert json.loads(target.read_text()) == {"query_id": "q1", "chunk_id": "c1", "relevance": 2}


def test_answer_evaluation_rejects_missing_blinded_rows(tmp_path):
    source = tmp_path / "rag_answers_blinded.csv"
    source.write_text("item_id,correctness_1_5,faithfulness_1_5,citation_correct_0_1\na,5,5,1\n")
    source.with_name("rag_answers_key.json").write_text(json.dumps({"a": {"system": "E1"}, "b": {"system": "E3"}}))
    with pytest.raises(ValueError, match="missing|complete"):
        summarize(source)


def test_tuning_cannot_use_a_test_only_benchmark(tmp_path):
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"query_id":"test1","text":"independent question","split":"test"}\n')
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text('{"query_id":"test1","chunk_id":"c1","relevance":2}\n')
    index = tmp_path / "corpus"
    index.mkdir()
    (index / "chunks.jsonl").write_text('{"chunk_id":"c1"}\n')
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"queries": str(queries), "qrels": str(qrels), "corpus_version": "corpus", "index_dir": str(index), "experiments": {"E3": {"method": "weighted"}}}))
    with pytest.raises(ValueError, match="split=dev"):
        tune(config, "E3", [0, 0.5, 1], [0])
