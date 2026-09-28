import json

from run_langsmith_eval import prepare_examples


def test_langsmith_examples_use_only_requested_split(tmp_path):
    queries = tmp_path / "queries.jsonl"
    qrels = tmp_path / "qrels.jsonl"
    queries.write_text("\n".join(json.dumps(row) for row in [
        {"query_id": "d1", "text": "Câu dev", "split": "dev"},
        {"query_id": "t1", "text": "Câu test", "split": "test"},
    ]), encoding="utf-8")
    qrels.write_text("\n".join(json.dumps(row) for row in [
        {"query_id": "d1", "chunk_id": "a", "relevance": 2},
        {"query_id": "t1", "chunk_id": "b", "relevance": 2},
    ]), encoding="utf-8")
    examples = prepare_examples({"queries": str(queries), "qrels": str(qrels)}, "dev")
    assert examples == [{"inputs": {"query_id": "d1", "text": "Câu dev"}, "outputs": {"qrels": {"a": 2}}}]
