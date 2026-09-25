import json

import pytest

from src.bench.manifest import MANIFEST_NAME, check_test_lock, write_manifest
from src.bench.split import split_queries
from src.io_utils import write_jsonl

QUERIES = [{"query_id": f"e{i}", "category": "exact"} for i in range(6)] + [
    {"query_id": f"k{i}", "category": "concept"} for i in range(4)
]
QRELS = [{"query_id": query["query_id"], "chunk_id": "c", "relevance": 1} for query in QUERIES if query["query_id"] != "k3"]
QRELS.append({"query_id": "k3", "chunk_id": "c", "relevance": 0})


def test_split_is_stratified_deterministic_and_drops_unanswerable():
    queries, dropped = split_queries(QUERIES, QRELS, dev_ratio=0.3, seed=42)
    assert dropped == ["k3"]
    assert len(queries) == 9
    dev = [query for query in queries if query["split"] == "dev"]
    assert sorted(query["category"] for query in dev) == ["concept", "exact", "exact"]
    again, _ = split_queries(QUERIES, QRELS, dev_ratio=0.3, seed=42)
    assert [query["split"] for query in again] == [query["split"] for query in queries]


def test_test_lock_records_first_run_and_flags_later_changes(tmp_path):
    write_jsonl(tmp_path / "queries.jsonl", [{"query_id": "e0"}])
    write_jsonl(tmp_path / "qrels.jsonl", [{"query_id": "e0", "chunk_id": "c", "relevance": 1}])
    manifest = write_manifest(tmp_path, index_version="v1", seed=42, dev_ratio=0.3)
    assert set(manifest["files"]) == {"queries.jsonl", "qrels.jsonl"}
    frozen = tmp_path / "frozen_params.yaml"
    frozen.write_text("alpha: 0.6\n", encoding="utf-8")

    first = check_test_lock(tmp_path, frozen)
    assert first["lock_violation"] is False and first["benchmark_changed"] is False
    assert check_test_lock(tmp_path, frozen)["lock_violation"] is False

    frozen.write_text("alpha: 0.7\n", encoding="utf-8")
    assert check_test_lock(tmp_path, frozen)["lock_violation"] is True
    saved = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert saved["test_lock"]["frozen_params_sha256"] == first["frozen_params_sha256"]

    write_jsonl(tmp_path / "qrels.jsonl", [{"query_id": "e0", "chunk_id": "c", "relevance": 2}])
    assert check_test_lock(tmp_path, frozen)["benchmark_changed"] is True


def test_test_lock_requires_frozen_params(tmp_path):
    write_jsonl(tmp_path / "queries.jsonl", [])
    write_jsonl(tmp_path / "qrels.jsonl", [])
    write_manifest(tmp_path, index_version="v1", seed=42, dev_ratio=0.3)
    with pytest.raises(FileNotFoundError, match="eval tune"):
        check_test_lock(tmp_path, tmp_path / "frozen_params.yaml")
