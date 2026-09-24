import json

import pytest

from src.eval.errors import classify_failures, export_error_sample, summarize_causes
from src.io_utils import read_csv, write_csv, write_jsonl
from tests.fakes import make_chunk


def _row(query_id, results):
    return {"config": "C4-WS", "query_id": query_id, "alpha_used": 0.5, "timings_ms": {}, "results": results}


@pytest.fixture
def run_dir(tmp_path):
    directory = tmp_path / "run"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps({"configs": {"C4-WS": {"fusion": "weighted", "rerank": True}}}))
    (directory / "qrels_used.json").write_text(json.dumps({"default": {
        "ok": {"a": 2}, "miss": {"z": 2}, "fusion": {"d": 1}, "rerank": {"b": 2},
    }}))
    write_jsonl(directory / "per_query.jsonl", [
        _row("ok", [["a", 1.0, 1, 0.9, 1, 0.9, 3.0]]),
        _row("miss", [["a", 1.0, 1, 0.9, 1, 0.9, 3.0]]),
        _row("fusion", [["a", None, None, 0.9, 1, 0.9, 3.0]] + [["x", None, None, 0.1, 2, 0.1, None]] * 10 + [["d", 0.5, 40, None, None, 0.01, None]]),
        _row("rerank", [["a", 1.0, 1, 0.9, 1, 0.9, 3.0]] * 10 + [["b", 2.0, 1, 0.8, 2, 0.95, -1.0]]),
    ])
    return directory


def test_classify_failures_by_stage(run_dir):
    failures = {row["query_id"]: row for row in classify_failures(run_dir, "C4-WS", k=10)}
    assert set(failures) == {"miss", "fusion", "rerank"}
    assert failures["miss"]["stage"] == "first_stage_miss" and failures["miss"]["final_rank"] is None
    assert failures["fusion"]["stage"] == "fusion_demoted" and failures["fusion"]["sparse_rank"] == 40
    assert failures["rerank"]["stage"] == "rerank_demoted" and failures["rerank"]["final_rank"] == 11
    assert failures["rerank"]["top1_chunk_id"] == "a"


def test_export_sample_and_summarize_causes(tmp_path, run_dir):
    failures = classify_failures(run_dir, "C4-WS")
    queries = [{"query_id": q, "text": f"câu {q}", "category": "exact" if q == "miss" else "concept"} for q in ("miss", "fusion", "rerank")]
    chunks = {cid: make_chunk(cid, f"nội dung {cid}") for cid in ("a", "b", "d", "x", "z")}
    rows = export_error_sample(failures, queries, chunks, tmp_path / "sample.csv", sample=2, seed=1)
    assert len(rows) == 2 and {row["category"] for row in rows} == {"exact", "concept"}
    saved = read_csv(tmp_path / "sample.csv")
    saved[0]["cause"] = "tokenization"
    write_csv(tmp_path / "sample.csv", saved)
    summary = summarize_causes(tmp_path / "sample.csv")
    assert summary["by_cause"] == {"tokenization": 1, "unlabeled": 1}
    saved[1]["cause"] = "bad"
    write_csv(tmp_path / "sample.csv", saved)
    with pytest.raises(ValueError, match=saved[1]["query_id"]):
        summarize_causes(tmp_path / "sample.csv")
