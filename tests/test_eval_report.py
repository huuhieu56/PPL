import json

from src.eval.report import FIGURES, TABLES, build_report, markdown_table
from src.io_utils import write_csv


def test_markdown_table_formats_numbers_and_missing_values():
    table = markdown_table([{"a": "C1", "b": 0.12345, "c": None}], ["a", "b", "c"], ["Cấu hình", "MRR", "X"])
    assert table.splitlines() == ["| Cấu hình | MRR | X |", "| --- | --- | --- |", "| C1 | 0.123 | – |"]


def _minimal_run(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    configs = {
        "C1": {"fusion": "none", "rerank": False, "rerank_n": 30, "tokenizer": "whitespace", "reranker_model": "r", "index_dir": "i", "index_version": "v1"},
        "C4-WS": {"fusion": "weighted", "rerank": True, "rerank_n": 30, "tokenizer": "whitespace", "reranker_model": "r", "index_dir": "i", "index_version": "v1"},
        "X5-N10": {"fusion": "weighted", "rerank": True, "rerank_n": 10, "tokenizer": "whitespace", "reranker_model": "r", "index_dir": "i", "index_version": "v1"},
    }
    (run_dir / "config.json").write_text(json.dumps({
        "split": "test", "frozen": {"alpha": 0.6, "rrf_k": 60, "adaptive_beta": 0.3, "rerank_n": 30, "best_single": "C2"},
        "lock": {"lock_violation": False}, "configs": configs,
        "indexes": {"i": {"embedding_model": "BAAI/bge-m3", "chunk_count": 120, "tokenizers": ["whitespace"]}},
    }), encoding="utf-8")
    metric = {"mrr@10": 0.5, "ndcg@10": 0.6, "recall@5": 0.7, "hit_rate@1": 0.4, "hit_rate@5": 0.8, "precision@5": 0.2}
    (run_dir / "metrics.json").write_text(json.dumps({
        "overall": {name: metric for name in configs},
        "by_category": {name: {"exact": {**metric, "n": 3}, "concept": {**metric, "n": 2}} for name in configs},
        "by_origin": {name: {"llm": {**metric, "n": 4}, "human": {**metric, "n": 1}} for name in configs},
        "n_queries": 5,
    }), encoding="utf-8")
    return run_dir


def test_report_skips_missing_inputs(tmp_path):
    run_dir = _minimal_run(tmp_path)
    written = build_report(run_dir, tmp_path / "bench")
    names = sorted(path.name for path in written)
    assert "table_3_5.md" in names and "table_3_6.md" in names and "table_3_11.md" in names
    assert "table_3_8.md" not in names and "table_3_2.md" not in names
    main = (run_dir / "report" / "table_3_5.md").read_text(encoding="utf-8")
    assert main.startswith(f"**Bảng 3.5. {TABLES['3.5']}**")
    assert "X5-N10" not in main
    assert "X5-N10" in (run_dir / "report" / "table_3_11.md").read_text(encoding="utf-8")
    assert (run_dir / "report" / "figure_3_3.png").exists()


def test_report_with_all_inputs_writes_every_table_and_figure(tmp_path):
    run_dir = _minimal_run(tmp_path)
    bench = tmp_path / "bench"
    bench.mkdir()
    (bench / "benchmark_description.json").write_text(json.dumps({
        "counts": [{"category": "exact", "origin": "llm", "split": "test", "queries": 3}],
        "relevant_per_query": [{"category": "exact", "mean": 1.5, "median": 1.0}],
        "pool_contribution": [{"system": "C1", "relevant_found": 4, "unique_relevant": 1}],
        "agreement": {"overlap": 20, "kappa": 0.71, "raw_agreement": 0.8},
        "totals": {"queries": 5, "qrels": 30, "relevant": 9},
    }), encoding="utf-8")
    write_csv(bench / "tune_results.csv", [
        {"param": "alpha", "value": value, "category": category, "metric": "mrr@10", "score": score}
        for value, score in ((0.0, 0.3), (0.5, 0.5), (1.0, 0.4))
        for category in ("all", "exact")
    ])
    write_csv(run_dir / "comparisons.csv", [
        {"family": "RQ3", "system": "C4-WS", "baseline": "C3-WS", "metric": "mrr@10", "category": "all", "n": 5,
         "mean_system": 0.6, "mean_baseline": 0.5, "diff": 0.1, "ci_low": 0.02, "ci_high": 0.2, "p_value": 0.01, "p_holm": 0.03},
    ])
    stage = {"mean": 10.0, "p50": 9.0, "p95": 20.0}
    (run_dir / "latency.json").write_text(json.dumps({
        "hardware": {"platform": "Linux", "cuda_device": "Tesla T4", "torch": "2.x", "python": "3.11"},
        "configs": {name: {s: stage for s in ("sparse", "dense", "fusion", "rerank", "total")} for name in ("C1", "C4-WS", "X5-N10")},
    }), encoding="utf-8")
    (run_dir / "errors_summary.json").write_text(json.dumps({
        "rows": [{"stage": "rerank_demoted", "cause": "tokenization", "count": 2}],
    }), encoding="utf-8")

    written = {path.name for path in build_report(run_dir, bench)}
    assert {f"table_3_{key.split('.')[1]}.md" for key in TABLES} <= written
    assert {f"figure_3_{key.split('.')[1]}.png" for key in FIGURES} <= written
    environment = (run_dir / "report" / "table_3_1.md").read_text(encoding="utf-8")
    assert "Tesla T4" in environment and "BAAI/bge-m3" in environment
