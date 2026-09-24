import json
from collections import defaultdict
from pathlib import Path

from src.io_utils import read_csv, write_csv

TABLES = {
    "3.1": "Môi trường và thiết lập thực nghiệm",
    "3.2": "Thống kê bộ benchmark theo nhóm, nguồn và tập",
    "3.3": "Độ đồng thuận dán nhãn và đóng góp của từng hệ thống vào pool",
    "3.4": "Tham số tối ưu chọn trên tập dev",
    "3.5": "Kết quả tổng thể trên tập test",
    "3.6": "Kết quả theo nhóm truy vấn trên tập test (RQ1)",
    "3.7": "So sánh ghép cặp có kiểm định thống kê (RQ2, RQ3, X2)",
    "3.8": "Độ trễ theo tầng xử lý (ms)",
    "3.9": "Kết quả theo nguồn câu hỏi (phân tích độ nhạy)",
    "3.10": "Phân bố truy vấn thất bại theo tầng và nguyên nhân",
    "3.11": "Kết quả các hướng khai thác X3–X6",
}
FIGURES = {
    "3.1": "Ảnh hưởng của α đến chỉ số chính trên tập dev",
    "3.2": "Đánh đổi chất lượng – độ trễ theo số ứng viên rerank N",
    "3.3": "Chỉ số chính theo nhóm truy vấn trên tập test",
}
MAIN_METRICS = ["mrr@10", "ndcg@10", "recall@5", "hit_rate@1", "hit_rate@5", "precision@5"]
EXPLORATION_PREFIXES = ("X3", "X4", "X5", "X6")


def markdown_table(rows: list[dict], columns: list[str], headers: list[str] | None = None, digits: int = 3) -> str:
    def cell(value) -> str:
        if value is None or value == "":
            return "–"
        if isinstance(value, float):
            return f"{value:.{digits}f}"
        return str(value)

    lines = [
        "| " + " | ".join(headers or columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines += ["| " + " | ".join(cell(row.get(column)) for column in columns) + " |" for row in rows]
    return "\n".join(lines)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _table(report_dir: Path, key: str, rows: list[dict], columns: list[str], headers: list[str] | None = None) -> list[Path]:
    if not rows:
        return []
    suffix = key.split(".")[1]
    markdown = report_dir / f"table_3_{suffix}.md"
    markdown.write_text(f"**Bảng {key}. {TABLES[key]}**\n\n{markdown_table(rows, columns, headers)}\n", encoding="utf-8")
    write_csv(report_dir / f"table_3_{suffix}.csv", rows, columns)
    return [markdown]


def _figure(report_dir: Path, key: str, draw) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7, 4))
    drawn = draw(axis)
    if not drawn:
        plt.close(figure)
        return []
    axis.set_title(f"Hình {key}. {FIGURES[key]}", fontsize=10)
    figure.tight_layout()
    path = report_dir / f"figure_3_{key.split('.')[1]}.png"
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return [path]


def build_report(run_dir, bench_dir, primary_metric: str = "mrr@10") -> list[Path]:
    run = Path(run_dir)
    bench = Path(bench_dir)
    report_dir = run / "report"
    report_dir.mkdir(exist_ok=True)
    config = _load_json(run / "config.json") or {}
    metrics = _load_json(run / "metrics.json") or {"overall": {}, "by_category": {}, "by_origin": {}}
    latency = _load_json(run / "latency.json")
    description = _load_json(bench / "benchmark_description.json")
    errors = _load_json(run / "errors_summary.json")
    configs = config.get("configs", {})
    main_names = [name for name in metrics["overall"] if not name.startswith(EXPLORATION_PREFIXES)]
    exploration = [name for name in metrics["overall"] if name.startswith(EXPLORATION_PREFIXES)]
    metric_columns = [metric for metric in MAIN_METRICS if any(metric in values for values in metrics["overall"].values())]
    written: list[Path] = []

    environment = [
        {"item": "Tập đánh giá", "value": config.get("split")},
        {"item": "Số truy vấn", "value": metrics.get("n_queries")},
        {"item": "Vi phạm khóa test", "value": (config.get("lock") or {}).get("lock_violation")},
    ]
    for directory, meta in (config.get("indexes") or {}).items():
        environment += [
            {"item": f"Index {Path(directory).name} — mô hình embedding", "value": meta.get("embedding_model")},
            {"item": f"Index {Path(directory).name} — số chunk", "value": meta.get("chunk_count")},
            {"item": f"Index {Path(directory).name} — tokenizer", "value": ", ".join(meta.get("tokenizers", []))},
        ]
    rerankers = sorted({values.get("reranker_model") for values in configs.values() if values.get("rerank")})
    environment.append({"item": "Mô hình reranker", "value": ", ".join(item for item in rerankers if item)})
    if latency:
        environment += [{"item": f"Phần cứng — {key}", "value": value} for key, value in latency["hardware"].items()]
    written += _table(report_dir, "3.1", environment, ["item", "value"], ["Thành phần", "Giá trị"])

    if description:
        written += _table(report_dir, "3.2", description["counts"], ["category", "origin", "split", "queries"],
                          ["Nhóm", "Nguồn", "Tập", "Số câu hỏi"])
        agreement = description.get("agreement") or {}
        rows = [{"item": "Cohen's κ (trọng số bậc hai)", "value": agreement.get("kappa")},
                {"item": "Tỉ lệ đồng ý thô", "value": agreement.get("raw_agreement")},
                {"item": "Số cặp dán nhãn đôi", "value": agreement.get("overlap")}]
        rows += [{"item": f"Pool — {row['system']}: tìm thấy / duy nhất", "value": f"{row['relevant_found']} / {row['unique_relevant']}"}
                 for row in description.get("pool_contribution", [])]
        written += _table(report_dir, "3.3", rows, ["item", "value"], ["Chỉ tiêu", "Giá trị"])

    frozen = config.get("frozen") or {}
    written += _table(report_dir, "3.4", [{"param": key, "value": value} for key, value in frozen.items()
                                          if key in ("alpha", "rrf_k", "adaptive_beta", "rerank_n", "best_single")],
                      ["param", "value"], ["Tham số", "Giá trị"])
    written += _table(report_dir, "3.5", [{"config": name, **metrics["overall"][name]} for name in main_names],
                      ["config", *metric_columns], ["Cấu hình", *metric_columns])

    categories = sorted({category for groups in metrics["by_category"].values() for category in groups})
    written += _table(report_dir, "3.6", [
        {"config": name, **{category: metrics["by_category"][name].get(category, {}).get(primary_metric) for category in categories}}
        for name in main_names
    ], ["config", *categories], ["Cấu hình", *categories])

    comparisons = read_csv(run / "comparisons.csv") if (run / "comparisons.csv").exists() else []
    comparison_rows = [
        {**row, **{key: float(row[key]) for key in ("diff", "ci_low", "ci_high", "p_value") if row[key] != ""},
         "p_holm": float(row["p_holm"]) if row["p_holm"] != "" else None}
        for row in comparisons if row["category"] == "all"
    ]
    written += _table(report_dir, "3.7", comparison_rows,
                      ["family", "system", "baseline", "metric", "diff", "ci_low", "ci_high", "p_value", "p_holm"],
                      ["Họ", "Hệ thống", "Đối chứng", "Chỉ số", "Chênh lệch", "CI thấp", "CI cao", "p", "p (Holm)"])

    if latency:
        written += _table(report_dir, "3.8", [
            {"config": name, "sparse": stages["sparse"]["mean"], "dense": stages["dense"]["mean"],
             "rerank": stages["rerank"]["mean"], "total": stages["total"]["mean"], "total_p95": stages["total"]["p95"]}
            for name, stages in latency["configs"].items()
        ], ["config", "sparse", "dense", "rerank", "total", "total_p95"],
            ["Cấu hình", "BM25", "Dense", "Rerank", "Tổng (TB)", "Tổng (P95)"])

    origins = sorted({origin for groups in metrics["by_origin"].values() for origin in groups})
    written += _table(report_dir, "3.9", [
        {"config": name, **{origin: metrics["by_origin"][name].get(origin, {}).get(primary_metric) for origin in origins}}
        for name in main_names
    ], ["config", *origins], ["Cấu hình", *origins])

    if errors:
        written += _table(report_dir, "3.10", errors["rows"], ["stage", "cause", "count"], ["Tầng thất bại", "Nguyên nhân", "Số truy vấn"])
    written += _table(report_dir, "3.11", [{"config": name, **metrics["overall"][name]} for name in exploration],
                      ["config", *metric_columns], ["Cấu hình", *metric_columns])

    tune_rows = read_csv(bench / "tune_results.csv") if (bench / "tune_results.csv").exists() else []

    def draw_alpha(axis) -> bool:
        series = defaultdict(list)
        for row in tune_rows:
            if row["param"] == "alpha":
                series[row["category"]].append((float(row["value"]), float(row["score"])))
        for category, points in sorted(series.items()):
            points.sort()
            axis.plot([x for x, _ in points], [y for _, y in points], marker="o", label=category,
                      linewidth=2.5 if category == "all" else 1.2)
        axis.set_xlabel("α (trọng số BM25)")
        axis.set_ylabel(primary_metric)
        if series:
            axis.legend(fontsize=8)
        return bool(series)

    def draw_tradeoff(axis) -> bool:
        points = sorted(
            (values["rerank_n"], metrics["overall"][name][primary_metric], latency["configs"][name]["total"]["mean"])
            for name, values in configs.items()
            if latency and values.get("rerank") and values.get("fusion") == "weighted"
            and name in metrics["overall"] and name in latency["configs"]
        )
        if len(points) < 2:
            return False
        axis.plot([n for n, _, _ in points], [q for _, q, _ in points], marker="o", color="tab:blue")
        axis.set_xlabel("N (số ứng viên rerank)")
        axis.set_ylabel(primary_metric, color="tab:blue")
        twin = axis.twinx()
        twin.plot([n for n, _, _ in points], [t for _, _, t in points], marker="s", color="tab:red")
        twin.set_ylabel("Độ trễ trung bình (ms)", color="tab:red")
        return True

    def draw_categories(axis) -> bool:
        if not categories or not main_names:
            return False
        width = 0.8 / len(main_names)
        for offset, name in enumerate(main_names):
            axis.bar([position + offset * width for position in range(len(categories))],
                     [metrics["by_category"][name].get(category, {}).get(primary_metric, 0.0) for category in categories],
                     width, label=name)
        axis.set_xticks([position + 0.4 - width / 2 for position in range(len(categories))], categories)
        axis.set_ylabel(primary_metric)
        axis.legend(fontsize=7, ncol=2)
        return True

    written += _figure(report_dir, "3.1", draw_alpha)
    written += _figure(report_dir, "3.2", draw_tradeoff)
    written += _figure(report_dir, "3.3", draw_categories)
    return written
