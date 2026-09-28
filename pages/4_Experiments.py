import json
from pathlib import Path

import streamlit as st
import yaml

from src.config import load_settings
from src.experiments import select_queries, validate_config
from src.ui import database, require_role, stage_uploads


st.set_page_config(page_title="Thực nghiệm", page_icon="📊", layout="wide")
require_role("admin")
settings = load_settings()
st.title("Thực nghiệm retrieval")

active = database().get_active_corpus()
queries = st.file_uploader("Queries JSONL", type="jsonl")
qrels = st.file_uploader("Qrels JSONL", type="jsonl")
selected = st.multiselect("Cấu hình", [f"E{i}" for i in range(8)], default=[f"E{i}" for i in range(8)])
split = st.selectbox("Tập đánh giá", ["dev", "test"])
st.caption("Tune trên dev. Chỉ chạy test sau khi đã đóng băng cấu hình; không trộn dev/test.")
if st.button("Tạo cấu hình CLI", disabled=not (active and queries and qrels and selected)):
    benchmark_dir = settings.data_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    query_path = stage_uploads([queries], benchmark_dir)[0]
    qrels_path = stage_uploads([qrels], benchmark_dir)[0]
    template = yaml.safe_load(Path("configs/experiments.yaml").read_text(encoding="utf-8"))
    template["corpus_version"] = active["version_id"]
    template["index_dir"] = str(settings.data_dir / "indexes" / active["version_id"])
    template["queries"] = str(query_path)
    template["qrels"] = str(qrels_path)
    template["experiments"] = {
        key: value for key, value in template["experiments"].items() if key in selected
    }
    try:
        validate_config(template, require_index=True)
        select_queries(query_path, split)
    except (ValueError, KeyError, TypeError, OSError) as error:
        st.error(f"Benchmark không hợp lệ: {error}")
        st.stop()
    generated = Path("configs/generated_experiment.yaml")
    generated.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    st.code(f"conda run -n ppl-rag python run_experiments.py --config {generated} --split {split}")

st.subheader("Kết quả đã chạy")
for run_dir in sorted(settings.runs_dir.glob("*"), reverse=True):
    if not run_dir.is_dir():
        continue
    with st.expander(run_dir.name):
        public_results = run_dir / "results.json"
        if public_results.is_file():
            result = json.loads(public_results.read_text(encoding="utf-8"))
            st.caption(f"{result['source']} · dev {len(result['dev_query_ids'])} · test {len(result['test_query_ids'])} · chọn {result['chosen']}")
            st.dataframe([
                {"Phương pháp": method, "nDCG@10": scores["ndcg@10"], "MRR@10": scores["mrr@10"], "Hit@1": scores["hit_rate@1"]}
                for method, scores in result["test"].items()
            ], width="stretch")
            st.download_button("Tải results.json", public_results.read_bytes(), file_name="results.json", key=f"{run_dir.name}-results")
            per_query_jsonl = run_dir / "per_query.jsonl"
            if per_query_jsonl.is_file():
                st.download_button("Tải per_query.jsonl", per_query_jsonl.read_bytes(), file_name="per_query.jsonl", key=f"{run_dir.name}-per-query-jsonl")
            continue
        status_path = run_dir / "status.json"
        st.write(json.loads(status_path.read_text()) if status_path.exists() else {"status": "unknown"})
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            st.json(json.loads(metrics_path.read_text(encoding="utf-8")))
            st.download_button("Tải metrics.json", metrics_path.read_bytes(), file_name="metrics.json", key=f"{run_dir.name}-metrics")
        per_query = run_dir / "per_query.csv"
        if per_query.exists():
            st.download_button("Tải per_query.csv", per_query.read_bytes(), file_name="per_query.csv", key=f"{run_dir.name}-per-query-csv")
