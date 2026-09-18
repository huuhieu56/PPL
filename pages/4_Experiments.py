import json
from pathlib import Path

import streamlit as st
import yaml

from src.config import load_settings
from src.ui import database, require_role, safe_upload_name


st.set_page_config(page_title="Thực nghiệm", page_icon="📊", layout="wide")
require_role("admin")
settings = load_settings()
st.title("Thực nghiệm retrieval")

active = database().get_active_corpus()
queries = st.file_uploader("Queries JSONL", type="jsonl")
qrels = st.file_uploader("Qrels JSONL", type="jsonl")
selected = st.multiselect("Cấu hình", [f"E{i}" for i in range(8)], default=[f"E{i}" for i in range(8)])
if st.button("Tạo cấu hình CLI", disabled=not (active and queries and qrels and selected)):
    benchmark_dir = settings.data_dir / "benchmark"
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    query_path = benchmark_dir / safe_upload_name(queries.name)
    qrels_path = benchmark_dir / safe_upload_name(qrels.name)
    query_path.write_bytes(queries.getvalue())
    qrels_path.write_bytes(qrels.getvalue())
    template = yaml.safe_load(Path("configs/experiments.yaml").read_text(encoding="utf-8"))
    template["corpus_version"] = active["version_id"]
    template["index_dir"] = str(settings.data_dir / "indexes" / active["version_id"])
    template["queries"] = str(query_path)
    template["qrels"] = str(qrels_path)
    template["experiments"] = {
        key: value for key, value in template["experiments"].items() if key in selected
    }
    generated = Path("configs/generated_experiment.yaml")
    generated.write_text(yaml.safe_dump(template, sort_keys=False), encoding="utf-8")
    st.code(f".venv/bin/python run_experiments.py --config {generated}")

st.subheader("Kết quả đã chạy")
for run_dir in sorted(settings.runs_dir.glob("*"), reverse=True):
    if not run_dir.is_dir():
        continue
    with st.expander(run_dir.name):
        status_path = run_dir / "status.json"
        st.write(json.loads(status_path.read_text()) if status_path.exists() else {"status": "unknown"})
        metrics_path = run_dir / "metrics.json"
        if metrics_path.exists():
            st.json(json.loads(metrics_path.read_text(encoding="utf-8")))
            st.download_button("Tải metrics.json", metrics_path.read_bytes(), file_name="metrics.json")
        per_query = run_dir / "per_query.csv"
        if per_query.exists():
            st.download_button("Tải per_query.csv", per_query.read_bytes(), file_name="per_query.csv")
