import json

import streamlit as st

from src.config import load_settings
from src.ui import database, require_role

st.set_page_config(page_title="Thực nghiệm", page_icon="📊", layout="wide")
require_role("admin")
settings = load_settings()
st.title("Thực nghiệm truy xuất")

active = database().get_active_corpus()
st.caption(f"Corpus đang active: {active['version_id'] if active else 'chưa có'}")
st.markdown(
    """
Quy trình (chạy trong terminal hoặc notebook Colab `notebooks/colab_pipeline.ipynb`):

```bash
python -m src.cli eval tune --config configs/experiment.yaml
python -m src.cli eval run --config configs/experiment.yaml --split test
python -m src.cli eval compare --config configs/experiment.yaml --run <RUN_DIR>
python -m src.cli eval errors --config configs/experiment.yaml --run <RUN_DIR>
python -m src.cli eval report --config configs/experiment.yaml --run <RUN_DIR>
```
"""
)

st.subheader("Các lần chạy")
runs = sorted((path for path in settings.runs_dir.glob("*") if path.is_dir()), reverse=True)
if not runs:
    st.info("Chưa có lần chạy nào.")
for run_dir in runs:
    with st.expander(run_dir.name, expanded=run_dir == runs[0]):
        status_path, config_path = run_dir / "status.json", run_dir / "config.json"
        status = json.loads(status_path.read_text(encoding="utf-8"))["status"] if status_path.exists() else "unknown"
        record = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        st.write(f"Trạng thái: **{status}** — tập: **{record.get('split', '?')}**")
        if (record.get("lock") or {}).get("lock_violation"):
            st.warning("Vi phạm khóa tập test: frozen_params.yaml đã thay đổi sau lần chạy test đầu tiên.")
        for table in sorted((run_dir / "report").glob("table_*.md"), key=lambda path: int(path.stem.split("_")[-1])):
            st.markdown(table.read_text(encoding="utf-8"))
        for figure in sorted((run_dir / "report").glob("figure_*.png")):
            st.image(str(figure))
        for name in ("metrics.json", "comparisons.csv", "error_sample.csv"):
            path = run_dir / name
            if path.exists():
                st.download_button(f"Tải {name}", path.read_bytes(), file_name=name, key=f"{run_dir.name}-{name}")
