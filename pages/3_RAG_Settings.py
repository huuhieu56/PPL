import streamlit as st

from src.config import load_settings
from src.models import PipelineConfig
from src.text import TOKENIZERS
from src.ui import database, require_role

BRANCHES = {"BM25": (True, False), "Dense": (False, True), "Hybrid": (True, True)}

st.set_page_config(page_title="Cấu hình RAG", page_icon="⚙️")
require_role("admin")
settings = load_settings()
db = database()
st.title("Cấu hình RAG")

with st.form("rag_config"):
    name = st.text_input("Tên cấu hình", "hybrid-default", key="config_name")
    branch = st.selectbox("Nhánh truy xuất", list(BRANCHES), index=2, key="branches")
    fusion = st.selectbox("Dung hợp (khi Hybrid)", ["weighted", "rrf", "adaptive"], key="fusion")
    rerank = st.checkbox("Dùng reranker", True, key="rerank")
    tokenizer = st.selectbox("Tách từ BM25", list(TOKENIZERS), key="tokenizer")
    top_l = st.number_input("Top-L mỗi nhánh", 10, 500, 100, key="top_l")
    rerank_n = st.number_input("Top-N rerank", 1, 100, 30, key="rerank_n")
    context_k = st.number_input("Số chunk đưa vào context", 1, 20, 5, key="context_k")
    alpha = st.slider("Alpha (trọng số BM25)", 0.0, 1.0, 0.5, 0.05, key="alpha")
    rrf_k = st.number_input("RRF k", 1, 200, 60, key="rrf_k")
    adaptive_beta = st.slider("Beta (adaptive)", 0.0, 0.5, 0.3, 0.05, key="adaptive_beta")
    reranker_model = st.text_input("Reranker", "BAAI/bge-reranker-v2-m3", key="reranker_model")
    threshold = st.number_input("Ngưỡng từ chối", value=0.0, format="%.4f", key="threshold")
    llm_model = st.text_input("LLM model", settings.openai_model, key="llm_model")
    temperature = st.number_input("Temperature", 0.0, 2.0, 0.0, 0.1, key="temperature")
    submitted = st.form_submit_button("Lưu cấu hình", type="primary", key="save_config")

if submitted:
    sparse, dense = BRANCHES[branch]
    try:
        config = PipelineConfig(
            sparse=sparse,
            dense=dense,
            fusion=fusion if sparse and dense else "none",
            alpha=float(alpha),
            rrf_k=int(rrf_k),
            adaptive_beta=float(adaptive_beta),
            rerank=rerank,
            rerank_n=int(rerank_n),
            top_l=int(top_l),
            context_k=int(context_k),
            tokenizer=tokenizer,
            reranker_model=reranker_model.strip(),
            refusal_threshold=float(threshold),
            llm_model=llm_model.strip(),
            temperature=float(temperature),
        )
    except ValueError as error:
        st.error(f"Cấu hình không hợp lệ: {error}")
    else:
        if not name.strip():
            st.error("Tên cấu hình không được để trống.")
        else:
            db.save_rag_config(name.strip(), config)
            st.success("Đã lưu cấu hình.")

configs, invalid = db.load_pipeline_configs()
if invalid:
    st.warning(f"Bỏ qua cấu hình cũ không còn tương thích: {', '.join(invalid)}")
if configs:
    st.dataframe(
        [{"name": name, **config.to_dict()} for name, config in configs.items()],
        use_container_width=True,
    )
