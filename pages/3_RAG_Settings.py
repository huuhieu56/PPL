import streamlit as st

from src.config import load_settings
from src.models import RagConfig
from src.ui import database, require_role


st.set_page_config(page_title="Cấu hình RAG", page_icon="⚙️")
require_role("admin")
settings = load_settings()
db = database()
st.title("Cấu hình RAG")

with st.form("rag_config"):
    name = st.text_input("Tên cấu hình", "adaptive-default")
    method = st.selectbox("Phương pháp", ["bm25", "dense", "rrf", "weighted", "adaptive"])
    use_reranker = st.checkbox("Dùng reranker", True)
    top_l = st.number_input("Top-L mỗi retriever", 10, 500, 100)
    rerank_n = st.number_input("Candidate rerank", 5, 100, 20)
    context_k = st.number_input("Context chunks", 1, 20, 5)
    alpha = st.slider("Alpha BM25", 0.0, 1.0, 0.5, 0.05)
    rrf_k = st.number_input("RRF k", 1, 200, 60)
    threshold = st.number_input("Ngưỡng từ chối", value=0.0, format="%.4f")
    model = st.text_input("LLM model", settings.openai_model)
    temperature = st.number_input("Temperature", 0.0, 2.0, 0.0, 0.1)
    submitted = st.form_submit_button("Lưu cấu hình", type="primary")
if submitted:
    if rerank_n > top_l or context_k > rerank_n:
        st.error("Cần context_k ≤ rerank_n ≤ top_l.")
    elif not name.strip():
        st.error("Tên cấu hình không được để trống.")
    else:
        db.save_rag_config(
            name.strip(),
            RagConfig(
                method=method,
                top_l=int(top_l),
                rerank_n=int(rerank_n),
                context_k=int(context_k),
                alpha=alpha,
                rrf_k=int(rrf_k),
                refusal_threshold=threshold,
                use_reranker=use_reranker,
                model=model,
                temperature=temperature,
            ),
        )
        st.success("Đã lưu cấu hình.")

configs = db.list_rag_configs()
if configs:
    st.dataframe(configs, use_container_width=True)
