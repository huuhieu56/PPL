import uuid

import streamlit as st
from openai import OpenAI

from src.config import load_settings
from src.index import RetrievalIndex
from src.models import PipelineConfig
from src.pipeline import RetrievalPipeline
from src.rag import answer_question
from src.ui import database, require_role


st.set_page_config(page_title="Hỏi đáp", page_icon="💬", layout="wide")
user = require_role("student", "admin")
settings = load_settings()
db = database()
active = db.get_active_corpus()
st.title("Hỏi đáp tài liệu học tập")

if not active:
    st.info("Chưa có kho tài liệu đang hoạt động. Quản trị viên cần lập chỉ mục trước.")
    st.stop()
if not settings.openai_api_key or not settings.openai_model:
    st.error("Thiếu OPENAI_API_KEY hoặc OPENAI_MODEL trong .env.")
    st.stop()


@st.cache_resource(show_spinner="Đang tải chỉ mục...")
def load_pipeline(path: str) -> RetrievalPipeline:
    return RetrievalPipeline(RetrievalIndex.load(path))


pipeline = load_pipeline(str(settings.indexes_dir / active["version_id"]))
config_by_name, invalid = db.load_pipeline_configs()
if invalid:
    st.sidebar.warning(f"Bỏ qua cấu hình cũ: {', '.join(invalid)}")
config_name = st.sidebar.selectbox("Cấu hình RAG", list(config_by_name) or ["Mặc định"])
config = config_by_name.get(config_name, PipelineConfig(llm_model=settings.openai_model))
enable_rewrite = st.sidebar.checkbox(
    "Chuẩn hóa câu hỏi (Query Rewrite)",
    value=True,
    help="Tự động chuẩn hóa câu hỏi, liên kết ngữ cảnh từ các câu hỏi trước và tối ưu từ khóa tìm kiếm học liệu.",
)
client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


def citation_label(citation: dict) -> str:
    source = " > ".join([citation.get("doc_title", ""), *citation.get("heading_path", [])]).strip(" >")
    return f"[{citation['number']}] {source or citation['doc_id']} — trang/slide {citation['page']}"

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
    st.session_state.chat_session_id = uuid.uuid4().hex

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("rewritten_query"):
            st.caption(f"🔍 **Truy vấn đã chuẩn hóa:** *{message['rewritten_query']}*")
        for citation in message.get("citations", []):
            with st.expander(citation_label(citation)):
                st.write(citation["text"])

if query := st.chat_input("Nhập câu hỏi về tài liệu..."):
    st.session_state.chat_messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm tài liệu và tạo câu trả lời..."):
            try:
                answer = answer_question(
                    query,
                    pipeline,
                    config,
                    client,
                    config.llm_model or settings.openai_model,
                    chat_history=st.session_state.chat_messages[:-1],
                    enable_rewrite=enable_rewrite,
                )
            except ValueError as error:
                st.error(f"Không thể truy xuất với cấu hình '{config_name}': {error}")
                st.stop()
        st.markdown(answer.text)
        show_rewritten = bool(
            answer.rewritten_query
            and answer.rewritten_query.strip().lower() != query.strip().lower()
        )
        if show_rewritten:
            st.caption(f"🔍 **Truy vấn đã chuẩn hóa:** *{answer.rewritten_query}*")
        st.caption(f"Thời gian: {answer.retrieval_ms + answer.generation_ms:.0f} ms")
        for citation in answer.citations:
            with st.expander(citation_label(citation)):
                st.write(citation["text"])
    message_id = db.save_message(
        {
            "session_id": st.session_state.chat_session_id,
            "username": user["username"],
            "query": query,
            "rewritten_query": answer.rewritten_query if show_rewritten else "",
            "answer": answer.text,
            "citations": answer.citations,
            "latency_ms": answer.retrieval_ms + answer.generation_ms,
        }
    )
    st.session_state.chat_messages.append(
        {
            "role": "assistant",
            "content": answer.text,
            "citations": answer.citations,
            "rewritten_query": answer.rewritten_query if show_rewritten else "",
            "message_id": message_id,
        }
    )

if st.session_state.chat_messages:
    latest = next(
        (message for message in reversed(st.session_state.chat_messages) if message["role"] == "assistant"),
        None,
    )
    if latest and "message_id" in latest:
        with st.form("feedback", clear_on_submit=True):
            useful = st.radio("Câu trả lời có hữu ích không?", ["Có", "Không"], horizontal=True)
            comment = st.text_input("Nhận xét thêm")
            if st.form_submit_button("Gửi phản hồi"):
                db.save_feedback(
                    {
                        "message_id": latest["message_id"],
                        "useful": useful == "Có",
                        "comment": comment,
                    }
                )
                st.success("Đã lưu phản hồi.")
