import uuid

import streamlit as st
from openai import OpenAI

from src.config import load_settings
from src.models import RagConfig
from src.rag import answer_question
from src.retrieval import RetrievalIndex
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
def load_index(path: str):
    return RetrievalIndex.load(path)


index = load_index(str(settings.data_dir / "indexes" / active["version_id"]))
saved_configs = db.list_rag_configs()
config_by_name = {item["name"]: RagConfig(**item["config"]) for item in saved_configs}
config_name = st.sidebar.selectbox("Cấu hình RAG", list(config_by_name) or ["Mặc định"])
config = config_by_name.get(config_name, RagConfig(model=settings.openai_model))
client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
    st.session_state.chat_session_id = uuid.uuid4().hex

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        for citation in message.get("citations", []):
            with st.expander(
                f"[{citation['number']}] {citation['doc_id']} — trang/slide {citation['page']}"
            ):
                st.write(citation["text"])

if query := st.chat_input("Nhập câu hỏi về tài liệu..."):
    st.session_state.chat_messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm tài liệu và tạo câu trả lời..."):
            answer = answer_question(
                query,
                index,
                config,
                client,
                config.model or settings.openai_model,
            )
        st.markdown(answer.text)
        st.caption(f"Thời gian: {answer.retrieval_ms + answer.generation_ms:.0f} ms")
        for citation in answer.citations:
            with st.expander(
                f"[{citation['number']}] {citation['doc_id']} — trang/slide {citation['page']}"
            ):
                st.write(citation["text"])
    message_id = db.save_message(
        {
            "session_id": st.session_state.chat_session_id,
            "username": user["username"],
            "query": query,
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
