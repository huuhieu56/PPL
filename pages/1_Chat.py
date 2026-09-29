import logging
import uuid

import streamlit as st
from openai import OpenAI

from src.config import load_settings
from src.index import RetrievalIndex
from src.models import PipelineConfig
from src.pipeline import RetrievalPipeline
from src.rag import answer_question
from src.ui import citation_label, database, require_role


st.set_page_config(page_title="Hỏi đáp", page_icon="💬", layout="centered")
user = require_role("student", "admin")
settings = load_settings()
db = database()
active = db.get_active_corpus()
st.title("Hỏi đáp tài liệu học tập")
st.caption("Hỏi theo nội dung học liệu. Mở dẫn nguồn dưới câu trả lời để kiểm tra; hệ thống có thể từ chối khi thiếu bằng chứng.")

if not active:
    st.info("Chưa có kho tài liệu đang hoạt động. Quản trị viên cần lập chỉ mục trước.")
    st.stop()
if not settings.openai_api_key or not settings.openai_model:
    st.error("Thiếu OPENAI_API_KEY hoặc OPENAI_MODEL trong .env.")
    st.stop()


@st.cache_resource(show_spinner="Đang tải chỉ mục...")
def load_pipeline(path: str) -> RetrievalPipeline:
    return RetrievalPipeline(RetrievalIndex.load(path))


try:
    pipeline = load_pipeline(str(settings.indexes_dir / active["version_id"]))
except Exception:
    logging.exception("Chat index load failed")
    st.error("Không tải được chỉ mục của kho tài liệu đang hoạt động. Quản trị viên cần lập chỉ mục lại.")
    st.stop()

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

titles = {chunk.doc_id: chunk.doc_title for chunk in pipeline.index.chunk_list}
filenames = {item["doc_id"]: item.get("filename", "") for item in db.list_documents()}
doc_ids = tuple(sorted(st.sidebar.multiselect(
    "Tài liệu muốn hỏi",
    sorted(titles, key=lambda doc_id: titles[doc_id]),
    format_func=lambda doc_id: titles[doc_id] or filenames.get(doc_id) or doc_id,
    help="Để trống để tìm trong toàn bộ kho. Lịch sử được tách theo nhóm tài liệu đã chọn.",
)))
st.caption(f"Phạm vi: {len(doc_ids)} tài liệu được chọn" if doc_ids else "Phạm vi: toàn bộ kho tài liệu")

sessions = db.list_chat_sessions(user["username"], active["version_id"], doc_ids)
if st.sidebar.button("Cuộc trò chuyện mới") or not sessions:
    session_id = uuid.uuid4().hex
    db.start_chat_session(session_id, user["username"], active["version_id"], doc_ids)
    sessions = db.list_chat_sessions(user["username"], active["version_id"], doc_ids)
    st.session_state.chat_session_id = session_id
session_ids = [item["session_id"] for item in sessions]
created = {item["session_id"]: item["created_at"] for item in sessions}
if st.session_state.get("chat_session_id") not in session_ids:
    st.session_state.chat_session_id = session_ids[0]
session_id = st.sidebar.selectbox(
    "Lịch sử trò chuyện",
    session_ids,
    index=session_ids.index(st.session_state.chat_session_id),
    format_func=lambda value: f"{created[value]} · {value[:6]}",
)
st.session_state.chat_session_id = session_id
turns = db.chat_turns(session_id, user["username"])


def render_answer(turn: dict) -> None:
    st.markdown(turn["answer"])
    if turn.get("rewritten_query"):
        st.caption(f"🔍 **Truy vấn đã chuẩn hóa:** *{turn['rewritten_query']}*")
    if turn.get("latency_ms"):
        st.caption(f"Thời gian: {turn['latency_ms']:.0f} ms")
    for citation in turn.get("citations", []):
        with st.expander(citation_label(citation, filenames)):
            st.write(citation["text"])


for turn in turns:
    with st.chat_message("user"):
        st.markdown(turn["query"])
    with st.chat_message("assistant"):
        render_answer(turn)

if query := st.chat_input("Nhập câu hỏi về tài liệu..."):
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm tài liệu và tạo câu trả lời..."):
            history = [
                message
                for turn in turns
                for message in ({"role": "user", "content": turn["query"]}, {"role": "assistant", "content": turn["answer"]})
            ]
            try:
                answer = answer_question(
                    query,
                    pipeline,
                    config,
                    client,
                    config.llm_model or settings.openai_model,
                    chat_history=history,
                    enable_rewrite=enable_rewrite,
                    doc_ids=frozenset(doc_ids),
                )
            except ValueError as error:
                st.error(f"Không thể truy xuất với cấu hình '{config_name}': {error}")
                st.stop()
            except Exception:
                logging.exception("Chat answer failed")
                st.error("Không thể trả lời lúc này. Kiểm tra kết nối mô hình rồi thử lại.")
                st.stop()
    show_rewritten = bool(
        answer.rewritten_query and answer.rewritten_query.strip().lower() != query.strip().lower()
    )
    db.save_message(
        {
            "session_id": session_id,
            "username": user["username"],
            "query": query,
            "rewritten_query": answer.rewritten_query if show_rewritten else "",
            "answer": answer.text,
            "citations": answer.citations,
            "refused": answer.refused,
            "latency_ms": answer.retrieval_ms + answer.generation_ms,
        }
    )
    st.rerun()

if turns:
    latest_id = turns[-1]["message_id"]
    with st.form(f"feedback-{latest_id}", clear_on_submit=True):
        useful = st.radio("Câu trả lời có hữu ích không?", ["Có", "Không"], horizontal=True)
        comment = st.text_input("Nhận xét thêm")
        if st.form_submit_button("Gửi phản hồi"):
            db.save_feedback({"message_id": latest_id, "useful": useful == "Có", "comment": comment})
            st.success("Đã lưu phản hồi.")
