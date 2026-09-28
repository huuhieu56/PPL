import uuid
import logging

import streamlit as st

from src.config import load_settings
from src.models import RagConfig
from src.rag import answer_question, chat_model
from src.retrieval import RetrievalIndex
from src.ui import citation_label, database, require_role


st.set_page_config(page_title="Hỏi đáp", page_icon="💬", layout="centered")
user = require_role("student", "admin")
settings = load_settings()
db = database()
document_names = {item["doc_id"]: item["filename"] for item in db.list_documents()}
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
def load_index(path: str):
    return RetrievalIndex.load(path)


try:
    index = load_index(str(settings.data_dir / "indexes" / active["version_id"]))
except Exception:
    logging.exception("Chat index load failed")
    st.error("Không tải được kho tài liệu. Kiểm tra Qdrant và chỉ mục của kho đang hoạt động.")
    st.stop()
saved_configs = db.list_rag_configs()
config_by_name = {item["name"]: RagConfig(**item["config"]) for item in saved_configs}
config_name = st.sidebar.selectbox("Cấu hình RAG", list(config_by_name) or ["Mặc định"])
config = config_by_name.get(config_name, RagConfig(model=settings.openai_model))
client = chat_model(settings, config.model or settings.openai_model)

available_docs = sorted({chunk.doc_id for chunk in index.chunks.values()}, key=lambda key: document_names.get(key, key))
saved_scope = st.session_state.get("chat_document_scope", {})
scope_default = saved_scope.get("doc_ids", ()) if saved_scope.get("corpus_version") == active["version_id"] else ()
doc_ids = tuple(sorted(st.sidebar.multiselect(
    "Tài liệu muốn hỏi", available_docs, format_func=lambda key: document_names.get(key, key),
    default=[doc_id for doc_id in scope_default if doc_id in available_docs],
    help="Để trống để tìm trong toàn bộ kho. Lịch sử được tách theo nhóm tài liệu đã chọn.",
)))
st.session_state.chat_document_scope = {"corpus_version":active["version_id"], "doc_ids":doc_ids}
if doc_ids:
    index = index.scoped(doc_ids)
st.caption(f"Phạm vi: {len(doc_ids)} tài liệu được chọn" if doc_ids else "Phạm vi: toàn bộ kho tài liệu")
sessions = db.list_chat_sessions(user["username"], active["version_id"], doc_ids)
session_ids = [item["session_id"] for item in sessions]
if st.sidebar.button("Cuộc trò chuyện mới") or not session_ids:
    session_id = uuid.uuid4().hex
    db.start_chat_session(session_id, user["username"], active["version_id"], doc_ids)
    session_ids.insert(0, session_id)
    st.session_state.chat_session_id = session_id
if st.session_state.get("chat_session_id") not in session_ids:
    st.session_state.chat_session_id = session_ids[0]
session_id = st.sidebar.selectbox(
    "Lịch sử trò chuyện",
    session_ids,
    index=session_ids.index(st.session_state.chat_session_id),
    format_func=lambda value: f"Phiên {value[:8]}",
)
st.session_state.chat_session_id = session_id
turns = db.chat_turns(session_id, user["username"])
for turn in turns:
    with st.chat_message("user"):
        st.markdown(turn["query"])
    with st.chat_message("assistant"):
        st.markdown(turn["answer"])
        for citation in turn.get("citations", []):
            with st.expander(citation_label(citation, document_names)):
                st.write(citation["text"])

if query := st.chat_input("Nhập câu hỏi về tài liệu..."):
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        with st.spinner("Đang tìm tài liệu và tạo câu trả lời..."):
            try:
                answer = answer_question(
                    query, index, config, client, config.model or settings.openai_model,
                    history=turns,
                )
            except Exception:
                logging.exception("Chat answer failed")
                st.error("Không thể trả lời lúc này. Kiểm tra kết nối mô hình và chỉ mục rồi thử lại.")
                st.stop()
        st.markdown(answer.text)
        st.caption(f"Thời gian: {answer.retrieval_ms + answer.generation_ms:.0f} ms")
        for citation in answer.citations:
            with st.expander(
                citation_label(citation, document_names)
            ):
                st.write(citation["text"])
    message_id = db.save_message(
        {
            "session_id": session_id,
            "username": user["username"],
            "query": query,
            "answer": answer.text,
            "citations": answer.citations,
            "refused": answer.refused,
            "latency_ms": answer.retrieval_ms + answer.generation_ms,
        }
    )
    latest_message_id = message_id

else:
    latest_message_id = turns[-1]["message_id"] if turns else None
if latest_message_id:
    if st.button("Góp ý câu trả lời", key=f"feedback-{latest_message_id}"):
        st.session_state.feedback_message_id = latest_message_id
    if st.session_state.get("feedback_message_id") == latest_message_id:
        with st.form("feedback", clear_on_submit=True):
            useful = st.radio("Câu trả lời có hữu ích không?", ["Có", "Không"], horizontal=True)
            comment = st.text_input("Nhận xét thêm")
            if st.form_submit_button("Gửi phản hồi"):
                db.save_feedback(
                    {
                        "message_id": latest_message_id,
                        "useful": useful == "Có",
                        "comment": comment,
                    }
                )
                st.session_state.feedback_message_id = None
                st.success("Đã lưu phản hồi.")
