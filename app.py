import json
from pathlib import Path

import streamlit as st

from src.ui import authenticate, database, seed_users


st.set_page_config(page_title="Trợ lý học tập RAG", page_icon="🎓", layout="wide")
db = database()
seed_users(db)


def login():
    introduction, form = st.columns([1.2, 1], gap="large")
    with introduction:
        st.caption("HỌC LIỆU · KHÔNG GIAN HỌC TẬP")
        st.title("Học từ tài liệu.\nKiểm tra từ nguồn.")
        st.write("Hỏi đáp sách, giáo trình và bài giảng. Mỗi câu trả lời dựa trên học liệu đều có dẫn nguồn để bạn đối chiếu.")
        st.caption("Hệ thống demo/nghiên cứu chạy trên máy cá nhân. Không thay thế việc đọc và kiểm chứng tài liệu.")
    with form:
        st.subheader("Đăng nhập")
        with st.form("login"):
            username = st.text_input("Tên đăng nhập")
            password = st.text_input("Mật khẩu", type="password")
            submitted = st.form_submit_button("Đăng nhập", type="primary", width="stretch")
        if submitted:
            user = authenticate(db, username, password)
            if user:
                st.session_state.user = user
                st.rerun()
            st.error("Tên đăng nhập hoặc mật khẩu không đúng.")


def overview():
    st.title("Không gian học tập")
    st.caption("Chọn học liệu, đặt câu hỏi và đối chiếu bằng chứng.")
    active = db.get_active_corpus()
    if not active:
        st.info("Chưa có học liệu đang hoạt động. Quản trị viên cần tải tài liệu và tạo chỉ mục.")
        return
    try:
        manifest_path = Path(active["chunks_path"]).parent / "manifest.json"
        documents = json.loads(manifest_path.read_text(encoding="utf-8"))["documents"]
    except (OSError, ValueError, KeyError):
        st.error("Không đọc được danh sách tài liệu của corpus đang hoạt động. Kiểm tra manifest hoặc lập chỉ mục lại.")
        return
    counts = st.columns(3)
    counts[0].metric("Tài liệu", len(documents))
    counts[1].metric("Môn học", len({document["course"] for document in documents}))
    counts[2].metric("Đoạn được lập chỉ mục", active.get("chunk_count", 0))
    st.page_link("pages/1_Chat.py", label="Mở hỏi đáp tài liệu", icon=":material/chat:")
    st.caption("Điểm truy xuất trong thực nghiệm không đồng nghĩa độ đúng của mọi câu trả lời. Luôn mở dẫn nguồn để kiểm tra.")


user = st.session_state.get("user")
if user:
    st.sidebar.title("Học liệu")
    st.sidebar.caption(f"{user['username']} · {'Quản trị viên' if user['role'] == 'admin' else 'Người học'}")
    if st.sidebar.button("Đăng xuất", width="stretch"):
        st.session_state.clear()
        st.rerun()
pages = {"Học tập": [
    st.Page(overview if user else login, title="Tổng quan" if user else "Đăng nhập", icon=":material/home:", default=True),
    st.Page("pages/1_Chat.py", title="Hỏi đáp", icon=":material/chat:", url_path="Chat", visibility="visible" if user else "hidden"),
]}
pages["Quản trị & nghiên cứu"] = [
    st.Page("pages/2_Documents.py", title="Tài liệu", icon=":material/library_books:", url_path="Documents", visibility="visible" if user and user["role"] == "admin" else "hidden"),
    st.Page("pages/3_RAG_Settings.py", title="Cấu hình truy xuất", icon=":material/tune:", url_path="RAG_Settings", visibility="visible" if user and user["role"] == "admin" else "hidden"),
    st.Page("pages/4_Experiments.py", title="Thực nghiệm", icon=":material/analytics:", url_path="Experiments", visibility="visible" if user and user["role"] == "admin" else "hidden"),
]
st.session_state.home_page = pages["Học tập"][0]
navigation = st.navigation(pages, position="sidebar" if user else "hidden")
navigation.run()
