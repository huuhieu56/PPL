import streamlit as st

from src.ui import authenticate, database, seed_users


st.set_page_config(page_title="Trợ lý học tập RAG", page_icon="🎓", layout="wide")
db = database()
seed_users(db)

st.title("Trợ lý học tập RAG")
if "user" not in st.session_state:
    with st.form("login"):
        username = st.text_input("Tên đăng nhập")
        password = st.text_input("Mật khẩu", type="password")
        submitted = st.form_submit_button("Đăng nhập", type="primary")
    if submitted:
        user = authenticate(db, username, password)
        if user:
            st.session_state.user = user
            st.rerun()
        st.error("Tên đăng nhập hoặc mật khẩu không đúng.")
else:
    user = st.session_state.user
    st.success(f"Đã đăng nhập: {user['username']} ({user['role']})")
    st.write("Sử dụng thanh điều hướng để mở trang hỏi đáp hoặc quản trị.")
    if st.button("Đăng xuất"):
        st.session_state.clear()
        st.rerun()
