import hashlib
import json
from pathlib import Path

import streamlit as st

from src.config import load_settings, load_yaml
from src.corpus import index_corpus
from src.ingestion import extract_document
from src.ui import database, require_role, stage_uploads


st.set_page_config(page_title="Quản lý tài liệu", page_icon="📚", layout="wide")
require_role("admin")
settings = load_settings()
db = database()
defaults = load_yaml("configs/default.yaml")
st.title("Quản lý tài liệu")
active = db.get_active_corpus()
active_ids = set()
if active:
    if active.get("node_count"):
        st.caption(f"Corpus đang dùng: {active['version_id']} · {active['chunk_count']} chunk lá · {active['node_count']} nút cấu trúc")
    else:
        st.warning("Corpus đang dùng chưa có cây phân cấp; hãy lập chỉ mục lại.")
    manifest_path = Path(active["chunks_path"]).parent / "manifest.json" if active.get("chunks_path") else None
    if manifest_path and manifest_path.is_file():
        active_ids = {item["doc_id"] for item in json.loads(manifest_path.read_text(encoding="utf-8"))["documents"]}
    else:
        st.warning("Chỉ mục cũ không còn; tài liệu tải lên sẽ tạo corpus mới.")

course = st.text_input("Môn học", placeholder="AI101")
source_type = st.selectbox("Loại tài liệu", ["textbook", "slide", "exam"])
semester = st.text_input("Học kỳ")
uploads = st.file_uploader(
    "Tải PDF, DOCX hoặc PPTX",
    type=["pdf", "docx", "pptx"],
    accept_multiple_files=True,
)

if uploads:
    total_size = sum(upload.size for upload in uploads)
    if total_size > 200 * 1024 * 1024:
        st.error("Tổng dung lượng upload vượt 200 MB.")
        st.stop()
    staging = settings.data_dir / "raw" / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    paths = stage_uploads(uploads, staging)
    stored = {
        item["doc_id"]: item for item in db.list_documents()
        if item["doc_id"] in active_ids and item.get("source_path") and Path(item["source_path"]).is_file()
    }
    with st.expander("Tùy chọn: chọn tài liệu giữ trong corpus"):
        retained = st.multiselect(
            "Giữ tài liệu đã nạp trong corpus mới",
            list(stored), default=list(stored),
            format_func=lambda doc_id: f"{stored[doc_id]['filename']} · {stored[doc_id].get('course', '')}",
            key=f"retained-{active['version_id'] if active else 'empty'}",
        )
    if st.button("Xem trước trích xuất"):
        for path in paths:
            st.subheader(path.name)
            try:
                pages = extract_document(path)[:5]
            except Exception as error:
                st.error(f"Không đọc được {path.name}: {error}")
                continue
            if not any(page.text.strip() for page in pages):
                st.warning(f"{path.name}: không tìm thấy văn bản trong các trang xem trước.")
            for page in pages:
                if page.warning:
                    st.warning(f"Trang {page.page}: {page.warning}")
                locator = "Đoạn" if path.suffix.lower() == ".docx" else "Trang/slide"
                st.text_area(f"{locator} {page.page}", page.text, height=140, disabled=True, key=f"preview-{path.parent.name}-{page.page}")
    if not course.strip():
        st.info("Nhập môn học để tự động lập chỉ mục tài liệu đã tải lên.")
    else:
        uploaded = {hashlib.sha256(path.read_bytes()).hexdigest(): path for path in paths}
        signature = (tuple(sorted((digest, path.name) for digest, path in uploaded.items())), tuple(sorted(set(retained) | {digest[:24] for digest in uploaded})), course.strip(), source_type, semester)
        attempt = (signature, active["version_id"] if active else None)
        retry = st.button("Thử lại") if st.session_state.get("upload_error") and st.session_state.get("upload_attempt") == attempt else False
        if st.session_state.get("upload_attempt") != attempt or retry:
            st.session_state.upload_attempt = attempt
            with st.spinner("Đang trích xuất, chunk và tạo embedding..."):
                files_by_hash = {stored[doc_id]["sha256"]: Path(stored[doc_id]["source_path"]) for doc_id in retained}
                metadata = {
                    str(stored[doc_id]["source_path"]): {key: stored[doc_id][key] for key in ("course", "source_type", "semester") if key in stored[doc_id]}
                    for doc_id in retained
                }
                uploaded_metadata = {
                    str(path): {"course": course.strip(), "source_type": source_type, "semester": semester}
                    for path in paths
                }
                files_by_hash.update(uploaded)
                metadata.update(uploaded_metadata)
                retrieval = defaults["retrieval"]
                try:
                    result = index_corpus(
                        list(files_by_hash.values()),
                        metadata,
                        settings,
                        db,
                        {"chunk_tokens": 450, "overlap_tokens": 75},
                        retrieval,
                    )
                except Exception as error:
                    st.session_state.upload_error = f"Không thể lập chỉ mục: {error}. Corpus đang dùng không thay đổi."
                else:
                    st.session_state.upload_error = ""
                    st.session_state.upload_attempt = (signature, result.version_id)
                    st.session_state.upload_result = f"Đã kích hoạt corpus {result.version_id} với {result.chunk_count} chunks."
                    st.rerun()
        if st.session_state.get("upload_error"):
            st.error(st.session_state.upload_error)
        elif st.session_state.get("upload_attempt") == (signature, db.get_active_corpus()["version_id"]):
            st.success(st.session_state.upload_result)

documents = [item for item in db.list_documents() if item["doc_id"] in active_ids]
if documents:
    st.subheader("Tài liệu đã xử lý")
    st.dataframe([{"Tài liệu": item["filename"], "Môn học": item.get("course", ""), "Loại": item.get("source_type", ""), "Trạng thái": item["status"]} for item in documents], width="stretch", hide_index=True)
