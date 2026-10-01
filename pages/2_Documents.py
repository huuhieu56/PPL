import hashlib
import json
from pathlib import Path

import streamlit as st

from src.config import load_settings, load_yaml
from src.corpus import index_corpus, remove_document
from src.models import Chunk
from src.ui import database, require_role, stage_uploads


st.set_page_config(page_title="Quản lý tài liệu", page_icon="📚", layout="wide")
require_role("admin")
settings = load_settings()
db = database()
retrieval = load_yaml("configs/default.yaml")["retrieval"]
chunking = {"chunk_tokens": 450, "overlap_tokens": 75}
active = db.get_active_corpus()
st.title("Quản lý tài liệu")

manifest_path = Path(active["chunks_path"]).parent / "manifest.json" if active and active.get("chunks_path") else None
manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path and manifest_path.is_file() else None
active_ids = {item["doc_id"] for item in manifest["documents"]} if manifest else set()
documents = {item["doc_id"]: item for item in db.list_documents() if item["doc_id"] in active_ids}
if active:
    st.caption(f"Corpus đang dùng: {active['version_id']} · {active.get('chunk_count', 0)} chunk lá · {active.get('node_count', 0)} nút cấu trúc")
    if not manifest:
        st.error("Không đọc được manifest của corpus đang dùng. Không thể thay đổi kho tài liệu.")
if notice := st.session_state.pop("document_notice", None):
    st.success(notice)

st.subheader("Thêm tài liệu")
st.caption("Chọn file chỉ để chuẩn bị. Trích xuất, chunking và embedding chỉ bắt đầu khi bấm nút xử lý.")
with st.form("upload_documents"):
    course = st.text_input("Môn học", placeholder="Ví dụ: Sinh học")
    source_type = st.selectbox("Loại tài liệu", ["textbook", "slide", "exam"])
    semester = st.text_input("Học kỳ")
    uploads = st.file_uploader("Tải PDF, DOCX hoặc PPTX", type=["pdf", "docx", "pptx"], accept_multiple_files=True)
    submitted = st.form_submit_button("Bắt đầu xử lý", type="primary")

if submitted:
    if not uploads:
        st.warning("Hãy chọn ít nhất một tài liệu.")
    elif not course.strip():
        st.warning("Hãy nhập môn học.")
    elif sum(upload.size for upload in uploads) > 200 * 1024 * 1024:
        st.error("Tổng dung lượng upload vượt 200 MB.")
    elif active and not manifest:
        st.error("Không thể giữ tài liệu hiện có vì manifest bị thiếu.")
    else:
        try:
            paths = stage_uploads(uploads, settings.data_dir / "raw" / "staging")
            files_by_hash = {item["sha256"]: Path(item["source_path"]) for item in documents.values()}
            metadata = {
                str(item["source_path"]): {key: item[key] for key in ("course", "source_type", "semester") if key in item}
                for item in documents.values()
            }
            for path in paths:
                files_by_hash[hashlib.sha256(path.read_bytes()).hexdigest()] = path
                metadata[str(path)] = {"course": course.strip(), "source_type": source_type, "semester": semester}
            with st.spinner("Đang trích xuất, chunk và tạo embedding..."):
                result = index_corpus(list(files_by_hash.values()), metadata, settings, db, chunking, retrieval)
        except Exception as error:
            st.error(f"Không thể lập chỉ mục: {error}. Corpus đang dùng không thay đổi.")
        else:
            st.session_state.document_notice = f"Đã xử lý {len(paths)} file; corpus mới có {result.chunk_count} chunk."
            st.session_state.preview_doc_id = hashlib.sha256(paths[0].read_bytes()).hexdigest()[:24]
            st.rerun()

if documents:
    st.subheader("Tài liệu đã xử lý")
    st.dataframe(
        [{"Tài liệu": item["filename"], "Môn học": item.get("course", ""), "Loại": item.get("source_type", ""), "Trạng thái": item["status"]} for item in documents.values()],
        width="stretch", hide_index=True,
    )

    st.subheader("Kiểm tra chunk")
    chunks = [Chunk(**json.loads(line)) for line in Path(active["chunks_path"]).read_text(encoding="utf-8").splitlines() if line]
    doc_ids = list(documents)
    if st.session_state.get("preview_doc_id") not in doc_ids:
        st.session_state.preview_doc_id = doc_ids[0]
    preview_id = st.selectbox("Tài liệu", doc_ids, key="preview_doc_id",
                              format_func=lambda doc_id: documents[doc_id]["filename"])
    selected_chunks = [chunk for chunk in chunks if chunk.doc_id == preview_id]
    st.caption(f"{len(selected_chunks)} chunk · chọn một chunk để đọc toàn bộ nội dung và kiểm tra ranh giới cắt.")
    st.dataframe([
        {"Chunk": number, "Vị trí": f"{'Đoạn' if chunk.file_type == 'docx' else 'Trang/slide'} {chunk.page}–{chunk.page_end or chunk.page}", "Mục": chunk.section,
         "Khoảng từ": f"{chunk.word_start}–{chunk.word_end}", "Nội dung": chunk.text[:160]}
        for number, chunk in enumerate(selected_chunks, 1)
    ], width="stretch", hide_index=True)
    if selected_chunks:
        number = st.selectbox("Xem đầy đủ chunk", range(1, len(selected_chunks) + 1))
        chunk = selected_chunks[number - 1]
        st.caption(f"ID: {chunk.chunk_id} · nút cha: {chunk.parent_id} · {'đoạn' if chunk.file_type == 'docx' else 'trang/slide'} {chunk.page}–{chunk.page_end or chunk.page}")
        st.text_area("Nội dung chunk", chunk.text, height=240, disabled=True)

    with st.expander("Xóa một tài liệu"):
        st.warning("Thao tác này tạo lại chỉ mục cho các tài liệu còn lại, xóa embedding cũ và xóa file upload khỏi volume. Không thể hoàn tác.")
        with st.form("remove_document"):
            delete_id = st.selectbox("Tài liệu cần xóa", doc_ids, format_func=lambda doc_id: documents[doc_id]["filename"])
            confirmed = st.checkbox("Tôi xác nhận xóa tài liệu này")
            delete_submitted = st.form_submit_button("Xóa tài liệu")
        if delete_submitted:
            if not confirmed:
                st.warning("Hãy xác nhận trước khi xóa.")
            else:
                try:
                    with st.spinner("Đang tạo lại chỉ mục và xóa dữ liệu cũ..."):
                        source_deleted = remove_document(delete_id, settings, db, chunking, retrieval)
                except Exception as error:
                    st.error(f"Không thể xóa tài liệu: {error}")
                else:
                    st.session_state.document_notice = "Đã xóa tài liệu, metadata và embedding. " + (
                        "File upload đã bị xóa." if source_deleted else "File nguồn nằm ngoài volume của ứng dụng nên được giữ nguyên."
                    )
                    st.session_state.pop("preview_doc_id", None)
                    st.rerun()
