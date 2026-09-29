import hashlib
import json
from pathlib import Path

import streamlit as st

from src.config import load_settings, load_yaml
from src.ingestion import build_corpus, extract_blocks
from src.index import RetrievalIndex
from src.models import Chunk
from src.ui import database, require_role, stage_uploads

MAX_UPLOAD_BYTES = 200 * 1024 * 1024
CARRIED_METADATA = ("course", "source_type", "semester", "doc_title")

st.set_page_config(page_title="Quản lý tài liệu", page_icon="📚", layout="wide")
require_role("admin")
settings = load_settings()
db = database()
defaults = load_yaml("configs/default.yaml")
st.title("Quản lý tài liệu")

active = db.get_active_corpus()
active_ids: set[str] = set()
if active:
    st.caption(f"Corpus đang dùng: {active['version_id']} · {active.get('chunk_count', '?')} chunk")
    manifest_path = Path(active["chunks_path"]).parent / "manifest.json" if active.get("chunks_path") else None
    if manifest_path and manifest_path.is_file():
        active_ids = {item["doc_id"] for item in json.loads(manifest_path.read_text(encoding="utf-8"))["documents"]}
    else:
        st.warning("Không tìm thấy manifest của corpus đang dùng; tài liệu tải lên sẽ tạo corpus mới.")

# Documents of the active corpus whose source file is still on disk can be carried into a rebuild.
stored = {
    item["doc_id"]: item
    for item in db.list_documents()
    if item["doc_id"] in active_ids and item.get("source_path") and Path(item["source_path"]).is_file()
}

course = st.text_input("Môn học", placeholder="AI101")
source_type = st.selectbox("Loại tài liệu", ["textbook", "slide", "exam"])
semester = st.text_input("Học kỳ")
strategy = st.selectbox("Chiến lược chunking", ["structure", "fixed"])
use_prefix = st.checkbox("Gắn tiêu đề (Tài liệu > Chương > Mục) vào đầu chunk", True)
uploads = st.file_uploader(
    "Tải PDF, DOCX hoặc PPTX",
    type=["pdf", "docx", "pptx"],
    accept_multiple_files=True,
)

if uploads:
    total_size = sum(upload.size for upload in uploads)
    if total_size > MAX_UPLOAD_BYTES:
        st.error("Tổng dung lượng upload vượt 200 MB.")
        st.stop()
    paths = stage_uploads(uploads, settings.data_dir / "raw" / "uploads")
    retained = []
    if stored:
        with st.expander("Tùy chọn: chọn tài liệu cũ giữ lại trong corpus mới"):
            retained = st.multiselect(
                "Giữ tài liệu đã nạp",
                list(stored),
                default=list(stored),
                format_func=lambda doc_id: f"{stored[doc_id]['filename']} · {stored[doc_id].get('course', '')}",
            )
    if st.button("Xem trước trích xuất"):
        for path in paths:
            st.subheader(path.name)
            try:
                blocks = extract_blocks(path)[:8]
            except Exception as error:
                st.error(f"Không đọc được {path.name}: {error}")
                continue
            if not any(block.text.strip() for block in blocks):
                st.warning(f"{path.name}: không tìm thấy văn bản trong phần xem trước.")
            locator = "" if path.suffix.lower() == ".docx" else "Trang/slide "
            for position, block in enumerate(blocks):
                if block.warning:
                    st.warning(f"Trang {block.page}: {block.warning}")
                where = f"{locator}{block.page} — " if locator else ""
                st.caption(f"{where}{' > '.join(block.heading_path) or '(chưa có tiêu đề)'}")
                st.text_area("Nội dung", block.text, height=120, disabled=True, key=f"{path.parent.name}-{position}")
    if st.button("Tạo và kích hoạt chỉ mục", type="primary", disabled=not course.strip()):
        with st.spinner("Đang trích xuất, chunk và tạo embedding..."):
            metadata = {
                stored[doc_id]["source_path"]: {key: stored[doc_id][key] for key in CARRIED_METADATA if key in stored[doc_id]}
                for doc_id in retained
            }
            metadata.update({
                str(path): {"course": course.strip(), "source_type": source_type, "semester": semester}
                for path in paths
            })
            files = [Path(path) for path in metadata]
            chunking = {**defaults["chunking"], "strategy": strategy, "prefix": use_prefix}
            try:
                result = build_corpus(files, metadata, settings, db, chunking)
                index_dir = settings.indexes_dir / result.version_id
                if not (index_dir / "index_meta.json").exists():
                    chunks = [
                        Chunk.from_dict(json.loads(line))
                        for line in result.chunks_path.read_text(encoding="utf-8").splitlines()
                        if line
                    ]
                    index_options = defaults["index"]
                    RetrievalIndex.build(
                        chunks,
                        index_dir,
                        embedding_model=index_options["embedding_model"],
                        tokenizers=index_options["tokenizers"],
                        dense_backend=index_options["dense_backend"],
                        bm25_k1=index_options["bm25_k1"],
                        bm25_b=index_options["bm25_b"],
                        chunking=chunking,
                    )
            except Exception as error:
                st.error(f"Không thể lập chỉ mục: {error}. Corpus đang dùng không thay đổi.")
                st.stop()
            # build_corpus rewrites each document row, so record every source path afterwards.
            for path in files:
                db.set_document_source(hashlib.sha256(path.read_bytes()).hexdigest()[:24], str(path))
            db.set_active_corpus(result.version_id)
        st.session_state.index_result = f"Đã kích hoạt corpus {result.version_id} với {result.chunk_count} chunks."
        st.rerun()

if message := st.session_state.pop("index_result", None):
    st.success(message)

documents = [item for item in db.list_documents() if item["doc_id"] in active_ids]
if documents:
    st.subheader("Tài liệu trong corpus đang dùng")
    st.dataframe(
        [
            {
                "Tài liệu": item.get("doc_title") or item.get("filename", ""),
                "Tệp": item.get("filename", ""),
                "Môn học": item.get("course", ""),
                "Loại": item.get("source_type", ""),
                "Học kỳ": item.get("semester", ""),
                "Trạng thái": item["status"],
            }
            for item in documents
        ],
        width="stretch",
        hide_index=True,
    )
