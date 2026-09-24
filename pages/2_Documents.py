import json

import streamlit as st

from src.config import load_settings, load_yaml
from src.ingestion import build_corpus, extract_blocks
from src.index import RetrievalIndex
from src.models import Chunk
from src.ui import database, require_role, safe_upload_name

st.set_page_config(page_title="Quản lý tài liệu", page_icon="📚", layout="wide")
require_role("admin")
settings = load_settings()
db = database()
defaults = load_yaml("configs/default.yaml")
st.title("Quản lý tài liệu")

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
    if total_size > 50 * 1024 * 1024:
        st.error("Tổng dung lượng upload vượt 50 MB.")
        st.stop()
    staging = settings.data_dir / "raw" / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    paths = []
    for upload in uploads:
        path = staging / safe_upload_name(upload.name)
        path.write_bytes(upload.getvalue())
        paths.append(path)
    if st.button("Xem trước trích xuất"):
        for path in paths:
            st.subheader(path.name)
            for block in extract_blocks(path)[:8]:
                if block.warning:
                    st.warning(f"Trang {block.page}: {block.warning}")
                st.caption(f"Trang/slide {block.page} — {' > '.join(block.heading_path) or '(chưa có tiêu đề)'}")
                st.text_area("Nội dung", block.text, height=120, disabled=True, key=f"{path.name}-{id(block)}")
    if st.button("Tạo và kích hoạt chỉ mục", type="primary", disabled=not course.strip()):
        with st.spinner("Đang trích xuất, chunk và tạo embedding..."):
            metadata = {
                str(path): {"course": course.strip(), "source_type": source_type, "semester": semester}
                for path in paths
            }
            chunking = {**defaults["chunking"], "strategy": strategy, "prefix": use_prefix}
            result = build_corpus(paths, metadata, settings, db, chunking)
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
            db.set_active_corpus(result.version_id)
        st.success(f"Đã kích hoạt corpus {result.version_id} với {result.chunk_count} chunks.")

documents = db.list_documents()
if documents:
    st.subheader("Tài liệu đã xử lý")
    st.dataframe(documents, use_container_width=True)
