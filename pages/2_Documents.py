import json
from pathlib import Path

import streamlit as st

from src.config import load_settings, load_yaml
from src.ingestion import build_corpus, extract_document
from src.models import Chunk
from src.retrieval import RetrievalIndex
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
            for page in extract_document(path)[:5]:
                if page.warning:
                    st.warning(f"Trang {page.page}: {page.warning}")
                st.text_area(f"Trang/slide {page.page}", page.text, height=140, disabled=True)
    if st.button("Tạo và kích hoạt chỉ mục", type="primary", disabled=not course.strip()):
        with st.spinner("Đang trích xuất, chunk và tạo embedding..."):
            metadata = {
                str(path): {"course": course.strip(), "source_type": source_type, "semester": semester}
                for path in paths
            }
            retrieval = defaults["retrieval"]
            result = build_corpus(
                paths,
                metadata,
                settings,
                db,
                {"chunk_tokens": 450, "overlap_tokens": 75},
            )
            chunks = [
                Chunk(**json.loads(line))
                for line in result.chunks_path.read_text(encoding="utf-8").splitlines()
                if line
            ]
            RetrievalIndex.build(
                chunks,
                settings.data_dir / "indexes" / result.version_id,
                retrieval["tokenizer"],
                retrieval["embedding_model"],
            )
            db.set_active_corpus(result.version_id)
        st.success(f"Đã kích hoạt corpus {result.version_id} với {result.chunk_count} chunks.")

documents = db.list_documents()
if documents:
    st.subheader("Tài liệu đã xử lý")
    st.dataframe(documents, use_container_width=True)
