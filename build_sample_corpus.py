"""Index local teaching slides and other example learning materials."""

from pathlib import Path

from src.config import load_settings, load_yaml
from src.corpus import index_corpus
from src.storage import Database


def main() -> None:
    settings = load_settings()
    slides = sorted((settings.data_dir / "sample_slides").rglob("*.pdf"))
    other = sorted(path for path in (settings.data_dir / "sample_materials").rglob("*") if path.suffix.lower() in {".pdf", ".docx", ".pptx"})
    files = slides + other
    if not files:
        raise FileNotFoundError("No supported documents found under data/sample_slides or data/sample_materials")
    database = Database(settings.db_path)
    database.initialize()
    metadata = {
        str(path): {"course": path.parent.name, "source_type": "docx" if path.suffix.lower() == ".docx" else "slide" if path in slides or path.suffix.lower() == ".pptx" else "textbook"}
        for path in files
    }
    retrieval = load_yaml("configs/default.yaml")["retrieval"]
    corpus = index_corpus(files, metadata, settings, database, {"chunk_tokens": 450, "overlap_tokens": 75}, retrieval)
    print(f"Đã kích hoạt {corpus.version_id}: {len(files)} tài liệu, {corpus.chunk_count} chunks")


if __name__ == "__main__":
    main()
