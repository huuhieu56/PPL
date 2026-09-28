import pymupdf
from qdrant_client import QdrantClient

import build_sample_corpus
from src.config import load_settings
from src.retrieval import RetrievalIndex
from src.storage import Database


def test_sample_builder_activates_index_with_hierarchy(tmp_path, monkeypatch):
    settings = load_settings(tmp_path)
    source = settings.data_dir / "sample_materials" / "Toan" / "lesson.pdf"
    source.parent.mkdir(parents=True)
    pdf = pymupdf.open()
    page = pdf.new_page()
    page.insert_text((72, 72), "Định lý Pythagoras trong tam giác vuông")
    pdf.save(source)
    pdf.close()

    class Encoder:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, texts, **_kwargs):
            return [[1.0, 0.0] for _ in texts]

    client = QdrantClient(":memory:")
    monkeypatch.setattr("sentence_transformers.SentenceTransformer", Encoder)
    monkeypatch.setattr("src.retrieval.QdrantClient", lambda *args, **kwargs: client)
    monkeypatch.setattr(build_sample_corpus, "load_settings", lambda: settings)
    build_sample_corpus.main()

    active = Database(settings.db_path).get_active_corpus()
    index = RetrievalIndex.load(settings.data_dir / "indexes" / active["version_id"])
    assert active["chunk_count"] == 1
    assert index.hierarchy
    assert next(iter(index.chunks.values())).parent_id in index.hierarchy
