import json

from pptx import Presentation
from pptx.util import Inches

from src.config import load_settings
from src.ingestion import build_corpus, chunk_pages, extract_document
from src.models import RagConfig
from src.storage import Database


def test_foundation_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-persisted")
    settings = load_settings(tmp_path)
    db = Database(settings.db_path)
    db.initialize()
    db.save_rag_config("demo", RagConfig(method="rrf", use_reranker=False))

    saved = db.list_rag_configs()
    assert saved[0]["name"] == "demo"
    assert saved[0]["config"]["method"] == "rrf"
    assert "must-not-be-persisted" not in json.dumps(saved)


def test_ingestion_preserves_source_and_reading_order(tmp_path):
    path = tmp_path / "lesson.pptx"
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = "Bài 1"
    left = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(3), Inches(1))
    right = slide.shapes.add_textbox(Inches(5), Inches(2), Inches(3), Inches(1))
    left.text = "Nội dung bên trái"
    right.text = "Nội dung bên phải"
    deck.save(path)

    pages = extract_document(path)
    assert pages[0].text.index("Bài 1") < pages[0].text.index("Nội dung bên trái")
    assert pages[0].text.index("Nội dung bên trái") < pages[0].text.index("Nội dung bên phải")

    chunks = chunk_pages(
        pages,
        doc_id="doc-1",
        course="AI101",
        source_type="slide",
        chunk_tokens=450,
        overlap_tokens=75,
    )
    assert chunks
    assert all(
        chunk.doc_id == "doc-1"
        and chunk.course == "AI101"
        and chunk.page == 1
        and chunk.text
        for chunk in chunks
    )

    settings = load_settings(tmp_path)
    db = Database(settings.db_path)
    db.initialize()
    result = build_corpus(
        [path],
        {str(path): {"course": "AI101", "source_type": "slide"}},
        settings,
        db,
        {"chunk_tokens": 450, "overlap_tokens": 75},
    )
    assert result.chunk_count == len(chunks)
    assert result.chunks_path.exists()
    assert db.get_active_corpus() is None
