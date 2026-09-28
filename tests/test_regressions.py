from pathlib import Path
from types import SimpleNamespace

import pymupdf
import pytest

from src.ingestion import PageText, build_hierarchy, extract_document, build_corpus
from src.config import load_settings
from src.storage import Database
from src.ui import citation_label, stage_uploads
from src.models import RagConfig
import src.rag as rag
from src.corpus import index_corpus


def test_docx_locator_independent_of_course_type():
    assert "đoạn 2" in citation_label(
        {"number": 1, "doc_id": "a", "page": 2, "source_type": "textbook"},
        {"a": "biology.docx"},
    )


def test_columns_read_column_before_next_column(tmp_path):
    path = tmp_path / "columns.pdf"
    doc = pymupdf.open()
    page = doc.new_page()
    for i in range(4):
        page.insert_text((30, 50 + i * 25), f"LEFT{i}")
        page.insert_text((320, 50 + i * 25), f"RIGHT{i}")
    doc.save(path)
    text = extract_document(path)[0].text
    assert text.index("LEFT3") < text.index("RIGHT0")


def test_empty_rejected_before_saving_corpus(tmp_path):
    path = tmp_path / "blank.pdf"
    doc = pymupdf.open()
    doc.new_page()
    doc.save(path)
    settings = load_settings(tmp_path)
    db = Database(settings.db_path)
    db.initialize()
    with pytest.raises(ValueError, match="không có nội dung"):
        build_corpus([path], {str(path): {"course": "test", "source_type": "textbook"}}, settings, {"chunk_tokens": 450, "overlap_tokens": 75})
    assert not db.list_documents()


def test_same_basename_preserves_both_uploads(tmp_path):
    class Upload:
        def __init__(self, content):
            self.name = "same.pdf"
            self.content = content

        def getvalue(self):
            return self.content

    first, second = stage_uploads([Upload(b"one"), Upload(b"two")], tmp_path)
    assert first != second
    assert first.name == second.name == "same.pdf"
    assert first.read_bytes() == b"one" and second.read_bytes() == b"two"


def test_followup_question_retrieves_with_conversation_context(monkeypatch):
    searched = []
    monkeypatch.setattr(rag, "retrieve", lambda query, index, config: searched.append(query) or [])
    client = SimpleNamespace(invoke=lambda messages, **kwargs: SimpleNamespace(text="Định nghĩa quang hợp trong sách Sinh học"))
    result = rag.answer_question(
        "Giải thích ý đó", object(), RagConfig(), client, "test-model",
        history=[{"query": "Quang hợp là gì?", "answer": "Quang hợp là quá trình..."}],
    )
    assert searched == ["Định nghĩa quang hợp trong sách Sinh học"]
    assert result.refused


def test_repeated_slide_title_keeps_separate_source_blocks():
    pages = [PageText(1, "Khái niệm", "Định nghĩa A"), PageText(2, "Khái niệm", "Định nghĩa B")]
    chunks, nodes = build_hierarchy(
        pages, doc_id="slide-deck", course="Bất kỳ", source_type="slide",
        file_type="pptx", chunk_tokens=100, overlap_tokens=0,
    )
    assert len(chunks) == 2
    assert len({chunk.parent_id for chunk in chunks}) == 2
    assert {chunk.page for chunk in chunks} == {1, 2}
    assert len([node for node in nodes if node["kind"] == "section"]) == 1


def test_failed_index_does_not_publish_corpus(tmp_path, monkeypatch):
    path = tmp_path / "lesson.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((50, 50), "Bài học Sinh học: tế bào là đơn vị cấu trúc của cơ thể.")
    document.save(path)
    settings = load_settings(tmp_path)
    db = Database(settings.db_path)
    db.initialize()

    def fail_index(*args, **kwargs):
        raise RuntimeError("Qdrant unavailable")

    monkeypatch.setattr("src.corpus.RetrievalIndex.build", fail_index)
    with pytest.raises(RuntimeError, match="Qdrant unavailable"):
        index_corpus(
            [path], {str(path): {"course": "Sinh học", "source_type": "textbook"}},
            settings, db, {"chunk_tokens": 100, "overlap_tokens": 0},
            {"tokenizer": "whitespace", "embedding_model": "test"},
        )
    assert db.get_active_corpus() is None
    assert db.list_documents() == []


def test_chat_history_is_private_to_its_owner(tmp_path):
    db = Database(tmp_path / "app.db")
    db.initialize()
    db.start_chat_session("session-1", "student-a", "corpus-1")
    with pytest.raises(PermissionError):
        db.save_message({"session_id": "session-1", "username": "student-b", "query": "Q", "answer": "A"})
    db.save_message({"session_id": "session-1", "username": "student-a", "query": "Q", "answer": "A"})
    assert db.chat_turns("session-1", "student-b") == []
    assert len(db.chat_turns("session-1", "student-a")) == 1
    db.start_chat_session("scoped", "student-a", "corpus-1", ("d2", "d1"))
    assert [s["session_id"] for s in db.list_chat_sessions("student-a", "corpus-1")] == ["session-1"]
    assert [s["session_id"] for s in db.list_chat_sessions("student-a", "corpus-1", ("d1", "d2"))] == ["scoped"]


@pytest.mark.parametrize("text", ["", "Tự bịa một đáp án.", "Đáp án [99].", rag.REFUSAL_TEXT])
def test_generated_answer_without_valid_source_is_refused(monkeypatch, text):
    from src.models import Chunk, SearchResult
    chunk = Chunk("c1", "d1", "Sinh học", "textbook", 1, "", "Tế bào là đơn vị cấu trúc.")
    monkeypatch.setattr(rag, "retrieve", lambda *_args: [SearchResult(chunk, 1.0, 1, "dense")])
    client = SimpleNamespace(invoke=lambda *_args, **_kwargs: SimpleNamespace(text=text, usage_metadata={}))
    answer = rag.answer_question("Tế bào là gì?", SimpleNamespace(expand_chunk=lambda c, _n: c), RagConfig(), client, "test")
    assert answer.refused and answer.text == rag.REFUSAL_TEXT and not answer.citations
