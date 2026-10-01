from streamlit.testing.v1 import AppTest
from dataclasses import replace
from pathlib import Path
import json
import pytest

from src.config import load_settings
from src.storage import Database
from src.ui import citation_label
from src.models import Chunk


@pytest.mark.parametrize("page", ["2_Documents.py", "3_RAG_Settings.py"])
def test_student_cannot_execute_admin_page_even_without_navigation(page):
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "pages" / page))
    app.session_state["user"] = {"username": "student", "role": "student"}
    app.run(timeout=30)
    assert not app.exception
    assert any("không có quyền" in error.value for error in app.error)
    assert len(app.button) == 0


def test_unauthenticated_documents_page_does_not_crash():
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "pages" / "2_Documents.py"))
    app.run(timeout=30)
    assert not app.exception


def test_citation_uses_readable_source_name():
    citation = {"number": 1, "doc_id": "sha123", "page": 4, "page_end": 5}
    assert citation_label(citation, {"sha123": "Giáo trình Toán.pdf"}) == "[1] Giáo trình Toán.pdf — trang/slide 4–5"
    assert citation_label({**citation, "source_type": "docx"}, {"sha123": "Giáo trình Toán.docx"}) == "[1] Giáo trình Toán.docx — đoạn 4–5"


def test_chat_does_not_prompt_for_feedback_by_default(tmp_path, monkeypatch):
    settings = replace(load_settings(tmp_path), llm_api_key="test-key", llm_model="test-model")
    db = Database(settings.db_path)
    db.initialize()
    db.publish_corpus([], {"version_id": "test-corpus"})
    db.start_chat_session("s1", "test", "test-corpus")
    db.save_message({"message_id": "m1", "session_id": "s1", "username": "test", "query": "Câu hỏi", "answer": "Trả lời [1]", "citations": []})
    monkeypatch.setattr("src.config.load_settings", lambda: settings)
    monkeypatch.setattr("src.ui.load_settings", lambda: settings)
    from types import SimpleNamespace
    monkeypatch.setattr("src.index.RetrievalIndex.load", lambda path: SimpleNamespace(chunks={}))
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.session_state["user"] = {"username": "test", "role": "student"}
    app.run(timeout=30).switch_page("pages/1_Chat.py").run(timeout=30)
    assert not app.exception
    assert len(app.radio) == 0
    app.button(key="feedback-m1").click().run(timeout=30)
    assert len(app.radio) == 1


def test_rag_settings_exposes_parent_context_budget(tmp_path, monkeypatch):
    settings = load_settings(tmp_path)
    monkeypatch.setattr("src.config.load_settings", lambda: settings)
    monkeypatch.setattr("src.ui.load_settings", lambda: settings)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.session_state["user"] = {"username": "admin", "role": "admin"}
    app.run(timeout=30).switch_page("pages/3_RAG_Settings.py").run(timeout=30)
    assert not app.exception
    assert any(control.label == "Ngữ cảnh cha tối đa (từ)" for control in app.number_input)


def test_documents_page_shows_active_hierarchy_counts(tmp_path, monkeypatch):
    settings = load_settings(tmp_path)
    db = Database(settings.db_path)
    db.initialize()
    db.publish_corpus([], {"version_id": "tree-v1", "chunk_count": 42, "node_count": 9})
    monkeypatch.setattr("src.config.load_settings", lambda: settings)
    monkeypatch.setattr("src.ui.load_settings", lambda: settings)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.session_state["user"] = {"username": "admin", "role": "admin"}
    app.run(timeout=30).switch_page("pages/2_Documents.py").run(timeout=30)
    assert not app.exception
    assert any("42 chunk lá" in item.value and "9 nút cấu trúc" in item.value for item in app.caption)


def test_admin_can_submit_upload_and_inspect_saved_chunk(tmp_path, monkeypatch):
    settings = load_settings(tmp_path)
    db = Database(settings.db_path)
    db.initialize()
    corpus = settings.data_dir / "processed" / "test-version"
    corpus.mkdir(parents=True)
    (corpus / "manifest.json").write_text(json.dumps({"documents": [{"doc_id": "doc-1", "course": "Sinh học"}]}), encoding="utf-8")
    chunk = Chunk("chunk-1", "doc-1", "Sinh học", "slide", 3, "Tế bào", "Tế bào là đơn vị cơ bản của sự sống.")
    (corpus / "chunks.jsonl").write_text(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n", encoding="utf-8")
    db.publish_corpus([{"doc_id": "doc-1", "filename": "biology.pdf", "source_path": str(tmp_path / "biology.pdf")}], {
        "version_id": "test-version", "chunks_path": str(corpus / "chunks.jsonl"), "chunk_count": 1, "node_count": 1,
    })
    monkeypatch.setattr("src.config.load_settings", lambda: settings)
    monkeypatch.setattr("src.ui.load_settings", lambda: settings)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.session_state["user"] = {"username": "admin", "role": "admin"}
    app.run().switch_page("pages/2_Documents.py").run()
    assert not app.exception
    assert any(button.label == "Bắt đầu xử lý" for button in app.button)
    assert any(button.label == "Xóa tài liệu" for button in app.button)
    assert any("Tế bào là đơn vị cơ bản" in area.value for area in app.text_area)


def test_overview_counts_active_manifest_not_historical_registry(tmp_path, monkeypatch):
    settings = load_settings(tmp_path)
    db = Database(settings.db_path)
    db.initialize()
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "manifest.json").write_text(json.dumps({"documents":[{"doc_id":"current", "course":"Sinh học"}]}))
    db.publish_corpus([{"doc_id":"old","course":"Toán"},{"doc_id":"current","course":"Sinh học"}], {"version_id":"one", "chunks_path":str(corpus / "chunks.jsonl"), "chunk_count":5})
    monkeypatch.setattr("src.ui.load_settings", lambda: settings)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.session_state["user"] = {"username":"student", "role":"student"}
    app.run()
    assert not app.exception
    assert [metric.value for metric in app.metric] == ["1", "1", "5"]


def test_document_scope_survives_leaving_chat(tmp_path, monkeypatch):
    from types import SimpleNamespace
    settings = replace(load_settings(tmp_path), llm_api_key="test-key", llm_model="test-model")
    db = Database(settings.db_path)
    db.initialize()
    db.publish_corpus([], {"version_id":"scope-navigation"})
    index = SimpleNamespace(chunks={"c":SimpleNamespace(doc_id="doc")})
    index.scoped = lambda _ids: index
    monkeypatch.setattr("src.config.load_settings", lambda: settings)
    monkeypatch.setattr("src.ui.load_settings", lambda: settings)
    monkeypatch.setattr("src.index.RetrievalIndex.load", lambda _path: index)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py"))
    app.session_state["user"] = {"username":"admin", "role":"admin"}
    app.run().switch_page("pages/1_Chat.py").run()
    app.multiselect[0].set_value(["doc"]).run()
    app.switch_page("pages/2_Documents.py").run()
    app.switch_page("pages/1_Chat.py").run()
    assert not app.exception and app.multiselect[0].value == ["doc"]
