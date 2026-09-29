from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.storage import Database

PAGES = Path(__file__).resolve().parent.parent / "pages"


def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("PPL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PPL_RUNS_DIR", str(tmp_path / "runs"))


def test_rag_settings_page_saves_pipeline_config(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    app = AppTest.from_file(str(PAGES / "3_RAG_Settings.py"), default_timeout=30)
    app.session_state["user"] = {"username": "admin", "role": "admin"}
    app.run()
    app.text_input(key="config_name").set_value("hybrid-rrf")
    app.selectbox(key="branches").set_value("Hybrid")
    app.selectbox(key="fusion").set_value("rrf")
    app.button(key="save_config").click()
    app.run()
    assert not app.exception
    configs, _ = Database(tmp_path / "data" / "app.db").load_pipeline_configs()
    assert configs["hybrid-rrf"].fusion == "rrf"


def test_rag_settings_page_shows_validation_error(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    app = AppTest.from_file(str(PAGES / "3_RAG_Settings.py"), default_timeout=30)
    app.session_state["user"] = {"username": "admin", "role": "admin"}
    app.run()
    app.number_input(key="rerank_n").set_value(5)
    app.number_input(key="context_k").set_value(10)
    app.button(key="save_config").click()
    app.run()
    assert any("context_k" in error.value for error in app.error)


def test_chat_page_without_active_corpus_shows_info(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    app = AppTest.from_file(str(PAGES / "1_Chat.py"), default_timeout=30)
    app.session_state["user"] = {"username": "student", "role": "student"}
    app.run()
    assert not app.exception
    assert app.info


def test_experiments_page_warns_on_lock_violation(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    run_dir = tmp_path / "runs" / "20260101T000000Z-test-abcd1234"
    (run_dir / "report").mkdir(parents=True)
    (run_dir / "status.json").write_text('{"status": "completed"}', encoding="utf-8")
    (run_dir / "config.json").write_text('{"split": "test", "lock": {"lock_violation": true}}', encoding="utf-8")
    (run_dir / "report" / "table_3_5.md").write_text("**Bảng 3.5. Kết quả tổng thể trên tập test**", encoding="utf-8")
    app = AppTest.from_file(str(PAGES / "4_Experiments.py"), default_timeout=30)
    app.session_state["user"] = {"username": "admin", "role": "admin"}
    app.run()
    assert not app.exception
    assert any("khóa" in warning.value for warning in app.warning)
    assert any("Bảng 3.5" in block.value for block in app.markdown)


def test_chat_page_scopes_documents_and_keeps_sessions(tmp_path, monkeypatch):
    from src.index import RetrievalIndex
    from tests.fakes import FakeEncoder, make_chunk

    _env(tmp_path, monkeypatch)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("OPENAI_MODEL", "test-model")
    RetrievalIndex.build(
        [make_chunk(f"c{n}", f"nội dung {n}", doc_id=f"d{n % 2}", doc_title=f"Tài liệu {n % 2}") for n in range(4)],
        tmp_path / "data" / "indexes" / "v1",
        embedding_model="fake",
        encoder=FakeEncoder(),
    )
    db = Database(tmp_path / "data" / "app.db")
    db.initialize()
    db.save_corpus_version({"version_id": "v1", "chunk_count": 4})
    db.set_active_corpus("v1")

    app = AppTest.from_file(str(PAGES / "1_Chat.py"), default_timeout=30)
    app.session_state["user"] = {"username": "student", "role": "student"}
    app.run()
    assert not app.exception
    assert app.sidebar.multiselect[0].options == ["Tài liệu 0", "Tài liệu 1"]
    assert len(db.list_chat_sessions("student", "v1", ())) == 1
    app.sidebar.multiselect[0].select("d1")
    app.run()
    assert not app.exception
    assert len(db.list_chat_sessions("student", "v1", ("d1",))) == 1
    assert len(db.list_chat_sessions("student", "v1", ())) == 1


def test_documents_page_renders_without_corpus(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    monkeypatch.chdir(PAGES.parent)
    app = AppTest.from_file(str(PAGES / "2_Documents.py"), default_timeout=30)
    app.session_state["user"] = {"username": "admin", "role": "admin"}
    app.run()
    assert not app.exception
