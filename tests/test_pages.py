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
