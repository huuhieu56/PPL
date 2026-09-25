from types import SimpleNamespace

from src.models import PipelineConfig, RetrievedChunk, StageScores
from src.pipeline import PipelineResult
from src.rag import answer_question
from src.rewrite import clean_conversational_fillers, rewrite_query
from tests.fakes import make_chunk


class FakePipeline:
    def __init__(self, results):
        self.results = results
        self.queries_received = []

    def run(self, query, config, use_cache=True):
        self.queries_received.append(query)
        return PipelineResult(self.results, {"sparse": 1.0, "dense": 1.0, "fusion": 0.0, "rerank": 0.0, "total": 2.0})


class FakeCompletions:
    def __init__(self, rewrite_reply="Ai dạy môn Nhập môn Khoa học Dữ liệu?"):
        self.rewrite_reply = rewrite_reply
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        messages = kwargs.get("messages", [])
        system_content = messages[0].get("content", "") if messages else ""
        if "Query Rewriter" in system_content:
            content = self.rewrite_reply
        else:
            content = "Theo tài liệu [1], TS. Nguyễn Tất Thắng giảng dạy môn học này."
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
            usage=SimpleNamespace(prompt_tokens=15, completion_tokens=10),
        )


def _sample_results():
    chunk = make_chunk(
        "c1",
        "Giảng viên: TS. Nguyễn Tất Thắng, Bộ môn KHMT",
        doc_title="1 0 Introduction",
        heading_path=("Thông tin liên quan",),
        page=4,
    )
    return [RetrievedChunk(chunk, StageScores(rank=1, fusion_score=0.9))]


def test_clean_conversational_fillers():
    raw_query = "Bạn ơi cho mình hỏi môn AI101 có mấy tín chỉ với ạ?"
    cleaned = clean_conversational_fillers(raw_query)
    assert "bạn ơi" not in cleaned.lower()
    assert "với ạ" not in cleaned.lower()
    assert "AI101" in cleaned
    assert "tín chỉ" in cleaned


def test_rewrite_query_fallback_rule_based():
    query = "Cho em hỏi lịch thi kết thúc học phần với ạ"
    rewritten, changed = rewrite_query(query, enable_llm=False)
    assert changed is True
    assert "cho em hỏi" not in rewritten.lower()
    assert "với ạ" not in rewritten.lower()
    assert "lịch thi kết thúc học phần" in rewritten


def test_rewrite_query_with_llm_coreference():
    completions = FakeCompletions(rewrite_reply="Ai giảng dạy môn Nhập môn Khoa học Dữ liệu?")
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    history = [
        {"role": "user", "content": "Môn Nhập môn Khoa học Dữ liệu học những gì?"},
        {"role": "assistant", "content": "Môn học giới thiệu về phân tích dữ liệu, Python và học máy."},
    ]
    query = "Ai dạy môn đó thế?"
    rewritten, changed = rewrite_query(query, chat_history=history, client=client, model="test-model")
    assert changed is True
    assert rewritten == "Ai giảng dạy môn Nhập môn Khoa học Dữ liệu?"
    assert len(completions.calls) == 1
    assert "Môn Nhập môn Khoa học Dữ liệu" in completions.calls[0]["messages"][1]["content"]


def test_answer_question_integrates_rewritten_query():
    pipeline = FakePipeline(_sample_results())
    completions = FakeCompletions(rewrite_reply="Ai giảng dạy môn Nhập môn Khoa học Dữ liệu?")
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = PipelineConfig(rerank=False)
    history = [{"role": "user", "content": "Môn Nhập môn Khoa học Dữ liệu gồm những gì?"}]

    answer = answer_question(
        query="Ai dạy môn đó thế ạ?",
        pipeline=pipeline,
        config=config,
        client=client,
        model="test-model",
        chat_history=history,
        enable_rewrite=True,
    )

    # Đảm bảo pipeline nhận câu hỏi đã được rewrite để truy xuất
    assert pipeline.queries_received == ["Ai giảng dạy môn Nhập môn Khoa học Dữ liệu?"]
    # Đảm bảo answer lưu lại rewritten_query
    assert answer.rewritten_query == "Ai giảng dạy môn Nhập môn Khoa học Dữ liệu?"
    assert "[1]" in answer.text
    assert answer.citations[0]["chunk_id"] == "c1"
