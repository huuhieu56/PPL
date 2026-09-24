from types import SimpleNamespace

from src.models import PipelineConfig, RetrievedChunk, StageScores
from src.pipeline import PipelineResult
from src.rag import REFUSAL_TEXT, answer_question
from tests.fakes import make_chunk


class FakePipeline:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def run(self, query, config, use_cache=True):
        self.calls.append(use_cache)
        return PipelineResult(self.results, {"sparse": 1.0, "dense": 1.0, "fusion": 0.0, "rerank": 0.0, "total": 2.0})


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Theo tài liệu [1] và [99]."))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def _results():
    chunk = make_chunk("c1", "Mã môn AI101", doc_title="Slide AI", heading_path=("Bài 1",), page=3)
    return [RetrievedChunk(chunk, StageScores(rank=1, fusion_score=0.8))]


def test_answer_keeps_valid_citations_and_uses_config():
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    pipeline = FakePipeline(_results())
    config = PipelineConfig(rerank=False, temperature=0.2, timeout_seconds=30)

    answer = answer_question("Mã môn AI101 là gì?", pipeline, config, client, "test-model")

    assert "[1]" in answer.text and "[99]" not in answer.text
    assert answer.citations[0]["chunk_id"] == "c1"
    assert answer.citations[0]["heading_path"] == ["Bài 1"]
    assert "Slide AI > Bài 1" in completions.calls[0]["messages"][1]["content"]
    assert completions.calls[0]["temperature"] == 0.2 and completions.calls[0]["timeout"] == 30
    assert pipeline.calls == [False]
    assert answer.retrieval_ms == 2.0


def test_answer_refuses_below_threshold_without_calling_llm():
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = PipelineConfig(rerank=False, refusal_threshold=2.0)
    answer = answer_question("Câu hỏi ngoài tài liệu", FakePipeline(_results()), config, client, "m")
    assert answer.refused is True and answer.text == REFUSAL_TEXT
    assert completions.calls == []
