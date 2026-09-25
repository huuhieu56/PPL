import json
import re
from types import SimpleNamespace

import pytest

from src.bench.generate import find_partners, generate_drafts, parse_json_reply, sample_chunks
from src.index import RetrievalIndex
from tests.fakes import FakeEncoder, make_chunk

BODIES = {
    "c1": "Học phần AI101 giới thiệu học máy có giám sát và không giám sát cho sinh viên năm nhất",
    "c2": "Mã CS202 là môn cơ sở dữ liệu với khóa chính khóa ngoại và chuẩn hóa lược đồ quan hệ",
    "c3": "Thuật toán ID3 xây dựng cây quyết định bằng độ lợi thông tin trên tập huấn luyện nhỏ",
    "c4": "Giao thức TCP cổng 80 đảm bảo truyền tin cậy nhờ cơ chế xác nhận và truyền lại gói tin",
}


@pytest.fixture
def index(tmp_path):
    chunks = [make_chunk(chunk_id, body, heading_path=("Chương 1",)) for chunk_id, body in BODIES.items()]
    return RetrievalIndex.build(chunks, tmp_path / "idx", embedding_model="fake", encoder=FakeEncoder())


class ScriptedClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)

    def create(self, model, messages, temperature, **kwargs):
        prompt = messages[-1]["content"]
        passages = re.findall(r"\[Đoạn \d+\][^\n]*\n(.+?)(?=\n\n\[Đoạn|\Z)", prompt, flags=re.DOTALL)
        first = passages[0].strip()
        quote = " ".join(first.split()[:5])
        if "nhiều đoạn" in prompt:
            payload = {
                "question": "So sánh hai nội dung trên?",
                "evidence_quotes": [" ".join(passage.split()[:5]) for passage in passages],
            }
        elif "BẮT BUỘC chứa nguyên văn" in prompt:
            code = next(word for word in first.split() if any(ch.isdigit() for ch in word))
            payload = {"question": f"{code} là gì?", "evidence_quote": quote}
        elif "KHÔNG dùng lại" in prompt:
            payload = {"question": first, "evidence_quote": quote}
        else:
            payload = {"question": "Khái niệm này có ý nghĩa ra sao?", "evidence_quote": quote}
        content = f"Đây là kết quả:\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_parse_json_reply_handles_fences_and_garbage():
    assert parse_json_reply('```json\n{"question": "A?"}\n```') == {"question": "A?"}
    assert parse_json_reply('Kết quả: {"a": {"b": 1}} xong') == {"a": {"b": 1}}
    with pytest.raises(ValueError):
        parse_json_reply("không có JSON")


def test_sample_chunks_is_stratified_and_deterministic():
    chunks = [
        make_chunk(f"{doc}-{number}", "một hai ba bốn năm sáu bảy tám", doc_id=doc)
        for doc in ("d1", "d2")
        for number in range(3)
    ]
    picked = sample_chunks(chunks, 4, seed=7, min_words=5)
    assert len(picked) == 4
    assert {chunk.doc_id for chunk in picked} == {"d1", "d2"}
    assert picked == sample_chunks(chunks, 4, seed=7, min_words=5)
    assert sample_chunks(chunks, 4, seed=7, min_words=50) == []


def test_find_partners_prefers_same_document_section(index):
    partners = find_partners(index.chunks["c1"], index, max_partners=2)
    assert len(partners) == 2
    assert all(partner.chunk_id != "c1" for partner in partners)


def test_generate_drafts_validates_each_category(index):
    accepted, rejected = generate_drafts(index, ScriptedClient(), "fake-llm", per_category=1, seed=3, min_words=5)
    categories = sorted(draft["category"] for draft in accepted)
    assert categories == ["concept", "exact", "multi"]
    assert [row["category"] for row in rejected] == ["paraphrase"]
    assert rejected[0]["reasons"][0].startswith("paraphrase:")
    multi = next(draft for draft in accepted if draft["category"] == "multi")
    assert 2 <= len(multi["source_chunk_ids"]) <= 3
    assert len(multi["evidence"]) == len(multi["source_chunk_ids"])
    assert all(
        draft["origin"] == "llm" and draft["split"] is None and draft["generator"] == "fake-llm"
        for draft in accepted
    )
    again, _ = generate_drafts(index, ScriptedClient(), "fake-llm", per_category=1, seed=3, min_words=5)
    assert [draft["query_id"] for draft in again] == [draft["query_id"] for draft in accepted]


def test_generate_drafts_records_non_json_reply_as_rejected(index):
    class BrokenClient(ScriptedClient):
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="xin lỗi"))])

    accepted, rejected = generate_drafts(
        index, BrokenClient(), "m", per_category=1, categories=("concept",), min_words=5
    )
    assert accepted == []
    assert rejected[0]["reasons"][0].startswith("json:")
