import re
import time
from dataclasses import dataclass

from src.models import RagConfig
from src.reranking import rerank
from src.retrieval import RetrievalIndex, retrieve


REFUSAL_TEXT = "Không tìm thấy đủ thông tin trong tài liệu để trả lời câu hỏi này."


@dataclass(frozen=True)
class RagAnswer:
    text: str
    citations: list[dict]
    refused: bool
    prompt_tokens: int
    completion_tokens: int
    retrieval_ms: float
    generation_ms: float


def _prompt(query: str, results) -> list[dict]:
    context = "\n\n".join(
        f"[{number}] Tài liệu: {result.chunk.doc_id}; trang/slide: {result.chunk.page}; "
        f"mục: {result.chunk.section}\n{result.chunk.text}"
        for number, result in enumerate(results, start=1)
    )
    return [
        {
            "role": "system",
            "content": (
                "Bạn là trợ lý học tập. Chỉ sử dụng ngữ cảnh được cung cấp. "
                "Gắn [n] ngay sau mỗi phát biểu dựa trên nguồn tương ứng. "
                f"Nếu ngữ cảnh không đủ, trả lời đúng câu: {REFUSAL_TEXT} "
                "Không tạo nguồn, số liệu hoặc citation mới."
            ),
        },
        {"role": "user", "content": f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {query}"},
    ]


def _valid_citations(text: str, results) -> tuple[str, list[dict]]:
    valid_numbers = set(range(1, len(results) + 1))
    cited = []
    seen = set()

    def replace(match):
        number = int(match.group(1))
        if number not in valid_numbers:
            return ""
        if number not in seen:
            seen.add(number)
            result = results[number - 1]
            cited.append(
                {
                    "number": number,
                    "chunk_id": result.chunk.chunk_id,
                    "doc_id": result.chunk.doc_id,
                    "page": result.chunk.page,
                    "section": result.chunk.section,
                    "text": result.chunk.text,
                }
            )
        return match.group(0)

    cleaned = re.sub(r"\[(\d+)\]", replace, text)
    return re.sub(r"\s+([.,;:])", r"\1", cleaned), cited


def answer_question(
    query: str,
    index: RetrievalIndex,
    config: RagConfig,
    client,
    model: str,
) -> RagAnswer:
    retrieval_started = time.perf_counter()
    results = retrieve(query, index, config)
    if config.use_reranker:
        results, _ = rerank(query, results, config.rerank_n)
    results = results[: config.context_k]
    retrieval_ms = (time.perf_counter() - retrieval_started) * 1000
    confidence = results[0].score if results else float("-inf")
    if confidence < config.refusal_threshold:
        return RagAnswer(REFUSAL_TEXT, [], True, 0, 0, retrieval_ms, 0.0)

    generation_started = time.perf_counter()
    last_error = None
    response = None
    for attempt in range(3):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=_prompt(query, results),
                temperature=0,
                timeout=60,
            )
            break
        except Exception as error:
            last_error = error
            if attempt == 2:
                raise
            time.sleep(0.5 * (2**attempt))
    if response is None:
        raise RuntimeError("LLM did not return a response") from last_error
    text, citations = _valid_citations(response.choices[0].message.content or "", results)
    usage = getattr(response, "usage", None)
    refused = text.strip() == REFUSAL_TEXT
    return RagAnswer(
        text=text,
        citations=citations,
        refused=refused,
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        retrieval_ms=retrieval_ms,
        generation_ms=(time.perf_counter() - generation_started) * 1000,
    )
