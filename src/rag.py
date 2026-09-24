import re
import time
from dataclasses import dataclass

from src.models import PipelineConfig


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
        f"[{number}] Nguồn: {result.chunk.breadcrumb()}; trang/slide: {result.chunk.page}\n"
        f"{result.chunk.body}"
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
                    "doc_title": result.chunk.doc_title,
                    "heading_path": list(result.chunk.heading_path),
                    "page": result.chunk.page,
                    "text": result.chunk.body,
                }
            )
        return match.group(0)

    cleaned = re.sub(r"\[(\d+)\]", replace, text)
    return re.sub(r"\s+([.,;:])", r"\1", cleaned), cited


def answer_question(
    query: str,
    pipeline,
    config: PipelineConfig,
    client,
    model: str,
) -> RagAnswer:
    retrieval = pipeline.run(query, config, use_cache=False)
    results = retrieval.results[: config.context_k]
    retrieval_ms = retrieval.timings_ms["total"]
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
                temperature=config.temperature,
                timeout=config.timeout_seconds,
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
    return RagAnswer(
        text=text,
        citations=citations,
        refused=text.strip() == REFUSAL_TEXT,
        prompt_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        completion_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        retrieval_ms=retrieval_ms,
        generation_ms=(time.perf_counter() - generation_started) * 1000,
    )
