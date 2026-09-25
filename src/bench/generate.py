import hashlib
import json
import random
from collections import defaultdict

from src.bench.checks import check_exact, check_paraphrase, contains_quote
from src.text import normalize_text

CATEGORIES = ("exact", "concept", "paraphrase", "multi")
_SYSTEM = (
    "Bạn là giảng viên soạn câu hỏi để kiểm tra hệ thống tìm kiếm tài liệu học tập tiếng Việt. "
    "Chỉ dựa vào các đoạn tài liệu được cung cấp. Chỉ trả về DUY NHẤT một đối tượng JSON."
)
_INSTRUCTIONS = {
    "exact": (
        "Viết một câu hỏi ngắn như người học tra cứu. Câu hỏi BẮT BUỘC chứa nguyên văn một mã, ký hiệu, "
        "số hiệu hoặc thuật ngữ hiếm xuất hiện trong đoạn."
    ),
    "concept": (
        "Viết một câu hỏi về định nghĩa, nguyên lý hoặc ý nghĩa của một khái niệm trong đoạn, "
        "diễn đạt như sinh viên hỏi, không chép nguyên câu trong đoạn."
    ),
    "paraphrase": (
        "Viết một câu hỏi mà câu trả lời nằm trong đoạn nhưng KHÔNG dùng lại các cụm từ đặc trưng của đoạn: "
        "dùng từ đồng nghĩa, đổi cấu trúc câu hoặc diễn đạt gián tiếp."
    ),
    "multi": (
        "Có nhiều đoạn tài liệu. Viết một câu hỏi so sánh hoặc tổng hợp chỉ trả lời đầy đủ được khi dùng "
        "TẤT CẢ các đoạn."
    ),
}
_SINGLE_FORMAT = '{"question": "<câu hỏi>", "evidence_quote": "<trích NGUYÊN VĂN một câu ngắn trong đoạn chứa câu trả lời>"}'
_MULTI_FORMAT = '{"question": "<câu hỏi>", "evidence_quotes": ["<trích nguyên văn từ Đoạn 1>", "<trích nguyên văn từ Đoạn 2>", "..."]}'


def sample_chunks(chunks, count: int, seed: int, min_words: int = 20):
    groups = defaultdict(list)
    for chunk in chunks:
        if len(chunk.body.split()) >= min_words:
            groups[(chunk.course, chunk.source_type, chunk.doc_id)].append(chunk)
    rng = random.Random(seed)
    queues = []
    for key in sorted(groups):
        members = sorted(groups[key], key=lambda chunk: chunk.chunk_id)
        rng.shuffle(members)
        queues.append(members)
    picked = []
    while len(picked) < count and any(queues):
        for queue in queues:
            if queue and len(picked) < count:
                picked.append(queue.pop(0))
    return picked


def find_partners(chunk, index, max_partners: int = 2, min_cosine: float = 0.6):
    position = {chunk_id: row for row, chunk_id in enumerate(index.chunk_ids)}
    similarities = index.embeddings @ index.embeddings[position[chunk.chunk_id]]
    root = chunk.heading_path[:1]
    same_section = [
        other
        for other in index.chunk_list
        if other.chunk_id != chunk.chunk_id and other.doc_id == chunk.doc_id and other.heading_path[:1] == root
    ]
    candidates = same_section or [
        other
        for other in index.chunk_list
        if other.chunk_id != chunk.chunk_id and similarities[position[other.chunk_id]] >= min_cosine
    ]
    candidates.sort(key=lambda other: (-float(similarities[position[other.chunk_id]]), other.chunk_id))
    return candidates[:max_partners]


def build_prompt(category: str, sources) -> list[dict]:
    passages = "\n\n".join(
        f"[Đoạn {number}] ({source.breadcrumb()})\n{source.body}" for number, source in enumerate(sources, start=1)
    )
    output_format = _MULTI_FORMAT if category == "multi" else _SINGLE_FORMAT
    return [
        {"role": "system", "content": _SYSTEM},
        {"role": "user", "content": f"{_INSTRUCTIONS[category]}\nĐịnh dạng JSON: {output_format}\n\n{passages}"},
    ]


def parse_json_reply(text: str) -> dict:
    start = text.find("{")
    while start != -1:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == "{":
                depth += 1
            elif text[end] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        value = json.loads(text[start : end + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(value, dict):
                        return value
                    break
        start = text.find("{", start + 1)
    raise ValueError("Reply does not contain a JSON object")


def _validate(category: str, question: str, quotes: list[str], sources, max_overlap: float) -> list[str]:
    reasons = []
    if not question:
        return ["empty: thiếu câu hỏi"]
    if len(quotes) != len(sources):
        reasons.append(f"evidence: cần {len(sources)} trích dẫn, nhận {len(quotes)}")
    for quote, source in zip(quotes, sources):
        if not contains_quote(quote, source.body):
            reasons.append(f"evidence: trích dẫn không có trong chunk {source.chunk_id}")
    if category == "exact" and (reason := check_exact(question, sources[0].body)):
        reasons.append(reason)
    if category == "paraphrase" and (reason := check_paraphrase(question, sources[0].body, max_overlap)):
        reasons.append(reason)
    return reasons


def generate_drafts(
    index,
    client,
    model: str,
    per_category: int,
    seed: int = 42,
    categories=CATEGORIES,
    max_overlap: float = 0.2,
    min_words: int = 20,
) -> tuple[list[dict], list[dict]]:
    accepted, rejected = [], []
    for offset, category in enumerate(categories):
        for chunk in sample_chunks(index.chunk_list, per_category, seed + offset, min_words):
            sources = [chunk]
            if category == "multi":
                partners = find_partners(chunk, index)
                if not partners:
                    rejected.append({"category": category, "source_chunk_ids": [chunk.chunk_id], "reasons": ["multi: không tìm được chunk liên quan"]})
                    continue
                sources += partners
            source_ids = [source.chunk_id for source in sources]
            try:
                response = client.chat.completions.create(
                    model=model, messages=build_prompt(category, sources), temperature=0.7
                )
                reply = parse_json_reply(response.choices[0].message.content or "")
            except ValueError as error:
                rejected.append({"category": category, "source_chunk_ids": source_ids, "reasons": [f"json: {error}"]})
                continue
            question = normalize_text(str(reply.get("question", "")))
            quotes = reply.get("evidence_quotes") if category == "multi" else [reply.get("evidence_quote", "")]
            quotes = [normalize_text(str(quote)) for quote in (quotes or [])]
            query_id = "q-" + hashlib.sha256(f"{category}|{'|'.join(source_ids)}|{question}".encode("utf-8")).hexdigest()[:10]
            draft = {
                "query_id": query_id,
                "text": question,
                "category": category,
                "origin": "llm",
                "split": None,
                "source_chunk_ids": source_ids,
                "evidence": [{"chunk_id": source.chunk_id, "quote": quote} for source, quote in zip(sources, quotes)],
                "generator": model,
            }
            reasons = _validate(category, question, quotes, sources, max_overlap)
            if reasons:
                rejected.append({**draft, "reasons": reasons})
            else:
                accepted.append(draft)
    return accepted, rejected
