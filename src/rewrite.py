import re
from typing import Any

from src.text import normalize_text

CONVERSATIONAL_PREFIXES = [
    r"^(?:bạn\s+ơi\s*[,:]?\s*)",
    r"^(?:ad\s+ơi\s*[,:]?\s*)",
    r"^(?:admin\s+ơi\s*[,:]?\s*)",
    r"^(?:thầy\s+cô\s+ơi\s*[,:]?\s*)",
    r"^(?:cho\s+(?:mình|em|tôi)\s+hỏi\s*[,:]?\s*)",
    r"^(?:làm\s+ơn\s+(?:cho\s+biết|nói\s+cho\s+tôi|giải\s+thích)\s*[,:]?\s*)",
    r"^(?:hãy\s+(?:cho\s+tôi\s+biết|giải\s+thích|tóm\s+tắt|trình\s+bày)\s*[,:]?\s*)",
    r"^(?:xin\s+hỏi\s*[,:]?\s*)",
    r"^(?:có\s+thể\s+cung\s+cấp\s+cho\s+tôi\s+về\s*)",
    r"^(?:có\s+thể\s+cho\s+(?:mình|em|tôi)\s+biết\s*)",
    r"^(?:bạn\s+có\s+biết\s*)",
    r"^(?:tôi\s+muốn\s+biết\s*)",
    r"^(?:cho\s+hỏi\s*)",
]

CONVERSATIONAL_SUFFIXES = [
    r"(?:\s+(?:ạ|nha|nhé|nhe|được\s+không|được\s+ko|với\s+ạ|với|giúp\s+(?:em|mình)\s+với)\s*[?!.]*)$",
]

REWRITE_SYSTEM_PROMPT = """Bạn là chuyên gia tiền xử lý và chuẩn hóa câu hỏi truy xuất học liệu (Query Rewriter).
Nhiệm vụ: Chuyển câu hỏi của người dùng (kèm lịch sử hội thoại nếu có) thành MỘT CÂU TRUY VẤN TÌM KIẾM ĐỘC LẬP DUY NHẤT bằng tiếng Việt, tối ưu cho việc tìm kiếm tài liệu giáo trình và slide bài giảng.

Nguyên tắc bắt buộc:
1. Giải quyết từ quy chiếu (Coreference Resolution): Nếu câu hỏi chứa đại từ hoặc từ thay thế ("nó", "môn này", "thầy đó", "môn đó", "phần này", "cái này"), hãy thay thế bằng tên thực thể cụ thể đã xuất hiện trong lịch sử hội thoại.
2. Loại bỏ từ ngữ xã giao, đàm thoại: Lược bỏ các từ như "bạn ơi", "cho mình hỏi", "hãy cho tôi biết", "ạ", "nhé", "với".
3. Tối ưu từ khóa chuyên môn: Diễn đạt rõ ràng ý định tìm kiếm, giữ nguyên các mã môn học (ví dụ: AI101, INT1340), thuật ngữ kỹ thuật, công thức, số liệu.
4. Tuyệt đối KHÔNG trả lời câu hỏi. CHỈ XUẤT RA DUY NHẤT 1 CÂU TRUY VẤN ĐÃ ĐƯỢC CHUẨN HÓA.
5. Không thêm dấu ngoặc kép, không thêm lời dẫn như "Câu hỏi chuẩn hóa:", "Truy vấn:".
6. Nếu câu hỏi ban đầu đã rõ ràng, đầy đủ ngữ cảnh và độc lập, hãy giữ nguyên cấu trúc và chỉ chuẩn hóa từ ngữ."""


def clean_conversational_fillers(query: str) -> str:
    """Làm sạch các từ đệm xã giao bằng biểu thức chính quy (Rule-based)."""
    text = normalize_text(query)
    for pattern in CONVERSATIONAL_PREFIXES:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    for pattern in CONVERSATIONAL_SUFFIXES:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
    return text.strip() or query.strip()


def _format_history_context(chat_history: list[dict], max_turns: int = 3) -> str:
    """Định dạng các lượt hội thoại gần nhất để làm ngữ cảnh."""
    if not chat_history:
        return ""
    relevant = chat_history[-(max_turns * 2) :]
    lines = []
    for msg in relevant:
        role = "Người dùng" if msg.get("role") == "user" else "Trợ lý"
        content = msg.get("content", "").strip()
        if content:
            lines.append(f"{role}: {content}")
    return "\n".join(lines)


def rewrite_query(
    query: str,
    chat_history: list[dict] | None = None,
    client: Any = None,
    model: str = "",
    enable_llm: bool = True,
    timeout_seconds: int = 8,
) -> tuple[str, bool]:
    """Chuẩn hóa và viết lại câu hỏi để tối ưu hóa truy xuất.

    Returns:
        tuple[str, bool]: (câu truy vấn đã chuẩn hóa, có thay đổi so với câu gốc không)
    """
    clean_query = clean_conversational_fillers(query)
    if not clean_query:
        return query, False

    history_text = _format_history_context(chat_history or [])
    has_history = bool(history_text)

    # Nếu không dùng LLM hoặc không có client/model, fallback về rule-based
    if not enable_llm or client is None or not model:
        is_changed = clean_query.strip().lower() != query.strip().lower()
        return clean_query, is_changed

    # Xây dựng prompt cho LLM
    if has_history:
        user_content = (
            f"Lịch sử hội thoại gần nhất:\n{history_text}\n\n"
            f"Câu hỏi mới của người dùng:\n{query}\n\n"
            "Hãy viết lại câu hỏi trên thành một câu truy vấn tìm kiếm độc lập và đầy đủ ý nghĩa:"
        )
    else:
        user_content = (
            f"Câu hỏi của người dùng:\n{query}\n\n"
            "Hãy chuẩn hóa câu hỏi trên thành một câu truy vấn tìm kiếm tài liệu học tập rõ ràng, loại bỏ từ đệm:"
        )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": REWRITE_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.0,
            timeout=timeout_seconds,
        )
        raw_text = response.choices[0].message.content or ""
        rewritten = raw_text.strip().strip('"').strip("'").strip()
        rewritten = re.sub(r"^(?:Truy vấn|Câu hỏi chuẩn hóa|Query|Rewritten query)\s*[:\-]\s*", "", rewritten, flags=re.IGNORECASE).strip()
        if not rewritten:
            return clean_query, clean_query.strip().lower() != query.strip().lower()
        is_changed = rewritten.strip().lower() != query.strip().lower()
        return rewritten, is_changed
    except Exception:
        # Fallback an toàn nếu LLM lỗi mạng / timeout
        is_changed = clean_query.strip().lower() != query.strip().lower()
        return clean_query, is_changed
