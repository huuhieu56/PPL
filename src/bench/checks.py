import re

from src.text import normalize_text, tokenize

STOPWORDS = frozenset(
    """
    là của và các có được cho trong một những với này đó thì không khi để từ theo như về ra vào
    bị đã sẽ đang rất cũng nào gì sao hay hoặc nếu mà nhưng do tại bởi ở trên dưới gồm hãy
    nêu cho biết thế nào bao nhiêu vì
    """.split()
)
_WORD = re.compile(r"\w+", re.UNICODE)


def content_tokens(text: str) -> set[str]:
    return {token for token in tokenize(text) if token not in STOPWORDS}


def lexical_overlap(question: str, source: str) -> float:
    left, right = content_tokens(question), content_tokens(source)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def check_exact(question: str, source: str) -> str | None:
    candidates = {
        word
        for word in _WORD.findall(normalize_text(question))
        if any(character.isdigit() for character in word) or (len(word) >= 2 and word.isupper())
    }
    source_tokens = set(tokenize(source))
    if not any(word.lower() in source_tokens for word in candidates):
        return "exact: câu hỏi không chứa mã/ký hiệu xuất hiện trong chunk"
    return None


def check_paraphrase(question: str, source: str, max_overlap: float = 0.2) -> str | None:
    overlap = lexical_overlap(question, source)
    if overlap > max_overlap:
        return f"paraphrase: trùng từ vựng {overlap:.2f} > {max_overlap}"
    return None


def contains_quote(quote: str, text: str) -> bool:
    needle = normalize_text(quote).lower()
    return bool(needle) and needle in normalize_text(text).lower()
