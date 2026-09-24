import hashlib
import re
from itertools import groupby

from src.models import Chunk

DEFAULT_CHUNKING = {
    "strategy": "structure",
    "max_words": 350,
    "overlap_words": 50,
    "chunk_words": 450,
    "fixed_overlap_words": 75,
    "prefix": True,
}
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+")


def split_sentences(text: str) -> list[str]:
    return [sentence.strip() for sentence in _SENTENCE_END.split(text) if sentence.strip()]


def _word_count(sentences: list[tuple[int, list[str]]]) -> int:
    return sum(len(words) for _, words in sentences)


def _overlap_tail(sentences: list[tuple[int, list[str]]], limit: int) -> list[tuple[int, list[str]]]:
    tail: list[tuple[int, list[str]]] = []
    for sentence in reversed(sentences):
        if _word_count(tail) + len(sentence[1]) > limit:
            break
        tail.insert(0, sentence)
    return tail


def _structure_pieces(blocks, max_words: int, overlap_words: int):
    if max_words <= 0 or not 0 <= overlap_words < max_words:
        raise ValueError("max_words must be positive and overlap_words smaller than max_words")
    pieces = []
    for path, group in groupby((block for block in blocks if block.text), key=lambda block: block.heading_path):
        sentences: list[tuple[int, list[str]]] = []
        for block in group:
            for sentence in split_sentences(block.text):
                words = sentence.split()
                for start in range(0, len(words), max_words):
                    sentences.append((block.page, words[start : start + max_words]))
        current: list[tuple[int, list[str]]] = []
        for sentence in sentences:
            if current and _word_count(current) + len(sentence[1]) > max_words:
                pieces.append((current[0][0], path, " ".join(" ".join(words) for _, words in current)))
                current = _overlap_tail(current, overlap_words)
                if _word_count(current) + len(sentence[1]) > max_words:
                    current = []
            current.append(sentence)
        if current:
            pieces.append((current[0][0], path, " ".join(" ".join(words) for _, words in current)))
    return pieces


def _fixed_pieces(blocks, chunk_words: int, overlap_words: int):
    if chunk_words <= 0 or not 0 <= overlap_words < chunk_words:
        raise ValueError("chunk_words must be positive and overlap smaller than chunk_words")
    step = chunk_words - overlap_words
    pieces = []
    for page, group in groupby((block for block in blocks if block.text), key=lambda block: block.page):
        page_blocks = list(group)
        words = " ".join(block.text for block in page_blocks).split()
        for start in range(0, len(words), step):
            pieces.append((page, page_blocks[0].heading_path, " ".join(words[start : start + chunk_words])))
            if start + chunk_words >= len(words):
                break
    return pieces


def chunk_blocks(
    blocks,
    *,
    doc_id: str,
    doc_title: str,
    course: str,
    source_type: str,
    strategy: str = "structure",
    max_words: int = 350,
    overlap_words: int = 50,
    chunk_words: int = 450,
    fixed_overlap_words: int = 75,
    prefix: bool = True,
) -> list[Chunk]:
    if strategy == "structure":
        pieces = _structure_pieces(blocks, max_words, overlap_words)
    elif strategy == "fixed":
        pieces = _fixed_pieces(blocks, chunk_words, fixed_overlap_words)
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")
    chunks = []
    for index, (page, path, body) in enumerate(pieces):
        header = " > ".join(part for part in (doc_title, *path) if part)
        text = f"{header}\n{body}" if prefix and header else body
        digest = hashlib.sha256(
            f"{doc_id}|{strategy}|{page}|{' > '.join(path)}|{index}|{body}".encode("utf-8")
        ).hexdigest()[:24]
        chunks.append(Chunk(digest, doc_id, doc_title, course, source_type, page, tuple(path), body, text))
    return chunks
