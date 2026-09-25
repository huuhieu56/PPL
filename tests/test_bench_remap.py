from src.bench.remap import remap_qrels
from tests.fakes import make_chunk

QUOTE = "một hai ba bốn năm sáu bảy tám chín mười"


def test_full_quote_inside_chunk_gets_relevance():
    chunks = [make_chunk("a", f"mở đầu {QUOTE} kết thúc"), make_chunk("b", "không liên quan")]
    evidence = [{"query_id": "q1", "doc_id": "doc-1", "page": 1, "quote": QUOTE, "relevance": 2}]
    assert remap_qrels(evidence, chunks) == [{"query_id": "q1", "chunk_id": "a", "relevance": 2}]


def test_quote_split_across_chunks_needs_sixty_percent():
    chunks = [
        make_chunk("a", "phần trước một hai ba bốn năm sáu"),
        make_chunk("b", "bảy tám chín mười phần sau"),
    ]
    evidence = [{"query_id": "q1", "doc_id": "doc-1", "page": 1, "quote": QUOTE, "relevance": 1}]
    assert remap_qrels(evidence, chunks) == [{"query_id": "q1", "chunk_id": "a", "relevance": 1}]


def test_other_documents_and_max_grade():
    chunks = [make_chunk("a", QUOTE), make_chunk("x", QUOTE, doc_id="doc-2")]
    evidence = [
        {"query_id": "q1", "doc_id": "doc-1", "page": 1, "quote": QUOTE, "relevance": 1},
        {"query_id": "q1", "doc_id": "doc-1", "page": 1, "quote": "hai ba bốn", "relevance": 2},
    ]
    assert remap_qrels(evidence, chunks) == [{"query_id": "q1", "chunk_id": "a", "relevance": 2}]
