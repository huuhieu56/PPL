import unicodedata

import pytest

from src.bench.checks import check_exact, check_paraphrase, contains_quote, content_tokens, lexical_overlap

SOURCE = "Học phần AI101 giới thiệu học máy. Khóa chính xác định duy nhất một bản ghi trong bảng."


def test_content_tokens_drop_stopwords():
    assert content_tokens("Khóa chính là gì và có tác dụng gì?") == {"khóa", "chính", "tác", "dụng"}


def test_lexical_overlap_is_jaccard():
    assert lexical_overlap("khóa chính", "khóa chính") == 1.0
    assert lexical_overlap("", SOURCE) == 0.0
    assert lexical_overlap("khóa ngoại", "khóa chính") == pytest.approx(1 / 3)


def test_check_exact_requires_identifier_present_in_source():
    assert check_exact("Học phần AI101 dạy gì?", SOURCE) is None
    assert check_exact("Mã CS999 là môn nào?", SOURCE).startswith("exact:")
    assert check_exact("Khóa chính là gì?", SOURCE).startswith("exact:")


def test_check_paraphrase_rejects_copied_wording():
    assert check_paraphrase("Khóa chính xác định duy nhất một bản ghi trong bảng?", SOURCE).startswith("paraphrase:")
    assert check_paraphrase("Thuộc tính nào giúp phân biệt từng dòng dữ liệu?", SOURCE) is None


def test_contains_quote_ignores_case_spacing_and_unicode_form():
    assert contains_quote("khóa  chính xác định", SOURCE)
    assert contains_quote(unicodedata.normalize("NFD", "Khóa chính"), SOURCE)
    assert not contains_quote("khóa ngoại", SOURCE)
    assert not contains_quote("   ", SOURCE)
