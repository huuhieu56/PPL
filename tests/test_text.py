import unicodedata

import pytest

from src.text import TOKENIZERS, normalize_text, tokenize


def test_normalize_text_produces_nfc_and_strips_invisible_characters():
    decomposed = unicodedata.normalize("NFD", "Học máy")
    raw = f"  {decomposed}​ là­  gì?\x07\n"
    assert normalize_text(raw) == "Học máy là gì?"
    assert unicodedata.is_normalized("NFC", normalize_text(raw))


def test_normalize_text_joins_hyphenated_line_breaks():
    assert normalize_text("infor-\nmation retrieval") == "information retrieval"
    assert normalize_text("CSDL - quan hệ") == "CSDL - quan hệ"


def test_whitespace_tokenizer_lowercases_and_drops_punctuation():
    assert tokenize("Mã môn AI101 là gì?") == ["mã", "môn", "ai101", "là", "gì"]
    assert tokenize(unicodedata.normalize("NFD", "Học")) == ["học"]
    assert tokenize("   ") == []


def test_pyvi_tokenizer_joins_compound_words():
    tokens = tokenize("Học máy là một lĩnh vực.", "pyvi")
    assert "lĩnh_vực" in tokens
    assert all(token.strip(".,") == token for token in tokens)


def test_unknown_tokenizer_is_rejected():
    assert TOKENIZERS == ("whitespace", "pyvi", "vncorenlp")
    with pytest.raises(ValueError, match="Unsupported tokenizer"):
        tokenize("abc", "spacy")
