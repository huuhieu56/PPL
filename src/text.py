import os
import re
import unicodedata

TOKENIZERS = ("whitespace", "pyvi", "vncorenlp")

_INVISIBLE = re.compile("[​‌‍⁠﻿­]")
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_HYPHEN_BREAK = re.compile(r"(\w)-\r?\n(\w)")
_WORD = re.compile(r"\w+", re.UNICODE)
_EDGE_PUNCTUATION = re.compile(r"^[^\w]+|[^\w]+$", re.UNICODE)
_vncorenlp_model = None


def normalize_text(text: str) -> str:
    value = unicodedata.normalize("NFC", text or "")
    value = _INVISIBLE.sub("", value)
    value = _CONTROL.sub(" ", value)
    value = _HYPHEN_BREAK.sub(r"\1\2", value)
    return re.sub(r"\s+", " ", value).strip()


def _vncorenlp():
    global _vncorenlp_model
    if _vncorenlp_model is None:
        import py_vncorenlp

        save_dir = os.path.abspath(os.getenv("PPL_VNCORENLP_DIR", "vncorenlp"))
        os.makedirs(save_dir, exist_ok=True)
        working_dir = os.getcwd()
        try:
            if not os.path.exists(os.path.join(save_dir, "VnCoreNLP-1.2.jar")):
                py_vncorenlp.download_model(save_dir=save_dir)
            _vncorenlp_model = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=save_dir)
        finally:
            os.chdir(working_dir)
    return _vncorenlp_model


def tokenize(text: str, mode: str = "whitespace") -> list[str]:
    if mode not in TOKENIZERS:
        raise ValueError(f"Unsupported tokenizer: {mode}")
    normalized = normalize_text(text).lower()
    if not normalized:
        return []
    if mode == "whitespace":
        return _WORD.findall(normalized)
    if mode == "pyvi":
        from pyvi import ViTokenizer

        segmented = ViTokenizer.tokenize(normalized)
    else:
        segmented = " ".join(_vncorenlp().word_segment(normalized))
    tokens = (_EDGE_PUNCTUATION.sub("", token) for token in segmented.split())
    return [token for token in tokens if token]
