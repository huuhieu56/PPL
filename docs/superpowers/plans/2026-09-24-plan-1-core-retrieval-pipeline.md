# Kế hoạch 1/3: Pipeline truy xuất lõi — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Thay pipeline truy xuất hiện tại bằng pipeline theo Chương 2 của báo cáo: chuẩn hóa NFC, structure-aware chunking có breadcrumb, index nhiều tokenizer (numpy/FAISS), fusion tách tầng với log điểm phân rã (Bảng 2.1), reranker có cache — và nối app Streamlit vào pipeline mới.

**Architecture:** Các module nhỏ, mỗi module một trách nhiệm: `text` (chuẩn hóa/tách từ) → `ingestion` (trích xuất Block có heading_path) → `chunking` (structure/fixed) → `index` (BM25 + dense lưu đĩa) → `fusion` (hàm thuần) → `pipeline` (điều phối tầng, đo thời gian, cache) → `rag` (LLM). App và (ở Kế hoạch 3) bộ đánh giá dùng chung `RetrievalPipeline`.

**Tech Stack:** Python 3.11, rank-bm25, sentence-transformers (bge-m3, bge-reranker-v2-m3), numpy, faiss-cpu (tùy chọn), pyvi, py_vncorenlp (tùy chọn), PyMuPDF, python-docx, python-pptx, Streamlit, SQLite, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-hybrid-retrieval-evaluation-design.md` (mục 3, 4, 5, 10, 11). Kế hoạch 2 (benchmark) và 3 (đánh giá, báo cáo, notebook) nối tiếp kế hoạch này.

## Global Constraints

- Python 3.11; venv ở `.venv`. Trên Windows dùng `.venv/Scripts/python`, trên Linux/Colab dùng `.venv/bin/python` (hoặc `python`).
- Chạy mọi lệnh từ thư mục gốc repo. Chạy test bằng `.venv/Scripts/python -m pytest -q`.
- Test không được tải model, không gọi mạng: dùng `tests/fakes.py` (encoder/cross-encoder giả).
- BM25 cố định `k1=1.5, b=0.75`.
- Rerank N mặc định **30**; top-L mặc định 100; context_k mặc định 5.
- Tokenizer hợp lệ: `whitespace`, `pyvi`, `vncorenlp`.
- Fusion hợp lệ: `none`, `rrf`, `weighted`, `adaptive`.
- Chunking mặc định: `structure`, `max_words=350`, `overlap_words=50`, `prefix=True`; đối chứng `fixed`: `chunk_words=450`, `fixed_overlap_words=75`.
- Chuỗi giao diện, prompt, câu từ chối: tiếng Việt. Code, tên hàm: tiếng Anh, theo phong cách hiện có (dataclass frozen, type hint, không comment thừa).
- `rank_bm25` cho idf ≤ 0 khi một từ xuất hiện ở ≥ nửa số tài liệu: fixture test phải có ≥ 4 chunk và từ truy vấn chỉ xuất hiện ở 1 chunk.
- Console Windows mặc định cp1252: khi chạy lệnh in tiếng Việt, đặt `PYTHONIOENCODING=utf-8` (Kế hoạch 2 cho CLI tự `reconfigure` stdout sang UTF-8).
- Commit sau mỗi task, message kết thúc bằng dòng `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.

## Review Focus

1. **Query có dấu dạng NFD hoặc có ký tự zero-width** (gõ từ macOS/copy từ PDF) — phải cho cùng kết quả BM25 như query NFC. Pin: `test_sparse_search_matches_nfd_query` (Task 6).
2. **Index chưa build tokenizer mà config yêu cầu** (ví dụ cấu hình lưu `pyvi` nhưng index chỉ có `whitespace`) — phải báo lỗi rõ ràng nêu lệnh khắc phục, không crash khó hiểu. Pin: `test_missing_tokenizer_error_names_fix` (Task 6); trang Chat bắt `ValueError` và hiển thị (Task 9).
3. **Query không có từ nào khớp BM25 / query rỗng sau chuẩn hóa** — pipeline hybrid vẫn trả kết quả từ nhánh dense, không chia cho 0. Pin: `test_hybrid_with_no_sparse_hits_falls_back_to_dense` (Task 7).
4. **PDF có header/footer lặp lại mỗi trang** (tên giáo trình cỡ chữ lớn) — không được coi là tiêu đề và reset breadcrumb. Pin: `test_pdf_repeated_header_is_ignored` (Task 5).
5. **Cấu hình RAG cũ trong `data/app.db`** (schema `method/use_reranker`) — app không crash, cấu hình cũ bị bỏ qua kèm cảnh báo. Pin: `test_legacy_rag_config_is_reported_invalid` (Task 8).

---

## File Structure

| File | Trạng thái | Trách nhiệm |
|---|---|---|
| `pytest.ini` | tạo | `pythonpath = .`, lọc warning |
| `src/text.py` | tạo | `normalize_text`, `tokenize`, `TOKENIZERS` |
| `src/models.py` | sửa | `Block`, `Chunk` (mới), `StageScores`, `RetrievedChunk`, `PipelineConfig`; xóa `SearchResult`, `RagConfig` |
| `src/fusion.py` | tạo | `minmax_scores`, `rank_ids`, `fuse_rrf`, `fuse_weighted`, `adaptive_alpha` |
| `src/cache.py` | tạo | `RetrievalCache` (SQLite) |
| `src/reranking.py` | viết lại | `load_cross_encoder`, `Reranker` |
| `src/chunking.py` | tạo | `DEFAULT_CHUNKING`, `split_sentences`, `chunk_blocks` |
| `src/ingestion.py` | viết lại | `extract_blocks`, `heading_level`, `build_corpus` |
| `src/index.py` | tạo | `RetrievalIndex`, `load_encoder` |
| `src/pipeline.py` | tạo | `RetrievalPipeline`, `PipelineResult` |
| `src/rag.py` | sửa | dùng `RetrievalPipeline` + `PipelineConfig` |
| `src/config.py` | sửa | `PPL_DATA_DIR`, `PPL_RUNS_DIR`, `indexes_dir`, `cache_path` |
| `src/storage.py` | sửa | `load_pipeline_configs`, `INSERT OR IGNORE` corpus |
| `src/retrieval.py` | xóa | thay bằng `index.py` + `fusion.py` + `pipeline.py` |
| `src/experiments.py`, `run_rag_evaluation.py` | sửa tạm | map cấu hình cũ sang `PipelineConfig` (Kế hoạch 3 thay thế hẳn) |
| `pages/*.py`, `configs/default.yaml` | sửa | dùng pipeline mới |
| `tests/fakes.py` | tạo | `FakeEncoder`, `FakeCrossEncoder`, `make_chunk` |
| `tests/test_text.py`, `test_models.py`, `test_fusion.py`, `test_cache_rerank.py`, `test_ingestion.py`, `test_index.py`, `test_pipeline.py`, `test_rag.py`, `test_storage.py`, `test_pages.py` | tạo | test hành vi theo module |
| `tests/test_core.py` | sửa dần | chỉ còn test metrics + resume (Kế hoạch 3 chuyển tiếp) |

---

### Task 1: Môi trường, pytest.ini và module `text`

**Files:**
- Create: `pytest.ini`, `src/text.py`, `tests/test_text.py`
- Modify: `requirements.in`, `requirements.txt`, `.gitignore`

**Interfaces:**
- Produces: `normalize_text(text: str) -> str`; `tokenize(text: str, mode: str = "whitespace") -> list[str]`; `TOKENIZERS: tuple[str, ...] = ("whitespace", "pyvi", "vncorenlp")`.

- [ ] **Step 1: Tạo venv (nếu chưa có) và thêm phụ thuộc**

Thêm vào cuối `requirements.in`:
```
faiss-cpu
py_vncorenlp
matplotlib
ranx
```
Chạy:
```bash
uv venv --python 3.11 .venv
uv pip compile requirements.in -o requirements.txt --torch-backend cpu
uv pip install --python .venv/Scripts/python.exe -r requirements.txt --torch-backend cpu
```
Expected: cài đặt thành công; `requirements.txt` có dòng `faiss-cpu==`, `py-vncorenlp==`, `matplotlib==`, `ranx==`.

Thêm vào `.gitignore` (dưới mục "Local model caches and databases"):
```
vncorenlp/
```

- [ ] **Step 2: Tạo `pytest.ini`**

```ini
[pytest]
pythonpath = .
testpaths = tests
filterwarnings =
    ignore::DeprecationWarning
    ignore::FutureWarning
```

- [ ] **Step 3: Viết test thất bại `tests/test_text.py`**

```python
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
```

- [ ] **Step 4: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_text.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.text'`

- [ ] **Step 5: Viết `src/text.py`**

```python
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
```

(`py_vncorenlp.VnCoreNLP` đổi thư mục làm việc khi khởi tạo — khối `finally` khôi phục lại.)

- [ ] **Step 6: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_text.py -q`
Expected: `5 passed`

- [ ] **Step 7: Chạy toàn bộ test cũ vẫn pass**

Run: `.venv/Scripts/python -m pytest -q`
Expected: `9 passed`

- [ ] **Step 8: Commit**

```bash
git add pytest.ini src/text.py tests/test_text.py requirements.in requirements.txt .gitignore
git commit -m "feat: add Vietnamese text normalization and tokenizers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Kiểu dữ liệu `PipelineConfig`, `StageScores`, `RetrievedChunk`, `Block`

**Files:**
- Modify: `src/models.py`
- Create: `tests/test_models.py`

**Interfaces:**
- Consumes: `TOKENIZERS` (Task 1).
- Produces (thêm, chưa xóa kiểu cũ):
  - `FUSIONS = ("none", "rrf", "weighted", "adaptive")`
  - `Block(page: int, heading_path: tuple[str, ...], text: str, warning: str = "")`
  - `StageScores(rank: int, sparse_score: float|None=None, sparse_rank: int|None=None, dense_score: float|None=None, dense_rank: int|None=None, fusion_score: float|None=None, rerank_score: float|None=None)` với `to_dict() -> dict`
  - `RetrievedChunk(chunk, scores: StageScores)` với property `score -> float` (ưu tiên rerank → fusion → sparse → dense)
  - `PipelineConfig(...)` (trường xem code) với `to_dict()`, `from_dict(data) -> PipelineConfig` (từ chối khóa lạ bằng `ValueError`), `replace(**changes) -> PipelineConfig`

- [ ] **Step 1: Viết test thất bại `tests/test_models.py`**

```python
import pytest

from src.models import PipelineConfig, RetrievedChunk, StageScores


def test_pipeline_config_defaults_follow_report():
    config = PipelineConfig()
    assert (config.sparse, config.dense, config.fusion, config.rerank) == (True, True, "weighted", True)
    assert (config.top_l, config.rerank_n, config.context_k) == (100, 30, 5)
    assert PipelineConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"fusion": "sum"}, "Unsupported fusion"),
        ({"fusion": "none"}, "exactly one"),
        ({"dense": False}, "requires both"),
        ({"alpha": 1.5}, "alpha"),
        ({"rerank_n": 200}, "context_k <= rerank_n <= top_l"),
        ({"rerank": False, "context_k": 101}, "context_k <= top_l"),
        ({"tokenizer": "spacy"}, "Unsupported tokenizer"),
    ],
)
def test_pipeline_config_rejects_inconsistent_values(changes, message):
    with pytest.raises(ValueError, match=message):
        PipelineConfig(**changes)


def test_pipeline_config_from_dict_rejects_legacy_keys():
    with pytest.raises(ValueError, match="Unknown config keys"):
        PipelineConfig.from_dict({"method": "rrf", "use_reranker": False})


def test_sparse_only_config_is_valid():
    config = PipelineConfig(dense=False, fusion="none", rerank=False)
    assert config.replace(tokenizer="pyvi").tokenizer == "pyvi"


def test_retrieved_chunk_score_prefers_latest_stage():
    assert RetrievedChunk(None, StageScores(rank=1, sparse_score=3.0)).score == 3.0
    assert RetrievedChunk(None, StageScores(rank=1, dense_score=0.4)).score == 0.4
    fused = StageScores(rank=1, sparse_score=3.0, dense_score=0.4, fusion_score=0.7)
    assert RetrievedChunk(None, fused).score == 0.7
    reranked = StageScores(rank=1, fusion_score=0.7, rerank_score=-1.2)
    assert RetrievedChunk(None, reranked).score == -1.2
    assert StageScores(rank=2).to_dict()["rerank_score"] is None
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_models.py -q`
Expected: FAIL với `ImportError: cannot import name 'PipelineConfig'`

- [ ] **Step 3: Thêm kiểu mới vào `src/models.py`**

Sửa dòng import đầu file thành:
```python
import dataclasses
from dataclasses import asdict, dataclass, fields
from typing import Any

from src.text import TOKENIZERS

FUSIONS = ("none", "rrf", "weighted", "adaptive")
```
Giữ nguyên `Chunk`, `SearchResult`, `RagConfig` hiện có. Thêm vào cuối file:
```python
@dataclass(frozen=True)
class Block:
    page: int
    heading_path: tuple[str, ...]
    text: str
    warning: str = ""


@dataclass(frozen=True)
class StageScores:
    rank: int
    sparse_score: float | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Any
    scores: StageScores

    @property
    def score(self) -> float:
        stages = self.scores
        for value in (stages.rerank_score, stages.fusion_score, stages.sparse_score, stages.dense_score):
            if value is not None:
                return value
        return float("-inf")


@dataclass(frozen=True)
class PipelineConfig:
    sparse: bool = True
    dense: bool = True
    fusion: str = "weighted"
    alpha: float = 0.5
    rrf_k: int = 60
    adaptive_beta: float = 0.3
    rerank: bool = True
    rerank_n: int = 30
    top_l: int = 100
    context_k: int = 5
    tokenizer: str = "whitespace"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    refusal_threshold: float = 0.0
    llm_model: str = ""
    temperature: float = 0.0
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        if self.fusion not in FUSIONS:
            raise ValueError(f"Unsupported fusion: {self.fusion}")
        if self.fusion == "none" and self.sparse == self.dense:
            raise ValueError("fusion 'none' requires exactly one of sparse or dense")
        if self.fusion != "none" and not (self.sparse and self.dense):
            raise ValueError(f"fusion '{self.fusion}' requires both sparse and dense")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if min(self.top_l, self.rerank_n, self.context_k, self.rrf_k) < 1:
            raise ValueError("top_l, rerank_n, context_k and rrf_k must be positive")
        if self.rerank and not self.context_k <= self.rerank_n <= self.top_l:
            raise ValueError("Require context_k <= rerank_n <= top_l")
        if not self.rerank and self.context_k > self.top_l:
            raise ValueError("Require context_k <= top_l")
        if self.tokenizer not in TOKENIZERS:
            raise ValueError(f"Unsupported tokenizer: {self.tokenizer}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def replace(self, **changes) -> "PipelineConfig":
        return dataclasses.replace(self, **changes)
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_models.py -q`
Expected: `11 passed`

- [ ] **Step 5: Chạy toàn bộ test**

Run: `.venv/Scripts/python -m pytest -q`
Expected: tất cả pass.

- [ ] **Step 6: Commit**

```bash
git add src/models.py tests/test_models.py
git commit -m "feat: add pipeline config and stage score models

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Module `fusion` (hàm thuần)

**Files:**
- Create: `src/fusion.py`, `tests/test_fusion.py`

**Interfaces:**
- Produces:
  - `minmax_scores(scores: dict[str, float]) -> dict[str, float]`
  - `rank_ids(scores: dict[str, float]) -> list[str]` (giảm dần, hòa → `chunk_id` tăng dần)
  - `fuse_rrf(rankings: list[list[str]], k: int) -> dict[str, float]`
  - `fuse_weighted(sparse: dict[str, float], dense: dict[str, float], alpha: float) -> dict[str, float]` (hợp hai tập; vắng mặt → 0 sau chuẩn hóa)
  - `adaptive_alpha(query: str, query_tokens: list[str], alpha0: float, beta: float, idf: dict[str, float]) -> tuple[float, dict[str, float]]`

**Ghi chú thiết kế (sửa lỗi so với code cũ, đã phản ánh vào spec mục 5.1):** code cũ cộng `max_idf` thô (thường > 1) nên `lexical_score` gần như luôn bão hòa = 1 → α luôn = α0 + β. Bản mới chuẩn hóa `max_idf` theo idf lớn nhất của corpus, và dùng công thức đối xứng `α = clamp(α0 + β·(2·lexical − 1), 0.1, 0.9)`: truy vấn thiên từ vựng tăng α, truy vấn ngữ nghĩa giảm α.

- [ ] **Step 1: Viết test thất bại `tests/test_fusion.py`**

```python
import pytest

from src.fusion import adaptive_alpha, fuse_rrf, fuse_weighted, minmax_scores, rank_ids


def test_minmax_handles_constant_and_empty_scores():
    assert minmax_scores({}) == {}
    assert minmax_scores({"a": 5.0, "b": 5.0}) == {"a": 1.0, "b": 1.0}
    assert minmax_scores({"a": 2.0, "b": 4.0, "c": 3.0}) == {"a": 0.0, "b": 1.0, "c": 0.5}


def test_rank_ids_breaks_ties_by_chunk_id():
    assert rank_ids({"b": 1.0, "a": 1.0, "c": 2.0}) == ["c", "a", "b"]


def test_rrf_sums_reciprocal_ranks_across_lists():
    fused = fuse_rrf([["a", "b"], ["b", "c"]], k=60)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["a"] == pytest.approx(1 / 61)
    assert rank_ids(fused) == ["b", "a", "c"]


def test_weighted_fusion_uses_union_and_zero_for_missing_branch():
    fused = fuse_weighted({"a": 8.0, "b": 1.0}, {"b": 0.9, "c": 0.5}, alpha=0.7)
    assert fused["a"] == pytest.approx(0.7 * 1.0 + 0.3 * 0.0)
    assert fused["b"] == pytest.approx(0.7 * 0.0 + 0.3 * 1.0)
    assert fused["c"] == pytest.approx(0.0)
    assert fuse_weighted({"a": 1.0}, {}, alpha=0.5) == {"a": 0.5}


def test_adaptive_alpha_raises_for_codes_and_lowers_for_concepts():
    idf = {"ai101": 2.0, "học": 0.1, "máy": 0.1, "giải": 0.2, "thích": 0.2}
    exact, signals = adaptive_alpha(
        "Mã môn AI101 là gì?", ["mã", "môn", "ai101", "là", "gì"], 0.5, 0.3, idf
    )
    concept, _ = adaptive_alpha(
        "Giải thích học máy", ["giải", "thích", "học", "máy"], 0.5, 0.3, idf
    )
    assert signals["code"] == 1.0
    assert signals["max_idf"] == pytest.approx(1.0)
    assert exact > 0.5 > concept
    assert 0.1 <= concept and exact <= 0.9


def test_adaptive_alpha_with_empty_idf_stays_in_bounds():
    alpha, signals = adaptive_alpha("???", [], 0.5, 0.3, {})
    assert signals["max_idf"] == 0.0
    assert 0.1 <= alpha <= 0.9
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_fusion.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.fusion'`

- [ ] **Step 3: Viết `src/fusion.py`**

```python
import math
import re

_CODE = re.compile(r"\b[A-Z]{2,}[A-Z0-9-]*\d+[A-Z0-9-]*\b")


def minmax_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if math.isclose(low, high):
        return {key: 1.0 for key in scores}
    scale = high - low
    return {key: (value - low) / scale for key, value in scores.items()}


def rank_ids(scores: dict[str, float]) -> list[str]:
    return sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))


def fuse_rrf(rankings: list[list[str]], k: int) -> dict[str, float]:
    fused: dict[str, float] = {}
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking, start=1):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (k + rank)
    return fused


def fuse_weighted(
    sparse: dict[str, float], dense: dict[str, float], alpha: float
) -> dict[str, float]:
    sparse_norm, dense_norm = minmax_scores(sparse), minmax_scores(dense)
    return {
        chunk_id: alpha * sparse_norm.get(chunk_id, 0.0) + (1 - alpha) * dense_norm.get(chunk_id, 0.0)
        for chunk_id in set(sparse_norm) | set(dense_norm)
    }


def adaptive_alpha(
    query: str,
    query_tokens: list[str],
    alpha0: float,
    beta: float,
    idf: dict[str, float],
) -> tuple[float, dict[str, float]]:
    compact = query.strip()
    length = max(len(compact), 1)
    corpus_max = max(idf.values(), default=0.0)
    query_max = max((idf.get(token, 0.0) for token in query_tokens), default=0.0)
    signals = {
        "digit": sum(character.isdigit() for character in compact) / length,
        "symbol": sum(not character.isalnum() and not character.isspace() for character in compact)
        / length,
        "code": float(bool(_CODE.search(compact))),
        "max_idf": query_max / corpus_max if corpus_max > 0 else 0.0,
    }
    lexical = min(1.0, sum(signals.values()))
    signals["lexical_score"] = lexical
    return min(0.9, max(0.1, alpha0 + beta * (2 * lexical - 1))), signals
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_fusion.py -q`
Expected: `6 passed`

- [ ] **Step 5: Cập nhật spec mục 5.1**

Trong `docs/superpowers/specs/2026-09-24-hybrid-retrieval-evaluation-design.md`, thay dòng:
```
`adaptive_alpha` giữ công thức hiện tại, β lấy từ config, clamp [0.1, 0.9].
```
bằng:
```
`adaptive_alpha`: `max_idf` được chuẩn hóa theo idf lớn nhất của corpus (code cũ dùng idf thô nên luôn bão hòa); α = clamp(α0 + β·(2·lexical − 1), 0.1, 0.9) — truy vấn từ vựng tăng α, truy vấn ngữ nghĩa giảm α; β lấy từ config.
```

- [ ] **Step 6: Commit**

```bash
git add src/fusion.py tests/test_fusion.py docs/superpowers/specs/2026-09-24-hybrid-retrieval-evaluation-design.md
git commit -m "feat: add pure fusion functions with normalized adaptive alpha

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Cache SQLite và `Reranker` có cache

**Files:**
- Create: `src/cache.py`, `tests/fakes.py`, `tests/test_cache_rerank.py`
- Modify: `src/reranking.py` (thêm, giữ hàm `rerank` cũ tới Task 8)

**Interfaces:**
- Produces:
  - `query_key(text: str) -> str` (sha256 hex)
  - `RetrievalCache(path: Path | str)` với `first_stage_key(index_version, branch, variant, query, top_l) -> str` (staticmethod), `get_first_stage(key) -> list[tuple[str, float]] | None`, `put_first_stage(key, results)`, `get_rerank(model, query, chunk_ids) -> dict[str, float]`, `put_rerank(model, query, scores: dict[str, float])`
  - `load_cross_encoder(model_name: str)` (cache cấp module)
  - `Reranker(model_name: str, model=None, cache: RetrievalCache | None = None)` với `score(query: str, chunks: list, use_cache: bool = True) -> list[float]`
  - `tests/fakes.py`: `FakeEncoder`, `FakeCrossEncoder`, `make_chunk(chunk_id, text, **fields)` — dùng lại ở mọi task sau.

- [ ] **Step 1: Tạo `tests/fakes.py`**

`make_chunk` trả `SimpleNamespace` ở task này; Task 5 đổi sang `Chunk` thật khi `Chunk` có trường mới.
```python
import hashlib
import re
from types import SimpleNamespace

import numpy as np


class FakeEncoder:
    dim = 64

    def __init__(self):
        self.calls = 0

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32):
        self.calls += 1
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                column = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
                vectors[row, column] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class FakeCrossEncoder:
    def __init__(self):
        self.pairs_seen = 0

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        self.pairs_seen += len(pairs)
        scores = []
        for query, text in pairs:
            query_tokens = set(re.findall(r"\w+", query.lower()))
            text_tokens = set(re.findall(r"\w+", text.lower()))
            scores.append(float(len(query_tokens & text_tokens)))
        return np.asarray(scores, dtype=np.float32)


def make_chunk(chunk_id, text, **fields):
    return SimpleNamespace(chunk_id=chunk_id, text=text, **fields)
```

- [ ] **Step 2: Viết test thất bại `tests/test_cache_rerank.py`**

```python
from src.cache import RetrievalCache
from src.reranking import Reranker
from tests.fakes import FakeCrossEncoder, make_chunk


def test_first_stage_cache_round_trip(tmp_path):
    cache = RetrievalCache(tmp_path / "cache" / "retrieval.sqlite")
    key = RetrievalCache.first_stage_key("v1", "sparse", "whitespace", "câu hỏi", 100)
    assert cache.get_first_stage(key) is None
    cache.put_first_stage(key, [("a", 2.5), ("b", 1.0)])
    reopened = RetrievalCache(tmp_path / "cache" / "retrieval.sqlite")
    assert reopened.get_first_stage(key) == [("a", 2.5), ("b", 1.0)]
    assert key != RetrievalCache.first_stage_key("v1", "sparse", "pyvi", "câu hỏi", 100)


def test_reranker_scores_only_uncached_chunks(tmp_path):
    cache = RetrievalCache(tmp_path / "retrieval.sqlite")
    model = FakeCrossEncoder()
    reranker = Reranker("fake-model", model=model, cache=cache)
    chunks = [make_chunk("a", "học máy là gì"), make_chunk("b", "cơ sở dữ liệu")]

    first = reranker.score("học máy", chunks)
    assert first == [2.0, 0.0]
    assert model.pairs_seen == 2

    again = reranker.score("học máy", [*chunks, make_chunk("c", "máy tính")])
    assert again == [2.0, 0.0, 1.0]
    assert model.pairs_seen == 3


def test_reranker_bypasses_cache_when_requested(tmp_path):
    cache = RetrievalCache(tmp_path / "retrieval.sqlite")
    model = FakeCrossEncoder()
    reranker = Reranker("fake-model", model=model, cache=cache)
    chunks = [make_chunk("a", "học máy")]
    reranker.score("học máy", chunks, use_cache=False)
    reranker.score("học máy", chunks, use_cache=False)
    assert model.pairs_seen == 2
    assert cache.get_rerank("fake-model", "học máy", ["a"]) == {}
    assert Reranker("fake-model", model=model).score("x", []) == []
```

- [ ] **Step 3: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_cache_rerank.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.cache'`

- [ ] **Step 4: Viết `src/cache.py`**

```python
import hashlib
import json
import sqlite3
from pathlib import Path


def query_key(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class RetrievalCache:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(self.path, check_same_thread=False)
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS first_stage (
                key TEXT PRIMARY KEY,
                results_json TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS rerank (
                model TEXT NOT NULL,
                query_key TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                score REAL NOT NULL,
                PRIMARY KEY (model, query_key, chunk_id)
            );
            """
        )
        self._connection.commit()

    @staticmethod
    def first_stage_key(index_version: str, branch: str, variant: str, query: str, top_l: int) -> str:
        return "|".join([index_version, branch, variant, query_key(query), str(top_l)])

    def get_first_stage(self, key: str) -> list[tuple[str, float]] | None:
        row = self._connection.execute(
            "SELECT results_json FROM first_stage WHERE key = ?", (key,)
        ).fetchone()
        if row is None:
            return None
        return [(chunk_id, float(score)) for chunk_id, score in json.loads(row[0])]

    def put_first_stage(self, key: str, results: list[tuple[str, float]]) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO first_stage(key, results_json) VALUES (?, ?)",
            (key, json.dumps([[chunk_id, score] for chunk_id, score in results])),
        )
        self._connection.commit()

    def get_rerank(self, model: str, query: str, chunk_ids: list[str]) -> dict[str, float]:
        if not chunk_ids:
            return {}
        placeholders = ",".join("?" for _ in chunk_ids)
        rows = self._connection.execute(
            f"SELECT chunk_id, score FROM rerank WHERE model = ? AND query_key = ? "
            f"AND chunk_id IN ({placeholders})",
            (model, query_key(query), *chunk_ids),
        ).fetchall()
        return {chunk_id: float(score) for chunk_id, score in rows}

    def put_rerank(self, model: str, query: str, scores: dict[str, float]) -> None:
        key = query_key(query)
        self._connection.executemany(
            "INSERT OR REPLACE INTO rerank(model, query_key, chunk_id, score) VALUES (?, ?, ?, ?)",
            [(model, key, chunk_id, float(score)) for chunk_id, score in scores.items()],
        )
        self._connection.commit()
```

- [ ] **Step 5: Thêm vào `src/reranking.py`** (giữ nguyên hàm `rerank` cũ; thêm sau nó)

```python
_CROSS_ENCODERS: dict = {}


def load_cross_encoder(model_name: str):
    if model_name not in _CROSS_ENCODERS:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODERS[model_name] = CrossEncoder(model_name)
    return _CROSS_ENCODERS[model_name]


class Reranker:
    def __init__(self, model_name: str, model=None, cache=None):
        self.model_name = model_name
        self._model = model
        self.cache = cache

    def score(self, query: str, chunks: list, use_cache: bool = True) -> list[float]:
        if not chunks:
            return []
        caching = self.cache is not None and use_cache
        known = (
            self.cache.get_rerank(self.model_name, query, [chunk.chunk_id for chunk in chunks])
            if caching
            else {}
        )
        missing = [chunk for chunk in chunks if chunk.chunk_id not in known]
        if missing:
            model = self._model or load_cross_encoder(self.model_name)
            values = model.predict(
                [(query, chunk.text) for chunk in missing],
                batch_size=min(16, len(missing)),
                show_progress_bar=False,
            )
            fresh = {chunk.chunk_id: float(value) for chunk, value in zip(missing, values)}
            if caching:
                self.cache.put_rerank(self.model_name, query, fresh)
            known = {**known, **fresh}
        return [known[chunk.chunk_id] for chunk in chunks]
```

- [ ] **Step 6: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_cache_rerank.py -q`
Expected: `3 passed`

- [ ] **Step 7: Commit**

```bash
git add src/cache.py src/reranking.py tests/fakes.py tests/test_cache_rerank.py
git commit -m "feat: add retrieval cache and cached cross-encoder reranker

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: `Chunk` mới, structure-aware chunking và trích xuất có tiêu đề

**Files:**
- Create: `src/chunking.py`, `tests/test_ingestion.py`
- Modify: `src/models.py` (thay `Chunk`), `src/ingestion.py` (viết lại), `src/retrieval.py` (chỉ dòng import `tokenize_vi`), `src/rag.py` (chỉ `_prompt` và `_valid_citations`), `tests/fakes.py` (`make_chunk`), `tests/test_core.py` (bỏ test PPTX cũ, sửa `Chunk(...)` trong test retrieval)

**Interfaces:**
- Consumes: `Block`, `normalize_text`.
- Produces:
  - `Chunk(chunk_id, doc_id, doc_title, course, source_type, page: int, heading_path: tuple[str, ...], body, text)` với `breadcrumb() -> str`, `to_dict() -> dict` (heading_path là list), `Chunk.from_dict(data) -> Chunk`
  - `DEFAULT_CHUNKING: dict` = `{"strategy": "structure", "max_words": 350, "overlap_words": 50, "chunk_words": 450, "fixed_overlap_words": 75, "prefix": True}`
  - `split_sentences(text: str) -> list[str]`
  - `chunk_blocks(blocks, *, doc_id, doc_title, course, source_type, strategy="structure", max_words=350, overlap_words=50, chunk_words=450, fixed_overlap_words=75, prefix=True) -> list[Chunk]`
  - `extract_blocks(path) -> list[Block]`; `heading_level(text, size, bold, median_size, chapter_offset) -> int | None`
  - `build_corpus(files, metadata, settings, database, chunking: dict) -> CorpusBuildResult(version_id, chunks_path, chunk_count, manifest_hash, reused: bool)`; `metadata[str(path)]` có `course`, `source_type`, tùy chọn `doc_title`, `semester`.

- [ ] **Step 1: Viết test thất bại `tests/test_ingestion.py`**

```python
import json

import pymupdf
from docx import Document
from pptx import Presentation
from pptx.util import Inches

from src.chunking import chunk_blocks, split_sentences
from src.config import load_settings
from src.ingestion import build_corpus, extract_blocks, heading_level
from src.models import Block, Chunk
from src.storage import Database


def _chunk(blocks, **options):
    return chunk_blocks(
        blocks, doc_id="d1", doc_title="Giáo trình CSDL", course="CS101", source_type="textbook", **options
    )


def test_heading_level_rules():
    assert heading_level("Chương 2: Mô hình quan hệ", 11, False, 11, 0) == 1
    assert heading_level("2.3 Chuẩn hóa", 11, False, 11, 1) == 3
    assert heading_level("1. Giới thiệu", 11, False, 11, 0) is None
    assert heading_level("1. Giới thiệu", 11, True, 11, 0) == 1
    assert heading_level("Tổng quan", 20, False, 11, 0) == 1
    assert heading_level("42", 20, False, 11, 0) is None
    assert heading_level("x" * 130, 20, True, 11, 0) is None


def test_docx_headings_become_breadcrumbs(tmp_path):
    path = tmp_path / "giao_trinh.docx"
    document = Document()
    document.add_heading("Chương 1. Tổng quan", level=1)
    document.add_paragraph("Cơ sở dữ liệu là tập hợp dữ liệu có tổ chức.")
    document.add_heading("1.1 Khái niệm", level=2)
    document.add_paragraph("Khóa chính xác định duy nhất một bản ghi.")
    table = document.add_table(rows=1, cols=2)
    table.rows[0].cells[0].text = "PK"
    table.rows[0].cells[1].text = "Khóa chính"
    document.save(path)

    blocks = extract_blocks(path)
    assert blocks[0] == Block(1, ("Chương 1. Tổng quan",), "Cơ sở dữ liệu là tập hợp dữ liệu có tổ chức.")
    assert blocks[1].heading_path == ("Chương 1. Tổng quan", "1.1 Khái niệm")
    assert blocks[2].text == "PK | Khóa chính"


def test_pdf_headings_detected_from_font_size_and_numbering(tmp_path):
    path = tmp_path / "slides.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "1 Tong quan", fontsize=20)
    page.insert_text((72, 110), "1.1 Khai niem co ban", fontsize=11)
    page.insert_text((72, 130), "Noi dung dong mot.", fontsize=11)
    page.insert_text((72, 144), "Noi dung dong hai.", fontsize=11)
    page.insert_text((72, 170), "2 cach tiep can chinh.", fontsize=11)
    document.save(path)

    blocks = extract_blocks(path)
    assert blocks[0].heading_path == ("1 Tong quan", "1.1 Khai niem co ban")
    assert blocks[0].text == "Noi dung dong mot. Noi dung dong hai."
    assert blocks[1].text == "2 cach tiep can chinh."


def test_pdf_repeated_header_is_ignored(tmp_path):
    path = tmp_path / "book.pdf"
    document = pymupdf.open()
    for number in range(1, 5):
        page = document.new_page()
        page.insert_text((72, 40), "GIAO TRINH CSDL", fontsize=20)
        if number == 1:
            page.insert_text((72, 90), "1.1 Mo dau", fontsize=11)
        page.insert_text((72, 120), f"Noi dung trang {number}.", fontsize=11)
    document.save(path)

    blocks = extract_blocks(path)
    assert [block.page for block in blocks] == [1, 2, 3, 4]
    assert all(block.heading_path == ("1.1 Mo dau",) for block in blocks)


def test_pptx_title_is_heading_and_body_keeps_reading_order(tmp_path):
    path = tmp_path / "lesson.pptx"
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = "Bài 1"
    left = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(3), Inches(1))
    right = slide.shapes.add_textbox(Inches(5), Inches(2), Inches(3), Inches(1))
    left.text = "Nội dung bên trái"
    right.text = "Nội dung bên phải"
    deck.save(path)

    blocks = extract_blocks(path)
    assert blocks == [Block(1, ("Bài 1",), "Nội dung bên trái Nội dung bên phải")]


def test_split_sentences_keeps_punctuation():
    assert split_sentences("Câu một. Câu hai? Câu ba…") == ["Câu một.", "Câu hai?", "Câu ba…"]


def test_structure_chunking_respects_sections_and_prefix():
    blocks = [
        Block(1, ("Chương 1",), "Một hai ba. Bốn năm sáu."),
        Block(2, ("Chương 1",), "Bảy tám chín."),
        Block(2, ("Chương 2",), "Mười."),
    ]
    chunks = _chunk(blocks, max_words=6, overlap_words=3)
    assert [chunk.body for chunk in chunks] == [
        "Một hai ba. Bốn năm sáu.",
        "Bốn năm sáu. Bảy tám chín.",
        "Mười.",
    ]
    assert [chunk.page for chunk in chunks] == [1, 1, 2]
    assert chunks[2].heading_path == ("Chương 2",)
    assert chunks[0].text == "Giáo trình CSDL > Chương 1\nMột hai ba. Bốn năm sáu."
    assert _chunk(blocks, max_words=6, overlap_words=3, prefix=False)[0].text == chunks[0].body
    assert len({chunk.chunk_id for chunk in chunks}) == 3


def test_structure_chunking_splits_overlong_sentence():
    words = " ".join(f"w{i}" for i in range(10))
    chunks = _chunk([Block(1, (), words + ".")], max_words=4, overlap_words=0)
    assert [len(chunk.body.split()) for chunk in chunks] == [4, 4, 2]
    assert chunks[0].text == "Giáo trình CSDL\n" + chunks[0].body


def test_fixed_chunking_uses_word_windows_per_page():
    blocks = [Block(1, ("A",), " ".join(f"w{i}" for i in range(10)))]
    chunks = _chunk(blocks, strategy="fixed", chunk_words=6, fixed_overlap_words=2)
    assert [chunk.body.split()[0] for chunk in chunks] == ["w0", "w4"]
    assert chunks[0].heading_path == ("A",)


def test_chunk_round_trips_through_dict():
    chunk = _chunk([Block(3, ("X", "Y"), "Nội dung.")])[0]
    data = json.loads(json.dumps(chunk.to_dict(), ensure_ascii=False))
    assert Chunk.from_dict(data) == chunk
    assert chunk.breadcrumb() == "Giáo trình CSDL > X > Y"


def test_build_corpus_is_versioned_by_chunking_and_reusable(tmp_path, monkeypatch):
    monkeypatch.delenv("PPL_DATA_DIR", raising=False)
    monkeypatch.delenv("PPL_RUNS_DIR", raising=False)
    settings = load_settings(tmp_path)
    db = Database(settings.db_path)
    db.initialize()
    path = tmp_path / "Giao_trinh_CSDL.docx"
    document = Document()
    document.add_heading("Chương 1", level=1)
    document.add_paragraph("Khóa chính xác định duy nhất một bản ghi.")
    document.save(path)
    metadata = {str(path): {"course": "CS101", "source_type": "textbook"}}

    structure = build_corpus([path], metadata, settings, db, {"strategy": "structure"})
    fixed = build_corpus([path], metadata, settings, db, {"strategy": "fixed"})
    again = build_corpus([path], metadata, settings, db, {"strategy": "structure"})

    assert structure.version_id != fixed.version_id
    assert again.reused is True and again.version_id == structure.version_id
    first = Chunk.from_dict(json.loads(structure.chunks_path.read_text(encoding="utf-8").splitlines()[0]))
    assert first.doc_title == "Giao trinh CSDL"
    assert first.heading_path == ("Chương 1",)
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_ingestion.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.chunking'`

- [ ] **Step 3: Thay `Chunk` trong `src/models.py`**

Thay toàn bộ class `Chunk` cũ bằng:
```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    doc_title: str
    course: str
    source_type: str
    page: int
    heading_path: tuple[str, ...]
    body: str
    text: str

    def breadcrumb(self) -> str:
        return " > ".join(part for part in (self.doc_title, *self.heading_path) if part)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["heading_path"] = list(self.heading_path)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(**{**data, "heading_path": tuple(data.get("heading_path", ()))})
```

- [ ] **Step 4: Viết `src/chunking.py`**

```python
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
```

- [ ] **Step 5: Viết lại `src/ingestion.py`**

```python
import hashlib
import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pymupdf
from docx import Document
from docx.table import Table
from pptx import Presentation
from pptx.enum.shapes import MSO_SHAPE_TYPE

from src.chunking import DEFAULT_CHUNKING, chunk_blocks
from src.models import Block
from src.text import normalize_text

MAX_HEADING_CHARS = 120
_CHAPTER = re.compile(r"^(chương|bài|phần)\s+([0-9]+|[ivxlc]+)\b", re.IGNORECASE)
_NUMBERED = re.compile(r"^(\d+(?:\.\d+){0,3})\.?\s+\S")
_DOCX_HEADING = re.compile(r"Heading ([1-3])")


@dataclass(frozen=True)
class CorpusBuildResult:
    version_id: str
    chunks_path: Path
    chunk_count: int
    manifest_hash: str
    reused: bool = False


def heading_level(text: str, size: float, bold: bool, median_size: float, chapter_offset: int) -> int | None:
    candidate = text.strip()
    if not candidate or len(candidate) >= MAX_HEADING_CHARS or candidate.isdigit():
        return None
    large = median_size > 0 and size >= median_size * 1.2
    if _CHAPTER.match(candidate):
        return 1
    numbered = _NUMBERED.match(candidate)
    if numbered:
        depth = numbered.group(1).count(".") + 1
        return depth + chapter_offset if depth >= 2 or bold or large else None
    if large:
        return 1 + chapter_offset
    return None


def _push_heading(stack: list[str], level: int, title: str) -> list[str]:
    return [*stack[: level - 1], title]


def extract_blocks(path: Path | str) -> list[Block]:
    source = Path(path)
    suffix = source.suffix.lower()
    if suffix == ".pdf":
        return _extract_pdf(source)
    if suffix == ".docx":
        return _extract_docx(source)
    if suffix == ".pptx":
        return _extract_pptx(source)
    raise ValueError(f"Unsupported document type: {suffix}")


def _pdf_pages(document) -> list[tuple[int, object, list[tuple[int, str, float, bool]]]]:
    pages = []
    for number, page in enumerate(document, start=1):
        lines = []
        for block in page.get_text("dict").get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                spans = [span for span in line.get("spans", []) if span.get("text", "").strip()]
                if spans:
                    lines.append(
                        (
                            block["number"],
                            "".join(span["text"] for span in spans),
                            max(span["size"] for span in spans),
                            all(span["flags"] & 16 for span in spans),
                        )
                    )
        pages.append((number, page, lines))
    return pages


def _repeated_lines(pages) -> set[str]:
    if len(pages) < 4:
        return set()
    counts = Counter()
    for _, _, lines in pages:
        counts.update({normalize_text(raw) for _, raw, _, _ in lines if len(normalize_text(raw)) <= 80})
    return {text for text, count in counts.items() if count > len(pages) / 2}


def _flush(blocks: list[Block], page: int, stack: list[str], buffer: list[str]) -> None:
    text = normalize_text("\n".join(buffer))
    if text:
        blocks.append(Block(page, tuple(stack), text))
    buffer.clear()


def _ocr_page(page) -> tuple[str, str]:
    try:
        return normalize_text(page.get_text("text", textpage=page.get_textpage_ocr())), ""
    except RuntimeError:
        return "", "Trang không có text layer và OCR không khả dụng."


def _extract_pdf(path: Path) -> list[Block]:
    with pymupdf.open(path) as document:
        pages = _pdf_pages(document)
        all_lines = [line for _, _, lines in pages for line in lines]
        median_size = statistics.median(line[2] for line in all_lines) if all_lines else 0.0
        repeated = _repeated_lines(pages)
        has_chapters = any(_CHAPTER.match(normalize_text(line[1])) for line in all_lines)
        offset = 1 if has_chapters else 0
        stack: list[str] = []
        blocks: list[Block] = []
        for number, page, lines in pages:
            if not lines:
                text, warning = _ocr_page(page)
                blocks.append(Block(number, tuple(stack), text, warning))
                continue
            buffer: list[str] = []
            current = None
            for block_number, raw, size, bold in lines:
                clean = normalize_text(raw)
                if not clean or clean in repeated:
                    continue
                level = heading_level(clean, size, bold, median_size, offset)
                if level is not None or block_number != current:
                    _flush(blocks, number, stack, buffer)
                    current = block_number
                if level is not None:
                    stack = _push_heading(stack, level, clean)
                else:
                    buffer.append(raw)
            _flush(blocks, number, stack, buffer)
    return blocks


def _extract_docx(path: Path) -> list[Block]:
    document = Document(path)
    stack: list[str] = []
    blocks: list[Block] = []
    for item in document.iter_inner_content():
        if isinstance(item, Table):
            rows = [
                " | ".join(normalize_text(cell.text) for cell in row.cells)
                for row in item.rows
                if any(cell.text.strip() for cell in row.cells)
            ]
            if rows:
                blocks.append(Block(1, tuple(stack), "\n".join(rows)))
            continue
        text = normalize_text(item.text)
        if not text:
            continue
        style = item.style.name if item.style is not None else ""
        match = _DOCX_HEADING.fullmatch(style)
        level = 1 if style == "Title" else int(match.group(1)) if match else None
        if level is not None:
            stack = _push_heading(stack, level, text)
        else:
            blocks.append(Block(1, tuple(stack), text))
    return blocks


def _shape_texts(shapes, title_shape=None):
    items: list[tuple[int, int, str]] = []
    title = ""
    for shape in shapes:
        if shape.shape_type == MSO_SHAPE_TYPE.GROUP:
            nested_title, nested_items = _shape_texts(shape.shapes)
            title = title or nested_title
            items.extend(nested_items)
            continue
        if shape is title_shape and getattr(shape, "has_text_frame", False):
            title = shape.text.strip()
            continue
        if getattr(shape, "has_table", False):
            text = "\n".join(
                " | ".join(cell.text.strip() for cell in row.cells)
                for row in shape.table.rows
            ).strip()
        elif getattr(shape, "has_text_frame", False):
            text = shape.text.strip()
        else:
            text = ""
        if text:
            items.append((int(shape.top), int(shape.left), text))
    return title, items


def _extract_pptx(path: Path) -> list[Block]:
    presentation = Presentation(path)
    blocks: list[Block] = []
    band_height = 228600  # 0.25 inch in EMU
    for number, slide in enumerate(presentation.slides, start=1):
        title, items = _shape_texts(slide.shapes, slide.shapes.title)
        ordered = sorted(items, key=lambda item: (round(item[0] / band_height), item[1]))
        body = normalize_text("\n".join(item[2] for item in ordered))
        heading = normalize_text(title)
        if body or heading:
            blocks.append(Block(number, (heading,) if heading else (), body or heading))
    return blocks


def build_corpus(files, metadata: dict[str, dict], settings, database, chunking: dict) -> CorpusBuildResult:
    options = {**DEFAULT_CHUNKING, **chunking}
    manifest_items = []
    all_chunks = []
    for path in sorted((Path(file) for file in files), key=str):
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        item = metadata[str(path)]
        doc_id = file_hash[:24]
        doc_title = item.get("doc_title") or path.stem.replace("_", " ")
        chunks = chunk_blocks(
            extract_blocks(path),
            doc_id=doc_id,
            doc_title=doc_title,
            course=item["course"],
            source_type=item["source_type"],
            **options,
        )
        all_chunks.extend(chunks)
        record = {"doc_id": doc_id, "filename": path.name, "sha256": file_hash, "doc_title": doc_title, **item}
        manifest_items.append(record)
        database.save_document({**record, "status": "processed"})

    manifest_json = json.dumps({"documents": manifest_items, "chunking": options}, ensure_ascii=False, sort_keys=True)
    manifest_hash = hashlib.sha256(manifest_json.encode("utf-8")).hexdigest()
    version_id = manifest_hash[:16]
    output_dir = settings.data_dir / "processed" / version_id
    chunks_path = output_dir / "chunks.jsonl"
    if chunks_path.exists():
        count = sum(1 for line in chunks_path.read_text(encoding="utf-8").splitlines() if line)
        return CorpusBuildResult(version_id, chunks_path, count, manifest_hash, reused=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    chunks_path.write_text(
        "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in all_chunks) + "\n",
        encoding="utf-8",
    )
    (output_dir / "manifest.json").write_text(manifest_json, encoding="utf-8")
    database.save_corpus_version(
        {
            "version_id": version_id,
            "manifest_hash": manifest_hash,
            "chunks_path": str(chunks_path),
            "chunk_count": len(all_chunks),
        }
    )
    return CorpusBuildResult(version_id, chunks_path, len(all_chunks), manifest_hash)
```

- [ ] **Step 6a: Giữ `src/retrieval.py` cũ chạy được tới Task 8**

`tokenize_vi` không còn trong `src/ingestion.py`. Trong `src/retrieval.py` thay dòng `from src.ingestion import tokenize_vi` bằng:
```python
from src.text import tokenize as tokenize_vi
```

- [ ] **Step 6: Sửa `src/rag.py` cho `Chunk` mới**

Trong `_prompt`, thay khối tạo `context` bằng:
```python
    context = "\n\n".join(
        f"[{number}] Nguồn: {result.chunk.breadcrumb()}; trang/slide: {result.chunk.page}\n"
        f"{result.chunk.body}"
        for number, result in enumerate(results, start=1)
    )
```
Trong `_valid_citations`, thay dict `cited.append({...})` bằng:
```python
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
```

- [ ] **Step 7: Sửa `tests/fakes.py` và `tests/test_core.py`**

Trong `tests/fakes.py` thay `make_chunk` và bỏ import `SimpleNamespace`:
```python
from src.models import Chunk


def make_chunk(chunk_id, text, **fields):
    values = {
        "doc_id": "doc-1",
        "doc_title": "Tài liệu",
        "course": "AI101",
        "source_type": "slide",
        "page": 1,
        "heading_path": (),
        "body": text,
        **fields,
    }
    return Chunk(chunk_id=chunk_id, text=text, **values)
```
Trong `tests/test_core.py`:
- Xóa hàm `test_ingestion_preserves_source_and_reading_order` và các import không còn dùng (`Presentation`, `Inches`, `build_corpus`, `chunk_pages`, `extract_document`).
- Trong `test_retrieval_fusion_and_metrics`, thay định nghĩa `chunks` bằng:
```python
    chunks = {
        "c1": Chunk("c1", "d1", "Slide", "AI101", "slide", 1, ("Mã môn",), "Mã môn AI101", "Mã môn AI101"),
        "c2": Chunk("c2", "d1", "Slide", "AI101", "slide", 2, ("Khái niệm",), "Giải thích học máy", "Giải thích học máy"),
    }
```

- [ ] **Step 8: Chạy toàn bộ test**

Run: `.venv/Scripts/python -m pytest -q`
Expected: tất cả pass (bao gồm `tests/test_ingestion.py`: 11 passed).

- [ ] **Step 9: Commit**

```bash
git add src/models.py src/chunking.py src/ingestion.py src/rag.py tests/fakes.py tests/test_ingestion.py tests/test_core.py
git commit -m "feat: extract heading paths and add structure-aware chunking

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: `RetrievalIndex` nhiều tokenizer, numpy/FAISS

**Files:**
- Create: `src/index.py`, `tests/test_index.py`

**Interfaces:**
- Consumes: `Chunk`, `tokenize`, `normalize_text`.
- Produces:
  - `load_encoder(model_name: str)` (cache cấp module)
  - `RetrievalIndex` với thuộc tính `directory: Path`, `chunk_list: list[Chunk]`, `chunk_ids: list[str]`, `chunks: dict[str, Chunk]`, `embeddings: np.ndarray`, `meta: dict`; property `version -> str` (tên thư mục), `embedding_model -> str`, `tokenizers -> list[str]`
  - `RetrievalIndex.build(chunks, output_dir, *, embedding_model, tokenizers=("whitespace",), dense_backend="numpy", bm25_k1=1.5, bm25_b=0.75, chunking=None, encoder=None) -> RetrievalIndex`
  - `RetrievalIndex.load(directory, encoder=None) -> RetrievalIndex`
  - `add_tokenizer(mode: str) -> None`; `idf(mode: str) -> dict[str, float]`
  - `sparse_search(query: str, tokenizer: str, limit: int) -> list[tuple[str, float]]` (chỉ điểm > 0, giảm dần, hòa → id)
  - `dense_search(query: str, limit: int) -> list[tuple[str, float]]`

- [ ] **Step 1: Viết test thất bại `tests/test_index.py`**

```python
import unicodedata

import pytest

from src.index import RetrievalIndex
from tests.fakes import FakeEncoder, make_chunk

TEXTS = {
    "c1": "Mã môn AI101 là học phần nhập môn trí tuệ nhân tạo",
    "c2": "Học máy là lĩnh vực nghiên cứu thuật toán học từ dữ liệu",
    "c3": "Cơ sở dữ liệu quan hệ lưu trữ bảng và khóa chính",
    "c4": "Mạng nơ ron sâu gồm nhiều tầng biến đổi phi tuyến",
}


@pytest.fixture
def chunks():
    return [make_chunk(chunk_id, text) for chunk_id, text in TEXTS.items()]


@pytest.fixture
def index(tmp_path, chunks):
    return RetrievalIndex.build(
        chunks, tmp_path / "idx-v1", embedding_model="fake", tokenizers=("whitespace",), encoder=FakeEncoder()
    )


def test_build_and_load_round_trip(tmp_path, index, chunks):
    loaded = RetrievalIndex.load(tmp_path / "idx-v1", encoder=FakeEncoder())
    assert loaded.version == "idx-v1"
    assert loaded.chunk_list == chunks
    assert loaded.tokenizers == ["whitespace"]
    assert loaded.meta["bm25"] == {"k1": 1.5, "b": 0.75}
    assert loaded.sparse_search("AI101", "whitespace", 10) == index.sparse_search("AI101", "whitespace", 10)


def test_sparse_search_returns_only_matching_chunks(index):
    results = index.sparse_search("mã môn AI101", "whitespace", 10)
    assert [chunk_id for chunk_id, _ in results] == ["c1"]
    assert results[0][1] > 0
    assert index.sparse_search("???", "whitespace", 10) == []


def test_sparse_search_matches_nfd_query(index):
    nfd = unicodedata.normalize("NFD", "Học máy thuật toán")
    assert index.sparse_search(nfd, "whitespace", 10) == index.sparse_search("Học máy thuật toán", "whitespace", 10)


def test_dense_search_ranks_by_cosine(index):
    results = index.dense_search("cơ sở dữ liệu quan hệ", 2)
    assert results[0][0] == "c3"
    assert len(results) == 2
    assert results[0][1] >= results[1][1]


def test_missing_tokenizer_error_names_fix(index):
    with pytest.raises(ValueError, match="add-tokenizer"):
        index.sparse_search("học máy", "pyvi", 10)


def test_add_tokenizer_persists(tmp_path, index):
    index.add_tokenizer("pyvi")
    loaded = RetrievalIndex.load(tmp_path / "idx-v1", encoder=FakeEncoder())
    assert loaded.tokenizers == ["whitespace", "pyvi"]
    assert loaded.sparse_search("nghiên cứu", "pyvi", 10)[0][0] == "c2"
    assert "nghiên_cứu" in loaded.idf("pyvi")


def test_faiss_backend_matches_numpy(tmp_path, chunks, index):
    pytest.importorskip("faiss")
    faiss_index = RetrievalIndex.build(
        chunks, tmp_path / "idx-faiss", embedding_model="fake", dense_backend="faiss", encoder=FakeEncoder()
    )
    assert (tmp_path / "idx-faiss" / "faiss.index").exists()
    reloaded = RetrievalIndex.load(tmp_path / "idx-faiss", encoder=FakeEncoder())
    expected = index.dense_search("học máy dữ liệu", 3)
    actual = reloaded.dense_search("học máy dữ liệu", 3)
    assert [chunk_id for chunk_id, _ in actual] == [chunk_id for chunk_id, _ in expected]
    assert [score for _, score in actual] == pytest.approx([score for _, score in expected], abs=1e-5)


def test_build_rejects_empty_corpus(tmp_path):
    with pytest.raises(ValueError, match="without chunks"):
        RetrievalIndex.build([], tmp_path / "empty", embedding_model="fake", encoder=FakeEncoder())
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_index.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.index'`

- [ ] **Step 3: Viết `src/index.py`**

```python
import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.models import Chunk
from src.text import normalize_text, tokenize

_ENCODERS: dict = {}


def load_encoder(model_name: str):
    if model_name not in _ENCODERS:
        from sentence_transformers import SentenceTransformer

        _ENCODERS[model_name] = SentenceTransformer(model_name)
    return _ENCODERS[model_name]


def _top(chunk_ids: list[str], values, limit: int, positive_only: bool) -> list[tuple[str, float]]:
    pairs = [
        (chunk_id, float(value))
        for chunk_id, value in zip(chunk_ids, values)
        if not positive_only or value > 0
    ]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return pairs[:limit]


class RetrievalIndex:
    def __init__(self, directory: Path | str, chunks: list[Chunk], embeddings: np.ndarray, meta: dict, encoder=None):
        self.directory = Path(directory)
        self.chunk_list = chunks
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.embeddings = embeddings
        self.meta = meta
        self._encoder = encoder
        self._bm25: dict[str, BM25Okapi] = {}
        self._faiss = None

    @property
    def version(self) -> str:
        return self.directory.name

    @property
    def embedding_model(self) -> str:
        return self.meta["embedding_model"]

    @property
    def tokenizers(self) -> list[str]:
        return list(self.meta["tokenizers"])

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        output_dir: Path | str,
        *,
        embedding_model: str,
        tokenizers=("whitespace",),
        dense_backend: str = "numpy",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        chunking: dict | None = None,
        encoder=None,
    ) -> "RetrievalIndex":
        if not chunks:
            raise ValueError("Cannot build an index without chunks")
        if dense_backend not in ("numpy", "faiss"):
            raise ValueError(f"Unsupported dense backend: {dense_backend}")
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        encoder = encoder or load_encoder(embedding_model)
        embeddings = np.asarray(
            encoder.encode([chunk.text for chunk in chunks], normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )
        np.save(target / "embeddings.npy", embeddings)
        (target / "chunks.jsonl").write_text(
            "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks) + "\n",
            encoding="utf-8",
        )
        meta = {
            "embedding_model": embedding_model,
            "dense_backend": dense_backend,
            "bm25": {"k1": bm25_k1, "b": bm25_b},
            "chunk_count": len(chunks),
            "chunking": chunking or {},
            "tokenizers": [],
        }
        index = cls(target, chunks, embeddings, meta, encoder)
        if dense_backend == "faiss":
            import faiss

            flat = faiss.IndexFlatIP(embeddings.shape[1])
            flat.add(embeddings)
            faiss.write_index(flat, str(target / "faiss.index"))
        for mode in tokenizers:
            index.add_tokenizer(mode)
        index._write_meta()
        return index

    @classmethod
    def load(cls, directory: Path | str, encoder=None) -> "RetrievalIndex":
        source = Path(directory)
        meta = json.loads((source / "index_meta.json").read_text(encoding="utf-8"))
        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in (source / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(chunks) != meta["chunk_count"]:
            raise ValueError("Index metadata does not match chunk count")
        return cls(source, chunks, np.load(source / "embeddings.npy"), meta, encoder)

    def _write_meta(self) -> None:
        (self.directory / "index_meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add_tokenizer(self, mode: str) -> None:
        tokens = [tokenize(chunk.text, mode) for chunk in self.chunk_list]
        (self.directory / f"tokens_{mode}.json").write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")
        if mode not in self.meta["tokenizers"]:
            self.meta["tokenizers"].append(mode)
        self._bm25.pop(mode, None)
        self._write_meta()

    def _bm25_for(self, mode: str) -> BM25Okapi:
        if mode not in self.meta["tokenizers"]:
            raise ValueError(
                f"Tokenizer '{mode}' is not built for index {self.version}; run "
                f"`python -m src.cli index add-tokenizer --index {self.directory} --tokenizer {mode}`"
            )
        if mode not in self._bm25:
            tokens = json.loads((self.directory / f"tokens_{mode}.json").read_text(encoding="utf-8"))
            parameters = self.meta["bm25"]
            self._bm25[mode] = BM25Okapi(tokens, k1=parameters["k1"], b=parameters["b"])
        return self._bm25[mode]

    def idf(self, mode: str) -> dict[str, float]:
        return {token: float(value) for token, value in self._bm25_for(mode).idf.items()}

    def sparse_search(self, query: str, tokenizer: str, limit: int) -> list[tuple[str, float]]:
        bm25 = self._bm25_for(tokenizer)
        query_tokens = tokenize(query, tokenizer)
        if not query_tokens:
            return []
        return _top(self.chunk_ids, bm25.get_scores(query_tokens), limit, positive_only=True)

    def _encoder_instance(self):
        if self._encoder is None:
            self._encoder = load_encoder(self.embedding_model)
        return self._encoder

    def dense_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        vector = np.asarray(
            self._encoder_instance().encode([normalize_text(query)], normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )[0]
        if self.meta["dense_backend"] == "faiss":
            import faiss

            if self._faiss is None:
                self._faiss = faiss.read_index(str(self.directory / "faiss.index"))
            scores, positions = self._faiss.search(vector[None, :], min(limit, len(self.chunk_ids)))
            pairs = [
                (self.chunk_ids[position], float(score))
                for position, score in zip(positions[0], scores[0])
                if position >= 0
            ]
            return sorted(pairs, key=lambda item: (-item[1], item[0]))
        return _top(self.chunk_ids, self.embeddings @ vector, limit, positive_only=False)
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_index.py -q`
Expected: `8 passed`

- [ ] **Step 5: Commit**

```bash
git add src/index.py tests/test_index.py
git commit -m "feat: add multi-tokenizer retrieval index with numpy and faiss backends

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: `RetrievalPipeline` tách tầng, log điểm và thời gian

**Files:**
- Create: `src/pipeline.py`, `tests/test_pipeline.py`

**Interfaces:**
- Consumes: `RetrievalIndex`, `RetrievalCache`, `Reranker`, các hàm `fusion`, `PipelineConfig`, `StageScores`, `RetrievedChunk`.
- Produces:
  - `PipelineResult(results: list[RetrievedChunk], timings_ms: dict[str, float], alpha_used: float | None = None)`; `timings_ms` luôn có đủ khóa `sparse`, `dense`, `fusion`, `rerank`, `total`.
  - `RetrievalPipeline(index, cache=None, reranker_factory=None, synchronize=None)`; `run(query: str, config: PipelineConfig, use_cache: bool = True) -> PipelineResult`. `reranker_factory(model_name) -> Reranker`. `synchronize=None` → tự dùng `torch.cuda.synchronize` khi có CUDA.
  - Thứ tự kết quả: top-N sau rerank, tiếp theo phần còn lại của danh sách fusion theo thứ tự cũ. `rank` bắt đầu từ 1.

- [ ] **Step 1: Viết test thất bại `tests/test_pipeline.py`**

```python
import pytest

from src.cache import RetrievalCache
from src.index import RetrievalIndex
from src.models import PipelineConfig
from src.pipeline import RetrievalPipeline
from src.reranking import Reranker
from tests.fakes import FakeCrossEncoder, FakeEncoder, make_chunk

TEXTS = {
    "c1": "Mã môn AI101 là học phần nhập môn trí tuệ nhân tạo",
    "c2": "Học máy là lĩnh vực nghiên cứu thuật toán học từ dữ liệu",
    "c3": "Cơ sở dữ liệu quan hệ lưu trữ bảng và khóa chính",
    "c4": "Mạng nơ ron sâu gồm nhiều tầng biến đổi phi tuyến",
}
BM25 = PipelineConfig(dense=False, fusion="none", rerank=False, top_l=10)
DENSE = PipelineConfig(sparse=False, fusion="none", rerank=False, top_l=10)
HYBRID = PipelineConfig(fusion="weighted", rerank=False, top_l=10)


@pytest.fixture
def pipeline(tmp_path):
    index = RetrievalIndex.build(
        [make_chunk(chunk_id, text) for chunk_id, text in TEXTS.items()],
        tmp_path / "idx-v1",
        embedding_model="fake",
        encoder=FakeEncoder(),
    )
    cross = FakeCrossEncoder()
    cache = RetrievalCache(tmp_path / "cache.sqlite")
    built = RetrievalPipeline(
        index,
        cache=cache,
        reranker_factory=lambda name: Reranker(name, model=cross, cache=cache),
        synchronize=lambda: None,
    )
    built.cross = cross
    return built


def test_bm25_only_records_sparse_stage(pipeline):
    result = pipeline.run("mã môn AI101", BM25)
    top = result.results[0]
    assert top.chunk.chunk_id == "c1"
    assert top.scores.rank == 1 and top.scores.sparse_rank == 1
    assert top.scores.dense_score is None and top.scores.fusion_score is None
    assert set(result.timings_ms) == {"sparse", "dense", "fusion", "rerank", "total"}
    assert result.timings_ms["dense"] == 0.0


def test_dense_only_ranks_all_chunks(pipeline):
    result = pipeline.run("cơ sở dữ liệu quan hệ", DENSE)
    assert result.results[0].chunk.chunk_id == "c3"
    assert len(result.results) == 4
    assert result.results[0].scores.sparse_score is None


def test_hybrid_logs_every_stage_score(pipeline):
    result = pipeline.run("học máy dữ liệu", HYBRID)
    by_id = {item.chunk.chunk_id: item.scores for item in result.results}
    assert by_id["c2"].sparse_rank is not None and by_id["c2"].dense_rank is not None
    assert by_id["c4"].sparse_score is None and by_id["c4"].dense_score is not None
    assert [item.scores.rank for item in result.results] == list(range(1, len(result.results) + 1))
    fusion_scores = [item.scores.fusion_score for item in result.results]
    assert fusion_scores == sorted(fusion_scores, reverse=True)
    assert result.alpha_used == 0.5


def test_hybrid_with_no_sparse_hits_falls_back_to_dense(pipeline):
    result = pipeline.run("zzz qqq", HYBRID)
    assert len(result.results) == 4
    assert all(item.scores.sparse_score is None for item in result.results)


def test_rrf_and_adaptive_fusion(pipeline):
    rrf = pipeline.run("học máy dữ liệu", HYBRID.replace(fusion="rrf", rrf_k=60))
    assert rrf.results[0].scores.fusion_score == pytest.approx(
        sum(1 / (60 + rank) for rank in (rrf.results[0].scores.sparse_rank, rrf.results[0].scores.dense_rank) if rank)
    )
    adaptive = pipeline.run("Mã môn AI101", HYBRID.replace(fusion="adaptive", adaptive_beta=0.3))
    assert adaptive.alpha_used > 0.5


def test_rerank_reorders_head_and_keeps_tail(pipeline):
    config = HYBRID.replace(rerank=True, rerank_n=2, context_k=2)
    result = pipeline.run("học máy thuật toán dữ liệu", config)
    head = result.results[:2]
    assert all(item.scores.rerank_score is not None for item in head)
    assert head[0].scores.rerank_score >= head[1].scores.rerank_score
    assert all(item.scores.rerank_score is None for item in result.results[2:])
    assert result.timings_ms["rerank"] >= 0.0


def test_cache_skips_recomputation_and_can_be_bypassed(pipeline):
    config = HYBRID.replace(rerank=True, rerank_n=2, context_k=2)
    pipeline.run("học máy", config)
    seen = pipeline.cross.pairs_seen
    encoder_calls = pipeline.index._encoder.calls
    pipeline.run("học máy", config)
    assert pipeline.cross.pairs_seen == seen
    assert pipeline.index._encoder.calls == encoder_calls
    pipeline.run("học máy", config, use_cache=False)
    assert pipeline.cross.pairs_seen == seen + 2
    assert pipeline.index._encoder.calls == encoder_calls + 1
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.pipeline'`

- [ ] **Step 3: Viết `src/pipeline.py`**

```python
import time
from dataclasses import dataclass, field
from typing import Callable

from src.cache import RetrievalCache
from src.fusion import adaptive_alpha, fuse_rrf, fuse_weighted, rank_ids
from src.models import PipelineConfig, RetrievedChunk, StageScores
from src.reranking import Reranker
from src.text import normalize_text, tokenize

STAGES = ("sparse", "dense", "fusion", "rerank")


@dataclass(frozen=True)
class PipelineResult:
    results: list[RetrievedChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)
    alpha_used: float | None = None


def _cuda_synchronize() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class RetrievalPipeline:
    def __init__(
        self,
        index,
        cache: RetrievalCache | None = None,
        reranker_factory: Callable[[str], Reranker] | None = None,
        synchronize: Callable[[], None] | None = None,
    ):
        self.index = index
        self.cache = cache
        self._reranker_factory = reranker_factory or (lambda name: Reranker(name, cache=cache))
        self._rerankers: dict[str, Reranker] = {}
        self._synchronize = synchronize or _cuda_synchronize

    def _clock(self) -> float:
        self._synchronize()
        return time.perf_counter()

    def _reranker(self, model_name: str) -> Reranker:
        if model_name not in self._rerankers:
            self._rerankers[model_name] = self._reranker_factory(model_name)
        return self._rerankers[model_name]

    def _first_stage(self, branch: str, variant: str, query: str, top_l: int, search, use_cache: bool):
        if self.cache is None or not use_cache:
            return search()
        key = RetrievalCache.first_stage_key(self.index.version, branch, variant, query, top_l)
        cached = self.cache.get_first_stage(key)
        if cached is None:
            cached = search()
            self.cache.put_first_stage(key, cached)
        return cached

    def run(self, query: str, config: PipelineConfig, use_cache: bool = True) -> PipelineResult:
        query = normalize_text(query)
        timings = {stage: 0.0 for stage in STAGES}
        started = self._clock()
        sparse: list[tuple[str, float]] = []
        dense: list[tuple[str, float]] = []
        if config.sparse:
            mark = self._clock()
            sparse = self._first_stage(
                "sparse", config.tokenizer, query, config.top_l,
                lambda: self.index.sparse_search(query, config.tokenizer, config.top_l), use_cache,
            )
            timings["sparse"] = (self._clock() - mark) * 1000
        if config.dense:
            mark = self._clock()
            dense = self._first_stage(
                "dense", self.index.embedding_model, query, config.top_l,
                lambda: self.index.dense_search(query, config.top_l), use_cache,
            )
            timings["dense"] = (self._clock() - mark) * 1000

        mark = self._clock()
        sparse_scores, dense_scores = dict(sparse), dict(dense)
        sparse_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(sparse, start=1)}
        dense_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(dense, start=1)}
        fused: dict[str, float] | None = None
        alpha_used = None
        if config.fusion == "none":
            order = [chunk_id for chunk_id, _ in (sparse if config.sparse else dense)]
        elif config.fusion == "rrf":
            fused = fuse_rrf([[chunk_id for chunk_id, _ in sparse], [chunk_id for chunk_id, _ in dense]], config.rrf_k)
            order = rank_ids(fused)
        else:
            alpha_used = config.alpha
            if config.fusion == "adaptive":
                alpha_used, _ = adaptive_alpha(
                    query, tokenize(query, config.tokenizer), config.alpha,
                    config.adaptive_beta, self.index.idf(config.tokenizer),
                )
            fused = fuse_weighted(sparse_scores, dense_scores, alpha_used)
            order = rank_ids(fused)
        timings["fusion"] = (self._clock() - mark) * 1000

        rerank_scores: dict[str, float] = {}
        if config.rerank and order:
            mark = self._clock()
            head = order[: config.rerank_n]
            values = self._reranker(config.reranker_model).score(
                query, [self.index.chunks[chunk_id] for chunk_id in head], use_cache=use_cache
            )
            rerank_scores = dict(zip(head, values))
            order = sorted(head, key=lambda chunk_id: (-rerank_scores[chunk_id], chunk_id)) + order[config.rerank_n :]
            timings["rerank"] = (self._clock() - mark) * 1000
        timings["total"] = (self._clock() - started) * 1000

        results = [
            RetrievedChunk(
                self.index.chunks[chunk_id],
                StageScores(
                    rank=rank,
                    sparse_score=sparse_scores.get(chunk_id),
                    sparse_rank=sparse_ranks.get(chunk_id),
                    dense_score=dense_scores.get(chunk_id),
                    dense_rank=dense_ranks.get(chunk_id),
                    fusion_score=fused.get(chunk_id) if fused is not None else None,
                    rerank_score=rerank_scores.get(chunk_id),
                ),
            )
            for rank, chunk_id in enumerate(order, start=1)
        ]
        return PipelineResult(results, timings, alpha_used)
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_pipeline.py -q`
Expected: `7 passed`

- [ ] **Step 5: Commit**

```bash
git add src/pipeline.py tests/test_pipeline.py
git commit -m "feat: add staged retrieval pipeline with score decomposition and timings

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Chuyển RAG, storage, config sang pipeline mới; xóa code cũ

**Files:**
- Modify: `src/rag.py`, `src/storage.py`, `src/config.py`, `src/models.py` (xóa `SearchResult`, `RagConfig`), `src/reranking.py` (xóa hàm `rerank` cũ), `src/experiments.py`, `run_rag_evaluation.py`, `tests/test_core.py`
- Delete: `src/retrieval.py`
- Create: `tests/test_rag.py`, `tests/test_storage.py`

**Interfaces:**
- Produces:
  - `answer_question(query: str, pipeline, config: PipelineConfig, client, model: str) -> RagAnswer` (gọi `pipeline.run(query, config, use_cache=False)`; dùng `config.temperature`, `config.timeout_seconds`)
  - `Settings` thêm `indexes_dir: Path`, `cache_path: Path`; đọc `PPL_DATA_DIR`, `PPL_RUNS_DIR`
  - `Database.save_rag_config(name: str, config: PipelineConfig)`; `Database.load_pipeline_configs() -> tuple[dict[str, PipelineConfig], list[str]]` (hợp lệ, tên không hợp lệ)
  - `src.experiments.execute_query` dùng `RetrievalPipeline` (tạm thời, Kế hoạch 3 thay thế)

- [ ] **Step 1: Viết test thất bại `tests/test_rag.py` và `tests/test_storage.py`**

`tests/test_rag.py`:
```python
from types import SimpleNamespace

from src.models import PipelineConfig, RetrievedChunk, StageScores
from src.pipeline import PipelineResult
from src.rag import REFUSAL_TEXT, answer_question
from tests.fakes import make_chunk


class FakePipeline:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def run(self, query, config, use_cache=True):
        self.calls.append(use_cache)
        return PipelineResult(self.results, {"sparse": 1.0, "dense": 1.0, "fusion": 0.0, "rerank": 0.0, "total": 2.0})


class FakeCompletions:
    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Theo tài liệu [1] và [99]."))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )


def _results():
    chunk = make_chunk("c1", "Mã môn AI101", doc_title="Slide AI", heading_path=("Bài 1",), page=3)
    return [RetrievedChunk(chunk, StageScores(rank=1, fusion_score=0.8))]


def test_answer_keeps_valid_citations_and_uses_config():
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    pipeline = FakePipeline(_results())
    config = PipelineConfig(rerank=False, temperature=0.2, timeout_seconds=30)

    answer = answer_question("Mã môn AI101 là gì?", pipeline, config, client, "test-model")

    assert "[1]" in answer.text and "[99]" not in answer.text
    assert answer.citations[0]["chunk_id"] == "c1"
    assert answer.citations[0]["heading_path"] == ["Bài 1"]
    assert "Slide AI > Bài 1" in completions.calls[0]["messages"][1]["content"]
    assert completions.calls[0]["temperature"] == 0.2 and completions.calls[0]["timeout"] == 30
    assert pipeline.calls == [False]
    assert answer.retrieval_ms == 2.0


def test_answer_refuses_below_threshold_without_calling_llm():
    completions = FakeCompletions()
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    config = PipelineConfig(rerank=False, refusal_threshold=2.0)
    answer = answer_question("Câu hỏi ngoài tài liệu", FakePipeline(_results()), config, client, "m")
    assert answer.refused is True and answer.text == REFUSAL_TEXT
    assert completions.calls == []
```

`tests/test_storage.py`:
```python
import json
import sqlite3

from src.config import load_settings
from src.models import PipelineConfig
from src.storage import Database


def _db(tmp_path, monkeypatch):
    monkeypatch.delenv("PPL_DATA_DIR", raising=False)
    monkeypatch.delenv("PPL_RUNS_DIR", raising=False)
    settings = load_settings(tmp_path)
    db = Database(settings.db_path)
    db.initialize()
    return settings, db


def test_pipeline_config_round_trip_without_secrets(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-persisted")
    _, db = _db(tmp_path, monkeypatch)
    db.save_rag_config("hybrid", PipelineConfig(fusion="rrf", rerank=False))
    configs, invalid = db.load_pipeline_configs()
    assert configs["hybrid"].fusion == "rrf" and invalid == []
    assert "must-not-be-persisted" not in json.dumps(db.list_rag_configs())


def test_legacy_rag_config_is_reported_invalid(tmp_path, monkeypatch):
    settings, db = _db(tmp_path, monkeypatch)
    with sqlite3.connect(settings.db_path) as connection:
        connection.execute(
            "INSERT INTO rag_configs(name, config_json) VALUES (?, ?)",
            ("old", json.dumps({"method": "adaptive", "use_reranker": True})),
        )
    configs, invalid = db.load_pipeline_configs()
    assert configs == {} and invalid == ["old"]


def test_settings_honor_data_dir_override(tmp_path, monkeypatch):
    monkeypatch.setenv("PPL_DATA_DIR", str(tmp_path / "drive" / "data"))
    monkeypatch.setenv("PPL_RUNS_DIR", str(tmp_path / "drive" / "runs"))
    settings = load_settings(tmp_path / "repo")
    data_dir = (tmp_path / "drive" / "data").resolve()
    assert settings.db_path == data_dir / "app.db"
    assert settings.indexes_dir == data_dir / "indexes"
    assert settings.cache_path == data_dir / "cache" / "retrieval.sqlite"
    assert settings.runs_dir.is_dir()


def test_saving_same_corpus_version_twice_is_idempotent(tmp_path, monkeypatch):
    _, db = _db(tmp_path, monkeypatch)
    record = {"version_id": "v1", "manifest_hash": "h", "chunks_path": "p", "chunk_count": 1}
    db.save_corpus_version(record)
    db.save_corpus_version(record)
    db.set_active_corpus("v1")
    assert db.get_active_corpus()["version_id"] == "v1"
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_rag.py tests/test_storage.py -q`
Expected: FAIL (`answer_question` còn dùng `retrieve`; `load_pipeline_configs` chưa có; `indexes_dir` chưa có).

- [ ] **Step 3: Viết lại phần truy xuất trong `src/rag.py`**

Đổi import đầu file thành:
```python
import re
import time
from dataclasses import dataclass

from src.models import PipelineConfig
```
Thay toàn bộ hàm `answer_question` bằng:
```python
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
```

- [ ] **Step 4: Sửa `src/config.py`**

Thay `Settings` và `load_settings` bằng:
```python
@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    runs_dir: Path
    db_path: Path
    indexes_dir: Path
    cache_path: Path
    openai_api_key: str
    openai_base_url: str
    openai_model: str


def load_settings(root: Path | None = None) -> Settings:
    load_dotenv()
    project_root = Path(root or Path.cwd()).resolve()
    data_dir = Path(os.getenv("PPL_DATA_DIR") or project_root / "data").resolve()
    runs_dir = Path(os.getenv("PPL_RUNS_DIR") or project_root / "runs").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        root=project_root,
        data_dir=data_dir,
        runs_dir=runs_dir,
        db_path=data_dir / "app.db",
        indexes_dir=data_dir / "indexes",
        cache_path=data_dir / "cache" / "retrieval.sqlite",
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=os.getenv("OPENAI_MODEL", ""),
    )
```

- [ ] **Step 5: Sửa `src/storage.py`**

Đổi `from src.models import RagConfig` thành `from src.models import PipelineConfig`. Đổi chữ ký `save_rag_config(self, name: str, config: RagConfig)` thành `save_rag_config(self, name: str, config: PipelineConfig)`. Trong `save_corpus_version` đổi `"INSERT INTO corpus_versions(version_id, metadata_json) VALUES (?, ?)"` thành `"INSERT OR IGNORE INTO corpus_versions(version_id, metadata_json) VALUES (?, ?)"`. Thêm phương thức sau `list_rag_configs`:
```python
    def load_pipeline_configs(self) -> tuple[dict[str, PipelineConfig], list[str]]:
        valid: dict[str, PipelineConfig] = {}
        invalid: list[str] = []
        for item in self.list_rag_configs():
            try:
                valid[item["name"]] = PipelineConfig.from_dict(item["config"])
            except (TypeError, ValueError):
                invalid.append(item["name"])
        return valid, invalid
```

- [ ] **Step 6: Xóa code cũ**

- Xóa file `src/retrieval.py`.
- Trong `src/models.py` xóa class `SearchResult` và `RagConfig`.
- Trong `src/reranking.py` xóa hàm `rerank` cũ và import `from src.models import SearchResult`; giữ `import time` chỉ nếu còn dùng (không còn → xóa).

- [ ] **Step 7: Sửa tạm `src/experiments.py`**

Thay import:
```python
from src.config import load_yaml
from src.evaluation import evaluate_rankings
from src.index import RetrievalIndex
from src.models import PipelineConfig
from src.pipeline import RetrievalPipeline
```
Thay `_load_index` và `execute_query` bằng:
```python
_LEGACY_METHODS = {
    "bm25": {"sparse": True, "dense": False, "fusion": "none"},
    "dense": {"sparse": False, "dense": True, "fusion": "none"},
    "rrf": {"fusion": "rrf"},
    "weighted": {"fusion": "weighted"},
    "adaptive": {"fusion": "adaptive"},
}


@lru_cache(maxsize=2)
def _load_pipeline(path: str) -> RetrievalPipeline:
    return RetrievalPipeline(RetrievalIndex.load(path))


def execute_query(experiment_id: str, query: dict, config: dict) -> dict:
    experiment = config["experiments"][experiment_id]
    retrieval_config = config.get("retrieval", {})
    pipeline_config = PipelineConfig(
        **_LEGACY_METHODS[experiment["method"]],
        alpha=float(experiment.get("alpha", 0.5)),
        rrf_k=int(experiment.get("rrf_k", 60)),
        rerank=bool(experiment.get("use_reranker", False)),
        top_l=int(retrieval_config.get("top_l", 100)),
        rerank_n=int(retrieval_config.get("rerank_n", 30)),
        context_k=int(retrieval_config.get("context_k", 5)),
        tokenizer=config.get("tokenizer", "whitespace"),
        reranker_model=config.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
    )
    result = _load_pipeline(str(config["index_dir"])).run(query["text"], pipeline_config)
    return {
        "ranked_chunk_ids": [item.chunk.chunk_id for item in result.results],
        "latency_ms": result.timings_ms["total"],
        "rerank_ms": result.timings_ms["rerank"],
    }
```
Xóa import `time`, `rerank`, `RetrievalIndex, retrieve` cũ không còn dùng.

- [ ] **Step 8: Sửa `run_rag_evaluation.py`**

Thay các import `from src.models import RagConfig` và `from src.retrieval import RetrievalIndex` bằng:
```python
from src.index import RetrievalIndex
from src.models import PipelineConfig
from src.pipeline import RetrievalPipeline
```
Thay `SYSTEMS` bằng:
```python
SYSTEMS = {
    "C2": PipelineConfig(sparse=False, dense=True, fusion="none", rerank=False),
    "C3-WS": PipelineConfig(fusion="weighted", alpha=0.5, rerank=False),
    "X2-R": PipelineConfig(fusion="adaptive", alpha=0.5, rerank=True),
}
```
Trong `generate`, thay `index = RetrievalIndex.load(config["index_dir"])` bằng `pipeline = RetrievalPipeline(RetrievalIndex.load(config["index_dir"]))`, và lời gọi `answer_question(query["text"], index, SYSTEMS[system_name], client, settings.openai_model)` bằng `answer_question(query["text"], pipeline, SYSTEMS[system_name], client, settings.openai_model)`.

- [ ] **Step 9: Dọn `tests/test_core.py`**

- Xóa `test_foundation_round_trip` (đã thay bằng `tests/test_storage.py`).
- Trong `test_retrieval_fusion_and_metrics`: xóa toàn bộ phần sau khối `metrics = evaluate_rankings(...)` và các assert của nó (phần RAG đã chuyển sang `tests/test_rag.py`); xóa phần `adaptive_alpha`, `fuse_weighted`, `minmax_scores` (đã có ở `tests/test_fusion.py`); đổi tên hàm thành `test_ranking_metrics`. Hàm còn lại:
```python
def test_ranking_metrics():
    metrics = evaluate_rankings(
        rankings={"q1": ["c1", "c2"]},
        qrels={"q1": {"c1": 2}},
        ks=(1, 10),
    )
    assert metrics["hit_rate@1"] == 1.0
    assert metrics["mrr@10"] == 1.0
```
- Import còn lại ở đầu file: `csv`, `json`, `pytest`, `yaml`, `evaluate_rankings`, `run_experiment`.

- [ ] **Step 10: Chạy toàn bộ test**

Run: `.venv/Scripts/python -m pytest -q`
Expected: tất cả pass; `grep -rn "src.retrieval\|RagConfig\|SearchResult" --include=*.py .` không còn kết quả nào ngoài `.venv`.

Run: `grep -rn "src.retrieval\|RagConfig\|SearchResult" --include=*.py src pages tests app.py run_*.py`
Expected: không có dòng nào.

- [ ] **Step 11: Commit**

```bash
git add -A src tests run_rag_evaluation.py
git commit -m "refactor: route RAG, storage and experiments through retrieval pipeline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: App Streamlit dùng pipeline mới

**Files:**
- Modify: `configs/default.yaml`, `pages/1_Chat.py`, `pages/2_Documents.py`, `pages/3_RAG_Settings.py`
- Create: `tests/test_pages.py`

**Interfaces:**
- Consumes: `build_corpus`, `extract_blocks`, `RetrievalIndex`, `RetrievalPipeline`, `PipelineConfig`, `Database.load_pipeline_configs`, `Settings.indexes_dir`.

- [ ] **Step 1: Viết test thất bại `tests/test_pages.py`**

```python
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.storage import Database

PAGES = Path(__file__).resolve().parent.parent / "pages"


def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("PPL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PPL_RUNS_DIR", str(tmp_path / "runs"))


def test_rag_settings_page_saves_pipeline_config(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    app = AppTest.from_file(str(PAGES / "3_RAG_Settings.py"), default_timeout=30)
    app.session_state["user"] = {"username": "admin", "role": "admin"}
    app.run()
    app.text_input(key="config_name").set_value("hybrid-rrf")
    app.selectbox(key="branches").set_value("Hybrid")
    app.selectbox(key="fusion").set_value("rrf")
    app.button(key="save_config").click()
    app.run()
    assert not app.exception
    configs, _ = Database(tmp_path / "data" / "app.db").load_pipeline_configs()
    assert configs["hybrid-rrf"].fusion == "rrf"


def test_rag_settings_page_shows_validation_error(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    app = AppTest.from_file(str(PAGES / "3_RAG_Settings.py"), default_timeout=30)
    app.session_state["user"] = {"username": "admin", "role": "admin"}
    app.run()
    app.number_input(key="rerank_n").set_value(5)
    app.number_input(key="context_k").set_value(10)
    app.button(key="save_config").click()
    app.run()
    assert any("context_k" in error.value for error in app.error)


def test_chat_page_without_active_corpus_shows_info(tmp_path, monkeypatch):
    _env(tmp_path, monkeypatch)
    app = AppTest.from_file(str(PAGES / "1_Chat.py"), default_timeout=30)
    app.session_state["user"] = {"username": "student", "role": "student"}
    app.run()
    assert not app.exception
    assert app.info
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_pages.py -q`
Expected: FAIL (trang RAG Settings chưa có key `config_name`/`branches`; trang Chat import `src.retrieval` đã xóa).

- [ ] **Step 3: Thay `configs/default.yaml`**

```yaml
chunking:
  strategy: structure
  max_words: 350
  overlap_words: 50
  chunk_words: 450
  fixed_overlap_words: 75
  prefix: true

index:
  embedding_model: BAAI/bge-m3
  tokenizers: [whitespace, pyvi]
  dense_backend: numpy
  bm25_k1: 1.5
  bm25_b: 0.75

pipeline:
  fusion: weighted
  alpha: 0.5
  rrf_k: 60
  adaptive_beta: 0.3
  rerank: true
  rerank_n: 30
  top_l: 100
  context_k: 5
  tokenizer: whitespace
  reranker_model: BAAI/bge-reranker-v2-m3
```

- [ ] **Step 4: Viết lại `pages/3_RAG_Settings.py`**

```python
import streamlit as st

from src.config import load_settings
from src.models import PipelineConfig
from src.text import TOKENIZERS
from src.ui import database, require_role

BRANCHES = {"BM25": (True, False), "Dense": (False, True), "Hybrid": (True, True)}

st.set_page_config(page_title="Cấu hình RAG", page_icon="⚙️")
require_role("admin")
settings = load_settings()
db = database()
st.title("Cấu hình RAG")

with st.form("rag_config"):
    name = st.text_input("Tên cấu hình", "hybrid-default", key="config_name")
    branch = st.selectbox("Nhánh truy xuất", list(BRANCHES), index=2, key="branches")
    fusion = st.selectbox("Dung hợp (khi Hybrid)", ["weighted", "rrf", "adaptive"], key="fusion")
    rerank = st.checkbox("Dùng reranker", True, key="rerank")
    tokenizer = st.selectbox("Tách từ BM25", list(TOKENIZERS), key="tokenizer")
    top_l = st.number_input("Top-L mỗi nhánh", 10, 500, 100, key="top_l")
    rerank_n = st.number_input("Top-N rerank", 1, 100, 30, key="rerank_n")
    context_k = st.number_input("Số chunk đưa vào context", 1, 20, 5, key="context_k")
    alpha = st.slider("Alpha (trọng số BM25)", 0.0, 1.0, 0.5, 0.05, key="alpha")
    rrf_k = st.number_input("RRF k", 1, 200, 60, key="rrf_k")
    adaptive_beta = st.slider("Beta (adaptive)", 0.0, 0.5, 0.3, 0.05, key="adaptive_beta")
    reranker_model = st.text_input("Reranker", "BAAI/bge-reranker-v2-m3", key="reranker_model")
    threshold = st.number_input("Ngưỡng từ chối", value=0.0, format="%.4f", key="threshold")
    llm_model = st.text_input("LLM model", settings.openai_model, key="llm_model")
    temperature = st.number_input("Temperature", 0.0, 2.0, 0.0, 0.1, key="temperature")
    submitted = st.form_submit_button("Lưu cấu hình", type="primary", key="save_config")

if submitted:
    sparse, dense = BRANCHES[branch]
    try:
        config = PipelineConfig(
            sparse=sparse,
            dense=dense,
            fusion=fusion if sparse and dense else "none",
            alpha=float(alpha),
            rrf_k=int(rrf_k),
            adaptive_beta=float(adaptive_beta),
            rerank=rerank,
            rerank_n=int(rerank_n),
            top_l=int(top_l),
            context_k=int(context_k),
            tokenizer=tokenizer,
            reranker_model=reranker_model.strip(),
            refusal_threshold=float(threshold),
            llm_model=llm_model.strip(),
            temperature=float(temperature),
        )
    except ValueError as error:
        st.error(f"Cấu hình không hợp lệ: {error}")
    else:
        if not name.strip():
            st.error("Tên cấu hình không được để trống.")
        else:
            db.save_rag_config(name.strip(), config)
            st.success("Đã lưu cấu hình.")

configs, invalid = db.load_pipeline_configs()
if invalid:
    st.warning(f"Bỏ qua cấu hình cũ không còn tương thích: {', '.join(invalid)}")
if configs:
    st.dataframe(
        [{"name": name, **config.to_dict()} for name, config in configs.items()],
        use_container_width=True,
    )
```

- [ ] **Step 5: Sửa `pages/1_Chat.py`**

Thay các import `from src.models import RagConfig` và `from src.retrieval import RetrievalIndex` bằng:
```python
from src.index import RetrievalIndex
from src.models import PipelineConfig
from src.pipeline import RetrievalPipeline
```
Thay khối từ `@st.cache_resource` tới dòng `client = OpenAI(...)` bằng:
```python
@st.cache_resource(show_spinner="Đang tải chỉ mục...")
def load_pipeline(path: str) -> RetrievalPipeline:
    return RetrievalPipeline(RetrievalIndex.load(path))


pipeline = load_pipeline(str(settings.indexes_dir / active["version_id"]))
config_by_name, invalid = db.load_pipeline_configs()
if invalid:
    st.sidebar.warning(f"Bỏ qua cấu hình cũ: {', '.join(invalid)}")
config_name = st.sidebar.selectbox("Cấu hình RAG", list(config_by_name) or ["Mặc định"])
config = config_by_name.get(config_name, PipelineConfig(llm_model=settings.openai_model))
client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)


def citation_label(citation: dict) -> str:
    source = " > ".join([citation.get("doc_title", ""), *citation.get("heading_path", [])]).strip(" >")
    return f"[{citation['number']}] {source or citation['doc_id']} — trang/slide {citation['page']}"
```
Thay cả hai chỗ `st.expander(f"[{citation['number']}] {citation['doc_id']} — trang/slide {citation['page']}")` bằng `st.expander(citation_label(citation))`. Thay khối gọi `answer_question` bằng:
```python
        with st.spinner("Đang tìm tài liệu và tạo câu trả lời..."):
            try:
                answer = answer_question(query, pipeline, config, client, config.llm_model or settings.openai_model)
            except ValueError as error:
                st.error(f"Không thể truy xuất với cấu hình '{config_name}': {error}")
                st.stop()
```

- [ ] **Step 6: Sửa `pages/2_Documents.py`**

Thay import:
```python
import json

import streamlit as st

from src.config import load_settings, load_yaml
from src.ingestion import build_corpus, extract_blocks
from src.index import RetrievalIndex
from src.models import Chunk
from src.ui import database, require_role, safe_upload_name
```
Sau `semester = st.text_input("Học kỳ")` thêm:
```python
strategy = st.selectbox("Chiến lược chunking", ["structure", "fixed"])
use_prefix = st.checkbox("Gắn tiêu đề (Tài liệu > Chương > Mục) vào đầu chunk", True)
```
Thay khối xem trước bằng:
```python
    if st.button("Xem trước trích xuất"):
        for path in paths:
            st.subheader(path.name)
            for block in extract_blocks(path)[:8]:
                if block.warning:
                    st.warning(f"Trang {block.page}: {block.warning}")
                st.caption(f"Trang/slide {block.page} — {' > '.join(block.heading_path) or '(chưa có tiêu đề)'}")
                st.text_area("Nội dung", block.text, height=120, disabled=True, key=f"{path.name}-{id(block)}")
```
Thay khối trong `with st.spinner(...)` bằng:
```python
            metadata = {
                str(path): {"course": course.strip(), "source_type": source_type, "semester": semester}
                for path in paths
            }
            chunking = {**defaults["chunking"], "strategy": strategy, "prefix": use_prefix}
            result = build_corpus(paths, metadata, settings, db, chunking)
            index_dir = settings.indexes_dir / result.version_id
            if not (index_dir / "index_meta.json").exists():
                chunks = [
                    Chunk.from_dict(json.loads(line))
                    for line in result.chunks_path.read_text(encoding="utf-8").splitlines()
                    if line
                ]
                index_options = defaults["index"]
                RetrievalIndex.build(
                    chunks,
                    index_dir,
                    embedding_model=index_options["embedding_model"],
                    tokenizers=index_options["tokenizers"],
                    dense_backend=index_options["dense_backend"],
                    bm25_k1=index_options["bm25_k1"],
                    bm25_b=index_options["bm25_b"],
                    chunking=chunking,
                )
            db.set_active_corpus(result.version_id)
```

- [ ] **Step 7: Chạy test trang và toàn bộ test**

Run: `.venv/Scripts/python -m pytest tests/test_pages.py -q`
Expected: `3 passed`

Run: `.venv/Scripts/python -m pytest -q && .venv/Scripts/python -m compileall -q app.py pages src run_experiments.py run_rag_evaluation.py`
Expected: tất cả pass, compileall không in lỗi.

- [ ] **Step 8: Kiểm tra thủ công app (cần mạng để tải bge-m3 lần đầu)**

Run: `.venv/Scripts/streamlit run app.py`
Expected: đăng nhập admin → Documents: tải 1 file DOCX có Heading → "Xem trước" hiện breadcrumb → "Tạo và kích hoạt chỉ mục" thành công → Chat: câu trả lời có trích dẫn dạng `Tài liệu > Chương > Mục — trang/slide N`. Ghi lại kết quả vào mô tả commit nếu có lỗi.

- [ ] **Step 9: Commit**

```bash
git add configs/default.yaml pages tests/test_pages.py
git commit -m "feat: wire Streamlit pages to the staged retrieval pipeline

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Tài liệu và kiểm tra toàn bộ

**Files:**
- Modify: `README.md`, `CLAUDE.md`

- [ ] **Step 1: Cập nhật `README.md`**

- Mục "Setup": thêm dòng Windows `uv pip install --python .venv/Scripts/python.exe -r requirements.txt --torch-backend cpu` và ghi chú "VnCoreNLP (tùy chọn) cần Java 8+; model tự tải về `vncorenlp/` hoặc `PPL_VNCORENLP_DIR`".
- Thêm mục "Data locations": `PPL_DATA_DIR` (mặc định `data/`), `PPL_RUNS_DIR` (mặc định `runs/`); index mới ở `data/indexes/<version>/` gồm `chunks.jsonl`, `embeddings.npy`, `tokens_<tokenizer>.json`, `index_meta.json`, tùy chọn `faiss.index`; **index tạo trước thay đổi này không tương thích — build lại từ trang Documents**.
- Mục "Run the application" bước 2: "chọn chiến lược chunking (mặc định structure + tiêu đề)"; bước 3: "chọn nhánh BM25/Dense/Hybrid, kiểu dung hợp, reranker".
- Mục "Verification": thay khối lệnh bằng
```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q app.py pages src run_experiments.py run_rag_evaluation.py
```
và xóa câu "The project intentionally has four behavior-level tests…".

- [ ] **Step 2: Cập nhật `CLAUDE.md`**

- Mục Commands: lệnh test thành `.venv/Scripts/python -m pytest -q` (Windows) / `.venv/bin/python -m pytest -q`; test đơn: `.venv/Scripts/python -m pytest tests/test_pipeline.py::test_rerank_reorders_head_and_keeps_tail -q`.
- Thay đoạn "Pipeline" trong Architecture bằng mô tả: `text` (NFC, tokenizers) → `ingestion.extract_blocks` (Block có `heading_path`; PDF dùng regex + cỡ chữ, bỏ header lặp) → `chunking.chunk_blocks` (structure/fixed, prefix breadcrumb, `chunk_id` băm) → `index.RetrievalIndex` (BM25 theo từng tokenizer + dense numpy/FAISS) → `pipeline.RetrievalPipeline.run` (sparse/dense → fusion none/rrf/weighted/adaptive → rerank top-N; trả `StageScores` từng tầng và `timings_ms`; cache SQLite ở `data/cache/retrieval.sqlite`) → `rag.answer_question`.
- Thay mục "Gotchas" bằng: test dùng `tests/fakes.py` (FakeEncoder/FakeCrossEncoder); fixture BM25 cần ≥ 4 chunk vì idf của `rank_bm25` ≤ 0 khi từ có mặt ở ≥ nửa số tài liệu; `PipelineConfig` validate chặt (fusion ≠ none cần cả hai nhánh, `context_k ≤ rerank_n ≤ top_l`); cấu hình RAG cũ trong DB bị bỏ qua; `src/experiments.py` là cầu nối tạm tới khi Kế hoạch 3 thay bằng `src/eval/`.

- [ ] **Step 3: Chạy kiểm tra cuối**

Run:
```bash
.venv/Scripts/python -m pytest -q
.venv/Scripts/python run_experiments.py --config configs/example_experiments.yaml --dry-run
.venv/Scripts/python -m compileall -q app.py pages src run_experiments.py run_rag_evaluation.py
```
Expected: pytest toàn bộ pass; dry-run in `{"status": "valid", "experiments": ["E0", ..., "E7"]}`; compileall không lỗi.

- [ ] **Step 4: Commit**

```bash
git add README.md CLAUDE.md
git commit -m "docs: describe staged retrieval pipeline and data locations

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
