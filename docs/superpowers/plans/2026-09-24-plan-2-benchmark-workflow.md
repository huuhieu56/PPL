# Kế hoạch 2/3: Quy trình xây dựng benchmark — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Công cụ CLI xây benchmark theo mục 2.4.3 của báo cáo: LLM sinh câu hỏi nháp 4 nhóm có kiểm tra tự động → người rà soát (+ câu hỏi do người viết) → pooling mù từ nhiều hệ thống → dán nhãn đôi, Cohen's κ có trọng số → qrels + evidence → chia dev/test phân tầng có manifest và khóa test → thống kê mô tả → ánh xạ qrels sang cách chunking khác.

**Architecture:** Package `src/bench/` gồm các module hàm thuần nhận/trả `list[dict]` (đọc/ghi file tách riêng trong `src/io_utils.py`), cộng `src/indexing.py` (build index từ thư mục) và `src/cli.py` (argparse, lệnh `index …`, `bench …`). Mọi file benchmark nằm trong một thư mục (mặc định `<data_dir>/benchmark/`).

**Tech Stack:** Python 3.11, numpy, OpenAI-compatible client (OpenRouter), pipeline từ Kế hoạch 1, pytest.

**Spec:** `docs/superpowers/specs/2026-09-24-hybrid-retrieval-evaluation-design.md` (mục 6). Yêu cầu Kế hoạch 1 đã hoàn tất (`RetrievalIndex`, `RetrievalPipeline`, `PipelineConfig`, `Chunk`, `normalize_text`, `tokenize`, `tests/fakes.py`).

## Global Constraints

- Mọi Global Constraints của Kế hoạch 1 vẫn áp dụng (Python 3.11, `.venv/Scripts/python -m pytest -q`, không tải model trong test, commit kèm `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`).
- Nhóm câu hỏi: `exact`, `concept`, `paraphrase`, `multi`. Nguồn: `origin ∈ {llm, human}`. Split: `dev`, `test` hoặc `null`.
- Ngưỡng: paraphrase Jaccard ≤ **0.2**; trùng lặp cosine ≥ **0.92**; ghép chunk multi cosine ≥ **0.6**; ánh xạ evidence cắt ngang ≥ **60%** số từ của quote.
- Pooling mặc định: depth **15**, hệ thống `C1, C2, C3-RRF, C3-WS, C4-WS, X1` với tham số mặc định.
- Nhãn qrels: `0/1/2`. κ: Cohen's κ **trọng số bậc hai**.
- Split: dev **0.3**, seed **42**, phân tầng theo `category`; query không có chunk liên quan (relevance ≥ 1) bị loại kèm cảnh báo.
- Khóa test: ghi sha256 của `frozen_params.yaml` vào `benchmark_manifest.json` lần chạy test đầu; về sau chỉ **cảnh báo + ghi `lock_violation`**, không chặn.
- Tên file cố định trong thư mục benchmark: `drafts.jsonl`, `rejected.jsonl`, `review.csv`, `queries.jsonl`, `duplicates.jsonl`, `pool_map.json`, `annotation_<tên>.csv`, `agreement.json`, `disagreements.csv`, `qrels.jsonl`, `evidence.jsonl`, `benchmark_manifest.json`, `benchmark_description.json`.
- CSV ghi bằng `utf-8-sig` (Excel mở đúng tiếng Việt), đọc bằng `utf-8-sig`.
- `src/cli.py` gọi `sys.stdout.reconfigure(encoding="utf-8")` để in tiếng Việt trên console Windows.

## Review Focus

1. **CSV đã mở/lưu bằng Excel** (có BOM, dòng trống cuối, ô số dạng `2.0`) — đọc lại phải đúng. Pin: `test_read_csv_handles_bom_and_excel_numbers` (Task 1) và `parse_relevance` chấp nhận `"2.0"` (Task 6).
2. **LLM trả lời không phải JSON thuần** (có ```json fence, lời dẫn) — vẫn trích được JSON; hoàn toàn không có JSON → vào `rejected.jsonl` với lý do `json`, không dừng cả lượt sinh. Pin: `test_parse_json_reply_handles_fences_and_garbage` (Task 3).
3. **Người dán nhãn để trống ô relevance hoặc nhập giá trị ngoài 0/1/2** — trống = chưa dán (không tính κ), giá trị lạ → lỗi nêu `pool_id`. Pin: `test_read_annotations_rejects_invalid_label` (Task 6).
4. **Chạy lại `bench pool` sau khi đã có file annotation** — không ghi đè nhãn đã dán. Pin: `test_write_pool_refuses_to_overwrite_annotations` (Task 5).
5. **Cửa sổ trích dẫn evidence không nằm trong chunk** (annotator gõ sai) — merge vẫn chạy nhưng trả cảnh báo nêu `pool_id`. Pin: trong `test_merge_labels_resolves_and_builds_evidence` (Task 6).

---

## File Structure

| File | Trách nhiệm |
|---|---|
| `src/io_utils.py` | `read_jsonl`, `write_jsonl`, `read_csv`, `write_csv`, `sha256_file` |
| `src/indexing.py` | `build_index_from_folder`, `resolve_index_dir` |
| `src/bench/__init__.py` | trống |
| `src/bench/checks.py` | `STOPWORDS`, `content_tokens`, `lexical_overlap`, `check_exact`, `check_paraphrase`, `contains_quote` |
| `src/bench/generate.py` | `CATEGORIES`, `sample_chunks`, `find_partners`, `build_prompt`, `parse_json_reply`, `generate_drafts` |
| `src/bench/review.py` | `export_review`, `import_review` |
| `src/bench/pool.py` | `DEFAULT_POOL_SYSTEMS`, `build_pool`, `write_pool` |
| `src/bench/agreement.py` | `weighted_kappa`, `parse_relevance`, `read_annotations`, `agreement_report`, `write_disagreements`, `merge_labels` |
| `src/bench/split.py` | `split_queries` |
| `src/bench/manifest.py` | `write_manifest`, `check_test_lock` |
| `src/bench/remap.py` | `remap_qrels` |
| `src/bench/describe.py` | `describe_benchmark` |
| `src/cli.py` | `main(argv)`: `index build`, `index add-tokenizer`, `bench *` |
| `tests/test_io_indexing.py`, `tests/test_bench_*.py`, `tests/test_cli_bench.py` | test |

---

### Task 1: `io_utils`, build index từ thư mục, khung CLI `index`

**Files:**
- Create: `src/io_utils.py`, `src/indexing.py`, `src/cli.py`, `tests/test_io_indexing.py`

**Interfaces:**
- Consumes: `build_corpus`, `DEFAULT_CHUNKING`, `RetrievalIndex`, `load_settings`, `load_yaml`, `Database`, `Chunk`.
- Produces:
  - `read_jsonl(path) -> list[dict]`, `write_jsonl(path, rows) -> None`, `read_csv(path) -> list[dict]`, `write_csv(path, rows, fieldnames: list[str] | None = None) -> None`, `sha256_file(path) -> str`
  - `build_index_from_folder(input_dir, *, course: str, settings, database, chunking: dict, index_options: dict, metadata_csv=None, activate: bool = True, encoder=None) -> tuple[str, Path]` (version_id, index_dir)
  - `resolve_index_dir(value: str | None, settings, database) -> Path` (None → index của corpus đang active)
  - `main(argv: list[str] | None = None) -> int` trong `src/cli.py`; `build_parser() -> argparse.ArgumentParser` (Kế hoạch 3 thêm nhóm `eval`)

- [ ] **Step 1: Viết test thất bại `tests/test_io_indexing.py`**

```python
import json

import pytest
from docx import Document

from src.cli import main
from src.config import load_settings
from src.indexing import build_index_from_folder, resolve_index_dir
from src.io_utils import read_csv, read_jsonl, sha256_file, write_csv, write_jsonl
from src.storage import Database
from tests.fakes import FakeEncoder

INDEX_OPTIONS = {
    "embedding_model": "fake",
    "tokenizers": ["whitespace"],
    "dense_backend": "numpy",
    "bm25_k1": 1.5,
    "bm25_b": 0.75,
}


def _env(tmp_path, monkeypatch):
    monkeypatch.setenv("PPL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PPL_RUNS_DIR", str(tmp_path / "runs"))
    settings = load_settings(tmp_path)
    db = Database(settings.db_path)
    db.initialize()
    return settings, db


def _docs(folder):
    folder.mkdir()
    for name, heading in (("giao_trinh.docx", "Chương 1"), ("de_thi.docx", "Đề 1")):
        document = Document()
        document.add_heading(heading, level=1)
        document.add_paragraph(f"Nội dung {heading} về khóa chính và khóa ngoại.")
        document.save(folder / name)


def test_jsonl_and_csv_round_trip(tmp_path):
    write_jsonl(tmp_path / "a.jsonl", [{"x": "Học", "y": [1, 2]}])
    assert read_jsonl(tmp_path / "a.jsonl") == [{"x": "Học", "y": [1, 2]}]
    write_csv(tmp_path / "a.csv", [{"a": "Tiếng Việt", "b": ""}])
    assert read_csv(tmp_path / "a.csv") == [{"a": "Tiếng Việt", "b": ""}]
    assert len(sha256_file(tmp_path / "a.csv")) == 64


def test_read_csv_handles_bom_and_excel_numbers(tmp_path):
    (tmp_path / "excel.csv").write_bytes("﻿pool_id,relevance\r\np1,2.0\r\n\r\n".encode("utf-8"))
    assert read_csv(tmp_path / "excel.csv") == [{"pool_id": "p1", "relevance": "2.0"}]


def test_build_index_from_folder_uses_metadata_and_activates(tmp_path, monkeypatch):
    settings, db = _env(tmp_path, monkeypatch)
    _docs(tmp_path / "docs")
    write_csv(tmp_path / "meta.csv", [{"filename": "de_thi.docx", "source_type": "exam", "doc_title": "Đề thi CSDL"}])
    version, index_dir = build_index_from_folder(
        tmp_path / "docs",
        course="CS101",
        settings=settings,
        database=db,
        chunking={"strategy": "structure"},
        index_options=INDEX_OPTIONS,
        metadata_csv=tmp_path / "meta.csv",
        encoder=FakeEncoder(),
    )
    assert index_dir == settings.indexes_dir / version
    assert db.get_active_corpus()["version_id"] == version
    chunks = read_jsonl(index_dir / "chunks.jsonl")
    titles = {chunk["doc_title"]: chunk["source_type"] for chunk in chunks}
    assert titles == {"Đề thi CSDL": "exam", "giao trinh": "textbook"}
    assert resolve_index_dir(None, settings, db) == index_dir
    assert resolve_index_dir(str(index_dir), settings, db) == index_dir


def test_resolve_index_dir_without_active_corpus_fails(tmp_path, monkeypatch):
    settings, db = _env(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="No active corpus"):
        resolve_index_dir(None, settings, db)


def test_cli_index_build_and_add_tokenizer(tmp_path, monkeypatch, capsys):
    settings, _ = _env(tmp_path, monkeypatch)
    monkeypatch.setattr("src.index.load_encoder", lambda name: FakeEncoder())
    _docs(tmp_path / "docs")
    assert main(["index", "build", "--input", str(tmp_path / "docs"), "--course", "CS101", "--tokenizers", "whitespace"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert main(["index", "add-tokenizer", "--index", output["index_dir"], "--tokenizer", "pyvi"]) == 0
    meta = json.loads((settings.indexes_dir / output["version_id"] / "index_meta.json").read_text(encoding="utf-8"))
    assert meta["tokenizers"] == ["whitespace", "pyvi"]
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_io_indexing.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.cli'`

- [ ] **Step 3: Viết `src/io_utils.py`**

```python
import csv
import hashlib
import json
from pathlib import Path


def read_jsonl(path: Path | str) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path | str, rows: list[dict]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8"
    )


def read_csv(path: Path | str) -> list[dict]:
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return [
            {key: (value or "") for key, value in row.items()}
            for row in csv.DictReader(stream)
            if any((value or "").strip() for value in row.values())
        ]


def write_csv(path: Path | str, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    columns = fieldnames or (list(rows[0]) if rows else [])
    with target.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path | str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
```

- [ ] **Step 4: Viết `src/indexing.py`**

```python
import json
from pathlib import Path

from src.index import RetrievalIndex
from src.ingestion import build_corpus
from src.io_utils import read_csv
from src.models import Chunk

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".pptx"}


def _default_source_type(path: Path) -> str:
    return "slide" if path.suffix.lower() == ".pptx" else "textbook"


def build_index_from_folder(
    input_dir,
    *,
    course: str,
    settings,
    database,
    chunking: dict,
    index_options: dict,
    metadata_csv=None,
    activate: bool = True,
    encoder=None,
) -> tuple[str, Path]:
    files = sorted(path for path in Path(input_dir).iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        raise ValueError(f"No PDF/DOCX/PPTX files in {input_dir}")
    overrides = {row["filename"]: row for row in read_csv(metadata_csv)} if metadata_csv else {}
    metadata = {}
    for path in files:
        row = overrides.get(path.name, {})
        metadata[str(path)] = {
            "course": row.get("course") or course,
            "source_type": row.get("source_type") or _default_source_type(path),
            **({"doc_title": row["doc_title"]} if row.get("doc_title") else {}),
        }
    result = build_corpus(files, metadata, settings, database, chunking)
    index_dir = settings.indexes_dir / result.version_id
    if not (index_dir / "index_meta.json").exists():
        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in result.chunks_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        RetrievalIndex.build(
            chunks,
            index_dir,
            embedding_model=index_options["embedding_model"],
            tokenizers=index_options["tokenizers"],
            dense_backend=index_options["dense_backend"],
            bm25_k1=index_options["bm25_k1"],
            bm25_b=index_options["bm25_b"],
            chunking=chunking,
            encoder=encoder,
        )
    if activate:
        database.set_active_corpus(result.version_id)
    return result.version_id, index_dir


def resolve_index_dir(value: str | None, settings, database) -> Path:
    if value:
        return Path(value)
    active = database.get_active_corpus()
    if not active:
        raise ValueError("No active corpus; pass --index or build one with `python -m src.cli index build`")
    return settings.indexes_dir / active["version_id"]
```

- [ ] **Step 5: Viết `src/cli.py`**

```python
import argparse
import json
import sys

from src.config import load_settings, load_yaml
from src.index import RetrievalIndex
from src.indexing import build_index_from_folder
from src.storage import Database


def _context():
    settings = load_settings()
    database = Database(settings.db_path)
    database.initialize()
    return settings, database


def _print(payload) -> None:
    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


def _index_build(args) -> int:
    settings, database = _context()
    defaults = load_yaml("configs/default.yaml")
    chunking = {**defaults["chunking"], "strategy": args.strategy, "prefix": not args.no_prefix}
    index_options = {**defaults["index"], "dense_backend": args.dense_backend}
    if args.tokenizers:
        index_options["tokenizers"] = [mode.strip() for mode in args.tokenizers.split(",") if mode.strip()]
    if args.embedding_model:
        index_options["embedding_model"] = args.embedding_model
    version, index_dir = build_index_from_folder(
        args.input,
        course=args.course,
        settings=settings,
        database=database,
        chunking=chunking,
        index_options=index_options,
        metadata_csv=args.metadata,
        activate=not args.no_activate,
    )
    _print({"version_id": version, "index_dir": str(index_dir)})
    return 0


def _index_add_tokenizer(args) -> int:
    RetrievalIndex.load(args.index).add_tokenizer(args.tokenizer)
    _print({"index_dir": args.index, "tokenizer": args.tokenizer})
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    groups = parser.add_subparsers(dest="group", required=True)

    index = groups.add_parser("index").add_subparsers(dest="command", required=True)
    build = index.add_parser("build")
    build.add_argument("--input", required=True)
    build.add_argument("--course", required=True)
    build.add_argument("--metadata")
    build.add_argument("--strategy", choices=["structure", "fixed"], default="structure")
    build.add_argument("--no-prefix", action="store_true")
    build.add_argument("--tokenizers")
    build.add_argument("--embedding-model")
    build.add_argument("--dense-backend", choices=["numpy", "faiss"], default="numpy")
    build.add_argument("--no-activate", action="store_true")
    build.set_defaults(handler=_index_build)
    add_tokenizer = index.add_parser("add-tokenizer")
    add_tokenizer.add_argument("--index", required=True)
    add_tokenizer.add_argument("--tokenizer", required=True, choices=["whitespace", "pyvi", "vncorenlp"])
    add_tokenizer.set_defaults(handler=_index_add_tokenizer)
    return parser


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 6: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_io_indexing.py -q`
Expected: `5 passed`

- [ ] **Step 7: Commit**

```bash
git add src/io_utils.py src/indexing.py src/cli.py tests/test_io_indexing.py
git commit -m "feat: add folder indexing CLI and shared file helpers

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Kiểm tra tự động cho câu hỏi (`bench/checks.py`)

**Files:**
- Create: `src/bench/__init__.py` (trống), `src/bench/checks.py`, `tests/test_bench_checks.py`

**Interfaces:**
- Produces:
  - `STOPWORDS: frozenset[str]`
  - `content_tokens(text: str) -> set[str]`
  - `lexical_overlap(question: str, source: str) -> float` (Jaccard trên content tokens)
  - `check_exact(question: str, source: str) -> str | None` (None = đạt; chuỗi = lý do bắt đầu `"exact:"`)
  - `check_paraphrase(question: str, source: str, max_overlap: float = 0.2) -> str | None` (lý do bắt đầu `"paraphrase:"`)
  - `contains_quote(quote: str, text: str) -> bool`

- [ ] **Step 1: Viết test thất bại `tests/test_bench_checks.py`**

```python
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
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_bench_checks.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.bench'`

- [ ] **Step 3: Viết `src/bench/__init__.py` (file rỗng) và `src/bench/checks.py`**

```python
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
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_bench_checks.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/bench/__init__.py src/bench/checks.py tests/test_bench_checks.py
git commit -m "feat: add automatic checks for generated benchmark questions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Sinh câu hỏi nháp bằng LLM (`bench/generate.py`)

**Files:**
- Create: `src/bench/generate.py`, `tests/test_bench_generate.py`

**Interfaces:**
- Consumes: `RetrievalIndex` (`chunk_list`, `chunk_ids`, `chunks`, `embeddings`), `checks.*`, `normalize_text`.
- Produces:
  - `CATEGORIES = ("exact", "concept", "paraphrase", "multi")`
  - `sample_chunks(chunks: list[Chunk], count: int, seed: int, min_words: int = 20) -> list[Chunk]`
  - `find_partners(chunk: Chunk, index, max_partners: int = 2, min_cosine: float = 0.6) -> list[Chunk]`
  - `build_prompt(category: str, sources: list[Chunk]) -> list[dict]` (messages)
  - `parse_json_reply(text: str) -> dict` (ValueError nếu không có JSON object)
  - `generate_drafts(index, client, model: str, per_category: int, seed: int = 42, categories=CATEGORIES, max_overlap: float = 0.2, min_words: int = 20) -> tuple[list[dict], list[dict]]` — (accepted, rejected)
  - Mỗi draft: `{"query_id", "text", "category", "origin": "llm", "split": None, "source_chunk_ids": [...], "evidence": [{"chunk_id", "quote"}], "generator": model}`; rejected thêm `"reasons": [...]`.

- [ ] **Step 1: Viết test thất bại `tests/test_bench_generate.py`**

```python
import json
import re
from types import SimpleNamespace

import pytest

from src.bench.generate import find_partners, generate_drafts, parse_json_reply, sample_chunks
from src.index import RetrievalIndex
from tests.fakes import FakeEncoder, make_chunk

BODIES = {
    "c1": "Học phần AI101 giới thiệu học máy có giám sát và không giám sát cho sinh viên năm nhất",
    "c2": "Mã CS202 là môn cơ sở dữ liệu với khóa chính khóa ngoại và chuẩn hóa lược đồ quan hệ",
    "c3": "Thuật toán ID3 xây dựng cây quyết định bằng độ lợi thông tin trên tập huấn luyện nhỏ",
    "c4": "Giao thức TCP cổng 80 đảm bảo truyền tin cậy nhờ cơ chế xác nhận và truyền lại gói tin",
}


@pytest.fixture
def index(tmp_path):
    chunks = [make_chunk(chunk_id, body, heading_path=("Chương 1",)) for chunk_id, body in BODIES.items()]
    return RetrievalIndex.build(chunks, tmp_path / "idx", embedding_model="fake", encoder=FakeEncoder())


class ScriptedClient:
    def __init__(self):
        self.chat = SimpleNamespace(completions=self)

    def create(self, model, messages, temperature, **kwargs):
        prompt = messages[-1]["content"]
        passages = re.findall(r"\[Đoạn \d+\][^\n]*\n(.+?)(?=\n\n\[Đoạn|\Z)", prompt, flags=re.DOTALL)
        first = passages[0].strip()
        quote = " ".join(first.split()[:5])
        if "nhiều đoạn" in prompt:
            payload = {
                "question": "So sánh hai nội dung trên?",
                "evidence_quotes": [" ".join(passage.split()[:5]) for passage in passages],
            }
        elif "BẮT BUỘC chứa nguyên văn" in prompt:
            code = next(word for word in first.split() if any(ch.isdigit() for ch in word))
            payload = {"question": f"{code} là gì?", "evidence_quote": quote}
        elif "KHÔNG dùng lại" in prompt:
            payload = {"question": first, "evidence_quote": quote}
        else:
            payload = {"question": "Khái niệm này có ý nghĩa ra sao?", "evidence_quote": quote}
        content = f"Đây là kết quả:\n```json\n{json.dumps(payload, ensure_ascii=False)}\n```"
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])


def test_parse_json_reply_handles_fences_and_garbage():
    assert parse_json_reply('```json\n{"question": "A?"}\n```') == {"question": "A?"}
    assert parse_json_reply('Kết quả: {"a": {"b": 1}} xong') == {"a": {"b": 1}}
    with pytest.raises(ValueError):
        parse_json_reply("không có JSON")


def test_sample_chunks_is_stratified_and_deterministic():
    chunks = [
        make_chunk(f"{doc}-{number}", "một hai ba bốn năm sáu bảy tám", doc_id=doc)
        for doc in ("d1", "d2")
        for number in range(3)
    ]
    picked = sample_chunks(chunks, 4, seed=7, min_words=5)
    assert len(picked) == 4
    assert {chunk.doc_id for chunk in picked} == {"d1", "d2"}
    assert picked == sample_chunks(chunks, 4, seed=7, min_words=5)
    assert sample_chunks(chunks, 4, seed=7, min_words=50) == []


def test_find_partners_prefers_same_document_section(index):
    partners = find_partners(index.chunks["c1"], index, max_partners=2)
    assert len(partners) == 2
    assert all(partner.chunk_id != "c1" for partner in partners)


def test_generate_drafts_validates_each_category(index):
    accepted, rejected = generate_drafts(index, ScriptedClient(), "fake-llm", per_category=1, seed=3, min_words=5)
    categories = sorted(draft["category"] for draft in accepted)
    assert categories == ["concept", "exact", "multi"]
    assert [row["category"] for row in rejected] == ["paraphrase"]
    assert rejected[0]["reasons"][0].startswith("paraphrase:")
    multi = next(draft for draft in accepted if draft["category"] == "multi")
    assert 2 <= len(multi["source_chunk_ids"]) <= 3
    assert len(multi["evidence"]) == len(multi["source_chunk_ids"])
    assert all(
        draft["origin"] == "llm" and draft["split"] is None and draft["generator"] == "fake-llm"
        for draft in accepted
    )
    again, _ = generate_drafts(index, ScriptedClient(), "fake-llm", per_category=1, seed=3, min_words=5)
    assert [draft["query_id"] for draft in again] == [draft["query_id"] for draft in accepted]


def test_generate_drafts_records_non_json_reply_as_rejected(index):
    class BrokenClient(ScriptedClient):
        def create(self, **kwargs):
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="xin lỗi"))])

    accepted, rejected = generate_drafts(
        index, BrokenClient(), "m", per_category=1, categories=("concept",), min_words=5
    )
    assert accepted == []
    assert rejected[0]["reasons"][0].startswith("json:")
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_bench_generate.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.bench.generate'`

- [ ] **Step 3: Viết `src/bench/generate.py`**

```python
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
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_bench_generate.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/bench/generate.py tests/test_bench_generate.py
git commit -m "feat: generate categorized draft questions with automatic validation

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Rà soát câu hỏi, câu hỏi do người viết, loại trùng (`bench/review.py`)

**Files:**
- Create: `src/bench/review.py`, `tests/test_bench_review.py`

**Interfaces:**
- Consumes: `write_csv`, `read_csv`, `CATEGORIES`, `normalize_text`, encoder có `.encode(texts, normalize_embeddings=True)`.
- Produces:
  - `REVIEW_COLUMNS = ["query_id", "category", "text", "source_text", "action", "new_text", "new_category"]`
  - `export_review(drafts: list[dict], index, path) -> None`
  - `import_review(review_path, drafts: list[dict], encoder, human_path=None, duplicate_threshold: float = 0.92) -> tuple[list[dict], list[dict]]` — (queries, duplicates); duplicates là draft kèm `"duplicate_of"`.
  - Câu hỏi của người: CSV cột `text`, `category`, tùy chọn `query_id`; nhận `origin="human"`, `generator="human"`, `source_chunk_ids=[]`, `evidence=[]`.

- [ ] **Step 1: Viết test thất bại `tests/test_bench_review.py`**

```python
import pytest

from src.bench.review import export_review, import_review
from src.io_utils import read_csv, write_csv
from tests.fakes import FakeEncoder, make_chunk


class TinyIndex:
    chunks = {"c1": make_chunk("c1", "Khóa chính"), "c2": make_chunk("c2", "Khóa ngoại")}


DRAFTS = [
    {"query_id": "q1", "text": "Khóa chính là gì?", "category": "concept", "origin": "llm", "split": None,
     "source_chunk_ids": ["c1"], "evidence": [], "generator": "m"},
    {"query_id": "q2", "text": "Khóa ngoại dùng làm gì?", "category": "concept", "origin": "llm", "split": None,
     "source_chunk_ids": ["c2"], "evidence": [], "generator": "m"},
    {"query_id": "q3", "text": "Câu hỏi tệ", "category": "exact", "origin": "llm", "split": None,
     "source_chunk_ids": ["c1"], "evidence": [], "generator": "m"},
]


def test_export_review_lists_sources_and_default_action(tmp_path):
    export_review(DRAFTS, TinyIndex(), tmp_path / "review.csv")
    rows = read_csv(tmp_path / "review.csv")
    assert [row["action"] for row in rows] == ["keep", "keep", "keep"]
    assert rows[1]["source_text"] == "Khóa ngoại"


def test_import_review_applies_actions_adds_humans_and_dedups(tmp_path):
    export_review(DRAFTS, TinyIndex(), tmp_path / "review.csv")
    rows = read_csv(tmp_path / "review.csv")
    rows[1].update(action="edit", new_text="Vai trò của khóa ngoại trong lược đồ?", new_category="paraphrase")
    rows[2].update(action="drop")
    write_csv(tmp_path / "review.csv", rows)
    write_csv(
        tmp_path / "human.csv",
        [
            {"text": "Khóa chính là gì?", "category": "concept"},
            {"text": "Làm sao chuẩn hóa bảng dữ liệu?", "category": "concept"},
        ],
    )

    queries, duplicates = import_review(tmp_path / "review.csv", DRAFTS, FakeEncoder(), tmp_path / "human.csv")

    assert [query["query_id"] for query in queries][:2] == ["q1", "q2"]
    assert queries[1]["text"] == "Vai trò của khóa ngoại trong lược đồ?"
    assert queries[1]["category"] == "paraphrase"
    assert queries[2]["origin"] == "human" and queries[2]["query_id"].startswith("h-")
    assert len(queries) == 3
    assert duplicates[0]["duplicate_of"] == "q1"


def test_import_review_rejects_unknown_action_and_category(tmp_path):
    export_review(DRAFTS[:1], TinyIndex(), tmp_path / "review.csv")
    rows = read_csv(tmp_path / "review.csv")
    rows[0]["action"] = "maybe"
    write_csv(tmp_path / "review.csv", rows)
    with pytest.raises(ValueError, match="q1"):
        import_review(tmp_path / "review.csv", DRAFTS, FakeEncoder())
    rows[0].update(action="edit", new_text="x", new_category="other")
    write_csv(tmp_path / "review.csv", rows)
    with pytest.raises(ValueError, match="category"):
        import_review(tmp_path / "review.csv", DRAFTS, FakeEncoder())
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_bench_review.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.bench.review'`

- [ ] **Step 3: Viết `src/bench/review.py`**

```python
import hashlib

import numpy as np

from src.bench.generate import CATEGORIES
from src.io_utils import read_csv, write_csv
from src.text import normalize_text

REVIEW_COLUMNS = ["query_id", "category", "text", "source_text", "action", "new_text", "new_category"]


def export_review(drafts: list[dict], index, path) -> None:
    rows = [
        {
            "query_id": draft["query_id"],
            "category": draft["category"],
            "text": draft["text"],
            "source_text": "\n---\n".join(index.chunks[chunk_id].body for chunk_id in draft["source_chunk_ids"]),
            "action": "keep",
            "new_text": "",
            "new_category": "",
        }
        for draft in drafts
    ]
    write_csv(path, rows, REVIEW_COLUMNS)


def _checked_category(value: str, where: str) -> str:
    if value not in CATEGORIES:
        raise ValueError(f"Unknown category '{value}' for {where}; expected one of {CATEGORIES}")
    return value


def _human_queries(path) -> list[dict]:
    queries = []
    for row in read_csv(path):
        text = normalize_text(row["text"])
        if not text:
            continue
        category = _checked_category(row["category"].strip(), f"human question '{text}'")
        query_id = row.get("query_id", "").strip() or "h-" + hashlib.sha256(text.encode("utf-8")).hexdigest()[:10]
        queries.append(
            {"query_id": query_id, "text": text, "category": category, "origin": "human", "split": None,
             "source_chunk_ids": [], "evidence": [], "generator": "human"}
        )
    return queries


def import_review(review_path, drafts: list[dict], encoder, human_path=None, duplicate_threshold: float = 0.92):
    by_id = {draft["query_id"]: draft for draft in drafts}
    reviewed = []
    for row in read_csv(review_path):
        query_id = row["query_id"]
        action = row["action"].strip().lower() or "keep"
        if action == "drop":
            continue
        if action not in ("keep", "edit"):
            raise ValueError(f"Unknown action '{action}' for {query_id}; use keep, edit or drop")
        query = dict(by_id[query_id])
        if action == "edit":
            new_text = normalize_text(row["new_text"])
            if not new_text:
                raise ValueError(f"Edited question {query_id} has empty new_text")
            query["text"] = new_text
            query["category"] = _checked_category(row["new_category"].strip() or query["category"], query_id)
        reviewed.append(query)
    if human_path:
        reviewed.extend(_human_queries(human_path))
    if not reviewed:
        return [], []
    vectors = np.asarray(encoder.encode([query["text"] for query in reviewed], normalize_embeddings=True))
    kept_rows: list[int] = []
    queries, duplicates = [], []
    for row, query in enumerate(reviewed):
        if kept_rows:
            similarities = vectors[kept_rows] @ vectors[row]
            best = int(np.argmax(similarities))
            if similarities[best] >= duplicate_threshold:
                duplicates.append({**query, "duplicate_of": reviewed[kept_rows[best]]["query_id"]})
                continue
        kept_rows.append(row)
        queries.append(query)
    return queries, duplicates
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_bench_review.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/bench/review.py tests/test_bench_review.py
git commit -m "feat: add question review import/export with human questions and dedup

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Pooling mù cho dán nhãn (`bench/pool.py`)

**Files:**
- Create: `src/bench/pool.py`, `tests/test_bench_pool.py`

**Interfaces:**
- Consumes: `RetrievalPipeline.run`, `PipelineConfig`, `write_csv`, `read_csv`.
- Produces:
  - `DEFAULT_POOL_SYSTEMS: dict[str, PipelineConfig]` (C1, C2, C3-RRF, C3-WS, C4-WS, X1)
  - `ANNOTATION_COLUMNS = ["pool_id", "query_id", "query", "heading_path", "chunk_text", "relevance", "evidence_quote"]`
  - `build_pool(queries, pipeline, systems: dict[str, PipelineConfig], depth: int = 15, manual_additions: list[dict] | None = None) -> list[dict]` — mỗi entry `{"pool_id", "query_id", "chunk_id", "doc_id", "page", "systems": [sorted names]}`; tên đặc biệt `"source"` (chunk nguồn của draft) và `"manual"`.
  - `write_pool(pool, queries, index, out_dir, annotators=("A", "B"), seed: int = 42) -> list[Path]` — ghi `pool_map.json` và `annotation_<tên>.csv` (xáo trộn trong từng query theo seed); ValueError nếu file annotation đã tồn tại.

- [ ] **Step 1: Viết test thất bại `tests/test_bench_pool.py`**

```python
import json

import pytest

from src.bench.pool import build_pool, write_pool
from src.index import RetrievalIndex
from src.io_utils import read_csv
from src.models import PipelineConfig
from src.pipeline import RetrievalPipeline
from tests.fakes import FakeEncoder, make_chunk

TEXTS = {
    "c1": "Mã môn AI101 là học phần nhập môn trí tuệ nhân tạo",
    "c2": "Học máy là lĩnh vực nghiên cứu thuật toán học từ dữ liệu",
    "c3": "Cơ sở dữ liệu quan hệ lưu trữ bảng và khóa chính",
    "c4": "Mạng nơ ron sâu gồm nhiều tầng biến đổi phi tuyến",
}
SYSTEMS = {
    "C1": PipelineConfig(dense=False, fusion="none", rerank=False, top_l=10),
    "C2": PipelineConfig(sparse=False, fusion="none", rerank=False, top_l=10),
}
QUERIES = [{"query_id": "q1", "text": "mã môn AI101", "source_chunk_ids": ["c3"]}]


@pytest.fixture
def index(tmp_path):
    chunks = [make_chunk(cid, text, heading_path=("Bài 1",), page=2) for cid, text in TEXTS.items()]
    return RetrievalIndex.build(chunks, tmp_path / "idx", embedding_model="fake", encoder=FakeEncoder())


def test_build_pool_merges_systems_source_and_manual(index):
    pipeline = RetrievalPipeline(index, synchronize=lambda: None)
    pool = build_pool(QUERIES, pipeline, SYSTEMS, depth=2, manual_additions=[{"query_id": "q1", "chunk_id": "c4"}])
    by_chunk = {entry["chunk_id"]: entry for entry in pool}
    assert "C1" in by_chunk["c1"]["systems"] and "C2" in by_chunk["c1"]["systems"]
    assert "source" in by_chunk["c3"]["systems"]
    assert "manual" in by_chunk["c4"]["systems"]
    assert by_chunk["c1"]["page"] == 2 and by_chunk["c1"]["doc_id"] == "doc-1"
    assert len({entry["pool_id"] for entry in pool}) == len(pool)


def test_build_pool_rejects_unknown_manual_chunk(index):
    pipeline = RetrievalPipeline(index, synchronize=lambda: None)
    with pytest.raises(ValueError, match="c99"):
        build_pool(QUERIES, pipeline, SYSTEMS, depth=2, manual_additions=[{"query_id": "q1", "chunk_id": "c99"}])


def test_write_pool_creates_blind_shuffled_annotation_files(tmp_path, index):
    pipeline = RetrievalPipeline(index, synchronize=lambda: None)
    pool = build_pool(QUERIES, pipeline, SYSTEMS, depth=4)
    paths = write_pool(pool, QUERIES, index, tmp_path / "bench", annotators=("A", "B"), seed=1)
    assert [path.name for path in paths] == ["annotation_A.csv", "annotation_B.csv"]
    rows = read_csv(tmp_path / "bench" / "annotation_A.csv")
    assert len(rows) == len(pool)
    assert "systems" not in rows[0] and rows[0]["relevance"] == ""
    assert rows[0]["heading_path"] == "Tài liệu > Bài 1"
    saved = json.loads((tmp_path / "bench" / "pool_map.json").read_text(encoding="utf-8"))
    assert {entry["pool_id"] for entry in saved} == {row["pool_id"] for row in rows}
    assert [row["pool_id"] for row in rows] == [
        row["pool_id"] for row in read_csv(tmp_path / "bench" / "annotation_B.csv")
    ]


def test_write_pool_refuses_to_overwrite_annotations(tmp_path, index):
    pipeline = RetrievalPipeline(index, synchronize=lambda: None)
    pool = build_pool(QUERIES, pipeline, SYSTEMS, depth=2)
    write_pool(pool, QUERIES, index, tmp_path / "bench")
    with pytest.raises(ValueError, match="already exists"):
        write_pool(pool, QUERIES, index, tmp_path / "bench")
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_bench_pool.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.bench.pool'`

- [ ] **Step 3: Viết `src/bench/pool.py`**

```python
import hashlib
import json
import random
from pathlib import Path

from src.io_utils import write_csv
from src.models import PipelineConfig

DEFAULT_POOL_SYSTEMS = {
    "C1": PipelineConfig(dense=False, fusion="none", rerank=False),
    "C2": PipelineConfig(sparse=False, fusion="none", rerank=False),
    "C3-RRF": PipelineConfig(fusion="rrf", rerank=False),
    "C3-WS": PipelineConfig(fusion="weighted", rerank=False),
    "C4-WS": PipelineConfig(fusion="weighted", rerank=True),
    "X1": PipelineConfig(sparse=False, fusion="none", rerank=True),
}
ANNOTATION_COLUMNS = ["pool_id", "query_id", "query", "heading_path", "chunk_text", "relevance", "evidence_quote"]


def build_pool(queries, pipeline, systems: dict, depth: int = 15, manual_additions: list[dict] | None = None) -> list[dict]:
    chunks = pipeline.index.chunks
    found: dict[tuple[str, str], set[str]] = {}
    for query in queries:
        for name, config in systems.items():
            for item in pipeline.run(query["text"], config).results[:depth]:
                found.setdefault((query["query_id"], item.chunk.chunk_id), set()).add(name)
        for chunk_id in query.get("source_chunk_ids", []):
            found.setdefault((query["query_id"], chunk_id), set()).add("source")
    for addition in manual_additions or []:
        if addition["chunk_id"] not in chunks:
            raise ValueError(f"Manual addition refers to unknown chunk {addition['chunk_id']}")
        found.setdefault((addition["query_id"], addition["chunk_id"]), set()).add("manual")
    return [
        {
            "pool_id": hashlib.sha256(f"{query_id}|{chunk_id}".encode("utf-8")).hexdigest()[:12],
            "query_id": query_id,
            "chunk_id": chunk_id,
            "doc_id": chunks[chunk_id].doc_id,
            "page": chunks[chunk_id].page,
            "systems": sorted(names),
        }
        for (query_id, chunk_id), names in sorted(found.items())
    ]


def write_pool(pool, queries, index, out_dir, annotators=("A", "B"), seed: int = 42) -> list[Path]:
    target = Path(out_dir)
    paths = [target / f"annotation_{name}.csv" for name in annotators]
    existing = [path for path in paths if path.exists()]
    if existing:
        raise ValueError(f"Annotation file already exists: {existing[0]}; move it away before re-pooling")
    target.mkdir(parents=True, exist_ok=True)
    (target / "pool_map.json").write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
    texts = {query["query_id"]: query["text"] for query in queries}
    rng = random.Random(seed)
    ordered = []
    for query in queries:
        entries = [entry for entry in pool if entry["query_id"] == query["query_id"]]
        rng.shuffle(entries)
        ordered.extend(entries)
    rows = [
        {
            "pool_id": entry["pool_id"],
            "query_id": entry["query_id"],
            "query": texts[entry["query_id"]],
            "heading_path": index.chunks[entry["chunk_id"]].breadcrumb(),
            "chunk_text": index.chunks[entry["chunk_id"]].body,
            "relevance": "",
            "evidence_quote": "",
        }
        for entry in ordered
    ]
    for path in paths:
        write_csv(path, rows, ANNOTATION_COLUMNS)
    return paths
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_bench_pool.py -q`
Expected: `4 passed`

- [ ] **Step 5: Commit**

```bash
git add src/bench/pool.py tests/test_bench_pool.py
git commit -m "feat: build blind multi-system relevance pools for annotators

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Độ đồng thuận và hợp nhất nhãn (`bench/agreement.py`)

**Files:**
- Create: `src/bench/agreement.py`, `tests/test_bench_agreement.py`

**Interfaces:**
- Consumes: `read_csv`, `write_csv`, `contains_quote`.
- Produces:
  - `weighted_kappa(first: list[int], second: list[int], labels=(0, 1, 2)) -> float`
  - `parse_relevance(value: str, pool_id: str) -> int | None` (`""` → None; chấp nhận `"2"`, `"2.0"`; khác → ValueError nêu pool_id)
  - `read_annotations(path) -> dict[str, dict]` — `{pool_id: {"relevance": int|None, "evidence_quote": str}}`
  - `agreement_report(first: dict, second: dict) -> dict` — `{"overlap", "kappa", "raw_agreement", "disagreements": [pool_id]}`
  - `write_disagreements(path, report, pool, first, second, queries, index) -> None` (cột `pool_id, query, chunk_text, label_a, label_b, final`)
  - `merge_labels(pool: list[dict], annotations: list[dict], index, resolved_path=None) -> tuple[list[dict], list[dict], list[str]]` — (qrels, evidence, warnings); ValueError nếu còn bất đồng chưa giải quyết.

- [ ] **Step 1: Viết test thất bại `tests/test_bench_agreement.py`**

```python
import pytest

from src.bench.agreement import (
    agreement_report,
    merge_labels,
    parse_relevance,
    read_annotations,
    weighted_kappa,
    write_disagreements,
)
from src.io_utils import read_csv, write_csv
from tests.fakes import make_chunk


class TinyIndex:
    chunks = {
        "c1": make_chunk("c1", "Khóa chính xác định duy nhất bản ghi", page=4),
        "c2": make_chunk("c2", "Khóa ngoại tham chiếu bảng khác", page=5),
        "c3": make_chunk("c3", "Chuẩn hóa loại bỏ dư thừa", page=6),
    }


POOL = [
    {"pool_id": "p1", "query_id": "q1", "chunk_id": "c1", "doc_id": "doc-1", "page": 4, "systems": ["C1"]},
    {"pool_id": "p2", "query_id": "q1", "chunk_id": "c2", "doc_id": "doc-1", "page": 5, "systems": ["C2"]},
    {"pool_id": "p3", "query_id": "q1", "chunk_id": "c3", "doc_id": "doc-1", "page": 6, "systems": ["C2"]},
]
QUERIES = [{"query_id": "q1", "text": "Khóa chính là gì?"}]


def _annotation(path, labels, quotes=None):
    quotes = quotes or {}
    write_csv(
        path,
        [{"pool_id": pid, "relevance": label, "evidence_quote": quotes.get(pid, "")} for pid, label in labels.items()],
    )
    return read_annotations(path)


def test_weighted_kappa_matches_hand_computation():
    assert weighted_kappa([0, 1, 2, 2], [0, 1, 2, 1]) == pytest.approx(0.8)
    assert weighted_kappa([1, 1], [1, 1]) == 1.0
    assert weighted_kappa([0, 2], [2, 0]) < 0


def test_parse_relevance_accepts_excel_numbers():
    assert parse_relevance("2.0", "p1") == 2
    assert parse_relevance(" ", "p1") is None


def test_read_annotations_rejects_invalid_label(tmp_path):
    write_csv(tmp_path / "a.csv", [{"pool_id": "p9", "relevance": "3", "evidence_quote": ""}])
    with pytest.raises(ValueError, match="p9"):
        read_annotations(tmp_path / "a.csv")


def test_agreement_report_uses_only_double_labeled_items(tmp_path):
    first = _annotation(tmp_path / "a.csv", {"p1": "2", "p2": "0", "p3": "1"})
    second = _annotation(tmp_path / "b.csv", {"p1": "2", "p2": "1", "p3": ""})
    report = agreement_report(first, second)
    assert report["overlap"] == 2
    assert report["raw_agreement"] == 0.5
    assert report["disagreements"] == ["p2"]
    write_disagreements(tmp_path / "dis.csv", report, POOL, first, second, QUERIES, TinyIndex())
    rows = read_csv(tmp_path / "dis.csv")
    assert rows == [{"pool_id": "p2", "query": "Khóa chính là gì?", "chunk_text": "Khóa ngoại tham chiếu bảng khác",
                     "label_a": "0", "label_b": "1", "final": ""}]


def test_merge_labels_resolves_and_builds_evidence(tmp_path):
    first = _annotation(
        tmp_path / "a.csv",
        {"p1": "2", "p2": "0", "p3": "1"},
        {"p1": "xác định duy nhất bản ghi", "p3": "câu không có trong chunk"},
    )
    second = _annotation(tmp_path / "b.csv", {"p1": "2", "p2": "1", "p3": ""})
    with pytest.raises(ValueError, match="p2"):
        merge_labels(POOL, [first, second], TinyIndex())

    write_csv(tmp_path / "dis.csv", [{"pool_id": "p2", "final": "0"}])
    qrels, evidence, warnings = merge_labels(POOL, [first, second], TinyIndex(), tmp_path / "dis.csv")

    assert qrels == [
        {"query_id": "q1", "chunk_id": "c1", "relevance": 2},
        {"query_id": "q1", "chunk_id": "c2", "relevance": 0},
        {"query_id": "q1", "chunk_id": "c3", "relevance": 1},
    ]
    assert evidence == [
        {"query_id": "q1", "doc_id": "doc-1", "page": 4, "quote": "xác định duy nhất bản ghi", "relevance": 2}
    ]
    assert len(warnings) == 1 and "p3" in warnings[0]
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_bench_agreement.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.bench.agreement'`

- [ ] **Step 3: Viết `src/bench/agreement.py`**

```python
import numpy as np

from src.bench.checks import contains_quote
from src.io_utils import read_csv, write_csv
from src.text import normalize_text


def weighted_kappa(first: list[int], second: list[int], labels=(0, 1, 2)) -> float:
    if len(first) != len(second) or not first:
        raise ValueError("kappa needs two non-empty label lists of equal length")
    position = {label: row for row, label in enumerate(labels)}
    size = len(labels)
    observed = np.zeros((size, size))
    for left, right in zip(first, second):
        observed[position[left], position[right]] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    grid = np.arange(size)
    weights = (grid[:, None] - grid[None, :]) ** 2 / (size - 1) ** 2
    expected_disagreement = float((weights * expected).sum())
    observed_disagreement = float((weights * observed).sum())
    if expected_disagreement == 0:
        return 1.0 if observed_disagreement == 0 else 0.0
    return 1.0 - observed_disagreement / expected_disagreement


def parse_relevance(value: str, pool_id: str) -> int | None:
    text = (value or "").strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        number = -1.0
    if number not in (0.0, 1.0, 2.0):
        raise ValueError(f"Invalid relevance '{value}' for pool_id {pool_id}; use 0, 1 or 2")
    return int(number)


def read_annotations(path) -> dict[str, dict]:
    return {
        row["pool_id"]: {
            "relevance": parse_relevance(row.get("relevance", ""), row["pool_id"]),
            "evidence_quote": normalize_text(row.get("evidence_quote", "")),
        }
        for row in read_csv(path)
    }


def agreement_report(first: dict, second: dict) -> dict:
    shared = sorted(
        pool_id
        for pool_id in set(first) & set(second)
        if first[pool_id]["relevance"] is not None and second[pool_id]["relevance"] is not None
    )
    if not shared:
        return {"overlap": 0, "kappa": None, "raw_agreement": None, "disagreements": []}
    left = [first[pool_id]["relevance"] for pool_id in shared]
    right = [second[pool_id]["relevance"] for pool_id in shared]
    return {
        "overlap": len(shared),
        "kappa": weighted_kappa(left, right),
        "raw_agreement": sum(a == b for a, b in zip(left, right)) / len(shared),
        "disagreements": [pool_id for pool_id, a, b in zip(shared, left, right) if a != b],
    }


def write_disagreements(path, report, pool, first, second, queries, index) -> None:
    entries = {entry["pool_id"]: entry for entry in pool}
    texts = {query["query_id"]: query["text"] for query in queries}
    rows = [
        {
            "pool_id": pool_id,
            "query": texts[entries[pool_id]["query_id"]],
            "chunk_text": index.chunks[entries[pool_id]["chunk_id"]].body,
            "label_a": first[pool_id]["relevance"],
            "label_b": second[pool_id]["relevance"],
            "final": "",
        }
        for pool_id in report["disagreements"]
    ]
    write_csv(path, rows, ["pool_id", "query", "chunk_text", "label_a", "label_b", "final"])


def merge_labels(pool: list[dict], annotations: list[dict], index, resolved_path=None):
    resolved = {}
    if resolved_path:
        for row in read_csv(resolved_path):
            final = parse_relevance(row.get("final", ""), row["pool_id"])
            if final is not None:
                resolved[row["pool_id"]] = final
    qrels, evidence, warnings, unresolved = [], [], [], []
    for entry in pool:
        pool_id = entry["pool_id"]
        labels = [item[pool_id]["relevance"] for item in annotations if pool_id in item and item[pool_id]["relevance"] is not None]
        if pool_id in resolved:
            final = resolved[pool_id]
        elif labels and len(set(labels)) == 1:
            final = labels[0]
        elif labels:
            unresolved.append(pool_id)
            continue
        else:
            continue
        qrels.append({"query_id": entry["query_id"], "chunk_id": entry["chunk_id"], "relevance": final})
        if final < 1:
            continue
        quote = next((item[pool_id]["evidence_quote"] for item in annotations if pool_id in item and item[pool_id]["evidence_quote"]), "")
        if not quote:
            continue
        if not contains_quote(quote, index.chunks[entry["chunk_id"]].body):
            warnings.append(f"Evidence quote for pool_id {pool_id} is not found in chunk {entry['chunk_id']}")
            continue
        evidence.append(
            {"query_id": entry["query_id"], "doc_id": entry["doc_id"], "page": entry["page"], "quote": quote, "relevance": final}
        )
    if unresolved:
        raise ValueError(f"Unresolved disagreements (fill 'final' in disagreements.csv): {', '.join(unresolved)}")
    return qrels, evidence, warnings
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_bench_agreement.py -q`
Expected: `5 passed`

- [ ] **Step 5: Commit**

```bash
git add src/bench/agreement.py tests/test_bench_agreement.py
git commit -m "feat: compute weighted kappa and merge adjudicated relevance labels

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Chia dev/test, manifest và khóa tập test (`bench/split.py`, `bench/manifest.py`)

**Files:**
- Create: `src/bench/split.py`, `src/bench/manifest.py`, `tests/test_bench_split.py`

**Interfaces:**
- Consumes: `sha256_file`.
- Produces:
  - `split_queries(queries, qrels, dev_ratio: float = 0.3, seed: int = 42) -> tuple[list[dict], list[str]]` — (queries có `split`, query_id bị loại vì không có chunk liên quan)
  - `MANIFEST_NAME = "benchmark_manifest.json"`
  - `write_manifest(bench_dir, index_version: str, seed: int, dev_ratio: float) -> dict`
  - `check_test_lock(bench_dir, frozen_params_path) -> dict` — `{"lock_violation": bool, "benchmark_changed": bool, "frozen_params_sha256": str}`; FileNotFoundError nếu thiếu `frozen_params.yaml` hoặc manifest.

- [ ] **Step 1: Viết test thất bại `tests/test_bench_split.py`**

```python
import json

import pytest

from src.bench.manifest import MANIFEST_NAME, check_test_lock, write_manifest
from src.bench.split import split_queries
from src.io_utils import write_jsonl

QUERIES = [{"query_id": f"e{i}", "category": "exact"} for i in range(6)] + [
    {"query_id": f"k{i}", "category": "concept"} for i in range(4)
]
QRELS = [{"query_id": query["query_id"], "chunk_id": "c", "relevance": 1} for query in QUERIES if query["query_id"] != "k3"]
QRELS.append({"query_id": "k3", "chunk_id": "c", "relevance": 0})


def test_split_is_stratified_deterministic_and_drops_unanswerable():
    queries, dropped = split_queries(QUERIES, QRELS, dev_ratio=0.3, seed=42)
    assert dropped == ["k3"]
    assert len(queries) == 9
    dev = [query for query in queries if query["split"] == "dev"]
    assert sorted(query["category"] for query in dev) == ["concept", "exact", "exact"]
    again, _ = split_queries(QUERIES, QRELS, dev_ratio=0.3, seed=42)
    assert [query["split"] for query in again] == [query["split"] for query in queries]


def test_test_lock_records_first_run_and_flags_later_changes(tmp_path):
    write_jsonl(tmp_path / "queries.jsonl", [{"query_id": "e0"}])
    write_jsonl(tmp_path / "qrels.jsonl", [{"query_id": "e0", "chunk_id": "c", "relevance": 1}])
    manifest = write_manifest(tmp_path, index_version="v1", seed=42, dev_ratio=0.3)
    assert set(manifest["files"]) == {"queries.jsonl", "qrels.jsonl"}
    frozen = tmp_path / "frozen_params.yaml"
    frozen.write_text("alpha: 0.6\n", encoding="utf-8")

    first = check_test_lock(tmp_path, frozen)
    assert first["lock_violation"] is False and first["benchmark_changed"] is False
    assert check_test_lock(tmp_path, frozen)["lock_violation"] is False

    frozen.write_text("alpha: 0.7\n", encoding="utf-8")
    assert check_test_lock(tmp_path, frozen)["lock_violation"] is True
    saved = json.loads((tmp_path / MANIFEST_NAME).read_text(encoding="utf-8"))
    assert saved["test_lock"]["frozen_params_sha256"] == first["frozen_params_sha256"]

    write_jsonl(tmp_path / "qrels.jsonl", [{"query_id": "e0", "chunk_id": "c", "relevance": 2}])
    assert check_test_lock(tmp_path, frozen)["benchmark_changed"] is True


def test_test_lock_requires_frozen_params(tmp_path):
    write_jsonl(tmp_path / "queries.jsonl", [])
    write_jsonl(tmp_path / "qrels.jsonl", [])
    write_manifest(tmp_path, index_version="v1", seed=42, dev_ratio=0.3)
    with pytest.raises(FileNotFoundError, match="eval tune"):
        check_test_lock(tmp_path, tmp_path / "frozen_params.yaml")
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_bench_split.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.bench.manifest'`

- [ ] **Step 3: Viết `src/bench/split.py`**

```python
import random
from collections import defaultdict


def split_queries(queries, qrels, dev_ratio: float = 0.3, seed: int = 42) -> tuple[list[dict], list[str]]:
    answerable = {row["query_id"] for row in qrels if int(row["relevance"]) >= 1}
    dropped = [query["query_id"] for query in queries if query["query_id"] not in answerable]
    groups = defaultdict(list)
    for query in queries:
        if query["query_id"] in answerable:
            groups[query["category"]].append(query["query_id"])
    rng = random.Random(seed)
    dev_ids = set()
    for category in sorted(groups):
        members = sorted(groups[category])
        rng.shuffle(members)
        dev_ids.update(members[: round(len(members) * dev_ratio)])
    kept = [
        {**query, "split": "dev" if query["query_id"] in dev_ids else "test"}
        for query in queries
        if query["query_id"] in answerable
    ]
    return kept, dropped
```

- [ ] **Step 4: Viết `src/bench/manifest.py`**

```python
import json
from datetime import datetime, timezone
from pathlib import Path

from src.io_utils import sha256_file

MANIFEST_NAME = "benchmark_manifest.json"
TRACKED_FILES = ("queries.jsonl", "qrels.jsonl", "evidence.jsonl")


def _file_hashes(bench_dir: Path) -> dict[str, str]:
    return {name: sha256_file(bench_dir / name) for name in TRACKED_FILES if (bench_dir / name).exists()}


def write_manifest(bench_dir, index_version: str, seed: int, dev_ratio: float) -> dict:
    directory = Path(bench_dir)
    manifest = {
        "index_version": index_version,
        "seed": seed,
        "dev_ratio": dev_ratio,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "files": _file_hashes(directory),
    }
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def check_test_lock(bench_dir, frozen_params_path) -> dict:
    directory = Path(bench_dir)
    frozen = Path(frozen_params_path)
    if not frozen.exists():
        raise FileNotFoundError(f"{frozen} not found; run `python -m src.cli eval tune` on the dev split first")
    manifest_path = directory / MANIFEST_NAME
    if not manifest_path.exists():
        raise FileNotFoundError(f"{manifest_path} not found; run `python -m src.cli bench split` first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    digest = sha256_file(frozen)
    if "test_lock" not in manifest:
        manifest["test_lock"] = {"frozen_params_sha256": digest, "locked_at": datetime.now(timezone.utc).isoformat()}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "lock_violation": manifest["test_lock"]["frozen_params_sha256"] != digest,
        "benchmark_changed": _file_hashes(directory) != manifest["files"],
        "frozen_params_sha256": digest,
    }
```

- [ ] **Step 5: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_bench_split.py -q`
Expected: `3 passed`

- [ ] **Step 6: Commit**

```bash
git add src/bench/split.py src/bench/manifest.py tests/test_bench_split.py
git commit -m "feat: add stratified dev/test split, benchmark manifest and test lock

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Ánh xạ qrels sang cách chunking khác (`bench/remap.py`)

**Files:**
- Create: `src/bench/remap.py`, `tests/test_bench_remap.py`

**Interfaces:**
- Consumes: `normalize_text`.
- Produces: `remap_qrels(evidence: list[dict], chunks: list[Chunk], min_fraction: float = 0.6) -> list[dict]` — qrels `{"query_id", "chunk_id", "relevance"}` sắp theo (query_id, chunk_id); mỗi cặp lấy mức cao nhất.

- [ ] **Step 1: Viết test thất bại `tests/test_bench_remap.py`**

```python
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
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_bench_remap.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.bench.remap'`

- [ ] **Step 3: Viết `src/bench/remap.py`**

```python
from collections import defaultdict

from src.text import normalize_text


def _words(text: str) -> list[str]:
    return normalize_text(text).lower().split()


def _longest_common_run(needle: list[str], haystack: list[str]) -> int:
    best = 0
    previous = [0] * (len(haystack) + 1)
    for left in needle:
        current = [0] * (len(haystack) + 1)
        for position, right in enumerate(haystack, start=1):
            if left == right:
                current[position] = previous[position - 1] + 1
                best = max(best, current[position])
        previous = current
    return best


def remap_qrels(evidence: list[dict], chunks, min_fraction: float = 0.6) -> list[dict]:
    by_doc = defaultdict(list)
    for chunk in chunks:
        by_doc[chunk.doc_id].append((chunk.chunk_id, _words(chunk.body)))
    grades: dict[tuple[str, str], int] = {}
    for item in evidence:
        quote = _words(item["quote"])
        if not quote:
            continue
        needed = min_fraction * len(quote)
        joined_quote = " ".join(quote)
        for chunk_id, words in by_doc[item["doc_id"]]:
            if joined_quote in " ".join(words) or _longest_common_run(quote, words) >= needed:
                key = (item["query_id"], chunk_id)
                grades[key] = max(grades.get(key, 0), int(item["relevance"]))
    return [
        {"query_id": query_id, "chunk_id": chunk_id, "relevance": grade}
        for (query_id, chunk_id), grade in sorted(grades.items())
    ]
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_bench_remap.py -q`
Expected: `3 passed`

- [ ] **Step 5: Commit**

```bash
git add src/bench/remap.py tests/test_bench_remap.py
git commit -m "feat: remap relevance labels to alternative chunkings via evidence quotes

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Thống kê mô tả benchmark (`bench/describe.py`)

**Files:**
- Create: `src/bench/describe.py`, `tests/test_bench_describe.py`

**Interfaces:**
- Produces: `describe_benchmark(queries, qrels, pool: list[dict] | None = None, agreement: dict | None = None) -> dict` với khóa:
  - `"counts"`: list `{"category", "origin", "split", "queries"}` sắp xếp
  - `"relevant_per_query"`: list `{"category", "mean", "median"}` (relevance ≥ 1)
  - `"pool_contribution"`: list `{"system", "relevant_found", "unique_relevant"}` (unique = chỉ hệ thống đó tìm thấy trong pool)
  - `"agreement"`: dict truyền vào hoặc None
  - `"totals"`: `{"queries", "qrels", "relevant"}`

- [ ] **Step 1: Viết test thất bại `tests/test_bench_describe.py`**

```python
from src.bench.describe import describe_benchmark

QUERIES = [
    {"query_id": "q1", "category": "exact", "origin": "llm", "split": "dev"},
    {"query_id": "q2", "category": "exact", "origin": "human", "split": "test"},
    {"query_id": "q3", "category": "concept", "origin": "llm", "split": "test"},
]
QRELS = [
    {"query_id": "q1", "chunk_id": "a", "relevance": 2},
    {"query_id": "q1", "chunk_id": "b", "relevance": 1},
    {"query_id": "q2", "chunk_id": "a", "relevance": 1},
    {"query_id": "q3", "chunk_id": "c", "relevance": 2},
    {"query_id": "q3", "chunk_id": "d", "relevance": 0},
]
POOL = [
    {"query_id": "q1", "chunk_id": "a", "systems": ["C1", "C2"]},
    {"query_id": "q1", "chunk_id": "b", "systems": ["C1"]},
    {"query_id": "q2", "chunk_id": "a", "systems": ["C2"]},
    {"query_id": "q3", "chunk_id": "c", "systems": ["manual"]},
    {"query_id": "q3", "chunk_id": "d", "systems": ["C1"]},
]


def test_describe_counts_relevance_and_pool_contribution():
    report = describe_benchmark(QUERIES, QRELS, POOL, {"kappa": 0.8})
    assert {"category": "exact", "origin": "human", "split": "test", "queries": 1} in report["counts"]
    exact = next(row for row in report["relevant_per_query"] if row["category"] == "exact")
    assert exact == {"category": "exact", "mean": 1.5, "median": 1.5}
    contribution = {row["system"]: row for row in report["pool_contribution"]}
    assert contribution["C1"] == {"system": "C1", "relevant_found": 2, "unique_relevant": 1}
    assert contribution["C2"] == {"system": "C2", "relevant_found": 2, "unique_relevant": 1}
    assert contribution["manual"]["unique_relevant"] == 1
    assert report["totals"] == {"queries": 3, "qrels": 5, "relevant": 4}
    assert report["agreement"] == {"kappa": 0.8}


def test_describe_without_pool():
    report = describe_benchmark(QUERIES, QRELS)
    assert report["pool_contribution"] == []
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_bench_describe.py -q`
Expected: FAIL với `ModuleNotFoundError: No module named 'src.bench.describe'`

- [ ] **Step 3: Viết `src/bench/describe.py`**

```python
import statistics
from collections import Counter, defaultdict


def describe_benchmark(queries, qrels, pool: list[dict] | None = None, agreement: dict | None = None) -> dict:
    counts = Counter((query["category"], query.get("origin", "llm"), query.get("split")) for query in queries)
    relevant = defaultdict(set)
    for row in qrels:
        if int(row["relevance"]) >= 1:
            relevant[row["query_id"]].add(row["chunk_id"])
    per_category = defaultdict(list)
    for query in queries:
        per_category[query["category"]].append(len(relevant[query["query_id"]]))
    contribution = []
    if pool:
        found = defaultdict(int)
        unique = defaultdict(int)
        for entry in pool:
            if entry["chunk_id"] not in relevant[entry["query_id"]]:
                continue
            for system in entry["systems"]:
                found[system] += 1
            if len(entry["systems"]) == 1:
                unique[entry["systems"][0]] += 1
        contribution = [
            {"system": system, "relevant_found": found[system], "unique_relevant": unique[system]}
            for system in sorted(found)
        ]
    return {
        "counts": [
            {"category": category, "origin": origin, "split": split, "queries": number}
            for (category, origin, split), number in sorted(counts.items(), key=lambda item: tuple(str(part) for part in item[0]))
        ],
        "relevant_per_query": [
            {"category": category, "mean": statistics.mean(values), "median": statistics.median(values)}
            for category, values in sorted(per_category.items())
        ],
        "pool_contribution": contribution,
        "agreement": agreement,
        "totals": {
            "queries": len(queries),
            "qrels": len(qrels),
            "relevant": sum(len(chunk_ids) for chunk_ids in relevant.values()),
        },
    }
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_bench_describe.py -q`
Expected: `2 passed`

- [ ] **Step 5: Commit**

```bash
git add src/bench/describe.py tests/test_bench_describe.py
git commit -m "feat: add benchmark descriptive statistics and pool contribution

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Lệnh CLI `bench …` và tài liệu quy trình

**Files:**
- Modify: `src/cli.py`, `README.md`, `CLAUDE.md`
- Create: `tests/test_cli_bench.py`

**Interfaces:**
- Consumes: mọi hàm của Task 1–9, `RetrievalIndex`, `RetrievalPipeline`, `RetrievalCache`, `load_encoder`, `OpenAI`.
- Produces: các lệnh (mọi lệnh nhận `--bench DIR`, mặc định `<data_dir>/benchmark`; `--index DIR`, mặc định index đang active):
  - `bench generate [--per-category 60] [--seed 42] [--model M] [--categories exact,concept,paraphrase,multi]` → `drafts.jsonl`, `rejected.jsonl`
  - `bench review-export` → `review.csv`
  - `bench review-import [--human FILE]` → `queries.jsonl`, `duplicates.jsonl`
  - `bench pool [--depth 15] [--annotators A,B] [--manual FILE] [--seed 42]` → `pool_map.json`, `annotation_*.csv`
  - `bench agreement --annotations F1 F2 [--resolved FILE]` → `agreement.json`, `disagreements.csv`; nếu không còn bất đồng chưa giải quyết thì ghi `qrels.jsonl`, `evidence.jsonl`
  - `bench split [--dev 0.3] [--seed 42]` → cập nhật `queries.jsonl`, ghi `benchmark_manifest.json`
  - `bench describe` → `benchmark_description.json`
  - `bench remap --target-index DIR --out FILE` → qrels cho index khác

- [ ] **Step 1: Viết test thất bại `tests/test_cli_bench.py`**

```python
import json

from src.cli import main
from src.index import RetrievalIndex
from src.io_utils import read_jsonl, write_csv, write_jsonl
from tests.fakes import FakeCrossEncoder, FakeEncoder, make_chunk

TEXTS = {
    "c1": "Mã môn AI101 là học phần nhập môn trí tuệ nhân tạo",
    "c2": "Học máy là lĩnh vực nghiên cứu thuật toán học từ dữ liệu",
    "c3": "Cơ sở dữ liệu quan hệ lưu trữ bảng và khóa chính",
    "c4": "Mạng nơ ron sâu gồm nhiều tầng biến đổi phi tuyến",
}


def _setup(tmp_path, monkeypatch):
    monkeypatch.setenv("PPL_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("PPL_RUNS_DIR", str(tmp_path / "runs"))
    monkeypatch.setattr("src.index.load_encoder", lambda name: FakeEncoder())
    monkeypatch.setattr("src.reranking.load_cross_encoder", lambda name: FakeCrossEncoder())
    index = RetrievalIndex.build(
        [make_chunk(cid, text) for cid, text in TEXTS.items()], tmp_path / "idx", embedding_model="fake", encoder=FakeEncoder()
    )
    bench = tmp_path / "bench"
    queries = [
        {"query_id": f"q{i}", "text": text, "category": "exact" if i < 2 else "concept", "origin": "llm",
         "split": None, "source_chunk_ids": [cid], "evidence": [], "generator": "m"}
        for i, (cid, text) in enumerate(TEXTS.items())
    ]
    write_jsonl(bench / "queries.jsonl", queries)
    return index, bench


def test_pool_agreement_split_describe_remap_flow(tmp_path, monkeypatch):
    index, bench = _setup(tmp_path, monkeypatch)
    common = ["--bench", str(bench), "--index", str(index.directory)]

    assert main(["bench", "pool", *common, "--depth", "2", "--annotators", "A,B"]) == 0
    pool = json.loads((bench / "pool_map.json").read_text(encoding="utf-8"))
    labels = [{"pool_id": entry["pool_id"], "relevance": "2" if "source" in entry["systems"] else "0",
               "evidence_quote": index.chunks[entry["chunk_id"]].body.split(" là ")[0] if "source" in entry["systems"] else ""}
              for entry in pool]
    write_csv(bench / "annotation_A.csv", labels)
    write_csv(bench / "annotation_B.csv", labels)

    assert main(["bench", "agreement", *common, "--annotations", str(bench / "annotation_A.csv"), str(bench / "annotation_B.csv")]) == 0
    assert json.loads((bench / "agreement.json").read_text(encoding="utf-8"))["kappa"] == 1.0
    assert len(read_jsonl(bench / "evidence.jsonl")) >= 3

    assert main(["bench", "split", *common, "--dev", "0.5", "--seed", "1"]) == 0
    splits = [query["split"] for query in read_jsonl(bench / "queries.jsonl")]
    assert splits.count("dev") == 2 and splits.count("test") == 2
    assert (bench / "benchmark_manifest.json").exists()

    assert main(["bench", "describe", *common]) == 0
    description = json.loads((bench / "benchmark_description.json").read_text(encoding="utf-8"))
    assert description["totals"]["queries"] == 4

    assert main(["bench", "remap", *common, "--target-index", str(index.directory), "--out", str(tmp_path / "remapped.jsonl")]) == 0
    assert {row["chunk_id"] for row in read_jsonl(tmp_path / "remapped.jsonl")} >= {"c1", "c3"}


def test_agreement_with_open_disagreements_writes_no_qrels(tmp_path, monkeypatch, capsys):
    index, bench = _setup(tmp_path, monkeypatch)
    common = ["--bench", str(bench), "--index", str(index.directory)]
    main(["bench", "pool", *common, "--depth", "1"])
    pool = json.loads((bench / "pool_map.json").read_text(encoding="utf-8"))
    write_csv(bench / "annotation_A.csv", [{"pool_id": pool[0]["pool_id"], "relevance": "2", "evidence_quote": ""}])
    write_csv(bench / "annotation_B.csv", [{"pool_id": pool[0]["pool_id"], "relevance": "0", "evidence_quote": ""}])
    assert main(["bench", "agreement", *common, "--annotations", str(bench / "annotation_A.csv"), str(bench / "annotation_B.csv")]) == 1
    assert not (bench / "qrels.jsonl").exists()
    assert (bench / "disagreements.csv").exists()
    assert "disagreements.csv" in capsys.readouterr().out
```

- [ ] **Step 2: Chạy test, xác nhận thất bại**

Run: `.venv/Scripts/python -m pytest tests/test_cli_bench.py -q`
Expected: FAIL với `SystemExit: 2` (argparse: invalid choice 'bench').

- [ ] **Step 3: Thêm lệnh `bench` vào `src/cli.py`**

Thêm import:
```python
from pathlib import Path

from openai import OpenAI

from src.bench.agreement import agreement_report, merge_labels, read_annotations, write_disagreements
from src.bench.describe import describe_benchmark
from src.bench.generate import CATEGORIES, generate_drafts
from src.bench.manifest import write_manifest
from src.bench.pool import DEFAULT_POOL_SYSTEMS, build_pool, write_pool
from src.bench.remap import remap_qrels
from src.bench.review import export_review, import_review
from src.bench.split import split_queries
from src.cache import RetrievalCache
from src.index import load_encoder
from src.indexing import resolve_index_dir
from src.io_utils import read_csv, read_jsonl, write_jsonl
from src.pipeline import RetrievalPipeline
```
Thêm các hàm xử lý (trước `build_parser`):
```python
def _bench_context(args):
    settings, database = _context()
    bench = Path(args.bench) if args.bench else settings.data_dir / "benchmark"
    bench.mkdir(parents=True, exist_ok=True)
    index = RetrievalIndex.load(resolve_index_dir(args.index, settings, database))
    return settings, bench, index


def _bench_generate(args) -> int:
    settings, bench, index = _bench_context(args)
    model = args.model or settings.openai_model
    if not settings.openai_api_key or not model:
        raise SystemExit("OPENAI_API_KEY and OPENAI_MODEL (or --model) are required")
    client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    categories = tuple(args.categories.split(",")) if args.categories else CATEGORIES
    accepted, rejected = generate_drafts(index, client, model, args.per_category, args.seed, categories)
    write_jsonl(bench / "drafts.jsonl", accepted)
    write_jsonl(bench / "rejected.jsonl", rejected)
    _print({"accepted": len(accepted), "rejected": len(rejected), "bench": str(bench)})
    return 0


def _bench_review_export(args) -> int:
    _, bench, index = _bench_context(args)
    export_review(read_jsonl(bench / "drafts.jsonl"), index, bench / "review.csv")
    _print({"review": str(bench / "review.csv")})
    return 0


def _bench_review_import(args) -> int:
    _, bench, index = _bench_context(args)
    queries, duplicates = import_review(
        bench / "review.csv", read_jsonl(bench / "drafts.jsonl"), load_encoder(index.embedding_model), args.human
    )
    write_jsonl(bench / "queries.jsonl", queries)
    write_jsonl(bench / "duplicates.jsonl", duplicates)
    _print({"queries": len(queries), "duplicates": len(duplicates)})
    return 0


def _bench_pool(args) -> int:
    settings, bench, index = _bench_context(args)
    pipeline = RetrievalPipeline(index, cache=RetrievalCache(settings.cache_path))
    queries = read_jsonl(bench / "queries.jsonl")
    manual = read_csv(args.manual) if args.manual else None
    pool = build_pool(queries, pipeline, DEFAULT_POOL_SYSTEMS, args.depth, manual)
    paths = write_pool(pool, queries, index, bench, tuple(args.annotators.split(",")), args.seed)
    _print({"pool_entries": len(pool), "annotation_files": [str(path) for path in paths]})
    return 0


def _bench_agreement(args) -> int:
    _, bench, index = _bench_context(args)
    pool = json.loads((bench / "pool_map.json").read_text(encoding="utf-8"))
    annotations = [read_annotations(path) for path in args.annotations]
    report = agreement_report(annotations[0], annotations[1]) if len(annotations) >= 2 else {"overlap": 0, "kappa": None, "raw_agreement": None, "disagreements": []}
    (bench / "agreement.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    if len(annotations) >= 2:
        write_disagreements(bench / "disagreements.csv", report, pool, annotations[0], annotations[1], read_jsonl(bench / "queries.jsonl"), index)
    try:
        qrels, evidence, warnings = merge_labels(pool, annotations, index, args.resolved)
    except ValueError as error:
        _print({"agreement": report, "error": str(error), "next": f"Điền cột 'final' trong {bench / 'disagreements.csv'} rồi chạy lại với --resolved"})
        return 1
    write_jsonl(bench / "qrels.jsonl", qrels)
    write_jsonl(bench / "evidence.jsonl", evidence)
    _print({"agreement": report, "qrels": len(qrels), "evidence": len(evidence), "warnings": warnings})
    return 0


def _bench_split(args) -> int:
    _, bench, index = _bench_context(args)
    queries, dropped = split_queries(read_jsonl(bench / "queries.jsonl"), read_jsonl(bench / "qrels.jsonl"), args.dev, args.seed)
    write_jsonl(bench / "queries.jsonl", queries)
    manifest = write_manifest(bench, index.version, args.seed, args.dev)
    _print({"queries": len(queries), "dropped_without_relevant_chunks": dropped, "manifest": manifest})
    return 0


def _bench_describe(args) -> int:
    _, bench, _ = _bench_context(args)
    pool_path, agreement_path = bench / "pool_map.json", bench / "agreement.json"
    report = describe_benchmark(
        read_jsonl(bench / "queries.jsonl"),
        read_jsonl(bench / "qrels.jsonl"),
        json.loads(pool_path.read_text(encoding="utf-8")) if pool_path.exists() else None,
        json.loads(agreement_path.read_text(encoding="utf-8")) if agreement_path.exists() else None,
    )
    (bench / "benchmark_description.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    _print(report["totals"])
    return 0


def _bench_remap(args) -> int:
    _, bench, _ = _bench_context(args)
    target = RetrievalIndex.load(args.target_index)
    qrels = remap_qrels(read_jsonl(bench / "evidence.jsonl"), target.chunk_list)
    write_jsonl(args.out, qrels)
    _print({"target_index": target.version, "qrels": len(qrels), "out": args.out})
    return 0
```
Trong `build_parser`, trước `return parser` thêm:
```python
    bench = groups.add_parser("bench").add_subparsers(dest="command", required=True)

    def bench_command(name, handler):
        command = bench.add_parser(name)
        command.add_argument("--bench")
        command.add_argument("--index")
        command.set_defaults(handler=handler)
        return command

    generate = bench_command("generate", _bench_generate)
    generate.add_argument("--per-category", type=int, default=60)
    generate.add_argument("--seed", type=int, default=42)
    generate.add_argument("--model")
    generate.add_argument("--categories")
    bench_command("review-export", _bench_review_export)
    bench_command("review-import", _bench_review_import).add_argument("--human")
    pool = bench_command("pool", _bench_pool)
    pool.add_argument("--depth", type=int, default=15)
    pool.add_argument("--annotators", default="A,B")
    pool.add_argument("--manual")
    pool.add_argument("--seed", type=int, default=42)
    agreement = bench_command("agreement", _bench_agreement)
    agreement.add_argument("--annotations", nargs="+", required=True)
    agreement.add_argument("--resolved")
    split = bench_command("split", _bench_split)
    split.add_argument("--dev", type=float, default=0.3)
    split.add_argument("--seed", type=int, default=42)
    bench_command("describe", _bench_describe)
    remap = bench_command("remap", _bench_remap)
    remap.add_argument("--target-index", required=True)
    remap.add_argument("--out", required=True)
```

- [ ] **Step 4: Chạy test, xác nhận pass**

Run: `.venv/Scripts/python -m pytest tests/test_cli_bench.py -q`
Expected: `2 passed`

- [ ] **Step 5: Cập nhật `README.md`**

Thay mục "Benchmark schema" bằng mục "Build the benchmark" mô tả đúng thứ tự lệnh:
```bash
python -m src.cli index build --input path/to/docs --course CS101        # lập chỉ mục + kích hoạt
python -m src.cli bench generate --per-category 60                         # LLM sinh nháp → drafts.jsonl
python -m src.cli bench review-export                                      # review.csv: action keep/edit/drop
python -m src.cli bench review-import --human human_queries.csv            # queries.jsonl (+ câu người viết)
python -m src.cli bench pool --depth 15 --annotators A,B                   # annotation_A.csv, annotation_B.csv
python -m src.cli bench agreement --annotations annotation_A.csv annotation_B.csv [--resolved disagreements.csv]
python -m src.cli bench split --dev 0.3 --seed 42                          # dev/test + benchmark_manifest.json
python -m src.cli bench describe                                           # benchmark_description.json
```
Kèm schema: `queries.jsonl` (`query_id, text, category ∈ exact|concept|paraphrase|multi, origin ∈ llm|human, split, source_chunk_ids, evidence, generator`), `qrels.jsonl` (`query_id, chunk_id, relevance 0|1|2`), `evidence.jsonl` (`query_id, doc_id, page, quote, relevance`), `human_queries.csv` (`text, category[, query_id]`), `manual_additions.csv` (`query_id, chunk_id`). Ghi chú: annotator điền cột `relevance` (0/1/2) và `evidence_quote` (trích nguyên văn đoạn căn cứ khi relevance ≥ 1); nên để ≥ 30% câu hỏi được cả hai người dán; `bench remap` dùng cho thí nghiệm đổi chunking.

- [ ] **Step 6: Cập nhật `CLAUDE.md`**

Trong Commands thêm khối `python -m src.cli index …` / `python -m src.cli bench …` (một dòng mỗi lệnh như README). Trong Architecture thêm đoạn "Benchmark (`src/bench/`)": hàm thuần trên `list[dict]`, I/O tập trung ở `src/io_utils.py` (CSV `utf-8-sig`), thứ tự `generate → review → pool → agreement → split → describe`, khóa test ở `bench/manifest.check_test_lock` (Kế hoạch 3 gọi khi `eval run --split test`).

- [ ] **Step 7: Chạy toàn bộ test**

Run: `.venv/Scripts/python -m pytest -q`
Expected: tất cả pass.

- [ ] **Step 8: Commit**

```bash
git add src/cli.py tests/test_cli_bench.py README.md CLAUDE.md
git commit -m "feat: expose benchmark workflow through the bench CLI

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
