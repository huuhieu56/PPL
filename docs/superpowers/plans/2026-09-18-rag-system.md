# Vietnamese Learning RAG System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Xây dựng hệ thống RAG local hoàn chỉnh bằng Streamlit cho sinh viên và quản trị viên, đồng thời chạy được thí nghiệm E0–E7 có khả năng tái lập.

**Architecture:** Một ứng dụng Python dùng chung lõi ingestion, retrieval, reranking, RAG và evaluation. Streamlit chỉ phục vụ thao tác/người dùng; benchmark dài chạy qua CLI và ghi artifacts vào `runs/`. SQLite lưu metadata, còn tài liệu, chunks, embeddings và kết quả nằm trên filesystem.

**Tech Stack:** Python 3.11, Streamlit, SQLite, PyMuPDF, python-docx, python-pptx, rank-bm25, pyvi, sentence-transformers, NumPy, OpenAI-compatible API, PyYAML, pytest.

**Spec:** `RESEARCH_EXPERIMENT_PLAN.md`

## Global Constraints

- Chạy local; không FastAPI, React, Docker, vector database hoặc background queue.
- API LLM lấy từ `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL`; không ghi secret vào log/database.
- Cùng corpus/chunks/query/qrels cho mọi cấu hình E0–E7.
- Weighted Sum chuẩn hóa min-max theo query trên top-L cố định; mặc định `L=100`.
- Demo CPU dùng reranker `N=20`; `N=20/50/100` chỉ chạy offline.
- Benchmark chạy bằng `python run_experiments.py --config <path>` và có checkpoint/resume.
- Chỉ tạo bốn test hành vi cốt lõi trong một file `tests/test_core.py`; không tạo test theo từng hàm.

---

## File Map

```text
app.py                         # entrypoint, login và điều hướng Streamlit
run_experiments.py             # CLI benchmark/resume
run_rag_evaluation.py          # sinh bộ câu trả lời mù để chấm RAG end-to-end
pages/1_Chat.py                # chat RAG và feedback
pages/2_Documents.py           # upload, preview, indexing
pages/3_RAG_Settings.py        # chỉnh/lưu cấu hình
pages/4_Experiments.py         # tạo config, đọc và vẽ kết quả
src/models.py                  # dataclass dùng chung
src/config.py                  # env/YAML config
src/storage.py                 # SQLite và filesystem metadata
src/ingestion.py               # extract, reading order, chunk, corpus version
src/retrieval.py               # BM25, dense, normalization, fusion, adaptive alpha
src/reranking.py               # batch cross-encoder reranking
src/evaluation.py              # metrics, bootstrap CI, per-query output
src/experiments.py             # E0–E7 và checkpoint
src/rag.py                     # retrieve → prompt → LLM → citation/refusal
src/ui.py                      # auth/session helpers dùng chung cho pages
tests/test_core.py             # bốn kiểm thử hành vi quan trọng
configs/default.yaml           # cấu hình demo
configs/experiments.yaml       # ma trận E0–E7
.env.example                   # biến môi trường không chứa secret
.gitignore                     # loại .env, data, runs, models/cache
requirements.in                # dependency trực tiếp
requirements.txt               # lock do uv compile tạo
README.md                      # cài đặt, chạy demo, chạy benchmark
```

---

### Task 1: Project foundation, configuration, and storage

**Files:**
- Create: `.gitignore`, `.env.example`, `requirements.in`, `requirements.txt`, `src/__init__.py`, `src/models.py`, `src/config.py`, `src/storage.py`, `configs/default.yaml`
- Create: `tests/test_core.py`

**Interfaces:**
- Produces: `Settings`, `Chunk`, `SearchResult`, `RagConfig`, `Database`, `load_settings()`, `load_yaml(path)`.

- [ ] **Step 1: Khởi tạo Git và môi trường Python 3.11**

Run:

```bash
git init
uv venv --python 3.11 .venv
```

Expected: `.git/` và `.venv/` được tạo; `.venv/bin/python --version` là Python 3.11.x.

- [ ] **Step 2: Khai báo dependency tối thiểu và khóa phiên bản**

`requirements.in`:

```text
streamlit
pymupdf
python-docx
python-pptx
rank-bm25
pyvi
sentence-transformers
numpy
openai
python-dotenv
pyyaml
pytest
```

Run:

```bash
uv pip compile requirements.in -o requirements.txt
uv pip install --python .venv/bin/python -r requirements.txt
```

- [ ] **Step 3: Tạo model/config contracts**

Implement these exact public types in `src/models.py`:

```python
@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    course: str
    source_type: str
    page: int
    section: str
    text: str

@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    rank: int
    source: str

@dataclass(frozen=True)
class RagConfig:
    method: str = "adaptive"
    top_l: int = 100
    rerank_n: int = 20
    context_k: int = 5
    alpha: float = 0.5
    rrf_k: int = 60
    refusal_threshold: float = 0.0
    use_reranker: bool = True
```

Implement `Settings` in `src/config.py` with paths and OpenAI-compatible env values. `load_settings()` must create `data/`, `runs/` and the SQLite parent directory without printing secrets.

- [ ] **Step 4: Tạo SQLite schema và repository tối thiểu**

`Database.initialize()` creates tables: `users`, `documents`, `corpus_versions`, `rag_configs`, `chat_sessions`, `messages`, `feedback`, `experiment_runs`. Add only methods used by later tasks:

```python
Database.upsert_user(username: str, password_hash: str, role: str) -> None
Database.save_document(record: dict) -> None
Database.list_documents() -> list[dict]
Database.save_corpus_version(record: dict) -> None
Database.get_active_corpus() -> dict | None
Database.set_active_corpus(version_id: str) -> None
Database.save_rag_config(name: str, config: RagConfig) -> None
Database.list_rag_configs() -> list[dict]
Database.save_message(record: dict) -> None
Database.save_feedback(record: dict) -> None
Database.save_run(record: dict) -> None
Database.update_run(run_id: str, status: str, result_path: str | None) -> None
```

- [ ] **Step 5: Viết một test storage/config duy nhất**

Add this first behavior test to `tests/test_core.py`:

```python
import json

from src.config import load_settings
from src.models import RagConfig
from src.storage import Database


def test_foundation_round_trip(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-be-persisted")
    settings = load_settings(tmp_path)
    db = Database(settings.db_path)
    db.initialize()
    db.save_rag_config("demo", RagConfig(method="rrf", use_reranker=False))

    saved = db.list_rag_configs()
    assert saved[0]["name"] == "demo"
    assert saved[0]["config"]["method"] == "rrf"
    assert "must-not-be-persisted" not in json.dumps(saved)
```

- [ ] **Step 6: Chạy test và commit**

Run:

```bash
.venv/bin/pytest tests/test_core.py::test_foundation_round_trip -q
git add .
git commit -m "feat: establish project configuration and storage"
```

Expected: `1 passed`.

---

### Task 2: Ingestion, Vietnamese tokenization, and corpus indexing

**Files:**
- Create: `src/ingestion.py`
- Modify: `tests/test_core.py`, `configs/default.yaml`

**Interfaces:**
- Consumes: `Chunk`, `Settings`, `Database`.
- Produces: `extract_document(path: Path) -> list[PageText]`, `chunk_pages(pages: list[PageText], *, doc_id: str, course: str, source_type: str, chunk_tokens: int, overlap_tokens: int) -> list[Chunk]`, `build_corpus(files: list[Path], metadata: dict[str, dict], settings: Settings, database: Database, config: dict) -> CorpusBuildResult`, `tokenize_vi(text: str, mode: str) -> list[str]`.

- [ ] **Step 1: Viết test ingestion theo hành vi**

Add this one ingestion test; it creates its own fixture rather than committing a binary PPTX:

```python
from pptx import Presentation
from pptx.util import Inches

from src.ingestion import chunk_pages, extract_document


def test_ingestion_preserves_source_and_reading_order(tmp_path):
    path = tmp_path / "lesson.pptx"
    deck = Presentation()
    slide = deck.slides.add_slide(deck.slide_layouts[5])
    slide.shapes.title.text = "Bài 1"
    left = slide.shapes.add_textbox(Inches(1), Inches(2), Inches(3), Inches(1))
    right = slide.shapes.add_textbox(Inches(5), Inches(2), Inches(3), Inches(1))
    left.text = "Nội dung bên trái"
    right.text = "Nội dung bên phải"
    deck.save(path)

    pages = extract_document(path)
    assert pages[0].text.index("Bài 1") < pages[0].text.index("Nội dung bên trái")
    assert pages[0].text.index("Nội dung bên trái") < pages[0].text.index("Nội dung bên phải")

    chunks = chunk_pages(
        pages, doc_id="doc-1", course="AI101", source_type="slide",
        chunk_tokens=450, overlap_tokens=75,
    )
    assert chunks
    assert all(c.doc_id == "doc-1" and c.course == "AI101" and c.page == 1 and c.text for c in chunks)
```

- [ ] **Step 2: Chạy test để xác nhận thất bại**

Run: `.venv/bin/pytest tests/test_core.py::test_ingestion_preserves_source_and_reading_order -q`

Expected: FAIL because `src.ingestion` does not exist.

- [ ] **Step 3: Implement extraction and reading order**

Use PyMuPDF pages, DOCX paragraphs/tables, and PPTX shapes. For a PDF page without a text layer, call PyMuPDF OCR only when local Tesseract is available; otherwise retain a structured extraction warning for the admin preview. For PPTX: title first; recursively flatten groups; group remaining text shapes into horizontal bands using their top coordinate; sort bands top-to-bottom and shapes left-to-right; serialize table cells row-by-row. Ignore empty/decorative shapes.

Define:

```python
@dataclass(frozen=True)
class PageText:
    page: int
    section: str
    text: str

@dataclass(frozen=True)
class CorpusBuildResult:
    version_id: str
    chunks_path: Path
    chunk_count: int
    manifest_hash: str
```

- [ ] **Step 4: Implement chunking and immutable corpus version**

Chunk by heading with token-window fallback. Default 450 tokens/75 overlap. Derive deterministic `doc_id` and `chunk_id` from SHA-256, write UTF-8 `chunks.jsonl`, `manifest.json`, and register the version in SQLite. Never modify an existing version directory.

- [ ] **Step 5: Implement tokenizer modes**

`tokenize_vi(text, "whitespace")` lowercases and splits whitespace; `tokenize_vi(text, "pyvi")` uses `ViTokenizer.tokenize(text)` and returns underscore-joined compound words as tokens. Corpus and query must call the same mode from config.

- [ ] **Step 6: Run test and commit**

Run:

```bash
.venv/bin/pytest tests/test_core.py -q
git add src/ingestion.py tests/test_core.py configs/default.yaml
git commit -m "feat: ingest and version learning documents"
```

Expected: `2 passed`.

---

### Task 3: Retrieval, fusion, reranking, and metrics

**Files:**
- Create: `src/retrieval.py`, `src/reranking.py`, `src/evaluation.py`
- Modify: `tests/test_core.py`, `configs/experiments.yaml`

**Interfaces:**
- Consumes: `Chunk`, `SearchResult`, `RagConfig`, `tokenize_vi`.
- Produces: `RetrievalIndex.build(chunks, output_dir, tokenizer_mode, embedding_model)`, `RetrievalIndex.load(index_dir)`, `retrieve(query, index, config) -> list[SearchResult]`, `minmax_scores(scores: dict[str, float]) -> dict[str, float]`, `adaptive_alpha(query, alpha0, beta, idf) -> tuple[float, dict[str, float]]`, `fuse_weighted(bm25_scores, dense_scores, chunks, alpha) -> list[SearchResult]`, `fuse_rrf(result_lists, rrf_k) -> list[SearchResult]`, `rerank(query, results, limit) -> list[SearchResult]`, `evaluate_rankings(rankings, qrels, ks) -> dict[str, float]`.

- [ ] **Step 1: Viết một test retrieval tổng hợp**

Add this third test with in-memory chunks; later Task 5 extends the same test rather than adding another one:

```python
from src.evaluation import evaluate_rankings
from src.models import Chunk
from src.retrieval import adaptive_alpha, fuse_weighted, minmax_scores


def test_retrieval_fusion_and_metrics():
    chunks = {
        "c1": Chunk("c1", "d1", "AI101", "slide", 1, "Mã môn", "Mã môn AI101"),
        "c2": Chunk("c2", "d1", "AI101", "slide", 2, "Khái niệm", "Giải thích học máy"),
    }
    assert minmax_scores({"c1": 5.0, "c2": 5.0}) == {"c1": 1.0, "c2": 1.0}

    exact_alpha, exact_signals = adaptive_alpha(
        "Mã môn AI101 là gì?", alpha0=0.5, beta=0.3, idf={"ai101": 1.0}
    )
    semantic_alpha, _ = adaptive_alpha(
        "Giải thích học máy", alpha0=0.5, beta=0.3, idf={"học": 0.1, "máy": 0.1}
    )
    assert exact_signals["code"] == 1.0
    assert exact_alpha > semantic_alpha

    fused = fuse_weighted(
        bm25_scores={"c1": 8.0, "c2": 1.0},
        dense_scores={"c1": 0.6, "c2": 0.5},
        chunks=chunks,
        alpha=0.7,
    )
    assert fused[0].chunk.chunk_id == "c1"

    metrics = evaluate_rankings(
        rankings={"q1": ["c1", "c2"]},
        qrels={"q1": {"c1": 2}},
        ks=(1, 10),
    )
    assert metrics["hit_rate@1"] == 1.0
    assert metrics["mrr@10"] == 1.0
```

This single test covers normalization, adaptive behavior, ranking and metric correctness.

- [ ] **Step 2: Chạy test để xác nhận thất bại**

Run: `.venv/bin/pytest tests/test_core.py::test_retrieval_fusion_and_metrics -q`

Expected: FAIL on missing retrieval module.

- [ ] **Step 3: Implement persistent BM25 and dense index**

`RetrievalIndex.build(chunks, output_dir, tokenizer_mode, embedding_model)` writes `chunks.jsonl`, dense `embeddings.npy`, `index_meta.json`, and BM25 token data. `load()` verifies chunk/hash/model metadata. Use normalized embeddings and NumPy matrix multiplication; do not add FAISS.

- [ ] **Step 4: Implement fusion exactly as specified**

For Weighted Sum: retrieve top-L from both branches, union by `chunk_id`, min-max each returned list, use 0 for missing results and 1 for every returned result when all scores tie. Implement RRF as `sum(1 / (rrf_k + rank))`. Sort deterministically by descending score then `chunk_id`.

`adaptive_alpha()` computes digit, symbol, code-pattern and max-IDF signals; clips to `[0.1, 0.9]`; returns the chosen alpha plus a signal dictionary for logging.

- [ ] **Step 5: Implement reranker with CPU-safe defaults**

Load `BAAI/bge-reranker-v2-m3` lazily, score `(query, text)` pairs in batches, default candidate count 20, preserve source metadata, and record rerank latency. When disabled, return input order unchanged.

- [ ] **Step 6: Implement evaluation**

Calculate Hit Rate@K, MRR@10, NDCG@K, Precision@K, Recall@K and per-query rows. Add paired bootstrap CI with an explicit seed. Write `metrics.json` and `per_query.csv`.

- [ ] **Step 7: Run tests and commit**

Run:

```bash
.venv/bin/pytest tests/test_core.py -q
git add src/retrieval.py src/reranking.py src/evaluation.py configs/experiments.yaml tests/test_core.py
git commit -m "feat: add adaptive hybrid retrieval and evaluation"
```

Expected: `3 passed`.

---

### Task 4: Reproducible E0–E7 experiment CLI

**Files:**
- Create: `src/experiments.py`, `run_experiments.py`
- Modify: `tests/test_core.py`, `configs/experiments.yaml`

**Interfaces:**
- Consumes: retrieval/evaluation interfaces and JSONL queries/qrels.
- Produces: `run_experiment(config_path) -> Path`, `runs/<run_id>/{config,status,metrics,per_query,errors}`.

- [ ] **Step 1: Viết test end-to-end CLI/resume duy nhất**

Add the fourth behavior test. `run_experiment()` accepts `resume_run_id` only for resuming an existing run; `execute_query()` is the single query execution seam:

```python
import csv
import json

import pytest
import yaml

from src.experiments import run_experiment


def test_experiment_run_resumes_without_repeating_completed_queries(tmp_path, monkeypatch):
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        "\n".join(json.dumps({"query_id": q, "text": q}) for q in ["q1", "q2"]),
        encoding="utf-8",
    )
    qrels = tmp_path / "qrels.jsonl"
    qrels.write_text(
        "\n".join(json.dumps({"query_id": q, "chunk_id": "c1", "relevance": 2}) for q in ["q1", "q2"]),
        encoding="utf-8",
    )
    config_path = tmp_path / "experiment.yaml"
    config_path.write_text(yaml.safe_dump({
        "runs_dir": str(tmp_path / "runs"),
        "queries": str(queries),
        "qrels": str(qrels),
        "corpus_version": "fixture",
        "experiments": ["E0"],
    }), encoding="utf-8")

    calls = []
    interrupted = {"value": False}

    def fake_execute_query(experiment_id, query, config):
        calls.append(query["query_id"])
        if query["query_id"] == "q2" and not interrupted["value"]:
            interrupted["value"] = True
            raise KeyboardInterrupt
        return {"ranked_chunk_ids": ["c1"], "latency_ms": 1.0}

    monkeypatch.setattr("src.experiments.execute_query", fake_execute_query)
    with pytest.raises(KeyboardInterrupt):
        run_experiment(config_path)

    run_dir = next((tmp_path / "runs").iterdir())
    run_experiment(config_path, resume_run_id=run_dir.name)
    rows = list(csv.DictReader((run_dir / "per_query.csv").open(encoding="utf-8")))
    assert calls.count("q1") == 1
    assert {row["query_id"] for row in rows} == {"q1", "q2"}
```

- [ ] **Step 2: Chạy test để xác nhận thất bại**

Run: `.venv/bin/pytest tests/test_core.py::test_experiment_run_resumes_without_repeating_completed_queries -q`

Expected: FAIL on missing experiment runner.

- [ ] **Step 3: Implement config validation and run identity**

Validate corpus path, query/qrels schema, method names E0–E7, K values and model identifiers. Derive `run_id` from timestamp plus a short hash of frozen config and data hashes. Copy the resolved config to the run directory.

- [ ] **Step 4: Implement checkpoint/resume**

Append one JSONL record per completed query, flush immediately, skip completed `(experiment_id, query_id)` pairs on resume, and write errors without stopping unrelated queries. Aggregate CSV/metrics only from checkpoint records.

- [ ] **Step 5: Implement CLI**

`run_experiments.py` accepts `--config`, `--resume <run_id>` and `--dry-run`. `--dry-run` validates files/models/config without loading heavy models.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
.venv/bin/pytest tests/test_core.py -q
git add src/experiments.py run_experiments.py configs/experiments.yaml tests/test_core.py
git commit -m "feat: run reproducible retrieval experiments"
```

Expected: `4 passed`.

---

### Task 5: Grounded RAG and end-to-end answer evaluation

**Files:**
- Create: `src/rag.py`, `run_rag_evaluation.py`
- Modify: `tests/test_core.py`, `configs/default.yaml`

**Interfaces:**
- Consumes: active `RetrievalIndex`, optional reranker, `RagConfig`, OpenAI-compatible settings.
- Produces: `answer_question(query, index, config, client) -> RagAnswer`.

- [ ] **Step 1: Mở rộng test retrieval hiện có thay vì tạo test mới**

Extend `test_retrieval_fusion_and_metrics` with a fake OpenAI client. Assert a grounded answer maps `[1]` to a real chunk; invalid citation `[99]` is removed; below-threshold retrieval returns the refusal string without invoking the fake client.

- [ ] **Step 2: Chạy test để xác nhận thất bại**

Run: `.venv/bin/pytest tests/test_core.py::test_retrieval_fusion_and_metrics -q`

Expected: FAIL because `answer_question` is absent.

- [ ] **Step 3: Implement prompt and API call**

Build numbered context with strict Vietnamese grounding instructions. Call the configured OpenAI-compatible chat completion with temperature 0, timeout and at most two retries using short exponential backoff. Return token usage and stage latencies.

- [ ] **Step 4: Implement refusal and citation validation**

Calibrate method-specific thresholds from config. If confidence is below threshold, do not call the API. Parse only citations present in supplied context, remove invalid markers, and return citation metadata containing `chunk_id`, document, page/slide and section.

- [ ] **Step 5: Implement blinded RAG evaluation export**

`run_rag_evaluation.py --config <path>` runs the same selected test questions through Dense RAG, best fixed Hybrid RAG and E7 using identical LLM/prompt/temperature/context budget. Randomize system order per query with a fixed seed and write `rag_answers_blinded.csv` containing `item_id`, query, answer, citations, latency, input/output tokens plus empty human columns `correctness_1_5`, `faithfulness_1_5`, `citation_correct_0_1`, `notes`. A second `--summarize <completed.csv>` mode validates filled scores and writes aggregate JSON without calling the LLM.

- [ ] **Step 6: Run tests and commit**

Run:

```bash
.venv/bin/pytest tests/test_core.py -q
git add src/rag.py run_rag_evaluation.py configs/default.yaml tests/test_core.py
git commit -m "feat: generate grounded answers with citations"
```

Expected: `4 passed`.

---

### Task 6: Streamlit application for student and administrator

**Files:**
- Create: `app.py`, `src/ui.py`, `pages/1_Chat.py`, `pages/2_Documents.py`, `pages/3_RAG_Settings.py`, `pages/4_Experiments.py`
- Modify: `src/storage.py`, `.env.example`

**Interfaces:**
- Consumes: all core modules; no duplicate retrieval/RAG logic in pages.
- Produces: working two-role local UI.

- [ ] **Step 1: Implement local authentication**

Use standard-library `hashlib.scrypt` with per-user salts. Seed `ADMIN_USERNAME/ADMIN_PASSWORD` and `STUDENT_USERNAME/STUDENT_PASSWORD` from environment on first start. Store only salt/hash and role. `require_role(*roles)` stops unauthorized pages.

- [ ] **Step 2: Implement student chat page**

Use `st.chat_input`, keep visible session history, call `answer_question`, render citations in expanders, show total latency, and save thumbs-up/down plus optional comment. Do not show secrets or raw system prompts.

- [ ] **Step 3: Implement document admin page**

Validate PDF/DOCX/PPTX extension and size, sanitize names, save inside the data root, collect course/type/semester, preview extracted pages/chunks, build a new immutable corpus version, and activate it only after successful index creation.

- [ ] **Step 4: Implement RAG configuration page**

Expose method, reranker toggle, top-L, rerank-N, context-K, alpha, RRF k, refusal threshold, model and temperature. Validate ranges and store named configurations in SQLite.

- [ ] **Step 5: Implement experiment result page**

Create validated YAML config files, display the exact CLI command, list `runs/`, read status/metrics/errors, filter results and offer CSV/JSON downloads. Do not execute the benchmark inside Streamlit.

- [ ] **Step 6: Manual smoke check and commit**

Run:

```bash
.venv/bin/streamlit run app.py
```

Verify: both roles can log in; student cannot access admin pages; admin can preview a small document; chat refuses cleanly without active index; experiment page shows a dry-run command.

Then:

```bash
git add app.py pages src/ui.py src/storage.py .env.example
git commit -m "feat: add Streamlit student and admin workflows"
```

---

### Task 7: Documentation and final system verification

**Files:**
- Create: `README.md`, `data/benchmark/queries.example.jsonl`, `data/benchmark/qrels.example.jsonl`
- Modify: `.gitignore`, `configs/default.yaml`, `configs/experiments.yaml`

**Interfaces:**
- Consumes: complete system.
- Produces: reproducible local setup and operator instructions.

- [ ] **Step 1: Write README with exact commands**

Document Python 3.11 setup, dependency install, optional local Tesseract, `.env`, Streamlit start, document ingestion, benchmark schemas, `--dry-run`, actual run, resume, blinded RAG evaluation, artifact interpretation and test command. State that test-set results must not be opened before configuration lock.

- [ ] **Step 2: Add tiny example benchmark**

Provide two schema-valid example queries and qrels referencing clearly marked example chunk IDs. They demonstrate file shape only and are not research data.

- [ ] **Step 3: Run automated verification**

Run:

```bash
.venv/bin/pytest tests/test_core.py -q
.venv/bin/python run_experiments.py --config configs/experiments.yaml --dry-run
```

Expected: exactly `4 passed`; dry-run validates or reports only the explicitly missing real corpus/benchmark paths without loading models.

- [ ] **Step 4: Run final local smoke test**

Run Streamlit, ingest one small document, build an index, ask one grounded and one unanswerable question, save feedback, then verify the SQLite rows and citations displayed in UI.

- [ ] **Step 5: Check secret/data hygiene and commit**

Run:

```bash
git status --short
git grep -n "OPENAI_API_KEY=" -- ':!*.example' || true
git check-ignore .env data/raw runs
```

Expected: no secret value tracked; local corpus and run outputs ignored.

Then:

```bash
git add README.md data/benchmark .gitignore configs
git commit -m "docs: document local RAG and experiment workflow"
```

---

## Completion Gate

Implementation is complete only when:

1. The four behavior tests pass.
2. One document can be uploaded, previewed, indexed and activated.
3. Student chat returns grounded citations and refuses an unsupported question.
4. `run_experiments.py --dry-run` validates E0–E7 config without model loading.
5. A tiny real run can resume from checkpoint and exports metrics/per-query/errors.
6. Blinded Dense/Hybrid/E7 RAG answers can be exported for human scoring and summarized without another API call.
7. Streamlit only reads experiment runs and never holds the long benchmark process.
8. `.env`, raw documents, indexes and run artifacts are not tracked by Git.
