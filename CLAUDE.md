# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Local Streamlit app for grounded Vietnamese-language question answering over course materials (PDF/DOCX/PPTX), plus a reproducible research harness comparing BM25, dense, RRF, weighted hybrid, adaptive hybrid, and reranked retrieval. UI strings, the LLM prompt, and the refusal text are in Vietnamese. Keep them that way.

## Commands

Python 3.11, venv at `.venv`. The README uses POSIX paths (`.venv/bin/python`). On this Windows machine use `.venv/Scripts/python`.

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt --torch-backend cpu   # requirements.txt is compiled from requirements.in
cp .env.example .env                        # OPENAI_API_KEY / OPENAI_BASE_URL / OPENAI_MODEL + ADMIN_/STUDENT_ USERNAME/PASSWORD

.venv/bin/streamlit run app.py              # run the app

# Verification (the project's full check)
.venv/bin/python -m pytest -q                                   # Windows: .venv/Scripts/python -m pytest -q
.venv/bin/python -m pytest tests/test_pipeline.py::test_rerank_reorders_head_and_keeps_tail -q   # single test
.venv/bin/python -m compileall -q app.py pages src run_experiments.py run_rag_evaluation.py

# Experiments / RAG evaluation
.venv/bin/python run_experiments.py --config configs/generated_experiment.yaml [--dry-run | --resume RUN_ID]
.venv/bin/python run_rag_evaluation.py --config configs/generated_experiment.yaml
.venv/bin/python run_rag_evaluation.py --summarize runs/RAG_RUN/rag_answers_blinded.csv
```

Tests are behavior-level, one file per module. They never download models or call the LLM: use `tests/fakes.py` (`FakeEncoder`, `FakeCrossEncoder`, `make_chunk`) and inject fake pipelines/clients.

## Architecture

**Entry points**: `app.py` (login) plus Streamlit multipage `pages/` (Chat for student/admin; Documents, RAG Settings, and Experiments for admin only, gated by `src.ui.require_role`). There are also two CLI scripts, `run_experiments.py` and `run_rag_evaluation.py`. Paths resolve from the CWD (pages read `configs/*.yaml` relatively), so run everything from the repo root. `PPL_DATA_DIR`/`PPL_RUNS_DIR` override the `data/` and `runs/` locations.

**Pipeline** (`src/`):
1. `text.py`: `normalize_text` (NFC, strips zero-width/control chars, joins hyphenated line breaks) and `tokenize(text, mode)` with `whitespace | pyvi | vncorenlp`. Documents and queries both go through it.
2. `ingestion.py`: `extract_blocks` yields `Block(page, heading_path, text)`. DOCX headings come from `Heading 1-3` styles; PDF headings from regex (`Chương/Bài`, `1.2 …`) plus font size/bold, and lines repeated on more than half the pages (headers/footers) are dropped; each PPTX slide title is its heading. `build_corpus` writes `data/processed/<version_id>/`, where `version_id` hashes the files and the chunking config. Rebuilding an identical corpus reuses it.
3. `chunking.py`: `chunk_blocks` with strategy `structure` (sentences packed within one section, max 350 words, 50 overlap) or `fixed` (450/75 word windows). With `prefix=True`, `text = "Doc > Chapter > Section
" + body`. `chunk_id` hashes doc, strategy, page, heading path, index and body.
4. `index.py`: `RetrievalIndex` stores embeddings (numpy exact, or FAISS `IndexFlatIP`) plus one BM25 token file per tokenizer (`add_tokenizer` adds more without re-embedding). BM25 uses `k1=1.5`, `b=0.75`.
5. `pipeline.py`: `RetrievalPipeline.run(query, PipelineConfig)`: sparse/dense top-L → fusion (`none | rrf | weighted | adaptive`, and `alpha` weights **BM25**) → rerank top-N → the full ordered list. Each `RetrievedChunk` carries `StageScores` (sparse/dense score and rank, fusion score, rerank score, rank), and the result carries `timings_ms`. An optional SQLite `RetrievalCache` stores first-stage lists and rerank scores.
6. `rag.py`: `answer_question(query, pipeline, config, client, model)` → top `context_k` → refusal below `refusal_threshold` → LLM (uses `config.temperature`/`timeout_seconds`) → `_valid_citations`.

**State**: `storage.Database` is a thin sqlite3 wrapper (`data/app.db`). Most tables store a JSON payload column. Exactly one `corpus_versions` row is `active`, and Chat loads the index for that version. Named `PipelineConfig`s saved from the RAG Settings page are what Chat uses; `load_pipeline_configs` skips legacy/invalid rows and reports their names. Users are seeded from `.env` on startup with scrypt hashes, and existing users are never overwritten.

**Experiments**: `configs/experiments.yaml` is the template. The Experiments page fills in the active corpus/index and uploaded benchmark files, then writes `configs/generated_experiment.yaml` (gitignored). Experiment IDs are restricted to `E0`–`E7`. `run_experiment` writes `runs/<timestamp>-<confighash>/` with `config.json`, `status.json`, an append-only `checkpoint.jsonl` (resume skips completed `(experiment_id, query_id)` pairs), `metrics.json`, `per_query.csv`, and `errors.csv`. Metrics live in `evaluation.py` (MRR@10, hit/precision/recall/nDCG@k with graded 0/1/2 qrels, paired bootstrap CI). Benchmark JSONL schemas are in README.md.

## Gotchas

- BM25 fixtures in tests need at least 4 chunks, with each query term in only one chunk. `rank_bm25` gives idf ≤ 0 to a term that appears in at least half of the documents.
- `PipelineConfig` validates strictly: any fusion other than `none` needs both branches, and it requires `context_k ≤ rerank_n ≤ top_l`. RAG configs saved to the DB in the old format are skipped.
- `src/experiments.py` + `run_experiments.py` are a temporary bridge (legacy E0–E7 YAML mapped onto `PipelineConfig`) until `src/eval/` replaces them.
- python-pptx returns fresh shape proxies, so compare shapes by `shape_id`, never with `is`.
- Windows console is cp1252; set `PYTHONIOENCODING=utf-8` when printing Vietnamese.
- Research protocol from README: don't look at test-split results until tokenizer, chunking, models, fusion params, and thresholds are locked using train/dev.
- `data/raw`, `data/processed`, `data/indexes`, `runs/`, `*.db`, `.env`, and `*.docx` are gitignored. `configs/example_experiments.yaml` points at `examples/index`, which is not a real index and only works with `--dry-run`.
