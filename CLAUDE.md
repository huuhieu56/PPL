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
.venv/bin/python -m src.cli eval run --config configs/experiment.example.yaml --split test --dry-run
.venv/bin/python -m compileall -q app.py pages src run_rag_evaluation.py

# Indexing and benchmark CLI (see README "Build the benchmark")
.venv/bin/python -m src.cli index build --input DIR --course CS101 [--strategy fixed] [--no-prefix] [--tokenizers whitespace,pyvi]
.venv/bin/python -m src.cli index add-tokenizer --index DIR --tokenizer vncorenlp
.venv/bin/python -m src.cli bench {generate|review-export|review-import|pool|agreement|split|describe|remap} ...

# Evaluation / RAG answer scoring (see README "Run experiments")
.venv/bin/python -m src.cli eval {tune|run|compare|errors|report} --config configs/experiment.yaml ...
.venv/bin/python run_rag_evaluation.py --config configs/experiment.yaml
.venv/bin/python run_rag_evaluation.py --summarize runs/RAG_RUN/rag_answers_blinded.csv
```

Tests are behavior-level, one file per module. They never download models or call the LLM: use `tests/fakes.py` (`FakeEncoder`, `FakeCrossEncoder`, `make_chunk`) and inject fake pipelines/clients.

## Architecture

**Entry points**: `app.py` (login) plus Streamlit multipage `pages/` (Chat for student/admin; Documents, RAG Settings, and Experiments for admin only, gated by `src.ui.require_role`). `src/cli.py` is the single argparse entry point (`index`, `bench`, `eval` command groups); `run_rag_evaluation.py` is a separate CLI script for blinded LLM-answer scoring. Paths resolve from the CWD (pages read `configs/*.yaml` relatively), so run everything from the repo root. `PPL_DATA_DIR`/`PPL_RUNS_DIR` override the `data/` and `runs/` locations.

**Pipeline** (`src/`):
1. `text.py`: `normalize_text` (NFC, strips zero-width/control chars, joins hyphenated line breaks) and `tokenize(text, mode)` with `whitespace | pyvi | vncorenlp`. Documents and queries both go through it.
2. `ingestion.py`: `extract_blocks` yields `Block(page, heading_path, text)`. DOCX headings come from `Heading 1-3` styles; PDF headings from regex (`Chương/Bài`, `1.2 …`) plus font size/bold, and lines repeated on more than half the pages (headers/footers) are dropped; each PPTX slide title is its heading. `build_corpus` writes `data/processed/<version_id>/`, where `version_id` hashes the files and the chunking config. Rebuilding an identical corpus reuses it.
3. `chunking.py`: `chunk_blocks` with strategy `structure` (sentences packed within one section, max 350 words, 50 overlap) or `fixed` (450/75 word windows). With `prefix=True`, `text = "Doc > Chapter > Section
" + body`. `chunk_id` hashes doc, strategy, page, heading path, index and body.
4. `index.py`: `RetrievalIndex` stores embeddings (numpy exact, or FAISS `IndexFlatIP`) plus one BM25 token file per tokenizer (`add_tokenizer` adds more without re-embedding). BM25 uses `k1=1.5`, `b=0.75`.
5. `pipeline.py`: `RetrievalPipeline.run(query, PipelineConfig)`: sparse/dense top-L → fusion (`none | rrf | weighted | adaptive`, and `alpha` weights **BM25**) → rerank top-N → the full ordered list. Each `RetrievedChunk` carries `StageScores` (sparse/dense score and rank, fusion score, rerank score, rank), and the result carries `timings_ms`. An optional SQLite `RetrievalCache` stores first-stage lists and rerank scores.
6. `rag.py`: `answer_question(query, pipeline, config, client, model)` → top `context_k` → refusal below `refusal_threshold` → LLM (uses `config.temperature`/`timeout_seconds`) → `_valid_citations`.

**Benchmark** (`src/bench/`): pure functions over `list[dict]`. File I/O goes through `src/io_utils.py`, and CSVs are written as `utf-8-sig` so Excel opens them correctly. Order: `generate → review → pool → agreement → split → describe`. `bench/manifest.check_test_lock` records a sha256 of `frozen_params.yaml` at the first test run and flags later changes (it warns, never blocks). `src/cli.py` is the single argparse entry point.

**State**: `storage.Database` is a thin sqlite3 wrapper (`data/app.db`). Most tables store a JSON payload column. Exactly one `corpus_versions` row is `active`, and Chat loads the index for that version. Named `PipelineConfig`s saved from the RAG Settings page are what Chat uses; `load_pipeline_configs` skips legacy/invalid rows and reports their names. Users are seeded from `.env` on startup with scrypt hashes, and existing users are never overwritten.

**Evaluation** (`src/eval/`): `spec.load_spec` reads the experiment YAML (`configs/experiment.yaml`; `${DATA_DIR}`/`${RUNS_DIR}` placeholders), resolving each config's `frozen` values (`alpha`, `rrf_k`, `adaptive_beta`, `rerank_n`) from `<bench_dir>/frozen_params.yaml` — `eval run --split test` refuses to run until that file exists. `runner.run_evaluation` checkpoints per-query results to `per_query.jsonl` (per-stage scores, see `RESULT_FIELDS`), supports `--resume`/`--only`, calls `bench/manifest.check_test_lock` on the test split (records `config.json`'s `lock.lock_violation`, never blocks), remaps qrels for configs pointing at a different index, and measures latency without the retrieval cache. `tune.tune` grid-searches on the dev split and writes `frozen_params.yaml`. `compare.compare_run` runs paired bootstrap CI + Holm-corrected randomization tests per `comparisons` family. `errors.classify_failures` labels failed queries by pipeline stage (`first_stage_miss | fusion_demoted | rerank_demoted`) and exports a sample for manual `cause` labeling. `report.build_report` writes Chapter 3 tables/figures to `runs/<RUN_ID>/report/`; the table/figure numbering lives in `report.TABLES`/`report.FIGURES` and must match `b_o_c_o_nh_m_3.md` (checked by `tests/test_notebook.py`). The Experiments page (`pages/4_Experiments.py`) lists runs, shows the generated tables/figures, and warns on `lock_violation`.

## Gotchas

- BM25 fixtures in tests need at least 4 chunks, with each query term in only one chunk. `rank_bm25` gives idf ≤ 0 to a term that appears in at least half of the documents.
- `PipelineConfig` validates strictly: any fusion other than `none` needs both branches, and it requires `context_k ≤ rerank_n ≤ top_l`. RAG configs saved to the DB in the old format are skipped.
- python-pptx returns fresh shape proxies, so compare shapes by `shape_id`, never with `is`.
- Windows console is cp1252; set `PYTHONIOENCODING=utf-8` when printing Vietnamese.
- Research protocol from README: don't look at test-split results until tokenizer, chunking, models, fusion params, and thresholds are locked using train/dev.
- `data/raw`, `data/processed`, `data/indexes`, `runs/`, `*.db`, `.env`, and `*.docx` are gitignored. `configs/experiment.example.yaml` points at `examples/index`, which is not a real index and only works with `--dry-run`.
- On Colab, install from `requirements.in`, not `requirements.txt` (which pins the CPU-only torch build via `--torch-backend cpu`).
