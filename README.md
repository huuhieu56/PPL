# Vietnamese Learning RAG

Local Streamlit application for grounded question answering over Vietnamese learning materials and reproducible comparison of BM25, dense, hybrid, adaptive hybrid, and reranked retrieval.

## Requirements

- Python 3.11
- An OpenAI-compatible chat API
- Optional: Tesseract for OCR of scanned PDF pages
- Optional: a CUDA GPU; CPU mode is supported and is the default installation below

## Setup

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt --torch-backend cpu
cp .env.example .env
```

Set these values in `.env`:

```env
OPENAI_API_KEY=your-key
OPENAI_BASE_URL=https://your-provider.example/v1
OPENAI_MODEL=your-model
ADMIN_USERNAME=admin
ADMIN_PASSWORD=change-this-password
STUDENT_USERNAME=student
STUDENT_PASSWORD=change-this-password
```

`.env`, uploaded documents, indexes, SQLite databases, and experiment outputs are ignored by Git.

## Run the application

```bash
.venv/bin/streamlit run app.py
```

1. Sign in as administrator.
2. Open **Documents**, upload PDF/DOCX/PPTX, preview extraction, then build and activate the index.
3. Open **RAG Settings** and save a named configuration.
4. Sign in as student or remain admin, then use **Chat**.

The first index build downloads `BAAI/bge-m3`. Enabling reranking downloads `BAAI/bge-reranker-v2-m3`. On CPU, the interactive default reranks 20 candidates.

## Benchmark schema

Each line of `queries.jsonl` must contain:

```json
{"query_id":"q1","text":"Câu hỏi","category":"concept","split":"test"}
```

Each line of `qrels.jsonl` represents one graded relevance judgment:

```json
{"query_id":"q1","chunk_id":"chunk-id","relevance":2}
```

Relevance is `0` (irrelevant), `1` (supporting), or `2` (direct answer). Example-only files are under `data/benchmark/`; replace their chunk IDs with IDs from the active corpus.

## Run retrieval experiments

The Streamlit **Experiments** page creates `configs/generated_experiment.yaml` and displays the command. Validate before loading models:

```bash
.venv/bin/python run_experiments.py --config configs/generated_experiment.yaml --dry-run
```

Run E0–E7:

```bash
.venv/bin/python run_experiments.py --config configs/generated_experiment.yaml
```

Resume an interrupted run:

```bash
.venv/bin/python run_experiments.py --config configs/generated_experiment.yaml --resume RUN_ID
```

Every run writes frozen config, checkpoint, status, `metrics.json`, `per_query.csv`, and `errors.csv` under `runs/RUN_ID/`. Do not inspect test-set results until tokenizer, chunking, models, fusion parameters, thresholds, and primary comparisons have been locked using train/dev.

## Evaluate RAG answers

Use the same index/query configuration to produce blinded outputs for Dense, fixed Hybrid, and Adaptive Hybrid + Reranker:

```bash
.venv/bin/python run_rag_evaluation.py --config configs/generated_experiment.yaml
```

Human raters fill the empty scoring columns in `rag_answers_blinded.csv`. Keep `rag_answers_key.json` hidden until scoring finishes, then summarize without another API call:

```bash
.venv/bin/python run_rag_evaluation.py --summarize runs/RAG_RUN/rag_answers_blinded.csv
```

## Verification

The project intentionally has four behavior-level tests rather than tests generated per function:

```bash
.venv/bin/python -m pytest tests/test_core.py -q
.venv/bin/python run_experiments.py --config configs/example_experiments.yaml --dry-run
.venv/bin/python -m compileall -q app.py pages src run_experiments.py run_rag_evaluation.py
```

The example experiment config validates structure only; its index directory is not a real searchable index.


