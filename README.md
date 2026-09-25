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
uv pip install --python .venv/bin/python -r requirements.txt --torch-backend cpu      # Linux/macOS
uv pip install --python .venv/Scripts/python.exe -r requirements.txt --torch-backend cpu # Windows
cp .env.example .env
```

VnCoreNLP (optional tokenizer) needs Java 8+; its model downloads to `vncorenlp/` or `PPL_VNCORENLP_DIR`.

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

## Data locations

- `PPL_DATA_DIR` (default `data/`) holds uploads, processed corpora, indexes, `app.db`, and the retrieval cache (`cache/retrieval.sqlite`).
- `PPL_RUNS_DIR` (default `runs/`) holds experiment runs.
- An index lives in `data/indexes/<version>/`: `chunks.jsonl`, `embeddings.npy`, `tokens_<tokenizer>.json`, `index_meta.json`, and optionally `faiss.index`.

**Indexes built before the structure-aware pipeline are not compatible — rebuild them from the Documents page.**

## Run the application

```bash
.venv/bin/streamlit run app.py
```

1. Sign in as administrator.
2. Open **Documents**, upload PDF/DOCX/PPTX, choose the chunking strategy (default: structure-aware with `Document > Chapter > Section` prefix), preview extraction, then build and activate the index.
3. Open **RAG Settings**, choose BM25/Dense/Hybrid, the fusion method and the reranker, and save a named configuration.
4. Sign in as student or remain admin, then use **Chat**.

The first index build downloads `BAAI/bge-m3`. Enabling reranking downloads `BAAI/bge-reranker-v2-m3`. The default reranks the top 30 fused candidates.

## Build the benchmark

Each command reads and writes files in `<data_dir>/benchmark/` (override with `--bench DIR`) and uses the active index (override with `--index DIR`):

```bash
python -m src.cli index build --input path/to/docs --course CS101        # index + activate
python -m src.cli bench generate --per-category 60                         # LLM drafts → drafts.jsonl, rejected.jsonl
python -m src.cli bench review-export                                      # review.csv: action keep/edit/drop
python -m src.cli bench review-import --human human_queries.csv            # queries.jsonl (+ human-written questions)
python -m src.cli bench pool --depth 15 --annotators A,B                   # annotation_A.csv, annotation_B.csv
python -m src.cli bench agreement --annotations annotation_A.csv annotation_B.csv [--resolved disagreements.csv]
python -m src.cli bench split --dev 0.3 --seed 42                          # dev/test + benchmark_manifest.json
python -m src.cli bench describe                                           # benchmark_description.json
python -m src.cli bench remap --target-index DIR --out qrels_remapped.jsonl  # relabel for another chunking
```

File formats:

- `queries.jsonl`: `query_id, text, category (exact|concept|paraphrase|multi), origin (llm|human), split, source_chunk_ids, evidence, generator`
- `qrels.jsonl`: `query_id, chunk_id, relevance (0|1|2)`
- `evidence.jsonl`: `query_id, doc_id, page, quote, relevance`
- `human_queries.csv`: `text, category[, query_id]`
- `manual_additions.csv`: `query_id, chunk_id`

Annotators fill `relevance` (0 = irrelevant, 1 = supporting, 2 = direct answer) and, when relevance ≥ 1, `evidence_quote` (a verbatim excerpt from the chunk). Have both annotators label at least 30% of the questions so κ is meaningful. Queries with no relevant chunk are dropped at `split`.

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

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q app.py pages src run_experiments.py run_rag_evaluation.py
```

Tests use fake encoders (`tests/fakes.py`) and never download models.

The example experiment config validates structure only; its index directory is not a real searchable index.
