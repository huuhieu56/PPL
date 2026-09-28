# Vietnamese Learning RAG

Local Streamlit application for grounded question answering over Vietnamese learning materials and reproducible comparison of BM25, dense, hybrid, adaptive hybrid, and reranked retrieval.

## Requirements

- Python 3.11
- An OpenAI-compatible chat API
- Qdrant running locally on port 6333
- Tesseract with `vie` and `eng` language packs for scanned PDF pages
- Optional CUDA GPU; CPU installation below is reproducible but slower

## Setup

```bash
conda create -n ppl-rag python=3.11 -y
uv pip install --python "$(conda run -n ppl-rag which python | tail -1)" -r requirements.txt --torch-backend cpu
cp .env.example .env
mkdir -p data/qdrant
docker run -d --restart unless-stopped --name ppl-rag-qdrant -p 127.0.0.1:6333:6333 -v "$PWD/data/qdrant:/qdrant/storage" qdrant/qdrant
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

If the container already exists, use `docker start ppl-rag-qdrant`. `.env`, local teaching documents, indexes, SQLite databases, and experiment outputs are ignored by Git.

On this machine the GTX 1650 Max-Q works with CUDA 11.8 after replacing CPU Torch in the same Conda environment:

```bash
uv pip install --python "$(conda run -n ppl-rag which python | tail -1)" --torch-backend cu118 'torch==2.7.1+cu118'
conda run -n ppl-rag python -c 'import torch; print(torch.cuda.is_available())'
```

Keep the CPU Torch wheel in `requirements.txt` for the portable base install; run the CUDA command afterwards if this GPU/driver combination is present. BGE-M3 embedding on CUDA uses FP16 and batch size 1 on this 4 GB GPU; CPU keeps full precision.
Run only one GPU embedding/indexing process at a time on this card. Stop Streamlit before a separate GPU benchmark, or set `CUDA_VISIBLE_DEVICES=''` for the benchmark to run on CPU.

## Run the application

```bash
conda run -n ppl-rag streamlit run app.py
```

1. Sign in as administrator.
2. Open **Documents**, enter a course and upload PDF/DOCX/PPTX. The app automatically builds and activates the index; preview and the choice of previously indexed files to retain are optional. A failed build leaves the previous active corpus unchanged.
3. Open **RAG Settings** and save a named configuration.
4. Sign in as student or remain admin, then use **Chat**. Select documents in the sidebar, or leave the selection empty to search the whole active corpus. Both BM25 and Qdrant respect the selection. Conversations are stored per user, corpus version and document selection; follow-up questions are rewritten using recent turns.

Answers without any valid source citation are rejected. A valid citation number alone does not prove that every claim is supported: open the cited passage to verify it. Restart Streamlit after changing retrieval code so cached index objects use the new implementation.

The current lexical retriever uses BM25L: unlike Okapi's zero/negative IDF on very small corpora, it remains usable when selecting a single short document. Historical experiment outputs used the earlier lexical implementation; do not treat them as measurements of this revision. Benchmark reruns are a separate step.

The first index build downloads `BAAI/bge-m3`. Enabling reranking downloads `BAAI/bge-reranker-v2-m3`. The vector index is stored in Qdrant; the app and experiment CLI use the same local service.
After the model is cached, `HF_HUB_OFFLINE=1 conda run -n ppl-rag streamlit run app.py` avoids Hugging Face network checks on this local demo machine.

To index all local example slides under `data/sample_slides/<course>/*.pdf` plus PDF/DOCX/PPTX materials under `data/sample_materials/<course>/` (when present):

```bash
conda run -n ppl-rag python build_sample_corpus.py
```

OCR reads text only; diagrams and formulas embedded in images are not interpreted.

Chat and answer evaluation use `langchain-openai` (`ChatOpenAI`) with the configured compatible API. Retrieval remains deterministic and shared by chat/experiments, not delegated to an unbounded agent. Optional LangSmith tracing is off by default. To enable it, set `LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`, and `LANGSMITH_PROJECT`; traces include questions, answers and supplied document context, so enable it only when those may be uploaded. See the [official integration](https://docs.langchain.com/oss/python/integrations/chat/openai).

The local demo files cover databases, introductory AI, image processing, Party history, and writing skills. The current chunker creates document → heading (when available) → page/slide → leaf chunks; DOCX headings contain paragraph/table text. Only leaves are embedded in Qdrant. Answer generation expands nearby leaves and, when the word budget allows, adjacent pages in the same section. `chunk_tokens`/`overlap_tokens` count whitespace-separated words, not model tokens. DOCX citations use content-block ordinals, not fabricated page numbers. This demo does not establish coverage across all subjects or school levels, and the VieQuAD result does not measure this corpus or its chunking.

## Benchmark schema

Each line of `queries.jsonl` must contain:

```json
{"query_id":"q1","text":"Câu hỏi","category":"concept","split":"test","group_id":"source-book-1"}
```

Each line of `qrels.jsonl` represents one graded relevance judgment:

```json
{"query_id":"q1","chunk_id":"chunk-id","relevance":2}
```

Relevance is `0` (irrelevant), `1` (supporting), or `2` (direct answer). Create these files against a frozen corpus; the repository does not ship example labels as a benchmark.

For independently authored questions, create a blinded pool from the selected retrievers on the frozen corpus (this step does not need qrels):

```bash
conda run -n ppl-rag python build_judgment_pool.py --config configs/generated_experiment.yaml --split dev --depth 10
```

Give raters only `judgments.csv`, not `pool_key.json`. Fill relevance, reviewer and notes after reading the evidence; blank labels cannot be exported. Review possible relevant chunks beyond the pool and use a second rater/adjudication for disagreements. The tool does not generate questions or label relevance. Export reviewed labels:

```bash
conda run -n ppl-rag python build_judgment_pool.py --judgments runs/JUDGMENT_RUN/judgments.csv
```

Point the experiment config at the resulting `.qrels.jsonl`. Freeze document-disjoint dev/test questions before tuning. Pooling is incomplete by nature; report its depth, participating methods and judgment coverage rather than claiming exhaustive gold labels.

## Run retrieval experiments

The Streamlit **Experiments** page creates `configs/generated_experiment.yaml` and displays the command. Validate before loading models:

```bash
conda run -n ppl-rag python run_experiments.py --config configs/generated_experiment.yaml --split dev --dry-run
```

Run E0–E7:

```bash
conda run -n ppl-rag python run_experiments.py --config configs/generated_experiment.yaml --split dev
```

Tune fusion on dev only, then freeze the selected file before the test run:

```bash
conda run -n ppl-rag python tune_retrieval.py --config configs/generated_experiment.yaml --experiment E3
conda run -n ppl-rag python tune_retrieval.py --config configs/generated_experiment.yaml --experiment E6
conda run -n ppl-rag python run_experiments.py --config runs/TUNING_RUN/selected_config.yaml --split test
```

E3 searches alpha; E6 searches alpha and `adaptive_beta`. Default grid is alpha `{0, .25, .5, .75, 1}`, beta `{-.5, -.25, 0, .25, .5}`; override with `--alphas`/`--betas` before evaluation. The tuner records dev-only trials, query IDs, selection metric/tie-break and input hashes. Reranking is an explicit separate comparison, not silently mixed into the fusion search.

Resume an interrupted run:

```bash
conda run -n ppl-rag python run_experiments.py --config configs/generated_experiment.yaml --split dev --resume RUN_ID
```

Every run writes frozen config, input hashes, checkpoint, status, `metrics.json`, `per_query.csv`, and errors (if any) under `runs/RUN_ID/`. Resume rejects changed configuration, queries, qrels or index files. Every query must carry a train/dev/test split; one run evaluates only its selected split. Duplicate question text or the same supplied `group_id` across splits is rejected. Supply group IDs from source books/duplicate document families to enforce source-disjoint evaluation. Do not inspect test-set results until tokenizer, chunking, models, fusion parameters, thresholds, and primary comparisons have been locked using train/dev; then use `--split test`.

To compare a selected retrieval method in LangSmith, set `LANGSMITH_API_KEY` in `.env`, then run an explicit split:

```bash
conda run -n ppl-rag python run_langsmith_eval.py --config configs/generated_experiment.yaml --experiment E0 --split dev --dry-run
conda run -n ppl-rag python run_langsmith_eval.py --config configs/generated_experiment.yaml --experiment E0 --split dev
```

The runner uploads query text, qrels, rankings and run metadata to LangSmith; it does not upload the source files. Run `--split test` only after selecting and freezing parameters on dev. `--dry-run` validates/counts locally and does not require a LangSmith key. The example files are schema examples, not a research benchmark.

For a separate Vietnamese public retrieval check, download the pinned VieQuADRetrieval corpus/queries/qrels and run the dev-only alpha selection plus held-out query evaluation:

```bash
hf download mteb/VieQuADRetrieval corpus/validation-00000-of-00001.parquet queries/validation-00000-of-00001.parquet qrels/validation-00000-of-00001.parquet --repo-type dataset --revision f956535 --local-dir data/external/VieQuADRetrieval
conda run -n ppl-rag python run_viequad_benchmark.py
```

VieQuAD is Wikipedia text, **not** a benchmark of the PDF/DOCX/PPTX/OCR ingestion pipeline. An independently judged, multi-subject learning-material retrieval test set is still needed before claiming an accuracy gain for this application.
The measured dev/test result and limitations are recorded in `docs/EXPERIMENTS_VIEQUAD.md`.

## Evaluate RAG answers

Use the selected experiments and tuned parameters from the same index/query configuration to produce blinded outputs (no hard-coded alternative systems):

```bash
conda run -n ppl-rag python run_rag_evaluation.py --config configs/generated_experiment.yaml --split dev
```

Human raters fill the empty scoring columns in `rag_answers_blinded.csv`. Keep `rag_answers_key.json` hidden until scoring finishes, then summarize without another API call:

```bash
conda run -n ppl-rag python run_rag_evaluation.py --summarize runs/RAG_RUN/rag_answers_blinded.csv
```

## Verification

Run the behavior tests and syntax check:

```bash
conda run -n ppl-rag python -m pytest -q
conda run -n ppl-rag python -m compileall -q app.py pages src run_experiments.py run_rag_evaluation.py
```
