# Hệ Thống RAG Cho Tài Liệu Học Tập Tiếng Việt (Vietnamese Learning RAG)

Ứng dụng Streamlit cục bộ phục vụ hỏi đáp có trích dẫn nguồn trên tài liệu học tập tiếng Việt, đồng thời cung cấp môi trường thực nghiệm chuẩn hóa, có thể tái lập để so sánh các phương pháp truy xuất: **BM25 (Sparse)**, **BGE-M3 (Dense)**, **Hybrid Fusion (RRF, Weighted Sum, Adaptive)** và **Reranker (Cross-Encoder)**.

---

## Mục lục

- [Tổng quan tính năng](#tổng-quan-tính-năng)
- [Yêu cầu hệ thống](#yêu-cầu-hệ-thống)
- [Cài đặt & Thiết lập](#cài-đặt--thiết-lập)
- [Cấu trúc lưu trữ dữ liệu](#cấu-trúc-lưu-trữ-dữ-liệu)
- [Khởi chạy ứng dụng Web (Streamlit)](#khởi-chạy-ứng-dụng-web-streamlit)
- [Quy trình dòng lệnh (CLI Workflow)](#quy-trình-dòng-lệnh-cli-workflow)
  - [1. Xây dựng Index từ tài liệu](#1-xây-dựng-index-từ-tài-liệu)
  - [2. Xây dựng bộ Benchmark đánh giá](#2-xây-dựng-bộ-benchmark-đánh-giá)
  - [3. Thực nghiệm, Đánh giá & Báo cáo khoa học](#3-thực-nghiệm-đánh-giá--báo-cáo-khoa-học)
- [Kiểm thử & Đảm bảo chất lượng](#kiểm-thử--đảm-bảo-chất-lượng)

---

## Tổng quan tính năng

- **Phân đoạn theo cấu trúc (Structure-aware Chunking)**: Hỗ trợ tài liệu PDF, DOCX, PPTX. Tự động nhận dạng cấu trúc đề mục, bổ sung tiền tố phân cấp ngữ cảnh (`Tài liệu > Chương > Mục`) giúp khắc phục triệt để hiện tượng mất ngữ cảnh khi phân đoạn tài liệu học tập.
- **Phương pháp truy xuất đa tầng (Multi-stage Retrieval)**:
  - **Sparse Retrieval**: BM25 hỗ trợ các bộ tách từ tiếng Việt (`whitespace`, `pyvi`, `vncorenlp`).
  - **Dense Retrieval**: `BAAI/bge-m3` đa ngôn ngữ, hỗ trợ tìm kiếm vector qua FAISS hoặc NumPy.
  - **Hybrid Fusion**: Hỗ trợ dung hợp xếp hạng nghịch đảo (Reciprocal Rank Fusion - RRF), dung hợp trọng số (Weighted Sum) và dung hợp thích ứng (Adaptive Fusion) dựa trên độ tương đồng giữa câu truy vấn và ngữ liệu.
  - **Reranking**: Mô hình Cross-Encoder `BAAI/bge-reranker-v2-m3` chấm điểm tương quan cặp câu hỏi - đoạn văn để tái xếp hạng top $N$ ứng viên tốt nhất.
- **Giao diện tương tác Streamlit**:
  - Phân quyền Quản trị viên (`admin`) và Sinh viên (`student`).
  - Trực quan hóa cấu trúc đoạn văn bản và quản lý đa phiên bản Corpus.
  - Khung chat hỏi đáp thông minh kèm trích dẫn nguồn minh bạch (đính kèm số trích dẫn `[1]`, tên tài liệu, phân cấp mục, đoạn trích dẫn đối sánh).
  - Trang theo dõi và phân tích các lượt chạy thực nghiệm khoa học.
- **Khung đánh giá khoa học chuẩn xác (Evaluation Framework)**:
  - Khóa tham số tập kiểm thử (`frozen_params.yaml`) chống rò rỉ dữ liệu (data leakage).
  - Kiểm định ý nghĩa thống kê bằng Paired Bootstrap và Randomization Test kèm hiệu chỉnh kiểm định bội Holm-Bonferroni.
  - Phân tích nguyên nhân lỗi truy xuất theo từng giai đoạn đường ống (Sparse, Dense, Fusion, Rerank).

---

## Yêu cầu hệ thống

- **Python**: 3.11
- **API LLM**: Bất kỳ dịch vụ nào tương thích chuẩn OpenAI Chat Completions (OpenAI, OpenRouter, vLLM, Ollama, v.v.).
- *(Tùy chọn)* **Tesseract OCR**: Để trích xuất nội dung từ các trang PDF dạng quét ảnh.
- *(Tùy chọn)* **CUDA GPU**: Mặc định hệ thống chạy mượt mà trên CPU với PyTorch CPU; có thể tận dụng GPU để tăng tốc độ mã hóa vector và reranking.
- *(Tùy chọn)* **Java 8+**: Cần thiết nếu sử dụng bộ tách từ `vncorenlp`.

---

## Cài đặt & Thiết lập

### 1. Khởi tạo môi trường ảo và cài đặt thư viện

Khuyến nghị sử dụng công cụ `uv` để cài đặt nhanh chóng, hoặc dùng `venv` tiêu chuẩn:

**Trên Windows (PowerShell / CMD):**
```powershell
# Tạo môi trường ảo với Python 3.11
uv venv --python 3.11 .venv
# hoặc: py -3.11 -m venv .venv

# Kích hoạt môi trường ảo
.venv\Scripts\activate

# Cài đặt các gói phụ thuộc (chế độ CPU)
uv pip install -r requirements.txt --torch-backend cpu
# hoặc: pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

**Trên Linux / macOS:**
```bash
# Tạo môi trường ảo
uv venv --python 3.11 .venv
# hoặc: python3.11 -m venv .venv

# Kích hoạt môi trường ảo
source .venv/bin/activate

# Cài đặt các gói phụ thuộc (chế độ CPU)
uv pip install -r requirements.txt --torch-backend cpu
# hoặc: pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### 2. Thiết lập biến môi trường `.env`

Tạo tệp `.env` từ tệp mẫu:
```bash
# Windows
copy .env.example .env

# Linux / macOS
cp .env.example .env
```

Cập nhật thông số API và tài khoản đăng nhập trong tệp `.env`:
```env
OPENAI_API_KEY=your-api-key-here
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini
ADMIN_USERNAME=admin
ADMIN_PASSWORD=mat-khau-quan-tri-vien
STUDENT_USERNAME=student
STUDENT_PASSWORD=mat-khau-sinh-vien
```

> **Lưu ý**: Các tệp `.env`, tài liệu tải lên, cơ sở dữ liệu SQLite, vector index và kết quả thực nghiệm đều đã được cấu hình trong `.gitignore` để không bị đẩy lên Git.

---

## Cấu trúc lưu trữ dữ liệu

- **`PPL_DATA_DIR`** (mặc định: `data/`):
  - `data/uploads/`: Lưu trữ các tệp gốc tải lên (PDF, DOCX, PPTX).
  - `data/indexes/<version>/`: Lưu các phiên bản index truy xuất gồm `chunks.jsonl`, `embeddings.npy`, `tokens_<tokenizer>.json`, `index_meta.json` và `faiss.index`.
  - `data/app.db`: Cơ sở dữ liệu SQLite quản lý tài khoản, danh mục tài liệu, corpus đang kích hoạt và cấu hình RAG.
  - `data/cache/retrieval.sqlite`: Bộ nhớ đệm cache kết quả truy xuất và vector hóa để tối ưu tốc độ.
  - `data/benchmark/`: Lưu trữ dữ liệu benchmark (`queries.jsonl`, `qrels.jsonl`, `benchmark_manifest.json`, v.v.).
- **`PPL_RUNS_DIR`** (mặc định: `runs/`): Chứa nhật ký, cấu hình đóng băng, ma trận so sánh và báo cáo của các đợt chạy thực nghiệm.

---

## Khởi chạy ứng dụng Web (Streamlit)

Khởi động giao diện người dùng bằng lệnh tương ứng với hệ điều hành:

**Trên Windows:**
```powershell
.venv\Scripts\streamlit run app.py
# hoặc nếu đã kích hoạt .venv:
streamlit run app.py
```

**Trên Linux / macOS:**
```bash
.venv/bin/streamlit run app.py
# hoặc nếu đã kích hoạt .venv:
streamlit run app.py
```

### Các bước sử dụng trên giao diện:

1. **Đăng nhập (`app.py`)**: Đăng nhập bằng tài khoản quản trị viên (`admin`) hoặc sinh viên (`student`).
2. **Quản lý Tài liệu (`pages/2_Documents.py`)**:
   - Tải lên tài liệu học tập bài giảng (PDF, DOCX, PPTX).
   - Chọn chiến lược cắt đoạn (khuyến nghị mặc định: `Structure-aware` kèm tiền tố phân cấp ngữ cảnh `Tài liệu > Chương > Mục`).
   - Xem trước kết quả cắt đoạn, sau đó bấm **Xây dựng & Kích hoạt Index**.
3. **Cấu hình RAG (`pages/3_RAG_Settings.py`)**:
   - Lựa chọn phương pháp tìm kiếm (BM25, Dense, Hybrid, Adaptive Hybrid).
   - Thiết lập trọng số dung hợp $\alpha$, hệ số RRF $k$, hoặc ngưỡng dung hợp thích ứng $\beta$.
   - Kích hoạt Reranker Cross-Encoder và lưu thành cấu hình đặt tên.
4. **Hỏi đáp sinh viên (`pages/1_Chat.py`)**:
   - Nhập câu hỏi thắc mắc bài học.
   - Nhận câu trả lời tổng hợp kèm các chỉ dẫn trích dẫn cụ thể `[1]`, `[2]`. Nhấp vào từng trích dẫn để đối chiếu đoạn văn gốc trong giáo trình.
5. **Xem kết quả thực nghiệm (`pages/4_Experiments.py`)**:
   - Theo dõi trạng thái các lần chạy thực nghiệm, cảnh báo khóa tập test (`lock_violation`), xem các bảng kết quả thống kê và biểu đồ trực quan.

---

## Quy trình dòng lệnh (CLI Workflow)

Toàn bộ đường ống từ lập chỉ mục, xây dựng benchmark đến thực nghiệm đánh giá được tự động hóa qua module `src.cli`:

### 1. Xây dựng Index từ tài liệu

```bash
# Xây dựng index từ thư mục tài liệu và kích hoạt cho môn học CS101
python -m src.cli index build --input data/courses/CS101 --course CS101
```

### 2. Xây dựng bộ Benchmark đánh giá

- `queries.jsonl`: `query_id, text, category (exact|concept|paraphrase|multi), origin (llm|human), split, source_chunk_ids, evidence, generator`
- `qrels.jsonl`: `query_id, chunk_id, relevance (0|1|2)`
- `evidence.jsonl`: `query_id, doc_id, page, quote, relevance`
- `human_queries.csv`: `text, category[, query_id]`
- `manual_additions.csv`: `query_id, chunk_id`

Annotators fill `relevance` (0 = irrelevant, 1 = supporting, 2 = direct answer) and, when relevance ≥ 1, `evidence_quote` (a verbatim excerpt from the chunk). Have both annotators label at least 30% of the questions so κ is meaningful. Queries with no relevant chunk are dropped at `split`.

## Run experiments

`configs/experiment.yaml` is the experiment spec (matrix C1–C4, X1–X6; see "Build the benchmark" above for `bench_dir`). Tune on dev before ever touching test:

```bash
.venv/bin/python -m src.cli eval run --config configs/experiment.yaml --split test --dry-run   # validate configs/index without loading models
.venv/bin/python -m src.cli eval tune --config configs/experiment.yaml                          # writes <bench_dir>/frozen_params.yaml from the dev split
.venv/bin/python -m src.cli eval run --config configs/experiment.yaml --split test              # locks frozen_params.yaml on first test run
.venv/bin/python -m src.cli eval compare --config configs/experiment.yaml --run RUN_DIR          # paired bootstrap CI + Holm-corrected randomization test
.venv/bin/python -m src.cli eval errors --config configs/experiment.yaml --run RUN_DIR           # error_sample.csv; label its `cause` column, then rerun with --summarize error_sample.csv
.venv/bin/python -m src.cli eval report --config configs/experiment.yaml --run RUN_DIR           # Chapter 3 tables/figures
```

`eval run --split test` refuses to run until `eval tune` has written `frozen_params.yaml`. Changing `frozen_params.yaml` after the first test run does not block later runs, but marks `config.json`'s `lock.lock_violation` and is flagged on the Streamlit **Experiments** page. `--resume RUN_ID` continues an interrupted run without repeating completed queries; `--only C1,C2` restricts which configs run.

Each run writes to `runs/<RUN_ID>/`: `config.json`, `per_query.jsonl` (per-stage scores), `metrics.json`, `metrics_per_query.csv`, `latency.json`, `comparisons.csv`, `error_sample.csv`, and `report/table_3_*.md` / `report/figure_3_*.png` (numbered to match Chapter 3 of `b_o_c_o_nh_m_3.md`).

`run_rag_evaluation.py --config configs/experiment.yaml` still works for blinded LLM-answer scoring (Dense, fixed Hybrid, Adaptive Hybrid + Reranker) on the test split:

```bash
.venv/bin/python run_rag_evaluation.py --config configs/experiment.yaml
```

Human raters fill the empty scoring columns in `rag_answers_blinded.csv`. Keep `rag_answers_key.json` hidden until scoring finishes, then summarize without another API call:

```bash
.venv/bin/python run_rag_evaluation.py --summarize runs/RAG_RUN/rag_answers_blinded.csv
```

## Google Colab

Open `notebooks/colab_pipeline.ipynb` in Colab and run the cells in order. Set `OPENAI_API_KEY`, `OPENAI_BASE_URL`, `OPENAI_MODEL` as Colab Secrets. Data and runs persist on Google Drive via `PPL_DATA_DIR`/`PPL_RUNS_DIR`. Install from `requirements.in` (not `requirements.txt`, which pins the CPU-only torch build).

## Verification

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m src.cli eval run --config configs/experiment.example.yaml --split test --dry-run
.venv/bin/python -m compileall -q app.py pages src run_rag_evaluation.py
```

Tests use fake encoders (`tests/fakes.py`) and never download models.

The example experiment config (`configs/experiment.example.yaml`, pointing at `examples/index`) validates structure only; its index directory is not a real searchable index.
