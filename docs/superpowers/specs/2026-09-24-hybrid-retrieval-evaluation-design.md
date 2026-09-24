# Thiết kế: Hybrid Retrieval theo báo cáo nhóm 3 và bộ đánh giá thực nghiệm

- Ngày: 2026-09-24
- Nguồn yêu cầu: `b_o_c_o_nh_m_3.md` (Chương 1–2) và các quyết định đã thống nhất trong buổi brainstorming
- Phạm vi: nâng cấp pipeline truy xuất, xây quy trình benchmark, bộ đánh giá thống kê, khung Chương 3, notebook Colab

## 1. Mục tiêu và tiêu chí thành công

**Mục tiêu:** code thực thi đúng thiết kế ở Chương 2 của báo cáo (C1–C4, NFC, structure-aware chunking, dev/test, bootstrap, latency P95), đồng thời hỗ trợ các hướng khai thác X1–X6 cho Chương 3.

**Thành công khi:**
1. Từ một kho tài liệu thô, notebook Colab chạy tuần tự được toàn bộ quy trình: lập chỉ mục → sinh câu hỏi nháp → rà soát → pooling → dán nhãn → κ → chia dev/test → tune trên dev → đánh giá test → kiểm định → error analysis → xuất bảng/hình.
2. Mỗi số liệu trong khung Chương 3 có một lệnh sinh ra nó, và kết quả tái lập được (seed, mã băm corpus/benchmark/tham số).
3. App Streamlit dùng cùng pipeline truy xuất với thực nghiệm.
4. Test hành vi chạy xong mà không tải model hay gọi API.

**Ngoài phạm vi:**
- Tối ưu mô hình sinh câu trả lời. `run_rag_evaluation.py` chỉ được sửa ở mức tương thích với `RagConfig` mới.
- Fine-tune embedding hay reranker.
- Vector DB phân tán (Milvus).
- Trang quản lý người dùng.

## 2. Quyết định đã chốt

| Chủ đề | Quyết định |
|---|---|
| Cách làm | Mở rộng code hiện có (Hướng 1); metrics tự viết, có test đối chiếu `ranx` |
| Khoảng trống nghiên cứu | Code hỗ trợ C1–C4 và X1–X6; thêm khung Chương 3 có placeholder vào báo cáo |
| Benchmark | Xây từ đầu: LLM sinh nháp → người rà soát → pooling → dán nhãn đôi → κ → dev/test |
| Hạ tầng | Tính nặng trên Colab Pro (GPU); phân tích được ở máy nhà nhờ cache |
| LLM | OpenAI-compatible (OpenRouter) qua `.env` hiện có |
| Chỉ số chính | MRR@10 (khai báo trước); phụ: NDCG@10, Recall@5; báo cáo đủ Hit/P/R/NDCG @1,3,5,10 |
| Tách từ BM25 | whitespace, pyvi, VnCoreNLP (VnCoreNLP tùy chọn, cần Java) |
| Vector search | numpy exact (mặc định) và FAISS `IndexFlatIP` (tùy chọn); cả hai đều tìm chính xác |
| Rerank N mặc định | 30 (trong khoảng 25–50 của báo cáo) |
| Dev/Test | Chia phân tầng theo nhóm câu hỏi, 30/70, seed 42 |

## 3. Kiến trúc module

```
src/
  text.py          # normalize_text (NFC, ký tự ẩn, khoảng trắng, gạch nối), tokenize(text, mode)
  ingestion.py     # trích xuất PDF/DOCX/PPTX → Block(page, heading_path, text); build_corpus
  chunking.py      # chunk_structure(), chunk_fixed(); prefix breadcrumb tùy chọn
  models.py        # Chunk, StageScores, RetrievedChunk, PipelineConfig (thay RagConfig)
  index.py         # RetrievalIndex: build/load, BM25 theo từng tokenizer, dense (numpy|faiss)
  fusion.py        # minmax, rrf, weighted, adaptive_alpha
  reranking.py     # Reranker với model được tải một lần, có cache điểm
  pipeline.py      # RetrievalPipeline.run(query, config) → list[RetrievedChunk] + timings
  cache.py         # SQLite cache: first-stage top-L và điểm rerank
  rag.py           # answer_question dùng RetrievalPipeline (giữ refusal và citation hiện có)
  storage.py, ui.py, config.py   # giữ nguyên, config thêm PPL_DATA_DIR
  bench/
    generate.py    # sinh câu hỏi nháp theo 4 nhóm, kiểm tra tự động
    review.py      # xuất/nhập CSV rà soát, loại câu gần trùng
    pool.py        # pooling mù từ nhiều hệ thống
    agreement.py   # Cohen's κ có trọng số, file bất đồng, hợp nhất qrels + evidence
    split.py       # chia dev/test, manifest, frozen_params lock
    describe.py    # thống kê mô tả benchmark, mức đóng góp vào pool
    remap.py       # ánh xạ qrels sang cách chunking khác qua evidence quote
  eval/
    metrics.py     # (chuyển từ evaluation.py) Hit/P/R/NDCG@k, MRR@10
    stats.py       # paired bootstrap CI, randomization test, Holm
    runner.py      # chạy cấu hình trên split, checkpoint/resume, lượt đo latency
    tune.py        # quét α, k, β, N trên dev → frozen_params.yaml
    errors.py      # phân loại tầng thất bại, xuất mẫu gắn nhãn nguyên nhân
    report.py      # bảng Markdown/CSV, hình PNG
  cli.py           # python -m src.cli {index,bench,eval} ...
notebooks/colab_pipeline.ipynb
configs/experiment.yaml          # thay experiments.yaml / example_experiments.yaml
```

Các file cũ `evaluation.py`, `experiments.py`, `retrieval.py`, `run_experiments.py` được thay bằng các module trên. README và CLAUDE.md được cập nhật theo. Index và corpus cũ không tương thích và phải build lại. Điều này chấp nhận được vì chúng chỉ là dữ liệu cục bộ đã gitignore.

## 4. Xử lý tài liệu và lập chỉ mục

### 4.1 Chuẩn hóa
`normalize_text` thực hiện:
1. `unicodedata.normalize("NFC")`.
2. Bỏ ký tự điều khiển và zero-width (U+200B–U+200D, U+FEFF, soft hyphen).
3. Nối chữ bị ngắt dòng bằng gạch nối (`-\n` giữa hai chữ cái).
4. Gộp khoảng trắng.

Hàm này được áp dụng cho text tài liệu, câu hỏi benchmark và query lúc chạy.

### 4.2 Nhận diện cấu trúc
Output của bước trích xuất là `Block(page, heading_path: list[str], text)`.

| Định dạng | Cách tìm tiêu đề |
|---|---|
| DOCX | Style `Heading 1..3` (và `Title`); cấp tiêu đề lấy theo style; bảng thành dòng `a \| b` như hiện tại |
| PDF | Dùng `page.get_text("dict")` để lấy span, cỡ chữ và cờ bold. Một dòng là tiêu đề nếu thỏa **một** trong hai điều kiện: (a) khớp regex `^(Chương\|CHƯƠNG\|Bài\|BÀI)\s+[\dIVXLC]+`, hoặc `^\d+(\.\d+){0,3}\.?\s+\S`, và dòng ngắn hơn 120 ký tự; (b) cỡ chữ ≥ trung vị cỡ chữ của tài liệu × 1.2 và dòng ngắn hơn 120 ký tự. Cấp tiêu đề: "Chương/Bài" = 1, số `a.b` = số phần của số. OCR fallback giữ nguyên như hiện tại |
| PPTX | Tiêu đề slide là heading cấp 2 dưới tên tài liệu; thứ tự đọc giữ như hiện tại |

`doc_title` lấy từ metadata của file, nếu không có thì dùng tên file bỏ phần mở rộng.

### 4.3 Chunking
- `structure` (mặc định):
  1. Nhóm các block liên tiếp có cùng `heading_path`.
  2. Trong mỗi nhóm, gom câu (tách câu bằng regex `(?<=[.!?…])\s+`) cho đến `max_words` (mặc định 350) và overlap `overlap_words` (mặc định 50, tính theo câu trọn vẹn).
  3. Không vượt ranh giới mục. Slide ngắn giữ nguyên cả slide.
- `fixed`: cửa sổ từ cố định như hiện tại (`chunk_words` 450, overlap 75), dùng làm đối chứng.
- `prefix: true|false`: nếu bật thì `text = "{doc_title} > {heading_path...}\n{body}"`.
- `chunk_id` = sha256(doc_id | strategy | page | heading_path | index | body)[:24].

`Chunk` gồm các trường `chunk_id, doc_id, doc_title, course, source_type, page, heading_path, body, text`. `text` là nội dung được đưa vào index và LLM.

### 4.4 Index
`data/indexes/<version>/` chứa:
- `chunks.jsonl`
- `embeddings.npy` (L2-normalized), tùy chọn thêm `faiss.index`
- `tokens_<mode>.json` cho mỗi tokenizer đã build
- `index_meta.json`: embedding model, tokenizer có sẵn, `bm25: {k1: 1.5, b: 0.75}`, chunk_count, cấu hình chunking

Thêm tokenizer mới cho index có sẵn bằng `index add-tokenizer`, không cần build lại embedding. `version_id` = sha256(manifest file + cấu hình chunking)[:16].

`PPL_DATA_DIR` (tùy chọn) ghi đè thư mục `data/`, dùng để trỏ vào Google Drive.

## 5. Truy xuất

### 5.1 Pipeline
```
normalize(query)
  → BM25 top-L (tokenizer theo config)      [nếu config dùng sparse]
  → Dense top-L                              [nếu config dùng dense]
  → fusion: none | rrf(k) | weighted(α) | adaptive(α0, β)
  → rerank top-N (tùy chọn)
  → top-K
```

**Quy ước fusion:**
- Danh sách fusion là **hợp** của hai top-L.
- Weighted và adaptive min-max chuẩn hóa từng nhánh trên top-L của nhánh đó; chunk vắng mặt ở một nhánh nhận 0 ở nhánh đó.
- RRF chỉ cộng các nhánh có mặt.
- Khi hòa điểm, sắp theo `chunk_id` tăng dần.

**`adaptive_alpha`** giữ công thức hiện tại, nhưng β lấy từ config và clamp [0.1, 0.9].

Mỗi `RetrievedChunk` mang `StageScores`: `sparse_score, sparse_rank, dense_score, dense_rank, fusion_score, rerank_score, rank` (None nếu tầng không chạy). Ngoài ra pipeline trả `timings_ms: {sparse, dense, fusion, rerank, total}` và `alpha_used` (cho adaptive).

### 5.2 PipelineConfig
Các trường: `sparse: bool, dense: bool, fusion: none|rrf|weighted|adaptive, alpha, rrf_k, adaptive_beta, rerank: bool, rerank_n, top_l, context_k, tokenizer, refusal_threshold, llm_model, temperature`.

`rag.py` dùng `temperature` và `timeout` từ config thay vì ghi cứng. Cấu hình lưu trong SQLite (bảng `rag_configs`) chuyển sang schema này. Cấu hình cũ không đọc được sẽ bị bỏ qua kèm cảnh báo trên trang RAG Settings.

### 5.3 Ma trận cấu hình (khai báo theo tên trong `configs/experiment.yaml`)

| ID | sparse | dense | fusion | rerank | Vai trò |
|---|---|---|---|---|---|
| C1 | ✓ | | none | | Đường cơ sở sparse |
| C2 | | ✓ | none | | Đường cơ sở dense |
| C3-RRF | ✓ | ✓ | rrf | | Hiệu quả dung hợp |
| C3-WS | ✓ | ✓ | weighted | | Hiệu quả dung hợp |
| C4-RRF | ✓ | ✓ | rrf | ✓ | Đóng góp reranker |
| C4-WS | ✓ | ✓ | weighted | ✓ | Đóng góp reranker |
| X1 | | ✓ | none | ✓ | Tách hiệu quả reranker khỏi fusion |
| X2 / X2-R | ✓ | ✓ | adaptive | – / ✓ | Trọng số dung hợp theo truy vấn |
| X3 | C1 và C3-WS với tokenizer ∈ {whitespace, pyvi, vncorenlp} | | | | Tách từ tiếng Việt |
| X4 | C1, C2, C4-WS trên corpus {structure, fixed} × prefix {on, off} | | | | Chunking; qrels ánh xạ qua evidence |
| X5 | C4-WS với N ∈ {10, 20, 30, 50} | | | | Chất lượng theo độ trễ (RQ3) |
| X6 | C2 và C4-WS với embedding ∈ {bge-m3, `x6_embedding_model` trong config — bắt buộc khai báo khi chạy X6, không có mặc định; mỗi model cần một index riêng} | | | | Độ nhạy embedding |

Tham số α, k, β, N của C3/C4/X2 được lấy từ `frozen_params.yaml` sau khi tune.

### 5.4 Cache
SQLite `data/cache/retrieval.sqlite`:
- `first_stage(index_version, branch, tokenizer_or_model, query_hash, top_l) → json[(chunk_id, score)]`
- `rerank(model, query_hash, chunk_id) → score`

Chỉ runner, tune và pool dùng cache. Lượt đo độ trễ và app chat không dùng cache first-stage.

### 5.5 Reranker
Model được tải một lần cho mỗi tên model (cache cấp module). Các call site đều truyền tên model lấy từ config.

## 6. Quy trình benchmark

### 6.1 Schema
- `queries.jsonl`: `query_id, text, category (exact|concept|paraphrase|multi), origin (llm|human), split (dev|test|null), source_chunk_ids, generator`.
- `qrels.jsonl`: `query_id, chunk_id, relevance (0|1|2)` (giữ như hiện tại).
- `evidence.jsonl`: `query_id, doc_id, page, quote, relevance`.
- `benchmark_manifest.json`: sha256 của queries, qrels, evidence; `index_version`; seed; tỉ lệ chia.

### 6.2 Các lệnh
1. **`bench generate --per-category 60`**
   - Chọn chunk phân tầng theo (course, source_type, doc).
   - Gọi LLM với prompt riêng cho từng nhóm; output JSON gồm `question, evidence_quote`.
   - Kiểm tra tự động:
     - `exact`: câu hỏi chứa ít nhất một token có chữ số hoặc viết hoa ≥ 2 ký tự, và token đó có trong chunk.
     - `paraphrase`: Jaccard (token, tokenizer whitespace, bỏ stopword tiếng Việt cơ bản) giữa câu hỏi và chunk ≤ 0.2.
     - `evidence_quote`: phải là chuỗi con của chunk sau khi chuẩn hóa.
     - `multi`: nhận 2–3 chunk cùng `heading_path` cấp 1, hoặc cosine ≥ 0.6. Evidence gồm một quote cho mỗi chunk.
   - Câu không qua kiểm tra bị ghi vào `rejected.jsonl` kèm lý do.
2. **`bench review-export` / `review-import`**
   - CSV có các cột `query_id, category, text, source_text, action(keep|edit|drop), new_text, new_category`.
   - Khi import, câu có cosine giữa hai câu hỏi ≥ 0.92 (dùng model dense) được đánh dấu trùng và giữ câu đầu.
   - Câu hỏi do người viết được thêm vào cùng file với `origin=human`. Mục tiêu 25–30% tổng số.
3. **`bench pool --depth 15`**
   - Gộp top-15 của C1, C2, C3-RRF, C3-WS, C4-WS và X1 (tham số mặc định vì chưa tune), cộng source chunks và các chunk người dán nhãn thêm thủ công (file `manual_additions.csv`).
   - Xáo trộn với seed.
   - Mỗi annotator nhận một CSV: `pool_id, query_id, query, heading_path, chunk_text, relevance, evidence_quote`. Map `pool_id → chunk_id, systems` lưu riêng, không giao cho annotator.
4. **`bench agreement`**
   - Nhập CSV của 2 annotator.
   - Tính κ có trọng số bậc hai trên phần giao, cùng tỉ lệ đồng ý thô.
   - Xuất `disagreements.csv`. Sau khi cả nhóm điền cột `final`, tạo `qrels.jsonl` và `evidence.jsonl`.
   - Phần không dán đôi lấy nhãn của annotator duy nhất.
5. **`bench split --dev 0.3 --seed 42`**
   - Chia phân tầng theo category, ghi `split` vào queries, tạo manifest.
6. **`bench describe`**
   - Bảng mô tả: số câu theo category × origin × split, số chunk liên quan trung bình, κ.
   - **Mức đóng góp vào pool:** với mỗi hệ thống, số chunk liên quan (relevance ≥ 1) chỉ hệ thống đó tìm thấy.

### 6.3 Khóa tập test
- `eval tune` ghi `frozen_params.yaml`.
- Lần đầu `eval run --split test` ghi sha256 của file này vào `benchmark_manifest.json` (`test_lock`).
- Các lần chạy test sau, nếu sha256 khác thì in cảnh báo rõ và ghi `lock_violation: true` vào `config.json` của run. Không chặn, vì nhóm có thể cần chạy lại vì lỗi kỹ thuật, nhưng việc vi phạm luôn được ghi lại.
- Chạy test khi chưa có `frozen_params.yaml` sẽ báo lỗi.

### 6.4 Ánh xạ qrels khi đổi chunking (X4)
Với corpus khác, chunk `c` nhận `relevance = max(evidence.relevance)` trên các evidence của query mà `normalize(quote)` là chuỗi con của `normalize(c.body)` và `c.doc_id` trùng với evidence. Nếu quote bị cắt ngang giữa hai chunk, cả hai chunk nhận mức liên quan khi mỗi chunk chứa ≥ 60% số từ của quote (so khớp theo cửa sổ từ liên tiếp dài nhất).

## 7. Đánh giá

### 7.1 Metrics
- Giữ các công thức hiện có. NDCG dùng gain `2^rel − 1`. MRR@10 tính trên relevance ≥ 1.
- Precision@k chia cho k.
- Recall@k với câu hỏi không có chunk liên quan: loại câu đó khỏi benchmark khi `bench split` và ghi cảnh báo.

### 7.2 `eval tune --split dev`
Lưới mặc định:
- α ∈ {0.0, 0.1, …, 1.0}
- k ∈ {10, 20, 40, 60, 100}
- β ∈ {0.1, 0.2, 0.3, 0.5}
- N ∈ {10, 20, 30, 50}

Cách chọn: chọn tham số tối đa hóa MRR@10 trên dev; khi hòa thì chọn giá trị gần mặc định (α 0.5, k 60, β 0.3, N 30). Output gồm `frozen_params.yaml`, `tune_results.csv`, hình α theo metric (tổng và theo category) và hình N theo metric/latency.

### 7.3 `eval run --split test`
- Chạy các cấu hình trong YAML và lưu per-query log (JSONL, gồm StageScores của top-L sau fusion hoặc top-N sau rerank).
- Metrics tổng, theo category và theo origin.
- Resume bằng checkpoint như hiện nay.

**Lượt đo latency:**
- 5 truy vấn warm-up, sau đó đo toàn bộ truy vấn test, không dùng cache.
- `torch.cuda.synchronize()` trước mỗi mốc thời gian nếu có CUDA.
- Ghi mean, P50, P95 cho từng tầng, cùng thông tin phần cứng (tên GPU và CPU, phiên bản torch).

### 7.4 `eval compare`
Các cặp so sánh khai báo trong YAML. Mặc định:
- RQ2: C3-RRF và C3-WS so với `best_single`, là cấu hình tốt hơn giữa C1 và C2 theo MRR@10 **trên dev**.
- RQ3: C4-RRF so với C3-RRF, C4-WS so với C3-WS, X1 so với C2.
- X2: X2 so với C3-WS, X2-R so với C4-WS.

Với mỗi cặp và mỗi metric chính/phụ, báo cáo:
- Hiệu số trung bình.
- CI 95% bằng paired bootstrap (10.000 mẫu, seed).
- p-value bằng paired randomization test (10.000 hoán vị, seed).
- p đã điều chỉnh Holm trong từng họ RQ.

Kết quả được tính cho toàn bộ và theo từng category. Riêng theo category, chỉ báo cáo CI và ghi chú cỡ mẫu nhỏ.

### 7.5 `eval errors`
Truy vấn thất bại là truy vấn không có chunk liên quan trong top-10 của cấu hình mục tiêu (mặc định C4-WS). Mỗi truy vấn thất bại được gán một trong các tầng sau, dựa vào StageScores của chunk liên quan tốt nhất:
- `first_stage_miss`: không có trong top-L của cả hai nhánh.
- `fusion_demoted`: có trong top-L của ít nhất một nhánh nhưng ngoài top-N sau fusion.
- `rerank_demoted`: có trong top-N nhưng reranker đẩy ra ngoài top-10.

Lệnh xuất `error_sample.csv` (tối đa 50 câu, phân tầng theo category), có cột `cause` để nhóm gắn nhãn thủ công với các giá trị: `extraction, chunk_boundary, tokenization, vocabulary_mismatch, multi_hop, label_error, other`. `eval errors --summarize` tổng hợp file đã gắn nhãn.

### 7.6 `eval report`
Sinh `runs/<id>/report/` gồm:
- Bảng Markdown/CSV: kết quả chính, theo category, theo origin, so sánh thống kê, latency, benchmark, κ, pool contribution, error stages và causes.
- Hình PNG (matplotlib): đường α, đường N theo latency, cột theo category.

Mỗi bảng có mã khớp với placeholder trong Chương 3 (ví dụ `Bảng 3.4`).

## 8. Khung Chương 3 trong báo cáo
Chương 3 được thêm vào cuối `b_o_c_o_nh_m_3.md` (trước danh mục tài liệu tham khảo). Mục lục, danh mục bảng và danh mục hình được cập nhật. Các mục:
- 3.1 Môi trường và thiết lập
- 3.2 Bộ dữ liệu và benchmark
- 3.3 Tinh chỉnh trên dev
- 3.4 Kết quả (3.4.1 RQ1, 3.4.2 RQ2, 3.4.3 RQ3)
- 3.5 Phân tích lỗi
- 3.6 Hướng khai thác X1–X6
- 3.7 Thảo luận và threats to validity
- 3.8 Kết luận chương

Quy ước:
- Mọi số liệu dùng dạng `[[ĐIỀN: <mô tả> — nguồn: <lệnh / file>]]`. Không có số liệu bịa.
- Phần threats to validity nêu sẵn: câu hỏi do LLM sinh, cỡ mẫu theo category, pooling depth, một bộ tài liệu/môn học, phần cứng Colab ảnh hưởng latency.
- Bảng 2.2 được bổ sung ghi chú về biến thể RRF/WS. Nội dung Chương 1–2 không đổi.

## 9. Notebook Colab
`notebooks/colab_pipeline.ipynb`:
1. Mount Drive, clone hoặc pull repo, `pip install -r requirements.txt`.
2. Đặt `PPL_DATA_DIR=/content/drive/MyDrive/ppl-data`; đọc `OPENAI_*` từ Colab Secrets.
3. Kiểm tra GPU.
4. Mỗi bước trong mục 6.2 và 7 là một cell gọi `python -m src.cli ...`, có markdown giải thích. Các bước cần người (rà soát, dán nhãn) ghi rõ file cần tải về hoặc upload.

## 10. Thay đổi app Streamlit
- **Documents:** chọn chiến lược chunking (mặc định structure + prefix); hiển thị breadcrumb trong phần xem trước.
- **RAG Settings:** form theo `PipelineConfig`.
- **Chat:** trích dẫn hiển thị `doc_title > heading_path, trang X`.
- **Experiments:** tạo `configs/generated_experiment.yaml` từ template mới; liệt kê các run và hiển thị bảng Markdown trong `report/`.

`run_rag_evaluation.py`: đổi `SYSTEMS` sang `PipelineConfig` (C2, C3-WS, X2-R), phần còn lại giữ nguyên.

## 11. Phụ thuộc
- **Thêm vào `requirements.in`:** `matplotlib`, `faiss-cpu`, `py_vncorenlp`, `ranx` (dùng cho test đối chiếu).
- VnCoreNLP chỉ được import khi chọn tokenizer `vncorenlp`, nên máy không có Java vẫn chạy các phần khác.

## 12. Kiểm thử
Test hành vi trong `tests/`, không tải model (encoder và reranker giả, client LLM giả):
1. **Chuẩn hóa:** chuỗi NFD tiếng Việt, zero-width và gạch nối ngắt dòng được đưa về dạng chuẩn.
2. **Structure chunking:** DOCX có Heading 1/2 và PDF có "Chương 1" / "1.2. …" cho ra `heading_path` đúng; chunk không vượt ranh giới mục; bật/tắt prefix.
3. **Pipeline:** các cấu hình C1/C3-WS/C4-WS với index giả cho StageScores đầy đủ và đúng thứ tự; quy ước chunk vắng mặt nhận 0.
4. **Metrics:** so với `ranx` trên run/qrels ngẫu nhiên có seed (MRR@10, NDCG@k, Recall@k, Precision@k, Hit@k).
5. **Thống kê:** κ khớp giá trị tính tay trên ví dụ nhỏ; bootstrap và randomization tất định theo seed, và p ≈ 1 khi hai hệ thống giống hệt nhau.
6. **Benchmark:** kiểm tra paraphrase/exact loại đúng câu; split phân tầng; remap qrels qua evidence.
7. **Runner:** resume sau khi bị ngắt (giữ test hiện có); lock test cảnh báo khi `frozen_params.yaml` bị sửa.
8. **RAG:** citation không hợp lệ bị loại và refusal theo ngưỡng (giữ test hiện có, chuyển sang `PipelineConfig`).

Lệnh kiểm tra đầy đủ: `pytest -q` cộng `python -m src.cli eval run --config configs/experiment.example.yaml --dry-run`.

## 13. Rủi ro
- **Heuristic tiêu đề PDF có thể sai** với giáo trình trình bày lạ. Giảm thiểu: xem trước breadcrumb ở trang Documents, và có chiến lược `fixed` để dự phòng.
- **LLM sinh câu hỏi có thể thiên về BM25.** Giảm thiểu: kiểm tra paraphrase, thêm câu hỏi do người viết, phân tích theo origin.
- **Cỡ mẫu theo category (khoảng 35 câu test mỗi nhóm) cho CI rộng.** Ghi rõ trong threats to validity.
- **Colab bị ngắt giữa chừng.** Giảm thiểu: checkpoint, resume và cache trên Drive.
