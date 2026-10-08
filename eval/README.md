# Ragas evaluation

Chạy lệnh từ thư mục gốc PPL. Cần cấu hình `.env` và có PDF nguồn trong `data/` theo đường dẫn ở `generate_dataset.py`.

## Cài dependency

```bash
.venv/bin/python -m pip install -r requirements-eval.txt
```

## Sinh 25 mẫu và chạy đánh giá

```bash
.venv/bin/python -u eval/generate_dataset.py &&
.venv/bin/python -u eval/run_eval.py
```

`generate_dataset.py` đọc PDF, tạo chunk và index mới bằng code hệ thống, rồi dùng Ragas sinh benchmark. Lệnh này xóa dữ liệu sinh trước đó trong `eval/`.

## Chạy lại đánh giá

Nếu đã có benchmark và index trong `eval/`, chỉ chạy:

```bash
.venv/bin/python -u eval/run_eval.py
```

Lệnh này chạy lại RAG và chấm Faithfulness, Answer Relevancy, Context Precision, Context Recall; ghi đè kết quả đánh giá cũ.

## Kết quả

- `eval/dataset25.jsonl`: benchmark do Ragas sinh.
- `eval/results/responses.jsonl`: câu trả lời và context truy xuất.
- `eval/results/scores.jsonl`: điểm từng mẫu.
- `eval/results/summary.json`: điểm trung bình và số điểm thiếu.

`scores.jsonl` và `summary.json` được lưu khi đánh giá hoàn tất.

```bash
cat eval/results/summary.json
```
