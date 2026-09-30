# Trợ lý hỏi đáp tài liệu học tập

Ứng dụng RAG cho PDF, DOCX và PPTX: trích xuất chữ (OCR cho PDF scan), chunk theo cấu trúc, tìm bằng BM25/Qdrant và trả lời có dẫn nguồn. Không diễn giải nội dung hình ảnh.

## Chạy local

```bash
conda create -n ppl-rag python=3.11 -y
conda run -n ppl-rag python -m pip install -r requirements.txt
cp .env.example .env
conda run -n ppl-rag streamlit run app.py
```

Điền LLM API, embedding API và Qdrant endpoint/key trong `.env`. Trên VM 1 GB RAM, dùng `EMBEDDING_PROVIDER=api`; không bật reranker local. Nếu chạy model trên máy cá nhân đủ mạnh, cài thêm `sentence-transformers` (và `pyvi` nếu chọn tokenizer đó), rồi đặt `EMBEDDING_PROVIDER=local`.

Muốn chạy bằng Docker trên máy đã có image `ppl-rag:local`:

```bash
docker build -t ppl-rag:local .
docker compose up -d
```

Docker lưu file upload, SQLite và chỉ mục BM25 trong named volume `app_data`; Qdrant chỉ lưu vector. Không dùng `docker compose down -v` nếu muốn giữ học liệu.

## Sử dụng

1. Đăng nhập admin, mở **Tài liệu**, nhập môn học và tải PDF/DOCX/PPTX. Hệ thống tự lập chỉ mục; nếu lỗi, corpus cũ vẫn hoạt động.
2. Mở **Hỏi đáp**, chọn tài liệu cần tìm hoặc tìm toàn bộ corpus. Kiểm tra câu trả lời bằng đoạn trích dẫn nguồn.
3. Nếu cần, thay phương pháp truy xuất trong **Cấu hình truy xuất**.

DOCX dẫn nguồn theo thứ tự khối nội dung, không dùng số trang giả. Chưa có benchmark hay tuyên bố tăng độ chính xác truy xuất.
