# Trợ lý hỏi đáp tài liệu học tập

Ứng dụng RAG chạy bằng Streamlit: tải PDF, DOCX hoặc PPTX; trích xuất văn bản (OCR cho PDF scan); chia đoạn theo cấu trúc tài liệu; lập chỉ mục BM25 + Qdrant; hỏi đáp có dẫn nguồn. OCR chỉ đọc chữ, không diễn giải sơ đồ hoặc hình ảnh.

## Triển khai lên Ubuntu VM

Máy chủ cần Docker Engine + Compose, tên miền trỏ về VM, cổng 80/443 mở và tài khoản SSH chạy được Docker. Tạo thư mục triển khai một lần, cho tài khoản SSH quyền ghi vào đó:

```bash
sudo install -d -o "$USER" -g "$USER" /opt/ppl-rag
```

Trong GitHub **Settings → Secrets and variables → Actions**, đặt hai Variables `VM_HOST`, `VM_USER` và hai Secrets `VM_SSH_KEY`, `APP_ENV`. `APP_ENV` là nguyên nội dung file `.env` theo `.env.example`, gồm domain, endpoint/model và các API key/mật khẩu. Không commit file này.

Mỗi lần push `main`, Actions chạy test và build Docker image **trên GitHub**. Khi đã đặt `VM_HOST`, bước deploy nén image thành tar.gz, chuyển qua SSH, `docker load` và khởi động Compose với `.env` lấy từ `APP_ENV`. Có thể chạy lại thủ công bằng **Actions → CI/CD → Run workflow** sau khi cấu hình Secrets/Variables. Lần kết nối SSH đầu của mỗi runner dùng `accept-new` (chấp nhận host key được máy chủ đưa ra); nếu cần xác thực host chặt hơn thì bổ sung host key đã kiểm tra sau.

Caddy phục vụ HTTPS; chỉ nó mở cổng công khai. Qdrant Cloud lưu vector, không lưu file gốc. File upload, SQLite và chỉ mục BM25 nằm trong Docker named volume `app_data`, tồn tại qua deploy/restart và `docker compose down` thông thường; **không dùng `down -v`** nếu muốn giữ học liệu. Cần sao lưu volume này riêng. Mật khẩu admin/student trong Secrets chỉ dùng khi tạo tài khoản lần đầu, không tự đổi mật khẩu của tài khoản đã có trong SQLite. VM 1 GB RAM nên dùng embedding/LLM qua API và tắt reranker local.

## Chạy local bằng Conda

```bash
conda create -n ppl-rag python=3.11 -y
conda run -n ppl-rag python -m pip install -r requirements-cloud.txt -c requirements.txt
cp .env.example .env
conda run -n ppl-rag streamlit run app.py
```

Nếu chạy Docker ngay trên máy local: `docker build -t ppl-rag:local .` rồi `docker compose up -d` sau khi điền `.env`.

Đặt thông số Qdrant Cloud và API trong `.env`. Nếu dùng Qdrant local, đổi `QDRANT_URL` và bỏ trống `QDRANT_API_KEY`. Nếu muốn embedding/reranker local trên máy đủ mạnh, cài `requirements.txt`, đặt `EMBEDDING_PROVIDER=local` và chọn model embedding trong `EMBEDDING_MODEL`.

## Sử dụng

1. Đăng nhập tài khoản admin và mở **Tài liệu**. Nhập môn học, chọn loại tài liệu, tải PDF/DOCX/PPTX. Ứng dụng tự trích xuất, chunk theo phân cấp và lập chỉ mục; nếu thất bại, corpus đang hoạt động không đổi.
2. Mở **Cấu hình truy xuất** nếu cần chọn BM25, dense, RRF, weighted hoặc adaptive. Mặc định dùng dense, không chạy reranker.
3. Mở **Hỏi đáp**. Có thể giới hạn vào vài tài liệu hoặc tìm toàn bộ corpus. Câu trả lời thiếu dẫn nguồn hợp lệ sẽ bị từ chối; hãy mở đoạn trích để kiểm tra nội dung.

DOCX dẫn nguồn theo thứ tự đoạn/khối nội dung, không giả định số trang. Cấu trúc lập chỉ mục gồm tài liệu → mục (nếu nhận diện được) → trang/slide → chunk lá; chỉ chunk lá được embedding. Thay model/provider embedding hoặc cách chunk sẽ tạo phiên bản corpus mới khi upload lại; không trộn vector từ các phiên bản khác nhau.

## Kiểm tra

```bash
conda run -n ppl-rag python -m pytest -q
conda run -n ppl-rag python -m compileall -q app.py pages src
```

Repo hiện tập trung vào hệ thống RAG; chưa có bộ benchmark hay tuyên bố cải thiện độ chính xác truy xuất.
