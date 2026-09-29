# 🚀 Hướng dẫn chạy hệ thống FE + BE

## Cấu trúc mới (không xóa file cũ)

```
PPL/
├── backend/
│   └── main.py          ← FastAPI REST API server
├── frontend/
│   ├── src/
│   │   ├── pages/
│   │   │   ├── teacher/  ← Dashboard, Documents (upload+xóa), Requests
│   │   │   └── student/  ← Dashboard, Chat (hỏi đáp), RequestDoc
│   │   └── ...
│   └── package.json
├── pages/               ← GIỮ NGUYÊN (Streamlit cũ)
├── src/                 ← GIỮ NGUYÊN (core logic)
├── app.py               ← GIỮ NGUYÊN (Streamlit cũ)
└── requirements_api.txt ← Dependency mới cho FastAPI
```

---

## ⚡ Bước 1 — Cài đặt dependencies

### Python (FastAPI backend)
```bash
pip install -r requirements_api.txt
pip install -r requirements.txt
```

### Node.js (React frontend)
```bash
cd frontend
npm install
```

---

## ⚡ Bước 2 — Cấu hình `.env`

Tạo file `.env` (copy từ `.env.example`):
```
OPENAI_API_KEY=sk-...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o-mini

TEACHER_USERNAME=teacher
TEACHER_PASSWORD=teacher123

STUDENT_USERNAME=student
STUDENT_PASSWORD=student123

JWT_SECRET=your-secret-key-here
```

---

## ⚡ Bước 3 — Chạy Backend (FastAPI)

```bash
# Từ thư mục gốc PPL/
uvicorn backend.main:app --reload --port 8000
```

API sẽ chạy tại: http://localhost:8000  
Tài liệu API (Swagger): http://localhost:8000/docs

---

## ⚡ Bước 4 — Chạy Frontend (React)

```bash
# Từ thư mục PPL/frontend/
npm run dev
```

Mở trình duyệt: **http://localhost:5173**

---

## 👩‍🏫 Giao diện Teacher

| Trang | Chức năng |
|---|---|
| Tổng quan | Thống kê corpus, yêu cầu đang chờ |
| **Tài liệu** | Upload PDF/DOCX/PPTX, xem danh sách, **xóa từng file** |
| **Yêu cầu** | Duyệt / từ chối yêu cầu tài liệu từ học sinh |

## 🎓 Giao diện Student

| Trang | Chức năng |
|---|---|
| Tổng quan | Xem tài liệu có sẵn |
| **Hỏi đáp** | Chat RAG, xem dẫn nguồn, phản hồi câu trả lời |
| **Yêu cầu tài liệu** | Gửi yêu cầu, theo dõi trạng thái |

---

## 🔌 API Endpoints chính

```
POST   /api/auth/login              Đăng nhập → JWT token
GET    /api/auth/me                 Thông tin user hiện tại

GET    /api/corpus/active           Corpus đang hoạt động
GET    /api/documents               Danh sách tài liệu
POST   /api/documents/upload        Upload & index [teacher]
DELETE /api/documents/{id}          Xóa tài liệu [teacher]

GET    /api/requests                Danh sách yêu cầu
POST   /api/requests                Tạo yêu cầu mới [student]
PATCH  /api/requests/{id}           Duyệt/từ chối [teacher]

POST   /api/chat/sessions           Tạo phiên chat [student]
GET    /api/chat/sessions           Danh sách phiên
POST   /api/chat/ask                Gửi câu hỏi RAG [student]
POST   /api/chat/feedback           Phản hồi câu trả lời
```

---

## 💡 Lưu ý

- Streamlit cũ (`app.py`, `pages/`) **vẫn hoạt động bình thường**
- React frontend proxy `/api` → FastAPI port 8000
- Sau khi xóa tài liệu, nên upload lại để lập chỉ mục corpus mới
