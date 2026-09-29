"""
FastAPI REST API — Trợ lý học tập RAG
Roles: teacher (upload/manage docs + handle requests) | student (chat only)

Run:  uvicorn backend.main:app --reload --port 8000
"""
from __future__ import annotations

import hashlib
import hmac


def safe_upload_name(name: str) -> str:
    """Sanitize uploaded filename (copied from src/ui.py to avoid streamlit import)."""
    from pathlib import Path as _Path
    candidate = _Path(name).name
    cleaned = "".join(c if c.isalnum() or c in " ._-" else "_" for c in candidate)
    return cleaned[:160].strip(" .") or "document"
import json
import logging
import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jose import JWTError, jwt
from pydantic import BaseModel

# ── Project imports ───────────────────────────────────────────────────────────
from src.config import load_settings
from src.storage import Database

# ── App & config ──────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Learning API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SECRET_KEY = os.getenv("JWT_SECRET", "rag-learning-secret-key-change-in-prod")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

bearer_scheme = HTTPBearer()
logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_db() -> Database:
    settings = load_settings()
    db = Database(settings.db_path)
    db.initialize()
    return db


def _password_hash(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1).hex()


def create_access_token(data: dict) -> str:
    expire = datetime.now(timezone.utc) + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    return jwt.encode({**data, "exp": expire}, SECRET_KEY, algorithm=ALGORITHM)


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    db: Database = Depends(get_db),
) -> dict:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub", "")
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Token không hợp lệ")
    user = db.get_user(username)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Người dùng không tồn tại")
    return user


def require_teacher(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "teacher":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Chỉ giáo viên được phép")
    return user


def require_student(user: dict = Depends(get_current_user)) -> dict:
    if user["role"] != "student":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Chỉ học sinh được phép")
    return user


# ── Seed initial users on startup ─────────────────────────────────────────────

@app.on_event("startup")
def seed_users() -> None:
    db = get_db()
    for role_key, role_value in [("TEACHER", "teacher"), ("STUDENT", "student")]:
        username = os.getenv(f"{role_key}_USERNAME", role_key.lower())
        password = os.getenv(f"{role_key}_PASSWORD", "")
        if not password or db.get_user(username):
            continue
        salt = os.urandom(16)
        db.upsert_user(username, _password_hash(password, salt), salt.hex(), role_value)
    logger.info("Users seeded.")


# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════

class LoginRequest(BaseModel):
    username: str
    password: str


@app.post("/api/auth/login")
def login(body: LoginRequest, db: Database = Depends(get_db)):
    user = db.get_user(body.username)
    if not user:
        raise HTTPException(status_code=401, detail="Tên đăng nhập hoặc mật khẩu không đúng")
    actual = _password_hash(body.password, bytes.fromhex(user["salt"]))
    if not hmac.compare_digest(actual, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Tên đăng nhập hoặc mật khẩu không đúng")
    token = create_access_token({"sub": user["username"], "role": user["role"]})
    return {"access_token": token, "token_type": "bearer", "role": user["role"], "username": user["username"]}


@app.get("/api/auth/me")
def me(user: dict = Depends(get_current_user)):
    return {"username": user["username"], "role": user["role"]}


# ══════════════════════════════════════════════════════════════════════════════
# CORPUS / DOCUMENTS — TEACHER
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/corpus/active")
def get_active_corpus(
    user: dict = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    active = db.get_active_corpus()
    if not active:
        return {"active": False}
    result = {"active": True, **active}
    # Try to read manifest for document list
    if active.get("chunks_path"):
        manifest_path = Path(active["chunks_path"]).parent / "manifest.json"
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                result["documents"] = manifest.get("documents", [])
            except Exception:
                result["documents"] = []
    return result


@app.get("/api/documents")
def list_documents(
    user: dict = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    """List all documents in the active corpus."""
    active = db.get_active_corpus()
    active_ids: set[str] = set()
    if active and active.get("chunks_path"):
        manifest_path = Path(active["chunks_path"]).parent / "manifest.json"
        if manifest_path.is_file():
            try:
                docs = json.loads(manifest_path.read_text(encoding="utf-8"))["documents"]
                active_ids = {d["doc_id"] for d in docs}
            except Exception:
                pass
    all_docs = db.list_documents()
    return [d for d in all_docs if d["doc_id"] in active_ids]


@app.post("/api/documents/upload")
async def upload_documents(
    course: str = Form(...),
    source_type: str = Form("textbook"),
    semester: str = Form(""),
    files: list[UploadFile] = File(...),
    teacher: dict = Depends(require_teacher),
    db: Database = Depends(get_db),
):
    """Upload documents and re-index corpus. Teacher only."""
    from src.config import load_settings, load_yaml
    from src.corpus import index_corpus
    from src.ui import safe_upload_name

    settings = load_settings()
    defaults = load_yaml("configs/default.yaml")

    # Check total size
    contents = []
    total_size = 0
    for f in files:
        content = await f.read()
        total_size += len(content)
        contents.append((f.filename, content))
    if total_size > 200 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="Tổng dung lượng vượt 200 MB")

    # Stage files
    staging = settings.data_dir / "raw" / "staging"
    staging.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    seen: set[str] = set()
    for idx, (filename, content) in enumerate(contents):
        digest = hashlib.sha256(content).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        safe_name = safe_upload_name(filename)
        path = staging / f"{idx:03}-{digest[:12]}" / safe_name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        paths.append(path)

    # Keep existing active docs
    active = db.get_active_corpus()
    active_ids: set[str] = set()
    if active and active.get("chunks_path"):
        manifest_path = Path(active["chunks_path"]).parent / "manifest.json"
        if manifest_path.is_file():
            try:
                docs = json.loads(manifest_path.read_text(encoding="utf-8"))["documents"]
                active_ids = {d["doc_id"] for d in docs}
            except Exception:
                pass

    stored = {
        item["doc_id"]: item
        for item in db.list_documents()
        if item["doc_id"] in active_ids
        and item.get("source_path")
        and Path(item["source_path"]).is_file()
    }

    uploaded = {hashlib.sha256(p.read_bytes()).hexdigest(): p for p in paths}
    files_by_hash = {stored[doc_id]["sha256"]: Path(stored[doc_id]["source_path"]) for doc_id in stored}
    metadata = {
        str(stored[doc_id]["source_path"]): {
            k: stored[doc_id][k] for k in ("course", "source_type", "semester") if k in stored[doc_id]
        }
        for doc_id in stored
    }
    uploaded_metadata = {
        str(p): {"course": course.strip(), "source_type": source_type, "semester": semester}
        for p in paths
    }
    files_by_hash.update(uploaded)
    metadata.update(uploaded_metadata)

    try:
        result = index_corpus(
            list(files_by_hash.values()),
            metadata,
            settings,
            db,
            {"chunk_tokens": 450, "overlap_tokens": 75},
            defaults["retrieval"],
        )
    except Exception as e:
        logger.exception("Index corpus failed")
        raise HTTPException(status_code=500, detail=f"Lập chỉ mục thất bại: {e}")

    return {
        "version_id": result.version_id,
        "chunk_count": result.chunk_count,
        "message": f"Đã tạo corpus {result.version_id} với {result.chunk_count} chunks",
    }


@app.delete("/api/documents/{doc_id}")
def delete_document(
    doc_id: str,
    teacher: dict = Depends(require_teacher),
    db: Database = Depends(get_db),
):
    """Delete a document from DB and disk. Teacher only."""
    all_docs = db.list_documents()
    doc = next((d for d in all_docs if d["doc_id"] == doc_id), None)
    if not doc:
        raise HTTPException(status_code=404, detail="Không tìm thấy tài liệu")

    deleted = db.delete_document(doc_id)
    if deleted and doc.get("source_path"):
        p = Path(doc["source_path"])
        p.unlink(missing_ok=True)
        if p.parent.exists() and not any(p.parent.iterdir()):
            try:
                p.parent.rmdir()
            except OSError:
                pass
    return {"deleted": deleted, "doc_id": doc_id}


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT REQUESTS — STUDENT → TEACHER
# ══════════════════════════════════════════════════════════════════════════════

class DocRequestCreate(BaseModel):
    subject: str
    description: str = ""


class DocRequestUpdate(BaseModel):
    status: str          # approved | rejected
    teacher_note: str = ""


@app.get("/api/requests")
def list_requests(
    filter_status: str | None = None,
    user: dict = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    """Teachers see all; students see only their own."""
    all_reqs = db.list_doc_requests(status=filter_status if filter_status else None)
    if user["role"] == "student":
        all_reqs = [r for r in all_reqs if r["username"] == user["username"]]
    return all_reqs


@app.post("/api/requests", status_code=201)
def create_request(
    body: DocRequestCreate,
    student: dict = Depends(require_student),
    db: Database = Depends(get_db),
):
    if not body.subject.strip():
        raise HTTPException(status_code=400, detail="Tên tài liệu không được để trống")
    request_id = db.save_doc_request({
        "username": student["username"],
        "subject": body.subject.strip(),
        "description": body.description.strip(),
    })
    return {"request_id": request_id, "message": "Đã gửi yêu cầu"}


@app.patch("/api/requests/{request_id}")
def update_request(
    request_id: str,
    body: DocRequestUpdate,
    teacher: dict = Depends(require_teacher),
    db: Database = Depends(get_db),
):
    if body.status not in ("approved", "rejected"):
        raise HTTPException(status_code=400, detail="Status phải là approved hoặc rejected")
    db.update_doc_request_status(request_id, body.status, body.teacher_note)
    return {"request_id": request_id, "status": body.status}


# ══════════════════════════════════════════════════════════════════════════════
# CHAT — STUDENT
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/chat/sessions")
def list_sessions(
    corpus_version: str = "",
    student: dict = Depends(require_student),
    db: Database = Depends(get_db),
):
    sessions = db.list_chat_sessions(student["username"], corpus_version)
    return sessions


@app.post("/api/chat/sessions", status_code=201)
def create_session(
    corpus_version: str = "",
    student: dict = Depends(require_student),
    db: Database = Depends(get_db),
):
    session_id = uuid.uuid4().hex
    db.start_chat_session(session_id, student["username"], corpus_version)
    return {"session_id": session_id}


@app.get("/api/chat/sessions/{session_id}/messages")
def get_messages(
    session_id: str,
    student: dict = Depends(require_student),
    db: Database = Depends(get_db),
):
    turns = db.chat_turns(session_id, student["username"])
    return turns


class ChatRequest(BaseModel):
    session_id: str
    query: str
    doc_ids: list[str] = []


@app.post("/api/chat/ask")
def ask(
    body: ChatRequest,
    student: dict = Depends(require_student),
    db: Database = Depends(get_db),
):
    """Send a question and get RAG answer."""
    from src.config import load_settings
    from src.models import RagConfig
    from src.rag import answer_question, chat_model
    from src.retrieval import RetrievalIndex

    settings = load_settings()
    if not settings.openai_api_key or not settings.openai_model:
        raise HTTPException(status_code=503, detail="Thiếu OPENAI_API_KEY hoặc OPENAI_MODEL trong .env")

    active = db.get_active_corpus()
    if not active:
        raise HTTPException(status_code=404, detail="Chưa có corpus đang hoạt động")

    try:
        index = RetrievalIndex.load(str(settings.data_dir / "indexes" / active["version_id"]))
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Không tải được chỉ mục: {e}")

    if body.doc_ids:
        index = index.scoped(tuple(sorted(body.doc_ids)))

    saved_configs = db.list_rag_configs()
    config = (
        RagConfig(**saved_configs[0]["config"]) if saved_configs
        else RagConfig(model=settings.openai_model)
    )
    client = chat_model(settings, config.model or settings.openai_model)

    turns = db.chat_turns(body.session_id, student["username"])
    try:
        answer = answer_question(
            body.query, index, config, client,
            config.model or settings.openai_model,
            history=turns,
        )
    except Exception as e:
        logger.exception("Chat answer failed")
        raise HTTPException(status_code=500, detail=f"Lỗi trả lời: {e}")

    message_id = db.save_message({
        "session_id": body.session_id,
        "username": student["username"],
        "query": body.query,
        "answer": answer.text,
        "citations": answer.citations,
        "refused": answer.refused,
        "latency_ms": answer.retrieval_ms + answer.generation_ms,
    })

    return {
        "message_id": message_id,
        "answer": answer.text,
        "citations": answer.citations,
        "refused": answer.refused,
        "latency_ms": answer.retrieval_ms + answer.generation_ms,
    }


class FeedbackRequest(BaseModel):
    message_id: str
    useful: bool
    comment: str = ""


@app.post("/api/chat/feedback", status_code=201)
def save_feedback(
    body: FeedbackRequest,
    student: dict = Depends(require_student),
    db: Database = Depends(get_db),
):
    feedback_id = db.save_feedback({
        "message_id": body.message_id,
        "useful": body.useful,
        "comment": body.comment,
    })
    return {"feedback_id": feedback_id}


# ══════════════════════════════════════════════════════════════════════════════
# RAG CONFIG — TEACHER
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/api/rag-configs")
def list_rag_configs(
    user: dict = Depends(get_current_user),
    db: Database = Depends(get_db),
):
    return db.list_rag_configs()
