import hashlib
import hmac
import os
from pathlib import Path

import streamlit as st

from src.config import load_settings
from src.storage import Database


def database() -> Database:
    settings = load_settings()
    db = Database(settings.db_path)
    db.initialize()
    return db


def _password_hash(password: str, salt: bytes) -> str:
    return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1).hex()


def seed_users(db: Database) -> None:
    for role in ("admin", "student"):
        username = os.getenv(f"{role.upper()}_USERNAME", role)
        password = os.getenv(f"{role.upper()}_PASSWORD", "")
        if not password or db.get_user(username):
            continue
        salt = os.urandom(16)
        db.upsert_user(username, _password_hash(password, salt), salt.hex(), role)


def authenticate(db: Database, username: str, password: str) -> dict | None:
    user = db.get_user(username)
    if not user:
        return None
    actual = _password_hash(password, bytes.fromhex(user["salt"]))
    return user if hmac.compare_digest(actual, user["password_hash"]) else None


def require_role(*roles: str) -> dict:
    user = st.session_state.get("user")
    if not user:
        st.switch_page(st.session_state.home_page)
    if roles and user["role"] not in roles:
        st.error("Bạn không có quyền truy cập trang này.")
        st.stop()
    return user


def safe_upload_name(name: str) -> str:
    candidate = Path(name).name
    cleaned = "".join(character if character.isalnum() or character in " ._-" else "_" for character in candidate)
    return cleaned[:160].strip(" .") or "document"


def stage_uploads(uploads, staging: Path) -> list[Path]:
    paths = []
    seen = set()
    for position, upload in enumerate(uploads):
        content = upload.getvalue()
        digest = hashlib.sha256(content).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        path = staging / f"{position:03}-{digest[:12]}" / safe_upload_name(upload.name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        paths.append(path)
    return paths


def citation_label(citation: dict, names: dict[str, str]) -> str:
    source = names.get(citation["doc_id"], citation["doc_id"])
    locator = "đoạn" if citation.get("file_type") == "docx" or source.lower().endswith(".docx") or citation.get("source_type") == "docx" else "trang/slide"
    return f"[{citation['number']}] {source} — {locator} {citation['page']}–{citation.get('page_end') or citation['page']}"
