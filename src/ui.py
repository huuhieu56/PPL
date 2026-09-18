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
        st.warning("Vui lòng đăng nhập từ trang chính.")
        st.stop()
    if roles and user["role"] not in roles:
        st.error("Bạn không có quyền truy cập trang này.")
        st.stop()
    return user


def safe_upload_name(name: str) -> str:
    candidate = Path(name).name
    cleaned = "".join(character if character.isalnum() or character in "._-" else "_" for character in candidate)
    return cleaned[:160] or "document"
