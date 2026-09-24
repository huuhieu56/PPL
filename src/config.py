import os
from dataclasses import dataclass
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    runs_dir: Path
    db_path: Path
    indexes_dir: Path
    cache_path: Path
    openai_api_key: str
    openai_base_url: str
    openai_model: str


def load_settings(root: Path | None = None) -> Settings:
    load_dotenv()
    project_root = Path(root or Path.cwd()).resolve()
    data_dir = Path(os.getenv("PPL_DATA_DIR") or project_root / "data").resolve()
    runs_dir = Path(os.getenv("PPL_RUNS_DIR") or project_root / "runs").resolve()
    data_dir.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        root=project_root,
        data_dir=data_dir,
        runs_dir=runs_dir,
        db_path=data_dir / "app.db",
        indexes_dir=data_dir / "indexes",
        cache_path=data_dir / "cache" / "retrieval.sqlite",
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        openai_model=os.getenv("OPENAI_MODEL", ""),
    )


def load_yaml(path: Path | str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}
