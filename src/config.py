import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    root: Path
    data_dir: Path
    db_path: Path
    llm_api_key: str = field(repr=False)
    llm_base_url: str
    llm_model: str


def load_settings(root: Path | None = None) -> Settings:
    load_dotenv()
    project_root = Path(root or Path.cwd()).resolve()
    data_dir = project_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return Settings(
        root=project_root,
        data_dir=data_dir,
        db_path=data_dir / "app.db",
        llm_api_key=os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", ""),
        llm_base_url=os.getenv("LLM_BASE_URL") or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        llm_model=os.getenv("LLM_MODEL") or os.getenv("OPENAI_MODEL", ""),
    )


def load_yaml(path: Path | str) -> dict:
    with Path(path).open(encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}
