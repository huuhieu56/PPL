import json
from pathlib import Path

from src.index import RetrievalIndex
from src.ingestion import build_corpus
from src.io_utils import read_csv
from src.models import Chunk

SUPPORTED_SUFFIXES = {".pdf", ".docx", ".pptx"}


def _default_source_type(path: Path) -> str:
    return "slide" if path.suffix.lower() == ".pptx" else "textbook"


def build_index_from_folder(
    input_dir,
    *,
    course: str,
    settings,
    database,
    chunking: dict,
    index_options: dict,
    metadata_csv=None,
    activate: bool = True,
    encoder=None,
) -> tuple[str, Path]:
    files = sorted(path for path in Path(input_dir).iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES)
    if not files:
        raise ValueError(f"No PDF/DOCX/PPTX files in {input_dir}")
    overrides = {row["filename"]: row for row in read_csv(metadata_csv)} if metadata_csv else {}
    metadata = {}
    for path in files:
        row = overrides.get(path.name, {})
        metadata[str(path)] = {
            "course": row.get("course") or course,
            "source_type": row.get("source_type") or _default_source_type(path),
            **({"doc_title": row["doc_title"]} if row.get("doc_title") else {}),
        }
    result = build_corpus(files, metadata, settings, database, chunking)
    index_dir = settings.indexes_dir / result.version_id
    if not (index_dir / "index_meta.json").exists():
        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in result.chunks_path.read_text(encoding="utf-8").splitlines()
            if line
        ]
        RetrievalIndex.build(
            chunks,
            index_dir,
            embedding_model=index_options["embedding_model"],
            tokenizers=index_options["tokenizers"],
            dense_backend=index_options["dense_backend"],
            bm25_k1=index_options["bm25_k1"],
            bm25_b=index_options["bm25_b"],
            chunking=chunking,
            encoder=encoder,
        )
    if activate:
        database.set_active_corpus(result.version_id)
    return result.version_id, index_dir


def resolve_index_dir(value: str | None, settings, database) -> Path:
    if value:
        return Path(value)
    active = database.get_active_corpus()
    if not active:
        raise ValueError("No active corpus; pass --index or build one with `python -m src.cli index build`")
    return settings.indexes_dir / active["version_id"]
