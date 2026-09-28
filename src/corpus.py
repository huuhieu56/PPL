"""Build and activate one complete, searchable corpus version."""

import json
import hashlib
import shutil
from pathlib import Path

from src.ingestion import CorpusBuildResult, build_corpus
from src.models import Chunk
from src.retrieval import RetrievalIndex


def index_corpus(
    files: list[Path],
    metadata: dict[str, dict],
    settings,
    database,
    chunking: dict,
    retrieval: dict,
) -> CorpusBuildResult:
    active = database.get_active_corpus()
    previous_dir = Path(active["chunks_path"]).parent if active and active.get("chunks_path") else None
    corpus = build_corpus(files, metadata, settings, chunking, previous_dir=previous_dir)
    chunks = [Chunk(**json.loads(line)) for line in corpus.chunks_path.read_text(encoding="utf-8").splitlines() if line]
    nodes = [json.loads(line) for line in corpus.hierarchy_path.read_text(encoding="utf-8").splitlines() if line]
    index_dir = settings.data_dir / "indexes" / corpus.version_id
    if index_dir.exists():
        try:
            RetrievalIndex.load(index_dir)
        except FileNotFoundError as error:
            if "Qdrant collection not found" not in str(error):
                raise
            shutil.rmtree(index_dir)
    if not index_dir.exists():
        RetrievalIndex.build(
            chunks, index_dir, retrieval["tokenizer"], retrieval["embedding_model"],
            hierarchy_nodes=nodes,
            reuse_from=settings.data_dir / "indexes" / active["version_id"] if active else None,
        )
    index = RetrievalIndex.load(index_dir)
    if len(index.chunks) != len(chunks) or len(index.hierarchy or {}) != len(nodes):
        raise ValueError("Chỉ mục không khớp với corpus; phiên bản cũ vẫn được giữ nguyên")
    manifest = json.loads((corpus.chunks_path.parent / "manifest.json").read_text(encoding="utf-8"))
    paths_by_hash = {hashlib.sha256(path.read_bytes()).hexdigest(): str(path) for path in files}
    documents = [{**record, "source_path": paths_by_hash[record["sha256"]]} for record in manifest["documents"]]
    database.publish_corpus(documents, {
        "version_id": corpus.version_id,
        "manifest_hash": corpus.manifest_hash,
        "chunks_path": str(corpus.chunks_path),
        "hierarchy_path": str(corpus.hierarchy_path),
        "chunk_count": len(chunks),
        "node_count": len(nodes),
    })
    return corpus
