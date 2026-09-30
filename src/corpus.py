"""Build and activate one complete, searchable corpus version."""

import json
import hashlib
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

from src.chunking import build_hierarchy
from src.documents import extract_document
from src.models import Chunk
from src.embeddings import embedding_config
from src.index import RetrievalIndex


@dataclass(frozen=True)
class CorpusBuildResult:
    version_id: str
    chunks_path: Path
    hierarchy_path: Path
    chunk_count: int
    manifest_hash: str


def build_corpus(
    files: list[Path],
    metadata: dict[str, dict],
    settings,
    config: dict,
    previous_dir: Path | None = None,
    index_identity: dict | None = None,
) -> CorpusBuildResult:
    cached_records: dict[str, dict] = {}
    cached_chunks: dict[str, list[Chunk]] = {}
    cached_nodes: dict[str, list[dict]] = {}
    if previous_dir and (previous_dir / "manifest.json").is_file():
        previous = json.loads((previous_dir / "manifest.json").read_text(encoding="utf-8"))
        if previous.get("config") == config and previous.get("chunking") == "hierarchical-v7":
            cached_records = {item["doc_id"]: item for item in previous["documents"]}
            for line in (previous_dir / "chunks.jsonl").read_text(encoding="utf-8").splitlines():
                chunk = Chunk(**json.loads(line))
                cached_chunks.setdefault(chunk.doc_id, []).append(chunk)
            for line in (previous_dir / "hierarchy.jsonl").read_text(encoding="utf-8").splitlines():
                node = json.loads(line)
                cached_nodes.setdefault(node["doc_id"], []).append(node)
    manifest_items = []
    all_chunks: list[Chunk] = []
    all_nodes: list[dict] = []
    for path in sorted((Path(file) for file in files), key=lambda item: str(item)):
        file_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        item_metadata = metadata[str(path)]
        doc_id = file_hash[:24]
        record = {
            "doc_id": doc_id,
            "filename": path.name,
            "sha256": file_hash,
            **item_metadata,
        }
        if cached_records.get(doc_id) == record and cached_chunks.get(doc_id) and cached_nodes.get(doc_id):
            chunks, nodes = cached_chunks[doc_id], cached_nodes[doc_id]
        else:
            pages = extract_document(path)
            chunks, nodes = build_hierarchy(
                pages,
                doc_id=doc_id,
                course=item_metadata["course"],
                source_type=item_metadata["source_type"],
                chunk_tokens=int(config["chunk_tokens"]),
                overlap_tokens=int(config["overlap_tokens"]),
                title=path.name,
                file_type=path.suffix.lower().lstrip("."),
            )
            if not chunks:
                raise ValueError(f"{path.name}: không có nội dung văn bản để lập chỉ mục")
        all_chunks.extend(chunks)
        all_nodes.extend(nodes)
        manifest_items.append(record)

    manifest = {"documents": manifest_items, "config": config, "chunking": "hierarchical-v7"}
    if index_identity is not None:
        manifest["index_identity"] = index_identity
    manifest_json = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    manifest_hash = hashlib.sha256(manifest_json.encode()).hexdigest()
    version_id = manifest_hash[:16]
    output_dir = settings.data_dir / "processed" / version_id
    chunks_path = output_dir / "chunks.jsonl"
    hierarchy_path = output_dir / "hierarchy.jsonl"
    chunks_jsonl = "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in all_chunks) + "\n"
    hierarchy_jsonl = "\n".join(json.dumps(node, ensure_ascii=False) for node in all_nodes) + "\n"
    if output_dir.exists():
        if (output_dir / "manifest.json").is_file() and chunks_path.is_file() and hierarchy_path.is_file() and (output_dir / "manifest.json").read_text(encoding="utf-8") == manifest_json and chunks_path.read_text(encoding="utf-8") == chunks_jsonl and hierarchy_path.read_text(encoding="utf-8") == hierarchy_jsonl:
            return CorpusBuildResult(version_id, chunks_path, hierarchy_path, len(all_chunks), manifest_hash)
        raise FileExistsError(f"Corpus version exists with different or incomplete contents: {version_id}")
    output_dir.mkdir(parents=True)
    chunks_path.write_text(chunks_jsonl, encoding="utf-8")
    hierarchy_path.write_text(hierarchy_jsonl, encoding="utf-8")
    (output_dir / "manifest.json").write_text(manifest_json, encoding="utf-8")
    return CorpusBuildResult(version_id, chunks_path, hierarchy_path, len(all_chunks), manifest_hash)


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
    embedding_model = os.getenv("EMBEDDING_MODEL") or retrieval["embedding_model"]
    identity = {"embedding": embedding_config(embedding_model), "tokenizer": retrieval["tokenizer"]}
    corpus = build_corpus(files, metadata, settings, chunking, previous_dir=previous_dir, index_identity=identity)
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
            chunks, index_dir, retrieval["tokenizer"], embedding_model,
            hierarchy_nodes=nodes,
        )
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
