import hashlib
import json
import random
from pathlib import Path

from src.io_utils import write_csv
from src.models import PipelineConfig

DEFAULT_POOL_SYSTEMS = {
    "C1": PipelineConfig(dense=False, fusion="none", rerank=False),
    "C2": PipelineConfig(sparse=False, fusion="none", rerank=False),
    "C3-RRF": PipelineConfig(fusion="rrf", rerank=False),
    "C3-WS": PipelineConfig(fusion="weighted", rerank=False),
    "C4-WS": PipelineConfig(fusion="weighted", rerank=True),
    "X1": PipelineConfig(sparse=False, fusion="none", rerank=True),
}
ANNOTATION_COLUMNS = ["pool_id", "query_id", "query", "heading_path", "chunk_text", "relevance", "evidence_quote"]


def build_pool(queries, pipeline, systems: dict, depth: int = 15, manual_additions: list[dict] | None = None) -> list[dict]:
    chunks = pipeline.index.chunks
    found: dict[tuple[str, str], set[str]] = {}
    for query in queries:
        for name, config in systems.items():
            for item in pipeline.run(query["text"], config).results[:depth]:
                found.setdefault((query["query_id"], item.chunk.chunk_id), set()).add(name)
        for chunk_id in query.get("source_chunk_ids", []):
            found.setdefault((query["query_id"], chunk_id), set()).add("source")
    for addition in manual_additions or []:
        if addition["chunk_id"] not in chunks:
            raise ValueError(f"Manual addition refers to unknown chunk {addition['chunk_id']}")
        found.setdefault((addition["query_id"], addition["chunk_id"]), set()).add("manual")
    return [
        {
            "pool_id": hashlib.sha256(f"{query_id}|{chunk_id}".encode("utf-8")).hexdigest()[:12],
            "query_id": query_id,
            "chunk_id": chunk_id,
            "doc_id": chunks[chunk_id].doc_id,
            "page": chunks[chunk_id].page,
            "systems": sorted(names),
        }
        for (query_id, chunk_id), names in sorted(found.items())
    ]


def write_pool(pool, queries, index, out_dir, annotators=("A", "B"), seed: int = 42) -> list[Path]:
    target = Path(out_dir)
    paths = [target / f"annotation_{name}.csv" for name in annotators]
    existing = [path for path in paths if path.exists()]
    if existing:
        raise ValueError(f"Annotation file already exists: {existing[0]}; move it away before re-pooling")
    target.mkdir(parents=True, exist_ok=True)
    (target / "pool_map.json").write_text(json.dumps(pool, ensure_ascii=False, indent=2), encoding="utf-8")
    texts = {query["query_id"]: query["text"] for query in queries}
    rng = random.Random(seed)
    ordered = []
    for query in queries:
        entries = [entry for entry in pool if entry["query_id"] == query["query_id"]]
        rng.shuffle(entries)
        ordered.extend(entries)
    rows = [
        {
            "pool_id": entry["pool_id"],
            "query_id": entry["query_id"],
            "query": texts[entry["query_id"]],
            "heading_path": index.chunks[entry["chunk_id"]].breadcrumb(),
            "chunk_text": index.chunks[entry["chunk_id"]].body,
            "relevance": "",
            "evidence_quote": "",
        }
        for entry in ordered
    ]
    for path in paths:
        write_csv(path, rows, ANNOTATION_COLUMNS)
    return paths
