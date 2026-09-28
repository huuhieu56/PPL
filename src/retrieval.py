import json
import math
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25L

from src.ingestion import tokenize_vi
from src.models import Chunk, RagConfig, SearchResult

DEFAULT_BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


@lru_cache(maxsize=2)
def load_encoder(model_name: str, revision: str | None):
    from sentence_transformers import SentenceTransformer

    try:
        encoder = SentenceTransformer(model_name, revision=revision, local_files_only=True)
    except OSError:
        encoder = SentenceTransformer(model_name, revision=revision, local_files_only=False)
    if getattr(getattr(encoder, "device", None), "type", None) == "cuda":
        encoder.half()
    return encoder


def minmax_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if math.isclose(low, high):
        return {key: float(high > 0) for key in scores}
    scale = high - low
    return {key: (value - low) / scale for key, value in scores.items()}


def adaptive_alpha(
    query: str,
    alpha0: float,
    beta: float,
    idf: dict[str, float],
) -> tuple[float, dict[str, float]]:
    compact = query.strip()
    length = max(len(compact), 1)
    tokens = re.findall(r"\w+", compact.lower(), flags=re.UNICODE)
    idf_scale = max(1.0, max(idf.values(), default=0.0))
    signals = {
        "digit": sum(character.isdigit() for character in compact) / length,
        "symbol": sum(not character.isalnum() and not character.isspace() for character in compact)
        / length,
        "code": float(bool(re.search(r"\b[A-Z]{2,}[A-Z0-9-]*\d+[A-Z0-9-]*\b", compact))),
        "max_idf": max((idf.get(token, 0.0) for token in tokens), default=0.0) / idf_scale,
    }
    lexical_score = min(
        1.0,
        signals["digit"] + signals["symbol"] + signals["code"] + signals["max_idf"],
    )
    signals["lexical_score"] = lexical_score
    return min(0.9, max(0.1, alpha0 + beta * lexical_score)), signals


def fuse_weighted(
    bm25_scores: dict[str, float],
    dense_scores: dict[str, float],
    chunks: dict[str, Chunk],
    alpha: float,
) -> list[SearchResult]:
    sparse = minmax_scores(bm25_scores)
    dense = minmax_scores(dense_scores)
    identifiers = set(sparse) | set(dense)
    ranked = sorted(
        identifiers,
        key=lambda chunk_id: (
            -(alpha * sparse.get(chunk_id, 0.0) + (1 - alpha) * dense.get(chunk_id, 0.0)),
            chunk_id,
        ),
    )
    return [
        SearchResult(
            chunks[chunk_id],
            alpha * sparse.get(chunk_id, 0.0) + (1 - alpha) * dense.get(chunk_id, 0.0),
            rank,
            "weighted",
        )
        for rank, chunk_id in enumerate(ranked, start=1)
    ]


def fuse_rrf(
    result_lists: list[list[SearchResult]], rrf_k: int = 60
) -> list[SearchResult]:
    scores: dict[str, float] = {}
    chunks: dict[str, Chunk] = {}
    for results in result_lists:
        for result in results:
            chunk_id = result.chunk.chunk_id
            chunks[chunk_id] = result.chunk
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1 / (rrf_k + result.rank)
    ranked = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    return [
        SearchResult(chunks[chunk_id], scores[chunk_id], rank, "rrf")
        for rank, chunk_id in enumerate(ranked, start=1)
    ]


def validate_hierarchy(chunks: list[Chunk], nodes: list[dict]) -> dict[str, dict]:
    hierarchy = {node["node_id"]: node for node in nodes}
    leaves = {chunk.chunk_id: chunk for chunk in chunks}
    if len(hierarchy) != len(nodes) or len(leaves) != len(chunks) or set(hierarchy) & set(leaves):
        raise ValueError("Duplicate hierarchy node or chunk ID")
    roots = [node for node in nodes if node["parent_id"] is None]
    if {node["doc_id"] for node in roots} != {chunk.doc_id for chunk in chunks} or len(roots) != len({node["doc_id"] for node in roots}) or any(node["kind"] != "document" for node in roots):
        raise ValueError("Hierarchy must have one document root per source")
    owners: dict[str, str] = {}
    for node in nodes:
        parent_id = node["parent_id"]
        if parent_id is not None and (parent_id not in hierarchy or hierarchy[parent_id]["doc_id"] != node["doc_id"]):
            raise ValueError("Hierarchy node has an invalid parent")
        for child_id in node["children"]:
            child = hierarchy.get(child_id) or leaves.get(child_id)
            if child is None or child_id in owners:
                raise ValueError("Hierarchy has an unknown or duplicate child")
            child_parent = child["parent_id"] if isinstance(child, dict) else child.parent_id
            child_doc = child["doc_id"] if isinstance(child, dict) else child.doc_id
            if child_parent != node["node_id"] or child_doc != node["doc_id"]:
                raise ValueError("Hierarchy child points to the wrong parent")
            owners[child_id] = node["node_id"]
    if set(owners) != (set(hierarchy) - {node["node_id"] for node in roots}) | set(leaves):
        raise ValueError("Hierarchy has an unlinked child")
    for chunk in chunks:
        parent = hierarchy[chunk.parent_id]
        words = parent["text"].split()
        if not 0 <= chunk.word_start < chunk.word_end <= len(words) or not chunk.text.endswith(" ".join(words[chunk.word_start:chunk.word_end])):
            raise ValueError("Hierarchy leaf span does not match parent text")
    visited: set[str] = set()

    def walk(node_id: str) -> None:
        if node_id in visited:
            raise ValueError("Hierarchy contains a cycle")
        visited.add(node_id)
        for child_id in hierarchy[node_id]["children"]:
            if child_id in hierarchy:
                walk(child_id)

    for root in roots:
        walk(root["node_id"])
    if len(visited) != len(hierarchy):
        raise ValueError("Hierarchy contains an unreachable node")
    return hierarchy


@dataclass
class RetrievalIndex:
    chunks: dict[str, Chunk]
    tokenizer_mode: str
    bm25: BM25L
    qdrant: QdrantClient
    collection: str
    embedding_model: str
    embedding_revision: str | None = None
    _encoder: object | None = None
    hierarchy: dict[str, dict] | None = None
    doc_ids: tuple[str, ...] = ()

    def scoped(self, doc_ids: tuple[str, ...]) -> "RetrievalIndex":
        if not doc_ids:
            return self
        selected = set(doc_ids)
        if not selected <= {chunk.doc_id for chunk in self.chunks.values()}:
            raise ValueError("Selected documents are not in this index")
        chunks = {key: chunk for key, chunk in self.chunks.items() if chunk.doc_id in selected}
        return replace(self, chunks=chunks, doc_ids=tuple(sorted(selected)),
                       bm25=BM25L([tokenize_vi(chunk.text, self.tokenizer_mode) for chunk in chunks.values()]))

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        output_dir: Path | str,
        tokenizer_mode: str,
        embedding_model: str,
        qdrant_client: QdrantClient | None = None,
        hierarchy_nodes: list[dict] | None = None,
        reuse_from: Path | None = None,
    ) -> "RetrievalIndex":
        if hierarchy_nodes is None and any(chunk.parent_id for chunk in chunks):
            raise ValueError("Parented chunks require hierarchy nodes")
        if not chunks:
            raise ValueError("Cannot build an empty retrieval index")
        target = Path(output_dir)
        if target.exists():
            raise FileExistsError(f"Index directory already exists: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        hierarchy = validate_hierarchy(chunks, hierarchy_nodes) if hierarchy_nodes is not None else None
        tokens = [tokenize_vi(chunk.text, tokenizer_mode) for chunk in chunks]
        revision = DEFAULT_BGE_M3_REVISION if embedding_model == "BAAI/bge-m3" else None
        client = qdrant_client or QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
        collection = "ppl_" + re.sub(r"[^A-Za-z0-9_]", "_", target.name)
        if client.collection_exists(collection):
            raise FileExistsError(f"Qdrant collection already exists: {collection}")
        vectors_by_chunk: dict[str, object] = {}
        if reuse_from and (reuse_from / "index_meta.json").is_file():
            previous = json.loads((reuse_from / "index_meta.json").read_text(encoding="utf-8"))
            old_collection = previous.get("collection")
            if previous.get("embedding_model") == embedding_model and previous.get("embedding_revision") == revision and old_collection and client.collection_exists(old_collection):
                for start in range(0, len(chunks), 128):
                    batch = chunks[start : start + 128]
                    identifiers = {str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)): chunk.chunk_id for chunk in batch}
                    for point in client.retrieve(old_collection, ids=list(identifiers), with_payload=False, with_vectors=True):
                        if point.vector is not None:
                            vectors_by_chunk[identifiers[str(point.id)]] = point.vector
        missing = [chunk for chunk in chunks if chunk.chunk_id not in vectors_by_chunk]
        encoder = None
        if missing:
            encoder = load_encoder(embedding_model, revision)
            encoded = encoder.encode(
                [chunk.text for chunk in missing],
                batch_size=1 if getattr(getattr(encoder, "device", None), "type", None) == "cuda" else 8,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            vectors_by_chunk.update(zip((chunk.chunk_id for chunk in missing), encoded))
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        created = False
        try:
            client.create_collection(collection, vectors_config=models.VectorParams(size=len(next(iter(vectors_by_chunk.values()))), distance=models.Distance.COSINE))
            created = True
            for start in range(0, len(chunks), 32):
                points = []
                for chunk in chunks[start : start + 32]:
                    vector = vectors_by_chunk[chunk.chunk_id]
                    points.append(models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                        vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
                        payload=chunk.to_dict(),
                    ))
                client.upsert(
                    collection,
                    points=points,
                    wait=True,
                )
            (staging / "chunks.jsonl").write_text(
                "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks) + "\n",
                encoding="utf-8",
            )
            (staging / "tokens.json").write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")
            if hierarchy_nodes is not None:
                (staging / "hierarchy.jsonl").write_text(
                    "\n".join(json.dumps(node, ensure_ascii=False) for node in hierarchy_nodes) + "\n",
                    encoding="utf-8",
                )
            (staging / "index_meta.json").write_text(
                json.dumps({
                    "tokenizer_mode": tokenizer_mode,
                    "embedding_model": embedding_model,
                    "embedding_revision": revision,
                    "chunk_count": len(chunks),
                    "backend": "qdrant",
                    "collection": collection,
                    "hierarchy_node_count": len(hierarchy_nodes) if hierarchy_nodes is not None else None,
                }, ensure_ascii=False), encoding="utf-8",
            )
            if client.count(collection, exact=True).count != len(chunks):
                raise ValueError("Qdrant point count does not match corpus")
            staging.rename(target)
        except Exception:
            if created:
                client.delete_collection(collection)
            shutil.rmtree(staging)
            raise
        return cls(
            {chunk.chunk_id: chunk for chunk in chunks},
            tokenizer_mode,
            BM25L(tokens),
            client,
            collection,
            embedding_model,
            revision,
            encoder,
            hierarchy,
        )

    @classmethod
    def load(cls, index_dir: Path | str, qdrant_client: QdrantClient | None = None) -> "RetrievalIndex":
        source = Path(index_dir)
        meta = json.loads((source / "index_meta.json").read_text(encoding="utf-8"))
        chunks = [
            Chunk(**json.loads(line))
            for line in (source / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(chunks) != meta["chunk_count"]:
            raise ValueError("Index metadata does not match chunk count")
        if meta.get("backend") != "qdrant":
            raise ValueError("Legacy NumPy index is unsupported; rebuild this corpus with Qdrant")
        if meta.get("hierarchy_node_count") is None and any(chunk.parent_id for chunk in chunks):
            raise ValueError("Parented chunks require hierarchy nodes")
        tokens = json.loads((source / "tokens.json").read_text(encoding="utf-8"))
        client = qdrant_client or QdrantClient(url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"))
        if not client.collection_exists(meta["collection"]):
            raise FileNotFoundError(f"Qdrant collection not found: {meta['collection']}")
        if client.count(meta["collection"], exact=True).count != len(chunks):
            raise ValueError("Qdrant collection does not match chunk count")
        hierarchy = None
        if meta.get("hierarchy_node_count") is not None:
            nodes = [json.loads(line) for line in (source / "hierarchy.jsonl").read_text(encoding="utf-8").splitlines() if line]
            hierarchy = validate_hierarchy(chunks, nodes)
            if len(hierarchy) != meta["hierarchy_node_count"]:
                raise ValueError("Index hierarchy does not match metadata")
        return cls(
            {chunk.chunk_id: chunk for chunk in chunks},
            meta["tokenizer_mode"],
            BM25L(tokens),
            client,
            meta["collection"],
            meta["embedding_model"],
            meta.get("embedding_revision", DEFAULT_BGE_M3_REVISION if meta["embedding_model"] == "BAAI/bge-m3" else None),
            hierarchy=hierarchy,
        )

    def expand_chunk(self, chunk: Chunk, max_words: int = 900) -> Chunk:
        if max_words <= 0:
            raise ValueError("max_words must be positive")
        parent = (self.hierarchy or {}).get(chunk.parent_id)
        if parent is None:
            return chunk
        siblings = [self.chunks[child_id] for child_id in parent["children"] if child_id in self.chunks]
        position = next(index for index, sibling in enumerate(siblings) if sibling.chunk_id == chunk.chunk_id)
        start, end = chunk.word_start, chunk.word_end
        selected = [chunk]
        left, right = position - 1, position + 1
        while left >= 0 or right < len(siblings):
            options = []
            if left >= 0:
                candidate = siblings[left]
                options.append((end - min(start, candidate.word_start), "left", candidate))
            if right < len(siblings):
                candidate = siblings[right]
                options.append((max(end, candidate.word_end) - start, "right", candidate))
            fitting = [item for item in options if item[0] <= max_words]
            if not fitting:
                break
            _, side, candidate = min(fitting, key=lambda item: item[0])
            selected.append(candidate)
            start, end = min(start, candidate.word_start), max(end, candidate.word_end)
            if side == "left":
                left -= 1
            else:
                right += 1
        body = " ".join(parent["text"].split()[start:end])
        page_start = min(item.page for item in selected)
        page_end = max(item.page_end or item.page for item in selected)
        section = (self.hierarchy or {}).get(parent["parent_id"])
        if section and section["kind"] in {"section", "document"} and parent["kind"] == "page":
            pages = [self.hierarchy[child_id] for child_id in section["children"] if child_id in self.hierarchy and self.hierarchy[child_id]["kind"] == "page"]
            position = next(index for index, page in enumerate(pages) if page["node_id"] == parent["node_id"])
            parts = [body]
            used = len(body.split())
            for side, adjacent in (("left", position - 1), ("right", position + 1)):
                if not 0 <= adjacent < len(pages):
                    continue
                page = pages[adjacent]
                if not page["text"] or used + len(page["text"].split()) > max_words:
                    continue
                if side == "left":
                    parts.insert(0, page["text"])
                    page_start = min(page_start, int(page["title"].split()[-1]))
                else:
                    parts.append(page["text"])
                    page_end = max(page_end, int(page["title"].split()[-1]))
                used += len(page["text"].split())
            body = " ".join(parts)
        text = f"{chunk.section}\n{body}" if chunk.section else body
        return replace(chunk, text=text, page=page_start, page_end=page_end, word_start=start, word_end=end)

    def _encoder_instance(self):
        if self._encoder is None:
            self._encoder = load_encoder(self.embedding_model, self.embedding_revision)
        return self._encoder

    def sparse_scores(self, query: str, limit: int) -> dict[str, float]:
        identifiers = list(self.chunks)
        values = self.bm25.get_scores(tokenize_vi(query, self.tokenizer_mode))
        order = sorted(range(len(values)), key=lambda index: -values[index])[:limit]
        return {identifiers[index]: float(values[index]) for index in order if values[index] > 0}

    def dense_scores(self, query: str, limit: int) -> dict[str, float]:
        vector = self._encoder_instance().encode([query], normalize_embeddings=True)[0]
        points = self.qdrant.query_points(
            collection_name=self.collection,
            query=vector.tolist() if hasattr(vector, "tolist") else list(vector),
            limit=limit,
            with_payload=True,
            query_filter=models.Filter(must=[models.FieldCondition(
                key="doc_id", match=models.MatchAny(any=list(self.doc_ids)),
            )]) if self.doc_ids else None,
        ).points
        return {point.payload["chunk_id"]: float(point.score) for point in points}


def retrieve(query: str, index: RetrievalIndex, config: RagConfig) -> list[SearchResult]:
    sparse = index.sparse_scores(query, config.top_l) if config.method != "dense" else {}
    dense = index.dense_scores(query, config.top_l) if config.method != "bm25" else {}
    if config.method == "bm25":
        ranked = sorted(sparse, key=lambda key: (-sparse[key], key))
        return [SearchResult(index.chunks[key], sparse[key], rank, "bm25") for rank, key in enumerate(ranked, 1)]
    if config.method == "dense":
        ranked = sorted(dense, key=lambda key: (-dense[key], key))
        return [SearchResult(index.chunks[key], dense[key], rank, "dense") for rank, key in enumerate(ranked, 1)]
    if config.method == "rrf":
        sparse_results = [SearchResult(index.chunks[key], value, rank, "bm25") for rank, (key, value) in enumerate(sorted(sparse.items(), key=lambda item: (-item[1], item[0])), 1)]
        dense_results = [SearchResult(index.chunks[key], value, rank, "dense") for rank, (key, value) in enumerate(sorted(dense.items(), key=lambda item: (-item[1], item[0])), 1)]
        return fuse_rrf([sparse_results, dense_results], config.rrf_k)
    alpha = config.alpha
    if config.method == "adaptive":
        alpha, _ = adaptive_alpha(query, config.alpha, config.adaptive_beta, self_idf(index))
    if config.method not in {"weighted", "adaptive"}:
        raise ValueError(f"Unsupported retrieval method: {config.method}")
    return fuse_weighted(sparse, dense, index.chunks, alpha)


def self_idf(index: RetrievalIndex) -> dict[str, float]:
    return {token: float(value) for token, value in index.bm25.idf.items()}
