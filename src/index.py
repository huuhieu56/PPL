import json
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass, replace
from pathlib import Path

from qdrant_client import QdrantClient, models
from rank_bm25 import BM25L

from src.embeddings import DEFAULT_BGE_M3_REVISION, embed_documents, embed_query, embedding_config
from src.chunking import tokenize_vi, validate_hierarchy
from src.models import Chunk


def make_qdrant_client() -> QdrantClient:
    return QdrantClient(
        url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
        api_key=os.getenv("QDRANT_API_KEY") or None,
    )


@dataclass
class RetrievalIndex:
    chunks: dict[str, Chunk]
    tokenizer_mode: str
    bm25: BM25L
    qdrant: QdrantClient
    collection: str
    embedding_model: str
    embedding_revision: str | None = None
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
        client = qdrant_client or make_qdrant_client()
        collection = "ppl_" + re.sub(r"[^A-Za-z0-9_]", "_", target.name)
        if client.collection_exists(collection):
            raise FileExistsError(f"Qdrant collection already exists: {collection}")
        vectors = embed_documents([chunk.text for chunk in chunks], embedding_model, revision)
        if len(vectors) != len(chunks):
            raise ValueError("Embedding API returned an unexpected number of vectors")
        staging = Path(tempfile.mkdtemp(prefix=f".{target.name}-", dir=target.parent))
        created = False
        try:
            client.create_collection(collection, vectors_config=models.VectorParams(size=len(vectors[0]), distance=models.Distance.COSINE))
            created = True
            for start in range(0, len(chunks), 32):
                points = []
                for chunk, vector in zip(chunks[start : start + 32], vectors[start : start + 32]):
                    points.append(models.PointStruct(
                        id=str(uuid.uuid5(uuid.NAMESPACE_URL, chunk.chunk_id)),
                        vector=vector.tolist() if hasattr(vector, "tolist") else list(vector),
                        payload={"chunk_id": chunk.chunk_id, "doc_id": chunk.doc_id},
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
                    "embedding_config": embedding_config(embedding_model),
                    "chunk_count": len(chunks),
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
        if meta["hierarchy_node_count"] is None and any(chunk.parent_id for chunk in chunks):
            raise ValueError("Parented chunks require hierarchy nodes")
        tokens = [tokenize_vi(chunk.text, meta["tokenizer_mode"]) for chunk in chunks]
        expected = meta["embedding_config"]
        if expected != embedding_config(os.getenv("EMBEDDING_MODEL") or meta["embedding_model"]):
            raise ValueError("Embedding configuration differs from this index; rebuild the corpus")
        client = qdrant_client or make_qdrant_client()
        if not client.collection_exists(meta["collection"]):
            raise FileNotFoundError(f"Qdrant collection not found: {meta['collection']}")
        hierarchy = None
        if meta["hierarchy_node_count"] is not None:
            nodes = [json.loads(line) for line in (source / "hierarchy.jsonl").read_text(encoding="utf-8").splitlines() if line]
            hierarchy = validate_hierarchy(chunks, nodes)
        return cls(
            {chunk.chunk_id: chunk for chunk in chunks},
            meta["tokenizer_mode"],
            BM25L(tokens),
            client,
            meta["collection"],
            meta["embedding_model"],
            meta["embedding_revision"],
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

    def sparse_scores(self, query: str, limit: int) -> dict[str, float]:
        identifiers = list(self.chunks)
        values = self.bm25.get_scores(tokenize_vi(query, self.tokenizer_mode))
        order = sorted(range(len(values)), key=lambda index: -values[index])[:limit]
        return {identifiers[index]: float(values[index]) for index in order if values[index] > 0}

    def dense_scores(self, query: str, limit: int) -> dict[str, float]:
        vector = embed_query(query, self.embedding_model, self.embedding_revision)
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
