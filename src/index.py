import json
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.models import Chunk
from src.text import normalize_text, tokenize

_ENCODERS: dict = {}


def load_encoder(model_name: str):
    if model_name not in _ENCODERS:
        from sentence_transformers import SentenceTransformer

        _ENCODERS[model_name] = SentenceTransformer(model_name)
    return _ENCODERS[model_name]


def _top(chunk_ids: list[str], values, limit: int, positive_only: bool) -> list[tuple[str, float]]:
    pairs = [
        (chunk_id, float(value))
        for chunk_id, value in zip(chunk_ids, values)
        if not positive_only or value > 0
    ]
    pairs.sort(key=lambda item: (-item[1], item[0]))
    return pairs[:limit]


class RetrievalIndex:
    def __init__(self, directory: Path | str, chunks: list[Chunk], embeddings: np.ndarray, meta: dict, encoder=None):
        self.directory = Path(directory)
        self.chunk_list = chunks
        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.embeddings = embeddings
        self.meta = meta
        self._encoder = encoder
        self._bm25: dict[str, BM25Okapi] = {}
        self._faiss = None

    @property
    def version(self) -> str:
        return self.directory.name

    @property
    def embedding_model(self) -> str:
        return self.meta["embedding_model"]

    @property
    def tokenizers(self) -> list[str]:
        return list(self.meta["tokenizers"])

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        output_dir: Path | str,
        *,
        embedding_model: str,
        tokenizers=("whitespace",),
        dense_backend: str = "numpy",
        bm25_k1: float = 1.5,
        bm25_b: float = 0.75,
        chunking: dict | None = None,
        encoder=None,
    ) -> "RetrievalIndex":
        if not chunks:
            raise ValueError("Cannot build an index without chunks")
        if dense_backend not in ("numpy", "faiss"):
            raise ValueError(f"Unsupported dense backend: {dense_backend}")
        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        encoder = encoder or load_encoder(embedding_model)
        embeddings = np.asarray(
            encoder.encode([chunk.text for chunk in chunks], normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )
        np.save(target / "embeddings.npy", embeddings)
        (target / "chunks.jsonl").write_text(
            "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks) + "\n",
            encoding="utf-8",
        )
        meta = {
            "embedding_model": embedding_model,
            "dense_backend": dense_backend,
            "bm25": {"k1": bm25_k1, "b": bm25_b},
            "chunk_count": len(chunks),
            "chunking": chunking or {},
            "tokenizers": [],
        }
        index = cls(target, chunks, embeddings, meta, encoder)
        if dense_backend == "faiss":
            import faiss

            flat = faiss.IndexFlatIP(embeddings.shape[1])
            flat.add(embeddings)
            faiss.write_index(flat, str(target / "faiss.index"))
        for mode in tokenizers:
            index.add_tokenizer(mode)
        index._write_meta()
        return index

    @classmethod
    def load(cls, directory: Path | str, encoder=None) -> "RetrievalIndex":
        source = Path(directory)
        meta = json.loads((source / "index_meta.json").read_text(encoding="utf-8"))
        chunks = [
            Chunk.from_dict(json.loads(line))
            for line in (source / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(chunks) != meta["chunk_count"]:
            raise ValueError("Index metadata does not match chunk count")
        return cls(source, chunks, np.load(source / "embeddings.npy"), meta, encoder)

    def _write_meta(self) -> None:
        (self.directory / "index_meta.json").write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def add_tokenizer(self, mode: str) -> None:
        tokens = [tokenize(chunk.text, mode) for chunk in self.chunk_list]
        (self.directory / f"tokens_{mode}.json").write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")
        if mode not in self.meta["tokenizers"]:
            self.meta["tokenizers"].append(mode)
        self._bm25.pop(mode, None)
        self._write_meta()

    def _bm25_for(self, mode: str) -> BM25Okapi:
        if mode not in self.meta["tokenizers"]:
            raise ValueError(
                f"Tokenizer '{mode}' is not built for index {self.version}; run "
                f"`python -m src.cli index add-tokenizer --index {self.directory} --tokenizer {mode}`"
            )
        if mode not in self._bm25:
            tokens = json.loads((self.directory / f"tokens_{mode}.json").read_text(encoding="utf-8"))
            parameters = self.meta["bm25"]
            self._bm25[mode] = BM25Okapi(tokens, k1=parameters["k1"], b=parameters["b"])
        return self._bm25[mode]

    def idf(self, mode: str) -> dict[str, float]:
        return {token: float(value) for token, value in self._bm25_for(mode).idf.items()}

    def sparse_search(self, query: str, tokenizer: str, limit: int) -> list[tuple[str, float]]:
        bm25 = self._bm25_for(tokenizer)
        query_tokens = tokenize(query, tokenizer)
        if not query_tokens:
            return []
        return _top(self.chunk_ids, bm25.get_scores(query_tokens), limit, positive_only=True)

    def _encoder_instance(self):
        if self._encoder is None:
            self._encoder = load_encoder(self.embedding_model)
        return self._encoder

    def dense_search(self, query: str, limit: int) -> list[tuple[str, float]]:
        vector = np.asarray(
            self._encoder_instance().encode([normalize_text(query)], normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )[0]
        if self.meta["dense_backend"] == "faiss":
            import faiss

            if self._faiss is None:
                self._faiss = faiss.read_index(str(self.directory / "faiss.index"))
            scores, positions = self._faiss.search(vector[None, :], min(limit, len(self.chunk_ids)))
            pairs = [
                (self.chunk_ids[position], float(score))
                for position, score in zip(positions[0], scores[0])
                if position >= 0
            ]
            return sorted(pairs, key=lambda item: (-item[1], item[0]))
        return _top(self.chunk_ids, self.embeddings @ vector, limit, positive_only=False)
