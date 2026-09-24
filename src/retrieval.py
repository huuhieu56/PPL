import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from src.text import tokenize as tokenize_vi
from src.models import Chunk, RagConfig, SearchResult


def minmax_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    low, high = min(scores.values()), max(scores.values())
    if math.isclose(low, high):
        return {key: 1.0 for key in scores}
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
    signals = {
        "digit": sum(character.isdigit() for character in compact) / length,
        "symbol": sum(not character.isalnum() and not character.isspace() for character in compact)
        / length,
        "code": float(bool(re.search(r"\b[A-Z]{2,}[A-Z0-9-]*\d+[A-Z0-9-]*\b", compact))),
        "max_idf": max((idf.get(token, 0.0) for token in tokens), default=0.0),
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


@dataclass
class RetrievalIndex:
    chunks: dict[str, Chunk]
    tokenizer_mode: str
    bm25: BM25Okapi
    embeddings: np.ndarray
    embedding_model: str
    _encoder: object | None = None

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        output_dir: Path | str,
        tokenizer_mode: str,
        embedding_model: str,
    ) -> "RetrievalIndex":
        from sentence_transformers import SentenceTransformer

        target = Path(output_dir)
        target.mkdir(parents=True, exist_ok=True)
        tokens = [tokenize_vi(chunk.text, tokenizer_mode) for chunk in chunks]
        encoder = SentenceTransformer(embedding_model)
        embeddings = np.asarray(
            encoder.encode(
                [chunk.text for chunk in chunks],
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=np.float32,
        )
        np.save(target / "embeddings.npy", embeddings)
        (target / "chunks.jsonl").write_text(
            "\n".join(json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in chunks) + "\n",
            encoding="utf-8",
        )
        (target / "tokens.json").write_text(json.dumps(tokens, ensure_ascii=False), encoding="utf-8")
        (target / "index_meta.json").write_text(
            json.dumps(
                {
                    "tokenizer_mode": tokenizer_mode,
                    "embedding_model": embedding_model,
                    "chunk_count": len(chunks),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        return cls(
            {chunk.chunk_id: chunk for chunk in chunks},
            tokenizer_mode,
            BM25Okapi(tokens),
            embeddings,
            embedding_model,
            encoder,
        )

    @classmethod
    def load(cls, index_dir: Path | str) -> "RetrievalIndex":
        source = Path(index_dir)
        meta = json.loads((source / "index_meta.json").read_text(encoding="utf-8"))
        chunks = [
            Chunk(**json.loads(line))
            for line in (source / "chunks.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        if len(chunks) != meta["chunk_count"]:
            raise ValueError("Index metadata does not match chunk count")
        tokens = json.loads((source / "tokens.json").read_text(encoding="utf-8"))
        return cls(
            {chunk.chunk_id: chunk for chunk in chunks},
            meta["tokenizer_mode"],
            BM25Okapi(tokens),
            np.load(source / "embeddings.npy"),
            meta["embedding_model"],
        )

    def _encoder_instance(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.embedding_model)
        return self._encoder

    def sparse_scores(self, query: str, limit: int) -> dict[str, float]:
        identifiers = list(self.chunks)
        values = self.bm25.get_scores(tokenize_vi(query, self.tokenizer_mode))
        order = np.argsort(values)[::-1][:limit]
        return {identifiers[index]: float(values[index]) for index in order}

    def dense_scores(self, query: str, limit: int) -> dict[str, float]:
        identifiers = list(self.chunks)
        query_embedding = np.asarray(
            self._encoder_instance().encode([query], normalize_embeddings=True), dtype=np.float32
        )[0]
        values = self.embeddings @ query_embedding
        order = np.argsort(values)[::-1][:limit]
        return {identifiers[index]: float(values[index]) for index in order}


def retrieve(query: str, index: RetrievalIndex, config: RagConfig) -> list[SearchResult]:
    sparse = index.sparse_scores(query, config.top_l)
    dense = index.dense_scores(query, config.top_l)
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
        alpha, _ = adaptive_alpha(query, config.alpha, 0.3, self_idf(index))
    if config.method not in {"weighted", "adaptive"}:
        raise ValueError(f"Unsupported retrieval method: {config.method}")
    return fuse_weighted(sparse, dense, index.chunks, alpha)


def self_idf(index: RetrievalIndex) -> dict[str, float]:
    return {token: float(value) for token, value in index.bm25.idf.items()}
