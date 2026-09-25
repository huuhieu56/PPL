import time
from dataclasses import dataclass, field
from typing import Callable

from src.cache import RetrievalCache
from src.fusion import adaptive_alpha, fuse_rrf, fuse_weighted, rank_ids
from src.models import PipelineConfig, RetrievedChunk, StageScores
from src.reranking import Reranker
from src.text import normalize_text, tokenize

STAGES = ("sparse", "dense", "fusion", "rerank")


@dataclass(frozen=True)
class PipelineResult:
    results: list[RetrievedChunk]
    timings_ms: dict[str, float] = field(default_factory=dict)
    alpha_used: float | None = None


def _cuda_synchronize() -> None:
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.synchronize()


class RetrievalPipeline:
    def __init__(
        self,
        index,
        cache: RetrievalCache | None = None,
        reranker_factory: Callable[[str], Reranker] | None = None,
        synchronize: Callable[[], None] | None = None,
    ):
        self.index = index
        self.cache = cache
        self._reranker_factory = reranker_factory or (lambda name: Reranker(name, cache=cache))
        self._rerankers: dict[str, Reranker] = {}
        self._synchronize = synchronize or _cuda_synchronize

    def _clock(self) -> float:
        self._synchronize()
        return time.perf_counter()

    def _reranker(self, model_name: str) -> Reranker:
        if model_name not in self._rerankers:
            self._rerankers[model_name] = self._reranker_factory(model_name)
        return self._rerankers[model_name]

    def _first_stage(self, branch: str, variant: str, query: str, top_l: int, search, use_cache: bool):
        if self.cache is None or not use_cache:
            return search()
        key = RetrievalCache.first_stage_key(self.index.version, branch, variant, query, top_l)
        cached = self.cache.get_first_stage(key)
        if cached is None:
            cached = search()
            self.cache.put_first_stage(key, cached)
        return cached

    def run(self, query: str, config: PipelineConfig, use_cache: bool = True) -> PipelineResult:
        query = normalize_text(query)
        timings = {stage: 0.0 for stage in STAGES}
        started = self._clock()
        sparse: list[tuple[str, float]] = []
        dense: list[tuple[str, float]] = []
        if config.sparse:
            mark = self._clock()
            sparse = self._first_stage(
                "sparse", config.tokenizer, query, config.top_l,
                lambda: self.index.sparse_search(query, config.tokenizer, config.top_l), use_cache,
            )
            timings["sparse"] = (self._clock() - mark) * 1000
        if config.dense:
            mark = self._clock()
            dense = self._first_stage(
                "dense", self.index.embedding_model, query, config.top_l,
                lambda: self.index.dense_search(query, config.top_l), use_cache,
            )
            timings["dense"] = (self._clock() - mark) * 1000

        mark = self._clock()
        sparse_scores, dense_scores = dict(sparse), dict(dense)
        sparse_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(sparse, start=1)}
        dense_ranks = {chunk_id: rank for rank, (chunk_id, _) in enumerate(dense, start=1)}
        fused: dict[str, float] | None = None
        alpha_used = None
        if config.fusion == "none":
            order = [chunk_id for chunk_id, _ in (sparse if config.sparse else dense)]
        elif config.fusion == "rrf":
            fused = fuse_rrf([[chunk_id for chunk_id, _ in sparse], [chunk_id for chunk_id, _ in dense]], config.rrf_k)
            order = rank_ids(fused)
        else:
            alpha_used = config.alpha
            if config.fusion == "adaptive":
                alpha_used, _ = adaptive_alpha(
                    query, tokenize(query, config.tokenizer), config.alpha,
                    config.adaptive_beta, self.index.idf(config.tokenizer),
                )
            fused = fuse_weighted(sparse_scores, dense_scores, alpha_used)
            order = rank_ids(fused)
        timings["fusion"] = (self._clock() - mark) * 1000

        rerank_scores: dict[str, float] = {}
        if config.rerank and order:
            mark = self._clock()
            head = order[: config.rerank_n]
            values = self._reranker(config.reranker_model).score(
                query, [self.index.chunks[chunk_id] for chunk_id in head], use_cache=use_cache
            )
            rerank_scores = dict(zip(head, values))
            order = sorted(head, key=lambda chunk_id: (-rerank_scores[chunk_id], chunk_id)) + order[config.rerank_n :]
            timings["rerank"] = (self._clock() - mark) * 1000
        timings["total"] = (self._clock() - started) * 1000

        results = [
            RetrievedChunk(
                self.index.chunks[chunk_id],
                StageScores(
                    rank=rank,
                    sparse_score=sparse_scores.get(chunk_id),
                    sparse_rank=sparse_ranks.get(chunk_id),
                    dense_score=dense_scores.get(chunk_id),
                    dense_rank=dense_ranks.get(chunk_id),
                    fusion_score=fused.get(chunk_id) if fused is not None else None,
                    rerank_score=rerank_scores.get(chunk_id),
                ),
            )
            for rank, chunk_id in enumerate(order, start=1)
        ]
        return PipelineResult(results, timings, alpha_used)
