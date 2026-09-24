import dataclasses
from dataclasses import asdict, dataclass, fields
from typing import Any

from src.text import TOKENIZERS

FUSIONS = ("none", "rrf", "weighted", "adaptive")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    course: str
    source_type: str
    page: int
    section: str
    text: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class SearchResult:
    chunk: Chunk
    score: float
    rank: int
    source: str


@dataclass(frozen=True)
class RagConfig:
    method: str = "adaptive"
    top_l: int = 100
    rerank_n: int = 20
    context_k: int = 5
    alpha: float = 0.5
    rrf_k: int = 60
    refusal_threshold: float = 0.0
    use_reranker: bool = True
    model: str = ""
    temperature: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Block:
    page: int
    heading_path: tuple[str, ...]
    text: str
    warning: str = ""


@dataclass(frozen=True)
class StageScores:
    rank: int
    sparse_score: float | None = None
    sparse_rank: int | None = None
    dense_score: float | None = None
    dense_rank: int | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Any
    scores: StageScores

    @property
    def score(self) -> float:
        stages = self.scores
        for value in (stages.rerank_score, stages.fusion_score, stages.sparse_score, stages.dense_score):
            if value is not None:
                return value
        return float("-inf")


@dataclass(frozen=True)
class PipelineConfig:
    sparse: bool = True
    dense: bool = True
    fusion: str = "weighted"
    alpha: float = 0.5
    rrf_k: int = 60
    adaptive_beta: float = 0.3
    rerank: bool = True
    rerank_n: int = 30
    top_l: int = 100
    context_k: int = 5
    tokenizer: str = "whitespace"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    refusal_threshold: float = 0.0
    llm_model: str = ""
    temperature: float = 0.0
    timeout_seconds: int = 60

    def __post_init__(self) -> None:
        if self.fusion not in FUSIONS:
            raise ValueError(f"Unsupported fusion: {self.fusion}")
        if self.fusion == "none" and self.sparse == self.dense:
            raise ValueError("fusion 'none' requires exactly one of sparse or dense")
        if self.fusion != "none" and not (self.sparse and self.dense):
            raise ValueError(f"fusion '{self.fusion}' requires both sparse and dense")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")
        if min(self.top_l, self.rerank_n, self.context_k, self.rrf_k) < 1:
            raise ValueError("top_l, rerank_n, context_k and rrf_k must be positive")
        if self.rerank and not self.context_k <= self.rerank_n <= self.top_l:
            raise ValueError("Require context_k <= rerank_n <= top_l")
        if not self.rerank and self.context_k > self.top_l:
            raise ValueError("Require context_k <= top_l")
        if self.tokenizer not in TOKENIZERS:
            raise ValueError(f"Unsupported tokenizer: {self.tokenizer}")

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "PipelineConfig":
        unknown = set(data) - {field.name for field in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def replace(self, **changes) -> "PipelineConfig":
        return dataclasses.replace(self, **changes)
