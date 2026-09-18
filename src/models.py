from dataclasses import asdict, dataclass


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
