import json
from types import SimpleNamespace

import pytest
from qdrant_client import QdrantClient

from src.ingestion import PageText, build_hierarchy
from src.models import Chunk, RagConfig
from src.retrieval import RetrievalIndex, load_encoder, minmax_scores


@pytest.fixture(autouse=True)
def clear_encoder_cache():
    load_encoder.cache_clear()
    yield
    load_encoder.cache_clear()


def test_encoder_reused_between_chat_and_index_build(monkeypatch):
    created = []

    class Encoder:
        def __init__(self, *_args, **_kwargs):
            created.append(self)

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", Encoder)
    assert load_encoder("same-model", None) is load_encoder("same-model", None)
    assert len(created) == 1


def test_qdrant_index_persists_searchable_chunks_without_numpy_file(tmp_path, monkeypatch):
    class Encoder:
        def __init__(self, model_name, **kwargs):
            assert model_name == "test-model"

        def encode(self, texts, **kwargs):
            return [[1.0, 0.0] if text in {"toán học", "định nghĩa"} else [0.0, 1.0] for text in texts]

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", Encoder)
    chunks = [
        Chunk("c1", "d1", "Toán", "textbook", 1, "", "toán học"),
        Chunk("c2", "d2", "Văn", "slide", 2, "", "văn học"),
    ]
    client = QdrantClient(":memory:")
    index_dir = tmp_path / "version1"
    RetrievalIndex.build(chunks, index_dir, "whitespace", "test-model", qdrant_client=client)
    index = RetrievalIndex.load(index_dir, qdrant_client=client)
    assert not (index_dir / "embeddings.npy").exists()
    assert index.dense_scores("định nghĩa", 2)["c1"] > index.dense_scores("định nghĩa", 2)["c2"]
    assert index.sparse_scores("không tồn tại", 2) == {}
    scoped = index.scoped(("d2",))
    assert set(scoped.dense_scores("định nghĩa", 2)) == {"c2"}
    assert set(scoped.chunks) == {"c2"}
    assert scoped.sparse_scores("văn học", 2)["c2"] > 0
    assert scoped.sparse_scores("không tồn tại", 2) == {}
    assert scoped.scoped(()) is scoped
    with pytest.raises(ValueError, match="not in this index"):
        index.scoped(("missing",))
    meta = json.loads((index_dir / "index_meta.json").read_text())
    assert meta["backend"] == "qdrant"
    assert "embedding_revision" in meta


def test_all_zero_scores_do_not_become_positive():
    assert minmax_scores({"a": 0.0, "b": 0.0}) == {"a": 0.0, "b": 0.0}


def test_gpu_embedding_uses_single_item_batch_for_small_vram(tmp_path, monkeypatch):
    class Encoder:
        device = SimpleNamespace(type="cuda")

        def __init__(self, model_name, **kwargs):
            self.half_precision = False

        def half(self):
            self.half_precision = True
            return self

        def encode(self, texts, **kwargs):
            assert self.half_precision
            assert kwargs["batch_size"] == 1
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", Encoder)
    RetrievalIndex.build([Chunk("c1", "d1", "M", "slide", 1, "", "nội dung")], tmp_path / "gpu", "whitespace", "test", qdrant_client=QdrantClient(":memory:"))


def test_hierarchy_survives_index_reload_and_expands_neighboring_leaf(tmp_path, monkeypatch):
    class Encoder:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, texts, **_kwargs):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", Encoder)
    chunks, nodes = build_hierarchy(
        [PageText(1, "Bài 1", "một hai ba bốn năm sáu bảy tám"), PageText(2, "Bài 1", "chín mười mười_một mười_hai mười_ba mười_bốn mười_lăm mười_sáu")],
        doc_id="d", course="M", source_type="slide", chunk_tokens=8, overlap_tokens=2,
    )
    client = QdrantClient(":memory:")
    index_dir = tmp_path / "hierarchical"
    RetrievalIndex.build(chunks, index_dir, "whitespace", "test", qdrant_client=client, hierarchy_nodes=nodes)
    loaded = RetrievalIndex.load(index_dir, qdrant_client=client)
    expanded = loaded.expand_chunk(chunks[0], max_words=24)

    assert len(loaded.hierarchy) == len(nodes)
    assert (index_dir / "hierarchy.jsonl").is_file()
    assert expanded.parent_id == chunks[0].parent_id
    assert expanded.page == 1 and expanded.page_end == 2
    assert "chín" in expanded.text
    assert expanded.text.count("một hai ba") == 1


def test_index_rejects_hierarchy_with_unlinked_leaf(tmp_path, monkeypatch):
    class Encoder:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, texts, **_kwargs):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", Encoder)
    chunks, nodes = build_hierarchy([PageText(1, "", "Một đoạn nội dung")], doc_id="d", course="M", source_type="textbook", chunk_tokens=20, overlap_tokens=2)
    client = QdrantClient(":memory:")
    index_dir = tmp_path / "broken-tree"
    RetrievalIndex.build(chunks, index_dir, "whitespace", "test", qdrant_client=client, hierarchy_nodes=nodes)
    nodes[-1]["children"].remove(chunks[0].chunk_id)
    (index_dir / "hierarchy.jsonl").write_text("\n".join(json.dumps(node) for node in nodes) + "\n")

    with pytest.raises(ValueError, match="child|parent"):
        RetrievalIndex.load(index_dir, qdrant_client=client)


def test_encoder_uses_cache_first_then_downloads_if_missing(tmp_path, monkeypatch):
    attempts = []

    class Encoder:
        def __init__(self, *_args, **kwargs):
            attempts.append(kwargs.get("local_files_only"))
            if kwargs.get("local_files_only"):
                raise OSError("not cached")

        def encode(self, texts, **_kwargs):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", Encoder)
    RetrievalIndex.build([Chunk("c1", "d1", "M", "slide", 1, "", "nội dung")], tmp_path / "fallback", "whitespace", "test", qdrant_client=QdrantClient(":memory:"))
    assert attempts == [True, False]


def test_default_parent_budget_expands_real_size_leaf(tmp_path, monkeypatch):
    class Encoder:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, texts, **_kwargs):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", Encoder)
    chunks, nodes = build_hierarchy([PageText(1, "Bài 1", " ".join(f"t{i}" for i in range(900)))], doc_id="d", course="M", source_type="slide", chunk_tokens=450, overlap_tokens=75)
    index = RetrievalIndex.build(chunks, tmp_path / "real-size", "whitespace", "test", qdrant_client=QdrantClient(":memory:"), hierarchy_nodes=nodes)
    expanded = index.expand_chunk(chunks[0], RagConfig().context_parent_words)
    assert expanded.word_end > chunks[0].word_end
    assert "t500" in expanded.text


def test_index_refuses_parented_chunks_without_hierarchy(tmp_path, monkeypatch):
    class Encoder:
        def __init__(self, *_args, **_kwargs):
            pass

        def encode(self, texts, **_kwargs):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("sentence_transformers.SentenceTransformer", Encoder)
    chunk = Chunk("c1", "d1", "M", "slide", 1, "", "nội dung", 1, "parent", 0, 1)
    with pytest.raises(ValueError, match="hierarchy"):
        RetrievalIndex.build([chunk], tmp_path / "missing-tree", "whitespace", "test", qdrant_client=QdrantClient(":memory:"))


def test_failed_upsert_removes_partial_index_and_collection(tmp_path, monkeypatch):
    class Encoder:
        def encode(self, texts, **kwargs):
            return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr("src.retrieval.load_encoder", lambda *_: Encoder())
    client = QdrantClient(":memory:")

    def fail_upsert(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(client, "upsert", fail_upsert)
    target = tmp_path / "partial"
    with pytest.raises(RuntimeError, match="disk full"):
        RetrievalIndex.build([Chunk("c1", "d1", "M", "slide", 1, "", "nội dung")], target, "whitespace", "test", qdrant_client=client)
    assert not target.exists()
    assert not client.collection_exists("ppl_partial")
