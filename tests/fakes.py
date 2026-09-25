import hashlib
import re
import numpy as np

from src.models import Chunk


class FakeEncoder:
    dim = 64

    def __init__(self):
        self.calls = 0

    def encode(self, texts, normalize_embeddings=True, show_progress_bar=False, batch_size=32):
        self.calls += 1
        vectors = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for token in re.findall(r"\w+", text.lower()):
                column = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.dim
                vectors[row, column] += 1.0
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms


class FakeCrossEncoder:
    def __init__(self):
        self.pairs_seen = 0

    def predict(self, pairs, batch_size=16, show_progress_bar=False):
        self.pairs_seen += len(pairs)
        scores = []
        for query, text in pairs:
            query_tokens = set(re.findall(r"\w+", query.lower()))
            text_tokens = set(re.findall(r"\w+", text.lower()))
            scores.append(float(len(query_tokens & text_tokens)))
        return np.asarray(scores, dtype=np.float32)


def make_chunk(chunk_id, text, **fields):
    values = {
        "doc_id": "doc-1",
        "doc_title": "Tài liệu",
        "course": "AI101",
        "source_type": "slide",
        "page": 1,
        "heading_path": (),
        "body": text,
        **fields,
    }
    return Chunk(chunk_id=chunk_id, text=text, **values)
