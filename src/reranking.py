_CROSS_ENCODERS: dict = {}


def load_cross_encoder(model_name: str):
    if model_name not in _CROSS_ENCODERS:
        from sentence_transformers import CrossEncoder

        _CROSS_ENCODERS[model_name] = CrossEncoder(model_name)
    return _CROSS_ENCODERS[model_name]


class Reranker:
    def __init__(self, model_name: str, model=None, cache=None):
        self.model_name = model_name
        self._model = model
        self.cache = cache

    def score(self, query: str, chunks: list, use_cache: bool = True) -> list[float]:
        if not chunks:
            return []
        caching = self.cache is not None and use_cache
        known = (
            self.cache.get_rerank(self.model_name, query, [chunk.chunk_id for chunk in chunks])
            if caching
            else {}
        )
        missing = [chunk for chunk in chunks if chunk.chunk_id not in known]
        if missing:
            model = self._model or load_cross_encoder(self.model_name)
            values = model.predict(
                [(query, chunk.text) for chunk in missing],
                batch_size=min(16, len(missing)),
                show_progress_bar=False,
            )
            fresh = {chunk.chunk_id: float(value) for chunk, value in zip(missing, values)}
            if caching:
                self.cache.put_rerank(self.model_name, query, fresh)
            known = {**known, **fresh}
        return [known[chunk.chunk_id] for chunk in chunks]
