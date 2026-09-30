import os
from functools import lru_cache


DEFAULT_BGE_M3_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"


def embedding_config(model: str) -> dict[str, str]:
    provider = os.getenv("EMBEDDING_PROVIDER", "local").strip().lower()
    if provider not in {"local", "api"}:
        raise ValueError("EMBEDDING_PROVIDER must be 'local' or 'api'")
    base_url = os.getenv("EMBEDDING_BASE_URL", "https://api.openai.com/v1").rstrip("/") if provider == "api" else ""
    if provider == "api" and (not os.getenv("EMBEDDING_MODEL") or not os.getenv("EMBEDDING_API_KEY")):
        raise ValueError("EMBEDDING_MODEL and EMBEDDING_API_KEY are required for API embeddings")
    return {"provider": provider, "model": model, "base_url": base_url}


@lru_cache(maxsize=2)
def api_embeddings(model: str, base_url: str, api_key: str):
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(
        model=model, api_key=api_key, base_url=base_url,
        check_embedding_ctx_length=False,
    )


def embed_documents(texts: list[str], model: str, revision: str | None):
    config = embedding_config(model)
    if config["provider"] == "api":
        return api_embeddings(model, config["base_url"], os.environ["EMBEDDING_API_KEY"]).embed_documents(texts)
    encoder = load_encoder(model, revision)
    return encoder.encode(
        texts,
        batch_size=1 if getattr(getattr(encoder, "device", None), "type", None) == "cuda" else 8,
        normalize_embeddings=True,
        show_progress_bar=False,
    )


def embed_query(text: str, model: str, revision: str | None):
    config = embedding_config(model)
    if config["provider"] == "api":
        return api_embeddings(model, config["base_url"], os.environ["EMBEDDING_API_KEY"]).embed_query(text)
    return load_encoder(model, revision).encode([text], normalize_embeddings=True)[0]


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
