"""Configuration centrale — FOURNI, FONCTIONNEL.

- Charge le fichier `.env` (python-dotenv). Aucune clé n'est jamais écrite en dur.
- Expose `settings` (valeurs de configuration), un client de chat LiteLLM
  (`get_chat_model`), des embeddings LiteLLM (`embed_texts`, `embed_query`) et
  un client Langfuse (`get_langfuse`).

Règle du projet : TOUS les appels LLM / embeddings passent par LiteLLM.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")
load_dotenv()  # filet de sécurité : .env du dossier courant


def _env(name: str, default: str = "") -> str:
    """Lit une variable d'environnement ; un placeholder REPLACE_ME vaut « non renseigné »."""
    value = os.getenv(name, default).strip()
    return "" if "REPLACE_ME" in value else value


@dataclass(frozen=True)
class Settings:
    llm_api_key: str
    llm_api_base: str
    chat_model: str
    embedding_model: str
    rerank_api_key: str
    rerank_api_base: str
    rerank_model: str
    langfuse_public_key: str
    langfuse_secret_key: str
    langfuse_host: str
    docs_dir: Path
    chroma_dir: Path
    collection_name: str
    chunk_size: int
    chunk_overlap: int

    @classmethod
    def from_env(cls) -> "Settings":
        chroma_dir = Path(_env("CHROMA_DIR", ".chroma"))
        if not chroma_dir.is_absolute():
            chroma_dir = PROJECT_ROOT / chroma_dir
        return cls(
            llm_api_key=_env("LLM_API_KEY"),
            llm_api_base=_env("LLM_API_BASE"),
            chat_model=_env("MODELE", "openai/gpt-4o-mini"),
            embedding_model=_env("MODELE_EMBEDDING", "openai/text-embedding-3-small"),
            rerank_api_key=_env("RERANK_API_KEY"),
            rerank_api_base=_env("RERANK_API_BASE"),
            rerank_model=_env("MODELE_RERANK", "azure_ai/cohere-rerank-v3-5"),
            langfuse_public_key=_env("LANGFUSE_PUBLIC_KEY"),
            langfuse_secret_key=_env("LANGFUSE_SECRET_KEY"),
            langfuse_host=_env("LANGFUSE_HOST", "https://cloud.langfuse.com"),
            docs_dir=PROJECT_ROOT / "data" / "docs",
            chroma_dir=chroma_dir,
            collection_name="orange_bank_docs",
            chunk_size=int(_env("CHUNK_SIZE", "700")),
            chunk_overlap=int(_env("CHUNK_OVERLAP", "100")),
        )

    @property
    def llm_configured(self) -> bool:
        return bool(self.llm_api_key)

    @property
    def rerank_enabled(self) -> bool:
        return bool(self.rerank_api_key and self.rerank_api_base)

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


settings = Settings.from_env()

# Langfuse lit ses variables dans l'environnement.
if settings.langfuse_enabled:
    os.environ["LANGFUSE_PUBLIC_KEY"] = settings.langfuse_public_key
    os.environ["LANGFUSE_SECRET_KEY"] = settings.langfuse_secret_key
    os.environ["LANGFUSE_HOST"] = settings.langfuse_host


def require_llm_config() -> None:
    """Échoue avec un message clair si la clé LiteLLM est absente."""
    if not settings.llm_configured:
        raise RuntimeError(
            "LLM_API_KEY n'est pas renseignée. Copiez .env.example en .env "
            "et renseignez vos clés LiteLLM."
        )


def litellm_kwargs() -> dict:
    """Arguments de connexion communs à tous les appels LiteLLM."""
    kwargs = {"api_key": settings.llm_api_key}
    if settings.llm_api_base:
        kwargs["api_base"] = settings.llm_api_base
    return kwargs


def _configure_litellm() -> None:
    import litellm

    litellm.drop_params = True  # ignore les paramètres non supportés par un modèle
    litellm.suppress_debug_info = True


def get_chat_model(temperature: float | None = None):
    """Modèle de chat LangChain adossé à LiteLLM (supporte `bind_tools`)."""
    require_llm_config()
    _configure_litellm()
    from langchain_litellm import ChatLiteLLM

    extra = {} if temperature is None else {"temperature": temperature}  # certains modèles refusent ce paramètre
    return ChatLiteLLM(model=settings.chat_model, **extra, **litellm_kwargs())


def embed_texts(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """Calcule les embeddings de `texts` via LiteLLM (ordre préservé)."""
    require_llm_config()
    _configure_litellm()
    import litellm

    vectors: list[list[float]] = []
    for start in range(0, len(texts), batch_size):
        response = litellm.embedding(
            model=settings.embedding_model,
            input=texts[start : start + batch_size],
            **litellm_kwargs(),
        )
        for item in response.data:
            vectors.append(item["embedding"] if isinstance(item, dict) else item.embedding)
    return vectors


def embed_query(text: str) -> list[float]:
    """Embedding d'une seule requête."""
    return embed_texts([text])[0]


@lru_cache(maxsize=1)
def get_langfuse():
    """Client Langfuse, ou `None` si les clés LANGFUSE_* ne sont pas renseignées."""
    if not settings.langfuse_enabled:
        return None
    from langfuse import get_client

    return get_client()
