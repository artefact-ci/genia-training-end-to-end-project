"""Ingestion de la documentation — FOURNI, FONCTIONNEL.

Découpe les fichiers de `data/docs/` en chunks (RecursiveCharacterTextSplitter)
et les indexe dans une collection Chroma locale (embeddings via LiteLLM).

Lancement :  python -m lagune_bank_agent.ingestion
"""
from __future__ import annotations

from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from lagune_bank_agent.config import embed_texts, settings


def load_documents(docs_dir: Path | None = None) -> list[tuple[str, str]]:
    """Retourne la liste des (nom_de_fichier, contenu) des .txt du dossier de docs."""
    docs_dir = docs_dir or settings.docs_dir
    files = sorted(docs_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"Aucun fichier .txt dans {docs_dir}")
    return [(f.name, f.read_text(encoding="utf-8")) for f in files]


def chunk_documents(
    documents: list[tuple[str, str]],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[dict]:
    """Découpe chaque document ; retourne des dicts {id, text, metadata}."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or settings.chunk_size,
        chunk_overlap=chunk_overlap if chunk_overlap is not None else settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for source, content in documents:
        for index, text in enumerate(splitter.split_text(content)):
            chunks.append(
                {
                    "id": f"{source}::{index}",
                    "text": text,
                    "metadata": {"source": source, "chunk_index": index},
                }
            )
    return chunks


def _client() -> chromadb.ClientAPI:
    settings.chroma_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(settings.chroma_dir),
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def get_collection():
    """Collection Chroma (créée vide si elle n'existe pas). Similarité cosinus."""
    return _client().get_or_create_collection(
        name=settings.collection_name, metadata={"hnsw:space": "cosine"}
    )


def build_index(reset: bool = True) -> int:
    """(Ré)indexe toute la documentation ; retourne le nombre de chunks indexés."""
    client = _client()
    if reset:
        try:
            client.delete_collection(settings.collection_name)
        except Exception:
            pass  # la collection n'existait pas
    collection = client.get_or_create_collection(
        name=settings.collection_name, metadata={"hnsw:space": "cosine"}
    )
    chunks = chunk_documents(load_documents())
    batch = 64
    for start in range(0, len(chunks), batch):
        part = chunks[start : start + batch]
        texts = [c["text"] for c in part]
        collection.add(
            ids=[c["id"] for c in part],
            documents=texts,
            metadatas=[c["metadata"] for c in part],
            embeddings=embed_texts(texts),
        )
    return len(chunks)


def ensure_index():
    """Retourne la collection, en indexant d'abord la documentation si elle est vide."""
    collection = get_collection()
    if collection.count() == 0:
        print("[ingestion] Index vide : indexation de la documentation…")
        build_index(reset=False)
        collection = get_collection()
    return collection


def main() -> None:
    print(f"Documents : {settings.docs_dir}")
    print(f"Modèle d'embeddings : {settings.embedding_model}")
    count = build_index(reset=True)
    print(f"OK — {count} chunks indexés dans {settings.chroma_dir}")


if __name__ == "__main__":
    main()
