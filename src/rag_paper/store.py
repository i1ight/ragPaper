from __future__ import annotations

from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from rag_paper.embeddings import EmbeddingProvider
from rag_paper.models import PaperChunk


def normalize_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    normalized: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            normalized[key] = value
        elif isinstance(value, list):
            normalized[key] = ", ".join(str(item) for item in value)
        else:
            normalized[key] = str(value)
    return normalized


class ChromaPaperStore:
    def __init__(self, persist_dir: Path, collection_name: str, embeddings: EmbeddingProvider) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        self.embeddings = embeddings

    def upsert_chunks(self, chunks: list[PaperChunk], batch_size: int = 32) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            vectors = self.embeddings.embed_texts([chunk.text for chunk in batch])
            self.collection.upsert(
                ids=[chunk.id for chunk in batch],
                documents=[chunk.text for chunk in batch],
                metadatas=[normalize_metadata(chunk.metadata) for chunk in batch],
                embeddings=vectors,
            )

    def delete_chunk_ids(self, chunk_ids: list[str]) -> None:
        if chunk_ids:
            self.collection.delete(ids=chunk_ids)

    def update_chunk_metadata(self, chunk_ids: list[str], metadata: dict[str, Any]) -> None:
        if chunk_ids:
            self.collection.update(
                ids=chunk_ids,
                metadatas=[normalize_metadata(metadata) for _ in chunk_ids],
            )

    def update_chunks_metadata(
        self,
        chunks: list[PaperChunk],
        shared_metadata: dict[str, Any],
    ) -> None:
        if chunks:
            self.collection.update(
                ids=[chunk.id for chunk in chunks],
                metadatas=[
                    normalize_metadata({**chunk.metadata, **shared_metadata}) for chunk in chunks
                ],
            )

    def delete_source(self, source_path: str) -> None:
        self.collection.delete(where={"source_path": source_path})

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        result = self.collection.get(ids=[chunk_id], include=["documents", "metadatas"])
        if not result["ids"]:
            return None
        return {
            "id": result["ids"][0],
            "text": result["documents"][0],
            "metadata": result["metadatas"][0],
        }

    def list_chunks(
        self,
        where: dict[str, Any] | None = None,
        limit: int | None = None,
        offset: int | None = None,
    ) -> dict[str, Any]:
        return self.collection.get(where=where, limit=limit, offset=offset)

    def query_vector(self, query: str, top_k: int) -> dict[str, Any]:
        vector = self.embeddings.embed_query(query)
        return self.collection.query(
            query_embeddings=[vector],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
