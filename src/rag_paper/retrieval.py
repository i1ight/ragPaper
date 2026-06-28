from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from rag_paper.config import AppConfig
from rag_paper.embeddings import build_embedding_provider
from rag_paper.logging import logger
from rag_paper.models import SearchResult
from rag_paper.store import ChromaPaperStore


TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return [item.lower() for item in TOKEN_RE.findall(text)]


def chroma_metadata_filter(
    *,
    year: int | None = None,
    file_name: str | None = None,
) -> dict[str, Any] | None:
    filters: list[dict[str, Any]] = []
    if year:
        filters.append({"year": year})
    if file_name:
        filters.append({"file_name": file_name})
    if not filters:
        return None
    if len(filters) == 1:
        return filters[0]
    return {"$and": filters}


def metadata_matches(
    metadata: dict[str, Any],
    *,
    author: str | None = None,
    year: int | None = None,
    tag: str | None = None,
    file_name: str | None = None,
) -> bool:
    if year is not None and metadata.get("year") != year:
        return False
    if file_name is not None and metadata.get("file_name") != file_name:
        return False
    if author and author.lower() not in str(metadata.get("authors", "")).lower():
        return False
    if tag and tag.lower() not in str(metadata.get("tags", "")).lower():
        return False
    return True


class RetrievalStats:
    def __init__(self, path: Path) -> None:
        self.path = path

    def write(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            **payload,
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


class HybridRetriever:
    def __init__(self, config: AppConfig) -> None:
        embeddings = build_embedding_provider(config)
        self.store = ChromaPaperStore(config.chroma_dir, config.chroma.collection, embeddings)
        self.config = config
        self.stats = RetrievalStats(config.stats_path)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        author: str | None = None,
        year: int | None = None,
        tag: str | None = None,
        file_name: str | None = None,
    ) -> list[SearchResult]:
        top_k = top_k or self.config.retrieval.default_top_k
        where = chroma_metadata_filter(year=year, file_name=file_name)
        vector_results = self._vector_search(query, where, author=author, tag=tag, year=year, file_name=file_name)
        bm25_results = self._bm25_search(query, where, author=author, tag=tag, year=year, file_name=file_name)
        merged = self._merge(vector_results, bm25_results)
        results = sorted(merged.values(), key=lambda item: item.score, reverse=True)[:top_k]
        self.stats.write(
            {
                "event": "search",
                "query": query,
                "top_k": top_k,
                "filters": {"author": author, "year": year, "tag": tag, "file_name": file_name},
                "vector_hits": len(vector_results),
                "bm25_hits": len(bm25_results),
                "returned": len(results),
            }
        )
        logger.info("retrieval.search", query=query, returned=len(results), filters=where)
        return results

    def search_by_metadata(
        self,
        *,
        author: str | None = None,
        year: int | None = None,
        tag: str | None = None,
        file_name: str | None = None,
        limit: int = 20,
    ) -> list[SearchResult]:
        where = chroma_metadata_filter(year=year, file_name=file_name)
        payload = self.store.list_chunks(where=where)
        results = [
            SearchResult(
                chunk_id=chunk_id,
                text=document,
                metadata=metadata,
                score=1.0,
            )
            for chunk_id, document, metadata in zip(
                payload.get("ids", []),
                payload.get("documents", []),
                payload.get("metadatas", []),
            )
            if metadata_matches(
                metadata,
                author=author,
                year=year,
                tag=tag,
                file_name=file_name,
            )
        ]
        results = results[:limit]
        self.stats.write(
            {
                "event": "search_by_metadata",
                "filters": {"author": author, "year": year, "tag": tag, "file_name": file_name},
                "returned": len(results),
            }
        )
        return results

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        return self.store.get_chunk(chunk_id)

    def export_context(self, chunk_ids: list[str]) -> str:
        sections: list[str] = []
        for chunk_id in chunk_ids:
            item = self.get_chunk(chunk_id)
            if not item:
                continue
            metadata = item["metadata"]
            title = metadata.get("title") or metadata.get("file_name") or chunk_id
            sections.append(
                "\n".join(
                    [
                        f"## {title}",
                        f"- chunk_id: {chunk_id}",
                        f"- source: {metadata.get('source_path', '')}",
                        "",
                        item["text"],
                    ]
                )
            )
        self.stats.write({"event": "export_context", "chunk_ids": chunk_ids, "returned": len(sections)})
        return "\n\n---\n\n".join(sections)

    def _vector_search(
        self,
        query: str,
        where: dict[str, Any] | None,
        *,
        author: str | None,
        tag: str | None,
        year: int | None,
        file_name: str | None,
    ) -> dict[str, SearchResult]:
        raw = self.store.collection.query(
            query_embeddings=[self.store.embeddings.embed_query(query)],
            n_results=max(self.config.retrieval.vector_top_k, self.config.retrieval.default_top_k * 3),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = raw.get("ids", [[]])[0]
        documents = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        results: dict[str, SearchResult] = {}
        for chunk_id, document, metadata, distance in zip(ids, documents, metadatas, distances):
            if not metadata_matches(
                metadata,
                author=author,
                year=year,
                tag=tag,
                file_name=file_name,
            ):
                continue
            similarity = max(0.0, 1.0 - float(distance))
            results[chunk_id] = SearchResult(
                chunk_id=chunk_id,
                text=document,
                metadata=metadata,
                score=similarity,
                vector_score=similarity,
            )
        return dict(list(results.items())[: self.config.retrieval.vector_top_k])

    def _bm25_search(
        self,
        query: str,
        where: dict[str, Any] | None,
        *,
        author: str | None,
        tag: str | None,
        year: int | None,
        file_name: str | None,
    ) -> dict[str, SearchResult]:
        payload = self.store.list_chunks(where=where)
        filtered = [
            (chunk_id, document, metadata)
            for chunk_id, document, metadata in zip(
                payload.get("ids", []),
                payload.get("documents", []),
                payload.get("metadatas", []),
            )
            if metadata_matches(
                metadata,
                author=author,
                year=year,
                tag=tag,
                file_name=file_name,
            )
        ]
        ids = [item[0] for item in filtered]
        documents = [item[1] for item in filtered]
        metadatas = [item[2] for item in filtered]
        if not ids:
            return {}

        tokenized_docs = [tokenize(document) for document in documents]
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(tokenize(query))
        ranked = sorted(enumerate(scores), key=lambda item: item[1], reverse=True)
        ranked = ranked[: self.config.retrieval.bm25_top_k]
        max_score = max((float(score) for _, score in ranked), default=0.0) or 1.0

        results: dict[str, SearchResult] = {}
        for index, score in ranked:
            normalized_score = float(score) / max_score
            results[ids[index]] = SearchResult(
                chunk_id=ids[index],
                text=documents[index],
                metadata=metadatas[index],
                score=normalized_score,
                bm25_score=normalized_score,
            )
        return results

    def _merge(
        self,
        vector_results: dict[str, SearchResult],
        bm25_results: dict[str, SearchResult],
    ) -> dict[str, SearchResult]:
        merged_scores: dict[str, dict[str, float]] = defaultdict(dict)
        for chunk_id, result in vector_results.items():
            merged_scores[chunk_id]["vector"] = result.vector_score or 0.0
        for chunk_id, result in bm25_results.items():
            merged_scores[chunk_id]["bm25"] = result.bm25_score or 0.0

        merged: dict[str, SearchResult] = {}
        for chunk_id, scores in merged_scores.items():
            base = vector_results.get(chunk_id) or bm25_results[chunk_id]
            vector_score = scores.get("vector", 0.0)
            bm25_score = scores.get("bm25", 0.0)
            score = (
                self.config.retrieval.vector_weight * vector_score
                + self.config.retrieval.bm25_weight * bm25_score
            )
            merged[chunk_id] = SearchResult(
                chunk_id=chunk_id,
                text=base.text,
                metadata=base.metadata,
                score=score,
                vector_score=vector_score,
                bm25_score=bm25_score,
            )
        return merged


def result_to_dict(result: SearchResult) -> dict[str, Any]:
    return {
        "chunk_id": result.chunk_id,
        "score": result.score,
        "vector_score": result.vector_score,
        "bm25_score": result.bm25_score,
        "metadata": result.metadata,
        "text": result.text,
    }
