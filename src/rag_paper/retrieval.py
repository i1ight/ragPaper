from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

from rag_paper.config import AppConfig
from rag_paper.embeddings import build_embedding_provider
from rag_paper.logging import logger
from rag_paper.models import SearchResult
from rag_paper.reranker import RerankerProvider, build_reranker_provider
from rag_paper.store import ChromaPaperStore


# CJK characters are matched one-at-a-time so a Chinese run yields per-character
# tokens; otherwise the whole run collapses into a single giant token and BM25
# can never match a Chinese query. Latin/alphanumeric runs match as whole words.
TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[^\W\u4e00-\u9fff]+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


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


class BM25Corpus:
    """A reusable BM25 index over the full chunk corpus.

    Tokenization and the BM25Okapi structure are built once; per-query filters
    are applied after scoring, so a single index serves every query regardless
    of author/year/tag/file_name filters. IDF is computed over the full corpus,
    which is more stable than re-scoring a filtered subset on every call.
    """

    def __init__(
        self, ids: list[str], documents: list[str], metadatas: list[dict[str, Any]]
    ) -> None:
        self.ids = ids
        self.documents = documents
        self.metadatas = metadatas
        self._tokenized = [tokenize(document) for document in documents]
        self.bm25: BM25Okapi | None = BM25Okapi(self._tokenized) if self._tokenized else None

    def score(
        self,
        query: str,
        *,
        top_k: int,
        author: str | None = None,
        year: int | None = None,
        tag: str | None = None,
        file_name: str | None = None,
    ) -> dict[str, SearchResult]:
        if self.bm25 is None:
            return {}
        raw_scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(
            (
                (index, float(score))
                for index, score in enumerate(raw_scores)
                if metadata_matches(
                    self.metadatas[index],
                    author=author,
                    year=year,
                    tag=tag,
                    file_name=file_name,
                )
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        ranked = ranked[:top_k]
        max_score = max((score for _, score in ranked), default=0.0) or 1.0

        results: dict[str, SearchResult] = {}
        for index, score in ranked:
            normalized = score / max_score
            results[self.ids[index]] = SearchResult(
                chunk_id=self.ids[index],
                text=self.documents[index],
                metadata=self.metadatas[index],
                score=normalized,
                bm25_score=normalized,
            )
        return results


class HybridRetriever:
    def __init__(self, config: AppConfig) -> None:
        embeddings = build_embedding_provider(config)
        self.store = ChromaPaperStore(config.chroma_dir, config.chroma.collection, embeddings)
        self.config = config
        self.stats = RetrievalStats(config.stats_path)
        self._bm25_corpus: BM25Corpus | None = None
        self._bm25_corpus_count: int | None = None
        self._reranker: RerankerProvider | None = build_reranker_provider(config)

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        author: str | None = None,
        year: int | None = None,
        tag: str | None = None,
        file_name: str | None = None,
        group_by_paper: bool = False,
    ) -> list[SearchResult]:
        top_k = top_k or self.config.retrieval.default_top_k
        where = chroma_metadata_filter(year=year, file_name=file_name)
        vector_results = self._vector_search(query, where, author=author, tag=tag, year=year, file_name=file_name)
        bm25_results = self._bm25_search(query, author=author, tag=tag, year=year, file_name=file_name)
        merged = self._merge(vector_results, bm25_results)
        ranked = sorted(merged.values(), key=lambda item: item.score, reverse=True)
        if self._reranker is not None:
            ranked = self._rerank(query, ranked)
        if group_by_paper:
            # Consider the full candidate pool (not just top_k chunks) so highly
            # represented papers don't crowd out breadth.
            results = _group_by_paper(ranked, top_k=top_k)
        else:
            results = ranked[:top_k]
        self.stats.write(
            {
                "event": "search",
                "query": query,
                "top_k": top_k,
                "group_by_paper": group_by_paper,
                "reranked": self._reranker is not None,
                "fusion": self.config.retrieval.fusion,
                "filters": {"author": author, "year": year, "tag": tag, "file_name": file_name},
                "vector_hits": len(vector_results),
                "bm25_hits": len(bm25_results),
                "returned": len(results),
            }
        )
        logger.info("retrieval.search", query=query, returned=len(results), filters=where)
        return results

    def _rerank(self, query: str, ranked: list[SearchResult]) -> list[SearchResult]:
        budget = self.config.reranker.top_k
        candidates = ranked[:budget]
        try:
            scores = self._reranker.score_pairs(  # type: ignore[union-attr]
                [(query, candidate.text) for candidate in candidates]
            )
        except Exception as exc:  # noqa: BLE001 - reranker unavailable: fall back to fused order
            logger.warning("reranker.failed", error=str(exc))
            return ranked
        reranked = [
            replace(candidate, rerank_score=score)
            for candidate, score in zip(candidates, scores)
        ]
        # Relevant (rerank_score=1) first; within a group, keep fused-score order.
        reranked.sort(key=lambda r: (r.rerank_score or 0.0, r.score), reverse=True)
        return reranked + ranked[budget:]

    def search_by_metadata(
        self,
        *,
        author: str | None = None,
        year: int | None = None,
        tag: str | None = None,
        file_name: str | None = None,
        limit: int = 20,
        group_by_paper: bool = False,
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
        if group_by_paper:
            # No relevance ranking here, so represent each paper by its earliest
            # chunk (lowest chunk_index; abstract chunks sort last) before folding
            # to one chunk per paper.
            results = sorted(results, key=lambda r: _chunk_index_key(r.metadata))
            results = _group_by_paper(results, top_k=limit)
        else:
            results = results[:limit]
        self.stats.write(
            {
                "event": "search_by_metadata",
                "filters": {"author": author, "year": year, "tag": tag, "file_name": file_name},
                "group_by_paper": group_by_paper,
                "returned": len(results),
            }
        )
        return results

    def get_chunk(self, chunk_id: str) -> dict[str, Any] | None:
        item = self.store.get_chunk(chunk_id)
        if item is None:
            return None
        item["metadata"] = _without_source_path(item["metadata"])
        return item

    def export_context(self, chunk_ids: list[str]) -> str:
        sections: list[str] = []
        for chunk_id in chunk_ids:
            item = self.get_chunk(chunk_id)
            if not item:
                continue
            metadata = item["metadata"]
            title = metadata.get("title") or metadata.get("file_name") or chunk_id
            header = [f"## {title}", f"- chunk_id: {chunk_id}"]
            doi = metadata.get("doi")
            if doi:
                header.append(f"- doi: {doi}")
            sections.append("\n".join([*header, "", item["text"]]))
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

    def _get_bm25_corpus(self) -> BM25Corpus:
        # Reuse one BM25 index across queries and rebuild only when the chunk
        # count changes (papers added/removed). count() is a cheap Chroma call,
        # so steady-state queries skip the O(N) tokenize + index build entirely.
        count = self.store.count()
        if self._bm25_corpus is None or self._bm25_corpus_count != count:
            payload = self.store.list_chunks()
            self._bm25_corpus = BM25Corpus(
                ids=list(payload.get("ids", [])),
                documents=list(payload.get("documents", [])),
                metadatas=list(payload.get("metadatas", [])),
            )
            self._bm25_corpus_count = count
        return self._bm25_corpus

    def _bm25_search(
        self,
        query: str,
        *,
        author: str | None,
        tag: str | None,
        year: int | None,
        file_name: str | None,
    ) -> dict[str, SearchResult]:
        corpus = self._get_bm25_corpus()
        return corpus.score(
            query,
            top_k=self.config.retrieval.bm25_top_k,
            author=author,
            year=year,
            tag=tag,
            file_name=file_name,
        )

    def _merge(
        self,
        vector_results: dict[str, SearchResult],
        bm25_results: dict[str, SearchResult],
    ) -> dict[str, SearchResult]:
        if self.config.retrieval.fusion == "rrf":
            fused = self._rrf_scores(vector_results, bm25_results)
        else:
            fused = self._linear_scores(vector_results, bm25_results)

        merged: dict[str, SearchResult] = {}
        chunk_ids = list(vector_results)
        chunk_ids += [cid for cid in bm25_results if cid not in vector_results]
        for chunk_id in chunk_ids:
            vector_hit = vector_results.get(chunk_id)
            bm25_hit = bm25_results.get(chunk_id)
            base = vector_hit or bm25_hit
            merged[chunk_id] = SearchResult(
                chunk_id=chunk_id,
                text=base.text,
                metadata=base.metadata,
                score=fused[chunk_id],
                vector_score=(vector_hit.vector_score or 0.0) if vector_hit else 0.0,
                bm25_score=(bm25_hit.bm25_score or 0.0) if bm25_hit else 0.0,
            )
        return merged

    def _rrf_scores(
        self,
        vector_results: dict[str, SearchResult],
        bm25_results: dict[str, SearchResult],
    ) -> dict[str, float]:
        # Reciprocal Rank Fusion: combine ranks, not scores, so the result is
        # immune to the very different distributions of cosine similarity and BM25.
        # Each retriever's dict is already in rank order by insertion.
        k = self.config.retrieval.rrf_k
        scores: dict[str, float] = defaultdict(float)
        for rank, chunk_id in enumerate(vector_results.keys(), start=1):
            scores[chunk_id] += 1.0 / (k + rank)
        for rank, chunk_id in enumerate(bm25_results.keys(), start=1):
            scores[chunk_id] += 1.0 / (k + rank)
        return scores

    def _linear_scores(
        self,
        vector_results: dict[str, SearchResult],
        bm25_results: dict[str, SearchResult],
    ) -> dict[str, float]:
        vector_weight = self.config.retrieval.vector_weight
        bm25_weight = self.config.retrieval.bm25_weight
        scores: dict[str, float] = {}
        chunk_ids = list(vector_results)
        chunk_ids += [cid for cid in bm25_results if cid not in vector_results]
        for chunk_id in chunk_ids:
            vector_hit = vector_results.get(chunk_id)
            bm25_hit = bm25_results.get(chunk_id)
            vector_score = (vector_hit.vector_score or 0.0) if vector_hit else 0.0
            bm25_score = (bm25_hit.bm25_score or 0.0) if bm25_hit else 0.0
            scores[chunk_id] = vector_weight * vector_score + bm25_weight * bm25_score
        return scores


def _chunk_index_key(metadata: dict[str, Any]) -> int:
    """Sort key for picking a paper's representative chunk: lowest chunk_index.

    Abstract chunks (and any chunk without chunk_index) sort last so a body chunk
    represents the paper.
    """
    index = metadata.get("chunk_index")
    return index if isinstance(index, int) else 10**9


def _group_by_paper(ranked: list[SearchResult], *, top_k: int) -> list[SearchResult]:
    """Collapse ranked chunks to the best chunk per paper.

    Keeps the highest-scoring chunk for each distinct ``source_path`` and records
    how many of the ranked chunks belonged to that paper plus their ids, so a
    caller (typically an LLM) sees coverage breadth instead of one paper's chunks
    crowding the whole result set. Returns up to ``top_k`` papers in score order.
    """
    best: dict[str, SearchResult] = {}
    counts: dict[str, int] = {}
    other_ids: dict[str, list[str]] = {}
    order: list[str] = []
    for result in ranked:
        key = str(result.metadata.get("source_path") or result.chunk_id)
        counts[key] = counts.get(key, 0) + 1
        if key not in best:
            best[key] = result
            other_ids[key] = []
            order.append(key)
        else:
            other_ids[key].append(result.chunk_id)

    grouped: list[SearchResult] = []
    for key in order[:top_k]:
        grouped.append(
            replace(best[key], chunk_count=counts[key], other_chunk_ids=other_ids[key])
        )
    return grouped


def _without_source_path(metadata: Any) -> Any:
    """Drop ``source_path`` from metadata before returning to MCP clients.

    The filesystem path is an implementation detail (grouping, deletion) and is
    not useful to an LLM consuming the Zotero-assist tools, so it is stripped at
    the response boundary.
    """
    if not isinstance(metadata, dict):
        return metadata
    return {key: value for key, value in metadata.items() if key != "source_path"}


def result_to_dict(
    result: SearchResult, *, max_text_chars: int | None = None
) -> dict[str, Any]:
    text = result.text
    if max_text_chars is not None and len(text) > max_text_chars:
        text = text[:max_text_chars].rstrip() + " …"
    payload: dict[str, Any] = {
        "chunk_id": result.chunk_id,
        "score": result.score,
        "vector_score": result.vector_score,
        "bm25_score": result.bm25_score,
        "metadata": _without_source_path(result.metadata),
        "text": text,
    }
    if result.chunk_count is not None:
        payload["chunk_count"] = result.chunk_count
    if result.other_chunk_ids is not None:
        payload["other_chunk_ids"] = result.other_chunk_ids
    if result.rerank_score is not None:
        payload["rerank_score"] = result.rerank_score
    return payload
