from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.api.models.Collection import Collection

from rag_paper.config import AppConfig
from rag_paper.manifest import IndexManifest
from rag_paper.title_quality import best_title


@dataclass
class IndexedPaperSummary:
    source_path: str
    file_name: str
    title: str = ""
    authors: str = ""
    year: int | str | None = None
    doi: str = ""
    chunk_count: int = 0


@dataclass
class IndexedPaperDetail(IndexedPaperSummary):
    metadata: dict[str, Any] = field(default_factory=dict)
    chunk_ids: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class IndexedLibrarySummary:
    paper_count: int
    chunk_count: int
    shown_count: int
    papers: list[IndexedPaperSummary]


@dataclass(frozen=True)
class DeleteIndexedPapersSummary:
    matched_papers: int
    deleted_papers: int
    deleted_chunks: int
    papers: list[IndexedPaperSummary]


class ChromaInspector:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.client = chromadb.PersistentClient(path=str(config.chroma_dir))
        self.collection: Collection = self.client.get_or_create_collection(
            name=config.chroma.collection,
            metadata={"hnsw:space": "cosine"},
        )

    def summarize(self, *, limit: int | None = None) -> IndexedLibrarySummary:
        payload = self.collection.get(include=["metadatas"])
        papers = _summaries_from_payload(payload)
        paper_count = len(papers)
        papers = sorted(
            papers,
            key=lambda item: (str(item.year or ""), item.title or item.file_name),
            reverse=True,
        )
        if limit is not None:
            papers = papers[:limit]
        return IndexedLibrarySummary(
            paper_count=paper_count,
            chunk_count=self.collection.count(),
            shown_count=len(papers),
            papers=papers,
        )

    def details(self, selector: str, *, limit: int = 10) -> list[IndexedPaperDetail]:
        selector = selector.strip()
        if not selector:
            return []

        payload = self.collection.get(include=["metadatas"])
        groups = _group_chunks(payload)
        details: list[IndexedPaperDetail] = []
        for source_path, items in groups.items():
            metadata = _merged_metadata([item["metadata"] for item in items])
            file_name = str(metadata.get("file_name") or Path(source_path).name)
            title = str(metadata.get("title") or "")
            doi = str(metadata.get("doi") or "")
            if not _matches_selector(selector, source_path, file_name, title, doi):
                continue
            summary = _summary_from_group(source_path, items)
            details.append(
                IndexedPaperDetail(
                    **summary.__dict__,
                    metadata=metadata,
                    chunk_ids=[item["id"] for item in items],
                )
            )
        return sorted(
            details,
            key=lambda item: (str(item.year or ""), item.title or item.file_name),
            reverse=True,
        )[:limit]

    def has_indexed_source(self, source_path: Path) -> bool:
        payload = self.collection.get(
            where={"source_path": str(source_path.expanduser().resolve())},
            limit=1,
            include=["metadatas"],
        )
        return bool(payload.get("ids"))

    def delete(self, selector: str, *, limit: int = 10) -> DeleteIndexedPapersSummary:
        return self.delete_details(self.details(selector, limit=limit))

    def delete_details(self, details: list[IndexedPaperDetail]) -> DeleteIndexedPapersSummary:
        manifest = IndexManifest(self.config.chroma_dir / "index_manifest.json")
        deleted_chunks = 0
        deleted_papers = 0
        papers: list[IndexedPaperSummary] = []

        for detail in details:
            if detail.chunk_ids:
                self.collection.delete(ids=detail.chunk_ids)
                deleted_chunks += len(detail.chunk_ids)
            manifest.remove(Path(detail.source_path))
            deleted_papers += 1
            papers.append(
                IndexedPaperSummary(
                    source_path=detail.source_path,
                    file_name=detail.file_name,
                    title=detail.title,
                    authors=detail.authors,
                    year=detail.year,
                    doi=detail.doi,
                    chunk_count=detail.chunk_count,
                )
            )

        if deleted_papers:
            manifest.save()

        return DeleteIndexedPapersSummary(
            matched_papers=len(details),
            deleted_papers=deleted_papers,
            deleted_chunks=deleted_chunks,
            papers=papers,
        )


def inspect_indexed_papers(config: AppConfig, *, limit: int | None = None) -> IndexedLibrarySummary:
    return ChromaInspector(config).summarize(limit=limit)


def inspect_indexed_paper(
    config: AppConfig,
    selector: str,
    *,
    limit: int = 10,
) -> list[IndexedPaperDetail]:
    return ChromaInspector(config).details(selector, limit=limit)


def is_pdf_indexed(config: AppConfig, pdf_path: str | Path) -> bool:
    return ChromaInspector(config).has_indexed_source(Path(pdf_path))


def delete_indexed_papers(
    config: AppConfig,
    selector: str,
    *,
    limit: int = 10,
) -> DeleteIndexedPapersSummary:
    return ChromaInspector(config).delete(selector, limit=limit)


def delete_indexed_paper_details(
    config: AppConfig,
    details: list[IndexedPaperDetail],
) -> DeleteIndexedPapersSummary:
    return ChromaInspector(config).delete_details(details)


def _summaries_from_payload(payload: dict[str, Any]) -> list[IndexedPaperSummary]:
    return [
        _summary_from_group(source_path, items)
        for source_path, items in _group_chunks(payload).items()
    ]


def _group_chunks(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    ids = payload.get("ids", [])
    metadatas = payload.get("metadatas", [])
    for chunk_id, metadata in zip(ids, metadatas, strict=False):
        if not isinstance(metadata, dict):
            continue
        source_path = str(metadata.get("source_path") or "")
        if not source_path:
            source_path = str(metadata.get("file_name") or chunk_id)
        groups[source_path].append({"id": str(chunk_id), "metadata": metadata})
    return groups


def _summary_from_group(source_path: str, items: list[dict[str, Any]]) -> IndexedPaperSummary:
    metadata = _merged_metadata([item["metadata"] for item in items])
    file_name = str(metadata.get("file_name") or Path(source_path).name)
    return IndexedPaperSummary(
        source_path=source_path,
        file_name=file_name,
        title=best_title(metadata.get("title"), file_name=file_name),
        authors=str(metadata.get("authors") or ""),
        year=metadata.get("year"),
        doi=str(metadata.get("doi") or ""),
        chunk_count=len(items),
    )


def _merged_metadata(items: list[dict[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for metadata in items:
        for key, value in metadata.items():
            if value not in ("", None, []):
                merged.setdefault(key, value)
    return merged


def _matches_selector(
    selector: str,
    source_path: str,
    file_name: str,
    title: str,
    doi: str,
) -> bool:
    needle = selector.lower()
    return (
        needle in source_path.lower()
        or needle in file_name.lower()
        or needle in title.lower()
        or needle in doi.lower()
    )
