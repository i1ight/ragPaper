from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PaperChunk:
    id: str
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True)
class PaperDocument:
    path: Path
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SearchResult:
    chunk_id: str
    text: str
    metadata: dict[str, Any]
    score: float
    vector_score: float | None = None
    bm25_score: float | None = None
    # Populated only when results are grouped by paper (group_by_paper):
    chunk_count: int | None = None
    other_chunk_ids: list[str] | None = None
    # Populated only when a reranker re-scored this candidate:
    rerank_score: float | None = None

