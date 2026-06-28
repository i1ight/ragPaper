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

