from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_paper.config import AppConfig
from rag_paper.embeddings import EmbeddingProvider, build_embedding_provider
from rag_paper.indexer import PdfCandidate, discover_configured_pdfs
from rag_paper.logging import logger
from rag_paper.metadata import load_paper_metadata, metadata_for_pdf
from rag_paper.pdf import extract_pdf_text


@dataclass(frozen=True)
class DuplicatePair:
    source_path: str
    duplicate_path: str
    similarity: float
    reason: str


@dataclass(frozen=True)
class DedupSummary:
    checked_files: int
    duplicate_pairs: int
    skipped_files: int
    report_path: str


def run_dedup_report(
    config: AppConfig,
    *,
    file_path: str | None = None,
    max_files: int | None = None,
) -> DedupSummary:
    metadata_map = load_paper_metadata(config.metadata_path)
    candidates = _dedup_candidates(config, file_path=file_path)
    if max_files is not None:
        candidates = candidates[:max_files]
    elif config.dedup.max_files is not None:
        candidates = candidates[: config.dedup.max_files]

    embeddings = build_embedding_provider(config) if config.dedup.semantic_enabled else None
    pairs = find_duplicate_pairs(
        config,
        candidates,
        metadata_map=metadata_map,
        embeddings=embeddings,
    )
    write_dedup_report(config.dedup_report_path, candidates, pairs)
    return DedupSummary(
        checked_files=len(candidates),
        duplicate_pairs=len(pairs),
        skipped_files=len(duplicate_paths(pairs)),
        report_path=str(config.dedup_report_path),
    )


def filter_duplicate_candidates(
    config: AppConfig,
    candidates: list[PdfCandidate],
    *,
    metadata_map: dict[str, dict[str, Any]],
    embeddings: EmbeddingProvider | None = None,
) -> tuple[list[PdfCandidate], list[DuplicatePair]]:
    if not config.dedup.enabled:
        return candidates, []

    effective_candidates = candidates
    if config.dedup.max_files is not None:
        effective_candidates = candidates[: config.dedup.max_files]
    pairs = find_duplicate_pairs(
        config,
        effective_candidates,
        metadata_map=metadata_map,
        embeddings=embeddings,
    )
    write_dedup_report(config.dedup_report_path, effective_candidates, pairs)
    if config.dedup.action != "skip":
        return candidates, pairs

    skipped = duplicate_paths(pairs)
    filtered = [candidate for candidate in candidates if str(candidate.path) not in skipped]
    for path in sorted(skipped):
        logger.info("dedup.skip_duplicate", file=path)
    return filtered, pairs


def find_duplicate_pairs(
    config: AppConfig,
    candidates: list[PdfCandidate],
    *,
    metadata_map: dict[str, dict[str, Any]],
    embeddings: EmbeddingProvider | None,
) -> list[DuplicatePair]:
    pairs = _metadata_duplicate_pairs(candidates, metadata_map)
    if config.dedup.semantic_enabled and embeddings is not None:
        pairs.extend(_semantic_duplicate_pairs(config, candidates, metadata_map, embeddings))
    return _unique_pairs(pairs)


def duplicate_paths(pairs: list[DuplicatePair]) -> set[str]:
    return {pair.duplicate_path for pair in pairs}


def write_dedup_report(
    path: Path,
    candidates: list[PdfCandidate],
    pairs: list[DuplicatePair],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checked_files": len(candidates),
        "duplicate_pairs": [
            {
                "source_path": pair.source_path,
                "duplicate_path": pair.duplicate_path,
                "similarity": pair.similarity,
                "reason": pair.reason,
            }
            for pair in pairs
        ],
        "skipped_files": sorted(duplicate_paths(pairs)),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")


def semantic_signature_text(
    config: AppConfig,
    pdf_path: Path,
    metadata: dict[str, Any],
) -> str:
    parts: list[str] = []
    for key in ("doi", "title", "abstract"):
        value = metadata.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    if not any(part for part in parts if len(part) > 40):
        document = extract_pdf_text(pdf_path, extra_metadata=metadata)
        parts.append(document.text[: config.dedup.signature_chars])

    return "\n".join(parts)[: config.dedup.signature_chars]


def _dedup_candidates(config: AppConfig, *, file_path: str | None) -> list[PdfCandidate]:
    if file_path:
        pdf_path = Path(file_path).expanduser().resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Only .pdf files can be deduplicated: {pdf_path}")
        candidates = discover_configured_pdfs(config)
        if str(pdf_path) not in {str(candidate.path) for candidate in candidates}:
            candidates.append(PdfCandidate(path=pdf_path, root_path=pdf_path.parent, tags=()))
        return candidates
    return discover_configured_pdfs(config)


def _metadata_duplicate_pairs(
    candidates: list[PdfCandidate],
    metadata_map: dict[str, dict[str, Any]],
) -> list[DuplicatePair]:
    pairs: list[DuplicatePair] = []
    seen_doi: dict[str, str] = {}
    seen_title_year: dict[tuple[str, Any], str] = {}

    for candidate in candidates:
        metadata = metadata_for_pdf(metadata_map, candidate.path)
        path = str(candidate.path)
        doi = _normalized(metadata.get("doi"))
        if doi:
            if doi in seen_doi:
                pairs.append(DuplicatePair(seen_doi[doi], path, 1.0, "doi"))
            else:
                seen_doi[doi] = path
            continue

        title = _normalized(metadata.get("title"))
        year = metadata.get("year")
        if title and year:
            key = (title, year)
            if key in seen_title_year:
                pairs.append(DuplicatePair(seen_title_year[key], path, 1.0, "title_year"))
            else:
                seen_title_year[key] = path

    return pairs


def _semantic_duplicate_pairs(
    config: AppConfig,
    candidates: list[PdfCandidate],
    metadata_map: dict[str, dict[str, Any]],
    embeddings: EmbeddingProvider,
) -> list[DuplicatePair]:
    signatures = [
        semantic_signature_text(config, candidate.path, metadata_for_pdf(metadata_map, candidate.path))
        for candidate in candidates
    ]
    vectors = embeddings.embed_texts(signatures) if signatures else []
    pairs: list[DuplicatePair] = []

    for right_index, right_vector in enumerate(vectors):
        for left_index in range(right_index):
            similarity = cosine_similarity(vectors[left_index], right_vector)
            if similarity >= config.dedup.similarity_threshold:
                pairs.append(
                    DuplicatePair(
                        source_path=str(candidates[left_index].path),
                        duplicate_path=str(candidates[right_index].path),
                        similarity=similarity,
                        reason="semantic",
                    )
                )
                break
    return pairs


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=False))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0
    return dot / (left_norm * right_norm)


def _unique_pairs(pairs: list[DuplicatePair]) -> list[DuplicatePair]:
    seen: set[tuple[str, str]] = set()
    unique: list[DuplicatePair] = []
    for pair in pairs:
        key = (pair.source_path, pair.duplicate_path)
        if key in seen:
            continue
        seen.add(key)
        unique.append(pair)
    return unique


def _normalized(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.lower().strip().split())
