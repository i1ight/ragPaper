from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_paper_metadata(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Metadata file must be a JSON object: {path}")
    return {str(key): dict(value) for key, value in payload.items()}


def save_paper_metadata(path: Path, metadata: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
        f.write("\n")


def metadata_for_pdf(all_metadata: dict[str, dict[str, Any]], pdf_path: Path) -> dict[str, Any]:
    resolved = str(pdf_path.resolve())
    entry = all_metadata.get(resolved)
    if entry is not None:
        return dict(entry)
    # A filename-only key is ambiguous when two PDFs in different directories
    # share a name. Only trust it when the entry has no provenance (legacy /
    # user-authored) or its source_path matches this exact file.
    name_entry = all_metadata.get(pdf_path.name)
    if name_entry is not None and _name_key_belongs_to_file(name_entry, resolved):
        return dict(name_entry)
    raw_entry = all_metadata.get(str(pdf_path))
    return dict(raw_entry) if raw_entry is not None else {}


def metadata_key_for_pdf(all_metadata: dict[str, dict[str, Any]], pdf_path: Path) -> str:
    resolved = str(pdf_path.resolve())
    if resolved in all_metadata:
        return resolved
    existing = all_metadata.get(pdf_path.name)
    if existing is not None and _name_key_belongs_to_file(existing, resolved):
        return pdf_path.name
    # Fall back to a resolved-path key for brand-new entries so that two same-named
    # PDFs never collide on a shared filename key.
    return resolved


def _name_key_belongs_to_file(entry: dict[str, Any], resolved_path: str) -> bool:
    source = entry.get("source_path")
    if not isinstance(source, str) or not source:
        # Legacy or user-authored entry with no provenance: assume the filename
        # match is intentional so existing paper_metadata.json files keep working.
        return True
    return source == resolved_path
