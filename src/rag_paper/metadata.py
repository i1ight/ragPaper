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
    return dict(all_metadata.get(pdf_path.name) or all_metadata.get(str(pdf_path)) or {})


def metadata_key_for_pdf(all_metadata: dict[str, dict[str, Any]], pdf_path: Path) -> str:
    resolved = str(pdf_path.resolve())
    if resolved in all_metadata:
        return resolved
    if pdf_path.name in all_metadata:
        return pdf_path.name
    return resolved
