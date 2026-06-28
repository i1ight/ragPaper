from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def file_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def file_stat_signature(path: Path) -> dict[str, int]:
    stat = path.stat()
    return {"size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class IndexManifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.payload: dict[str, Any] = {"files": {}}
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                self.payload = json.load(f)
            self.payload.setdefault("files", {})

    def get(self, pdf_path: Path) -> dict[str, Any] | None:
        return self.payload["files"].get(str(pdf_path.resolve()))

    def has_same_stat(self, pdf_path: Path) -> bool:
        item = self.get(pdf_path)
        if not item:
            return False
        signature = file_stat_signature(pdf_path)
        return item.get("size") == signature["size"] and item.get("mtime_ns") == signature["mtime_ns"]

    def is_current(self, pdf_path: Path, digest: str) -> bool:
        item = self.get(pdf_path)
        return bool(item and item.get("sha256") == digest)

    def update(self, pdf_path: Path, digest: str, chunk_ids: list[str]) -> None:
        signature = file_stat_signature(pdf_path)
        self.payload["files"][str(pdf_path.resolve())] = {
            "sha256": digest,
            "size": signature["size"],
            "mtime_ns": signature["mtime_ns"],
            "chunk_ids": chunk_ids,
            "chunk_count": len(chunk_ids),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
        }

    def remove(self, pdf_path: Path) -> list[str]:
        item = self.payload["files"].pop(str(pdf_path.resolve()), None)
        if not item:
            return []
        return list(item.get("chunk_ids") or [])

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as f:
            json.dump(self.payload, f, indent=2, ensure_ascii=False)
            f.write("\n")
