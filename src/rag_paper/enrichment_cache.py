from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

from rag_paper.config import MetadataProviderName


class MetadataEnrichmentCache:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(self.path)
        self.connection.execute(
            """
            CREATE TABLE IF NOT EXISTS metadata_cache (
                provider TEXT NOT NULL,
                query_type TEXT NOT NULL,
                query_value TEXT NOT NULL,
                payload TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (provider, query_type, query_value)
            )
            """
        )
        self.connection.commit()

    def close(self) -> None:
        self.connection.close()

    def get(
        self,
        provider: MetadataProviderName,
        query_type: str,
        query_value: str,
    ) -> dict[str, Any] | None:
        row = self.connection.execute(
            """
            SELECT payload FROM metadata_cache
            WHERE provider = ? AND query_type = ? AND query_value = ?
            """,
            (provider, query_type, query_value),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])

    def set(
        self,
        provider: MetadataProviderName,
        query_type: str,
        query_value: str,
        payload: dict[str, Any],
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO metadata_cache (provider, query_type, query_value, payload)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(provider, query_type, query_value)
            DO UPDATE SET payload = excluded.payload, updated_at = CURRENT_TIMESTAMP
            """,
            (provider, query_type, query_value, json.dumps(payload, ensure_ascii=False)),
        )
        self.connection.commit()
