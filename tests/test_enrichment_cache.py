from __future__ import annotations

from rag_paper.enrichment_cache import MetadataEnrichmentCache


def test_metadata_enrichment_cache_round_trip(tmp_path) -> None:
    cache = MetadataEnrichmentCache(tmp_path / "cache.sqlite3")
    try:
        payload = {
            "source": "crossref",
            "doi": "10.1000/test",
            "title": "Cached Paper",
        }
        cache.set("crossref", "doi", "10.1000/test", payload)

        assert cache.get("crossref", "doi", "10.1000/test") == payload
        assert cache.get("openalex", "doi", "10.1000/test") is None
    finally:
        cache.close()
