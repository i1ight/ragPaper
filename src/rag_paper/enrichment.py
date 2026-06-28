from __future__ import annotations

import re
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

from rag_paper.config import AppConfig, MetadataEnrichmentConfig, MetadataProviderName
from rag_paper.enrichment_cache import MetadataEnrichmentCache
from rag_paper.indexer import discover_configured_pdfs
from rag_paper.logging import logger
from rag_paper.metadata import (
    load_paper_metadata,
    metadata_for_pdf,
    metadata_key_for_pdf,
    save_paper_metadata,
)
from rag_paper.pdf import extract_pdf_text

DOI_RE = re.compile(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+", re.IGNORECASE)
TRAILING_DOI_CHARS = ".,;:)]}>\"'"


@dataclass(frozen=True)
class EnrichmentSummary:
    checked_files: int
    updated_files: int
    skipped_files: int
    failed_files: int


@dataclass(frozen=True)
class EnrichmentTarget:
    path: Path
    metadata: dict[str, Any]
    text: str = ""


@dataclass(frozen=True)
class MetadataWork:
    source: MetadataProviderName
    doi: str
    title: str
    authors: list[str]
    year: int | None
    container_title: str
    publisher: str
    url: str
    score: float | None
    openalex_id: str = ""
    abstract: str = ""
    referenced_work_ids: list[str] | None = None
    referenced_dois: list[str] | None = None
    related_work_ids: list[str] | None = None
    cited_by_count: int | None = None


CrossRefWork = MetadataWork


class RateLimiter:
    def __init__(self, requests_per_second: float) -> None:
        self.interval_seconds = 1.0 / requests_per_second
        self.last_request_at = 0.0

    def wait(self) -> None:
        now = time.monotonic()
        delay = self.interval_seconds - (now - self.last_request_at)
        if delay > 0:
            time.sleep(delay)
        self.last_request_at = time.monotonic()


class CrossRefClient:
    def __init__(self, config: MetadataEnrichmentConfig) -> None:
        self.config = config
        self.limiter = RateLimiter(config.requests_per_second)
        headers = {"User-Agent": config.user_agent}
        self.client = httpx.Client(
            base_url=config.base_url.rstrip("/"),
            headers=headers,
            timeout=config.timeout_seconds,
            proxy=self._proxy_url(),
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()

    def lookup_by_doi(self, doi: str) -> MetadataWork | None:
        self.limiter.wait()
        response = self.client.get(
            f"/works/{quote(doi, safe='')}",
            params=self._common_params(),
        )
        if response.status_code == 404:
            return None
        response.raise_for_status()
        message = response.json().get("message")
        if not isinstance(message, dict):
            return None
        return _work_from_crossref_message(message)

    def query_by_title(self, title: str) -> MetadataWork | None:
        title = title.strip()
        if not title:
            return None

        params: dict[str, object] = {
            "query.bibliographic": title[: self.config.max_query_chars],
            "rows": 1,
        }
        params.update(self._common_params())
        self.limiter.wait()
        response = self.client.get("/works", params=params)
        response.raise_for_status()
        message = response.json().get("message")
        if not isinstance(message, dict):
            return None
        items = message.get("items")
        if not isinstance(items, list) or not items:
            return None
        work = _work_from_crossref_message(items[0])
        if work is None:
            return None
        if work.score is not None and work.score < self.config.min_title_score:
            return None
        return work

    def _common_params(self) -> dict[str, str]:
        if self.config.mailto:
            return {"mailto": self.config.mailto}
        return {}

    def _proxy_url(self) -> str | None:
        if self.config.socks5_proxy:
            return self.config.socks5_proxy
        if self.config.https_proxy:
            return self.config.https_proxy
        if self.config.http_proxy:
            return self.config.http_proxy
        return None


class OpenAlexClient:
    def __init__(self, config: MetadataEnrichmentConfig) -> None:
        self.config = config
        self.limiter = RateLimiter(config.requests_per_second)
        headers = {"User-Agent": config.user_agent}
        self.client = httpx.Client(
            base_url=config.openalex_base_url.rstrip("/"),
            headers=headers,
            timeout=config.timeout_seconds,
            proxy=self._proxy_url(),
            follow_redirects=True,
        )

    def close(self) -> None:
        self.client.close()

    def lookup_by_doi(self, doi: str) -> MetadataWork | None:
        self.limiter.wait()
        response = self.client.get(
            "/works",
            params={
                **self._common_params(),
                "filter": f"doi:{_normalize_doi(doi)}",
                "per-page": 1,
            },
        )
        response.raise_for_status()
        return self._first_work(response.json())

    def query_by_title(self, title: str) -> MetadataWork | None:
        title = title.strip()
        if not title:
            return None
        self.limiter.wait()
        response = self.client.get(
            "/works",
            params={
                **self._common_params(),
                "search": title[: self.config.max_query_chars],
                "per-page": 1,
            },
        )
        response.raise_for_status()
        return self._first_work(response.json())

    def _first_work(self, payload: dict[str, Any]) -> MetadataWork | None:
        results = payload.get("results")
        if not isinstance(results, list) or not results:
            return None
        return _work_from_openalex_message(results[0])

    def _common_params(self) -> dict[str, str]:
        email = self.config.openalex_email or self.config.mailto
        if email:
            return {"mailto": email}
        return {}

    def _proxy_url(self) -> str | None:
        if self.config.socks5_proxy:
            return self.config.socks5_proxy
        if self.config.https_proxy:
            return self.config.https_proxy
        if self.config.http_proxy:
            return self.config.http_proxy
        return None


def enrich_metadata(
    config: AppConfig,
    *,
    force: bool = False,
    file_path: str | None = None,
    max_files: int | None = None,
) -> EnrichmentSummary:
    if not config.metadata_enrichment.enabled:
        logger.info("metadata.enrichment_disabled")
        return EnrichmentSummary(checked_files=0, updated_files=0, skipped_files=0, failed_files=0)

    metadata_map = load_paper_metadata(config.metadata_path)
    targets = discover_enrichment_targets(config, metadata_map, file_path=file_path)
    if max_files is not None:
        targets = targets[:max_files]

    summary = enrich_targets(
        config,
        targets,
        metadata_map=metadata_map,
        force=force,
    )
    save_paper_metadata(config.metadata_path, metadata_map)
    return summary


def enrich_targets(
    config: AppConfig,
    targets: list[EnrichmentTarget],
    *,
    metadata_map: dict[str, dict[str, Any]] | None = None,
    force: bool = False,
) -> EnrichmentSummary:
    if not config.metadata_enrichment.enabled:
        return EnrichmentSummary(checked_files=0, updated_files=0, skipped_files=0, failed_files=0)
    if metadata_map is None:
        metadata_map = load_paper_metadata(config.metadata_path)

    clients = _build_provider_clients(config.metadata_enrichment)
    cache = MetadataEnrichmentCache(config.metadata_cache_path)
    checked_files = 0
    updated_files = 0
    skipped_files = 0
    failed_files = 0

    try:
        try:
            for target in targets:
                checked_files += 1
                existing = metadata_for_pdf(metadata_map, target.path)
                if existing.get("doi") and not force:
                    skipped_files += 1
                    logger.info("metadata.skip_existing_doi", file=str(target.path), doi=existing["doi"])
                    continue

                merged = dict(existing)
                merged.update({key: value for key, value in target.metadata.items() if value})

                work, provider_failed = _lookup_work_with_fallback(
                    clients,
                    config.metadata_enrichment,
                    cache,
                    target,
                    merged,
                    force=force,
                )

                if work is None:
                    if provider_failed:
                        failed_files += 1
                        logger.warning("metadata.all_providers_failed", file=str(target.path))
                    else:
                        skipped_files += 1
                        logger.info("metadata.no_provider_match", file=str(target.path))
                    continue

                key = metadata_key_for_pdf(metadata_map, target.path)
                metadata_map[key] = _merge_work_metadata(merged, work, force=force)
                save_paper_metadata(config.metadata_path, metadata_map)
                updated_files += 1
                logger.info(
                    "metadata.enriched",
                    file=str(target.path),
                    doi=work.doi,
                    title=work.title,
                    source=work.source,
                )
        except KeyboardInterrupt:
            logger.warning("metadata.interrupted", updated_files=updated_files)
    finally:
        for client in clients.values():
            client.close()
        cache.close()

    save_paper_metadata(config.metadata_path, metadata_map)
    return EnrichmentSummary(
        checked_files=checked_files,
        updated_files=updated_files,
        skipped_files=skipped_files,
        failed_files=failed_files,
    )


def target_from_document(path: Path, metadata: dict[str, Any], text: str) -> EnrichmentTarget:
    return EnrichmentTarget(path=path.resolve(), metadata=dict(metadata), text=text)


def extract_doi(value: str) -> str | None:
    match = DOI_RE.search(value)
    if not match:
        return None
    return match.group(0).rstrip(TRAILING_DOI_CHARS)


def infer_title(metadata: dict[str, Any], text: str) -> str:
    title = metadata.get("title")
    if isinstance(title, str) and title.strip():
        return title.strip()

    lines = [
        line.strip()
        for line in text.splitlines()[:80]
        if 8 <= len(line.strip()) <= 220 and not line.strip().startswith("[Page ")
    ]
    if not lines:
        return ""
    return max(lines[:8], key=len)


def discover_enrichment_targets(
    config: AppConfig,
    metadata_map: dict[str, dict[str, Any]],
    *,
    file_path: str | None,
) -> list[EnrichmentTarget]:
    if file_path:
        pdf_path = Path(file_path).expanduser().resolve()
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        if not pdf_path.is_file() or pdf_path.suffix.lower() != ".pdf":
            raise ValueError(f"Only .pdf files can be enriched: {pdf_path}")
        candidates = [pdf_path]
    else:
        candidates = [candidate.path for candidate in discover_configured_pdfs(config)]

    targets: list[EnrichmentTarget] = []
    for pdf_path in candidates:
        document = extract_pdf_text(pdf_path, extra_metadata=metadata_for_pdf(metadata_map, pdf_path))
        targets.append(target_from_document(pdf_path, document.metadata, document.text))
    return targets


def _build_provider_clients(
    config: MetadataEnrichmentConfig,
) -> dict[MetadataProviderName, CrossRefClient | OpenAlexClient]:
    clients: dict[MetadataProviderName, CrossRefClient | OpenAlexClient] = {}
    for provider in _provider_order(config):
        if provider == "crossref":
            clients[provider] = CrossRefClient(config)
        elif provider == "openalex":
            clients[provider] = OpenAlexClient(config)
    return clients


def _provider_order(config: MetadataEnrichmentConfig) -> list[MetadataProviderName]:
    providers = config.providers or [config.provider]
    ordered: list[MetadataProviderName] = []
    for provider in providers:
        if provider not in ordered:
            ordered.append(provider)
    if not ordered:
        ordered.append(config.provider)
    return ordered


def _lookup_work_with_fallback(
    clients: dict[MetadataProviderName, CrossRefClient | OpenAlexClient],
    config: MetadataEnrichmentConfig,
    cache: MetadataEnrichmentCache,
    target: EnrichmentTarget,
    metadata: dict[str, Any],
    *,
    force: bool,
) -> tuple[MetadataWork | None, bool]:
    provider_failed = False
    for provider in _provider_order(config):
        client = clients.get(provider)
        if client is None:
            continue
        try:
            work = _lookup_work(client, provider, cache, target, metadata, force=force)
        except httpx.HTTPError as exc:
            provider_failed = True
            logger.warning(
                "metadata.provider_failed",
                provider=provider,
                file=str(target.path),
                error=str(exc),
            )
            continue
        if work is not None:
            return work, provider_failed
        logger.info("metadata.provider_no_match", provider=provider, file=str(target.path))
    return None, provider_failed


def _lookup_work(
    client: CrossRefClient | OpenAlexClient,
    provider: MetadataProviderName,
    cache: MetadataEnrichmentCache,
    target: EnrichmentTarget,
    metadata: dict[str, Any],
    *,
    force: bool,
) -> MetadataWork | None:
    possible_doi = _first_doi(metadata, target.text)
    if possible_doi:
        cached = None if force else _cache_get_work(cache, provider, "doi", possible_doi)
        if cached is not None:
            return cached
        work = client.lookup_by_doi(possible_doi)
        if work is not None:
            cache.set(provider, "doi", possible_doi, asdict(work))
            return work

    title = infer_title(metadata, target.text)
    if title:
        title_key = _normalize_cache_text(title)
        cached = None if force else _cache_get_work(cache, provider, "title", title_key)
        if cached is not None:
            return cached
        work = client.query_by_title(title)
        if work is not None:
            cache.set(provider, "title", title_key, asdict(work))
        return work
    return None


def _cache_get_work(
    cache: MetadataEnrichmentCache,
    provider: MetadataProviderName,
    query_type: str,
    query_value: str,
) -> MetadataWork | None:
    payload = cache.get(provider, query_type, query_value)
    if payload is None:
        return None
    return MetadataWork(**payload)


def _normalize_cache_text(value: str) -> str:
    return " ".join(value.lower().strip().split())


def _first_doi(metadata: dict[str, Any], text: str) -> str | None:
    doi = metadata.get("doi")
    if isinstance(doi, str) and doi.strip():
        return _normalize_doi(doi)

    for key in ("url", "external_url"):
        value = metadata.get(key)
        if isinstance(value, str):
            extracted = extract_doi(value)
            if extracted:
                return _normalize_doi(extracted)

    extracted = extract_doi(text[:8000])
    return _normalize_doi(extracted) if extracted else None


def _merge_work_metadata(
    metadata: dict[str, Any],
    work: CrossRefWork,
    *,
    force: bool,
) -> dict[str, Any]:
    enriched = dict(metadata)
    _set_metadata(enriched, "doi", work.doi, force=force)
    _set_metadata(enriched, "title", work.title, force=force)
    _set_metadata(enriched, "authors", work.authors, force=force)
    _set_metadata(enriched, "year", work.year, force=force)
    _set_metadata(enriched, "container_title", work.container_title, force=force)
    _set_metadata(enriched, "publisher", work.publisher, force=force)
    _set_metadata(enriched, "url", work.url, force=force)
    _set_metadata(enriched, "openalex_id", work.openalex_id, force=force)
    _set_metadata(enriched, "abstract", work.abstract, force=force)
    _set_metadata(enriched, "referenced_work_ids", work.referenced_work_ids or [], force=force)
    _set_metadata(enriched, "referenced_dois", work.referenced_dois or [], force=force)
    _set_metadata(enriched, "related_work_ids", work.related_work_ids or [], force=force)
    _set_metadata(enriched, "cited_by_count", work.cited_by_count, force=force)
    enriched["metadata_source"] = work.source
    enriched["metadata_enriched_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {key: value for key, value in enriched.items() if value not in ("", [], None)}


def _set_metadata(metadata: dict[str, Any], key: str, value: Any, *, force: bool) -> None:
    if value in ("", [], None):
        return
    if force or not metadata.get(key):
        metadata[key] = value


def _work_from_crossref_message(message: dict[str, Any]) -> CrossRefWork | None:
    doi = _first_string(message.get("DOI"))
    if not doi:
        return None

    title = _first_string(message.get("title"))
    authors = _authors_from_crossref(message.get("author"))
    year = _year_from_crossref(message)
    container_title = _first_string(message.get("container-title"))
    publisher = _first_string(message.get("publisher"))
    url = _first_string(message.get("URL"))
    score = message.get("score")
    referenced_dois = _referenced_dois_from_crossref(message.get("reference"))

    return CrossRefWork(
        source="crossref",
        doi=doi,
        title=title,
        authors=authors,
        year=year,
        container_title=container_title,
        publisher=publisher,
        url=url,
        score=float(score) if isinstance(score, (int, float)) else None,
        referenced_dois=referenced_dois,
    )


def _work_from_openalex_message(message: dict[str, Any]) -> MetadataWork | None:
    doi = _normalize_doi(_first_string(message.get("doi")))
    if not doi:
        return None

    primary_location = message.get("primary_location")
    source = primary_location.get("source") if isinstance(primary_location, dict) else None
    container_title = ""
    if isinstance(source, dict):
        container_title = _first_string(source.get("display_name"))

    score = message.get("relevance_score")
    cited_by_count = message.get("cited_by_count")
    return MetadataWork(
        source="openalex",
        doi=doi,
        title=_first_string(message.get("display_name")),
        authors=_authors_from_openalex(message.get("authorships")),
        year=_int_or_none(message.get("publication_year")),
        container_title=container_title,
        publisher=_first_string(message.get("publisher")),
        url=_first_string(message.get("doi")) or _first_string(message.get("id")),
        score=float(score) if isinstance(score, (int, float)) else None,
        openalex_id=_first_string(message.get("id")),
        abstract=_abstract_from_openalex(message.get("abstract_inverted_index")),
        referenced_work_ids=_strings_from_list(message.get("referenced_works")),
        related_work_ids=_strings_from_list(message.get("related_works")),
        cited_by_count=cited_by_count if isinstance(cited_by_count, int) else None,
    )


def _first_string(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item.strip()
    return ""


def _authors_from_crossref(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    authors: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        given = _first_string(item.get("given"))
        family = _first_string(item.get("family"))
        name = " ".join(part for part in [given, family] if part)
        if name:
            authors.append(name)
    return authors


def _authors_from_openalex(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    authors: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        author = item.get("author")
        if isinstance(author, dict):
            name = _first_string(author.get("display_name"))
            if name:
                authors.append(name)
    return authors


def _referenced_dois_from_crossref(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    dois: list[str] = []
    for item in value:
        if not isinstance(item, dict):
            continue
        doi = _normalize_doi(_first_string(item.get("DOI")) or _first_string(item.get("doi")))
        if doi:
            dois.append(doi)
    return sorted(set(dois))


def _abstract_from_openalex(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    positions: dict[int, str] = {}
    for word, raw_indexes in value.items():
        if not isinstance(word, str) or not isinstance(raw_indexes, list):
            continue
        for index in raw_indexes:
            if isinstance(index, int):
                positions[index] = word
    return " ".join(positions[index] for index in sorted(positions))


def _strings_from_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [item.strip() for item in value if isinstance(item, str) and item.strip()]


def _normalize_doi(value: str) -> str:
    value = value.strip()
    value = re.sub(r"^https?://(dx\.)?doi\.org/", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^doi:", "", value, flags=re.IGNORECASE)
    return value.rstrip(TRAILING_DOI_CHARS)


def _int_or_none(value: Any) -> int | None:
    return value if isinstance(value, int) else None


def _year_from_crossref(message: dict[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "published", "issued"):
        value = message.get(key)
        if not isinstance(value, dict):
            continue
        date_parts = value.get("date-parts")
        if (
            isinstance(date_parts, list)
            and date_parts
            and isinstance(date_parts[0], list)
            and date_parts[0]
            and isinstance(date_parts[0][0], int)
        ):
            return date_parts[0][0]
    return None
