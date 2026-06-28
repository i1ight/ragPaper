from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from rag_paper.citation_graph import build_citation_graph
from rag_paper.config import AppConfig
from rag_paper.dedup import run_dedup_report
from rag_paper.enrichment import enrich_metadata
from rag_paper.indexer import run_indexing
from rag_paper.inspection import inspect_indexed_paper, inspect_indexed_papers
from rag_paper.logging import configure_logging, logger
from rag_paper.retrieval import HybridRetriever, result_to_dict


def create_mcp_server(config: AppConfig) -> FastMCP:
    configure_logging(config)
    mcp = FastMCP(
        "rag-paper",
        host=config.mcp.host,
        port=config.mcp.port,
        instructions=(
            "Local paper knowledge retrieval service. Use tools to import PDFs, "
            "search indexed chunks, inspect metadata, fetch chunks, and export context. "
            "This service is not a chat model."
        ),
    )
    retriever = HybridRetriever(config)

    @mcp.tool()
    def search_papers(
        query: str,
        top_k: int | None = None,
        author: str | None = None,
        year: int | None = None,
        tag: str | None = None,
        file_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid BM25 + vector search over indexed paper chunks."""
        results = retriever.search(
            query,
            top_k=top_k,
            author=author,
            year=year,
            tag=tag,
            file_name=file_name,
        )
        return [result_to_dict(result) for result in results]

    @mcp.tool()
    def search_by_metadata(
        author: str | None = None,
        year: int | None = None,
        tag: str | None = None,
        file_name: str | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Find chunks by paper metadata such as author, year, tag, or file name."""
        results = retriever.search_by_metadata(
            author=author,
            year=year,
            tag=tag,
            file_name=file_name,
            limit=limit,
        )
        return [result_to_dict(result) for result in results]

    @mcp.tool()
    def get_chunk(chunk_id: str) -> dict[str, Any] | None:
        """Fetch one indexed chunk by chunk_id."""
        return retriever.get_chunk(chunk_id)

    @mcp.tool()
    def export_context(chunk_ids: list[str]) -> str:
        """Export selected chunks as a compact Markdown context block."""
        return retriever.export_context(chunk_ids)

    @mcp.tool()
    def list_indexed_papers(limit: int | None = 10) -> dict[str, Any]:
        """List papers currently stored in local Chroma."""
        summary = inspect_indexed_papers(config, limit=limit)
        return {
            "paper_count": summary.paper_count,
            "chunk_count": summary.chunk_count,
            "shown_count": summary.shown_count,
            "papers": [paper.__dict__ for paper in summary.papers],
        }

    @mcp.tool()
    def show_indexed_paper(
        selector: str,
        limit: int = 10,
        include_all_chunks: bool = False,
    ) -> list[dict[str, Any]]:
        """Show detailed Chroma metadata for matching indexed papers."""
        details = inspect_indexed_paper(config, selector, limit=limit)
        payload: list[dict[str, Any]] = []
        for detail in details:
            item = detail.__dict__.copy()
            chunk_ids = list(detail.chunk_ids)
            item["chunk_ids"] = chunk_ids if include_all_chunks else chunk_ids[:5]
            item["hidden_chunk_ids"] = 0 if include_all_chunks else max(0, len(chunk_ids) - 5)
            payload.append(item)
        return payload

    @mcp.tool()
    def import_papers(
        force: bool = False,
        file_path: str | None = None,
        only_new: bool = False,
        retry_failed: bool = False,
        max_files: int | None = None,
    ) -> dict[str, int]:
        """Index PDFs into local Chroma. Use force to rebuild existing files."""
        summary = run_indexing(
            config,
            force=force,
            file_path=file_path,
            only_new=only_new,
            retry_failed=retry_failed,
            max_files=max_files,
        )
        logger.info("mcp.import_papers", **summary.__dict__)
        return {
            "indexed_files": summary.indexed_files,
            "skipped_files": summary.skipped_files,
            "chunks": summary.chunks,
            "duplicate_files": summary.duplicate_files,
            "enriched_metadata_files": summary.enriched_metadata_files,
            "failed_metadata_files": summary.failed_metadata_files,
        }

    @mcp.tool()
    def enrich_paper_metadata(
        force: bool = False,
        file_path: str | None = None,
        max_files: int | None = None,
    ) -> dict[str, int]:
        """Fetch DOI and bibliographic metadata for PDFs."""
        summary = enrich_metadata(
            config,
            force=force,
            file_path=file_path,
            max_files=max_files,
        )
        logger.info("mcp.enrich_paper_metadata", **summary.__dict__)
        return {
            "checked_files": summary.checked_files,
            "updated_files": summary.updated_files,
            "skipped_files": summary.skipped_files,
            "failed_files": summary.failed_files,
        }

    @mcp.tool()
    def dedupe_papers(
        file_path: str | None = None,
        max_files: int | None = None,
    ) -> dict[str, int | str]:
        """Build a duplicate-paper report."""
        summary = run_dedup_report(config, file_path=file_path, max_files=max_files)
        logger.info("mcp.dedupe_papers", **summary.__dict__)
        return {
            "checked_files": summary.checked_files,
            "duplicate_pairs": summary.duplicate_pairs,
            "skipped_files": summary.skipped_files,
            "report_path": summary.report_path,
        }

    @mcp.tool()
    def build_paper_citation_graph() -> dict[str, int | str]:
        """Build a local citation graph JSON file from enriched metadata."""
        summary = build_citation_graph(config)
        logger.info("mcp.build_paper_citation_graph", **summary.__dict__)
        return {
            "nodes": summary.nodes,
            "edges": summary.edges,
            "path": summary.path,
            "mermaid_path": summary.mermaid_path,
        }

    @mcp.tool()
    def service_info() -> dict[str, Any]:
        """Return service configuration useful for MCP clients."""
        return {
            "name": "rag-paper",
            "transport": config.mcp.transport,
            "host": config.mcp.host,
            "port": config.mcp.port,
            "endpoint": config.mcp.endpoint_url,
            "embedding_provider": config.embedding.provider,
            "ollama_model": config.ollama.model,
            "paper_roots": [str(path) for path in config.paper_dirs],
            "chroma_dir": str(config.chroma_dir),
            "collection": config.chroma.collection,
        }

    return mcp


def run_mcp_server(config: AppConfig) -> None:
    server = create_mcp_server(config)
    server.run(transport=config.mcp.transport)
