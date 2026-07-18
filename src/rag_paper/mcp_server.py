from __future__ import annotations

from typing import Any

from mcp.server.fastmcp import FastMCP

from rag_paper.config import AppConfig
from rag_paper.inspection import inspect_indexed_paper, inspect_indexed_papers, paper_citations
from rag_paper.logging import configure_logging
from rag_paper.retrieval import HybridRetriever, result_to_dict


def create_mcp_server(config: AppConfig) -> FastMCP:
    configure_logging(config)
    mcp = FastMCP(
        "rag-paper",
        host=config.mcp.host,
        port=config.mcp.port,
        instructions=(
            "Local paper knowledge retrieval service (read-only). Query the indexed "
            "paper library with these tools: search_papers for hybrid BM25 + vector "
            "search over chunks, search_by_metadata to find chunks by author/year/tag/"
            "file name, get_chunk to fetch one chunk by id, and export_context to export "
            "selected chunks as a compact Markdown block. list_indexed_papers and "
            "show_indexed_paper describe what is currently indexed. Results are "
            "chunk-granular, so a single paper may yield several chunks; use "
            "export_context to assemble the final context for a question. This service "
            "does not modify the library (index/enrich/delete are CLI-only) and is not "
            "a chat model."
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
        group_by_paper: bool = False,
        max_text_chars: int | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid BM25 + vector search over indexed paper chunks.

        Returns chunk-granular results ranked by a fused score; a single paper can
        appear multiple times. Optional filters: author (substring), year (exact),
        tag, file_name.

        Set group_by_paper=True to collapse to the best chunk per paper (returns
        up to top_k distinct papers, each with chunk_count and other_chunk_ids) —
        use this for broad coverage. Set max_text_chars to truncate each chunk's
        text for a compact first pass, then call get_chunk/export_context for full
        text. Defaults to the configured top_k when omitted.
        """
        results = retriever.search(
            query,
            top_k=top_k,
            author=author,
            year=year,
            tag=tag,
            file_name=file_name,
            group_by_paper=group_by_paper,
        )
        return [result_to_dict(result, max_text_chars=max_text_chars) for result in results]

    @mcp.tool()
    def search_by_metadata(
        author: str | None = None,
        year: int | None = None,
        tag: str | None = None,
        file_name: str | None = None,
        limit: int = 20,
        group_by_paper: bool = False,
        max_text_chars: int | None = None,
    ) -> list[dict[str, Any]]:
        """Find chunks by paper metadata such as author, year, tag, or file name.

        Use this for structured lookups (e.g. "papers by X in 2024") without a
        semantic query. Returns matching chunks, up to `limit`. Set
        group_by_paper=True to get one representative chunk per paper (up to
        `limit` distinct papers); set max_text_chars to truncate each chunk's text.
        """
        results = retriever.search_by_metadata(
            author=author,
            year=year,
            tag=tag,
            file_name=file_name,
            limit=limit,
            group_by_paper=group_by_paper,
        )
        return [result_to_dict(result, max_text_chars=max_text_chars) for result in results]

    @mcp.tool()
    def get_chunk(chunk_id: str) -> dict[str, Any] | None:
        """Fetch one indexed chunk by chunk_id."""
        return retriever.get_chunk(chunk_id)

    @mcp.tool()
    def export_context(chunk_ids: list[str]) -> str:
        """Export selected chunks as a compact Markdown context block.

        Pass chunk_ids returned by search_papers/search_by_metadata to assemble the
        final reading context for a question.
        """
        return retriever.export_context(chunk_ids)

    @mcp.tool()
    def list_indexed_papers(limit: int | None = 10) -> dict[str, Any]:
        """List papers currently stored in local Chroma."""
        summary = inspect_indexed_papers(config, limit=limit)
        return {
            "paper_count": summary.paper_count,
            "chunk_count": summary.chunk_count,
            "shown_count": summary.shown_count,
            "papers": [
                {k: v for k, v in paper.__dict__.items() if k != "source_path"}
                for paper in summary.papers
            ],
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
            item = {k: v for k, v in detail.__dict__.items() if k != "source_path"}
            item["metadata"] = {
                k: v for k, v in (detail.metadata or {}).items() if k != "source_path"
            }
            chunk_ids = list(detail.chunk_ids)
            item["chunk_ids"] = chunk_ids if include_all_chunks else chunk_ids[:5]
            item["hidden_chunk_ids"] = 0 if include_all_chunks else max(0, len(chunk_ids) - 5)
            payload.append(item)
        return payload

    @mcp.tool()
    def get_paper_citations(
        selector: str,
        external_sample: int = 20,
        incoming_limit: int = 50,
    ) -> dict[str, Any] | None:
        """Locally-resolved citation view for one paper.

        Given a selector (title / file name / DOI / source-path substring), returns
        the paper's outgoing references (each resolved to whether it is in the local
        library, with local title/path) and the local papers that reference it
        (incoming), plus cited_by_count. Useful for "what does this build on?" and
        "what cites it here?" without re-reading chunk text.

        Note: reference completeness depends on the enrichment provider — CrossRef
        exposes reference DOIs (directly matchable to local DOIs), OpenAlex exposes
        work ids (matched via local openalex_id).
        """
        return paper_citations(
            config,
            selector,
            external_sample=external_sample,
            incoming_limit=incoming_limit,
        )

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
