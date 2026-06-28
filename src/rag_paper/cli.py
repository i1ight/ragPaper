from __future__ import annotations

import locale
from datetime import datetime, timezone, tzinfo
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import click
from rich.console import Console
from rich.table import Table

from rag_paper.citation_graph import build_citation_graph
from rag_paper.config import AppConfig, load_config, write_default_config
from rag_paper.dedup import run_dedup_report
from rag_paper.enrichment import discover_enrichment_targets, enrich_targets
from rag_paper.indexer import IndexPlan, SkipMarkerHit, run_indexing
from rag_paper.inspection import (
    IndexedPaperDetail,
    inspect_indexed_paper,
    inspect_indexed_papers,
    is_pdf_indexed,
)
from rag_paper.logging import configure_logging
from rag_paper.metadata import load_paper_metadata
from rag_paper.mcp_server import run_mcp_server
from rag_paper.retrieval import HybridRetriever, result_to_dict

console = Console()


def load_app_config(config_path: str) -> AppConfig:
    default_config_path = Path("./config.json").resolve()
    requested_config_path = Path(config_path).expanduser().resolve()
    if requested_config_path == default_config_path and not requested_config_path.exists():
        config = AppConfig()
    else:
        config = load_config(config_path)
    configure_logging(config)
    return config


@click.group()
def cli() -> None:
    """Local PDF paper indexing and MCP retrieval service."""


@cli.command("init-config")
@click.option("--path", default="./config.json", show_default=True, help="Output config path.")
def init_config(path: str) -> None:
    """Write a default JSON config file."""
    written = write_default_config(path)
    console.print(f"Config written: {written}")


@cli.command("index")
@click.option("--config", "config_path", default="./config.json", show_default=True)
@click.option("--force", is_flag=True, help="Re-index all selected PDFs.")
@click.option("--file", "file_path", help="Index or refresh one PDF file.")
@click.option("--only-new", is_flag=True, help="Only index PDFs not present in the manifest.")
@click.option("--retry-failed", is_flag=True, help="Retry PDFs recorded in the index failure log.")
@click.option("--yes", is_flag=True, help="Start vectorization without interactive confirmation.")
@click.option("--max-files", default=None, type=click.IntRange(min=0), help="Maximum PDFs to vectorize.")
def index(
    config_path: str,
    force: bool,
    file_path: str | None,
    only_new: bool,
    retry_failed: bool,
    yes: bool,
    max_files: int | None,
) -> None:
    """Vectorize PDFs and store chunks in Chroma."""
    config = load_app_config(config_path)
    confirm_plan = None if yes or config.indexing.assume_yes else _confirm_index_plan
    confirm_skip_markers = (
        _ack_skip_markers if yes or config.indexing.assume_yes else _confirm_skip_markers
    )
    summary = run_indexing(
        config,
        force=force,
        file_path=file_path,
        only_new=only_new,
        retry_failed=retry_failed,
        max_files=max_files,
        confirm_plan=confirm_plan,
        confirm_skip_markers=confirm_skip_markers,
    )
    table = Table(title="Index Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Indexed files", str(summary.indexed_files))
    table.add_row("Skipped files", str(summary.skipped_files))
    table.add_row("Chunks", str(summary.chunks))
    table.add_row("Duplicate files", str(summary.duplicate_files))
    table.add_row("Enriched metadata files", str(summary.enriched_metadata_files))
    table.add_row("Failed metadata files", str(summary.failed_metadata_files))
    table.add_row("Chroma dir", str(config.chroma_dir))
    table.add_row("Collection", config.chroma.collection)
    console.print(table)


@cli.command("list-indexed-papers")
@click.option("--config", "config_path", default="./config.json", show_default=True)
@click.option("--limit", default=10, show_default=True, type=click.IntRange(min=0), help="Maximum papers to show.")
def list_indexed_papers(config_path: str, limit: int | None) -> None:
    """Show papers currently stored in local Chroma."""
    config = load_app_config(config_path)
    summary = inspect_indexed_papers(config, limit=limit)

    table = Table(
        title=(
            f"Indexed Papers (showing {summary.shown_count} of "
            f"{summary.paper_count} papers, {summary.chunk_count} chunks)"
        ),
        show_lines=False,
    )
    table.add_column("#", justify="right", no_wrap=True, width=4)
    table.add_column("Title / File", overflow="fold", ratio=4)
    table.add_column("Year", justify="right", no_wrap=True, width=6)
    table.add_column("Chunks", justify="right", no_wrap=True, width=7)
    table.add_column("DOI", overflow="fold", ratio=2)
    table.add_column("Source", overflow="fold", ratio=3)

    for index, paper in enumerate(summary.papers, start=1):
        title = paper.title or paper.file_name
        table.add_row(
            str(index),
            title,
            str(paper.year or ""),
            str(paper.chunk_count),
            paper.doi,
            paper.source_path,
        )
    console.print(table)


@cli.command("show-indexed-paper")
@click.argument("selector")
@click.option("--config", "config_path", default="./config.json", show_default=True)
@click.option("--limit", default=10, show_default=True, type=click.IntRange(min=1), help="Maximum matching papers to show.")
@click.option("--all-chunks", is_flag=True, help="Show all chunk IDs instead of the first 5.")
def show_indexed_paper(
    selector: str,
    config_path: str,
    limit: int,
    all_chunks: bool,
) -> None:
    """Show detailed Chroma metadata for matching indexed papers."""
    config = load_app_config(config_path)
    details = inspect_indexed_paper(config, selector, limit=limit)
    if not details:
        raise click.ClickException(f"Indexed paper not found: {selector}")
    for detail in details:
        _print_indexed_paper_detail(detail, config, show_all_chunks=all_chunks)


@cli.command("dedupe-papers")
@click.option("--config", "config_path", default="./config.json", show_default=True)
@click.option("--file", "file_path", help="Check one PDF file against the configured set.")
@click.option("--max-files", default=None, type=click.IntRange(min=0), help="Maximum PDFs to check.")
def dedupe_papers(
    config_path: str,
    file_path: str | None,
    max_files: int | None,
) -> None:
    """Build a duplicate-paper report."""
    config = load_app_config(config_path)
    summary = run_dedup_report(config, file_path=file_path, max_files=max_files)

    table = Table(title="Dedup Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Checked files", str(summary.checked_files))
    table.add_row("Duplicate pairs", str(summary.duplicate_pairs))
    table.add_row("Skipped files", str(summary.skipped_files))
    table.add_row("Report path", summary.report_path)
    console.print(table)


@cli.command("build-citation-graph")
@click.option("--config", "config_path", default="./config.json", show_default=True)
def build_citation_graph_command(config_path: str) -> None:
    """Build a local citation graph JSON file from enriched metadata."""
    config = load_app_config(config_path)
    summary = build_citation_graph(config)

    table = Table(title="Citation Graph Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Nodes", str(summary.nodes))
    table.add_row("Edges", str(summary.edges))
    table.add_row("Path", summary.path)
    table.add_row("Mermaid path", summary.mermaid_path)
    console.print(table)


@cli.command("enrich-metadata")
@click.option("--config", "config_path", default="./config.json", show_default=True)
@click.option("--force", is_flag=True, help="Refresh metadata even when DOI already exists.")
@click.option("--file", "file_path", help="Enrich one PDF file.")
@click.option("--yes", is_flag=True, help="Start enrichment without interactive confirmation.")
@click.option("--max-files", default=None, type=click.IntRange(min=0), help="Maximum PDFs to enrich.")
def enrich_metadata_command(
    config_path: str,
    force: bool,
    file_path: str | None,
    yes: bool,
    max_files: int | None,
) -> None:
    """Fetch DOI and bibliographic metadata for PDFs."""
    config = load_app_config(config_path)
    if file_path and not is_pdf_indexed(config, file_path):
        raise click.ClickException(
            f"PDF is not indexed in local Chroma yet; run `rag-paper index --file {file_path}` first."
        )
    metadata_map = load_paper_metadata(config.metadata_path)
    targets = discover_enrichment_targets(config, metadata_map, file_path=file_path)
    if max_files is not None:
        targets = targets[:max_files]

    if not yes and not config.indexing.assume_yes:
        _confirm_enrichment_plan(len(targets))

    summary = enrich_targets(
        config,
        targets,
        metadata_map=metadata_map,
        force=force,
    )

    table = Table(title="Metadata Enrichment Summary")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("Checked files", str(summary.checked_files))
    table.add_row("Updated files", str(summary.updated_files))
    table.add_row("Skipped files", str(summary.skipped_files))
    table.add_row("Failed files", str(summary.failed_files))
    table.add_row("Metadata path", str(config.metadata_path))
    console.print(table)


@cli.command("serve")
@click.option("--config", "config_path", default="./config.json", show_default=True)
def serve(config_path: str) -> None:
    """Start the MCP knowledge retrieval service."""
    config = load_app_config(config_path)
    table = Table(title="MCP Service")
    table.add_column("Item")
    table.add_column("Value")
    table.add_row("Name", "rag-paper")
    table.add_row("Transport", config.mcp.transport)
    table.add_row("Host", config.mcp.host)
    table.add_row("Port", str(config.mcp.port))
    if config.mcp.endpoint_url is not None:
        table.add_row("Endpoint", config.mcp.endpoint_url)
    table.add_row("Embedding", f"{config.embedding.provider}:{config.ollama.model}")
    for paper_dir in config.paper_dirs:
        table.add_row("Paper root", str(paper_dir))
    table.add_row("Chroma dir", str(config.chroma_dir))
    table.add_row("Collection", config.chroma.collection)
    console.print(table)
    run_mcp_server(config)


@cli.command("search")
@click.argument("query")
@click.option("--config", "config_path", default="./config.json", show_default=True)
@click.option("--top-k", default=None, type=int)
@click.option("--author", default=None)
@click.option("--year", default=None, type=int)
@click.option("--tag", default=None)
@click.option("--file-name", default=None)
def search(
    query: str,
    config_path: str,
    top_k: int | None,
    author: str | None,
    year: int | None,
    tag: str | None,
    file_name: str | None,
) -> None:
    """Run a local hybrid search without starting MCP."""
    config = load_app_config(config_path)
    retriever = HybridRetriever(config)
    results = retriever.search(
        query,
        top_k=top_k,
        author=author,
        year=year,
        tag=tag,
        file_name=file_name,
    )
    for item in [result_to_dict(result) for result in results]:
        _print_search_result(item)


def _print_search_result(item: dict[str, Any]) -> None:
    metadata = item["metadata"]
    title = metadata.get("title") or metadata.get("file_name") or item["chunk_id"]
    console.print(f"\n[bold]{title}[/bold]")
    console.print(
        f"chunk_id={item['chunk_id']} score={item['score']:.4f} "
        f"vector={item.get('vector_score')} bm25={item.get('bm25_score')}"
    )
    console.print(str(Path(metadata.get("source_path", ""))))
    text = item["text"].replace("\n", " ")
    console.print(text[:800] + ("..." if len(text) > 800 else ""))


def _print_indexed_paper_detail(
    detail: IndexedPaperDetail,
    config: AppConfig,
    *,
    show_all_chunks: bool = False,
) -> None:
    summary = Table(title="Indexed Paper")
    summary.add_column("Field", no_wrap=True)
    summary.add_column("Value", overflow="fold")
    summary.add_row("Title", detail.title or detail.file_name)
    summary.add_row("File", detail.file_name)
    summary.add_row("Source path", detail.source_path)
    summary.add_row("Authors", detail.authors)
    summary.add_row("Year", str(detail.year or ""))
    summary.add_row("DOI", detail.doi)
    summary.add_row("Chunks", str(detail.chunk_count))
    console.print(summary)

    metadata = Table(title="Metadata")
    metadata.add_column("Key", no_wrap=True)
    metadata.add_column("Value", overflow="fold")
    for key in sorted(detail.metadata):
        metadata.add_row(key, _format_metadata_value(key, detail.metadata[key], config))
    console.print(metadata)

    chunks = Table(title="Chunk IDs")
    chunks.add_column("#", justify="right", no_wrap=True, width=4)
    chunks.add_column("Chunk ID", overflow="fold")
    visible_chunk_ids = detail.chunk_ids if show_all_chunks else detail.chunk_ids[:5]
    for index, chunk_id in enumerate(visible_chunk_ids, start=1):
        chunks.add_row(str(index), chunk_id)
    hidden_count = len(detail.chunk_ids) - len(visible_chunk_ids)
    if hidden_count > 0:
        chunks.add_section()
        chunks.add_row("", f"{hidden_count} more hidden; rerun with --all-chunks to show all.")
    console.print(chunks)


def _format_metadata_value(key: str, value: Any, config: AppConfig) -> str:
    if key == "metadata_enriched_at" and isinstance(value, str):
        return _format_datetime_value(
            value,
            timezone_name=config.display.datetime_timezone,
            datetime_format=config.display.datetime_format,
        )
    return str(value)


def _format_datetime_value(
    value: str,
    *,
    timezone_name: str,
    datetime_format: str,
) -> str:
    parsed = _parse_datetime(value)
    if parsed is None:
        return value

    display_timezone = _resolve_display_timezone(timezone_name)
    display_format = datetime_format.strip() or _default_datetime_format()
    return parsed.astimezone(display_timezone).strftime(display_format)


def _parse_datetime(value: str) -> datetime | None:
    normalized = value.strip()
    if not normalized:
        return None
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _resolve_display_timezone(timezone_name: str) -> tzinfo:
    normalized = timezone_name.strip()
    if not normalized or normalized.lower() in {"local", "system"}:
        return datetime.now().astimezone().tzinfo or timezone.utc
    if normalized.upper() == "UTC":
        return timezone.utc
    try:
        return ZoneInfo(normalized)
    except ZoneInfoNotFoundError as exc:
        raise click.ClickException(f"Unknown display timezone: {timezone_name}") from exc


def _default_datetime_format() -> str:
    try:
        locale.setlocale(locale.LC_TIME, "")
    except locale.Error:
        pass
    return "%c"


def _confirm_index_plan(plan: IndexPlan) -> None:
    table = Table(title="Index Plan")
    table.add_column("Paper root")
    table.add_column("PDFs to vectorize", justify="right")
    for root in plan.roots:
        table.add_row(root.root_path, str(root.files))
    if not plan.roots:
        table.add_row("-", "0")
    table.add_section()
    table.add_row("Total", str(plan.total_files))
    console.print(table)

    if plan.total_files == 0:
        console.print("No PDFs need vectorization.")
        return
    click.confirm("Start vectorization?", abort=True)


def _print_skip_marker_warning(hits: tuple[SkipMarkerHit, ...]) -> None:
    table = Table(title="[bold yellow]Skip Marker Detected[/bold yellow]")
    table.add_column("Marker", no_wrap=True)
    table.add_column("Directory", overflow="fold")
    table.add_column("Marker path", overflow="fold")
    table.add_column("Configured root", overflow="fold")
    for hit in hits:
        table.add_row(
            f"[bold yellow]{hit.marker}[/bold yellow]",
            str(hit.directory),
            str(hit.marker_path),
            str(hit.root_path or ""),
        )
    console.print(table)


def _confirm_skip_markers(hits: tuple[SkipMarkerHit, ...]) -> None:
    _print_skip_marker_warning(hits)
    click.confirm("Skip marker was detected. Continue indexing anyway?", abort=True)


def _ack_skip_markers(hits: tuple[SkipMarkerHit, ...]) -> None:
    _print_skip_marker_warning(hits)
    console.print("[bold yellow]Continuing because --yes or indexing.assume_yes is set.[/bold yellow]")


def _confirm_enrichment_plan(total_files: int) -> None:
    table = Table(title="Metadata Enrichment Plan")
    table.add_column("Metric")
    table.add_column("Value", justify="right")
    table.add_row("PDFs to enrich", str(total_files))
    console.print(table)

    if total_files == 0:
        console.print("No PDFs need metadata enrichment.")
        return
    click.confirm("Start metadata enrichment?", abort=True)


if __name__ == "__main__":
    cli()
