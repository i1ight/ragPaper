from __future__ import annotations

from rich.console import Console

import rag_paper.cli as cli_module
from rag_paper.cli import _format_datetime_value, _print_indexed_paper_detail
from rag_paper.config import AppConfig
from rag_paper.inspection import IndexedPaperDetail


def test_display_config_defaults_to_local_time_format() -> None:
    config = AppConfig()

    assert config.display.datetime_timezone == "local"
    assert config.display.datetime_format == ""


def test_data_dir_drives_default_runtime_paths() -> None:
    config = AppConfig.model_validate({"data_dir": "./portable_data"})

    assert config.chroma.persist_dir == "./portable_data/chroma_db"
    assert config.indexing.metadata_path == "./portable_data/paper_metadata.json"
    assert config.metadata_enrichment.cache_path == "./portable_data/cache/metadata_enrichment.sqlite3"
    assert config.citation_graph.mermaid_path == "./portable_data/citation_graph/citation_graph.md"


def test_format_datetime_value_with_configured_timezone_and_format() -> None:
    formatted = _format_datetime_value(
        "2026-06-28T10:00:00Z",
        timezone_name="Asia/Shanghai",
        datetime_format="%Y-%m-%d %H:%M:%S %Z",
    )

    assert formatted == "2026-06-28 18:00:00 CST"


def test_print_indexed_paper_detail_limits_chunk_ids_by_default() -> None:
    capture_console = Console(record=True, width=120)
    old_console = cli_module.console
    cli_module.console = capture_console
    try:
        _print_indexed_paper_detail(
            IndexedPaperDetail(
                source_path="/papers/a.pdf",
                file_name="a.pdf",
                title="A",
                chunk_count=7,
                chunk_ids=[f"chunk-{index}" for index in range(7)],
            ),
            AppConfig(),
        )
        output = capture_console.export_text()
    finally:
        cli_module.console = old_console

    assert "chunk-0" in output
    assert "chunk-4" in output
    assert "chunk-5" not in output
    assert "2 more hidden" in output


def test_print_indexed_paper_detail_can_show_all_chunk_ids() -> None:
    capture_console = Console(record=True, width=120)
    old_console = cli_module.console
    cli_module.console = capture_console
    try:
        _print_indexed_paper_detail(
            IndexedPaperDetail(
                source_path="/papers/a.pdf",
                file_name="a.pdf",
                title="A",
                chunk_count=6,
                chunk_ids=[f"chunk-{index}" for index in range(6)],
            ),
            AppConfig(),
            show_all_chunks=True,
        )
        output = capture_console.export_text()
    finally:
        cli_module.console = old_console

    assert "chunk-5" in output
    assert "more hidden" not in output
