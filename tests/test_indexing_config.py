from __future__ import annotations

from pathlib import Path

from rag_paper.config import AppConfig
from rag_paper.indexer import (
    _build_index_plan,
    _prepare_index_targets,
    clear_index_failure,
    discover_configured_pdfs,
    find_skip_markers_for_file,
    load_failed_index_paths,
    record_index_failure,
)
from rag_paper.manifest import IndexManifest


def _write_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n")


def test_papers_config_accepts_multiple_roots_and_tags(tmp_path: Path) -> None:
    root_a = tmp_path / "papers-a"
    root_b = tmp_path / "papers-b"
    _write_pdf(root_a / "one.pdf")
    _write_pdf(root_b / "two.pdf")

    config = AppConfig.model_validate(
        {
            "papers": [
                {"root_path": str(root_a), "tags": ["local"]},
                {"root_path": str(root_b), "tags": ["zetro"]},
            ]
        }
    )

    candidates = discover_configured_pdfs(config)

    assert [candidate.path.name for candidate in candidates] == ["one.pdf", "two.pdf"]
    assert [candidate.tags for candidate in candidates] == [("local",), ("zetro",)]


def test_papers_root_path_accepts_array(tmp_path: Path) -> None:
    root_a = tmp_path / "papers-a"
    root_b = tmp_path / "papers-b"
    _write_pdf(root_a / "one.pdf")
    _write_pdf(root_b / "two.pdf")

    config = AppConfig.model_validate(
        {
            "papers": [
                {
                    "root_path": [str(root_a), str(root_b)],
                    "tags": ["shared"],
                }
            ]
        }
    )

    candidates = discover_configured_pdfs(config)

    assert [candidate.path.name for candidate in candidates] == ["one.pdf", "two.pdf"]
    assert [candidate.tags for candidate in candidates] == [("shared",), ("shared",)]


def test_skip_marker_file_skips_marked_root_or_subdirectory(tmp_path: Path) -> None:
    root = tmp_path / "papers"
    skipped = root / "skipped"
    _write_pdf(root / "keep.pdf")
    _write_pdf(skipped / "drop.pdf")
    (skipped / ".skip-index").write_text("", encoding="utf-8")

    config = AppConfig.model_validate(
        {"papers": [{"root_path": str(root), "skip_marker_file": ".skip-index"}]}
    )

    candidates = discover_configured_pdfs(config)

    assert [candidate.path.name for candidate in candidates] == ["keep.pdf"]


def test_file_indexing_detects_skip_marker_in_parent_directory(tmp_path: Path) -> None:
    root = tmp_path / "papers"
    marked = root / "marked"
    pdf_path = marked / "paper.pdf"
    _write_pdf(pdf_path)
    (marked / ".skip-index").write_text("", encoding="utf-8")
    config = AppConfig.model_validate(
        {"papers": [{"root_path": [str(root)], "skip_marker_file": ".skip-index"}]}
    )

    hits = find_skip_markers_for_file(config, pdf_path)

    assert len(hits) == 1
    assert hits[0].directory == marked.resolve()
    assert hits[0].marker == ".skip-index"


def test_max_files_limits_prepared_targets_and_plan(tmp_path: Path) -> None:
    root = tmp_path / "papers"
    for name in ["a.pdf", "b.pdf", "c.pdf"]:
        _write_pdf(root / name)

    config = AppConfig.model_validate({"papers": [{"root_path": str(root)}]})
    candidates = discover_configured_pdfs(config)
    manifest = IndexManifest(tmp_path / "manifest.json")

    targets, skipped_files = _prepare_index_targets(
        candidates,
        manifest,
        force=False,
        only_new=False,
        max_files=2,
    )
    plan = _build_index_plan(targets)

    assert [target.candidate.path.name for target in targets] == ["a.pdf", "b.pdf"]
    assert skipped_files == 1
    assert plan.total_files == 2
    assert [(root.root_path, root.files) for root in plan.roots] == [(str(root.resolve()), 2)]


def test_prepare_targets_skips_current_file_by_stat_without_sha(tmp_path: Path) -> None:
    root = tmp_path / "papers"
    pdf_path = root / "a.pdf"
    _write_pdf(pdf_path)
    config = AppConfig.model_validate({"papers": [{"root_path": str(root)}]})
    candidate = discover_configured_pdfs(config)[0]
    manifest = IndexManifest(tmp_path / "manifest.json")
    manifest.update(pdf_path, "old-digest", ["chunk-1"])

    targets, skipped_files = _prepare_index_targets(
        [candidate],
        manifest,
        force=False,
        only_new=False,
        max_files=None,
    )

    assert targets == []
    assert skipped_files == 1


def test_index_failure_log_records_loads_and_clears(tmp_path: Path) -> None:
    failed_path = tmp_path / "failed.jsonl"
    pdf_path = tmp_path / "paper.pdf"
    _write_pdf(pdf_path)

    record_index_failure(failed_path, pdf_path, RuntimeError("boom"))

    assert load_failed_index_paths(failed_path) == [pdf_path.resolve()]
    clear_index_failure(failed_path, pdf_path)
    assert load_failed_index_paths(failed_path) == []
