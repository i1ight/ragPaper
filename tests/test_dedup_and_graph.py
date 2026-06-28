from __future__ import annotations

from pathlib import Path

from rag_paper.citation_graph import build_citation_graph
from rag_paper.config import AppConfig
from rag_paper.dedup import cosine_similarity, find_duplicate_pairs
from rag_paper.indexer import PdfCandidate
from rag_paper.metadata import save_paper_metadata


def _write_pdf(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"%PDF-1.4\n")


def test_cosine_similarity() -> None:
    assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == 1.0
    assert cosine_similarity([1.0, 0.0], [0.0, 1.0]) == 0.0


def test_metadata_dedup_finds_same_doi(tmp_path: Path) -> None:
    first = tmp_path / "a.pdf"
    second = tmp_path / "b.pdf"
    _write_pdf(first)
    _write_pdf(second)
    candidates = [
        PdfCandidate(path=first, root_path=tmp_path, tags=()),
        PdfCandidate(path=second, root_path=tmp_path, tags=()),
    ]
    config = AppConfig.model_validate({"dedup": {"semantic_enabled": False}})

    pairs = find_duplicate_pairs(
        config,
        candidates,
        metadata_map={
            str(first): {"doi": "10.1000/test"},
            str(second): {"doi": "10.1000/test"},
        },
        embeddings=None,
    )

    assert len(pairs) == 1
    assert pairs[0].reason == "doi"
    assert pairs[0].duplicate_path == str(second)


def test_build_citation_graph_links_indexed_and_external_nodes(tmp_path: Path) -> None:
    papers = tmp_path / "papers"
    first = papers / "a.pdf"
    second = papers / "b.pdf"
    metadata_path = tmp_path / "metadata.json"
    graph_path = tmp_path / "graph.json"
    mermaid_path = tmp_path / "graph.md"
    _write_pdf(first)
    _write_pdf(second)
    save_paper_metadata(
        metadata_path,
        {
            str(first.resolve()): {
                "doi": "10.1000/a",
                "title": "A",
                "referenced_dois": ["10.1000/b", "10.1000/external"],
            },
            str(second.resolve()): {
                "doi": "10.1000/b",
                "title": "B",
            },
        },
    )
    config = AppConfig.model_validate(
        {
            "papers": [{"root_path": str(papers)}],
            "indexing": {"metadata_path": str(metadata_path)},
            "citation_graph": {"path": str(graph_path), "mermaid_path": str(mermaid_path)},
        }
    )

    summary = build_citation_graph(config)

    assert summary.nodes == 3
    assert summary.edges == 2
    assert graph_path.exists()
    assert mermaid_path.exists()
    assert "```mermaid" in mermaid_path.read_text(encoding="utf-8")
