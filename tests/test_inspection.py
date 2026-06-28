from __future__ import annotations

from rag_paper.inspection import _matches_selector, _summaries_from_payload


def test_summaries_group_chunks_by_source_path() -> None:
    payload = {
        "ids": ["chunk-a", "chunk-b", "chunk-c"],
        "metadatas": [
            {
                "source_path": "/papers/a.pdf",
                "file_name": "a.pdf",
                "title": "Paper A",
                "year": 2024,
                "doi": "10.1000/a",
            },
            {
                "source_path": "/papers/a.pdf",
                "file_name": "a.pdf",
                "title": "Paper A",
                "year": 2024,
            },
            {
                "source_path": "/papers/b.pdf",
                "file_name": "b.pdf",
                "title": "Paper B",
            },
        ],
    }

    summaries = sorted(_summaries_from_payload(payload), key=lambda item: item.file_name)

    assert len(summaries) == 2
    assert summaries[0].file_name == "a.pdf"
    assert summaries[0].title == "Paper A"
    assert summaries[0].chunk_count == 2
    assert summaries[0].doi == "10.1000/a"
    assert summaries[1].file_name == "b.pdf"
    assert summaries[1].chunk_count == 1


def test_selector_matches_fuzzy_fields() -> None:
    assert _matches_selector("paper", "/papers/a.pdf", "a.pdf", "A Useful Paper", "10.1000/a")
    assert _matches_selector("1000/a", "/papers/a.pdf", "a.pdf", "A Useful Paper", "10.1000/a")
    assert _matches_selector("papers/a", "/papers/a.pdf", "a.pdf", "A Useful Paper", "10.1000/a")
