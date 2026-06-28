from __future__ import annotations

from rag_paper.enrichment import (
    _merge_work_metadata,
    _work_from_crossref_message,
    _work_from_openalex_message,
    extract_doi,
    infer_title,
)


def test_extract_doi_trims_trailing_punctuation() -> None:
    assert extract_doi("DOI: 10.1145/3368089.3409702.") == "10.1145/3368089.3409702"


def test_infer_title_uses_metadata_before_text() -> None:
    assert infer_title({"title": "Metadata Title"}, "A Much Longer Text Title") == "Metadata Title"


def test_parse_and_merge_crossref_work() -> None:
    work = _work_from_crossref_message(
        {
            "DOI": "10.1000/test",
            "title": ["Example Paper"],
            "author": [{"given": "Ada", "family": "Lovelace"}],
            "issued": {"date-parts": [[2024, 1, 1]]},
            "container-title": ["ExampleConf"],
            "publisher": "Example Publisher",
            "URL": "https://doi.org/10.1000/test",
            "score": 10.5,
        }
    )

    assert work is not None
    metadata = _merge_work_metadata({"tags": ["rag"]}, work, force=False)

    assert metadata["doi"] == "10.1000/test"
    assert metadata["title"] == "Example Paper"
    assert metadata["authors"] == ["Ada Lovelace"]
    assert metadata["year"] == 2024
    assert metadata["tags"] == ["rag"]


def test_parse_openalex_work() -> None:
    work = _work_from_openalex_message(
        {
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.1000/openalex",
            "display_name": "OpenAlex Paper",
            "authorships": [{"author": {"display_name": "Grace Hopper"}}],
            "publication_year": 2025,
            "primary_location": {"source": {"display_name": "Journal"}},
            "abstract_inverted_index": {"hello": [0], "world": [1]},
            "referenced_works": ["https://openalex.org/W456"],
            "cited_by_count": 7,
        }
    )

    assert work is not None
    assert work.source == "openalex"
    assert work.doi == "10.1000/openalex"
    assert work.authors == ["Grace Hopper"]
    assert work.abstract == "hello world"
    assert work.referenced_work_ids == ["https://openalex.org/W456"]
    assert work.cited_by_count == 7
