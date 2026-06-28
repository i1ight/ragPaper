from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rag_paper.config import AppConfig
from rag_paper.indexer import discover_configured_pdfs
from rag_paper.metadata import load_paper_metadata, metadata_for_pdf


@dataclass(frozen=True)
class CitationGraphSummary:
    nodes: int
    edges: int
    path: str
    mermaid_path: str


def build_citation_graph(config: AppConfig) -> CitationGraphSummary:
    metadata_map = load_paper_metadata(config.metadata_path)
    candidates = discover_configured_pdfs(config)

    nodes: dict[str, dict[str, Any]] = {}
    doi_to_node: dict[str, str] = {}
    openalex_to_node: dict[str, str] = {}
    path_to_node: dict[str, str] = {}

    for candidate in candidates:
        metadata = metadata_for_pdf(metadata_map, candidate.path)
        node_id = _node_id(candidate.path, metadata)
        nodes[node_id] = _node_payload(node_id, candidate.path, metadata, indexed=True)
        path_to_node[str(candidate.path)] = node_id
        doi = _normalized_doi(metadata.get("doi"))
        if doi:
            doi_to_node[doi] = node_id
        openalex_id = _string(metadata.get("openalex_id"))
        if openalex_id:
            openalex_to_node[openalex_id] = node_id

    edges: list[dict[str, str]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for candidate in candidates:
        metadata = metadata_for_pdf(metadata_map, candidate.path)
        source_id = path_to_node[str(candidate.path)]

        for referenced_doi in _string_list(metadata.get("referenced_dois")):
            target_id = doi_to_node.get(_normalized_doi(referenced_doi))
            if target_id is None and config.citation_graph.include_external_nodes:
                target_id = f"doi:{_normalized_doi(referenced_doi)}"
                nodes.setdefault(
                    target_id,
                    {
                        "id": target_id,
                        "doi": _normalized_doi(referenced_doi),
                        "indexed": False,
                    },
                )
            if target_id:
                _append_edge(edges, seen_edges, source_id, target_id, "references")

        for referenced_work in _string_list(metadata.get("referenced_work_ids")):
            target_id = openalex_to_node.get(referenced_work)
            if target_id is None and config.citation_graph.include_external_nodes:
                target_id = referenced_work
                nodes.setdefault(
                    target_id,
                    {
                        "id": target_id,
                        "openalex_id": referenced_work,
                        "indexed": False,
                    },
                )
            if target_id:
                _append_edge(edges, seen_edges, source_id, target_id, "references")

    payload = {
        "nodes": list(nodes.values()),
        "edges": edges,
    }
    config.citation_graph_path.parent.mkdir(parents=True, exist_ok=True)
    with config.citation_graph_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
        f.write("\n")

    config.citation_graph_mermaid_path.parent.mkdir(parents=True, exist_ok=True)
    with config.citation_graph_mermaid_path.open("w", encoding="utf-8") as f:
        f.write(_to_mermaid(nodes, edges))

    return CitationGraphSummary(
        nodes=len(nodes),
        edges=len(edges),
        path=str(config.citation_graph_path),
        mermaid_path=str(config.citation_graph_mermaid_path),
    )


def _node_id(path: Path, metadata: dict[str, Any]) -> str:
    doi = _normalized_doi(metadata.get("doi"))
    if doi:
        return f"doi:{doi}"
    openalex_id = _string(metadata.get("openalex_id"))
    if openalex_id:
        return openalex_id
    return f"path:{path.resolve()}"


def _node_payload(
    node_id: str,
    path: Path,
    metadata: dict[str, Any],
    *,
    indexed: bool,
) -> dict[str, Any]:
    return {
        "id": node_id,
        "source_path": str(path),
        "file_name": path.name,
        "doi": _normalized_doi(metadata.get("doi")),
        "openalex_id": _string(metadata.get("openalex_id")),
        "title": _string(metadata.get("title")),
        "authors": _string_list(metadata.get("authors")),
        "year": metadata.get("year"),
        "cited_by_count": metadata.get("cited_by_count"),
        "indexed": indexed,
    }


def _append_edge(
    edges: list[dict[str, str]],
    seen_edges: set[tuple[str, str, str]],
    source_id: str,
    target_id: str,
    relation: str,
) -> None:
    key = (source_id, target_id, relation)
    if source_id == target_id or key in seen_edges:
        return
    seen_edges.add(key)
    edges.append({"source": source_id, "target": target_id, "relation": relation})


def _string(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _string_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item.strip() for item in value if isinstance(item, str) and item.strip()]
    if isinstance(value, str) and value.strip():
        return [item.strip() for item in value.split(",") if item.strip()]
    return []


def _normalized_doi(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    value = value.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
    return value


def _to_mermaid(nodes: dict[str, dict[str, Any]], edges: list[dict[str, str]]) -> str:
    lines = ["```mermaid", "graph TD"]
    for node_id, node in nodes.items():
        lines.append(f"  {_mermaid_id(node_id)}[\"{_escape_mermaid_label(_node_label(node))}\"]")
    for edge in edges:
        lines.append(
            "  "
            f"{_mermaid_id(edge['source'])} --> {_mermaid_id(edge['target'])}"
        )
    lines.append("```")
    lines.append("")
    return "\n".join(lines)


def _mermaid_id(value: str) -> str:
    sanitized = "".join(char if char.isalnum() else "_" for char in value)
    if not sanitized or sanitized[0].isdigit():
        sanitized = f"n_{sanitized}"
    return sanitized


def _node_label(node: dict[str, Any]) -> str:
    title = _string(node.get("title"))
    if title:
        return title
    doi = _string(node.get("doi"))
    if doi:
        return doi
    openalex_id = _string(node.get("openalex_id"))
    if openalex_id:
        return openalex_id.rsplit("/", 1)[-1]
    return _string(node.get("id"))


def _escape_mermaid_label(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")
