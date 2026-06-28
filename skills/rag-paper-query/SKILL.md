---
name: rag-paper-query
description: Use when the user asks to search, inspect, summarize, compare, cite, or reason over locally indexed paper PDFs in rag-paper via MCP tools or direct local Chroma/rag-paper access.
---

# rag-paper-query

Use this skill whenever the user asks about papers that may exist in the local rag-paper knowledge base.

## Priority

Prefer MCP tools when available. Use direct local commands or Python access only when MCP is unavailable, incomplete, or the user explicitly asks for direct Chroma/local access.

## MCP Workflow

If MCP server `rag-paper` is available, use these tools:

1. `list_indexed_papers`
   - Use first when the user asks what papers are indexed.
   - Use to confirm whether a paper exists before detailed lookup.

2. `show_indexed_paper`
   - Use for metadata, DOI, authors, source path, chunk IDs, or indexed-paper details.
   - Use fuzzy selectors: title fragment, filename fragment, source path fragment, or DOI fragment.

3. `search_papers`
   - Use for semantic or keyword-like questions over paper content.
   - Prefer concise, content-rich search queries.
   - Use filters when the user mentions author, year, tag, or file name.

4. `search_by_metadata`
   - Use when the user asks for papers by author, year, tag, or filename.

5. `get_chunk`
   - Use when exact chunk text is needed.

6. `export_context`
   - Use when synthesizing an answer from multiple chunks.
   - Export only relevant chunks.

## Direct Local Workflow

If MCP is unavailable, use local rag-paper commands from the project environment:

```bash
rag-paper list-indexed-papers
rag-paper show-indexed-paper "selector"
rag-paper search "query" --top-k 8
rag-paper search "query" --author "name"
rag-paper search "query" --year 2024
rag-paper search "query" --tag transformer
```

If commands are unavailable but Python package access works, use:

```python
from rag_paper.config import load_config
from rag_paper.retrieval import HybridRetriever, result_to_dict

config = load_config("./config.json")
retriever = HybridRetriever(config)
results = retriever.search("your query", top_k=8)
items = [result_to_dict(result) for result in results]
```

For direct Chroma inspection without semantic search:

```python
import chromadb
from rag_paper.config import load_config

config = load_config("./config.json")
client = chromadb.PersistentClient(path=str(config.chroma_dir))
collection = client.get_collection(config.chroma.collection)
payload = collection.get(include=["documents", "metadatas"])
```

## Answering Rules

- Do not answer from memory when the local paper index can be queried.
- Always cite the paper title or filename when using retrieved content.
- Include DOI when available.
- If multiple chunks support an answer, synthesize rather than dumping raw chunks.
- If the result is uncertain, say what was searched and what was not found.
- Do not mutate the index, enrich metadata, rebuild citation graphs, or run indexing unless the user explicitly asks.
- For broad questions, first search, then inspect the most relevant papers, then answer.
- For "what papers do I have about X", use search plus metadata listing.
- For "summarize this paper", use `show_indexed_paper` first, then retrieve chunks by title/file selector.
- For "compare papers", retrieve relevant chunks from each paper before comparing.
