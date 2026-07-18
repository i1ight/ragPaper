# rag-paper

**Local-first paper RAG for PDF research workflows.**

rag-paper helps you build a private, local, searchable paper library from PDF files. It parses PDFs, chunks paper text, creates embeddings, stores vectors in local Chroma, and exposes retrieval through a CLI and MCP server for tools such as Codex CLI and Claude Code.

It is designed for researchers, students, engineers, and LLM power users who want to read papers efficiently while saving tokens and using affordable models such as DeepSeek, Qwen, local Ollama embeddings, or OpenAI-compatible embedding services.

[中文 README](./README.zh-CN.md)

## Keywords

paper RAG, local RAG, PDF RAG, academic search, Chroma, MCP server, Codex CLI, Claude Code, Zotero, Obsidian, citation graph, Mermaid, DOI enrichment, CrossRef, OpenAlex, semantic deduplication, local vector database, research assistant, low-cost LLM workflow

## Why rag-paper

- **Token-efficient paper reading**: retrieve only relevant chunks instead of sending whole PDFs to an LLM.
- **Local-first storage**: vectors and metadata are stored locally in Chroma.
- **MCP ready**: expose paper search tools to Codex CLI, Claude Code, and other MCP clients.
- **Low-cost model friendly**: use local Ollama embeddings by default, or an OpenAI-compatible embedding endpoint.
- **Zotero friendly**: point `root_path` to one or more Zotero storage/export directories.
- **Obsidian friendly**: export citation graphs as Mermaid Markdown.
- **Privacy controls**: use `skip_marker_file` to prevent sensitive folders from being indexed.

## Requirements

- Python **3.10+**
- A local or remote embedding provider
- Default embedding setup: Ollama with `qwen3-embedding:4b`

```bash
ollama pull qwen3-embedding:4b
```

## Installation

```bash
git clone https://github.com/your-name/rag-paper.git
cd rag-paper
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

Windows PowerShell:

```powershell
git clone https://github.com/your-name/rag-paper.git
Set-Location rag-paper
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e .
```

## Quick Start

Create a config file:

```bash
rag-paper init-config --path ./config.json
```

Put PDFs under `./papers`, or edit `config.json` and point `papers[].root_path` to your own directories.

Index PDFs:

```bash
rag-paper index
```

Search locally:

```bash
rag-paper search "retrieval augmented generation evaluation" --top-k 5
```

Inspect indexed papers:

```bash
rag-paper list-indexed-papers
rag-paper show-indexed-paper "attention" --limit 3
```

Delete a bad indexed record:

```bash
rag-paper delete-indexed-paper "10.55277/researchhub.x9vnpm0y.1"
rag-paper build-citation-graph
```

Start the MCP server:

```bash
rag-paper serve
```

By default this starts the MCP service with `streamable-http` on `http://127.0.0.1:8765/mcp`.
For stdio clients, start it with `rag-paper serve --transport stdio`.

## Tech Stack

- **Python 3.10+**: typed CLI application and local service runtime.
- **Click + Rich**: cross-platform command-line interface, confirmation prompts, and terminal tables.
- **PyMuPDF**: PDF text extraction and PDF metadata reading.
- **ChromaDB**: local persistent vector database for paper chunks.
- **Ollama embeddings**: default local embedding provider using `qwen3-embedding:4b`.
- **OpenAI-compatible embeddings**: optional remote embedding backend.
- **Rank BM25**: keyword retrieval path for hybrid search.
- **httpx**: CrossRef/OpenAlex metadata requests with HTTP, HTTPS, and SOCKS5 proxy support.
- **SQLite**: metadata enrichment cache to reduce repeated provider requests.
- **MCP Python SDK**: exposes paper search and inspection tools to Codex CLI, Claude Code, and other MCP clients.
- **Pydantic**: configuration validation and migration for older config shapes.
- **structlog**: structured logs for indexing, enrichment, search, and failures.

## Modules

### Indexing

`rag-paper index` scans configured PDF roots, respects `skip_marker_file`, extracts text, chunks papers, creates embeddings, and writes chunks to Chroma. Before vectorization it shows the number of files and the root paths they came from unless `--yes` or `indexing.assume_yes` is enabled.

Indexing is incremental. rag-paper stores `size + mtime_ns` in the manifest and only computes SHA256 when the quick file signature changes. Failed files are recorded in a JSONL log and can be retried with `rag-paper index --retry-failed`.

`rag-paper index --update-metadata-only` does **not** re-vectorize. It refreshes each indexed chunk's metadata (title, DOI, authors, …) from `paper_metadata.json` in place, so changes from `enrich-metadata` become visible to search without recomputing embeddings. Run it after `enrich-metadata`; chunk text (including abstract-chunk text) is not changed.

### Search and Retrieval

`rag-paper search` performs local hybrid retrieval over indexed chunks. It combines vector similarity from Chroma with BM25 keyword matching, then returns compact source-aware excerpts that are suitable for sending to an LLM instead of whole PDFs.

Search supports filters such as author, year, tag, and file name. The same retrieval engine powers the MCP tools.

Optionally enable a reranker (`reranker.enabled`) to re-score the top candidates by a continuous P(yes) relevance score read from the model's `yes`/`no` token logprobs (Qwen3-Reranker style, e.g. `dengcao/Qwen3-Reranker-4B` via Ollama); relevant chunks are promoted above spurious lexical/vector matches. It adds latency proportional to the candidate count, so keep `reranker.top_k` modest.

### Indexed Paper Inspection

`rag-paper list-indexed-papers` shows the current Chroma library as a terminal table, including total paper count, chunk count, title, DOI, year, and source path.

`rag-paper show-indexed-paper` supports fuzzy selectors over title, file name, source path, and DOI. It shows merged metadata and the first 5 chunk IDs by default; use `--all-chunks` to show every chunk ID.

`rag-paper delete-indexed-paper` deletes matching papers from Chroma and the index manifest. Without `--yes`, every matched paper must be confirmed one by one before deletion. After deleting indexed papers, run `rag-paper build-citation-graph` to refresh citation graph exports.

### Metadata Enrichment

`rag-paper enrich-metadata` enriches indexed papers with DOI and bibliographic metadata. It supports CrossRef and OpenAlex, provider fallback, rate limiting, custom User-Agent, contact email, and HTTP/HTTPS/SOCKS5 proxies.

The enrichment module is decoupled from vectorization. It can run per indexed file, after indexing finishes, or manually. Results are written to `paper_metadata.json` and provider responses are cached in SQLite.

Title quality checks reject obvious spam, URLs, ad-like strings, and symbol-heavy PDF metadata titles. When a PDF title is not trusted, rag-paper prefers a title inferred from the first page, a provider title, or the file name.

### Deduplication

`rag-paper dedupe-papers` reports duplicate candidates before indexing. It can compare DOI/title-year metadata and optional semantic signatures built from paper text or abstracts. Depending on configuration, duplicates can be reported or skipped.

### Citation Graph

`rag-paper build-citation-graph` builds a graph from enriched DOI/OpenAlex metadata and exports JSON plus Mermaid Markdown. The Mermaid output is designed to work well in Obsidian for browsing paper relationships.

### MCP Server

`rag-paper serve` starts an MCP server so external tools can query the local paper library. The server is **query-only**: clients can search chunks, find chunks by metadata, list and inspect indexed papers, fetch chunks by id, and export context. Importing, enriching, deduplicating, deleting, and citation-graph builds are CLI-only — the MCP server never mutates the library.

## Typical Workflows

### Use with Zotero

`root_path` is an array, so you can point rag-paper at multiple paper folders, including Zotero storage/export folders:

```json
{
  "papers": [
    {
      "root_path": [
        "D:/Zotero/storage",
        "D:/Zotero/exports/LLM"
      ],
      "skip_marker_file": ".rag-paper-skip",
      "tags": ["zotero"]
    }
  ]
}
```

rag-paper recursively scans these roots and indexes only `.pdf` files.

### Protect private folders with `skip_marker_file`

If a directory contains the configured marker file, rag-paper skips that directory and all of its children.

This is useful for privacy protection. For example, you can place `.rag-paper-skip` in folders containing unpublished papers, private notes, or papers that should not be exposed through MCP search.

```json
{
  "papers": [
    {
      "root_path": ["./papers"],
      "skip_marker_file": ".rag-paper-skip"
    }
  ]
}
```

When a marker is detected, rag-paper highlights the warning and asks whether to continue unless `--yes` or `indexing.assume_yes` is enabled.

### Back up and restore work

By default, core runtime data is stored under:

```text
rag_paper_data/
  chroma_db/
  paper_metadata.json
  cache/
  citation_graph/
  logs/
```

Copying `rag_paper_data/` is enough to back up the local Chroma vectors, metadata, cache, failure logs, retrieval stats, and citation graph exports when default paths are used.

To restore work on another device:

1. Install rag-paper.
2. Copy `rag_paper_data/` into the project directory.
3. Copy your `config.json` if you customized paths.
4. Run `rag-paper list-indexed-papers` to verify the restored index.

### Build a citation graph for Obsidian

After metadata enrichment, build a citation graph:

```bash
rag-paper build-citation-graph
```

rag-paper exports:

- JSON graph: `rag_paper_data/citation_graph/citation_graph.json`
- Mermaid Markdown: `rag_paper_data/citation_graph/citation_graph.md`

The Mermaid file can be opened directly in Obsidian or any Markdown tool with Mermaid support.

## Metadata Enrichment

rag-paper can enrich indexed papers with DOI and bibliographic metadata.

Supported providers:

- CrossRef
- OpenAlex

Default order:

```json
{
  "metadata_enrichment": {
    "providers": ["crossref", "openalex"]
  }
}
```

If the first provider fails or returns no match, rag-paper falls back to the next provider.

Run enrichment manually:

```bash
rag-paper enrich-metadata
```

Refresh existing metadata:

```bash
rag-paper enrich-metadata --force
```

Refresh one indexed PDF:

```bash
rag-paper enrich-metadata --file /path/to/paper.pdf --force
```

`--file` first checks whether the PDF has already been indexed in local Chroma. If not, rag-paper exits and asks you to index it first.

Re-check DOIs already in metadata and correct mismatches:

```bash
rag-paper enrich-metadata --reverify
```

`--reverify` re-checks every paper that already has a DOI. The stored DOI is treated as unverified and re-checked against the paper's title and year (see below); if it no longer matches, it is dropped and a fresh title search tries to replace it. If no confident replacement is found, the untrusted enrichment fields are **cleared** (DOI/title/authors/… removed, file/tags kept) instead of being left wrong — the run reports a `Cleared files` count. Corrections and clears are logged (`metadata.reverify_corrected`, `metadata.cleared_unmatched`). `--force` does the same re-derivation unconditionally for every paper (it re-derives, not just refreshes). Run either after upgrading, or whenever you suspect wrong associations.

### Title verification and association safety

To avoid grafting a citation's metadata onto the wrong paper, every lookup is checked against the document's title:

- DOIs typed into `paper_metadata.json` (`doi` / `url` / `external_url`) are trusted as-is.
- DOIs mined out of PDF body text are *speculative* — they are kept only if the returned work's title is similar enough to the paper's title.
- Title-search best matches from CrossRef/OpenAlex, and every cache hit, are checked the same way; a low-similarity match is rejected rather than written.
- During `--reverify` / `--force`, the reference title is taken from the **filename** (Zotero `Authors - Year - Title.pdf`), falling back to the PDF first page — never from the stored title (that would be circular). The filename year is cross-checked against the matched work's year (≈2-year slack for preprint→published lag), which rejects near-namesakes (e.g. *BERT* vs *Spectrum-BERT*) when the years diverge.
- If a stored DOI fails these checks and no confident replacement is found, the enrichment fields are cleared (see `--reverify` above) so wrong data does not persist.

Residual limitation: a different paper with a deliberately similar title *and* a close or missing year (a true near-namesake) can still slip through and may need a manual edit.

Relevant thresholds (see Configuration Reference):

- `min_title_similarity` (default `0.6`): title-similarity threshold for accepting a match.
- `min_title_score` (default `3.0`): minimum CrossRef relevance score.
- `min_openalex_score` (default `0.5`): minimum OpenAlex relevance score.

Metadata enrichment uses a SQLite cache by default:

```text
rag_paper_data/cache/metadata_enrichment.sqlite3
```

This avoids repeatedly calling CrossRef/OpenAlex for the same DOI or title query.

When a provider returns an abstract (OpenAlex), it is indexed as its own searchable chunk alongside the paper's body chunks, so abstract-level matches surface in `search_papers` / `rag-paper search`. Abstracts are written during indexing (per-file/after-index enrichment) and via `enrich-metadata`; papers enriched before this feature need a re-index or re-enrich to pick up the abstract chunk.

## MCP Usage

Start the MCP server:

```bash
rag-paper serve
```

rag-paper supports both MCP transports below:

- `streamable-http`: enabled by default through `mcp.transport` in `config.json`; use this for long-running HTTP MCP clients.
- `stdio`: supported for process-managed MCP clients such as `agent-infra/mcp-hub`; pass `--transport stdio` so startup status is written to stderr and stdout remains reserved for MCP protocol messages.

Streamable HTTP configuration example:

```json
{
  "mcpServers": {
    "rag-paper": {
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

stdio configuration example:

```json
{
  "mcpServers": {
    "rag-paper": {
      "command": "rag-paper",
      "args": ["serve", "--transport", "stdio"]
    }
  }
}
```

Available MCP tools (all read-only):

- `search_papers` — hybrid BM25 + vector search over chunks; supports `group_by_paper` (best chunk per paper) and `max_text_chars` (compact snippets)
- `search_by_metadata` — find chunks by author/year/tag/file name; supports `group_by_paper` and `max_text_chars` like `search_papers`
- `get_chunk` — fetch one chunk by id
- `export_context` — export selected chunks as a Markdown context block
- `list_indexed_papers` — list indexed papers
- `show_indexed_paper` — inspect a paper's metadata and chunk ids
- `get_paper_citations` — locally-resolved references (outgoing + incoming) for one paper
- `service_info` — service configuration

The server exposes no write tools. Use the CLI (`rag-paper index`, `enrich-metadata`, `dedupe-papers`, `delete-indexed-paper`, `build-citation-graph`) for anything that mutates the library.

## CLI Commands

```bash
rag-paper init-config
rag-paper index
rag-paper index --force
rag-paper index --file /path/to/paper.pdf
rag-paper index --only-new
rag-paper index --retry-failed
rag-paper enrich-metadata
rag-paper list-indexed-papers
rag-paper show-indexed-paper "selector"
rag-paper delete-indexed-paper "selector"
rag-paper search "query"
rag-paper dedupe-papers
rag-paper build-citation-graph
rag-paper serve
```

## Notes on Indexing

rag-paper stores indexing state in:

```text
rag_paper_data/chroma_db/index_manifest.json
```

It uses a two-stage change check:

1. Compare `size + mtime_ns`.
2. Compute SHA256 only when the quick check changed.

If indexing fails for a PDF, rag-paper records the failure in:

```text
rag_paper_data/logs/index_failed.jsonl
```

Retry failed files:

```bash
rag-paper index --retry-failed
```

If you press `Ctrl+C`, completed files are already persisted in Chroma and the manifest, so the next run continues from the remaining files.

## Development

```bash
pip install -e ".[dev]"
python -m pytest -q
```

## License

This project is licensed under the MIT License. See [LICENSE](./LICENSE).

## Configuration Reference

Common options:

- `data_dir`: core runtime data directory. Default: `./rag_paper_data`
- `papers[].root_path`: array of PDF root directories. Useful with Zotero folders.
- `papers[].skip_marker_file`: marker filename used to skip private directories.
- `papers[].tags`: default tags for papers under the root paths.
- `chroma.persist_dir`: Chroma persistence directory.
- `indexing.metadata_path`: paper metadata JSON path.
- `indexing.assume_yes`: skip interactive confirmation.
- `indexing.max_files`: maximum PDFs to index.
- `indexing.failed_path`: index failure JSONL path.
- `metadata_enrichment.enabled`: enable DOI and metadata enrichment.
- `metadata_enrichment.providers`: provider order, e.g. `["crossref", "openalex"]`.
- `metadata_enrichment.timing`: `per_file`, `after_index`, or `manual`.
- `metadata_enrichment.user_agent`: User-Agent for CrossRef/OpenAlex requests.
- `metadata_enrichment.mailto`: contact email for provider etiquette.
- `metadata_enrichment.openalex_email`: OpenAlex email parameter.
- `metadata_enrichment.requests_per_second`: provider request rate limit.
- `metadata_enrichment.http_proxy`: HTTP proxy.
- `metadata_enrichment.https_proxy`: HTTPS proxy.
- `metadata_enrichment.socks5_proxy`: SOCKS5 proxy.
- `metadata_enrichment.cache_path`: SQLite enrichment cache path.
- `metadata_enrichment.min_title_score`: minimum CrossRef relevance score for a title match. Default: `3.0`.
- `metadata_enrichment.min_openalex_score`: minimum OpenAlex relevance score for a title match. Default: `0.5`.
- `metadata_enrichment.min_title_similarity`: minimum title similarity (0–1) for accepting a DOI/title match and preventing wrong associations. Default: `0.6`.
- `reranker.enabled`: enable cross-encoder reranking of search results. Default: `false`.
- `reranker.model`: reranker model (Ollama), e.g. `dengcao/Qwen3-Reranker-4B:Q5_K_M`.
- `reranker.top_k`: number of first-stage candidates to rerank per query. Default: `20`.
- `reranker.concurrency`: parallel reranker requests. Default: `4` (raise Ollama's `OLLAMA_NUM_PARALLEL` to match).
- `reranker.top_logprobs`: number of top candidate tokens sampled for the P(yes) score. Default: `10`.
- `retrieval.fusion`: how vector and BM25 results are fused — `rrf` (default; reciprocal rank fusion, robust to score distributions) or `linear` (weighted score, uses `vector_weight`/`bm25_weight`).
- `retrieval.rrf_k`: RRF smoothing constant. Default: `60`.
- `dedup.enabled`: enable duplicate report before indexing.
- `dedup.action`: `report` or `skip`.
- `dedup.similarity_threshold`: semantic duplicate threshold.
- `citation_graph.path`: citation graph JSON output.
- `citation_graph.mermaid_path`: Mermaid Markdown output for Obsidian.
- `display.datetime_timezone`: display timezone for `metadata_enriched_at`.
- `display.datetime_format`: `strftime` format for displayed datetimes.
- `logging.level`: log level.
- `logging.stats_path`: retrieval stats JSONL path.
