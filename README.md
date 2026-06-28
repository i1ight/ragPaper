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

Start the MCP server:

```bash
rag-paper serve
```

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

Metadata enrichment uses a SQLite cache by default:

```text
rag_paper_data/cache/metadata_enrichment.sqlite3
```

This avoids repeatedly calling CrossRef/OpenAlex for the same DOI or title query.

## MCP Usage

Start the MCP server:

```bash
rag-paper serve
```

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
      "args": ["serve"]
    }
  }
}
```

Available MCP tools include:

- `service_info`
- `import_papers`
- `list_indexed_papers`
- `show_indexed_paper`
- `search_papers`
- `search_by_metadata`
- `get_chunk`
- `export_context`
- `enrich_paper_metadata`
- `dedupe_papers`
- `build_paper_citation_graph`

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
- `dedup.enabled`: enable duplicate report before indexing.
- `dedup.action`: `report` or `skip`.
- `dedup.similarity_threshold`: semantic duplicate threshold.
- `citation_graph.path`: citation graph JSON output.
- `citation_graph.mermaid_path`: Mermaid Markdown output for Obsidian.
- `display.datetime_timezone`: display timezone for `metadata_enriched_at`.
- `display.datetime_format`: `strftime` format for displayed datetimes.
- `logging.level`: log level.
- `logging.stats_path`: retrieval stats JSONL path.
