# rag-paper

**本地优先的论文 PDF RAG 工具。**

rag-paper 可以把本地 PDF 论文解析、切分、向量化并保存到本地 Chroma，再通过 CLI 和 MCP Server 提供检索能力，适合 Codex CLI、Claude Code 等工具调用。

它的目标是：在 token 和预算有限时，尽量只把相关论文片段交给模型，从而配合 DeepSeek、Qwen、Ollama、本地模型或兼容 OpenAI Embedding 的服务高效阅读论文。

[English README](./README.md)

## 关键词

论文 RAG、本地 RAG、PDF RAG、Chroma、MCP Server、Codex CLI、Claude Code、Zotero、Obsidian、citation graph、Mermaid、DOI 补全、CrossRef、OpenAlex、语义去重、本地向量数据库、低成本 LLM 工作流

## 环境要求

- Python **3.10+**
- 默认使用 Ollama + `qwen3-embedding:4b`

```bash
ollama pull qwen3-embedding:4b
```

## 安装

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

## 快速开始

生成配置：

```bash
rag-paper init-config --path ./config.json
```

把 PDF 放到 `./papers`，或修改 `config.json` 中的 `papers[].root_path`。

建立索引：

```bash
rag-paper index
```

搜索论文：

```bash
rag-paper search "retrieval augmented generation evaluation" --top-k 5
```

查看已索引论文：

```bash
rag-paper list-indexed-papers
rag-paper show-indexed-paper "attention" --limit 3
```

删除错误的已索引记录：

```bash
rag-paper delete-indexed-paper "10.55277/researchhub.x9vnpm0y.1"
rag-paper build-citation-graph
```

启动 MCP：

```bash
rag-paper serve
```

## 技术栈

- **Python 3.10+**：类型化 CLI 应用和本地服务运行时。
- **Click + Rich**：跨平台命令行、确认提示和终端表格展示。
- **PyMuPDF**：PDF 文本抽取和 PDF 内置 metadata 读取。
- **ChromaDB**：本地持久化向量数据库，用于保存论文 chunk。
- **Ollama Embeddings**：默认本地向量模型，使用 `qwen3-embedding:4b`。
- **OpenAI-compatible Embeddings**：可选远程 embedding 后端。
- **Rank BM25**：关键词检索路径，用于混合检索。
- **httpx**：访问 CrossRef/OpenAlex，支持 HTTP、HTTPS、SOCKS5 代理。
- **SQLite**：metadata enrichment 缓存，减少重复请求。
- **MCP Python SDK**：向 Codex CLI、Claude Code 等 MCP 客户端暴露检索工具。
- **Pydantic**：配置校验和旧配置结构迁移。
- **structlog**：索引、补全、检索和失败记录的结构化日志。

## 模块功能

### 索引模块

`rag-paper index` 会扫描配置中的 PDF 根目录，遵守 `skip_marker_file`，抽取 PDF 文本，切分 chunk，生成 embedding，并写入 Chroma。开始向量化前会展示待处理文件数量和对应根路径，除非使用 `--yes` 或启用 `indexing.assume_yes`。

索引是增量式的。rag-paper 会在 manifest 中记录 `size + mtime_ns`，只有快速文件签名变化时才计算 SHA256。索引失败的文件会写入 JSONL 失败记录，可用 `rag-paper index --retry-failed` 重试。

`rag-paper index --update-metadata-only` **不重新向量化**：直接用 `paper_metadata.json` 刷新每个已索引 chunk 的元数据（标题、DOI、作者……），让 `enrich-metadata` 的改动对检索可见而无需重算 embedding。建议在 `enrich-metadata` 之后运行；不会改动 chunk 正文（含摘要 chunk 的文本）。

### 检索模块

`rag-paper search` 在本地已索引 chunk 上执行混合检索。它结合 Chroma 向量相似度和 BM25 关键词匹配，返回适合交给 LLM 的精简上下文，避免把整篇 PDF 都发送给模型。

检索支持作者、年份、标签和文件名过滤。MCP Server 内部也复用同一套检索能力。

可选开启 reranker（`reranker.enabled`），用**连续 P(yes) 相关性分数**（从 `yes`/`no` token 的 logprobs 读取，Qwen3-Reranker 风格，如经 Ollama 调用 `dengcao/Qwen3-Reranker-4B`）对一阶段 top 候选重排，把真正相关的 chunk 提到词面/向量误匹配之上。重排会增加与候选数成正比的延迟，请把 `reranker.top_k` 控制在合理范围。

### 已索引论文查看与删除

`rag-paper list-indexed-papers` 以表格展示 Chroma 中已有论文，包括总论文数、chunk 数、标题、DOI、年份和来源路径。

`rag-paper show-indexed-paper` 支持对标题、文件名、来源路径和 DOI 进行模糊查询。默认只展示前 5 个 Chunk IDs，可用 `--all-chunks` 展示全部。

`rag-paper delete-indexed-paper` 会从 Chroma 和 index manifest 中删除匹配论文。未使用 `--yes` 时，无论匹配数量多少，都需要逐篇确认后才删除。删除后建议运行 `rag-paper build-citation-graph` 刷新引用图。

### DOI 和元数据补全模块

`rag-paper enrich-metadata` 会为已索引论文补全 DOI 和文献信息。它支持 CrossRef、OpenAlex、接口 fallback、访问速率控制、自定义 User-Agent、联系邮箱，以及 HTTP/HTTPS/SOCKS5 代理。

补全模块与向量化模块解耦。它可以在每篇论文向量化后执行、全部索引结束后执行，也可以手动执行。补全结果写入 `paper_metadata.json`，接口返回结果会缓存到 SQLite。

title 质量校验会拒绝明显广告、URL、特殊符号密度过高、疑似垃圾 PDF metadata title。PDF 内置 title 不可信时，会优先使用首页推断标题、DOI provider 标题或文件名。

### 去重模块

`rag-paper dedupe-papers` 用于索引前生成重复论文报告。它可以基于 DOI、title-year metadata 判断，也可以使用摘要或正文片段构建语义签名进行近似去重。根据配置，重复论文可以仅报告，也可以跳过。

### Citation Graph 模块

`rag-paper build-citation-graph` 会基于补全后的 DOI/OpenAlex 元数据构建引用图，并导出 JSON 和 Mermaid Markdown。Mermaid 输出适合直接放入 Obsidian 查看论文引用关系。

### MCP Server 模块

`rag-paper serve` 会启动 MCP Server，让外部工具查询本地论文库。该服务**仅供查询**：客户端可以检索 chunk、按 metadata 查找 chunk、列出与查看已索引论文、按 id 获取 chunk、导出上下文、查询论文引用关系（出向/入向，解析到本地）。导入、补全、去重、删除、构建引用图等写操作仅限 CLI，MCP 服务不会改动论文库。

## Zotero 与 root_path

`root_path` 是数组，可以同时指定多个论文根目录，适合直接指向 Zotero storage 或导出目录：

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

rag-paper 会递归扫描这些路径，只导入 `.pdf` 文件。

## skip_marker_file 与隐私保护

如果某个目录下存在配置的 `skip_marker_file`，rag-paper 会跳过该目录和所有子目录。

这可以用于保护隐私，例如：

- 未公开论文
- 私人笔记
- 不希望通过 MCP 暴露的 PDF

示例：

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

检测到 marker 时，rag-paper 会高亮提示；如果没有使用 `--yes` 或 `indexing.assume_yes`，会询问是否继续。

## 备份与迁移

默认核心数据目录是：

```text
rag_paper_data/
  chroma_db/
  paper_metadata.json
  cache/
  citation_graph/
  logs/
```

使用默认路径时，复制 `rag_paper_data/` 即可备份 Chroma 向量库、metadata、缓存、失败记录、检索统计和 citation graph。

在其他设备恢复：

1. 安装 rag-paper。
2. 复制 `rag_paper_data/` 到项目目录。
3. 如果自定义过路径，也复制 `config.json`。
4. 运行 `rag-paper list-indexed-papers` 验证。

## Obsidian 与 Citation Graph

补全元数据后可以构建引用图：

```bash
rag-paper build-citation-graph
```

默认导出：

- JSON: `rag_paper_data/citation_graph/citation_graph.json`
- Mermaid Markdown: `rag_paper_data/citation_graph/citation_graph.md`

Mermaid 文件可以直接放入 Obsidian 使用。

## DOI 和元数据补全

支持：

- CrossRef
- OpenAlex

默认顺序：

```json
{
  "metadata_enrichment": {
    "providers": ["crossref", "openalex"]
  }
}
```

手动补全：

```bash
rag-paper enrich-metadata
```

强制刷新：

```bash
rag-paper enrich-metadata --force
```

只补全某篇已索引 PDF：

```bash
rag-paper enrich-metadata --file /path/to/paper.pdf --force
```

重新校验已存 DOI 并纠正错配：

```bash
rag-paper enrich-metadata --reverify
```

`--reverify` 会重新检查每篇已有 DOI 的论文：把已存 DOI 当作未验证，重新与论文标题和年份比对（见下）；若不再匹配，则丢弃并用标题查询尝试重新匹配。若找不到可信替代，则**清空**这些不可信的补全字段（移除 DOI/标题/作者…，保留文件/标签），而不是保留错值——运行结果会报告 `Cleared files` 数。纠正与清空分别以 `metadata.reverify_corrected`、`metadata.cleared_unmatched` 记录日志。`--force` 行为相同，但对所有论文无条件重新派生（是重派生，而非简单刷新）。建议升级后或怀疑关联有误时运行。

### 标题校验与关联安全

为避免把某条引用的元数据错误关联到别的论文，每次查询都会与文档标题比对：

- 手写在 `paper_metadata.json` 里的 DOI（`doi` / `url` / `external_url` 字段）视为可信，直接采用。
- 从 PDF 正文里挖出的 DOI 视为*投机*，仅当返回记录的标题与论文标题足够相似时才保留。
- CrossRef/OpenAlex 的标题查询最佳匹配、以及每次缓存命中，都按同样规则校验；相似度过低会被拒绝而不写入。
- `--reverify` / `--force` 时，参考标题取自**文件名**（Zotero `作者 - 年份 - 标题.pdf`），文件名解析不出则退回首屏正文——绝不取自已存标题（否则等于自我确认）。同时把文件名里的年份与命中记录的年份交叉校验（约 2 年容差，兼容预印本→正式发表），可挡住年份对不上的近同名（如 BERT vs Spectrum-BERT）。
- 若已存 DOI 未通过这些校验、又找不到可信替代，则清空相关补全字段（见上文 `--reverify`），不让错值残留。

残留局限：标题蓄意相似**且**年份贴边/缺失的"真·近同名"，仍可能溜过，需人工订正。

相关阈值（见配置参数参考）：

- `min_title_similarity`（默认 `0.6`）：接受匹配的标题相似度阈值。
- `min_title_score`（默认 `3.0`）：CrossRef 最低相关性分数。
- `min_openalex_score`（默认 `0.5`）：OpenAlex 最低相关性分数。

补全结果会写入 `paper_metadata.json`，请求结果会缓存到 SQLite，避免重复请求 CrossRef/OpenAlex。

当 provider 返回摘要（OpenAlex）时，摘要会被作为独立可检索 chunk 与正文 chunk 一起写入，这样 `search_papers` / `rag-paper search` 能命中"摘要级"匹配。摘要在索引期间（per_file/after_index 补全）以及 `enrich-metadata` 时写入；在本特性之前已补全的论文需重新索引或重新补全一次以生成摘要 chunk。

## MCP 配置示例

rag-paper 支持以下 MCP 传输方式：

- `streamable-http`：默认由 `config.json` 里的 `mcp.transport` 启用，适合长期运行的 HTTP MCP 客户端。
- `stdio`：支持 `agent-infra/mcp-hub` 这类由客户端托管进程的 MCP 客户端；启动时传入 `--transport stdio`，stdout 会保留给 MCP 协议消息。

Streamable HTTP：

```json
{
  "mcpServers": {
    "rag-paper": {
      "url": "http://127.0.0.1:8765/mcp"
    }
  }
}
```

stdio：

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

## 常用命令

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

## 开发

```bash
pip install -e ".[dev]"
python -m pytest -q
```

## License

本项目使用 MIT License。见 [LICENSE](./LICENSE)。

## 配置参数参考

- `data_dir`: 核心运行数据目录，默认 `./rag_paper_data`
- `papers[].root_path`: PDF 根目录数组，可结合 Zotero 使用
- `papers[].skip_marker_file`: 隐私保护跳过标记文件名
- `papers[].tags`: 默认标签
- `chroma.persist_dir`: Chroma 保存目录
- `indexing.metadata_path`: 论文元数据 JSON
- `indexing.assume_yes`: 跳过交互确认
- `indexing.max_files`: 最大索引 PDF 数
- `indexing.failed_path`: 索引失败记录
- `metadata_enrichment.enabled`: 是否启用 DOI 补全
- `metadata_enrichment.providers`: CrossRef/OpenAlex 顺序
- `metadata_enrichment.timing`: `per_file`、`after_index` 或 `manual`
- `metadata_enrichment.cache_path`: SQLite 补全缓存
- `metadata_enrichment.min_title_score`: CrossRef 标题匹配最低相关性分数，默认 `3.0`
- `metadata_enrichment.min_openalex_score`: OpenAlex 标题匹配最低相关性分数，默认 `0.5`
- `metadata_enrichment.min_title_similarity`: 接受 DOI/标题匹配的标题相似度阈值（0–1），用于防止错误关联，默认 `0.6`
- `reranker.enabled`: 是否启用 reranker 重排搜索结果，默认 `false`
- `reranker.model`: reranker 模型（Ollama），如 `dengcao/Qwen3-Reranker-4B:Q5_K_M`
- `reranker.top_k`: 每次查询重排的一阶段候选数，默认 `20`
- `reranker.concurrency`: 重排并发请求数，默认 `4`（需同时调高 Ollama 的 `OLLAMA_NUM_PARALLEL`）
- `reranker.top_logprobs`: 计算 P(yes) 时采样的 top 候选 token 数，默认 `10`
- `retrieval.fusion`: 向量与 BM25 结果融合方式——`rrf`（默认；按排名融合，不受分数分布影响）或 `linear`（加权分数，用 `vector_weight`/`bm25_weight`）
- `retrieval.rrf_k`: RRF 平滑常数，默认 `60`
- `dedup.enabled`: 是否启用去重报告
- `dedup.action`: `report` 或 `skip`
- `citation_graph.path`: citation graph JSON 输出
- `citation_graph.mermaid_path`: Mermaid 输出，适合 Obsidian
- `display.datetime_timezone`: 展示时间时区
- `display.datetime_format`: 展示时间格式
- `logging.level`: 日志级别
- `logging.stats_path`: 检索统计路径
