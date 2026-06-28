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

- **PyMuPDF**：PDF 文本抽取和 PDF 内置 metadata 读取。
- **ChromaDB**：本地持久化向量数据库，用于保存论文 chunk。
- **Rank BM25**：关键词检索路径，用于混合检索。
- **MCP Python SDK**：向 Codex CLI、Claude Code 等 MCP 客户端暴露检索工具。

## 模块功能

### 索引模块

`rag-paper index` 会扫描配置中的 PDF 根目录，遵守 `skip_marker_file`，抽取 PDF 文本，切分 chunk，生成 embedding，并写入 Chroma。开始向量化前会展示待处理文件数量和对应根路径，除非使用 `--yes` 或启用 `indexing.assume_yes`。

索引是增量式的。rag-paper 会在 manifest 中记录 `size + mtime_ns`，只有快速文件签名变化时才计算 SHA256。索引失败的文件会写入 JSONL 失败记录，可用 `rag-paper index --retry-failed` 重试。

### 检索模块

`rag-paper search` 在本地已索引 chunk 上执行混合检索。它结合 Chroma 向量相似度和 BM25 关键词匹配，返回适合交给 LLM 的精简上下文，避免把整篇 PDF 都发送给模型。

检索支持作者、年份、标签和文件名过滤。MCP Server 内部也复用同一套检索能力。

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

`rag-paper serve` 会启动 MCP Server，让外部工具查询本地论文库。MCP 客户端可以导入论文、检索 chunk、查看已索引论文、查看 metadata、删除错误索引、补全 metadata、去重、获取特定 chunk、导出上下文和构建 citation graph。

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

补全结果会写入 `paper_metadata.json`，请求结果会缓存到 SQLite，避免重复请求 CrossRef/OpenAlex。

## MCP 配置示例

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
      "args": ["serve"]
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
- `dedup.enabled`: 是否启用去重报告
- `dedup.action`: `report` 或 `skip`
- `citation_graph.path`: citation graph JSON 输出
- `citation_graph.mermaid_path`: Mermaid 输出，适合 Obsidian
- `display.datetime_timezone`: 展示时间时区
- `display.datetime_format`: 展示时间格式
- `logging.level`: 日志级别
- `logging.stats_path`: 检索统计路径
