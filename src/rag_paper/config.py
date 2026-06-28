from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, field_validator, model_validator

MetadataProviderName = Literal["crossref", "openalex"]


class OllamaConfig(BaseModel):
    base_url: str = "http://127.0.0.1:11434"
    model: str = "qwen3-embedding:4b"
    timeout_seconds: int = 120


class OpenAIEmbeddingConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "text-embedding-3-large"


class EmbeddingConfig(BaseModel):
    provider: Literal["ollama", "openai"] = "ollama"
    dimension: int | None = 2560
    openai: OpenAIEmbeddingConfig = Field(default_factory=OpenAIEmbeddingConfig)


class PaperRootConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    root_path: list[str] = Field(
        default_factory=lambda: ["./papers"],
        validation_alias=AliasChoices("root_path", "pdf_dir", "path"),
        serialization_alias="root_path",
    )
    skip_marker_file: str = ""
    tags: list[str] = Field(default_factory=list)

    @field_validator("root_path", mode="before")
    @classmethod
    def normalize_root_path(cls, value: object) -> object:
        if isinstance(value, Path):
            return [str(value)]
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(item) if isinstance(item, Path) else item for item in value]
        return value

    @property
    def root_paths(self) -> list[str]:
        return self.root_path


class ChromaConfig(BaseModel):
    persist_dir: str = "./rag_paper_data/chroma_db"
    collection: str = "papers"


class IndexingConfig(BaseModel):
    chunk_size: int = 1200
    chunk_overlap: int = 180
    metadata_path: str = "./rag_paper_data/paper_metadata.json"
    assume_yes: bool = False
    max_files: int | None = Field(default=None, ge=0)
    failed_path: str = "./rag_paper_data/logs/index_failed.jsonl"


class MetadataEnrichmentConfig(BaseModel):
    enabled: bool = True
    provider: MetadataProviderName = "crossref"
    providers: list[MetadataProviderName] = Field(
        default_factory=lambda: ["crossref", "openalex"]
    )
    timing: Literal["per_file", "after_index", "manual"] = "per_file"
    base_url: str = "https://api.crossref.org"
    openalex_base_url: str = "https://api.openalex.org"
    user_agent: str = "rag-paper/0.1 (mailto:your-email@example.com)"
    mailto: str = ""
    openalex_email: str = ""
    timeout_seconds: int = 20
    requests_per_second: float = Field(default=1.0, gt=0)
    http_proxy: str = ""
    https_proxy: str = ""
    socks5_proxy: str = ""
    min_title_score: float = Field(default=0.0, ge=0)
    max_query_chars: int = Field(default=300, gt=0)
    cache_path: str = "./rag_paper_data/cache/metadata_enrichment.sqlite3"

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_provider(cls, value: object) -> object:
        if isinstance(value, dict) and "provider" in value and "providers" not in value:
            migrated = dict(value)
            provider = migrated.get("provider")
            if provider == "openalex":
                migrated["providers"] = ["openalex", "crossref"]
            elif provider == "crossref":
                migrated["providers"] = ["crossref", "openalex"]
            return migrated
        return value

    @field_validator("providers", mode="before")
    @classmethod
    def normalize_providers(cls, value: object) -> object:
        if value is None:
            return ["crossref", "openalex"]
        if isinstance(value, str):
            return [value]
        return value


class DedupConfig(BaseModel):
    enabled: bool = False
    semantic_enabled: bool = True
    action: Literal["report", "skip"] = "report"
    report_path: str = "./rag_paper_data/logs/duplicate_papers.json"
    similarity_threshold: float = Field(default=0.92, ge=0, le=1)
    signature_chars: int = Field(default=3000, gt=0)
    max_files: int | None = Field(default=None, ge=0)


class CitationGraphConfig(BaseModel):
    path: str = "./rag_paper_data/citation_graph/citation_graph.json"
    mermaid_path: str = "./rag_paper_data/citation_graph/citation_graph.md"
    include_external_nodes: bool = True


class DisplayConfig(BaseModel):
    datetime_timezone: str = "local"
    datetime_format: str = ""


class RetrievalConfig(BaseModel):
    vector_top_k: int = 12
    bm25_top_k: int = 12
    default_top_k: int = 8
    vector_weight: float = 0.65
    bm25_weight: float = 0.35


class McpConfig(BaseModel):
    host: str = "127.0.0.1"
    port: int = 8765
    transport: Literal["stdio", "sse", "streamable-http"] = "streamable-http"

    @property
    def endpoint_path(self) -> str | None:
        if self.transport == "streamable-http":
            return "/mcp"
        if self.transport == "sse":
            return "/sse"
        return None

    @property
    def endpoint_url(self) -> str | None:
        if self.endpoint_path is None:
            return None
        return f"http://{self.host}:{self.port}{self.endpoint_path}"


class LoggingConfig(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    level: str = "INFO"
    json_format: bool = Field(default=True, alias="json")
    stats_path: str = "./rag_paper_data/logs/retrieval_stats.jsonl"


class AppConfig(BaseModel):
    data_dir: str = "./rag_paper_data"
    ollama: OllamaConfig = Field(default_factory=OllamaConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    papers: list[PaperRootConfig] = Field(default_factory=lambda: [PaperRootConfig()])
    chroma: ChromaConfig = Field(default_factory=ChromaConfig)
    indexing: IndexingConfig = Field(default_factory=IndexingConfig)
    metadata_enrichment: MetadataEnrichmentConfig = Field(
        default_factory=MetadataEnrichmentConfig
    )
    dedup: DedupConfig = Field(default_factory=DedupConfig)
    citation_graph: CitationGraphConfig = Field(default_factory=CitationGraphConfig)
    display: DisplayConfig = Field(default_factory=DisplayConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    mcp: McpConfig = Field(default_factory=McpConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @model_validator(mode="before")
    @classmethod
    def apply_data_dir_defaults(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        data_dir = str(value.get("data_dir") or "./rag_paper_data").rstrip("/")
        normalized = dict(value)

        chroma = dict(normalized.get("chroma") or {})
        chroma.setdefault("persist_dir", f"{data_dir}/chroma_db")
        normalized["chroma"] = chroma

        indexing = dict(normalized.get("indexing") or {})
        indexing.setdefault("metadata_path", f"{data_dir}/paper_metadata.json")
        indexing.setdefault("failed_path", f"{data_dir}/logs/index_failed.jsonl")
        normalized["indexing"] = indexing

        enrichment = dict(normalized.get("metadata_enrichment") or {})
        enrichment.setdefault("cache_path", f"{data_dir}/cache/metadata_enrichment.sqlite3")
        normalized["metadata_enrichment"] = enrichment

        dedup = dict(normalized.get("dedup") or {})
        dedup.setdefault("report_path", f"{data_dir}/logs/duplicate_papers.json")
        normalized["dedup"] = dedup

        citation_graph = dict(normalized.get("citation_graph") or {})
        citation_graph.setdefault("path", f"{data_dir}/citation_graph/citation_graph.json")
        citation_graph.setdefault("mermaid_path", f"{data_dir}/citation_graph/citation_graph.md")
        normalized["citation_graph"] = citation_graph

        logging_config = dict(normalized.get("logging") or {})
        logging_config.setdefault("stats_path", f"{data_dir}/logs/retrieval_stats.jsonl")
        normalized["logging"] = logging_config

        return normalized

    @field_validator("papers", mode="before")
    @classmethod
    def normalize_papers(cls, value: object) -> object:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            return [{"root_path": value}]
        if isinstance(value, dict):
            roots = value.get("roots")
            if roots is not None:
                return roots

            skip_marker_file = value.get("skip_marker_file", "")
            tags = value.get("tags", [])
            normalized_roots: list[dict[str, object]] = []
            pdf_dir = value.get("pdf_dir", "./papers")
            if pdf_dir:
                normalized_roots.append(
                    {
                        "root_path": pdf_dir,
                        "skip_marker_file": skip_marker_file,
                        "tags": tags,
                    }
                )

            zetro_database_dir = value.get("zetro_database_dir")
            if zetro_database_dir:
                normalized_roots.append(
                    {
                        "root_path": zetro_database_dir,
                        "skip_marker_file": skip_marker_file,
                        "tags": tags,
                    }
                )

            return normalized_roots or [{"root_path": "./papers"}]
        return value

    @property
    def pdf_dir(self) -> Path:
        return Path(self.papers[0].root_paths[0]).expanduser().resolve()

    @property
    def zetro_database_dir(self) -> Path | None:
        return None

    @property
    def paper_dirs(self) -> list[Path]:
        return [
            Path(root_path).expanduser().resolve()
            for root in self.papers
            for root_path in root.root_paths
        ]

    @property
    def chroma_dir(self) -> Path:
        return Path(self.chroma.persist_dir).expanduser().resolve()

    @property
    def metadata_path(self) -> Path:
        return Path(self.indexing.metadata_path).expanduser().resolve()

    @property
    def index_failed_path(self) -> Path:
        return Path(self.indexing.failed_path).expanduser().resolve()

    @property
    def stats_path(self) -> Path:
        return Path(self.logging.stats_path).expanduser().resolve()

    @property
    def dedup_report_path(self) -> Path:
        return Path(self.dedup.report_path).expanduser().resolve()

    @property
    def citation_graph_path(self) -> Path:
        return Path(self.citation_graph.path).expanduser().resolve()

    @property
    def citation_graph_mermaid_path(self) -> Path:
        return Path(self.citation_graph.mermaid_path).expanduser().resolve()

    @property
    def metadata_cache_path(self) -> Path:
        return Path(self.metadata_enrichment.cache_path).expanduser().resolve()


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path).expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        return AppConfig.model_validate(json.load(f))


def write_default_config(path: str | Path) -> Path:
    config_path = Path(path).expanduser().resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(AppConfig().model_dump(by_alias=True), f, indent=2, ensure_ascii=False)
        f.write("\n")
    return config_path
