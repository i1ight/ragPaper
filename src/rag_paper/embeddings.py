from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

import httpx

from rag_paper.config import AppConfig


class EmbeddingProvider(ABC):
    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]


class OllamaEmbeddingProvider(EmbeddingProvider):
    def __init__(self, base_url: str, model: str, timeout_seconds: int) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout_seconds

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        with httpx.Client(timeout=self.timeout) as client:
            for text in texts:
                response = client.post(
                    f"{self.base_url}/api/embeddings",
                    json={"model": self.model, "prompt": text},
                )
                response.raise_for_status()
                payload = response.json()
                embeddings.append(payload["embedding"])
        return embeddings


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(self, base_url: str, api_key: str, model: str, timeout_seconds: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout_seconds

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        with httpx.Client(timeout=self.timeout, headers=headers) as client:
            response = client.post(
                f"{self.base_url}/embeddings",
                json={"model": self.model, "input": texts},
            )
            response.raise_for_status()
            payload: dict[str, Any] = response.json()
            return [item["embedding"] for item in sorted(payload["data"], key=lambda x: x["index"])]


def build_embedding_provider(config: AppConfig) -> EmbeddingProvider:
    if config.embedding.provider == "ollama":
        return OllamaEmbeddingProvider(
            base_url=config.ollama.base_url,
            model=config.ollama.model,
            timeout_seconds=config.ollama.timeout_seconds,
        )

    api_key = os.getenv(config.embedding.openai.api_key_env)
    if not api_key:
        raise RuntimeError(
            f"Missing API key env var: {config.embedding.openai.api_key_env}"
        )
    return OpenAICompatibleEmbeddingProvider(
        base_url=config.embedding.openai.base_url,
        api_key=api_key,
        model=config.embedding.openai.model,
    )

