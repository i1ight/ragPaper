from __future__ import annotations

import math
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed

import httpx

from rag_paper.config import AppConfig, RerankerConfig
from rag_paper.logging import logger

# Official Qwen3-Reranker judge prompt: the model is trained to answer "yes"/"no".
JUDGE_PROMPT = (
    'Judge whether the Document meets the requirements based on the Query and the '
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)

# Cap document length so very long chunks don't blow up the prompt. Default
# chunks (~1200 chars) are well within this; it only guards oversized inputs.
MAX_DOC_CHARS = 4000


def format_prompt(instruction: str, query: str, document: str) -> str:
    """Build the raw Qwen3-Reranker prompt.

    Hand-assembled chat tokens + an empty ``<think>`` block so the model skips
    reasoning and its first generated token is the yes/no judgment. Sent with
    Ollama ``raw=true`` so the template is not re-applied.
    """
    document = (document or "")[:MAX_DOC_CHARS]
    prefix = f"<|im_start|>system\n{JUDGE_PROMPT}<|im_end|>\n<|im_start|>user\n"
    body = f"<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    return prefix + body + suffix


def score_from_logprobs(entry: dict | None) -> float | None:
    """Continuous relevance P(yes) from a generated token's top_logprobs.

    Aggregates the probability mass of all ``yes``/``no`` case variants among the
    top candidate tokens and returns ``p_yes / (p_yes + p_no)``. Returns None when
    neither token appears (caller falls back to reading the generated text).
    """
    if not isinstance(entry, dict):
        return None
    top = entry.get("top_logprobs")
    if not isinstance(top, list) or not top:
        return None
    p_yes = 0.0
    p_no = 0.0
    for item in top:
        if not isinstance(item, dict):
            continue
        token = (item.get("token") or "").strip().lower()
        logprob = item.get("logprob")
        if not isinstance(logprob, (int, float)):
            continue
        if token == "yes":
            p_yes += math.exp(logprob)
        elif token == "no":
            p_no += math.exp(logprob)
    if p_yes + p_no == 0.0:
        return None
    return p_yes / (p_yes + p_no)


def parse_yes_no(content: str | None) -> float:
    """Binary fallback: map a yes/no text answer to 1.0/0.0.

    Used only when the logprobs path yields no yes/no token (rare). Unparseable
    output counts as not relevant (0.0).
    """
    if not content:
        return 0.0
    stripped = content.strip()
    if not stripped:
        return 0.0
    first = stripped.split()[0].lower().strip(".,;:!?)\"'")
    if first == "yes":
        return 1.0
    if first == "no":
        return 0.0
    lowered = stripped.lower()
    return 1.0 if "yes" in lowered and "no" not in lowered else 0.0


class RerankerProvider(ABC):
    @abstractmethod
    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Score a batch of (query, document) pairs, returning one 0..1 per pair."""
        raise NotImplementedError


class OllamaRerankerProvider(RerankerProvider):
    """Reranker via Ollama /api/generate reading the P(yes) logprob (Qwen3-Reranker)."""

    def __init__(self, config: RerankerConfig) -> None:
        self.config = config
        self.client = httpx.Client(timeout=config.timeout_seconds)

    def close(self) -> None:
        self.client.close()

    def score_pairs(self, pairs: list[tuple[str, str]]) -> list[float]:
        if not pairs:
            return []
        scores: list[float] = [0.0] * len(pairs)
        with ThreadPoolExecutor(max_workers=self.config.concurrency) as pool:
            future_to_index = {
                pool.submit(self._score_one, query, document): index
                for index, (query, document) in enumerate(pairs)
            }
            for future in as_completed(future_to_index):
                index = future_to_index[future]
                try:
                    scores[index] = future.result()
                except Exception as exc:  # noqa: BLE001 - degrade one pair, keep going
                    logger.warning("reranker.pair_failed", index=index, error=str(exc))
                    scores[index] = 0.0
        return scores

    def _score_one(self, query: str, document: str) -> float:
        body = {
            "model": self.config.model,
            "prompt": format_prompt(self.config.instruction, query, document),
            "raw": True,
            "stream": False,
            "logprobs": True,
            "top_logprobs": self.config.top_logprobs,
            "options": {"num_predict": 1, "temperature": 0},
        }
        response = self.client.post(
            f"{self.config.base_url.rstrip('/')}/api/generate", json=body
        )
        response.raise_for_status()
        data = response.json()
        logprobs = data.get("logprobs") or []
        entry = logprobs[0] if logprobs else None
        score = score_from_logprobs(entry)
        if score is not None:
            return score
        # Fallback: no yes/no in top_logprobs; read the generated text.
        return parse_yes_no(data.get("response"))


def build_reranker_provider(config: AppConfig) -> RerankerProvider | None:
    rc = config.reranker
    if not rc.enabled:
        return None
    if rc.provider == "ollama":
        return OllamaRerankerProvider(rc)
    raise ValueError(f"Unknown reranker provider: {rc.provider}")
