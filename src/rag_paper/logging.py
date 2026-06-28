from __future__ import annotations

import logging
import sys

import structlog

from rag_paper.config import AppConfig


def configure_logging(config: AppConfig) -> None:
    processors: list[object] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    if config.logging.json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(structlog.dev.ConsoleRenderer())

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=getattr(logging, config.logging.level.upper(), logging.INFO),
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, config.logging.level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


logger = structlog.get_logger("rag_paper")
