from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, clear_contextvars

_LOGGING_CONFIGURED = False


def configure_logging(*, level: str = "info", fmt: str = "text", force: bool = False) -> None:
    global _LOGGING_CONFIGURED
    if _LOGGING_CONFIGURED and not force:
        return

    renderer = (
        structlog.processors.JSONRenderer()
        if fmt.lower() == "json"
        else structlog.dev.ConsoleRenderer(colors=False)
    )

    shared_processors: list[Any] = [
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(normalize_log_level(level))

    structlog.reset_defaults()
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    _LOGGING_CONFIGURED = True


def normalize_log_level(level: str) -> int:
    return getattr(logging, level.strip().upper(), logging.INFO)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)


def bind_log_context(**values: object) -> None:
    bind_contextvars(**values)


def clear_log_context() -> None:
    clear_contextvars()
