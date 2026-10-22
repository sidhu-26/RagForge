"""Structured logging setup for RAGForge.

Configures standard library logging with sanitized, structured outputs without
leaking sensitive tokens, secrets, or internal stack traces to public handlers.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from ragforge.core.config import Settings


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured production logs."""

    def format(self, record: logging.LogRecord) -> str:
        log_obj: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include contextual metadata if provided in 'extra'
        record_extra = getattr(record, "extra", None)
        if isinstance(record_extra, dict):
            # Filter out sensitive keys if any
            sensitive_keys = {"authorization", "secret", "password", "token", "api_key"}
            clean_extra = {
                k: v for k, v in record_extra.items() if str(k).lower() not in sensitive_keys
            }
            log_obj["context"] = clean_extra

        if record.exc_info:
            log_obj["exception"] = self.formatException(record.exc_info)

        return json.dumps(log_obj)


def setup_logging(settings: Settings) -> None:
    """Configure root and application loggers based on settings."""
    log_level = getattr(logging, settings.LOG_LEVEL.value, logging.INFO)

    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Clear existing handlers to prevent duplicate lines
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)

    if settings.is_production:
        console_handler.setFormatter(JSONFormatter())
    else:
        text_format = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
        console_handler.setFormatter(logging.Formatter(text_format))

    root_logger.addHandler(console_handler)

    # Set third-party loggers to a reasonable level
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("uvicorn.error").setLevel(logging.INFO)
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
