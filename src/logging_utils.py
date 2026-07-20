"""Logging helpers for the Ames project."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(log_level: int = logging.INFO, log_file: Path | None = None) -> None:
    """Configure a consistent logging format for CLI scripts and the app."""

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )

