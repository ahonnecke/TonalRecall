"""Centralized logging configuration for Tonal Recall.

This module provides a consistent way to configure logging across the application.
"""

import logging
import sys


def setup_logging(level=logging.INFO):
    """Configure logging for the application.

    Args:
        level: Logging level (default: logging.INFO)
    """
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger("tonal_recall")
    root_logger.setLevel(level)

    # Add console handler if none exists
    if not any(isinstance(h, logging.StreamHandler) for h in root_logger.handlers):
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    return root_logger


def get_logger(name):
    """Get a logger with the given name, namespaced under 'tonal_recall'."""
    if not name.startswith("tonal_recall."):
        name = f"tonal_recall.{name}"
    return logging.getLogger(name)
