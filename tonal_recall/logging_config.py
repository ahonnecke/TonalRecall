"""Centralized logging configuration for Tonal Recall.

This module provides a consistent way to configure logging across the application.
"""

import logging
import sys
from typing import Optional, Dict

# Log levels for different modules
MODULE_LOG_LEVELS = {
    # Core modules
    "tonal_recall": logging.INFO,
    "tonal_recall.main": logging.INFO,
    # Game components
    "tonal_recall.note_matcher": logging.INFO,  # Set to DEBUG for detailed matching info
    "tonal_recall.note_detector": logging.DEBUG,
    "tonal_recall.core": logging.INFO,
    "tonal_recall.ui": logging.WARNING,  # UI modules often noisy, keep at WARNING
    "tonal_recall.stats": logging.INFO,
    "tonal_recall.logger": logging.WARNING,  # Logger module itself should be quiet
    # Libraries/third-party
    "aubio": logging.ERROR,
    "PIL": logging.ERROR,
    # Root logger
    "": logging.ERROR,
}

# Shared console handler
_console_handler: Optional[logging.Handler] = None

# Cache for loggers to avoid duplicate setup
_logger_cache: Dict[str, logging.Logger] = {}


def setup_logging(force_level: Optional[int] = None) -> None:
    """Set up logging configuration for the application.

    Args:
        force_level: If provided, override all log levels with this level.
    """
    global _console_handler

    # Configure root logger to show only errors
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR)

    # Clear all existing handlers and disable propagation for all loggers
    for logger_name in logging.root.manager.loggerDict:
        logger = logging.getLogger(logger_name)
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        logger.propagate = False

    # Clear root handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler with a formatter
    _console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    _console_handler.setFormatter(formatter)

    # Apply log levels
    if force_level is not None:
        # Apply forced level to all loggers
        for logger_name in logging.root.manager.loggerDict:
            logger = logging.getLogger(logger_name)
            logger.setLevel(force_level)
            # Only add handler if this is one of our loggers
            if logger_name.startswith("tonal_recall"):
                logger.addHandler(_console_handler)
                logger.propagate = False
    else:
        # Apply module-specific levels
        for module_name, module_level in MODULE_LOG_LEVELS.items():
            logger = logging.getLogger(
                module_name if module_name else None
            )  # Empty string for root
            logger.setLevel(module_level)
            logger.addHandler(_console_handler)
            logger.propagate = False

    # Confirm setup complete
    logging.getLogger("tonal_recall").info("Logging configuration complete")


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    The logger name must be explicitly listed in MODULE_LOG_LEVELS.

    Args:
        name: The full module name (e.g., 'tonal_recall.note_matcher')

    Returns:
        A configured logger instance

    Raises:
        ValueError: If the module name is not in MODULE_LOG_LEVELS
    """
    global _console_handler

    if name in _logger_cache:
        return _logger_cache[name]

    if name not in MODULE_LOG_LEVELS:
        raise ValueError(
            f"Logger '{name}' not found in MODULE_LOG_LEVELS. "
            "Please add it to the configuration."
        )

    logger = logging.getLogger(name)
    logger.setLevel(MODULE_LOG_LEVELS[name])

    # Only add handler if not already attached
    if _console_handler and not any(
        isinstance(h, type(_console_handler)) for h in logger.handlers
    ):
        logger.addHandler(_console_handler)

    logger.propagate = False
    _logger_cache[name] = logger
    return logger
