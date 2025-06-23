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


def setup_logging(level: Optional[str] = None) -> None:
    """Set up logging configuration for the application.

    Args:
        level: If provided, override all 'tonal_recall' log levels with this level (e.g., "DEBUG").
    """
    global _console_handler

    # Create a single, shared console handler if it doesn't exist
    if _console_handler is None:
        _console_handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        _console_handler.setFormatter(formatter)

    # Determine log levels
    log_levels = MODULE_LOG_LEVELS.copy()
    if level:
        numeric_level = logging.getLevelName(level.upper())
        if isinstance(numeric_level, int):
            for module_name in log_levels:
                if module_name.startswith("tonal_recall"):
                    log_levels[module_name] = numeric_level
        else:
            logging.getLogger(__name__).error(f"Invalid log level: {level}")

    # Apply module-specific levels
    for module_name, module_level in log_levels.items():
        logger = logging.getLogger(module_name if module_name else "")
        logger.setLevel(module_level)

        # Clear existing handlers and add the shared one
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        if _console_handler not in logger.handlers:
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
