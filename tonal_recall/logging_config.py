"""Centralized logging configuration for Tonal Recall.

This module provides a consistent way to configure logging across the application.
"""

import logging
import sys

# Log levels for different modules
# Each module must be explicitly listed with its desired log level
# No fallback to root logger level - each module must be explicitly configured
MODULE_LOG_LEVELS = {
    # Core modules
    "tonal_recall": logging.INFO,
    "tonal_recall.main": logging.INFO,
    # Game components
    "tonal_recall.note_matcher": logging.DEBUG,  # Set to DEBUG for detailed matching info
    "tonal_recall.note_detector": logging.WARNING,  # Typically noisy, keep at WARNING
    "tonal_recall.core": logging.INFO,
    "tonal_recall.ui": logging.WARNING,  # UI modules often noisy, keep at WARNING
    "tonal_recall.stats": logging.INFO,
    # Libraries/third-party (if needed)
    "aubio": logging.ERROR,
    "PIL": logging.ERROR,
}


def setup_logging() -> None:
    """Set up logging configuration for the application.

    This should be called as early as possible in the application startup.
    It configures all loggers listed in MODULE_LOG_LEVELS.

    Note: The level parameter has been removed as all levels are now
    explicitly defined in MODULE_LOG_LEVELS.
    """
    # Configure root logger to only show errors
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.ERROR)

    # Clear any existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create console handler with default formatter
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)-30s - %(levelname)-8s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)

    # Configure each module logger explicitly
    for module_name, module_level in MODULE_LOG_LEVELS.items():
        logger = logging.getLogger(module_name)
        logger.setLevel(module_level)

        # Clear existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

        # Add our console handler
        logger.addHandler(console_handler)
        logger.propagate = False

    # Log configuration complete
    logger = logging.getLogger("tonal_recall")
    logger.info("Logging configuration complete")


# Cache for loggers to avoid creating duplicate handlers
_logger_cache = {}


def get_logger(name: str) -> logging.Logger:
    """Get a logger with the given name.

    The logger name must be explicitly listed in MODULE_LOG_LEVELS.
    This is intentional to ensure all loggers are explicitly configured.

    Args:
        name: The full module name (e.g., 'tonal_recall.note_matcher')

    Returns:
        A configured logger instance

    Raises:
        ValueError: If the module name is not in MODULE_LOG_LEVELS
    """
    # Return cached logger if it exists
    if name in _logger_cache:
        return _logger_cache[name]

    # Verify the module is explicitly configured
    if name not in MODULE_LOG_LEVELS:
        raise ValueError(
            f"Logger '{name}' not found in MODULE_LOG_LEVELS. "
            "Please add it to the configuration."
        )

    # Get the logger
    logger = logging.getLogger(name)

    # Skip if this logger is already configured
    if logger.handlers:
        _logger_cache[name] = logger
        return logger

    # Set the log level from configuration
    logger.setLevel(MODULE_LOG_LEVELS[name])

    # Create a console handler if one doesn't exist
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)-30s - %(levelname)-8s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # Disable propagation to prevent duplicate logs
    logger.propagate = False

    # Cache the logger
    _logger_cache[name] = logger

    return logger
