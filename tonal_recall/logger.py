"""Centralized lazy-loading logger configuration for Tonal Recall."""
import logging
from typing import Optional, Dict, Any

# Module-level cache for loggers
_logger_cache: Dict[str, logging.Logger] = {}

def get_logger(name: str) -> logging.Logger:
    """
    Get a lazily initialized logger with the given name.
    
    Args:
        name: The full module name (e.g., 'tonal_recall.note_matcher')
        
    Returns:
        A configured logger instance
    """
    if name not in _logger_cache:
        logger = logging.getLogger(name)
        _logger_cache[name] = logger
    return _logger_cache[name]
