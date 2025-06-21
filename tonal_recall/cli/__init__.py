"""Command-line interface for Tonal Recall."""

# Import CLI modules for easier access
from .note_detector_cli import main as legacy_cli_main
from .note_detector_cli_refactored import main as refactored_cli_main

__all__ = ["legacy_cli_main", "refactored_cli_main"]
