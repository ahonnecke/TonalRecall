"""Core components for the Tonal Recall application."""

# Import interfaces for easier access
from .interfaces import (
    INoteDetector,
    IAudioInput,
    INoteDetectionService,
)

__all__ = ["INoteDetector", "IAudioInput", "INoteDetectionService"]
