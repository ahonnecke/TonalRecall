"""Defines the core interfaces for the Tonal Recall application."""

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, Callable

import numpy as np

from ..note_types import DetectedNote


class IAudioInput(ABC):
    """Interface for audio input handlers."""

    @abstractmethod
    def start(self, callback: Callable[[np.ndarray, float], None]) -> bool:
        """Start capturing audio."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing audio."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if audio is running."""
        pass


class INoteDetector(ABC):
    """Interface for note detection algorithms."""

    @abstractmethod
    def process_audio(self, audio_data: np.ndarray, timestamp: float) -> Optional[DetectedNote]:
        """Process a chunk of audio data and return a detected note if found."""
        pass

    @abstractmethod
    def set_callback(self, callback: Optional[Callable[[DetectedNote, float], None]]) -> None:
        """Set a callback to be invoked when a note is detected."""
        pass


class INoteDetectionService(ABC):
    """Interface for the main note detection service."""

    @abstractmethod
    def start(self, callback: Callable[[DetectedNote, float], None]) -> bool:
        """Start the note detection service."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop the note detection service."""
        pass

    @abstractmethod
    def is_running(self) -> bool:
        """Check if the service is running."""
        pass
