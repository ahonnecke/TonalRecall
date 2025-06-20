"""Core interfaces for the Tonal Recall application.

This module defines the interfaces that components must implement to be compatible
with the Tonal Recall architecture. These interfaces establish a clear contract
between components and allow for different implementations to be swapped in and out.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any, Protocol

from ..note_types import DetectedNote


class INoteDetectionService(ABC):
    """Interface for note detection service."""
    
    @abstractmethod
    def start(self, callback: Callable[[DetectedNote, float], None]) -> None:
        """Start note detection.
        
        Args:
            callback: Function to call when a note is detected
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop note detection."""
        pass
    
    @abstractmethod
    def get_current_note(self) -> Optional[DetectedNote]:
        """Get the current detected note.
        
        Returns:
            The current detected note or None if no note is detected
        """
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if note detection is running.
        
        Returns:
            True if running, False otherwise
        """
        pass
    
