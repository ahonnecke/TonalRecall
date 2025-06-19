"""Core interfaces for the Tonal Recall application.

This module defines the interfaces that components must implement to be compatible
with the Tonal Recall architecture. These interfaces establish a clear contract
between components and allow for different implementations to be swapped in and out.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any, Protocol

from ..note_types import DetectedNote


class INoteDetector(ABC):
    """Interface for note detection components."""
    
    @abstractmethod
    def process_audio(self, audio_data, timestamp: float) -> Optional[DetectedNote]:
        """Process audio data and detect notes.
        
        Args:
            audio_data: Audio data as numpy array
            timestamp: Current timestamp in seconds
            
        Returns:
            DetectedNote if a stable note is detected, None otherwise
        """
        pass
    
    @abstractmethod
    def get_current_note(self) -> Optional[DetectedNote]:
        """Get the current detected note.
        
        Returns:
            The current detected note or None if no note is detected
        """
        pass
    
    @abstractmethod
    def set_callback(self, callback: Optional[Callable]) -> None:
        """Set the callback function for note detection events.
        
        Args:
            callback: Callback function or None to remove
        """
        pass
    
    @abstractmethod
    def set_sample_rate(self, sample_rate: int) -> None:
        """Update the sample rate.
        
        Args:
            sample_rate: New sample rate in Hz
        """
        pass


class IAudioInput(ABC):
    """Interface for audio input components."""
    
    @abstractmethod
    def start(self, callback: Callable) -> bool:
        """Start audio input.
        
        Args:
            callback: Function to call with audio data
            
        Returns:
            True if started successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop audio input."""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if audio input is running.
        
        Returns:
            True if running, False otherwise
        """
        pass


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
    
    @abstractmethod
    def configure_detector(self, **params) -> None:
        """Configure the note detector with the given parameters.
        
        Args:
            **params: Parameters to set on the note detector
        """
        pass
    
    @abstractmethod
    def get_detector_config(self) -> Dict[str, Any]:
        """Get the current note detector configuration.
        
        Returns:
            Dictionary of parameter names and values
        """
        pass


class INoteDetectionCallback(Protocol):
    """Protocol for note detection callbacks."""
    
    def __call__(self, note: DetectedNote, timestamp: float) -> None:
        """Called when a note is detected.
        
        Args:
            note: The detected note
            timestamp: The timestamp when the note was detected
        """
        pass


class IConfigurable(Protocol):
    """Protocol for configurable components."""
    
    def configure(self, **params) -> None:
        """Configure the component with the given parameters.
        
        Args:
            **params: Parameters to set on the component
        """
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get the current component configuration.
        
        Returns:
            Dictionary of parameter names and values
        """
        pass
