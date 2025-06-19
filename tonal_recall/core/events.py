"""Event system for Tonal Recall components."""

from typing import Dict, List, Callable, Any, Optional
from enum import Enum, auto

from ..logger import get_logger

logger = get_logger(__name__)


class NoteDetectionEventType(Enum):
    """Event types for note detection."""
    
    NOTE_DETECTED = auto()
    NOTE_LOST = auto()
    AUDIO_LEVEL_CHANGED = auto()
    DETECTION_STARTED = auto()
    DETECTION_STOPPED = auto()
    ERROR = auto()


class EventEmitter:
    """Event emitter for Tonal Recall components."""
    
    def __init__(self):
        """Initialize the event emitter."""
        self._listeners: Dict[Any, List[Callable]] = {}
    
    def on(self, event_type: Any, callback: Callable) -> None:
        """Register a callback for an event type.
        
        Args:
            event_type: Event type to listen for
            callback: Function to call when the event occurs
        """
        if event_type not in self._listeners:
            self._listeners[event_type] = []
        
        if callback not in self._listeners[event_type]:
            self._listeners[event_type].append(callback)
            logger.debug(f"Added listener for event {event_type}")
    
    def off(self, event_type: Any, callback: Optional[Callable] = None) -> None:
        """Remove a callback for an event type.
        
        Args:
            event_type: Event type to remove listener from
            callback: Function to remove, or None to remove all listeners for the event type
        """
        if event_type not in self._listeners:
            return
        
        if callback is None:
            # Remove all listeners for this event type
            self._listeners[event_type] = []
            logger.debug(f"Removed all listeners for event {event_type}")
        else:
            # Remove specific listener
            if callback in self._listeners[event_type]:
                self._listeners[event_type].remove(callback)
                logger.debug(f"Removed listener for event {event_type}")
    
    def emit(self, event_type: Any, *args, **kwargs) -> None:
        """Emit an event.
        
        Args:
            event_type: Event type to emit
            *args: Positional arguments to pass to listeners
            **kwargs: Keyword arguments to pass to listeners
        """
        if event_type not in self._listeners:
            return
        
        for callback in self._listeners[event_type]:
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Error in event listener for {event_type}: {e}")
    
    def clear(self) -> None:
        """Remove all event listeners."""
        self._listeners = {}
        logger.debug("Cleared all event listeners")


class NoteDetectionEvents:
    """Event emitter specifically for note detection events."""
    
    def __init__(self):
        """Initialize the note detection events."""
        self._emitter = EventEmitter()
    
    def on_note_detected(self, callback: Callable) -> None:
        """Register a callback for note detection events.
        
        Args:
            callback: Function to call when a note is detected
        """
        self._emitter.on(NoteDetectionEventType.NOTE_DETECTED, callback)
    
    def on_note_lost(self, callback: Callable) -> None:
        """Register a callback for note lost events.
        
        Args:
            callback: Function to call when a note is lost
        """
        self._emitter.on(NoteDetectionEventType.NOTE_LOST, callback)
    
    def on_audio_level_changed(self, callback: Callable) -> None:
        """Register a callback for audio level changed events.
        
        Args:
            callback: Function to call when the audio level changes
        """
        self._emitter.on(NoteDetectionEventType.AUDIO_LEVEL_CHANGED, callback)
    
    def on_detection_started(self, callback: Callable) -> None:
        """Register a callback for detection started events.
        
        Args:
            callback: Function to call when detection starts
        """
        self._emitter.on(NoteDetectionEventType.DETECTION_STARTED, callback)
    
    def on_detection_stopped(self, callback: Callable) -> None:
        """Register a callback for detection stopped events.
        
        Args:
            callback: Function to call when detection stops
        """
        self._emitter.on(NoteDetectionEventType.DETECTION_STOPPED, callback)
    
    def on_error(self, callback: Callable) -> None:
        """Register a callback for error events.
        
        Args:
            callback: Function to call when an error occurs
        """
        self._emitter.on(NoteDetectionEventType.ERROR, callback)
    
    def emit_note_detected(self, note, timestamp: float) -> None:
        """Emit a note detected event.
        
        Args:
            note: The detected note
            timestamp: The timestamp when the note was detected
        """
        self._emitter.emit(NoteDetectionEventType.NOTE_DETECTED, note, timestamp)
    
    def emit_note_lost(self, note, timestamp: float) -> None:
        """Emit a note lost event.
        
        Args:
            note: The lost note
            timestamp: The timestamp when the note was lost
        """
        self._emitter.emit(NoteDetectionEventType.NOTE_LOST, note, timestamp)
    
    def emit_audio_level_changed(self, level: float, timestamp: float) -> None:
        """Emit an audio level changed event.
        
        Args:
            level: The new audio level
            timestamp: The timestamp when the level changed
        """
        self._emitter.emit(NoteDetectionEventType.AUDIO_LEVEL_CHANGED, level, timestamp)
    
    def emit_detection_started(self, timestamp: float) -> None:
        """Emit a detection started event.
        
        Args:
            timestamp: The timestamp when detection started
        """
        self._emitter.emit(NoteDetectionEventType.DETECTION_STARTED, timestamp)
    
    def emit_detection_stopped(self, timestamp: float) -> None:
        """Emit a detection stopped event.
        
        Args:
            timestamp: The timestamp when detection stopped
        """
        self._emitter.emit(NoteDetectionEventType.DETECTION_STOPPED, timestamp)
    
    def emit_error(self, error: Exception, timestamp: float) -> None:
        """Emit an error event.
        
        Args:
            error: The error that occurred
            timestamp: The timestamp when the error occurred
        """
        self._emitter.emit(NoteDetectionEventType.ERROR, error, timestamp)
    
    def clear(self) -> None:
        """Remove all event listeners."""
        self._emitter.clear()
