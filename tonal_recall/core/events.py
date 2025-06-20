"""Event system for Tonal Recall components."""

from typing import Dict, List, Callable, Any
from enum import Enum, auto

from ..logger import get_logger

logger = get_logger(__name__)


class NoteDetectionEventType(Enum):
    """Event types for note detection."""
    
    NOTE_DETECTED = auto()
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
                        
    def emit_note_detected(self, note, timestamp: float) -> None:
        """Emit a note detected event.
        
        Args:
            note: The detected note
            timestamp: The timestamp when the note was detected
        """
        self._emitter.emit(NoteDetectionEventType.NOTE_DETECTED, note, timestamp)
                        
    def clear(self) -> None:
        """Remove all event listeners."""
        self._emitter.clear()
