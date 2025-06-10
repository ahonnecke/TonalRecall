"""Type definitions for the Tonal Recall project."""

from typing import TypedDict, List, Callable, Any, Optional
from dataclasses import dataclass
from enum import Enum


@dataclass
class NotePosition:
    """Represents a position on the guitar fretboard."""

    string: int  # String number (0-5, where 0 is the thickest string)
    fret: int  # Fret number (0 for open string)

    def __str__(self):
        return f"S{self.string}F{self.fret}"


@dataclass
class DetectedNote:
    """Represents a detected musical note with its properties."""

    name: str  # Note name (e.g., 'A4', 'C#3')
    frequency: float  # Frequency in Hz
    confidence: float  # Detection confidence (0-1)
    signal: float  # Signal strength (0-1, e.g., max(abs(audio)))
    is_stable: bool  # Whether this is a stable note
    timestamp: float  # Timestamp when the note was detected
    position: Optional[NotePosition] = None  # Position on the fretboard, if known


class NoteDetectorConfig(TypedDict, total=False):
    """Configuration options for the NoteDetector class."""

    sample_rate: int
    frames_per_buffer: int
    channels: int
    min_confidence: float
    min_signal: float
    min_frequency: float
    max_frequency: float
    stability_window: int
    stability_threshold: float


class NoteGameDifficulty(Enum):
    """Difficulty levels for the note game."""

    SINGLE_NOTE = 0
    OPEN_STRINGS = 1
    WHOLE_NOTES = 2
    HALF_NOTES = 3


# Audio callback type for the note detector
AudioCallback = Callable[[List[float], int, float, Any], None]
