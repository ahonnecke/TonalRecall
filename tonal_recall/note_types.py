"""Type definitions for the Tonal Recall project."""

from typing import Optional
from dataclasses import dataclass


@dataclass
class NotePosition:
    """Represents a position on the guitar fretboard."""

    string: int  # String number (1-X, where 0 is the thinnest string)
    fret: int  # Fret number (0 for open string)

    def __str__(self):
        return f"S{self.string}F{self.fret}"


@dataclass
class DetectedNote:
    """Represents a detected musical note with its properties."""

    note_name: str  # Note name (e.g., 'A4', 'C#')
    frequency: float  # Frequency in Hz
    confidence: float  # Detection confidence (0-1)
    signal: float  # Signal strength (0-1, e.g., max(abs(audio)))
    is_stable: bool  # Whether this is a stable note
    timestamp: float  # Timestamp when the note was detected
    position: Optional[NotePosition] = None  # Position on the fretboard, if known
    octave: Optional[int] = None  # e.g. 2
