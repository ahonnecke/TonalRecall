import logging
from collections import deque
from typing import Optional, Deque

from ..note_types import DetectedNote
from ..note_utils import get_note_name

logger = logging.getLogger(__name__)


class StabilityAnalyzer:
    """
    Analyzes a stream of detected notes to determine when a stable note is being played.
    """

    def __init__(
        self,
        min_frequency: float,
        min_confidence: float,
        min_signal: float,
        min_stable_count: int,
        group_hz: float,
        history_size: int,
    ):
        self._min_frequency = min_frequency
        self._min_confidence = min_confidence
        self._min_signal = min_signal
        self._min_stable_count = min_stable_count
        self._group_hz = group_hz

        self._note_history: Deque[DetectedNote] = deque(maxlen=history_size)
        self._stable_note: Optional[DetectedNote] = None
        self._last_stable_note: Optional[str] = None

    def add_note(self, note: DetectedNote):
        """Adds a new detected note to the history for analysis."""
        self._note_history.append(note)

    def get_stable_note(self) -> Optional[DetectedNote]:
        """
        Determine if there's a stable note in the history.

        Returns:
            A DetectedNote if a stable note is found, None otherwise.
        """
        valid_notes = [
            n
            for n in self._note_history
            if n.frequency >= self._min_frequency
            and getattr(n, "confidence", 1.0) >= self._min_confidence
            and getattr(n, "signal_max", 1.0) >= self._min_signal
        ]

        if len(valid_notes) < self._min_stable_count:
            self._stable_note = None
            return None

        freq_groups = []
        for note in valid_notes:
            found_group = False
            for group in freq_groups:
                group_avg = sum(n.frequency for n in group) / len(group)
                if abs(note.frequency - group_avg) < self._group_hz:
                    group.append(note)
                    found_group = True
                    break
            if not found_group:
                freq_groups.append([note])

        if not freq_groups:
            self._stable_note = None
            return None

        freq_groups.sort(key=len, reverse=True)
        largest_group = freq_groups[0]

        avg_freq = sum(n.frequency for n in largest_group) / len(largest_group)
        avg_confidence = sum(n.confidence for n in largest_group) / len(largest_group)
        avg_signal = sum(n.signal_max for n in largest_group) / len(largest_group)
        most_recent = max(largest_group, key=lambda n: n.timestamp)

        new_note = DetectedNote(
            note_name=get_note_name(avg_freq),
            frequency=avg_freq,
            confidence=avg_confidence,
            signal_max=avg_signal,
            is_stable=True,
            timestamp=most_recent.timestamp,
        )

        if self._stable_note and new_note.note_name == self._stable_note.note_name:
            self._stable_note.frequency = new_note.frequency
            self._stable_note.confidence = new_note.confidence
            self._stable_note.signal_max = new_note.signal_max
            self._stable_note.timestamp = new_note.timestamp
            return self._stable_note
        else:
            self._stable_note = new_note
            return new_note
