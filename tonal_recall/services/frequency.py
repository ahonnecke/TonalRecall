#!/usr/bin/env python3

import numpy as np
import logging
from typing import (
    Optional,
    Callable,
    List,
    Dict,
    Any,
    ClassVar,
    TypeAlias,
)

logger = logging.getLogger(__name__)


class FrequencyService:
    # Type aliases
    NoteName: TypeAlias = str
    Frequency: TypeAlias = float
    Confidence: TypeAlias = float
    SignalStrength: TypeAlias = float

    # Note names and music theory constants
    SHARP_NOTES: ClassVar[List[NoteName]] = [
        "C",
        "C#",
        "D",
        "D#",
        "E",
        "F",
        "F#",
        "G",
        "G#",
        "A",
        "A#",
        "B",
    ]

    # Mapping between sharp and flat note names
    SHARP_TO_FLAT: ClassVar[Dict[NoteName, NoteName]] = {
        "C#": "Db",
        "D#": "Eb",
        "F#": "Gb",
        "G#": "Ab",
        "A#": "Bb",
    }

    FLAT_TO_SHARP: ClassVar[Dict[NoteName, NoteName]] = {
        v: k for k, v in SHARP_TO_FLAT.items()
    }

    def frequency_to_note(
        self, tuning, frequency: float, use_flats: bool = False
    ) -> str:
        """Convert a frequency in Hz to the nearest note name.

        Args:
            frequency: The frequency in Hz to convert
            use_flats: If True, use flat notes (e.g., Gb) instead of sharps (e.g., F#)

        Returns:
            str: The note name with octave (e.g., 'A4', 'F#2' or 'Gb2'), or empty string if invalid

        Raises:
            ValueError: If frequency is not a positive number
        """

        if not isinstance(frequency, (int, float)) or not np.isfinite(frequency):
            logger.warning(f"Invalid frequency value: {frequency}")
            return ""

        if frequency <= 0:
            logger.debug(f"Non-positive frequency detected: {frequency}")
            return ""

        try:
            # Calculate the number of half steps from A4
            n = 12 * np.log2(frequency / tuning)
            note_num = round(n) + 57  # 57 is the number of semitones from C0 to A4

            # Ensure note_num is within valid range
            if note_num < 0 or note_num >= len(self.SHARP_NOTES) * 10:  # 10 octaves
                logger.warning(
                    f"Note number {note_num} out of range for frequency {frequency}"
                )
                return ""

            # Get note name and octave
            note_name = self.SHARP_NOTES[note_num % 12]
            octave = (note_num // 12) - 1  # C0 is octave 0

            # Convert to flat if requested and applicable
            if use_flats and note_name in self.SHARP_TO_FLAT:
                note_name = self.SHARP_TO_FLAT[note_name]

            return f"{note_name}{octave}"

        except Exception as e:
            logger.error(
                f"Error converting frequency {frequency} to note: {e}", exc_info=True
            )
            raise e
