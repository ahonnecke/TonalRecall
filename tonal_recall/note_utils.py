"""Utility functions for working with musical notes and frequencies."""

import numpy as np
import logging

# Get logger for this module
logger = logging.getLogger(__name__)


def convert_note_notation(cls, note_name: str, to_flats: bool = False) -> str:
    """Convert a note name between sharp and flat notation.

    Args:
        note_name: The note name to convert (e.g., 'F#2' or 'Gb2')
        to_flats: If True, convert to flats (e.g., 'Gb2'), otherwise to sharps (e.g., 'F#2')

    Returns:
        str: The converted note name, or original if no conversion needed or invalid

    Examples:
        >>> convert_note_notation('F#2', to_flats=True)  # Returns 'Gb2'
        >>> convert_note_notation('Gb2', to_flats=False)  # Returns 'F#2'
    """
    if not note_name or not isinstance(note_name, str):
        return note_name or ""

    try:
        # Extract note letter and octave
        note_part = "".join(
            c for c in note_name if not c.isdigit() and c != "-"
        ).strip()
        octave_part = note_name[len(note_part) :] if note_name else ""

        # Check if conversion is needed
        if to_flats and note_part in cls.SHARP_TO_FLAT:
            return f"{cls.SHARP_TO_FLAT[note_part]}{octave_part}"
        elif not to_flats and note_part in cls.FLAT_TO_SHARP:
            return f"{cls.FLAT_TO_SHARP[note_part]}{octave_part}"

        # No conversion needed or possible
        return note_name

    except Exception as e:
        logger.error(f"Error converting note {note_name}: {e}", exc_info=True)
        return note_name


def get_note_name(freq: float, use_flats: bool = False) -> str:
    """Convert frequency to note name using Scientific Pitch Notation (SPN).

    Args:
        freq: Frequency in Hz
        use_flats: If True, use flat notes (e.g., 'Bb') instead of sharps (e.g., 'A#')
                  Note: Some notes will still use sharps (e.g., 'F#') to maintain SPN conventions

    Returns:
        Note name with octave in SPN (e.g., 'A4', 'C#4', 'Bb3')

    Note:
        - Middle C is C4 (261.63 Hz)
        - A4 is 440 Hz
        - Octave numbers change between B and C (e.g., B3 -> C4)
    """
    if freq <= 0:
        return "---"

    # Standard reference: A4 = 440Hz
    # Calculate half steps from A4 (A4 is 69 in MIDI)
    half_steps = round(12 * np.log2(freq / 440.0))
    midi_number = 69 + half_steps  # A4 = 69 in MIDI

    # Calculate note name and octave according to SPN
    note_names_sharps = [
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
    note_names_flats = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

    # SPN octave calculation (C4 is middle C)
    octave = (midi_number // 12) - 1
    note_idx = midi_number % 12

    # Always use sharps for E#/B# to follow SPN conventions
    if not use_flats or note_idx in [4, 11]:  # E or B
        note_name = note_names_sharps[note_idx]
    else:
        note_name = note_names_flats[note_idx]

    return f"{note_name}{octave}"
