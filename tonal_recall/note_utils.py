"""Utility functions for working with musical notes and frequencies."""

import numpy as np
import logging
import time
from typing import (
    Optional,
)
from .note_types import DetectedNote

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


def get_stable_note(cls) -> Optional[DetectedNote]:
    """Determine if there's a stable note in the history

    Returns:
        A DetectedNote if a stable note is found, None otherwise
    """
    logger.debug(
        f"Current note history ({len(cls._note_history)}): "
        f"{[f'{n.note_name}({n.confidence:.2f},{n.signal:.3f})' for n in cls._note_history]}"
    )

    # Filter notes by frequency, confidence, and signal
    valid_notes = [
        n
        for n in cls._note_history
        if n.frequency >= cls._min_frequency
        and getattr(n, "confidence", 1.0) >= cls._min_confidence
        and getattr(n, "signal", 1.0) >= cls._min_signal
    ]

    logger.debug(
        f"Valid readings: {len(valid_notes)}/{len(cls._note_history)} | "
        f"min_conf: {cls._min_confidence} min_sig: {cls._min_signal} min_freq: {cls._min_frequency}"
    )

    if valid_notes:
        logger.debug(
            f"Valid note range: {min(n.frequency for n in valid_notes):.1f}Hz - {max(n.frequency for n in valid_notes):.1f}Hz"
        )

    if len(valid_notes) < cls._min_stable_count:
        logger.debug(
            f"Not enough valid readings for stability: "
            f"{len(valid_notes)} < {cls._min_stable_count} (min_stable_count)"
        )
        if cls._stable_note is not None:
            logger.debug(f"Clearing stable note: {cls._stable_note.note_name}")
        cls._stable_note = None
        return None

    # Group frequencies that are close to each other (within group_hz)
    freq_groups = []
    for note in valid_notes:
        # Check if this frequency fits in an existing group
        found_group = False
        for group in freq_groups:
            group_avg = sum(n.frequency for n in group) / len(group)
            freq_diff = abs(note.frequency - group_avg)
            if freq_diff < cls.group_hz:
                group.append(note)
                found_group = True
                logger.debug(
                    f"Grouped {note.note_name} ({note.frequency:.1f}Hz) with "
                    f"group avg {group_avg:.1f}Hz (diff: {freq_diff:.1f} < {cls._group_hz}Hz)"
                )
                break

        # If no matching group, create a new one
        if not found_group:
            freq_groups.append([note])
            logger.debug(
                f"Created new group for {note.note_name} ({note.frequency:.1f}Hz)"
            )

    # Log all frequency groups
    for i, group in enumerate(freq_groups):
        freqs = [n.frequency for n in group]
        avg_freq = sum(freqs) / len(freqs)
        logger.debug(
            f"Group {i}: {len(group)} notes, "
            f"freq range: {min(freqs):.1f}-{max(freqs):.1f}Hz, "
            f"avg: {avg_freq:.1f}Hz"
        )

    # Find the largest group of similar frequencies
    if not freq_groups:
        logger.debug("No frequency groups found")
        cls._stable_note = None
        return None

    # Sort groups by size (largest first)
    freq_groups.sort(key=len, reverse=True)
    largest_group = freq_groups[0]

    freqs = [n.frequency for n in largest_group]
    avg_freq = sum(freqs) / len(freqs)
    logger.debug(
        f"Largest group: {len(largest_group)} notes, "
        f"freq range: {min(freqs):.1f}-{max(freqs):.1f}Hz, "
        f"avg: {avg_freq:.1f}Hz"
    )

    # Get average frequency of the largest group
    avg_freq = sum(n.frequency for n in largest_group) / len(largest_group)
    avg_confidence = sum(n.confidence for n in largest_group) / len(largest_group)
    avg_signal = sum(n.signal for n in largest_group) / len(largest_group)

    # Get most recent note in the group for its timestamp
    most_recent = max(largest_group, key=lambda n: n.timestamp)

    # Create a new note with the averaged values
    new_note = DetectedNote(
        note_name=get_note_name(avg_freq),
        frequency=avg_freq,
        confidence=avg_confidence,
        signal=avg_signal,
        is_stable=True,
        timestamp=most_recent.timestamp,
    )

    # Check if we had a previous stable note
    if cls._stable_note is not None:
        # If the note name changed, check if it's a significant change
        if new_note.note_name != cls._stable_note.note_name:
            # Check if the frequency change is significant
            freq_diff = abs(new_note.frequency - cls._stable_note.frequency)
            if freq_diff < cls.group_hz:
                # Not a significant change, keep the previous note
                logger.debug(
                    f"Ignoring small frequency change: "
                    f"{cls._stable_note.note_name} -> {new_note.note_name} "
                    f"(diff: {freq_diff:.1f}Hz < {cls._group_hz}Hz)"
                )
                return cls._stable_note
        else:
            if cls._last_stable_note != cls._stable_note.note_name:
                logger.debug(
                    f"Stable note held: {cls._stable_note.note_name} "
                    f"(confidence: {avg_confidence * 100:.1f}%, signal: {avg_signal:.3f})"
                )
            cls._last_stable_note = cls._stable_note.note_name
            return DetectedNote(
                cls._stable_note.note_name,
                cls._stable_note.frequency,
                avg_confidence,
                avg_signal,
                True,
                time.time(),
            )

    # If we get here, we have a new stable note
    if cls._last_stable_note != new_note.note_name:
        logger.info(
            f"New stable note: {new_note.note_name} "
            f"({len(largest_group)}/{len(valid_notes)} votes, {avg_confidence * 100:.1f}% confidence)"
        )

    cls._last_stable_note = new_note.note_name
    return new_note
