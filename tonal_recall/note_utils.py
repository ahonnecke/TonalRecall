"""Utility functions for working with musical notes and frequencies."""

import numpy as np
from typing import Tuple


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
    note_names_sharps = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
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

