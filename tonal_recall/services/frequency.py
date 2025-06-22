#!/usr/bin/env python3

import numpy as np
import logging
from typing import (
    List,
    Dict,
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

    # Standard chromatic scale notes (all notes in an octave)
    STANDARD_NOTES: ClassVar[List[NoteName]] = [
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

    def _analyze_fft(self, audio_data: np.ndarray, sample_rate) -> float:
        """Analyze the audio data with FFT to find the dominant frequency."""
        # Apply a window function to reduce spectral leakage
        window = np.hanning(len(audio_data))
        windowed_data = audio_data * window

        # Calculate FFT with zero-padding for better frequency resolution
        n_fft = 4 * len(audio_data)
        fft = np.fft.rfft(windowed_data, n=n_fft)
        fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / sample_rate)

        # Get the magnitude spectrum
        magnitude = np.abs(fft)

        # Filter to focus on bass/guitar frequency range (30-500 Hz)
        bass_range = (fft_freqs >= 30) & (fft_freqs <= 500)
        bass_freqs = fft_freqs[bass_range]
        bass_magnitude = magnitude[bass_range]

        # Find the dominant frequency in the bass range
        if len(bass_magnitude) > 0:
            max_idx = np.argmax(bass_magnitude)
            dom_freq = bass_freqs[max_idx]

            # Snap to the closest standard note frequency if within a tolerance
            def get_note_freq(note_idx, octave=4):
                semitones_from_a4 = (note_idx - 9) + (octave - 4) * 12
                return 440.0 * (2.0 ** (semitones_from_a4 / 12.0))

            closest_note = min(
                self.STANDARD_NOTES,
                key=lambda note: abs(
                    get_note_freq(self.STANDARD_NOTES.index(note)) - dom_freq
                ),
            )
            note_freq = get_note_freq(self.STANDARD_NOTES.index(closest_note))

            if abs(dom_freq - note_freq) / note_freq < 0.05:
                dom_freq = note_freq

            return dom_freq, bass_freqs, bass_magnitude
        return 0, np.array([]), np.array([])

    def select_and_correct_pitch(
        self,
        aubio_pitch: float,
        aubio_confidence: float,
        signal_max: float,
        current_stable_note,
        audio_data,
        sample_rate,
    ) -> float:
        """Select the best pitch, apply octave correction and weak signal handling."""
        CONFIDENCE_THRESHOLD = 0.7
        detected_freq = 0

        self.fft_freq, bass_freqs, bass_magnitude = self._analyze_fft(
            audio_data, sample_rate
        )

        if aubio_confidence >= CONFIDENCE_THRESHOLD and 30 < aubio_pitch < 1000:
            detected_freq = aubio_pitch
        elif 30 < self.fft_freq < 1000:
            detected_freq = self.fft_freq

        # Octave error correction for low notes
        if 70 < detected_freq < 150 and len(bass_freqs) > 0:
            sub_harmonic_freq = detected_freq / 2.0
            tolerance_hz = 2.0
            sub_harmonic_min = sub_harmonic_freq - tolerance_hz
            sub_harmonic_max = sub_harmonic_freq + tolerance_hz

            sub_harmonic_indices = np.where(
                (bass_freqs >= sub_harmonic_min) & (bass_freqs <= sub_harmonic_max)
            )
            if len(sub_harmonic_indices[0]) > 0:
                sub_harmonic_energy = np.sum(bass_magnitude[sub_harmonic_indices])

                peak_indices = np.where(
                    (bass_freqs >= detected_freq - tolerance_hz)
                    & (bass_freqs <= detected_freq + tolerance_hz)
                )
                peak_energy = (
                    np.sum(bass_magnitude[peak_indices])
                    if len(peak_indices[0]) > 0
                    else 0
                )

                if peak_energy > 0 and sub_harmonic_energy > (peak_energy * 0.3):
                    logger.debug(
                        f"Octave error detected. Original: {detected_freq:.1f}Hz. "
                        f"Correcting to {sub_harmonic_freq:.1f}Hz."
                    )
                    detected_freq = sub_harmonic_freq

        # If signal is weak, maintain the current stable note to prevent jumping
        if signal_max < 0.05 and current_stable_note:
            logger.debug(
                f"Signal weak ({signal_max:.4f} < 0.05), "
                f"maintaining current note: {current_stable_note.note_name}"
            )
            return current_stable_note.frequency

        return detected_freq
