from __future__ import annotations
import os
import numpy as np
import aubio
import sounddevice as sd
import time
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
from .audio_device import find_rocksmith_adapter
from .note_utils import get_note_name, convert_note_notation
from .detection.stability_analyzer import StabilityAnalyzer

from .note_types import DetectedNote

# Get logger for this module
logger = logging.getLogger(__name__)

# DetectedNote class moved to types.py


class NoteDetector:
    """A class for detecting musical notes from audio input in real-time."""

    # Type aliases
    NoteName: TypeAlias = str
    Frequency: TypeAlias = float
    Confidence: TypeAlias = float
    SignalStrength: TypeAlias = float

    # Audio configuration
    SAMPLE_RATE: ClassVar[int] = 44100  # Hz
    FRAMES_PER_BUFFER: ClassVar[int] = 4096  # Number of frames per buffer
    CHANNELS: ClassVar[int] = 1  # Mono audio

    # Note detection settings
    DEFAULT_MIN_CONFIDENCE: ClassVar[Confidence] = (
        0.7  # Minimum confidence to consider a note detection valid
    )
    DEFAULT_MIN_SIGNAL: ClassVar[SignalStrength] = (
        0.005  # Minimum signal level to process (avoids noise)
    )
    MIN_FREQUENCY: ClassVar[Frequency] = 30.0  # Hz - below this is probably noise
    MAX_FREQUENCY: ClassVar[Frequency] = (
        1000.0  # Hz - above this is probably noise for our purposes
    )

    # Note stability tracking
    STABILITY_WINDOW: ClassVar[int] = (
        5  # Number of consecutive detections to consider stable
    )

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

    # Standard note frequencies for reference (A4 = 440Hz)
    A4_FREQ: ClassVar[Frequency] = 440.0  # A4 is the reference note at 440Hz

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

    def __init__(
        self,
        device_id: Optional[int] = None,
        silence_threshold_db: float = -90,
        tolerance: float = 0.8,
        min_stable_count: int = 3,
        stability_majority: float = 0.7,
        min_confidence: Optional[Confidence] = None,
        min_signal: Optional[SignalStrength] = None,
        min_frequency: Optional[Frequency] = None,
        sample_rate: Optional[int] = None,
        frames_per_buffer: Optional[int] = None,
        channels: Optional[int] = None,
        use_flats: bool = False,
    ) -> None:
        """Initialize the NoteDetector with the given parameters.

        Args:
            device_id: Audio input device ID, or None to auto-detect
            silence_threshold_db: Silence threshold in dB (default -90)
            tolerance: aubio pitch tolerance (default 0.8)
            min_stable_count: Minimum consecutive stable detections for note stability (default 3)
            stability_majority: % of readings required for stability (default 0.7)
            min_confidence: Minimum confidence for aubio pitch (default 0.7)
            min_signal: Minimum signal level (0-1) to consider for detection (default 0.005)
            min_frequency: Minimum frequency (Hz) to consider valid (default 30.0)
            sample_rate: Audio sample rate in Hz (default 44100)
            frames_per_buffer: Number of audio frames per buffer (default 1024)
            channels: Number of audio channels (1=mono, 2=stereo) (default: 1)
            use_flats: Whether to use flat note names (e.g., Bb) instead of sharps (A#)
            channels: Number of audio channels (1=mono, 2=stereo, default 1)
        """
        # Audio configuration
        self._sample_rate = (
            int(sample_rate) if sample_rate is not None else self.SAMPLE_RATE
        )
        self._frames_per_buffer = (
            int(frames_per_buffer)
            if frames_per_buffer is not None
            else self.FRAMES_PER_BUFFER
        )
        self._channels = int(channels) if channels is not None else self.CHANNELS

        # Detection parameters with type conversion
        self._silence_threshold_db = float(silence_threshold_db)
        self._tolerance = float(tolerance)
        self._min_stable_count = int(min_stable_count)
        self._stability_majority = float(stability_majority)
        self._min_confidence = float(
            min_confidence
            if min_confidence is not None
            else self.DEFAULT_MIN_CONFIDENCE
        )
        self._min_signal = float(
            min_signal if min_signal is not None else self.DEFAULT_MIN_SIGNAL
        )
        self._min_frequency = float(
            min_frequency if min_frequency is not None else self.MIN_FREQUENCY
        )
        self._group_hz = 1.0  # Frequency grouping in Hz

        # Instrument configuration
        self._use_flats: bool = bool(use_flats)

        # Stability tracking
        self._stability_analyzer = StabilityAnalyzer(
            min_frequency=self._min_frequency,
            min_confidence=self._min_confidence,
            min_signal=self._min_signal,
            min_stable_count=self._min_stable_count,
            group_hz=self.group_hz,
            history_size=self.STABILITY_WINDOW,
        )
        self._current_stable_note: Optional[DetectedNote] = None
        self._running: bool = False  # Tracks if the audio stream is running
        self._callback: Optional[Callable[[DetectedNote, float], None]] = None
        self._buffer_size: int = self._frames_per_buffer

        # Audio processing state
        self._device_id: Optional[int] = device_id
        self._audio_stream: Optional[sd.InputStream] = None

        # Initialize aubio pitch detection
        # The buf_size (win_s) is crucial for detecting low-frequency notes accurately.
        # The hop_size (hop_s) determines how often we analyze a new chunk.
        win_s = self._frames_per_buffer  # The analysis window size.
        hop_s = (
            win_s // 4
        )  # The step size between analysis windows. Overlap improves stability.
        self._pitch_detector = aubio.pitch(
            "yin",
            win_s,
            hop_s,
            self._sample_rate,
        )
        self._pitch_detector.set_unit("Hz")
        self._pitch_detector.set_tolerance(self._tolerance)

        # Initialize audio device
        try:
            logger.info("Initializing audio device for note detection...")
            self._init_audio_device()
        except Exception as e:
            logger.error(f"Failed to initialize audio device: {e}")
            raise

        logger.info(
            f"Initialized NoteDetector with sample_rate={self._sample_rate}Hz, "
            f"min_confidence={self._min_confidence}, min_signal={self._min_signal}"
        )

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags
    ) -> None:
        """Callback for processing audio data from the input stream.

        Args:
            indata: The input audio data as a numpy array (frames x channels)
            frames: Number of frames in the buffer
            time_info: Dictionary containing timing information from PortAudio
            status: Callback status flags from PortAudio

        Note:
            This callback runs in a separate audio thread. Keep processing minimal
            and avoid any blocking operations to prevent audio glitches.
        """
        try:
            if status:
                logger.warning(f"Audio status: {status}")

            if not self._running:
                return

            # Convert input to float32 and process only the first channel
            audio_data = indata[:, 0].astype(np.float32)
            self._process_audio(audio_data)

        except Exception as e:
            logger.error(f"Error in _audio_callback: {e}", exc_info=True)
            raise

    def start(self, callback: Callable[[DetectedNote, float], None]) -> bool:
        """Start the note detection audio stream.

        Args:
            callback: Function to call when a note is detected.
                     The function should accept two parameters:
                     - note (DetectedNote): The detected note
                     - signal_strength (float): The strength of the detected signal (0-1)

        Returns:
            bool: True if started successfully, False otherwise

        Note:
            This method initializes the audio stream and starts listening for input.
            The callback will be called from a separate audio thread when notes are detected.
        """
        if self._running:
            logger.warning("Note detector is already running")
            return False

        self._callback = callback
        self._running = True

        try:
            # Set environment variable to use PortAudio explicitly
            os.environ["PYAUDIO_HOST"] = "portaudio"

            logger.info(
                f"Starting note detection on device: {self._device_info['name'] if self._device_info else 'default'}"
            )
            logger.debug(
                f"Audio settings - Sample rate: {self._sample_rate}Hz, "
                f"Buffer size: {self._buffer_size}, Channels: {self._channels}"
            )

            # Initialize and start the audio stream
            self._audio_stream = sd.InputStream(
                device=self._device_id,
                samplerate=self._sample_rate,
                channels=self._channels,
                callback=self._audio_callback,
                blocksize=self._buffer_size,
                dtype=np.float32,
                latency="high",
            )
            self._audio_stream.start()

            logger.info("Note detection started - stable notes will be logged")
            return True

        except Exception as e:
            error_msg = f"Failed to start audio stream: {e}"
            logger.error(error_msg)
            logger.debug(f"Device info: {self._device_info}", exc_info=True)
            self._running = False
            self._audio_stream = None
            raise e

    def stop(self) -> None:
        """Stop the note detector and clean up resources.

        This method stops the audio stream and cleans up resources. It's safe to call
        this method even if the detector is not currently running.
        """
        if not self._running:
            logger.debug("Note detector is not running, nothing to stop")
            return

        self._running = False
        self._callback = None

        if self._audio_stream is not None:
            try:
                if self._audio_stream.active:
                    self._audio_stream.stop()
                if not self._audio_stream.closed:
                    self._audio_stream.close()
                logger.info("Audio stream stopped and closed")
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}", exc_info=True)
            finally:
                self._audio_stream = None

        logger.debug("Note detector stopped")

    def _process_audio(self, audio_data: np.ndarray) -> None:
        """Process a chunk of audio data and detect notes.

        Args:
            audio_data: Numpy array containing audio samples (1D float32 array)

        Note:
            This method processes audio data, detects notes, and triggers callbacks
            when notes are detected. It handles all the audio processing pipeline
            including signal level checking, pitch detection, and note stability.

            The method follows these steps:
            1. Check if processing should continue
            2. Calculate signal level (RMS)
            3. Skip if signal is too weak
            4. Get pitch and confidence from aubio
            5. Skip if confidence is too low or pitch is out of range
            6. Convert frequency to note name
            7. Update note history and check for stability
            8. Trigger callback if a stable note is detected
        """
        if not self._running or self._callback is None:
            return

        # Calculate signal level (RMS)
        signal_level: float = float(np.sqrt(np.mean(audio_data**2)))
        logger.debug(f"Signal: {signal_level:.4f}")

        # Skip if signal is too weak
        if signal_level < self._min_signal:
            return

        # Get pitch and confidence from aubio
        pitch: float = float(self._pitch_detector(audio_data)[0])
        confidence: float = float(self._pitch_detector.get_confidence())
        logger.debug(f"Pitch: {pitch:.2f} Hz, Confidence: {confidence:.4f}")

        # Skip if confidence is too low or pitch is out of range
        if (
            confidence < self._min_confidence
            or pitch < self._min_frequency
            or pitch > self.MAX_FREQUENCY
        ):
            return

        # Convert frequency to note name
        note_name: Optional[str] = self._frequency_to_note(
            pitch, use_flats=self._use_flats
        )
        if not note_name:
            return

        # Create a DetectedNote object from the raw pitch detection
        detected = DetectedNote(
            note_name=note_name,
            frequency=pitch,
            confidence=confidence,
            signal=signal_level,
            is_stable=False,  # The analyzer will determine stability
            timestamp=time.time(),
        )

        # Add the new note to the stability analyzer
        self._stability_analyzer.add_note(detected)

        # Get the stable note from the analyzer
        new_stable_note = self._stability_analyzer.get_stable_note()

        # Only trigger the callback if the stable note has changed
        if new_stable_note != self._current_stable_note:
            self._current_stable_note = new_stable_note
            try:
                # The callback receives the new stable note (or None) and the signal strength
                self._callback(new_stable_note, signal_level)
            except Exception as e:
                logger.error(f"Error in note detection callback: {e}", exc_info=True)
                raise e

    def _frequency_to_note(self, frequency: float, use_flats: bool = False) -> str:
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
            n = 12 * np.log2(frequency / self.A4_FREQ)
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

    def convert_note_notation(self, note_name: str, to_flats: bool = False) -> str:
        return convert_note_notation(self, note_name, to_flats)

    @property
    def silence_threshold_db(self) -> float:
        """Get the silence threshold in dB.

        Returns:
            float: Silence threshold in dB (default -90.0)
        """
        return self._silence_threshold_db

    @silence_threshold_db.setter
    def silence_threshold_db(self, value: float) -> None:
        """Set the silence threshold in dB.

        Args:
            value: New silence threshold in dB
        """
        self._silence_threshold_db = float(value)
        if self._pitch_detector:
            self._pitch_detector.set_silence(self._silence_threshold_db)

    @property
    def tolerance(self) -> float:
        """Get the aubio pitch tolerance.

        Returns:
            float: Current tolerance value (default 0.8)
        """
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value: float) -> None:
        """Set the aubio pitch tolerance.

        Args:
            value: New tolerance value (0.0 to 1.0)
        """
        self._tolerance = float(value)
        if hasattr(self, "pitch_detector"):
            self._pitch_detector.set_tolerance(self._tolerance)

    @property
    def min_stable_count(self) -> int:
        """Get the minimum number of stable detections to consider a note stable.

        Returns:
            int: Minimum stable count (default 3)
        """
        return self._min_stable_count

    @min_stable_count.setter
    def min_stable_count(self, value: int) -> None:
        """Set the minimum number of stable detections.

        Args:
            value: New minimum stable count (positive integer)
        """
        self._min_stable_count = max(1, int(value))  # Ensure at least 1

    @property
    def stability_majority(self) -> float:
        """Get the proportion of detections needed for stability.

        Returns:
            float: Stability majority threshold (0.0 to 1.0, default 0.7)
        """
        return self._stability_majority

    @stability_majority.setter
    def stability_majority(self, value: float) -> None:
        """Set the stability majority threshold.

        Args:
            value: New threshold (0.0 to 1.0)
        """
        self._stability_majority = max(
            0.0, min(1.0, float(value))
        )  # Clamp to 0-1 range

    @property
    def group_hz(self) -> float:
        """Get the frequency grouping in Hz.

        Returns:
            float: Frequency grouping in Hz (default 10.0)
        """
        return self._group_hz

    @group_hz.setter
    def group_hz(self, value: float) -> None:
        """Set the frequency grouping in Hz.

        Args:
            value: New frequency grouping in Hz (must be positive)
        """
        self._group_hz = max(0.1, float(value))  # Ensure positive value

    @property
    def snap_percent(self) -> float:
        """Get the snap percentage for note detection.

        Returns:
            float: Snap percentage (0.0 to 1.0, default 0.05)
        """
        return self._snap_percent

    @snap_percent.setter
    def snap_percent(self, value: float) -> None:
        """Set the snap percentage for note detection.

        Args:
            value: New snap percentage (0.0 to 1.0)
        """
        self._snap_percent = max(0.0, min(1.0, float(value)))  # Clamp to 0-1 range

    @property
    def normal_majority(self) -> float:
        """Get the normal majority threshold.

        Returns:
            float: Normal majority threshold (0.0 to 1.0, default 0.7)
        """
        return self._normal_majority

    @normal_majority.setter
    def normal_majority(self, value: float) -> None:
        """Set the normal majority threshold.

        Args:
            value: New threshold (0.0 to 1.0)
        """
        self._normal_majority = max(0.0, min(1.0, float(value)))  # Clamp to 0-1 range

    @property
    def min_frequency(self) -> float:
        """Get the minimum frequency to consider for note detection.

        Returns:
            float: Minimum frequency in Hz (default 30.0)
        """
        return (
            self._min_frequency
            if hasattr(self, "_min_frequency")
            else float(self.MIN_FREQUENCY)
        )

    @min_frequency.setter
    def min_frequency(self, value: float) -> None:
        """Set the minimum frequency to consider for note detection.

        Args:
            value: New minimum frequency in Hz (must be positive)
        """
        self._min_frequency = max(1.0, float(value))  # Ensure positive value

    def _init_audio_device(self) -> None:
        """Initialize the audio input device.

        This method sets up the audio input device for note detection, with fallback
        to default settings if the preferred configuration fails.

        Raises:
            Exception: If both primary and fallback audio initialization fail.
        """
        try:
            # Set environment variable to use PortAudio explicitly

            os.environ["PYAUDIO_HOST"] = "portaudio"

            # Find the Rocksmith USB Guitar Adapter if available
            self._device_id, self._device_info = find_rocksmith_adapter()

            # If no Rocksmith adapter found, use default device
            if self._device_id is None:
                self._device_id = sd.default.device[0]  # Default input device
                self._device_info = sd.query_devices(self._device_id)

            # Update sample rate from device if available
            device_rate = int(
                self._device_info.get("default_samplerate", self._sample_rate)
            )
            if device_rate != self._sample_rate:
                logger.info(
                    f"Using device sample rate: {device_rate}Hz (requested {self._sample_rate}Hz)"
                )
                self._sample_rate = device_rate

            # Initialize the audio stream with the correct buffer size
            self._buffer_size = self._frames_per_buffer

            # Initialize aubio pitch detection with updated sample rate if needed
            self._pitch_detector = aubio.pitch(
                "yin",
                self._frames_per_buffer * 2,
                self._frames_per_buffer,
                self._sample_rate,
            )
            self._pitch_detector.set_unit("Hz")
            self._pitch_detector.set_tolerance(self._tolerance)

            logger.info(
                f"Initialized audio device: {self._device_info['name']} "
                f"(ID: {self._device_id}, {self._sample_rate}Hz, {self._channels}ch)"
            )

        except Exception as e:
            logger.error(f"Error initializing audio device: {e}")
            raise e

    def _audio_callback(
        self, indata: np.ndarray, frames: int, stream_time: dict, status: Any
    ) -> None:
        """Callback for processing audio data

        Args:
            indata: Input audio data as numpy array
            frames: Number of frames in the buffer
            stream_time: Dictionary containing timing information
            status: PortAudio status flags
        """
        try:
            if status:
                if status.input_overflow:
                    logger.warning("Input overflow")
                    return  # Skip processing this buffer on overflow

            # Get the audio data and check levels
            audio_data = indata[:, 0] if indata.shape[1] > 0 else indata.flatten()
            signal_max = np.max(np.abs(audio_data))

            # Buffer size check: skip if not expected length
            if len(audio_data) != self._buffer_size:
                logger.warning(
                    f"Skipping frame: got {len(audio_data)} samples, expected {self._buffer_size}"
                )
                return

            # Log detailed audio stats at debug level
            logger.debug(
                f"[AUDIO DEBUG] min={audio_data.min():.4f} max={audio_data.max():.4f} mean={audio_data.mean():.4f} dtype={audio_data.dtype}"
            )

            # Calculate RMS to get a better signal level measurement
            rms = np.sqrt(np.mean(audio_data**2))
            db = 20 * np.log10(rms) if rms > 0 else -100  # Convert to dB

            # Increase noise gate threshold to 0.01
            if signal_max > 0.01:  # Basic noise gate
                # Process the audio data with aubio
                raw_pitch = self._pitch_detector(audio_data.astype(np.float32))
                pitch = raw_pitch[0]  # The frequency in Hz
                confidence = self._pitch_detector.get_confidence()

                # Apply a window function to reduce spectral leakage
                window = np.hanning(len(audio_data))
                windowed_data = audio_data * window

                # Calculate FFT with zero-padding for better frequency resolution
                n_fft = 4 * len(
                    audio_data
                )  # Zero-padding for better frequency resolution
                fft = np.fft.rfft(windowed_data, n=n_fft)
                fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / self._sample_rate)

                # Get the magnitude spectrum
                magnitude = np.abs(fft)

                # Filter to focus on bass/guitar frequency range (30-500 Hz)
                bass_range = (fft_freqs >= 30) & (fft_freqs <= 500)
                bass_freqs = fft_freqs[bass_range]
                bass_magnitude = magnitude[bass_range]

                # Find the dominant frequency in the bass range
                if len(bass_magnitude) > 0:
                    # Find the strongest peak in the bass range
                    max_idx = np.argmax(bass_magnitude)
                    dom_freq = bass_freqs[max_idx]

                    # Find the closest note in the standard notes list
                    # Calculate frequency for each note in the chromatic scale (A4 = 440Hz)
                    def get_note_freq(note_idx, octave=4):
                        # A4 is the 9th note in STANDARD_NOTES (0-based index 9)
                        semitones_from_a4 = (note_idx - 9) + (octave - 4) * 12
                        return 440.0 * (2.0 ** (semitones_from_a4 / 12.0))

                    # Find the note with frequency closest to dom_freq
                    closest_note = min(
                        NoteDetector.STANDARD_NOTES,
                        key=lambda note: abs(
                            get_note_freq(NoteDetector.STANDARD_NOTES.index(note))
                            - dom_freq
                        ),
                    )
                    note_name = closest_note
                    note_freq = get_note_freq(
                        NoteDetector.STANDARD_NOTES.index(note_name)
                    )

                    # If we're within 5% of a standard note, snap to that frequency
                    if abs(dom_freq - note_freq) / note_freq < 0.05:
                        dom_freq = note_freq
                else:
                    dom_freq = 0

                current_time = time.strftime("%H:%M:%S")
                if pitch > 0:
                    note_name = get_note_name(pitch)
                    dom_note = get_note_name(dom_freq) if dom_freq > 0 else "---"
                    logger.debug(
                        f"{current_time} | Aubio: {pitch:.1f} Hz ({note_name}) | FFT: {dom_freq:.1f} Hz ({dom_note}) | Conf: {confidence:.2f} | Sig: {signal_max:.3f}"
                    )
                else:
                    dom_note = get_note_name(dom_freq) if dom_freq > 0 else "---"
                    logger.debug(
                        f"{current_time} | Aubio: {pitch:.1f} Hz | FFT: {dom_freq:.1f} Hz ({dom_note}) | Conf: {confidence:.2f} | Sig: {signal_max:.3f}"
                    )

                # Pitch selection logic: Use aubio pitch if confidence is high and pitch is valid, otherwise use FFT
                CONFIDENCE_THRESHOLD = 0.7  # Can be tuned or made configurable
                detected_method = None
                if confidence >= CONFIDENCE_THRESHOLD and 30 < pitch < 1000:
                    detected_freq = pitch
                    detected_method = "aubio"
                elif 30 < dom_freq < 1000:
                    detected_freq = dom_freq
                    detected_method = "fft"
                else:
                    detected_freq = 0
                    detected_method = "none"

                logger.debug(
                    f"Pitch selection - method: {detected_method}, confidence: {confidence:.2f}, "
                    f"aubio: {pitch:.1f}Hz, fft: {dom_freq:.1f}Hz"
                )

                # Octave error correction for low notes
                # If the detected frequency is in a range where it might be a harmonic (e.g., 80-150 Hz for bass)
                # check for energy at the sub-harmonic frequency (half the detected frequency).
                if 70 < detected_freq < 150:
                    sub_harmonic_freq = detected_freq / 2.0

                    # Define a small window around the sub-harmonic to check for energy
                    tolerance_hz = 2.0
                    sub_harmonic_min = sub_harmonic_freq - tolerance_hz
                    sub_harmonic_max = sub_harmonic_freq + tolerance_hz

                    # Find the energy in the sub-harmonic range within the FFT spectrum
                    sub_harmonic_indices = np.where(
                        (bass_freqs >= sub_harmonic_min)
                        & (bass_freqs <= sub_harmonic_max)
                    )
                    if len(sub_harmonic_indices[0]) > 0:
                        sub_harmonic_energy = np.sum(
                            bass_magnitude[sub_harmonic_indices]
                        )

                        # Find the energy of the originally detected peak
                        peak_indices = np.where(
                            (bass_freqs >= detected_freq - tolerance_hz)
                            & (bass_freqs <= detected_freq + tolerance_hz)
                        )
                        peak_energy = (
                            np.sum(bass_magnitude[peak_indices])
                            if len(peak_indices[0]) > 0
                            else 0
                        )

                        # If the sub-harmonic energy is significant (e.g., > 30% of the peak's energy), it's likely an octave error.
                        if peak_energy > 0 and sub_harmonic_energy > (
                            peak_energy * 0.3
                        ):
                            logger.debug(
                                f"Octave error detected. Original: {detected_freq:.1f}Hz. "
                                f"Found significant sub-harmonic energy ({sub_harmonic_energy:.2f}) "
                                f"at ~{sub_harmonic_freq:.1f}Hz compared to peak energy ({peak_energy:.2f}). "
                                f"Correcting pitch."
                            )
                            detected_freq = sub_harmonic_freq

                # If signal is weak (below 0.05), maintain the current stable note
                # This prevents jumping between notes during decay
                if signal_max < 0.05 and self._current_stable_note:
                    logger.debug(
                        f"Signal weak ({signal_max:.4f} < 0.05), "
                        f"maintaining current note: {self._current_stable_note.note_name}"
                    )
                    detected_freq = self._current_stable_note.frequency

                # Filter out unreasonable frequencies
                if (
                    detected_freq > 0 and 30 < detected_freq < 1000
                ):  # Reasonable range for guitar/bass
                    # Convert frequency to note name
                    note_name = get_note_name(detected_freq)
                    detected = DetectedNote(
                        note_name,
                        detected_freq,
                        confidence,
                        signal_max,
                        False,
                        time.time(),
                    )
                    self._stability_analyzer.add_note(detected)

                    # Track previous stable note for change detection
                    prev_stable_note = self._current_stable_note

                    # Get stable note
                    new_stable_note = self._stability_analyzer.get_stable_note()
                    self._stable_note = new_stable_note
                    self._current_stable_note = new_stable_note

                    if new_stable_note:
                        if self._callback:
                            self._callback(new_stable_note, signal_max)

                    # Log stable note changes
                    stable_note_changed = (
                        (prev_stable_note is None and new_stable_note is not None)
                        or (prev_stable_note is not None and new_stable_note is None)
                        or (
                            prev_stable_note
                            and new_stable_note
                            and (
                                prev_stable_note.note_name != new_stable_note.note_name
                                or abs(
                                    prev_stable_note.confidence
                                    - new_stable_note.confidence
                                )
                                > 0.1
                            )
                        )
                    )

                    if stable_note_changed and prev_stable_note:
                        logger.info(f"Note released: {prev_stable_note.note_name}")
                    elif not new_stable_note and not prev_stable_note and detected:
                        # Only log current note if we don't have a stable note and it's not just silence
                        self._current_note = detected
                        logger.debug(
                            f"{current_time} | Detected: {self._current_note.note_name} | "
                            f"{self._current_note.frequency:.1f} Hz | "
                            f"Conf: {self._current_note.confidence:.2f} | "
                            f"Signal: {signal_max:.4f}"
                        )

            else:
                current_time = time.strftime("%H:%M:%S")
                logger.debug(
                    f"{current_time} | Waiting for input | Signal: {signal_max:.4f} | dB: {db:.1f}"
                )
        except Exception as e:
            logger.error(f"Error in audio_callback: {e}", exc_info=True)
            raise e

    def get_current_note(self) -> Optional[DetectedNote]:
        """Get the current detected note

        Returns:
            Optional[DetectedNote]: The current stable note, or None if no note is detected
        """
        return self._current_stable_note

    def get_simple_note(self) -> Optional[str]:
        """Get just the note letter (A, B, C, etc.) without the octave

        Returns:
            Optional[str]: The note letter, or None if no note is detected
        """
        if self._current_stable_note:
            return self._current_stable_note.note_name[0]
        return None

    def is_note_playing(self, target_note: str) -> bool:
        """Check if a specific note is currently playing

        Args:
            target_note: The target note letter (A, B, C, etc.)

        Returns:
            bool: True if the target note is currently playing, False otherwise
        """
        current = self.get_simple_note()
        return current == target_note
