import numpy as np
import aubio
import sounddevice as sd
import time
from collections import deque, defaultdict
from dataclasses import dataclass
from typing import Optional, Callable, Dict, List, Tuple

from .logger import get_logger

# Get logger for this module
logger = get_logger(__name__)


@dataclass
class DetectedNote:
    """Represents a detected musical note with its properties."""

    name: str  # Note name (e.g., 'A4', 'C#3')
    frequency: float  # Frequency in Hz
    confidence: float  # Detection confidence (0-1)
    signal: float  # Signal strength (0-1, e.g., max(abs(audio)))
    is_stable: bool  # Whether this is a stable note
    timestamp: float  # Timestamp when the note was detected

    def __str__(self):
        return f"{self.name} ({self.frequency:.1f}Hz, {self.confidence:.2f}, {'stable' if self.is_stable else 'unstable'})"


class NoteDetector:
    """A class for detecting musical notes from audio input in real-time."""

    # Audio configuration
    SAMPLE_RATE = 44100  # Hz
    FRAMES_PER_BUFFER = 1024  # Number of frames per buffer
    CHANNELS = 1  # Mono audio

    # Note detection settings
    DEFAULT_MIN_CONFIDENCE = (
        0.7  # Minimum confidence to consider a note detection valid
    )
    DEFAULT_MIN_SIGNAL = 0.005  # Minimum signal level to process (avoids noise)
    MIN_FREQUENCY = 30.0  # Hz - below this is probably noise
    MAX_FREQUENCY = 1000.0  # Hz - above this is probably noise for our purposes

    # Note stability tracking
    STABILITY_WINDOW = 5  # Number of consecutive detections to consider stable
    STABILITY_THRESHOLD = 0.5  # Proportion of detections that must match to be stable

    # Standard note frequencies for reference (A4 = 440Hz)
    NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    A4_FREQ = 440.0  # A4 is the reference note at 440Hz

    # Standard chromatic scale notes (all notes in an octave)
    STANDARD_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

    # Standard tunings for reference
    STANDARD_TUNINGS = {
        "guitar_standard": ["E2", "A2", "D3", "G3", "B3", "E4"],
        "bass_standard": ["E1", "A1", "D2", "G2"],
        "ukulele_standard": ["G4", "C4", "E4", "A4"],
    }

    def __init__(
        self,
        device_id=None,
        silence_threshold_db=-90,
        tolerance=0.8,
        min_stable_count=3,
        stability_majority=0.7,
        min_confidence=None,
        min_signal=None,
        min_frequency=None,
        sample_rate=None,
        frames_per_buffer=None,
        channels=None,
    ):
        """Initialize the note detector with the given parameters.

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
            channels: Number of audio channels (1=mono, 2=stereo, default 1)
        """
        # Set audio configuration with defaults
        self.sample_rate = sample_rate if sample_rate is not None else self.SAMPLE_RATE
        self.frames_per_buffer = (
            frames_per_buffer
            if frames_per_buffer is not None
            else self.FRAMES_PER_BUFFER
        )
        self.channels = channels if channels is not None else self.CHANNELS

        # Set detection parameters with defaults
        self.min_confidence = (
            min_confidence
            if min_confidence is not None
            else self.DEFAULT_MIN_CONFIDENCE
        )
        self.min_signal = (
            min_signal if min_signal is not None else self.DEFAULT_MIN_SIGNAL
        )
        self.MIN_FREQUENCY = (
            min_frequency if min_frequency is not None else self.MIN_FREQUENCY
        )

        # Set device and stability parameters
        self.device_id = device_id
        
        # Initialize all private attributes with proper type conversion
        self._silence_threshold_db = float(silence_threshold_db)
        self._tolerance = float(tolerance)
        self._min_stable_count = int(min_stable_count)
        self._stability_majority = float(stability_majority)
        self._min_frequency = float(min_frequency) if min_frequency is not None else float(self.MIN_FREQUENCY)
        self._group_hz = 10.0  # Default value for frequency grouping
        self._snap_percent = 0.05  # Default 5% snapping threshold
        self._normal_majority = 0.7  # Default normal majority threshold
        self._min_confidence = float(self.min_confidence)  # Use the property to get default if needed
        self._min_signal = float(self.min_signal)  # Use the property to get default if needed

        # Initialize state
        self.running = False
        self.callback = None
        self.stream = None
        self.recent_notes = deque(maxlen=self.STABILITY_WINDOW)
        self.current_stable_note = None
        self.note_history = deque(maxlen=10)  # Store last 10 detected notes
        self.current_note = None
        self.stable_note = None
        self._last_stable_note = None  # For debug logging
        self.device_info = None
        self.buffer_size = self.frames_per_buffer

        # Initialize aubio pitch detection
        self.pitch_detector = aubio.pitch(
            "yin", self.frames_per_buffer * 2, self.frames_per_buffer, self.sample_rate
        )
        self.pitch_detector.set_unit("Hz")
        self.pitch_detector.set_tolerance(self._tolerance)

        # Initialize audio device
        try:
            logger.info("Initializing audio device for note detection...")
            self._init_audio_device()
        except Exception as e:
            logger.error(f"Failed to initialize audio device: {e}")
            raise

        logger.info(
            f"Initialized NoteDetector with sample_rate={self.sample_rate}Hz, "
            f"min_confidence={self.min_confidence}, min_signal={self.min_signal}"
        )

    def _audio_callback(self, indata, frames, time_info, status):
        """Callback for processing audio data from the input stream.

        Args:
            indata: The input audio data
            frames: Number of frames in the buffer
            time_info: Dictionary containing timing information
            status: Status flags
        """
        try:
            if status:
                logger.warning(f"Audio status: {status}")

            if not self.running:
                return

            # Convert input to float32 and process
            audio_data = indata[:, 0].astype(np.float32)
            self._process_audio(audio_data)

        except Exception as e:
            logger.error(f"Error in _audio_callback: {e}")

    def start(self, callback: Callable[[DetectedNote, float], None]) -> bool:
        """Start note detection

        Args:
            callback: Function to call when a note is detected.
                     The function should accept two parameters:
                     - note (DetectedNote): The detected note
                     - signal_strength (float): The strength of the detected signal (0-1)
        Returns:
            bool: True if started successfully, False otherwise
        """
        if self.running:
            logger.warning("Note detector is already running")
            return False

        self.callback = callback
        self.running = True

        try:
            # Double the blocksize to match what the pitch detector expects (2048 samples)
            self.stream = sd.InputStream(
                device=self.device_id,
                samplerate=self.sample_rate,
                channels=self.CHANNELS,
                callback=self._audio_callback,
                blocksize=self.FRAMES_PER_BUFFER
                * 2,  # Match pitch detector's expected size
                dtype=np.float32,
                latency="high",
            )
            self.stream.start()
            logger.info(
                f"Started note detection on device: {self.device_info['name'] if self.device_info else 'default'}"
            )
            logger.debug("Note detection started - stable notes will be logged")
            return True
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self.running = False
            return False

        try:
            # Double the blocksize to match what the pitch detector expects (2048 samples)
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.CHANNELS,
                callback=audio_callback,
                blocksize=self.FRAMES_PER_BUFFER
                * 2,  # Match pitch detector's expected size
                dtype=np.float32,
            )
            self.stream.start()
            logger.info("Started audio stream")
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            raise

    def stop(self) -> None:
        """Stop the note detector and clean up resources."""
        if not self.running:
            return

        self.running = False

        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
                logger.info("Stopped audio stream")
            except Exception as e:
                logger.error(f"Error stopping audio stream: {e}")
            finally:
                self.stream = None

    def _process_audio(self, audio_data: np.ndarray) -> None:
        """Process a chunk of audio data and detect notes."""
        if not self.running or self.callback is None:
            return

        # Calculate signal level (normalized RMS)
        signal_level = np.sqrt(np.mean(audio_data**2))

        # Skip if signal is too weak
        if signal_level < self.min_signal:
            return

        # Detect pitch
        frequency = self.pitch_detector(audio_data)[0]
        confidence = self.pitch_detector.get_confidence()

        # Skip if confidence is too low or frequency is out of range
        if (
            confidence < self.min_confidence
            or frequency < self.MIN_FREQUENCY
            or frequency > self.MAX_FREQUENCY
        ):
            return

        # Convert frequency to note name
        note_name = self._frequency_to_note(frequency)

        # Track note stability
        self.recent_notes.append(note_name)

        # Check if we have a stable note
        is_stable = self._check_note_stability()

        # Create detected note object
        detected_note = DetectedNote(
            name=note_name,
            frequency=frequency,
            confidence=confidence,
            signal=signal_level,
            is_stable=is_stable,
            timestamp=time.time(),
        )

        # Only update stable note if it's actually stable
        if is_stable:
            self.current_stable_note = detected_note

        # Call the callback with the detected note and signal strength
        try:
            self.callback(detected_note, signal_level)
        except Exception as e:
            logger.error(f"Error in note detection callback: {e}")

    def _check_note_stability(self) -> bool:
        """Check if the current note is stable based on recent detections."""
        if not self.recent_notes or len(self.recent_notes) < self.STABILITY_WINDOW:
            return False

        # Get the most recent note to check
        current_note = self.recent_notes[-1]

        # Count how many of the recent notes match the current note
        matching_notes = sum(1 for note in self.recent_notes if note == current_note)

        # Calculate the stability ratio
        stability_ratio = matching_notes / len(self.recent_notes)

        return stability_ratio >= self.STABILITY_THRESHOLD

    def _frequency_to_note(self, frequency: float) -> str:
        """Convert a frequency in Hz to the nearest note name."""
        if frequency <= 0:
            return ""

        # Calculate the number of half steps from A4
        n = 12 * np.log2(frequency / self.A4_FREQ)
        note_num = round(n) + 57  # 57 is the number of semitones from C0 to A4

        # Get note name and octave
        note_name = self.NOTE_NAMES[note_num % 12]
        octave = (note_num // 12) - 1  # C0 is octave 0

        return f"{note_name}{octave}"

        logger.info("Initializing audio device for note detection...")
        try:
            self._init_audio_device()
        except Exception as e:
            logger.error(f"Failed to initialize audio device: {e}")
            raise

        logger.info(
            f"Initialized NoteDetector with sample_rate={self.sample_rate}Hz, "
            f"min_confidence={self.min_confidence}, min_signal={self.min_signal}"
        )

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
        if self.pitch_detector:
            self.pitch_detector.set_silence(self._silence_threshold_db)

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
        if hasattr(self, 'pitch_detector'):
            self.pitch_detector.set_tolerance(self._tolerance)

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
        self._stability_majority = max(0.0, min(1.0, float(value)))  # Clamp to 0-1 range

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
        return self._min_frequency if hasattr(self, '_min_frequency') else float(self.MIN_FREQUENCY)

    @min_frequency.setter
    def min_frequency(self, value: float) -> None:
        """Set the minimum frequency to consider for note detection.
        
        Args:
            value: New minimum frequency in Hz (must be positive)
        """
        self._min_frequency = max(1.0, float(value))  # Ensure positive value

    def get_stability_params(self):
        """Get all current stability parameter values"""
        return {
            "silence_threshold_db": self._silence_threshold_db,
            "tolerance": self._tolerance,
            "min_stable_count": self._min_stable_count,
            "stability_majority": self._stability_majority,
            "group_hz": self._group_hz,
            "snap_percent": self._snap_percent,
            "normal_majority": self._normal_majority,
            "min_confidence": self._min_confidence,
            "min_signal": self._min_signal,
            "min_frequency": self._min_frequency,
        }

    def _init_audio_device(self):
        """Initialize the audio input device"""
        try:
            # If no device ID specified, try to find Rocksmith adapter
            if self.device_id is None:
                self.device_id, self.device_info = self._find_rocksmith_adapter()

            # If still no device, use default
            if self.device_id is None:
                self.device_id = sd.default.device[0]
                self.device_info = sd.query_devices(self.device_id)
            elif self.device_info is None:
                self.device_info = sd.query_devices(self.device_id)

            # Check if device has input channels
            if self.device_info["max_input_channels"] == 0:
                raise ValueError(
                    f"Selected device {self.device_id} has no input channels"
                )

            # Get device parameters
            self.sample_rate = int(self.device_info["default_samplerate"])
            self.buffer_size = 2048  # Larger buffer for better low frequency detection

            # Initialize pitch detection
            self.pitch_detector = aubio.pitch(
                method="yin",  # Best for guitar/bass
                buf_size=self.buffer_size,
                hop_size=self.buffer_size,
                samplerate=self.sample_rate,
            )
            self.pitch_detector.set_unit("Hz")
            self.pitch_detector.set_silence(
                self.silence_threshold_db
            )  # Exposed silence threshold
            self.pitch_detector.set_tolerance(
                self.tolerance
            )  # Exposed tolerance for responsiveness

            logger.info(
                f"Initialized audio device: {self.device_info['name']} (ID: {self.device_id})"
            )
            logger.debug(
                f"Audio settings - Sample rate: {self.sample_rate} Hz, "
                f"Buffer size: {self.buffer_size}"
            )
            logger.debug(
                f"Detection settings - Silence threshold: {self.silence_threshold_db} dB, "
                f"Tolerance: {self.tolerance}"
            )

        except Exception as e:
            logger.error(f"Error initializing audio device: {e}")
            raise RuntimeError(f"Error initializing audio device: {e}")

    def _find_rocksmith_adapter(self):
        """Find the Rocksmith USB Guitar Adapter in the device list"""
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if "rocksmith" in dev["name"].lower() and dev["max_input_channels"] > 0:
                logger.info(f"Found Rocksmith adapter: {dev['name']} (ID: {i})")
                return i, dev
        logger.warning("Rocksmith adapter not found, using default input device")
        return None, None

    def get_note_name(self, freq):
        """Convert frequency to note name

        Args:
            freq: Frequency in Hz

        Returns:
            Note name with octave (e.g., 'A4', 'C#3')
        """
        if freq <= 0:
            return "---"

        # Standard reference: A4 = 440Hz
        # Calculate half steps from A4
        half_steps = round(12 * np.log2(freq / 440.0))

        # Calculate octave (A4 is in octave 4)
        octave = 3 + (half_steps + 9) // 12

        # Get note name (0 = A, 1 = A#, etc.)
        notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
        note_idx = (half_steps % 12 + 12) % 12  # Ensure positive index

        return f"{notes[note_idx]}{octave}"

    def get_stable_note(self) -> Optional[DetectedNote]:
        """Determine if there's a stable note in the history

        Returns:
            A DetectedNote if a stable note is found, None otherwise
        """
        logger.debug(
            f"Current note history ({len(self.note_history)}): "
            f"{[f'{n.name}({n.confidence:.2f},{n.signal:.3f})' for n in self.note_history]}"
        )

        # Filter notes by frequency, confidence, and signal
        valid_notes = [
            n
            for n in self.note_history
            if n.frequency >= self._min_frequency
            and getattr(n, "confidence", 1.0) >= self._min_confidence
            and getattr(n, "signal", 1.0) >= self._min_signal
        ]

        logger.debug(
            f"Valid readings: {len(valid_notes)}/{len(self.note_history)} | "
            f"min_conf: {self._min_confidence} min_sig: {self._min_signal} min_freq: {self._min_frequency}"
        )

        if valid_notes:
            logger.debug(
                f"Valid note range: {min(n.frequency for n in valid_notes):.1f}Hz - {max(n.frequency for n in valid_notes):.1f}Hz"
            )

        if len(valid_notes) < self.min_stable_count:
            logger.debug(
                f"Not enough valid readings for stability: "
                f"{len(valid_notes)} < {self.min_stable_count} (min_stable_count)"
            )
            if self.stable_note is not None:
                logger.debug(f"Clearing stable note: {self.stable_note.name}")
            self.stable_note = None
            return None

        # Group frequencies that are close to each other (within group_hz)
        freq_groups = []
        for note in valid_notes:
            # Check if this frequency fits in an existing group
            found_group = False
            for group in freq_groups:
                group_avg = sum(n.frequency for n in group) / len(group)
                freq_diff = abs(note.frequency - group_avg)
                if freq_diff < self.group_hz:
                    group.append(note)
                    found_group = True
                    logger.debug(
                        f"Grouped {note.name} ({note.frequency:.1f}Hz) with "
                        f"group avg {group_avg:.1f}Hz (diff: {freq_diff:.1f} < {self.group_hz}Hz)"
                    )
                    break

            # If no matching group, create a new one
            if not found_group:
                freq_groups.append([note])
                logger.debug(
                    f"Created new group for {note.name} ({note.frequency:.1f}Hz)"
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
            self.stable_note = None
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
            name=self.get_note_name(avg_freq),
            frequency=avg_freq,
            confidence=avg_confidence,
            signal=avg_signal,
            is_stable=True,
            timestamp=most_recent.timestamp,
        )

        # Check if we had a previous stable note
        if self.stable_note is not None:
            # If the note name changed, check if it's a significant change
            if new_note.name != self.stable_note.name:
                # Check if the frequency change is significant
                freq_diff = abs(new_note.frequency - self.stable_note.frequency)
                if freq_diff < self.group_hz:
                    # Not a significant change, keep the previous note
                    logger.debug(
                        f"Ignoring small frequency change: "
                        f"{self.stable_note.name} -> {new_note.name} "
                        f"(diff: {freq_diff:.1f}Hz < {self.group_hz}Hz)"
                    )
                    return self.stable_note
            else:
                if self._last_stable_note != self.stable_note.name:
                    logger.debug(
                        f"Stable note held: {self.stable_note.name} "
                        f"(confidence: {avg_confidence * 100:.1f}%, signal: {avg_signal:.3f})"
                    )
                self._last_stable_note = self.stable_note.name
                return DetectedNote(
                    self.stable_note.name,
                    self.stable_note.frequency,
                    avg_confidence,
                    avg_signal,
                    True,
                    time.time(),
                )

        # If we get here, we have a new stable note
        if self._last_stable_note != new_note.name:
            logger.info(
                f"New stable note: {new_note.name} "
                f"({len(largest_group)}/{len(valid_notes)} votes, {avg_confidence * 100:.1f}% confidence)"
            )

        self._last_stable_note = new_note.name
        return new_note

    def _audio_callback(self, indata, frames, stream_time, status):
        """Callback for processing audio data"""
        try:
            if status:
                if status.input_overflow:
                    logger.warning("Input overflow")
                    return  # Skip processing this buffer on overflow

            # Get the audio data and check levels
            audio_data = indata[:, 0] if indata.shape[1] > 0 else indata.flatten()
            signal_max = np.max(np.abs(audio_data))

            # Buffer size check: skip if not expected length
            if len(audio_data) != self.buffer_size:
                logger.debug(
                    f"Skipping frame: got {len(audio_data)} samples, expected {self.buffer_size}"
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
                raw_pitch = self.pitch_detector(audio_data.astype(np.float32))
                pitch = raw_pitch[0]  # The frequency in Hz
                confidence = self.pitch_detector.get_confidence()

                # Apply a window function to reduce spectral leakage
                window = np.hanning(len(audio_data))
                windowed_data = audio_data * window

                # Calculate FFT with zero-padding for better frequency resolution
                n_fft = 4 * len(
                    audio_data
                )  # Zero-padding for better frequency resolution
                fft = np.fft.rfft(windowed_data, n=n_fft)
                fft_freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)

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
                        key=lambda note: abs(get_note_freq(NoteDetector.STANDARD_NOTES.index(note)) - dom_freq)
                    )
                    note_name = closest_note
                    note_freq = get_note_freq(NoteDetector.STANDARD_NOTES.index(note_name))

                    # If we're within 5% of a standard note, snap to that frequency
                    if abs(dom_freq - note_freq) / note_freq < 0.05:
                        dom_freq = note_freq
                else:
                    dom_freq = 0

                current_time = time.strftime("%H:%M:%S")
                if pitch > 0:
                    note_name = self.get_note_name(pitch)
                    dom_note = self.get_note_name(dom_freq) if dom_freq > 0 else "---"
                    logger.debug(
                        f"{current_time} | Aubio: {pitch:.1f} Hz ({note_name}) | FFT: {dom_freq:.1f} Hz ({dom_note}) | Conf: {confidence:.2f} | Sig: {signal_max:.3f}"
                    )
                else:
                    dom_note = self.get_note_name(dom_freq) if dom_freq > 0 else "---"
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

                # If signal is weak (below 0.05), maintain the current stable note
                # This prevents jumping between notes during decay
                if signal_max < 0.05 and self.stable_note:
                    logger.debug(
                        f"Signal weak ({signal_max:.4f} < 0.05), "
                        f"maintaining current note: {self.stable_note.name}"
                    )
                    detected_freq = self.stable_note.frequency

                # Filter out unreasonable frequencies
                if (
                    detected_freq > 0 and 30 < detected_freq < 1000
                ):  # Reasonable range for guitar/bass
                    # Convert frequency to note name
                    note_name = self.get_note_name(detected_freq)
                    detected = DetectedNote(
                        note_name,
                        detected_freq,
                        confidence,
                        signal_max,
                        False,
                        time.time(),
                    )
                    self.note_history.append(detected)

                    # Track previous stable note for change detection
                    prev_stable_note = self.stable_note

                    # Get stable note
                    new_stable_note = self.get_stable_note()
                    if new_stable_note:
                        self.stable_note = new_stable_note
                        if self.callback:
                            self.callback(self.stable_note, signal_max)

                    # Log stable note changes
                    stable_note_changed = (
                        (prev_stable_note is None and new_stable_note is not None)
                        or (prev_stable_note is not None and new_stable_note is None)
                        or (
                            prev_stable_note
                            and new_stable_note
                            and (
                                prev_stable_note.name != new_stable_note.name
                                or abs(
                                    prev_stable_note.confidence
                                    - new_stable_note.confidence
                                )
                                > 0.1
                            )
                        )
                    )

                    if stable_note_changed and prev_stable_note:
                        logger.info(f"Note released: {prev_stable_note.name}")
                    elif not new_stable_note and not prev_stable_note and detected:
                        # Only log current note if we don't have a stable note and it's not just silence
                        self.current_note = detected
                        logger.debug(
                            f"{current_time} | Detected: {self.current_note.name} | "
                            f"{self.current_note.frequency:.1f} Hz | "
                            f"Conf: {self.current_note.confidence:.2f} | "
                            f"Signal: {signal_max:.4f}"
                        )

            else:
                current_time = time.strftime("%H:%M:%S")
                logger.debug(
                    f"{current_time} | Waiting for input | Signal: {signal_max:.4f} | dB: {db:.1f}"
                )
        except Exception as e:
            logger.error(f"Error in audio callback: {e}")

    def stop(self):
        """Stop note detection"""
        self.running = False
        if getattr(self, "stream", None):
            self.stream.stop()
            self.stream.close()
            self.stream = None
        logger.info("Stopped note detection")

    def get_current_note(self) -> Optional[DetectedNote]:
        """Get the current detected note
        Returns:
            The current stable note, or None if no note is detected
        """
        return getattr(self, "stable_note", None)

    def get_simple_note(self) -> Optional[str]:
        """Get just the note letter (A, B, C, etc.) without the octave
        Returns:
            The note letter, or None if no note is detected
        """
        if getattr(self, "stable_note", None):
            return self.stable_note.name[0]
        return None

    def is_note_playing(self, target_note: str) -> bool:
        """Check if a specific note is currently playing
        Args:
            target_note: The target note letter (A, B, C, etc.)
        Returns:
            True if the target note is currently playing, False otherwise
        """
        current = self.get_simple_note()
        return current == target_note
