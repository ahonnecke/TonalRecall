import numpy as np
import aubio
import sounddevice as sd
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional

from .logging_config import get_logger

# Get logger for this module
note_detector_logger = get_logger("tonal_recall.note_detector")


@dataclass
class DetectedNote:
    name: str  # Note name (e.g., 'A4', 'C#3')
    frequency: float  # Frequency in Hz
    confidence: float  # Detection confidence (0-1)
    signal: float  # Signal strength (0-1, e.g., max(abs(audio)))
    is_stable: bool  # Whether this is a stable note
    timestamp: float  # Timestamp when the note was detected


class NoteDetector:
    """A class for detecting musical notes from audio input"""

    # Default thresholds
    DEFAULT_MIN_CONFIDENCE = 0.0
    DEFAULT_MIN_SIGNAL = 0.02
    DEFAULT_MIN_FREQUENCY = 30.0

    # Class attributes (will be overridden in __init__)
    pitch_detector = None

    """A class for detecting musical notes from audio input"""

    # Default signal threshold (lowered from 0.02 to 0.005 for better sensitivity)
    DEFAULT_MIN_SIGNAL = 0.005

    # Standard bass/guitar frequencies
    STANDARD_NOTES = {
        "E1": 41.2,  # Bass low E
        "A1": 55.0,  # Bass A
        "D2": 73.4,  # Bass D
        "G2": 98.0,  # Bass G
        "E2": 82.4,  # Guitar low E
        "A2": 110.0,  # Guitar A
        "D3": 146.8,  # Guitar D
        "G3": 196.0,  # Guitar G
        "B3": 246.9,  # Guitar B
        "E4": 329.6,  # Guitar high E
    }

    def __init__(
        self,
        device_id=None,
        silence_threshold_db=-90,
        tolerance=0.3,
        min_stable_count=3,
        stability_majority=0.7,
        group_hz=5,
        snap_percent=0.05,
        normal_majority=0.5,
        min_confidence=None,
        min_signal=None,
        min_frequency=None,
    ):
        """Initialize the note detector

        Args:
            device_id: Audio input device ID, or None to auto-detect
            silence_threshold_db: Silence threshold in dB (default -90)
            tolerance: aubio pitch tolerance (default 0.3)
            min_stable_count: Minimum consecutive stable detections for note stability (default 3)
            stability_majority: % of readings required for stability (default 0.7)
            group_hz: Group frequencies within this Hz as the same note (default 5)
            snap_percent: Snap to standard note if within this percent (default 0.05)
            normal_majority: % of readings required for initial stability (default 0.5)
            min_confidence: Minimum confidence for aubio pitch (default 0.0)
            min_signal: Minimum signal amplitude (default 0.02)
            min_frequency: Minimum frequency to consider (default 30.0)
        """
        self.device_id = device_id
        self.silence_threshold_db = silence_threshold_db
        self.tolerance = tolerance
        self.min_stable_count = min_stable_count
        self.stability_majority = stability_majority
        self._group_hz = group_hz
        self._snap_percent = snap_percent
        self._normal_majority = normal_majority
        self._min_confidence = (
            min_confidence
            if min_confidence is not None
            else self.DEFAULT_MIN_CONFIDENCE
        )
        self._min_signal = (
            min_signal if min_signal is not None else self.DEFAULT_MIN_SIGNAL
        )
        self._min_frequency = (
            min_frequency if min_frequency is not None else self.DEFAULT_MIN_FREQUENCY
        )

        # Note detection state
        self.note_history = deque(maxlen=10)  # Store last 10 detected notes
        self.current_note = None
        self.stable_note = None
        self._last_stable_note = None  # For debug logging

        # Callback function for note detection
        self.note_callback = None

        # Always initialize these attributes so they exist even if init fails
        self.device_info = None
        self.sample_rate = None
        self.buffer_size = None

        # Find and initialize audio device
        note_detector_logger.info("Initializing audio device for note detection...")
        self._init_audio_device()

    @property
    def silence_threshold_db(self):
        """Silence threshold in dB (default -90)"""
        return self._silence_threshold_db

    @silence_threshold_db.setter
    def silence_threshold_db(self, value):
        self._silence_threshold_db = value
        if self.pitch_detector:
            self.pitch_detector.set_silence(self._silence_threshold_db)

    @property
    def tolerance(self):
        """aubio pitch tolerance (default 0.3)"""
        return self._tolerance

    @tolerance.setter
    def tolerance(self, value):
        self._tolerance = value
        if self.pitch_detector:
            self.pitch_detector.set_tolerance(self._tolerance)

    @property
    def min_stable_count(self):
        """Minimum consecutive stable detections for note stability (default 3)"""
        return self._min_stable_count

    @min_stable_count.setter
    def min_stable_count(self, value):
        self._min_stable_count = value

    @property
    def stability_majority(self):
        """% of readings required for stability (default 0.7)"""
        return self._stability_majority

    @stability_majority.setter
    def stability_majority(self, value):
        self._stability_majority = value

    @property
    def group_hz(self):
        """Group frequencies within this Hz as the same note (default 5)"""
        return self._group_hz

    @group_hz.setter
    def group_hz(self, value):
        self._group_hz = value

    @property
    def snap_percent(self):
        """Snap to standard note if within this percent (default 0.05)"""
        return self._snap_percent

    @snap_percent.setter
    def snap_percent(self, value):
        self._snap_percent = value

    @property
    def normal_majority(self):
        """% of readings required for initial stability (default 0.5)"""
        return self._normal_majority

    @normal_majority.setter
    def normal_majority(self, value):
        self._normal_majority = value

    @property
    def min_confidence(self):
        """Minimum confidence for aubio pitch (default 0.0)"""
        return self._min_confidence

    @min_confidence.setter
    def min_confidence(self, value):
        self._min_confidence = value

    @property
    def min_signal(self):
        """Minimum signal amplitude (default 0.02)"""
        return self._min_signal

    @min_signal.setter
    def min_signal(self, value):
        self._min_signal = value

    @property
    def min_frequency(self):
        """Minimum frequency to consider (default 30.0)"""
        return self._min_frequency

    @min_frequency.setter
    def min_frequency(self, value):
        self._min_frequency = value

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

            note_detector_logger.info(
                f"Initialized audio device: {self.device_info['name']} (ID: {self.device_id})"
            )
            note_detector_logger.debug(
                f"Audio settings - Sample rate: {self.sample_rate} Hz, "
                f"Buffer size: {self.buffer_size}"
            )
            note_detector_logger.debug(
                f"Detection settings - Silence threshold: {self.silence_threshold_db} dB, "
                f"Tolerance: {self.tolerance}"
            )

        except Exception as e:
            note_detector_logger.error(f"Error initializing audio device: {e}")
            raise RuntimeError(f"Error initializing audio device: {e}")

    def _find_rocksmith_adapter(self):
        """Find the Rocksmith USB Guitar Adapter in the device list"""
        devices = sd.query_devices()
        for i, dev in enumerate(devices):
            if "rocksmith" in dev["name"].lower() and dev["max_input_channels"] > 0:
                note_detector_logger.info(
                    f"Found Rocksmith adapter: {dev['name']} (ID: {i})"
                )
                return i, dev
        note_detector_logger.warning(
            "Rocksmith adapter not found, using default input device"
        )
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
        note_detector_logger.debug(
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

        note_detector_logger.debug(
            f"Valid readings: {len(valid_notes)}/{len(self.note_history)} | "
            f"min_conf: {self._min_confidence} min_sig: {self._min_signal} min_freq: {self._min_frequency}"
        )

        if valid_notes:
            note_detector_logger.debug(
                f"Valid note range: {min(n.frequency for n in valid_notes):.1f}Hz - {max(n.frequency for n in valid_notes):.1f}Hz"
            )

        if len(valid_notes) < self.min_stable_count:
            note_detector_logger.debug(
                f"Not enough valid readings for stability: "
                f"{len(valid_notes)} < {self.min_stable_count} (min_stable_count)"
            )
            if self.stable_note is not None:
                note_detector_logger.debug(
                    f"Clearing stable note: {self.stable_note.name}"
                )
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
                    note_detector_logger.debug(
                        f"Grouped {note.name} ({note.frequency:.1f}Hz) with "
                        f"group avg {group_avg:.1f}Hz (diff: {freq_diff:.1f} < {self.group_hz}Hz)"
                    )
                    break

            # If no matching group, create a new one
            if not found_group:
                freq_groups.append([note])
                note_detector_logger.debug(
                    f"Created new group for {note.name} ({note.frequency:.1f}Hz)"
                )

        # Log all frequency groups
        for i, group in enumerate(freq_groups):
            freqs = [n.frequency for n in group]
            avg_freq = sum(freqs) / len(freqs)
            note_detector_logger.debug(
                f"Group {i}: {len(group)} notes, "
                f"freq range: {min(freqs):.1f}-{max(freqs):.1f}Hz, "
                f"avg: {avg_freq:.1f}Hz"
            )

        # Find the largest group of similar frequencies
        if not freq_groups:
            note_detector_logger.debug("No frequency groups found")
            self.stable_note = None
            return None

        # Sort groups by size (largest first)
        freq_groups.sort(key=len, reverse=True)
        largest_group = freq_groups[0]

        freqs = [n.frequency for n in largest_group]
        avg_freq = sum(freqs) / len(freqs)
        note_detector_logger.debug(
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
                    note_detector_logger.debug(
                        f"Ignoring small frequency change: "
                        f"{self.stable_note.name} -> {new_note.name} "
                        f"(diff: {freq_diff:.1f}Hz < {self.group_hz}Hz)"
                    )
                    return self.stable_note
            else:
                if self._last_stable_note != self.stable_note.name:
                    note_detector_logger.debug(
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
            note_detector_logger.info(
                f"New stable note: {new_note.name} "
                f"({len(largest_group)}/{len(valid_notes)} votes, {avg_confidence * 100:.1f}% confidence)"
            )

        self._last_stable_note = new_note.name
        return new_note

    def _audio_callback(self, indata, frames, stream_time, status):
        """Callback for processing audio data"""
        if status:
            if status.input_overflow:
                note_detector_logger.warning("Input overflow")
                return  # Skip processing this buffer on overflow

        # Get the audio data and check levels
        audio_data = indata[:, 0] if indata.shape[1] > 0 else indata.flatten()
        signal_max = np.max(np.abs(audio_data))

        # Buffer size check: skip if not expected length
        if len(audio_data) != self.buffer_size:
            note_detector_logger.debug(
                f"Skipping frame: got {len(audio_data)} samples, expected {self.buffer_size}"
            )
            return

        # Log detailed audio stats at debug level
        note_detector_logger.debug(
            f"[AUDIO DEBUG] min={audio_data.min():.4f} max={audio_data.max():.4f} mean={audio_data.mean():.4f} dtype={audio_data.dtype}"
        )

        # Calculate RMS to get a better signal level measurement
        rms = np.sqrt(np.mean(audio_data**2))
        db = 20 * np.log10(rms) if rms > 0 else -100  # Convert to dB

        # Increase noise gate threshold to 0.01
        if signal_max > 0.01:  # Basic noise gate
            try:
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

                    # Find the closest bass note
                    closest_note = min(
                        self.STANDARD_NOTES.items(), key=lambda x: abs(x[1] - dom_freq)
                    )
                    note_name, note_freq = closest_note

                    # If we're within 5% of a standard note, snap to that frequency
                    if abs(dom_freq - note_freq) / note_freq < 0.05:
                        dom_freq = note_freq
                else:
                    dom_freq = 0

                current_time = time.strftime("%H:%M:%S")
                if pitch > 0:
                    note_name = self.get_note_name(pitch)
                    dom_note = self.get_note_name(dom_freq) if dom_freq > 0 else "---"
                    note_detector_logger.debug(
                        f"{current_time} | Aubio: {pitch:.1f} Hz ({note_name}) | FFT: {dom_freq:.1f} Hz ({dom_note}) | Conf: {confidence:.2f} | Sig: {signal_max:.3f}"
                    )
                else:
                    dom_note = self.get_note_name(dom_freq) if dom_freq > 0 else "---"
                    note_detector_logger.debug(
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

                note_detector_logger.debug(
                    f"Pitch selection - method: {detected_method}, confidence: {confidence:.2f}, "
                    f"aubio: {pitch:.1f}Hz, fft: {dom_freq:.1f}Hz"
                )

                # If signal is weak (below 0.05), maintain the current stable note
                # This prevents jumping between notes during decay
                if signal_max < 0.05 and self.stable_note:
                    note_detector_logger.debug(
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
                        if self.note_callback:
                            self.note_callback(self.stable_note, signal_max)

                    # Log stable note changes
                    stable_note_changed = (
                        prev_stable_note is None
                        and new_stable_note is not None
                        or prev_stable_note is not None
                        and new_stable_note is None
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

                    if stable_note_changed:
                        if new_stable_note:
                            note_detector_logger.info(
                                f"Stable note: {self.stable_note.name} | "
                                f"{self.stable_note.frequency:.1f} Hz | "
                                f"Conf: {self.stable_note.confidence:.2f}"
                            )
                        elif prev_stable_note:
                            note_detector_logger.info(
                                f"Note released: {prev_stable_note.name}"
                            )
                        elif not new_stable_note and not prev_stable_note and detected:
                            # Only log current note if we don't have a stable note and it's not just silence
                            self.current_note = detected
                            note_detector_logger.debug(
                                f"Current note: {self.current_note.name} | "
                                f"{self.current_note.frequency:.1f} Hz | "
                                f"Conf: {self.current_note.confidence:.2f}"
                            )
            except ValueError as e:
                current_time = time.strftime("%H:%M:%S")
                note_detector_logger.warning(f"{current_time} | Error: {e}")
        else:
            current_time = time.strftime("%H:%M:%S")
            note_detector_logger.debug(
                f"{current_time} | Waiting for input | Signal: {signal_max:.4f} | dB: {db:.1f}"
            )

    def start(self, callback=None):
        """Start note detection

        Args:
            callback: Function to call when a stable note is detected
                     Function signature: callback(note: DetectedNote, signal_strength: float)
        Returns:
            True if started successfully, False otherwise
        """
        if getattr(self, "running", False):
            return True
        try:
            self.note_callback = callback
            self.running = True
            self.stream = sd.InputStream(
                device=self.device_id,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=getattr(self, "buffer_size", 2048),
                callback=self._audio_callback,
                dtype="float32",
                latency="high",
            )
            self.stream.start()
            note_detector_logger.info(
                f"Started note detection on device: {self.device_info['name']}"
            )
            note_detector_logger.debug(
                "Note detection started - stable notes will be logged"
            )
            return True
        except Exception as e:
            self.running = False
            note_detector_logger.error(f"Error starting note detection: {e}")
            return False

    def stop(self):
        """Stop note detection"""
        self.running = False
        if getattr(self, "stream", None):
            self.stream.stop()
            self.stream.close()
            self.stream = None
        note_detector_logger.info("Stopped note detection")

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
