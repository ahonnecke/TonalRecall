import numpy as np
import aubio
import sounddevice as sd
import time
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List


@dataclass
class DetectedNote:
    name: str  # Note name (e.g., 'A4', 'C#3')
    frequency: float  # Frequency in Hz
    confidence: float  # Detection confidence (0-1)
    signal: float  # Signal strength (0-1, e.g., max(abs(audio)))
    is_stable: bool  # Whether this is a stable note


class NoteDetector:
    # ... existing code ...
    # Add new default thresholds as class-level constants
    DEFAULT_MIN_CONFIDENCE = 0.0
    DEFAULT_MIN_SIGNAL = 0.02
    DEFAULT_MIN_FREQUENCY = 30.0

    """A class for detecting musical notes from audio input"""

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
        debug=False,
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
            debug: Whether to print debug information
            silence_threshold_db: Silence threshold in dB (default -90)
            tolerance: aubio pitch tolerance (default 0.3)
            min_stable_count: Minimum valid readings for stability (default 3)
            stability_majority: % of readings required to change stable note (default 0.7)
            group_hz: Frequency grouping window in Hz (default 5)
            snap_percent: Snap to standard note if within this percent (default 0.05)
            normal_majority: % of readings required for initial stability (default 0.5)
        """
        self.debug = debug
        self.device_id = device_id
        self.device_info = None
        self.stream = None
        self.running = False

        # Exposed parameters for strictness
        self._silence_threshold_db = silence_threshold_db
        self._tolerance = tolerance
        self._min_stable_count = min_stable_count
        self._stability_majority = stability_majority
        self._group_hz = group_hz
        self._snap_percent = snap_percent
        self._normal_majority = normal_majority

        # New tunable thresholds for filtering
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

        # Find and initialize audio device
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
        """Minimum valid readings for stability (default 3)"""
        return self._min_stable_count

    @min_stable_count.setter
    def min_stable_count(self, value):
        self._min_stable_count = value

    @property
    def stability_majority(self):
        """% of readings required to change stable note (default 0.7)"""
        return self._stability_majority

    @stability_majority.setter
    def stability_majority(self, value):
        self._stability_majority = value

    @property
    def group_hz(self):
        """Frequency grouping window in Hz (default 5)"""
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
        """Minimum confidence for a valid reading (default 0.5)"""
        return self._min_confidence

    @min_confidence.setter
    def min_confidence(self, value):
        self._min_confidence = value

    @property
    def min_signal(self):
        """Minimum signal strength for a valid reading (default 0.02)"""
        return self._min_signal

    @min_signal.setter
    def min_signal(self, value):
        self._min_signal = value

    @property
    def min_frequency(self):
        """Minimum frequency for a valid reading (default 30.0)"""
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
                method="yinfft",  # Best for guitar/bass
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

            if self.debug:
                print(
                    f"Initialized audio device: {self.device_info['name']} (ID: {self.device_id})"
                )
                print(
                    f"Sample rate: {self.sample_rate} Hz, Buffer size: {self.buffer_size}"
                )
                print(
                    f"Silence threshold: {self.silence_threshold_db} dB, Tolerance: {self.tolerance}"
                )

        except Exception as e:
            raise RuntimeError(f"Error initializing audio device: {e}")

    def _find_rocksmith_adapter(self):
        """Find the Rocksmith USB Guitar Adapter in the device list"""
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if (
                    device["max_input_channels"] > 0
                    and "rocksmith" in device["name"].lower()
                ):
                    if self.debug:
                        print(f"Found Rocksmith adapter: {device['name']} (ID: {i})")
                    return i, device
            return None, None
        except Exception as e:
            if self.debug:
                print(f"Error listing audio devices: {e}")
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
        octave = 4 + (half_steps + 9) // 12

        # Get note name (0 = A, 1 = A#, etc.)
        notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
        note_idx = (half_steps % 12 + 12) % 12  # Ensure positive index

        return f"{notes[note_idx]}{octave}"

    def get_stable_note(self) -> Optional[DetectedNote]:
        """Determine if there's a stable note in the history

        Returns:
            A DetectedNote if a stable note is found, None otherwise
        """
        # Filter notes by frequency, confidence, and signal
        valid_notes = [
            n
            for n in self.note_history
            if n.frequency >= self._min_frequency
            and getattr(n, "confidence", 1.0) >= self._min_confidence
            and getattr(n, "signal", 1.0) >= self._min_signal
        ]
        if self.debug:
            print(
                f"[NoteDetector] Valid readings: {len(valid_notes)}/{len(self.note_history)} | min_conf: {self._min_confidence} min_sig: {self._min_signal} min_freq: {self._min_frequency}"
            )
        if len(valid_notes) < self.min_stable_count:
            # Not enough valid readings: set to None or keep previous, but log
            if self.debug and self.stable_note is not None:
                print(
                    "[NoteDetector] Not enough valid readings for stability, clearing stable note."
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
                if abs(note.frequency - group_avg) < self.group_hz:
                    group.append(note)
                    found_group = True
                    break

            # If no matching group, create a new one
            if not found_group:
                freq_groups.append([note])

        # Find the largest group
        if not freq_groups:
            return (
                self.stable_note
            )  # Return the current stable note to maintain stability

        largest_group = max(freq_groups, key=len)

        # Calculate average frequency for this group
        avg_freq = sum(n.frequency for n in largest_group) / len(largest_group)
        avg_conf = sum(n.confidence for n in largest_group) / len(largest_group)
        avg_signal = sum(n.signal for n in largest_group) / len(largest_group)

        # Find the closest standard note
        closest_note = min(
            self.STANDARD_NOTES.items(), key=lambda x: abs(x[1] - avg_freq)
        )
        note_name_std, freq_std = closest_note

        # If we're within snap_percent of a standard note, snap to that frequency
        if abs(avg_freq - freq_std) / freq_std < self.snap_percent:
            avg_freq = freq_std
            note_name = note_name_std
        else:
            # Get the note name from the average frequency
            note_name = self.get_note_name(avg_freq)

        # Apply hysteresis - if we have a stable note already, require a stronger consensus to change
        if self.stable_note:
            # If the new note is different from the current stable note
            if note_name != self.stable_note.name:
                if len(largest_group) < len(valid_notes) * self.stability_majority:
                    if self.debug:
                        print(
                            f"[NoteDetector] Hysteresis: Keeping previous stable note {self.stable_note.name} (not enough majority for change)"
                        )
                    return self.stable_note
            else:
                if self.debug and (self._last_stable_note != self.stable_note.name):
                    print(f"[NoteDetector] Stable note held: {self.stable_note.name}")
                self._last_stable_note = self.stable_note.name
                return DetectedNote(
                    self.stable_note.name,
                    self.stable_note.frequency,
                    avg_conf,
                    avg_signal,
                    True,
                )
        if self.debug and (self._last_stable_note != note_name):
            print(f"[NoteDetector] Stable note set to: {note_name}")
        self._last_stable_note = note_name
        return DetectedNote(note_name, avg_freq, avg_conf, avg_signal, True)
        return DetectedNote(note_name, avg_freq, avg_conf, True)
    
    def _audio_callback(self, indata, frames, stream_time, status):
        """Callback for processing audio data"""
        if status:
            if status.input_overflow:
                if self.debug:
                    print("Input overflow")
                return  # Skip processing this buffer on overflow
            elif self.debug:
                print(f"Status: {status}")
        
        # Get the audio data and check levels
        audio_data = indata[:, 0] if indata.shape[1] > 0 else indata.flatten()
        signal_max = np.max(np.abs(audio_data))
        
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
                n_fft = 4 * len(audio_data)  # Zero-padding for better frequency resolution
                fft = np.fft.rfft(windowed_data, n=n_fft)
                fft_freqs = np.fft.rfftfreq(n_fft, 1.0/self.sample_rate)
                
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
                    closest_note = min(self.STANDARD_NOTES.items(), key=lambda x: abs(x[1] - dom_freq))
                    note_name, note_freq = closest_note
                    
                    # If we're within 5% of a standard note, snap to that frequency
                    if abs(dom_freq - note_freq) / note_freq < 0.05:
                        dom_freq = note_freq
                else:
                    dom_freq = 0
                
                if self.debug:
                    current_time = time.strftime("%H:%M:%S")
                    if pitch > 0:
                        note_name = self.get_note_name(pitch)
                        dom_note = self.get_note_name(dom_freq) if dom_freq > 0 else "---"
                        print(f"{current_time} | Aubio: {pitch:.1f} Hz ({note_name}) | FFT: {dom_freq:.1f} Hz ({dom_note}) | Conf: {confidence:.2f} | Sig: {signal_max:.3f}")
                    else:
                        dom_note = self.get_note_name(dom_freq) if dom_freq > 0 else "---"
                        print(f"{current_time} | Aubio: {pitch:.1f} Hz | FFT: {dom_freq:.1f} Hz ({dom_note}) | Conf: {confidence:.2f} | Sig: {signal_max:.3f}")
                
                # For bass guitar, prioritize FFT analysis which is more reliable for low frequencies
                # For higher frequencies, aubio might be more accurate
                detected_freq = dom_freq if 30 < dom_freq < 200 else pitch
                
                # If aubio returns 0 or a very high frequency, use the FFT frequency
                if pitch == 0 or pitch > 1000:
                    if 30 < dom_freq < 1000:
                        detected_freq = dom_freq
                        
                # If signal is weak (below 0.15), maintain the current stable note
                # This prevents jumping between notes during decay
                if signal_max < 0.15 and self.stable_note:
                    detected_freq = self.stable_note.frequency
                
                # Filter out unreasonable frequencies
                if detected_freq > 0 and 30 < detected_freq < 1000:  # Reasonable range for guitar/bass
                    # Convert frequency to note name
                    note_name = self.get_note_name(detected_freq)
                    detected = DetectedNote(note_name, detected_freq, confidence, signal_max, False)
                    self.note_history.append(detected)
                    
                    # Get stable note
                    new_stable_note = self.get_stable_note()
                    if new_stable_note:
                        self.stable_note = new_stable_note
                        if self.note_callback:
                            self.note_callback(self.stable_note, signal_max)
                    
                    if self.debug:
                        if new_stable_note:
                            print(f"● STABLE: {self.stable_note.name:4} | {self.stable_note.frequency:6.1f} Hz | {self.stable_note.confidence:4.2f}")
                        elif self.stable_note:
                            print(f"○ RECENT: {self.stable_note.name:4} | {self.stable_note.frequency:6.1f} Hz | {self.stable_note.confidence:4.2f}")
                        else:
                            self.current_note = detected
                            print(f"  CURRENT: {self.current_note.name:4} | {self.current_note.frequency:6.1f} Hz | {self.current_note.confidence:4.2f}")
            except ValueError as e:
                if self.debug:
                    current_time = time.strftime("%H:%M:%S")
                    print(f"{current_time} | Error: {e}")
        elif self.debug:
            current_time = time.strftime("%H:%M:%S")
            print(f"{current_time} | Waiting for input... | Signal: {signal_max:.4f} | dB: {db:.1f}")
    
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
            if self.debug:
                print(f"Started note detection on device: {self.device_info['name']}")
                print(
                    "● = stable note, ○ = previously stable note, no symbol = current reading"
                )
            return True
        except Exception as e:
            self.running = False
            if self.debug:
                print(f"Error starting note detection: {e}")
            return False

    def stop(self):
        """Stop note detection"""
        self.running = False
        if getattr(self, "stream", None):
            self.stream.stop()
            self.stream.close()
            self.stream = None
        if self.debug:
            print("Stopped note detection")

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
