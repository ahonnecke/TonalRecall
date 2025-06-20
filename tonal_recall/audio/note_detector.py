"""Note detection functionality for audio analysis."""

from __future__ import annotations
import numpy as np
import aubio
from collections import deque
from typing import Optional, Dict, Deque, ClassVar, TypeAlias, Callable, Any

from ..logger import get_logger
from ..note_types import DetectedNote
from ..note_utils import get_note_name
from ..core.interfaces import INoteDetector

logger = get_logger(__name__)


class NoteDetector(INoteDetector):
    """Class for detecting musical notes from audio data."""
    
    # Type aliases
    NoteName: TypeAlias = str
    Frequency: TypeAlias = float
    Confidence: TypeAlias = float
    SignalStrength: TypeAlias = float
    
    # Note detection settings
    DEFAULT_MIN_CONFIDENCE: ClassVar[Confidence] = 0.7  # Minimum confidence to consider a note detection valid
    DEFAULT_MIN_SIGNAL: ClassVar[SignalStrength] = 0.005  # Minimum signal level to process (avoids noise)
    MIN_FREQUENCY: ClassVar[float] = 30.0  # Hz - below this is probably noise
    MAX_FREQUENCY: ClassVar[float] = 1000.0  # Hz - above this is probably noise for our purposes
    
    # Note stability tracking
    STABILITY_WINDOW: ClassVar[int] = 5  # Number of consecutive detections to consider stable
    
    def __init__(
        self,
        hop_size: int = 256,
        win_size: int = 512,
        sample_rate: int = 44100,
        min_frequency: float = 30.0,
        min_confidence: float = 0.5,
        min_signal: float = 0.001,
        use_flats: bool = False,
        callback: Optional[Callable] = None,
        tolerance: float = 0.8,
        min_stable_count: int = 3,
        stability_majority: float = 0.6,
        group_hz: float = 5.0,
        snap_percent: float = 0.0,
        harmonic_correction: bool = True,  # Enable harmonic correction for low notes
    ) -> None:
        """Initialize the NoteDetector.
        
        Args:
            hop_size: Hop size for the pitch detector (typically 512)
            win_size: Window size for the pitch detector (typically 1024)
            sample_rate: Audio sample rate in Hz
            min_frequency: Minimum frequency to consider for note detection
            min_confidence: Minimum confidence to consider a note detection valid
            min_signal: Minimum signal level to process (avoids noise)
            use_flats: If True, use flat notes (e.g., 'Bb') instead of sharps (e.g., 'A#')
            callback: Optional callback function for note detection events
            tolerance: Aubio pitch detection tolerance (0.0 to 1.0)
            min_stable_count: Minimum number of consecutive detections to consider a note stable
            stability_majority: Proportion of detections that must match to be stable
            group_hz: Frequency grouping in Hz
            snap_percent: Snap percentage for note detection
            harmonic_correction: Enable harmonic correction for low notes
        """
        # Initialize parameters
        self._sample_rate = sample_rate
        self._hop_size = hop_size
        self._tolerance = tolerance
        self._min_confidence = min_confidence
        self._min_signal = min_signal
        self._min_frequency = min_frequency
        self._use_flats = use_flats
        self._callback = callback
        self._min_stable_count = min_stable_count
        self._stability_majority = stability_majority
        self._group_hz = group_hz
        self._snap_percent = snap_percent
        self._harmonic_correction = harmonic_correction
        
        
        
        # Initialize aubio pitch detection
        self._pitch_detector = aubio.pitch("yin", self._hop_size * 2, self._hop_size, self._sample_rate)
        self._pitch_detector.set_unit("Hz")
        self._pitch_detector.set_tolerance(self._tolerance)
        
        # Stability parameters already initialized in constructor
        
        # Initialize note history and current state
        self._note_history: Deque[str] = deque(maxlen=self.STABILITY_WINDOW)
        self._current_note: Optional[str] = None
        self._stable_note: Optional[DetectedNote] = None
        self._stable_count = 0
        
        # Frequency grouping for stability detection
        self._group_hz = 1.0  # Frequency grouping in Hz
        self._snap_percent = 0.05  # Snap percentage for note detection
        
        logger.info(f"Note detector initialized: sample_rate={sample_rate}, win_size={win_size}, hop_size={hop_size}")
    
    def process_audio(self, audio_data: np.ndarray, timestamp: float) -> Optional[DetectedNote]:
        """Process audio data and detect notes.
        
        Args:
            audio_data: Numpy array containing audio samples (1D float32 array)
            timestamp: Current timestamp in seconds
            
        Returns:
            DetectedNote if a stable note is detected, None otherwise
        """
        try:
            # Ensure audio data is the right format (float32)
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
                
            # Check if audio data size matches expected hop size
            if len(audio_data) != self._hop_size:
                # Resize or pad audio data to match hop size
                if len(audio_data) > self._hop_size:
                    # Truncate if too large
                    audio_data = audio_data[:self._hop_size]
                else:
                    # Pad with zeros if too small
                    padding = np.zeros(self._hop_size - len(audio_data), dtype=audio_data.dtype)
                    audio_data = np.concatenate((audio_data, padding))
            
            # Get the audio data and check levels
            signal_max = float(np.max(np.abs(audio_data)))
            
            # Calculate RMS to get a better signal level measurement
            rms = float(np.sqrt(np.mean(audio_data**2)))
            db = 20 * np.log10(rms) if rms > 0 else -100  # Convert to dB
            
            # Log detailed audio stats at debug level
            logger.debug(
                f"[AUDIO DEBUG] min={audio_data.min():.4f} max={audio_data.max():.4f} "
                f"mean={audio_data.mean():.4f} rms={rms:.4f} db={db:.1f} dtype={audio_data.dtype}"
            )
            
            # Apply a noise gate threshold - this is critical for detection
            if signal_max > 0.01:  # Basic noise gate - same as baseline
                # Apply a window function to reduce spectral leakage
                window = np.hanning(len(audio_data))
                audio_data * window
                
                # Get pitch and confidence from aubio
                pitch = float(self._pitch_detector(audio_data)[0])
                confidence = float(self._pitch_detector.get_confidence())
                
                # Skip if confidence is too low or pitch is out of range
                if (
                    confidence < self._min_confidence
                    or pitch < self._min_frequency
                    or pitch > self.MAX_FREQUENCY
                ):
                    return None
                
                # Apply harmonic correction for low frequencies, but only in specific cases
                corrected_pitch = pitch
                
                # Only apply harmonic correction for specific frequency ranges
                # E1 (41.2 Hz) might be detected as E2 (82.4 Hz)
                # F1 (43.65 Hz) might be detected as F2 (87.3 Hz)
                if self._harmonic_correction:
                    # Check specifically for E2 and F2 frequencies that might be harmonics
                    if 81.0 < pitch < 84.0:  # Potential E2 that might be E1
                        half_pitch = pitch / 2.0
                        if abs(half_pitch - 41.2) < 2.0:  # Close to E1
                            logger.debug(f"Harmonic correction for E: {pitch:.1f}Hz -> {half_pitch:.1f}Hz (E1)")
                            corrected_pitch = half_pitch
                    
                    elif 86.0 < pitch < 89.0:  # Potential F2 that might be F1
                        half_pitch = pitch / 2.0
                        if abs(half_pitch - 43.65) < 2.0:  # Close to F1
                            logger.debug(f"Harmonic correction for F: {pitch:.1f}Hz -> {half_pitch:.1f}Hz (F1)")
                            corrected_pitch = half_pitch
                    
                    elif 92.0 < pitch < 94.0:  # Potential F#2 that might be F#1
                        half_pitch = pitch / 2.0
                        if abs(half_pitch - 46.25) < 2.0:  # Close to F#1
                            logger.debug(f"Harmonic correction for F#: {pitch:.1f}Hz -> {half_pitch:.1f}Hz (F#1)")
                            corrected_pitch = half_pitch
                
                # Convert frequency to note name
                note_name = get_note_name(corrected_pitch, self._use_flats)
                
                # Add to note history for stability checking
                self._note_history.append(note_name)
                
                # Check if we have a stable note
                if len(self._note_history) >= self._min_stable_count:
                    # Count occurrences of each note in history
                    note_counts: Dict[str, int] = {}
                    for note in self._note_history:
                        if note is not None:
                            note_counts[note] = note_counts.get(note, 0) + 1
                    
                    if note_counts:
                        # Find the most common note and its count
                        most_common_note = max(note_counts.items(), key=lambda x: x[1])[0]
                        majority = note_counts[most_common_note] / len(self._note_history)
                        
                        # If we have a clear majority, consider it a stable note
                        if majority >= self._stability_majority:
                            # Only trigger callback if this is a new note
                            if most_common_note != self._current_note:
                                self._current_note = most_common_note
                                self._stable_count = 0
                                
                                # Create DetectedNote object
                                detected_note = DetectedNote(
                                    name=most_common_note,
                                    frequency=corrected_pitch,  # Use the corrected pitch
                                    confidence=confidence,
                                    signal=rms,  # Use RMS as signal strength
                                    is_stable=True,
                                    timestamp=timestamp
                                )
                                
                                # Update stable note
                                self._stable_note = detected_note
                                
                                # Call the callback with the detected note
                                if self._callback:
                                    try:
                                        self._callback(detected_note, timestamp)
                                    except Exception as e:
                                        logger.error(f"Error in note detection callback: {e}", exc_info=True)
                                
                                # Log the detection but don't print to stdout
                                # The NoteDetectionService will handle printing via the callback
                                logger.info(f"[{timestamp:.2f}s] {most_common_note} ({pitch:.1f}Hz, conf: {confidence:.2f}, signal: {rms:.4f})")
                                
                                return detected_note
                            
                            # Reset stability counter if we have a stable note
                            self._stable_count = 0
                        else:
                            # Increment stability counter if we don't have a stable note yet
                            self._stable_count += 1
                    
                    # If we've had too many unstable readings, clear the current note
                    if self._stable_count > self._min_stable_count * 2:
                        if self._current_note is not None:
                            self._current_note = None
                            self._stable_count = 0
                            self._stable_note = None
            else:
                # Signal is too weak, log at debug level
                logger.debug(f"Signal too weak: {signal_max:.4f} (RMS: {rms:.4f}, dB: {db:.1f})")
            
            return None
                    
        except Exception as e:
            logger.error(f"Error in audio processing: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
    
    def get_stable_note(self) -> Optional[DetectedNote]:
        """Get the current stable note if one exists.
        
        Returns:
            The current stable note, or None if no stable note exists
        """
        return self._stable_note
    
    def set_callback(self, callback: Optional[Callable]) -> None:
        """Set the callback function for note detection events.
        
        Args:
            callback: Callback function or None to remove
        """
        self._callback = callback
        
    def set_sample_rate(self, sample_rate: int) -> None:
        """Update the sample rate and reinitialize the pitch detector.
        
        Args:
            sample_rate: New sample rate in Hz
        """
        if sample_rate <= 0:
            raise ValueError("Sample rate must be positive")
            
        # Only update if sample rate has changed
        if sample_rate != self._sample_rate:
            logger.info(f"Updating note detector sample rate from {self._sample_rate} to {sample_rate} Hz")
            self._sample_rate = sample_rate
            
            # Reinitialize the pitch detector with the new sample rate
            self._pitch_detector = aubio.pitch(
                "yin",
                self._hop_size * 2,
                self._hop_size,
                self._sample_rate
            )
            self._pitch_detector.set_unit("Hz")
            self._pitch_detector.set_tolerance(self._tolerance)
    
    def get_current_note(self) -> Optional[DetectedNote]:
        """Get the current detected note.
        
        Returns:
            The current stable note, or None if no note is detected
        """
        return self._stable_note
    
    def get_simple_note(self) -> Optional[str]:
        """Get just the note letter (A, B, C, etc.) without the octave.
        
        Returns:
            The note letter, or None if no note is detected
        """
        if self._stable_note:
            return self._stable_note.name[0]
        return None
    
    def is_note_playing(self, target_note: str) -> bool:
        """Check if a specific note is currently playing.
        
        Args:
            target_note: The target note letter (A, B, C, etc.)
            
        Returns:
            True if the target note is currently playing, False otherwise
        """
        current = self.get_simple_note()
        return current == target_note
    
    # Property getters and setters
    @property
    def tolerance(self) -> float:
        """Get the aubio pitch tolerance."""
        return self._tolerance
    
    @tolerance.setter
    def tolerance(self, value: float) -> None:
        """Set the aubio pitch tolerance."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("Tolerance must be between 0.0 and 1.0")
        self._tolerance = value
        if self._pitch_detector:
            self._pitch_detector.set_tolerance(value)
    
    @property
    def min_stable_count(self) -> int:
        """Get the minimum number of stable detections to consider a note stable."""
        return self._min_stable_count
    
    @min_stable_count.setter
    def min_stable_count(self, value: int) -> None:
        """Set the minimum number of stable detections."""
        if value < 1:
            raise ValueError("min_stable_count must be at least 1")
        self._min_stable_count = value
    
    @property
    def stability_majority(self) -> float:
        """Get the proportion of detections needed for stability."""
        return self._stability_majority
    
    @stability_majority.setter
    def stability_majority(self, value: float) -> None:
        """Set the stability majority threshold."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("stability_majority must be between 0.0 and 1.0")
        self._stability_majority = value
    
    @property
    def group_hz(self) -> float:
        """Get the frequency grouping in Hz."""
        return self._group_hz
    
    @group_hz.setter
    def group_hz(self, value: float) -> None:
        """Set the frequency grouping in Hz."""
        if value <= 0:
            raise ValueError("group_hz must be positive")
        self._group_hz = value
    
    @property
    def snap_percent(self) -> float:
        """Get the snap percentage for note detection."""
        return self._snap_percent
    
    @snap_percent.setter
    def snap_percent(self, value: float) -> None:
        """Set the snap percentage for note detection."""
        if not 0.0 <= value <= 1.0:
            raise ValueError("snap_percent must be between 0.0 and 1.0")
        self._snap_percent = value
    
    @property
    def min_frequency(self) -> float:
        """Get the minimum frequency to consider for note detection."""
        return self._min_frequency
    
    @min_frequency.setter
    def min_frequency(self, value: float) -> None:
        """Set the minimum frequency to consider for note detection."""
        if value <= 0:
            raise ValueError("min_frequency must be positive")
        self._min_frequency = value
    
