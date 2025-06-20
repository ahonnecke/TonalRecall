"""Note detection service that integrates audio input and note detection."""

from __future__ import annotations
import time
from typing import Optional, Callable

from ..logger import get_logger
from ..note_types import DetectedNote
from .audio_input import SoundDeviceInput
from .note_detector import NoteDetector
from ..core.interfaces import INoteDetectionService, INoteDetector, IAudioInput

logger = get_logger(__name__)


class NoteDetectionService(INoteDetectionService):
    """Service that integrates audio input and note detection.
    
    This class acts as a facade for the audio input and note detection components,
    providing a simple interface for clients to use.
    """
    
    def __init__(
        self,
        audio_input: Optional[IAudioInput] = None,
        note_detector: Optional[INoteDetector] = None,
        device_id: Optional[int] = None,
        sample_rate: int = 44100,
        frames_per_buffer: int = 1024,
        channels: int = 1,
        **detector_params
    ) -> None:
        """Initialize the note detection service.
        
        Args:
            audio_input: Audio input handler, or None to create a default one
            note_detector: Note detector, or None to create a default one
            device_id: Audio input device ID, or None to auto-detect
            sample_rate: Sample rate in Hz
            frames_per_buffer: Buffer size in frames
            channels: Number of audio channels
            **detector_params: Additional parameters for the note detector
        """
        # Create default components if not provided
        self._audio_input = audio_input or SoundDeviceInput(
            device_id=device_id,
            sample_rate=sample_rate,
            frames_per_buffer=frames_per_buffer,
            channels=channels,
        )
        
        self._note_detector = note_detector or NoteDetector(
            sample_rate=sample_rate,
            win_size=frames_per_buffer * 2,  # win_size should be twice frames_per_buffer (matching baseline)
            hop_size=frames_per_buffer,  # hop_size should be frames_per_buffer (matching baseline)
            **detector_params
        )
        
        self._callback = None
        self._running = False
        self._start_time = 0.0
    
    def start(self, callback: Callable[[DetectedNote, float], None]) -> None:
        """Start note detection.
        
        Args:
            callback: Function to call when a note is detected
        """
        if self._running:
            logger.warning("Note detection already running")
            return
        
        self._callback = callback
        self._note_detector.set_callback(callback)
        
        # Start audio input with our processing callback
        self._audio_input.start(self._process_audio)
        
        # Update the note detector with the actual sample rate that was used
        # (in case the audio input had to use a different sample rate)
        actual_sample_rate = self._audio_input._sample_rate
        if hasattr(self._note_detector, 'set_sample_rate'):
            logger.info(f"Updating note detector sample rate to {actual_sample_rate} Hz")
            self._note_detector.set_sample_rate(actual_sample_rate)
        
        self._running = True
        self._start_time = time.time()
        logger.info("Note detection started")
    
    def stop(self) -> None:
        """Stop note detection."""
        if not self._running:
            return
        
        self._audio_input.stop()
        self._running = False
        logger.info("Note detection stopped")
    
    def _process_audio(self, audio_data, timestamp):
        """Process audio data and detect notes.
        
        Args:
            audio_data: Audio data as numpy array
            timestamp: Current timestamp
        """
        # Process the audio data to detect notes
        detected_note = self._note_detector.process_audio(audio_data, timestamp)
        
        # If a stable note was detected and we have a callback, call it
        if detected_note and self._callback:
            # Calculate elapsed time since start
            elapsed = timestamp - self._start_time
            
            # Call the callback with the detected note and elapsed time
            self._callback(detected_note, elapsed)
            
            # Log the detected note
            logger.debug(
                f"[{elapsed:.2f}s] {detected_note.name} ({detected_note.frequency:.1f}Hz, "
                f"conf: {detected_note.confidence:.2f}, signal: {detected_note.signal:.4f})"
            )
    
    def get_current_note(self) -> Optional[DetectedNote]:
        """Get the current detected note.
        
        Returns:
            The current detected note or None if no note is detected
        """
        return self._note_detector.get_current_note()
        
    def get_simple_note(self) -> Optional[str]:
        """Get just the note letter (A, B, C, etc.) without the octave.
        
        Returns:
            The note letter, or None if no note is detected
        """
        return self._note_detector.get_simple_note()
    
    def is_note_playing(self, target_note: str) -> bool:
        """Check if a specific note is currently playing.
        
        Args:
            target_note: The target note letter (A, B, C, etc.)
            
        Returns:
            True if the target note is currently playing, False otherwise
        """
        return self._note_detector.is_note_playing(target_note)
    
    def is_running(self) -> bool:
        """Check if note detection is running.
        
        Returns:
            True if note detection is running, False otherwise
        """
        return self._running
    
    
