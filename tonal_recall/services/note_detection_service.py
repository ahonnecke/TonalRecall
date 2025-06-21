from typing import Optional, Callable
import numpy as np

from tonal_recall.note_detector import NoteDetector

# This is the service-level, simplified note DTO
from tonal_recall.services.interfaces import (
    IAudioProvider,
    INoteDetectionService,
    DetectedNote,
)

# This is the detector's internal, more detailed note object
from tonal_recall.note_types import DetectedNote as DetectorNote


class NoteDetectionService(INoteDetectionService):
    """A service that detects musical notes from an audio stream."""

    def __init__(self, audio_provider: IAudioProvider, **config) -> None:
        self._audio_provider = audio_provider
        self._on_note_detected: Optional[Callable[[Optional[DetectedNote]], None]] = (
            None
        )

        # The actual note detector is now an internal implementation detail.
        self._detector = NoteDetector(sample_rate=audio_provider.sample_rate, **config)
        # The legacy detector uses an internal callback. We set it to our handler.
        self._detector._callback = self._internal_note_handler

    def _internal_note_handler(self, note: DetectorNote, signal: float) -> None:
        """Handles notes from the legacy detector and adapts them for the service's callback."""
        if self._on_note_detected:
            note_object = None
            if note and note.note_name:
                # Adapt from the detector's detailed object to the service's simple DTO
                note_object = DetectedNote(
                    note.note_name, note.frequency, note.confidence
                )
            self._on_note_detected(note_object)

    def start(self, on_note_detected: Callable[[Optional[DetectedNote]], None]) -> None:
        """Starts the note detection service."""
        self._on_note_detected = on_note_detected
        self._detector._running = True  # Set running state for the legacy detector
        self._audio_provider.start(self._audio_callback)

    def _audio_callback(self, audio_data: bytes) -> None:
        """Receives audio data from the provider and processes it."""
        # Convert bytes back to a numpy array of float32
        audio_chunk = np.frombuffer(audio_data, dtype=np.float32)

        # Reshape the array if the audio is multi-channel
        num_channels = self._audio_provider.channels
        if num_channels > 1 and audio_chunk.size > 0:
            audio_chunk = audio_chunk.reshape(-1, num_channels)

        # Call the detector's internal processing method
        self._detector._process_audio(audio_chunk)

    def stop(self) -> None:
        """Stops the note detection service."""
        self._audio_provider.stop()
        if self._detector:
            self._detector._running = False  # Clean up state

    def configure_detector(self, **kwargs) -> None:
        """Configures the underlying note detector."""
        # This would pass configuration changes to the self._detector instance.
        # For now, we can delegate to the existing property setters if they exist.
        for key, value in kwargs.items():
            if hasattr(self._detector, key):
                setattr(self._detector, key, value)

    def get_detector_config(self) -> dict:
        """Gets the current configuration of the note detector."""
        # In a real implementation, this would list the configurable properties.
        return {}
