from abc import ABC, abstractmethod
from typing import Optional, Callable

# Assuming the existence of a simple data structure for a detected note.
# This will be defined more concretely later.
class DetectedNote:
    def __init__(self, note_name: str, frequency: float, confidence: float):
        self.note_name = note_name
        self.frequency = frequency
        self.confidence = confidence

    def __repr__(self):
        return f"DetectedNote(note='{self.note_name}', freq={self.frequency:.2f}, conf={self.confidence:.2f})"

class IAudioProvider(ABC):
    """An abstract interface for audio providers."""
    @abstractmethod
    def start(self, on_data_callback: Callable[[bytes], None]) -> None:
        """Starts the audio stream, calling the callback with chunks of audio data."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops the audio stream."""
        pass

    @property
    @abstractmethod
    def sample_rate(self) -> int:
        """The sample rate of the audio stream."""
        pass

    @property
    @abstractmethod
    def channels(self) -> int:
        """The number of channels in the audio stream."""
        pass

class INoteDetectionService(ABC):
    """An abstract interface for a note detection service."""
    @abstractmethod
    def start(self, on_note_detected: Callable[[Optional[DetectedNote]], None]) -> None:
        """Starts the note detection service."""
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stops the note detection service."""
        pass

    @abstractmethod
    def configure_detector(self, **kwargs) -> None:
        """Configures the underlying note detector."""
        pass

    @abstractmethod
    def get_detector_config(self) -> dict:
        """Gets the current configuration of the note detector."""
        pass
