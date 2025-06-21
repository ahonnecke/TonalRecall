"""
A simple, non-interactive script to test live note detection.

This script uses LiveAudioProvider and NoteDetector to capture and identify
musical notes from an audio input device in real-time. It's designed for
quick, clean verification of the note detection system.
"""
import time
import sys
from pathlib import Path
import numpy as np
import sounddevice as sd
import argparse
from typing import List, Optional

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tonal_recall.note_detector import NoteDetector, DetectedNote
from tonal_recall.services.audio_providers import LiveAudioProvider
from tonal_recall.util.logs import get_logger

logger = get_logger(__name__)

class BaselineTester:
    """A simple class to test live note detection."""

    def __init__(self, device_id: int):
        """
        Initialize the tester.

        Args:
            device_id: The audio input device ID.
        """
        self.device_id = device_id
        self.test_start_time: float = 0.0
        self.test_duration: float = 0.0
        self.detected_notes: List[DetectedNote] = []
        self.last_note_name: Optional[str] = None
        self.gain = 15000.0  # Digital gain for the Rocksmith adapter

        # Initialize the audio provider
        self.audio_provider = LiveAudioProvider(
            device_id=self.device_id,
            sample_rate=48000,
            channels=1,
            callback=self._audio_data_handler
        )

        # Initialize the note detector
        self.detector = NoteDetector(
            sample_rate=self.audio_provider.sample_rate,
            callback=self.note_callback,
            min_confidence=0.80,
            min_stable_count=4,
            harmonic_correction=True
        )

    def _audio_data_handler(self, indata: np.ndarray, frames: int, time_info, status) -> None:
        """Handles raw audio data from the LiveAudioProvider."""
        try:
            audio_data = np.frombuffer(indata, dtype=np.float32)
            audio_data *= self.gain
            timestamp = time.time()
            self.detector.process_audio(audio_data, timestamp)
        except Exception as e:
            logger.error(f"Error in audio data handler: {e}", exc_info=True)

    def note_callback(self, detected_note: DetectedNote) -> None:
        """Callback for detected notes, printing only new stable notes."""
        self.detected_notes.append(detected_note)
        
        if detected_note.is_stable and detected_note.note_name != self.last_note_name:
            self.last_note_name = detected_note.note_name
            print(f"[STABLE] {detected_note.note_name}")

    def run_test(self, duration: float = 10.0) -> None:
        """Run the baseline test."""
        print("\n=== Guitar Note Detection Test ===")
        print(f"Listening for {duration:.1f} seconds on device {self.device_id}...\n")
        
        self.detected_notes = []
        self.test_start_time = time.time()
        
        self.audio_provider.start()
        
        try:
            time.sleep(duration)
        except KeyboardInterrupt:
            print("\nTest stopped by user.")
        finally:
            self.test_duration = time.time() - self.test_start_time
            self.audio_provider.stop()
            self._print_summary()

    def _print_summary(self) -> None:
        """Prints a summary of the test results."""
        print("\n=== Test Complete ===")
        print(f"Ran for {self.test_duration:.1f} seconds.")
        
        stable_notes = sorted(list(set(
            note.note_name for note in self.detected_notes if note.is_stable
        )))
        
        if stable_notes:
            print(f"Stable notes detected: {', '.join(stable_notes)}")
        else:
            print("No stable notes were detected.")


def list_audio_devices() -> None:
    """List available audio input devices."""
    print("\nAvailable audio input devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{i}: {device['name']} (Sample Rate: {device['default_samplerate'] / 1000:.1f}kHz)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run a simple live note detection test.')
    parser.add_argument('--device', type=int, help='Audio device ID. Required to run the test.')
    parser.add_argument('--duration', type=float, default=10.0, help='Test duration in seconds.')
    parser.add_argument('--list-devices', action='store_true', help='List available audio devices and exit.')
    
    args = parser.parse_args()

    if args.list_devices:
        list_audio_devices()
        sys.exit(0)

    if args.device is None:
        list_audio_devices()
        print("\nPlease specify the device ID with the --device argument.")
        sys.exit(1)
    
    tester = BaselineTester(device_id=args.device)
    tester.run_test(duration=args.duration)
