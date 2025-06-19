"""Command-line interface for note detection."""

import time
import argparse
import sys
from typing import Optional

from ..logger import get_logger
from ..note_types import DetectedNote
from ..audio.note_detection_service import NoteDetectionService

logger = get_logger(__name__)


class NoteDetectorCLI:
    """Command-line interface for note detection."""
    
    def __init__(
        self,
        test_duration: float = 15.0,
        device_id: Optional[int] = None,
        use_flats: bool = False,
        min_confidence: float = 0.7,
        min_signal: float = 0.005,
    ) -> None:
        """Initialize the CLI.
        
        Args:
            test_duration: Duration of the test in seconds
            device_id: Audio input device ID, or None to auto-detect
            use_flats: If True, use flat notes (e.g., 'Bb') instead of sharps (e.g., 'A#')
            min_confidence: Minimum confidence to consider a note detection valid
            min_signal: Minimum signal level to process (avoids noise)
        """
        self.test_duration = test_duration
        self.device_id = device_id
        self.use_flats = use_flats
        self.min_confidence = min_confidence
        self.min_signal = min_signal
        
        # Create the note detection service
        self.service = NoteDetectionService(
            device_id=device_id,
            use_flats=use_flats,
            min_confidence=min_confidence,
            min_signal=min_signal,
        )
    
    def run_test(self) -> None:
        """Run a note detection test for the specified duration."""
        print("\n=== Guitar Note Detection Baseline Test ===\n")
        print("Play each open string (E2, A2, D3, G3, B3, E4) during the test.")
        print(f"Test will run for {self.test_duration} seconds.\n")
        
        print("Starting note detector...")
        
        # Set up logging to display detected notes
        self.detected_notes = []
        
        # Start the note detection service
        try:
            self.service.start(self._note_callback)
            
            # Print test information
            start_time = time.strftime("%Y-%m-%d %H:%M:%S")
            print("\n=== TEST STARTED ===")
            print(f"Start time: {start_time}")
            print(f"Listening for {self.test_duration} seconds...")
            print("Play each open string at least once during this time.\n")
            
            try:
                # Wait for the test duration
                time.sleep(self.test_duration)
            except KeyboardInterrupt:
                print("\nTest interrupted by user.")
        except Exception as e:
            print(f"\nError starting note detector: {e}")
            import traceback
            print(traceback.format_exc())
        finally:
            # Stop the note detection service
            if self.service.is_running():
                self.service.stop()
            print("\n=== TEST COMPLETED ===")
            
            # Print summary
            if self.detected_notes:
                print(f"\nDetected {len(self.detected_notes)} notes.")
                # Group by note name
                note_counts = {}
                for note in self.detected_notes:
                    if note.name not in note_counts:
                        note_counts[note.name] = 0
                    note_counts[note.name] += 1
                
                print("\nNote distribution:")
                for note, count in sorted(note_counts.items()):
                    print(f"  {note}: {count} times")
            else:
                print("\nNo notes were detected. Try adjusting the sensitivity or check your audio input.")
                print("You may need to increase the volume or reduce the minimum confidence threshold.")
                print("Current settings:")
                print(f"  Minimum confidence: {self.min_confidence}")
                print(f"  Minimum signal: {self.min_signal}")
                print(f"  Use flats: {self.use_flats}")
                print("\nTry running with: --min-confidence 0.5 --min-signal 0.001")
    
    def _note_callback(self, note: DetectedNote, elapsed: float) -> None:
        """Callback for detected notes.
        
        Args:
            note: The detected note
            elapsed: Elapsed time since the test started
        """
        # Store the note for later analysis
        self.detected_notes.append(note)
        
        # Format and print the detected note
        print(f"[{elapsed:.2f}s] {note.name} ({note.frequency:.1f}Hz, conf: {note.confidence:.2f}, signal: {note.signal:.4f})")
        
        # Flush stdout to ensure immediate display
        sys.stdout.flush()


def main() -> None:
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="Guitar Note Detection Test")
    parser.add_argument(
        "--duration", type=float, default=15.0, help="Test duration in seconds"
    )
    parser.add_argument(
        "--device", type=int, default=None, help="Audio input device ID"
    )
    parser.add_argument(
        "--flats", action="store_true", help="Use flat notes instead of sharps"
    )
    parser.add_argument(
        "--min-confidence", type=float, default=0.7, help="Minimum confidence threshold"
    )
    parser.add_argument(
        "--min-signal", type=float, default=0.005, help="Minimum signal threshold"
    )
    
    args = parser.parse_args()
    
    # Create and run the CLI
    cli = NoteDetectorCLI(
        test_duration=args.duration,
        device_id=args.device,
        use_flats=args.flats,
        min_confidence=args.min_confidence,
        min_signal=args.min_signal,
    )
    cli.run_test()


if __name__ == "__main__":
    main()
