"""
Direct note detector CLI that follows the exact same pattern as the working baseline test.
"""

import time
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

from ..logger import get_logger
from ..note_detector import NoteDetector, DetectedNote

logger = get_logger(__name__)


def run_direct_test(device_id: Optional[int] = None, 
                   duration: float = 10.0,
                   min_confidence: float = 0.5,
                   min_signal: float = 0.001):
    """Run a direct note detection test using the original working approach.
    
    Args:
        device_id: Audio input device ID. If None, the default device will be used.
        duration: Test duration in seconds.
        min_confidence: Minimum confidence threshold for note detection.
        min_signal: Minimum signal threshold for note detection.
        harmonic_correction: Whether to enable harmonic correction.
    """
    print("\n=== Direct Note Detection Test ===\n")
    print(f"Device ID: {device_id}")
    print(f"Test will run for {duration} seconds.\n")
    
    # Initialize note detector directly (same as baseline test)
    detector = NoteDetector(
        device_id=device_id,
        min_confidence=min_confidence,
        min_signal=min_signal,
        silence_threshold_db=-45,  # Higher threshold to reduce false triggers
        min_stable_count=5,  # Require 5 stable readings
        stability_majority=0.8,  # 80% must agree for stable note
        min_frequency=30.0,  # Minimum frequency for bass
        sample_rate=44100,  # Standard sample rate
        frames_per_buffer=1024,  # Standard buffer size
        channels=1  # Mono audio
    )
    
    detected_notes = []
    test_start_time = time.time()
    
    # Callback for detected notes
    def note_callback(note: DetectedNote, signal_strength: float) -> None:
        """Callback for detected notes."""
        if note is None:
            return
            
        timestamp = time.time() - test_start_time
        note_data = {
            "timestamp": round(timestamp, 3),
            "name": note.name,
            "frequency": round(note.frequency, 2),
            "confidence": round(note.confidence, 3),
            "signal_strength": round(signal_strength, 4),
            "is_stable": note.is_stable
        }
        detected_notes.append(note_data)
        
        # Print a summary of the detection
        print(f"[{timestamp:.2f}s] {note.name} "
              f"({note.frequency:.1f}Hz, "
              f"conf: {note.confidence:.2f}, "
              f"signal: {signal_strength:.4f})")
    
    try:
        print("Starting note detector...")
        detector.start(note_callback)
        
        print("\n=== TEST STARTED ===")
        print(f"Listening for {duration} seconds...\n")
        
        # Wait for the specified duration
        time.sleep(duration)
        
    except KeyboardInterrupt:
        print("\nTest stopped by user")
    finally:
        test_duration = time.time() - test_start_time
        detector.stop()
        
        # Print test results
        print("\n=== TEST COMPLETED ===")
        print(f"Duration: {test_duration:.1f} seconds")
        print(f"Total notes detected: {len(detected_notes)}")
        
        # Print note statistics
        if detected_notes:
            note_counts = {}
            for note in detected_notes:
                name = note['name']
                if name not in note_counts:
                    note_counts[name] = 0
                note_counts[name] += 1
            
            print("Note statistics:")
            for note_name, count in sorted(note_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {note_name}: {count} detections")


def main():
    """Main entry point for the direct note detector CLI."""
    parser = argparse.ArgumentParser(description='Run direct note detection test.')
    parser.add_argument('--device', type=int, help='Audio device ID')
    parser.add_argument('--duration', type=float, default=10.0, 
                       help='Test duration in seconds (default: 10)')
    parser.add_argument('--min-confidence', type=float, default=0.5,
                       help='Minimum confidence threshold (default: 0.5)')
    parser.add_argument('--min-signal', type=float, default=0.001,
                       help='Minimum signal threshold (default: 0.001)')

    args = parser.parse_args()
    
    # Run the test
    run_direct_test(
        device_id=args.device,
        duration=args.duration,
        min_confidence=args.min_confidence,
        min_signal=args.min_signal
    )


if __name__ == "__main__":
    main()
