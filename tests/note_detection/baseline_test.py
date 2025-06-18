"""
Baseline test for note detection with guitar input.

This script captures note detection data from a guitar input to create a baseline
for regression testing during the migration to the new module structure.
"""
import time
import json
from pathlib import Path
import sounddevice as sd
import numpy as np
from datetime import datetime
from typing import List, Dict, Any, Optional

# Add the project root to the Python path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from tonal_recall.note_detector import NoteDetector, DetectedNote

class BaselineTester:
    def __init__(self, device_id: Optional[int] = None, output_dir: str = "baseline_data"):
        """Initialize the baseline tester.
        
        Args:
            device_id: Audio input device ID. If None, the default device will be used.
            output_dir: Directory to save baseline data.
        """
        self.device_id = device_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)  # Create parent directories if needed
        
        # Initialize note detector with optimized settings for bass
        self.detector = NoteDetector(
            device_id=device_id,
            silence_threshold_db=-45,       # Higher threshold to reduce false triggers
            min_confidence=0.75,            # Slightly higher confidence requirement
            min_signal=0.015,               # Higher threshold to reduce noise
            min_stable_count=5,             # Require 5 stable readings
            stability_majority=0.8,         # 80% must agree for stable note
            min_frequency=30.0,             # Minimum frequency for bass
            sample_rate=44100,              # Standard sample rate
            frames_per_buffer=1024,         # Standard buffer size
            channels=1                      # Mono audio
        )
        
        self.detected_notes: List[Dict[str, Any]] = []
        self.test_start_time: float = 0
        self.test_duration: float = 0
        self.string_notes = ["E1", "A1", "D2", "G2"]  # Standard 4-string bass tuning
    
    def note_callback(self, note: DetectedNote, signal_strength: float) -> None:
        """Callback for detected notes."""
        if note is None:
            return
            
        timestamp = time.time() - self.test_start_time
        note_data = {
            "timestamp": round(timestamp, 3),
            "name": note.name,
            "frequency": round(note.frequency, 2),
            "confidence": round(note.confidence, 3),
            "signal_strength": round(signal_strength, 4),
            "is_stable": note.is_stable
        }
        self.detected_notes.append(note_data)
        
        # Print a summary of the detection
        print(f"[{timestamp:.2f}s] {note.name} "
              f"({note.frequency:.1f}Hz, "
              f"conf: {note.confidence:.2f}, "
              f"signal: {signal_strength:.4f})")
    
    def run_test(self, duration: float = 30.0) -> None:
        """Run the baseline test.
        
        Args:
            duration: Test duration in seconds.
        """
        print("\n=== Guitar Note Detection Baseline Test ===\n")
        print("Play each open string (E2, A2, D3, G3, B3, E4) during the test.")
        print(f"Test will run for {duration} seconds.\n")
        
        self.detected_notes = []
        self.test_start_time = time.time()
        
        try:
            print("Starting note detector...")
            self.detector.start(self.note_callback)
            
            print("\n=== TEST STARTED ===")
            print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Listening for {duration} seconds...")
            print("Play each open string at least once during this time.\n")
            
            # Wait for the specified duration
            time.sleep(duration)
            
        except KeyboardInterrupt:
            print("\nTest stopped by user")
        finally:
            self.test_duration = time.time() - self.test_start_time
            self.detector.stop()
            self._save_results()
    
    def _save_results(self) -> None:
        """Save the test results to a JSON file."""
        if not self.detected_notes:
            print("No notes were detected during the test.")
            return
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy_types(obj):
            if isinstance(obj, (np.floating, np.integer)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(x) for x in obj]
            return obj
        
        # Create a summary of detected notes
        note_counts = {}
        for note in self.detected_notes:
            name = note['name']
            if name not in note_counts:
                note_counts[name] = 0
            note_counts[name] += 1
        
        # Prepare the results dictionary with converted numpy types
        results = {
            "test_info": {
                "start_time": datetime.fromtimestamp(self.test_start_time).isoformat(),
                "duration_seconds": round(self.test_duration, 2),
                "device_id": self.device_id,
                "total_notes_detected": len(self.detected_notes),
                "unique_notes_detected": len(note_counts),
            },
            "note_summary": note_counts,
            "detections": [convert_numpy_types(note) for note in self.detected_notes]
        }
        
        # Generate a filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"baseline_{timestamp}.json"
        
        # Save to JSON
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\n=== TEST COMPLETED ===")
        print(f"Duration: {self.test_duration:.1f} seconds")
        print(f"Total notes detected: {len(self.detected_notes)}")
        print(f"Unique notes: {', '.join(note_counts.keys())}")
        print(f"\nResults saved to: {output_file}")

def list_audio_devices() -> None:
    """List available audio input devices."""
    print("\nAvailable audio input devices:")
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:  # Only show input devices
            print(f"{i}: {device['name']} (Inputs: {device['max_input_channels']}, "
                  f"Sample Rate: {device['default_samplerate']/1000:.1f}kHz)")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run baseline note detection test.')
    parser.add_argument('--device', type=int, help='Audio device ID')
    parser.add_argument('--duration', type=float, default=30.0, 
                       help='Test duration in seconds (default: 30)')
    args = parser.parse_args()
    
    # List devices if no device ID is provided
    if args.device is None:
        list_audio_devices()
        device_id = input("\nEnter device ID to use (or press Enter for default): ")
        args.device = int(device_id) if device_id.strip() else None
    
    # Run the test
    tester = BaselineTester(device_id=args.device)
    tester.run_test(duration=args.duration)
