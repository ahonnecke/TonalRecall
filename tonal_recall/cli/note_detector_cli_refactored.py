"""CLI for testing note detection with the refactored architecture."""

import argparse
import logging
import time
from typing import List

from ..logger import get_logger
from ..logging_config import setup_logging
from ..note_types import DetectedNote

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments.
    
    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Test note detection")
    
    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device ID (default: auto-detect)",
    )
    
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Test duration in seconds (default: 10.0)",
    )
    
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=44100,
        help="Audio sample rate in Hz (default: 44100)",
    )
    
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence for note detection (default: 0.5)",
    )
    
    parser.add_argument(
        "--min-signal",
        type=float,
        default=0.001,
        help="Minimum signal level for note detection (default: 0.001)",
    )
    
    parser.add_argument(
        "--harmonic-correction",
        type=bool,
        default=True,
        help="Enable harmonic correction for low notes (default: True)",
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    
    return parser.parse_args()


def run_test(args):
    """Run the note detection test.
    
    Args:
        args: Command line arguments
    """
    # Configure logging
    setup_logging(force_level=logging.DEBUG if args.debug else None)
    
    # Create component factory and configuration manager

    
    # List of detected notes
    detected_notes: List[DetectedNote] = []
    
    # Callback for note detection
    def note_callback(note: DetectedNote, timestamp: float):
        detected_notes.append(note)
        logger.info(
            f"[{timestamp:.2f}s] {note.name} ({note.frequency:.1f}Hz, "
            f"conf: {note.confidence:.2f}, signal: {note.signal:.4f})"
        )
    
    # Create audio input and note detector directly, following the working baseline approach
    from tonal_recall.audio.note_detector import NoteDetector
    from tonal_recall.audio.audio_input import SoundDeviceInput
    from tonal_recall.audio.note_detection_service import NoteDetectionService
    
    # Create components directly instead of using factory
    try:
        # Create audio input
        audio_input = SoundDeviceInput(
            device_id=args.device,
            sample_rate=args.sample_rate
        )
        
        # Create note detector
        note_detector = NoteDetector(
            min_confidence=args.min_confidence,
            min_signal=args.min_signal,
            harmonic_correction=args.harmonic_correction
        )
        
        # Create note detection service
        note_detection_service = NoteDetectionService(
            audio_input=audio_input,
            note_detector=note_detector
        )
        
        logger.info(f"Created note detection components with device ID: {args.device}")
    except Exception as e:
        logger.error(f"Failed to create note detection components: {e}")
        return
    
    # Start note detection
    logger.info(f"Starting note detection test for {args.duration} seconds...")
    if not note_detection_service.start(note_callback):
        logger.error("Failed to start note detection service")
        return
    
    try:
        # Wait for the specified duration
        time.sleep(args.duration)
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
    finally:
        # Stop note detection
        note_detection_service.stop()
    
    # Print test results
    logger.info(f"Test completed. Detected {len(detected_notes)} notes.")
    
    # Print note statistics
    if detected_notes:
        note_counts = {}
        for note in detected_notes:
            if note.name not in note_counts:
                note_counts[note.name] = 0
            note_counts[note.name] += 1
        
        logger.info("Note statistics:")
        for note_name, count in sorted(note_counts.items(), key=lambda x: x[1], reverse=True):
            logger.info(f"  {note_name}: {count} detections")


def main():
    """Main entry point for the CLI."""
    args = parse_args()
    run_test(args)


if __name__ == "__main__":
    main()
