"""Main entry point for the Tonal Recall application using the refactored architecture."""

import argparse
import logging
import sys
import time
from typing import Dict, Any, Optional

from .logger import get_logger
from .logging_config import setup_logging
from .ui.adapters import PygameAdapter, CursesAdapter, UIAdapter

logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Tonal Recall - Guitar Note Detection")

    parser.add_argument(
        "--ui",
        choices=["pygame", "curses", "none"],
        default="pygame",
        help="UI to use (default: pygame)",
    )

    parser.add_argument(
        "--device",
        type=int,
        default=None,
        help="Audio input device ID (default: auto-detect)",
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
        action="store_true",
        default=True,
        help="Enable harmonic correction for low notes (default: True)",
    )

    parser.add_argument(
        "--gain",
        type=float,
        default=1.0,
        help="Audio gain to apply to the input signal",
    )

    parser.add_argument(
        "--difficulty",
        type=int,
        default=3,
        help="Game difficulty level (0-6)",
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def create_ui_adapter(
    ui_type: str, config: Dict[str, Any]
) -> Optional[UIAdapter]:
    """Create a UI adapter based on the specified type."""
    if ui_type == "pygame":
        # The note detection service is now created inside the adapter
        return PygameAdapter(note_detection_service=None, config=config)
    elif ui_type == "curses":
        # Curses adapter would need similar refactoring if used
        return CursesAdapter(note_detection_service=None, config=config)
    elif ui_type == "none":
        return None
    else:
        logger.error(f"Unknown UI type: {ui_type}")
        return None


def main():
    """Main entry point for the application."""
    # Parse command line arguments
    args = parse_args()

    # Configure logging
    setup_logging(force_level=logging.DEBUG if args.debug else None)

    # Create configuration from command line arguments
    config = {
        "device_id": args.device,
        "sample_rate": args.sample_rate,
        "min_confidence": args.min_confidence,
        "min_signal": args.min_signal,
        "harmonic_correction": args.harmonic_correction,
        "gain": args.gain,
        "difficulty": args.difficulty,
        "duration": 60,  # Game duration in seconds
        "width": 1024,
        "height": 768,
        "title": "Tonal Recall",
    }

    # Create UI adapter
    ui_adapter = create_ui_adapter(args.ui, config)

    if ui_adapter is None and args.ui != "none":
        logger.error("Failed to create UI adapter")
        return 1

    try:
        if ui_adapter:
            if ui_adapter.initialize():
                # The start method is now blocking and runs the whole game loop.
                ui_adapter.start()
                ui_adapter.cleanup()
            else:
                logger.error("Failed to initialize UI adapter.")
                return 1
        else:
            # Handle the 'none' UI case for headless operation if needed
            logger.info("No UI selected. Exiting.")

        return 0

    except Exception as e:
        logger.exception(f"Error in main: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
