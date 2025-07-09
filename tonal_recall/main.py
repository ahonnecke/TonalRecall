#!/usr/bin/env python3

import argparse

from tonal_recall.logging_config import setup_logging
from tonal_recall.note_game_core import NoteGame
from tonal_recall.ui import PygameUI
from tonal_recall.note_detector import NoteDetector
from tonal_recall.logger import get_logger


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tonal Recall - A musical note training game"
    )

    # UI settings
    parser.add_argument(
        "--ui",
        type=str,
        default="pygame",
        choices=["pygame"],
        help="UI to use (default: pygame). Curses is not supported.",
    )

    # Game settings
    difficulty_levels = sorted(NoteGame.note_sets.keys())
    parser.add_argument(
        "--difficulty",
        type=int,
        default=3,
        choices=difficulty_levels,
        help=f"Game difficulty level from {min(difficulty_levels)} to {max(difficulty_levels)}. (default: 3)",
    )

    # Audio settings
    parser.add_argument("--device", type=int, help="Audio input device ID.")
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=48000,
        help="Audio sample rate (default: 48000).",
    )
    parser.add_argument(
        "--gain", type=float, default=1.0, help="Audio gain factor."
    )
    parser.add_argument(
        "--silence",
        type=float,
        default=-60,
        help="Silence threshold in dB. Notes quieter than this will be ignored.",
    )

    # Debugging
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging."
    )
    parser.add_argument(
        "--note",
        type=str,
        help="Target a specific note for testing (e.g., 'C4'). Overrides difficulty to 0 and enables debug mode.",
    )

    return parser.parse_args()


def main():
    """Main entry point for the Tonal Recall game."""
    args = parse_arguments()

    # If a specific note is targeted for testing, override settings
    if args.note:
        args.difficulty = 0  # Use an empty note set
        args.debug = True

    # Configure logging
    setup_logging(level="DEBUG" if args.debug else "INFO")
    logger = get_logger(__name__)

    try:
        ui = PygameUI()

        def game_factory():
            """Factory to create a new game instance with a configured detector."""
            logger.info("Creating new game instance via factory.")

            detector_config = {
                "device_id": args.device,
                "sample_rate": args.sample_rate,
                "gain": args.gain,
                "silence_threshold_db": args.silence,
            }
            detector_config = {
                k: v for k, v in detector_config.items() if v is not None
            }

            detector = NoteDetector(**detector_config)

            game = NoteGame(
                note_detector=detector,
                difficulty=args.difficulty,
                target_note=args.note,  # Pass the specific note to the game
            )
            return game

        # The PygameUI.run() method now controls the entire application lifecycle.
        ui.run(game_factory, halt_on_match=bool(args.note))

    except Exception:
        logger.exception("An unhandled error occurred in the main application.")
        raise
    finally:
        logger.info("Tonal Recall is shutting down.")


if __name__ == "__main__":
    main()
