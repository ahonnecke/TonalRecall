"""Main entry point for the Tonal Recall CLI."""

import sys
import argparse
from typing import List, Optional

from .note_detector_cli import NoteDetectorCLI
from .note_detector_cli_refactored import run_test as run_refactored_test
from .direct_note_detector import run_direct_test


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.

    Args:
        args: Command line arguments, or None to use sys.argv

    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(description="Tonal Recall - Guitar Note Detection")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Note detection test command (legacy)
    test_parser = subparsers.add_parser(
        "test", help="Run note detection test (legacy implementation)"
    )
    test_parser.add_argument(
        "--duration", type=float, default=15.0, help="Test duration in seconds"
    )
    test_parser.add_argument(
        "--device", type=int, default=None, help="Audio input device ID"
    )
    test_parser.add_argument(
        "--flats", action="store_true", help="Use flat notes instead of sharps"
    )
    test_parser.add_argument(
        "--min-confidence", type=float, default=0.7, help="Minimum confidence threshold"
    )
    test_parser.add_argument(
        "--min-signal", type=float, default=0.005, help="Minimum signal threshold"
    )

    # Note detection test command (refactored)
    test_refactored_parser = subparsers.add_parser(
        "test-refactored", help="Run note detection test (refactored implementation)"
    )
    test_refactored_parser.add_argument(
        "--duration", type=float, default=10.0, help="Test duration in seconds"
    )
    test_refactored_parser.add_argument(
        "--device", type=int, default=None, help="Audio input device ID"
    )
    test_refactored_parser.add_argument(
        "--sample-rate", type=int, default=44100, help="Audio sample rate in Hz"
    )
    test_refactored_parser.add_argument(
        "--min-confidence", type=float, default=0.5, help="Minimum confidence threshold"
    )
    test_refactored_parser.add_argument(
        "--min-signal", type=float, default=0.001, help="Minimum signal threshold"
    )
    test_refactored_parser.add_argument(
        "--harmonic-correction",
        type=bool,
        default=True,
        help="Enable harmonic correction for low notes",
    )
    test_refactored_parser.add_argument(
        "--debug", action="store_true", help="Enable debug logging"
    )

    # Direct note detection test command (using original working approach)
    test_direct_parser = subparsers.add_parser(
        "test-direct", help="Run note detection test (direct implementation)"
    )
    test_direct_parser.add_argument(
        "--duration", type=float, default=10.0, help="Test duration in seconds"
    )
    test_direct_parser.add_argument(
        "--device", type=int, default=None, help="Audio input device ID"
    )
    test_direct_parser.add_argument(
        "--min-confidence", type=float, default=0.5, help="Minimum confidence threshold"
    )
    test_direct_parser.add_argument(
        "--min-signal", type=float, default=0.001, help="Minimum signal threshold"
    )

    # Parse arguments
    parsed_args = parser.parse_args(args)

    # Handle commands
    if parsed_args.command == "test":
        # Legacy implementation
        cli = NoteDetectorCLI(
            test_duration=parsed_args.duration,
            device_id=parsed_args.device,
            use_flats=parsed_args.flats,
            min_confidence=parsed_args.min_confidence,
            min_signal=parsed_args.min_signal,
        )
        cli.run_test()
    elif parsed_args.command == "test-refactored":
        # Refactored implementation
        run_refactored_test(parsed_args)
    elif parsed_args.command == "test-direct":
        # Direct implementation using original working approach
        run_direct_test(
            device_id=parsed_args.device,
            duration=parsed_args.duration,
            min_confidence=parsed_args.min_confidence,
            min_signal=parsed_args.min_signal,
        )
    else:
        parser.print_help()
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
