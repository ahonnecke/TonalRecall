"""Main entry point for the Tonal Recall CLI."""

import sys
import argparse
from typing import List, Optional

from .note_detector_cli import NoteDetectorCLI


def main(args: Optional[List[str]] = None) -> int:
    """Main entry point for the CLI.
    
    Args:
        args: Command line arguments, or None to use sys.argv
        
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = argparse.ArgumentParser(description="Tonal Recall - Guitar Note Detection")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Note detection test command
    test_parser = subparsers.add_parser("test", help="Run note detection test")
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
    
    # Parse arguments
    parsed_args = parser.parse_args(args)
    
    # Handle commands
    if parsed_args.command == "test":
        cli = NoteDetectorCLI(
            test_duration=parsed_args.duration,
            device_id=parsed_args.device,
            use_flats=parsed_args.flats,
            min_confidence=parsed_args.min_confidence,
            min_signal=parsed_args.min_signal,
        )
        cli.run_test()
    else:
        parser.print_help()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
