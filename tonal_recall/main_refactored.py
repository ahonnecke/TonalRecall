"""Main entry point for the Tonal Recall application using the refactored architecture."""

import argparse
import logging
import sys
import time
from typing import Dict, Any, Optional

from .logger import get_logger
from .logging_config import setup_logging
from .core.config import ConfigManager
from .core.factory import ComponentFactory
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

    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    return parser.parse_args()


def create_ui_adapter(
    ui_type: str, factory: ComponentFactory, config: Dict[str, Any]
) -> Optional[UIAdapter]:
    """Create a UI adapter based on the specified type.

    Args:
        ui_type: Type of UI to create (pygame, curses, none)
        factory: Component factory
        config: Configuration options

    Returns:
        UI adapter or None if no UI is requested
    """
    # Create note detection service
    note_detection_service = factory.create_note_detection_service(
        implementation="default",
        device_id=config.get("device_id"),
        sample_rate=config.get("sample_rate", 44100),
        frames_per_buffer=config.get("frames_per_buffer", 1024),
        min_confidence=config.get("min_confidence", 0.5),
        min_signal=config.get("min_signal", 0.001),
        harmonic_correction=config.get("harmonic_correction", True),
    )

    # Create UI adapter based on type
    if ui_type == "pygame":
        return PygameAdapter(note_detection_service, config)
    elif ui_type == "curses":
        return CursesAdapter(note_detection_service, config)
    elif ui_type == "none":
        # No UI, just return None
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

    # Create configuration manager and component factory
    config_manager = ConfigManager()
    factory = ComponentFactory(config_manager)

    # Create configuration from command line arguments
    config = {
        "device_id": args.device,
        "sample_rate": args.sample_rate,
        "min_confidence": args.min_confidence,
        "min_signal": args.min_signal,
        "harmonic_correction": args.harmonic_correction,
        "width": 800,
        "height": 600,
        "title": "Tonal Recall",
    }

    # Create UI adapter
    ui_adapter = create_ui_adapter(args.ui, factory, config)

    if ui_adapter is None and args.ui != "none":
        logger.error("Failed to create UI adapter")
        return 1

    try:
        # If we have a UI adapter, start it and run the main loop
        if ui_adapter:
            if not ui_adapter.start():
                logger.error("Failed to start UI")
                return 1

            # Main loop
            running = True
            last_time = time.time()

            while running:
                # Calculate delta time
                current_time = time.time()
                delta_time = current_time - last_time
                last_time = current_time

                # Update UI
                running = ui_adapter.update(delta_time)

                # Render UI
                ui_adapter.render()

                # Small sleep to avoid hogging CPU
                time.sleep(0.01)

            # Clean up
            ui_adapter.stop()
        else:
            # No UI, just start note detection service directly
            note_detection_service = factory.create_note_detection_service(
                implementation="default",
                device_id=args.device,
                sample_rate=args.sample_rate,
                min_confidence=args.min_confidence,
                min_signal=args.min_signal,
                harmonic_correction=args.harmonic_correction,
            )

            # Define a simple callback for note detection
            def note_callback(note, timestamp):
                logger.info(
                    f"[{timestamp:.2f}s] {note.name} ({note.frequency:.1f}Hz, "
                    f"conf: {note.confidence:.2f}, signal: {note.signal:.4f})"
                )

            # Start note detection
            note_detection_service.start(note_callback)

            logger.info("Press Ctrl+C to exit")

            # Run until interrupted
            try:
                while True:
                    time.sleep(0.1)
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
            finally:
                note_detection_service.stop()

        return 0

    except Exception as e:
        logger.exception(f"Error in main: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
