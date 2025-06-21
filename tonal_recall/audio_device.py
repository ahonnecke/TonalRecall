"""Audio device utilities for note detection."""

import logging
import sounddevice as sd
from typing import Optional, Dict, Any, Tuple

logger = logging.getLogger(__name__)


def find_rocksmith_adapter() -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
    """Find the Rocksmith audio adapter in the system's audio devices.

    Returns:
        A tuple of (device_id, device_info) if found, (None, None) otherwise
    """
    try:
        devices = sd.query_devices()
        for device_id, device in enumerate(devices):
            if (
                device["max_input_channels"] > 0
                and "rocksmith" in device["name"].lower()
            ):
                return device_id, device
        return None, None
    except Exception as e:
        logger.error(f"Error finding Rocksmith adapter: {e}")
        raise


def clear_unstable_notes(callback, _stable_count, _min_stable_count, _current_note):
    # If we've had too many unstable readings, clear the current note
    if _stable_count > _min_stable_count * 2:
        if _current_note is not None:
            _current_note = None
            _stable_count = 0
            # Optionally notify that the note was cleared
            try:
                callback(
                    DetectedNote(
                        timestamp=time.time(),
                        note_name="",
                        octave=4,
                        frequency=0.0,
                        confidence=0.0,
                        signal=0.0,
                        is_stable=False,
                    ),
                    0.0,
                )
            except Exception as e:
                logger.error(f"Error in note cleared callback: {e}", exc_info=True)
                raise
