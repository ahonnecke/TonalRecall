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
            if device["max_input_channels"] > 0 and "rocksmith" in device["name"].lower():
                return device_id, device
        return None, None
    except Exception as e:
        logger.error(f"Error finding Rocksmith adapter: {e}")
        return None, None
