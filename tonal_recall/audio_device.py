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

def init_audio_device(device_id: Optional[int] = None, sample_rate: int = 44100) -> bool:
    """Initialize the audio device.
    
    Args:
        device_id: Optional device ID to use. If None, will try to find Rocksmith adapter.
        sample_rate: Sample rate in Hz
        
    Returns:
        bool: True if initialization was successful, False otherwise
    """
    try:
        # If no device_id provided, try to find Rocksmith adapter
        if device_id is None:
            device_id, _ = find_rocksmith_adapter()
            if device_id is None:
                logger.warning("No Rocksmith adapter found, using default input device")
                device_id = sd.default.device[0]  # Default input device
        
        # Set the default device
        sd.default.device = device_id
        sd.default.samplerate = sample_rate
        
        # Test if the device is working
        sd.check_input_settings()
        return True
        
    except Exception as e:
        logger.error(f"Error initializing audio device: {e}")
        return False
