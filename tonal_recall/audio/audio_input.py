"""Audio input handling for note detection."""

from __future__ import annotations
import numpy as np
import sounddevice as sd
import time
from typing import Optional, Dict, Any, Tuple, Callable, ClassVar
from abc import ABC, abstractmethod

from ..logger import get_logger
from ..core.interfaces import IAudioInput

logger = get_logger(__name__)


class AudioInputHandler(IAudioInput, ABC):
    """Abstract base class for audio input handlers."""

    def is_running(self) -> bool:
        """Check if audio input is running.

        Returns:
            True if audio input is running, False otherwise
        """
        return self._running

    @abstractmethod
    def start(self, callback: Callable[[np.ndarray, float], None]) -> bool:
        """Start capturing audio and pass it to the callback.

        Args:
            callback: Function to call with audio data and timestamp

        Returns:
            True if started successfully, False otherwise
        """
        pass

    @abstractmethod
    def stop(self) -> None:
        """Stop capturing audio."""
        pass


class SoundDeviceInput(AudioInputHandler):
    """Audio input handler using sounddevice library."""

    # Audio configuration
    SAMPLE_RATE: ClassVar[int] = 44100  # Hz
    FRAMES_PER_BUFFER: ClassVar[int] = (
        512  # Number of frames per buffer - must match hop_size in NoteDetector
    )
    CHANNELS: ClassVar[int] = 1  # Mono audio

    def __init__(
        self,
        device_id: Optional[int] = None,
        sample_rate: Optional[int] = None,
        frames_per_buffer: Optional[int] = None,
        channels: Optional[int] = None,
    ) -> None:
        """Initialize the audio input handler.

        Args:
            device_id: Audio input device ID, or None to auto-detect
            sample_rate: Sample rate in Hz, or None for default (44100)
            frames_per_buffer: Buffer size in frames, or None for default (1024)
            channels: Number of audio channels, or None for default (1)
        """
        self._device_id = device_id
        self._sample_rate = sample_rate or self.SAMPLE_RATE
        self._frames_per_buffer = frames_per_buffer or self.FRAMES_PER_BUFFER
        self._channels = channels or self.CHANNELS

        self._stream = None
        self._callback = None
        self._running = False

        # Initialize audio device
        self._init_audio_device()

    def _init_audio_device(self) -> None:
        """Initialize the audio input device."""
        # Common supported sample rates to try
        common_rates = [44100, 48000, 22050, 16000, 8000]

        # If requested rate is not in the list, add it first
        if self._sample_rate not in common_rates:
            common_rates.insert(0, self._sample_rate)
        elif self._sample_rate != common_rates[0]:
            # Make sure requested rate is first if it's in the list
            common_rates.remove(self._sample_rate)
            common_rates.insert(0, self._sample_rate)

        # Try to find Rocksmith adapter if no device ID provided
        if self._device_id is None:
            self._device_id, _ = self._find_rocksmith_adapter()
            if self._device_id is None:
                logger.warning("No Rocksmith adapter found, using default input device")
                try:
                    self._device_id = sd.default.device[0]  # Default input device
                except Exception:
                    logger.warning("Could not get default input device, using None")
                    self._device_id = None

        # Try each sample rate until one works
        for rate in common_rates:
            try:
                logger.info(f"Trying sample rate: {rate} Hz")
                sd.default.device = self._device_id
                sd.default.samplerate = rate

                # Test if the device is working
                sd.check_input_settings()

                # If we get here, the settings work
                self._sample_rate = rate  # Update to the working rate
                logger.info(
                    f"Audio device initialized: ID={self._device_id}, Rate={rate}Hz"
                )
                return

            except Exception as e:
                logger.warning(f"Sample rate {rate} Hz not supported: {e}")
                continue

        # If we get here, none of the sample rates worked with the selected device
        # Try with default device
        logger.warning("Trying default audio device with various sample rates")
        sd.default.device = None

        for rate in common_rates:
            try:
                sd.default.samplerate = rate
                sd.check_input_settings()

                # If we get here, the settings work
                self._sample_rate = rate
                self._device_id = None
                logger.info(
                    f"Audio device initialized with default device, Rate={rate}Hz"
                )
                return

            except Exception as e:
                logger.warning(
                    f"Default device with sample rate {rate} Hz not supported: {e}"
                )
                continue

        # If we get here, nothing worked
        raise Exception(
            "Could not initialize audio device with any supported sample rate"
        )

    def _find_rocksmith_adapter(self) -> Tuple[Optional[int], Optional[Dict[str, Any]]]:
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
                    logger.info(f"Found Rocksmith adapter: {device['name']}")
                    return device_id, device
            return None, None
        except Exception as e:
            logger.error(f"Error finding Rocksmith adapter: {e}")
            return None, None

    def _audio_callback(
        self,
        indata: np.ndarray,
        _frames: int,
        _time_info: dict,
        status: sd.CallbackFlags,
    ) -> None:
        """Callback for processing audio data from the input stream.

        Args:
            indata: The input audio data as a numpy array (frames x channels)
            _frames: Number of frames in the buffer
            _time_info: Dictionary with timing information
            status: Status flags indicating whether input/output underflow or overflow occurred

        Note:
            This is called from a separate audio thread, so it should be fast
            and avoid any blocking operations to prevent audio glitches.
        """
        if status:
            logger.warning(f"Audio callback status: {status}")

        if self._callback:
            # Extract mono audio data (take first channel if multi-channel)
            audio_data = indata[:, 0] if indata.ndim > 1 else indata

            # Call the user's callback with the audio data and current time
            self._callback(audio_data, time.time())

    def start(self, callback: Callable[[np.ndarray, float], None]) -> bool:
        """Start capturing audio and pass it to the callback.

        Args:
            callback: Function to call with audio data and timestamp
        """
        if self._running:
            logger.warning("Audio input already running")
            return

        self._callback = callback

        # Try different sample rates if the default one fails
        sample_rates_to_try = [48000, 44100, 22050, 16000, 8000]

        # Make sure our current rate is first in the list
        if self._sample_rate in sample_rates_to_try:
            sample_rates_to_try.remove(self._sample_rate)
        sample_rates_to_try.insert(0, self._sample_rate)

        success = False
        for rate in sample_rates_to_try:
            try:
                logger.info(f"Trying to start audio input with sample rate: {rate} Hz")
                self._sample_rate = rate
                self._stream = sd.InputStream(
                    samplerate=rate,
                    blocksize=self._frames_per_buffer,
                    channels=self._channels,
                    callback=self._audio_callback,
                )
                self._stream.start()
                self._running = True
                logger.info(f"Audio input started with sample rate {rate} Hz")
                success = True
                break
            except Exception as e:
                logger.warning(
                    f"Failed to start audio input with sample rate {rate} Hz: {e}"
                )
                if self._stream:
                    try:
                        self._stream.close()
                    except Exception:  # pylint: disable=broad-except
                        pass
                    self._stream = None

        if not success:
            error_msg = "Could not start audio input with any sample rate"
            logger.error(error_msg)
            return False

        return True

    def stop(self) -> None:
        """Stop capturing audio."""
        if not self._running:
            return

        try:
            if self._stream:
                self._stream.stop()
                self._stream.close()
                self._stream = None
            self._running = False
            logger.info("Audio input stopped")
        except Exception as e:
            logger.error(f"Error stopping audio input: {e}")
