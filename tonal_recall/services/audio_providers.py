import sounddevice as sd
import soundfile as sf
import numpy as np
import threading
import time
from typing import Callable, Optional

from tonal_recall.services.interfaces import IAudioProvider


class LiveAudioProvider(IAudioProvider):
    """Provides live audio from an input device using sounddevice."""

    def __init__(
        self, device_id: int, sample_rate: int, channels: int, chunk_size: int
    ):
        self._device_id = device_id
        self._sample_rate = sample_rate
        self._channels = channels
        self._chunk_size = chunk_size
        self._stream: Optional[sd.InputStream] = None
        self._on_data_callback: Optional[Callable[[bytes], None]] = None

    def start(self, on_data_callback: Callable[[bytes], None]) -> None:
        self._on_data_callback = on_data_callback
        self._stream = sd.InputStream(
            device=self._device_id,
            channels=self._channels,
            samplerate=self._sample_rate,
            blocksize=self._chunk_size,
            callback=self._audio_callback,
            dtype="float32",  # Standard for audio processing
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def _audio_callback(
        self, indata: np.ndarray, _frames: int, _time_info, status
    ) -> None:
        if status:
            print(status, flush=True)
        if self._on_data_callback:
            # Convert numpy array to bytes for consistent interface
            self._on_data_callback(indata.tobytes())

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels


class WavFileAudioProvider(IAudioProvider):
    """Provides audio data by reading from a WAV file."""

    def __init__(
        self, file_path: str, chunk_size: int, loop: bool = False, gain: float = 1.0
    ):
        self._file_path = file_path
        self._chunk_size = chunk_size
        self._loop = loop
        self._gain = gain
        self._on_data_callback: Optional[Callable[[bytes], None]] = None
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

        with sf.SoundFile(self._file_path) as f:
            self._sample_rate = f.samplerate
            self._channels = f.channels

    def start(self, on_data_callback: Callable[[bytes], None]) -> None:
        if self._is_running:
            return

        self._on_data_callback = on_data_callback
        self._is_running = True
        self._thread = threading.Thread(target=self._stream_data)
        self._thread.start()

    def stop(self) -> None:
        self._is_running = False
        if self._thread:
            self._thread.join()
            self._thread = None

    @property
    def is_running(self) -> bool:
        """Returns True if the provider is currently streaming data."""
        return self._is_running

    def _stream_data(self) -> None:
        while self._is_running:
            try:
                with sf.SoundFile(self._file_path) as f:
                    while self._is_running:
                        data = f.read(self._chunk_size, dtype="float32", always_2d=True)
                        if len(data) == 0:
                            if self._loop:
                                f.seek(0)
                                continue
                            else:
                                break

                        # Apply gain if specified
                        if self._gain != 1.0:
                            data *= self._gain

                        if self._on_data_callback:
                            # Convert to bytes for consistent interface
                            self._on_data_callback(data.tobytes())

                        # Simulate real-time playback speed
                        time.sleep(self._chunk_size / self.sample_rate)

                if not self._loop:
                    break  # Exit outer loop if not looping

            except Exception as e:
                print(f"Error streaming WAV file: {e}")
                break

        self._is_running = False  # Ensure flag is reset on exit

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels
