import os
import pytest
import numpy as np
import time
import soundfile as sf
import logging



from tonal_recall.services.audio_providers import WavFileAudioProvider
from tonal_recall.note_detector import NoteDetector, DetectedNote




# Construct an absolute path to the recordings directory, which is at the project root
RECORDINGS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "recordings")
)
TEST_WAV_FILE = os.path.join(RECORDINGS_PATH, "A1.wav")


@pytest.fixture
def audio_provider():
    """Fixture to provide a WavFileAudioProvider instance for A1.wav."""
    if not os.path.exists(TEST_WAV_FILE):
        pytest.skip(f"Test WAV file not found at {TEST_WAV_FILE}")
    return WavFileAudioProvider(file_path=TEST_WAV_FILE, chunk_size=1024)


def test_wav_provider_initialization(audio_provider):
    """Test that the WavFileAudioProvider initializes correctly."""
    assert audio_provider.sample_rate == 48000
    assert audio_provider.channels > 0
    assert audio_provider._chunk_size == 1024


def test_read_and_process_one_chunk(audio_provider):
    """Test reading one chunk, applying gain, and verifying the signal."""
    captured_data = []

    def on_data_callback(audio_bytes: bytes):
        num_samples = len(audio_bytes) // (4 * audio_provider.channels)
        if num_samples > 0:
            audio_data_2d = np.frombuffer(audio_bytes, dtype=np.float32).reshape(
                num_samples, audio_provider.channels
            )
            mono_audio_data = audio_data_2d[:, 0].copy()

            # Check data before gain
            assert (
                np.max(np.abs(mono_audio_data)) > 1e-6
            ), "Initial audio data should not be silent"

            # Apply gain
            gain = 1.0
            mono_audio_data *= gain

            captured_data.append(mono_audio_data)

        # Stop after one chunk
        audio_provider.stop()

    audio_provider.start(on_data_callback=on_data_callback)

    # The provider runs in a separate thread, wait for it to finish
    timeout = 5  # seconds
    start_time = time.time()
    while audio_provider.is_running and (time.time() - start_time) < timeout:
        time.sleep(0.01)

    assert not audio_provider.is_running, "Audio provider did not stop in time"
    assert len(captured_data) > 0, "Callback should have been called at least once"

    processed_chunk = captured_data[0]
    assert (
        np.max(np.abs(processed_chunk)) > 0.0
    ), "Processed audio data should not be silent"
    assert (
        np.max(np.abs(processed_chunk)) < 1.0
    ), "Signal should not be clipped after gain"


def test_note_detector_processes_chunk(wav_file_path=TEST_WAV_FILE):
    """
    Tests that the NoteDetector can process a stream of audio chunks and detect a note.
    This test seeks past the initial silence in the WAV file and processes multiple
    chunks to ensure the pitch detector has enough data.
    """
    detected_notes = []

    def on_note_detected(note: DetectedNote, signal: float):
        detected_notes.append(note)

    with sf.SoundFile(wav_file_path, 'r') as f:
        assert f.samplerate == 48000
        assert f.channels == 1
        sample_rate = f.samplerate

        detector = NoteDetector(
            sample_rate=sample_rate,
            frames_per_buffer=1024,
            channels=1,
            min_confidence=0.7,
            min_signal=0.05,  # The raw signal is already strong enough
            min_stable_count=2  # Require two stable readings for a more robust test
        )
        detector._callback = on_note_detected
        detector._running = True

        # Seek 2.1 seconds into the file to bypass silence
        f.seek(int(2.1 * sample_rate))

        # Process enough chunks to satisfy min_stable_count and get a detection
        for _ in range(10):
            audio_chunk = f.read(1024, dtype='float32', always_2d=False)
            if not audio_chunk.any():
                break

            # The WAV file is a bit hot; reduce volume slightly to prevent clipping.
            audio_chunk *= 0.9

            detector._process_audio(audio_chunk)

            if detected_notes:
                break

    assert len(detected_notes) > 0, "NoteDetector should have detected a note"
    assert detected_notes[0].note_name == "A", f"Expected note A, but got {detected_notes[0].note_name}"

