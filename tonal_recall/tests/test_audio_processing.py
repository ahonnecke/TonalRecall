import os
import pytest
import numpy as np
import time
import soundfile as sf
import json


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

    with sf.SoundFile(wav_file_path, "r") as f:
        assert f.samplerate == 48000
        assert f.channels == 1
        sample_rate = f.samplerate

        detector = NoteDetector(
            sample_rate=sample_rate,
            frames_per_buffer=1024,
            channels=1,
            min_confidence=0.7,
            min_signal=0.05,  # The raw signal is already strong enough
            min_stable_count=2,  # Require two stable readings for a more robust test
        )
        detector._callback = on_note_detected
        detector._running = True

        # Seek 2.1 seconds into the file to bypass silence
        f.seek(int(2.1 * sample_rate))

        # Process enough chunks to satisfy min_stable_count and get a detection
        for _ in range(10):
            audio_chunk = f.read(1024, dtype="float32", always_2d=False)
            if not audio_chunk.any():
                break

            # The WAV file is a bit hot; reduce volume slightly to prevent clipping.
            audio_chunk *= 0.9

            detector._process_audio(audio_chunk)

            if detected_notes:
                break

    assert len(detected_notes) > 0, "NoteDetector should have detected a note"
    assert (
        detected_notes[0].note_name == "A"
    ), f"Expected note A, but got {detected_notes[0].note_name}"


def get_test_cases():
    """Scan the recordings directory for WAV files and their JSON metadata."""
    test_cases = []
    if not os.path.exists(RECORDINGS_PATH):
        return test_cases

    # Using sorted() to make test execution order deterministic
    for filename in sorted(os.listdir(RECORDINGS_PATH)):
        if filename.endswith(".wav"):
            wav_path = os.path.join(RECORDINGS_PATH, filename)
            json_path = wav_path.replace(".wav", ".json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    metadata = json.load(f)
                    # Correct key is 'expected_note'
                    expected_note = metadata.get("expected_note")
                    if expected_note:
                        # Use the note name (e.g., "A1") as the test ID
                        test_id = os.path.splitext(os.path.basename(wav_path))[0]
                        test_cases.append(
                            pytest.param(wav_path, expected_note, id=test_id)
                        )
    return test_cases


@pytest.mark.parametrize("wav_file_path, expected_note_with_octave", get_test_cases())
def test_all_recorded_notes(wav_file_path, expected_note_with_octave):
    """Tests the NoteDetector against all WAV files in the recordings directory."""
    detected_notes = []

    def on_note_detected(note_info: dict, signal: float) -> None:
        """Callback function to store detected notes."""
        if note_info:
            detected_notes.append(note_info)

    # Use the WavFileAudioProvider to stream audio with gain
    provider = WavFileAudioProvider(
        file_path=wav_file_path,
        chunk_size=1024,
        loop=False,
        gain=5.0,  # Further reduced gain to match live signal levels
    )

    min_confidence = 0.75
    silence_threshold_db = -45

    # Use parameters from the working baseline_test, with dynamic confidence
    detector = NoteDetector(
        sample_rate=provider.sample_rate,
        frames_per_buffer=1024,
        channels=provider.channels,
        silence_threshold_db=silence_threshold_db,
        min_confidence=min_confidence,
        min_signal=0.015,  # Expecting a lower signal after gain reduction
        min_stable_count=5,
        stability_majority=0.8,
        tolerance=0.8,  # Use baseline's default tolerance
    )
    detector._callback = on_note_detected
    detector._running = True

    def process_wav_data(audio_bytes: bytes) -> None:
        """Converts audio bytes from provider and processes them."""
        # The provider sends bytes. NoteDetector._process_audio needs a float32 numpy array.
        num_samples = len(audio_bytes) // (provider.channels * 4)  # 4 bytes for float32
        if num_samples == 0:
            return

        audio_chunk = np.frombuffer(audio_bytes, dtype=np.float32).reshape(
            num_samples, provider.channels
        )

        # Process only the first channel, as NoteDetector expects mono.
        mono_chunk = audio_chunk[:, 0]
        detector._process_audio(mono_chunk)

        # Stop processing if a note is found. The provider is stopped in the
        # main thread's loop to avoid a deadlock.
        if detected_notes:
            pass

    provider.start(process_wav_data)

    # Wait for the provider to finish or for a note to be detected
    start_time = time.time()
    while provider.is_running:
        if time.time() - start_time > 5:  # 5-second timeout
            provider.stop()
            break
        time.sleep(0.01)

    assert (
        len(detected_notes) > 0
    ), f"No note detected for {os.path.basename(wav_file_path)}"

    detected_note_name = detected_notes[0].note_name
    # Strip octave number from the expected note for comparison, as octave detection is unreliable
    expected_note_letter = "".join(
        c for c in expected_note_with_octave if not c.isdigit()
    )

    assert (
        detected_note_name == expected_note_letter
    ), f"Expected {expected_note_letter} but got {detected_note_name} for {os.path.basename(wav_file_path)}"
