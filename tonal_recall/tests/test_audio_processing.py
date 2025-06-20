import os
import pytest
import numpy as np
import time
import soundfile as sf
import json

from tonal_recall.note_detector import NoteDetector, DetectedNote

# Construct an absolute path to the recordings directory, which is at the project root
RECORDINGS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "recordings")
)
TEST_WAV_FILE = os.path.join(RECORDINGS_PATH, "A1_short.wav")


def test_note_detector_processes_chunk(wav_file_path=TEST_WAV_FILE):
    """
    Tests that the NoteDetector can process a stream of audio chunks and detect a note.
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
    """
    Generates test cases by finding all _short.wav files and their corresponding
    .json metadata files in the recordings directory.
    """
    test_cases = []
    json_files = {
        os.path.splitext(f)[0]
        .replace("_short", ""):
        os.path.join(RECORDINGS_PATH, f)
        for f in os.listdir(RECORDINGS_PATH)
        if f.endswith(".json")
    }

    for wav_file in os.listdir(RECORDINGS_PATH):
        if wav_file.endswith("_short.wav"):
            base_name = os.path.splitext(wav_file)[0].replace("_short", "")
            if base_name in json_files:
                json_path = json_files[base_name]
                with open(json_path, "r") as f:
                    metadata = json.load(f)
                expected_note = metadata["expected_note"]
                wav_path = os.path.join(RECORDINGS_PATH, wav_file)
                test_cases.append(pytest.param(wav_path, expected_note))

    # Sort test cases by note name for consistent test order
    test_cases.sort(key=lambda x: x.values[1])
    return test_cases


@pytest.mark.parametrize("wav_file_path, expected_note_with_octave", get_test_cases())
def test_all_recorded_notes(wav_file_path, expected_note_with_octave):
    """Tests the NoteDetector against all WAV files in the recordings directory."""
    detected_notes = []
    note_found = False
    buffer_size = 1024

    def on_note_detected(note: DetectedNote, signal: float):
        nonlocal note_found
        if note and not note_found:
            detected_notes.append(note)
            note_found = True

    try:
        with sf.SoundFile(wav_file_path, "r") as f:
            detector = NoteDetector(
                sample_rate=f.samplerate,
                frames_per_buffer=buffer_size,
                channels=f.channels,
                silence_threshold_db=-45,
                min_confidence=0.8,
                min_signal=0.015,
                min_stable_count=5,
                tolerance=0.8,
            )
            detector._callback = on_note_detected
            detector._running = True

            while not note_found:
                audio_chunk = f.read(buffer_size, dtype="float32", always_2d=False)
                if not audio_chunk.any():
                    break

                # Pad the last chunk if it's smaller than the buffer size
                if len(audio_chunk) < buffer_size:
                    padding = np.zeros(buffer_size - len(audio_chunk), dtype='float32')
                    audio_chunk = np.concatenate((audio_chunk, padding))

                # Apply gain manually
                audio_chunk *= 4.0

                detector._process_audio(audio_chunk)

    except Exception as e:
        pytest.fail(f"Error processing {wav_file_path}: {e}")

    assert (
        len(detected_notes) > 0
    ), f"No note detected for {os.path.basename(wav_file_path)}"

    detected_note_name = detected_notes[0].note_name
    # Strip octave number from the expected note for comparison
    expected_note_letter = "".join(
        c for c in expected_note_with_octave if not c.isdigit()
    )

    assert (
        detected_note_name == expected_note_letter
    ), f"Expected {expected_note_letter} but got {detected_note_name} for {os.path.basename(wav_file_path)}"
