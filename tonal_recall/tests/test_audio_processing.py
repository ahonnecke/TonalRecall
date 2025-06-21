import os
import pytest
import json
import time

# Imports for the new service-based architecture
from tonal_recall.services.note_detection_service import NoteDetectionService
from tonal_recall.services.audio_providers import WavFileAudioProvider
from tonal_recall.services.interfaces import DetectedNote

# Construct an absolute path to the recordings directory, which is at the project root
RECORDINGS_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "recordings")
)


def get_test_cases():
    """
    Generates test cases by finding all _test.wav files and their corresponding
    .json metadata files in the recordings directory.
    """
    test_cases = []
    json_files = {
        os.path.splitext(f)[0].replace("_test", ""): os.path.join(RECORDINGS_PATH, f)
        for f in os.listdir(RECORDINGS_PATH)
        if f.endswith(".json")
    }

    for wav_file in os.listdir(RECORDINGS_PATH):
        if wav_file.endswith("_test.wav"):
            base_name = os.path.splitext(wav_file)[0].replace("_test", "")
            if base_name in json_files:
                json_path = json_files[base_name]
                with open(json_path, "r") as f:
                    metadata = json.load(f)
                expected_note = metadata["expected_note"]
                wav_path = os.path.join(RECORDINGS_PATH, wav_file)
                # Mark known unstable tests
                marks = []
                if "E1_test.wav" in wav_path:
                    marks.append(
                        pytest.mark.xfail(
                            reason="E1 tests are known to be unstable with short samples."
                        )
                    )
                # Removed xfail mark for D2 test case

                test_cases.append(pytest.param(wav_path, expected_note, marks=marks))

    # Sort test cases by note name for consistent test order
    test_cases.sort(key=lambda x: x.values[1])
    return test_cases


@pytest.mark.parametrize("wav_file_path, expected_note_with_octave", get_test_cases())
def test_all_recorded_notes_with_service(wav_file_path, expected_note_with_octave):
    """
    Tests the NoteDetectionService against all WAV files in the recordings directory.
    """
    detected_notes = []
    note_found = False
    buffer_size = 8192  # Increased buffer for better low-frequency resolution

    def on_note_detected(note: DetectedNote | None):
        nonlocal note_found
        if note and not note_found:
            detected_notes.append(note)
            note_found = True

    provider = None
    service = None
    try:
        provider = WavFileAudioProvider(
            file_path=wav_file_path,
            chunk_size=buffer_size,
            gain=4.0,  # Gain applied in the original test
            loop=False,
        )

        service = NoteDetectionService(
            audio_provider=provider,
            frames_per_buffer=buffer_size,
            silence_threshold_db=-45,
            min_confidence=0.8,
            min_signal=0.015,
            min_stable_count=5,
            tolerance=0.8,
        )

        service.start(on_note_detected)

        start_time = time.time()
        while not note_found and time.time() - start_time < 3:
            time.sleep(0.1)

    except Exception as e:
        pytest.fail(f"Error setting up or running service for {wav_file_path}: {e}")
    finally:
        if service:
            service.stop()

    assert len(detected_notes) > 0, (
        f"No note detected for {os.path.basename(wav_file_path)}"
    )

    detected_note_name = detected_notes[0].note_name

    assert detected_note_name == expected_note_with_octave, (
        f"Expected {expected_note_with_octave} but got {detected_note_name} for {os.path.basename(wav_file_path)}"
    )
