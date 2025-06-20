import os
import json
import sys
import time
import argparse
from collections import Counter

import numpy as np
from typing import List, Dict, Any

from tonal_recall.services.audio_providers import WavFileAudioProvider
from tonal_recall.note_detector import NoteDetector
from tonal_recall.note_types import DetectedNote


class TestHarness:
    """A test harness to validate NoteDetector against pre-recorded samples."""

    def __init__(self, recordings_path: str):
        self.recordings_path = recordings_path
        self.test_files = self._find_test_files()
        if not self.test_files:
            print(
                f"Warning: No test files (.wav/.json pairs) found in {recordings_path}"
            )

    def _find_test_files(self) -> List[tuple[str, str]]:
        """Find all corresponding .wav and .json files in the recordings path."""
        pairs = []
        for filename in os.listdir(self.recordings_path):
            if filename.endswith(".wav"):
                wav_path = os.path.join(self.recordings_path, filename)
                json_path = os.path.splitext(wav_path)[0] + ".json"
                if os.path.exists(json_path):
                    pairs.append((wav_path, json_path))
        return pairs

    def run(self) -> bool:
        """Run all tests and print a summary report."""
        print(f"Found {len(self.test_files)} test cases.")
        results = []
        passed_count = 0

        for wav_path, json_path in self.test_files:
            result = self._run_single_test(wav_path, json_path)
            results.append(result)
            if result["passed"]:
                passed_count += 1
                status = "PASS"
            else:
                status = "FAIL"

            print(
                f"- Test: {os.path.basename(wav_path):<15} | Status: {status:<4} | Expected: {result['expected']:<4} | Got: {result['most_common'] or 'None'}"
            )

        print("\n--- Test Summary ---")
        print(f"{passed_count} / {len(self.test_files)} tests passed.")
        return passed_count == len(self.test_files)

    def _run_single_test(self, wav_path: str, json_path: str) -> Dict[str, Any]:
        """Run a single test case against one WAV file."""
        with open(json_path, "r") as f:
            ground_truth = json.load(f)
        expected_note = ground_truth["expected_note"]

        detected_notes_history: List[str] = []

        def on_note_detected(note: DetectedNote, signal_strength: float):
            if note and note.name:
                detected_notes_history.append(note.name)

        provider = WavFileAudioProvider(file_path=wav_path, chunk_size=1024)
        print(
            f"    [Debug] Initializing NoteDetector for {os.path.basename(wav_path)} with SR={provider.sample_rate}Hz"
        )

        detector = NoteDetector(
            sample_rate=provider.sample_rate,
            frames_per_buffer=provider._chunk_size,
            channels=provider.channels,
            min_confidence=0.2,  # Keep low confidence for now
            min_stable_count=1,  # Make stability check extremely lenient
        )
        detector._callback = on_note_detected

        def process_audio_chunk(audio_bytes: bytes):
            num_samples = len(audio_bytes) // (4 * provider.channels)
            if num_samples == 0:
                return
            audio_data_2d = np.frombuffer(audio_bytes, dtype=np.float32).reshape(
                num_samples, provider.channels
            )
            # Create a writable copy to apply gain, as the original buffer is read-only
            mono_audio_data = audio_data_2d[:, 0].copy()

            # The previous gain of 15000 was causing severe clipping.
            # A much lower gain is needed for these recordings.
            gain = 50.0
            mono_audio_data *= gain

            detector._process_audio(mono_audio_data)

        provider.start(on_data_callback=process_audio_chunk)
        while getattr(provider, "_is_running", False):
            time.sleep(0.1)
        provider.stop()

        most_common_note = None
        if detected_notes_history:
            note_counts = Counter(detected_notes_history)
            most_common_note = note_counts.most_common(1)[0][0]

        passed = most_common_note == expected_note

        return {
            "passed": passed,
            "expected": expected_note,
            "detected_notes": detected_notes_history,
            "most_common": most_common_note,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NoteDetector test harness.")
    parser.add_argument(
        "--recordings-path",
        type=str,
        default=os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "recordings")
        ),
        help="Path to the directory with WAV and JSON test files.",
    )
    args = parser.parse_args()

    harness = TestHarness(recordings_path=args.recordings_path)
    all_passed = harness.run()

    if not all_passed:
        sys.exit(1)  # Exit with a non-zero code if any test failed
    sys.exit(0)
