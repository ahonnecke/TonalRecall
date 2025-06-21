"""
Simple test harness for the note_detector module.
This test only imports the module and performs a basic assertion to verify syntax.
"""

import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_note_detector_import():
    """Test that the note_detector module can be imported without errors."""
    try:
        from tonal_recall.note_detector import NoteDetector
        from tonal_recall.note_types import DetectedNote

        print("✅ Successfully imported NoteDetector and DetectedNote")
        assert NoteDetector is not None, "NoteDetector should be importable"
        assert DetectedNote is not None, "DetectedNote should be importable"
        assert True  # Simple assertion to verify the test runs
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        assert False, f"Import Error: {e}"
    except SyntaxError as e:
        print(f"❌ Syntax Error: {e}")
        assert False, f"Syntax Error: {e}"
    except Exception as e:
        print(f"❌ Unexpected Error: {e}")
        assert False, f"Unexpected Error: {e}"


def test_note_detector_basic_functionality():
    """Test basic functionality of the NoteDetector class without running audio."""
    try:
        from tonal_recall.note_detector import NoteDetector

        # Create an instance without initializing audio
        detector = NoteDetector(device_id=None)

        # Test a simple method that doesn't require audio input
        note_name = "A4"
        flat_notation = detector.convert_note_notation(note_name, to_flats=True)
        sharp_notation = detector.convert_note_notation(flat_notation, to_flats=False)

        print(
            f"✅ Note conversion test: {note_name} -> {flat_notation} -> {sharp_notation}"
        )
        assert sharp_notation == note_name, (
            f"Expected {note_name}, got {sharp_notation}"
        )
    except Exception as e:
        print(f"❌ Functionality Error: {e}")
        assert False, f"Functionality Error: {e}"


if __name__ == "__main__":
    print("\n=== Note Detector Import Test ===\n")

    import_success = test_note_detector_import()
    if import_success:
        functionality_success = test_note_detector_basic_functionality()

        if functionality_success:
            print("\n✅ All tests passed successfully!")
            sys.exit(0)
        else:
            print("\n❌ Basic functionality test failed")
            sys.exit(1)
    else:
        print("\n❌ Import test failed")
        sys.exit(1)
