"""
Simple test harness that ONLY imports the files and asserts True=True.
This test is designed to verify syntax without changing any logic.
"""

import sys
from pathlib import Path
from tonal_recall import note_detector
from tonal_recall import note_types
from tonal_recall import audio_device
from tonal_recall import note_utils
from tonal_recall import logger


# Add the project root to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all core modules can be imported without errors."""
    try:
        assert note_detector
        assert note_types
        assert audio_device
        assert note_utils
        assert logger
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import Error: {e}")
        return False


if __name__ == "__main__":
    print("\n=== Simple Import Test ===\n")
    success = test_imports()
    if success:
        print("\n✅ Test passed successfully!")
        sys.exit(0)
    else:
        print("\n❌ Test failed")
        sys.exit(1)
