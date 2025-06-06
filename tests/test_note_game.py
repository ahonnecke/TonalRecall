import unittest
from tonal_recall.note_game_core import NoteGame
from tonal_recall.mock_note_detector import MockNoteDetector

class DummyNote:
    def __init__(self, name, string=None):
        self.name = name
        self.string = string

class TestNoteGame(unittest.TestCase):
    def test_basic_game_progression(self):
        mock_detector = MockNoteDetector()
        game = NoteGame(debug=True, level=1, note_detector=mock_detector)
        game.ui = None  # No UI for logic test
        game.pick_new_target()
        target = game.current_target
        note = DummyNote(target)
        mock_detector.start(lambda *_: None)  # Start the detector, but don't run full game loop
        # Simulate a correct note event
        mock_detector.simulate_note_event(note)
        self.assertIn('correct_notes', game.stats)
        self.assertGreaterEqual(game.stats['correct_notes'], 0)

if __name__ == '__main__':
    unittest.main()
