import unittest
from tonal_recall.note_game_core import NoteGame
from tonal_recall.mock_note_detector import MockNoteDetector


class DummyNote:
    def __init__(self, name, string=None):
        self.name = name
        self.string = string


class TestNoteGame(unittest.TestCase):
    def setUp(self):
        self.mock_detector = MockNoteDetector()

    def test_stats_notes_played(self):
        game = NoteGame(note_detector=self.mock_detector)
        game.ui = None
        game.running = True
        game.pick_new_target()
        note = DummyNote(game.current_target)
        game.note_detected_callback(note, 1.0)
        self.assertIn(note.name[0], game.stats["notes_played"])

    def test_no_callback_when_not_running(self):
        game = NoteGame(note_detector=self.mock_detector)
        game.ui = None
        game.running = False
        note = DummyNote("A")
        # Should not throw or update stats
        game.note_detected_callback(note, 1.0)
        self.assertEqual(game.stats["correct_notes"], 0)


if __name__ == "__main__":
    unittest.main()
