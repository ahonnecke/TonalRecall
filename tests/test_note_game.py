import unittest
from tonal_recall.note_game_core import NoteGame
from tonal_recall.mock_note_detector import MockNoteDetector


class DummyNote:
    def __init__(self, name, string=None):
        self.note_name = name
        self.string = string


class TestNoteGame(unittest.TestCase):
    def setUp(self):
        self.mock_detector = MockNoteDetector()

    def test_stats_notes_played(self):
        game = NoteGame(note_detector=self.mock_detector)
        game.ui = None
        game.running = True

        # Test with a sharp note to replicate the failure case deterministically
        sharp_note_name = "G#"
        game.current_target = sharp_note_name
        note = DummyNote(game.current_target)
        game.note_detected_callback(note, 1.0)

        # The key in stats should be the full note name, "G#", not "G"
        self.assertIn(sharp_note_name, game.stats["notes_played"])
        self.assertEqual(game.stats["notes_played"][sharp_note_name], 1)

        # Also test with a natural note
        natural_note_name = "A"
        game.current_target = natural_note_name
        note = DummyNote(game.current_target)
        game.note_detected_callback(note, 1.0)
        self.assertIn(natural_note_name, game.stats["notes_played"])
        self.assertEqual(game.stats["notes_played"][natural_note_name], 1)

    def test_no_callback_when_not_running(self):
        game = NoteGame(note_detector=self.mock_detector)
        game.ui = None
        game.running = False
        note = DummyNote("A#")
        # Should not throw or update stats
        game.note_detected_callback(note, 1.0)
        self.assertEqual(game.stats["correct_notes"], 0)


if __name__ == "__main__":
    unittest.main()
