import unittest
from tonal_recall.note_game_core import NoteGame
from tonal_recall.mock_note_detector import MockNoteDetector
from tonal_recall.note_types import DetectedNote


class TestNoteGame(unittest.TestCase):
    def setUp(self):
        self.mock_detector = MockNoteDetector()

    def test_stats_notes_played(self):
        game = NoteGame(note_detector=self.mock_detector, difficulty=3)
        game.ui = None
        game.running = True

        # Test with a sharp note
        sharp_note_name = "G#"
        game.current_target = sharp_note_name
        note = DetectedNote(
            note_name="G#4",
            frequency=415.3,
            confidence=0.9,
            signal_max=0.5,
            is_stable=True,
            timestamp=0,
        )
        game.note_detected_callback(note, 1.0)
        game.process_events()

        self.assertIn(sharp_note_name, game.stats["notes_played"])
        self.assertEqual(game.stats["notes_played"][sharp_note_name], 1)

        # Also test with a natural note
        natural_note_name = "A"
        game.current_target = natural_note_name
        note = DetectedNote(
            note_name="A4",
            frequency=440.0,
            confidence=0.9,
            signal_max=0.5,
            is_stable=True,
            timestamp=0,
        )
        game.note_detected_callback(note, 1.0)
        game.process_events()
        self.assertIn(natural_note_name, game.stats["notes_played"])
        self.assertEqual(game.stats["notes_played"][natural_note_name], 1)

    def test_no_callback_when_not_running(self):
        game = NoteGame(note_detector=self.mock_detector, difficulty=3)
        game.ui = None
        game.running = False
        note = DetectedNote(
            note_name="A#4",
            frequency=466.16,
            confidence=0.9,
            signal_max=0.5,
            is_stable=True,
            timestamp=0,
        )
        # Should not throw or update stats
        game.note_detected_callback(note, 1.0)
        game.process_events()
        self.assertEqual(game.stats["correct_notes"], 0)


if __name__ == "__main__":
    unittest.main()
