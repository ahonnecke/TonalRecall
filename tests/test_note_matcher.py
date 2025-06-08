import unittest
from tonal_recall.note_matcher import NoteMatcher


class TestNoteMatcher(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(NoteMatcher.match("C#1", "C#1"))
        self.assertTrue(NoteMatcher.match("A", "A"))

    def test_octave_insensitive(self):
        self.assertTrue(NoteMatcher.match("C#", "C#1"))
        self.assertTrue(NoteMatcher.match("A", "A0"))
        self.assertTrue(NoteMatcher.match("F#", "F#2"))
        self.assertTrue(NoteMatcher.match("F#", "F#0"))  # Added test for F# vs F#0

    def test_enharmonic_equivalence(self):
        self.assertTrue(NoteMatcher.match("Gb", "F#0"))
        self.assertTrue(NoteMatcher.match("Bb", "A#1"))
        self.assertTrue(NoteMatcher.match("Db", "C#2"))
        self.assertTrue(NoteMatcher.match("Eb", "D#3"))
        self.assertTrue(NoteMatcher.match("Cb", "B4"))
        self.assertTrue(NoteMatcher.match("Fb", "E5"))
        self.assertTrue(NoteMatcher.match("Ab", "G#6"))

    def test_negative_cases(self):
        self.assertFalse(NoteMatcher.match("C", "D1"))
        self.assertFalse(NoteMatcher.match("F#", "G0"))
        self.assertFalse(NoteMatcher.match("Bb", "B1"))


if __name__ == "__main__":
    unittest.main()
