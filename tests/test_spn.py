import unittest
from tonal_recall.note_utils import get_note_name


class TestScientificPitchNotation(unittest.TestCase):
    def test_middle_c(self):
        # Middle C (C4) should be ~261.63 Hz
        self.assertEqual(get_note_name(261.63), "C4")

    def test_a4(self):
        # A4 should be 440 Hz
        self.assertEqual(get_note_name(440.0), "A4")

    def test_octave_transitions(self):
        # Test octave transitions (B3 -> C4)
        self.assertEqual(get_note_name(246.94), "B3")
        self.assertEqual(get_note_name(261.63), "C4")

    def test_sharps_and_flats(self):
        # Test sharp notes
        self.assertEqual(get_note_name(277.18), "C#4")  # C#4/Db4
        self.assertEqual(get_note_name(311.13), "D#4")  # D#4/Eb4

        # Test flat notes when requested
        self.assertEqual(get_note_name(277.18, use_flats=True), "Db4")
        self.assertEqual(get_note_name(311.13, use_flats=True), "Eb4")

        # Some notes should always use sharps in SPN (E#, B#)
        self.assertEqual(get_note_name(329.63), "E4")  # Should not be Fb4
        self.assertEqual(get_note_name(493.88), "B4")  # Should not be Cb5


if __name__ == "__main__":
    unittest.main()
