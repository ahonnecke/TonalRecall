import logging

note_matcher_logger = logging.getLogger("tonal_recall.note_matcher")
note_matcher_logger.setLevel(logging.INFO)  # Set to DEBUG for verbose output

class NoteMatcher:
    """
    Encapsulates logic for comparing detected notes to target notes,
    including normalization and enharmonic equivalence.
    """
    @staticmethod
    def normalize_to_sharp(note):
        flat_to_sharp = {
            "Ab": "G#", "Bb": "A#", "Cb": "B", "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#"
        }
        if len(note) > 1 and note[1] == 'b':
            return flat_to_sharp.get(note[:2], note[:2]) + note[2:]
        return note

    @classmethod
    def match(cls, target, played):
        """
        Returns True if the played note matches the target (octave-insensitive, enharmonic-aware).
        """
        normalized_target = cls.normalize_to_sharp(target)
        normalized_played = cls.normalize_to_sharp(played)
        result = normalized_target in normalized_played
        note_matcher_logger.debug(
            f"Matching: target={target} ({normalized_target}), played={played} ({normalized_played}) -> {result}"
        )
        return result
