import re
from .logging_config import get_logger

# Get logger for this module
note_matcher_logger = get_logger("note_matcher")

# Compile regex to extract note name and octave
NOTE_PATTERN = re.compile(r"^([A-Ga-g][#b]?)([0-9]*)$")


class NoteMatcher:
    """
    Encapsulates logic for comparing detected notes to target notes,
    including normalization and enharmonic equivalence.
    """

    @staticmethod
    def normalize_to_sharp(note):
        flat_to_sharp = {
            "Ab": "G#",
            "Bb": "A#",
            "Cb": "B",
            "Db": "C#",
            "Eb": "D#",
            "Fb": "E",
            "Gb": "F#",
        }
        if len(note) > 1 and note[1] == "b":
            return flat_to_sharp.get(note[:2], note[:2]) + note[2:]
        return note

    @classmethod
    def match(cls, target, played):
        """
        Returns True if the played note matches the target (octave-insensitive, enharmonic-aware).

        Args:
            target: The target note (e.g., 'C', 'C#', 'Bb')
            played: The played note (e.g., 'C4', 'C#0', 'Bb2')

        Returns:
            bool: True if the notes match (considering enharmonic equivalents)
        """
        # Normalize both notes to sharp notation
        normalized_target = cls.normalize_to_sharp(target)
        normalized_played = cls.normalize_to_sharp(played)
        note_matcher_logger.debug(f"Normalized played {normalized_played}")

        # Extract just the note name (without octave) from the played note
        played_match = NOTE_PATTERN.match(normalized_played)
        if not played_match:
            note_matcher_logger.warning(f"Invalid played note format: {played}")
            return False

        played_note = played_match.group(1)  # Just the note name (e.g., 'C', 'C#')

        # Check if the normalized target matches the played note
        result = normalized_target == played_note

        # Log the matching attempt
        if result:
            note_matcher_logger.debug(
                f"Match SUCCESS: target={target} ({normalized_target}), "
                f"played={played} (normalized={played_note})"
            )
        else:
            note_matcher_logger.debug(
                f"Match FAILED: target={target} ({normalized_target}), "
                f"played={played} (normalized={played_note})"
            )

        return result
