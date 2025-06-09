import re
from .logging_config import get_logger

# Get logger for this module
note_matcher_logger = get_logger("tonal_recall.note_matcher")
note_matcher_logger.debug("Initializing note matcher")
# Compile regex to extract note name and octave
# This pattern matches:
# - Note name (A-G, case insensitive)
# - Optional accidental (# or b)
# - Optional octave number (0-9+)
NOTE_PATTERN = re.compile(r"^([A-Ga-g][#b]?)([0-9]*)$", re.IGNORECASE)


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
        Check if the played note matches the target note, ignoring octave.

        Args:
            target: The target note (e.g., 'A', 'A#', 'Bb')
            played: The played note (e.g., 'A4', 'A#3', 'Bb2')

        Returns:
            bool: True if the notes match (ignoring octave), False otherwise
        """
        # Log the initial matching attempt with more context
        note_matcher_logger.debug(
            f"Matching notes - Target: '{target}' (type: {type(target)}), "
            f"Played: '{played}' (type: {type(played)})"
        )

        # Ensure inputs are strings
        target = str(target).strip() if target is not None else ""
        played = str(played).strip() if played is not None else ""

        # Handle empty inputs
        if not target or not played:
            note_matcher_logger.debug(
                f"Empty input - Target: '{target}', Played: '{played}'"
            )
            return False

        try:
            # Normalize both notes to use sharps for consistency
            normalized_target = cls.normalize_to_sharp(target)
            normalized_played = cls.normalize_to_sharp(played)
            note_matcher_logger.debug(
                f"Normalized notes - Target: '{normalized_target}', Played: '{normalized_played}'"
            )

            # Log the raw regex matches for debugging
            target_match = NOTE_PATTERN.match(normalized_target)
            played_match = NOTE_PATTERN.match(normalized_played)
            note_matcher_logger.debug(
                f"Regex matches - Target match: {target_match.groups() if target_match else 'None'}, "
                f"Played match: {played_match.groups() if played_match else 'None'}"
            )

            # Parse the notes into components
            target_match = NOTE_PATTERN.match(normalized_target)
            played_match = NOTE_PATTERN.match(normalized_played)

            # Validate note formats
            if not target_match:
                note_matcher_logger.warning(
                    f"Invalid target note format: '{target}' (normalized: '{normalized_target}')"
                )
                return False

            if not played_match:
                note_matcher_logger.debug(
                    f"Invalid played note format: '{played}' (normalized: '{normalized_played}')"
                )
                return False

            # Extract note names (ignoring case)
            target_note = target_match.group(1).upper()
            played_note = played_match.group(1).upper()

            # Extract octaves if they exist
            target_octave = target_match.group(2) or None
            played_octave = played_match.group(2) or None

            # Log the parsed components
            note_matcher_logger.debug(
                f"Parsed components - Target note: '{target_note}', octave: {target_octave}, "
                f"Played note: '{played_note}', octave: {played_octave}"
            )

            note_matcher_logger.debug(
                f"Parsed notes - Target: '{target_note}' (octave: {target_octave}), "
                f"Played: '{played_note}' (octave: {played_octave})"
            )

            # Handle enharmonic equivalents
            enharmonic_map = {
                "B#": "C",
                "E#": "F",
                "Cb": "B",
                "Fb": "E",
                "A##": "B",
                "B##": "C#",
                "C##": "D",
                "D##": "E",
                "E##": "F#",
                "F##": "G",
                "G##": "A",
                "Abb": "G",
                "Bbb": "A",
                "Cbb": "Bb",
                "Dbb": "C",
                "Ebb": "D",
                "Fbb": "Eb",
                "Gbb": "F",
            }

            # Check for direct match
            if target_note == played_note:
                note_matcher_logger.debug(
                    f"Note match - Direct: '{played_note}' matches target '{target_note}'"
                )
                return True

            # Check for enharmonic equivalents
            if (
                target_note in enharmonic_map
                and enharmonic_map[target_note] == played_note
            ):
                note_matcher_logger.debug(
                    f"Note match - Enharmonic: '{played_note}' matches target '{target_note}' "
                    f"(via {enharmonic_map[target_note]})"
                )
                return True

            if (
                played_note in enharmonic_map
                and enharmonic_map[played_note] == target_note
            ):
                note_matcher_logger.debug(
                    f"Note match - Enharmonic: '{played_note}' (as {enharmonic_map[played_note]}) "
                    f"matches target '{target_note}'"
                )
                return True

            # If we get here, no match was found
            note_matcher_logger.debug(
                f"No match found - Target: '{target_note}', Played: '{played_note}'. "
                f"No enharmonic equivalent found."
            )
            return False

        except Exception as e:
            note_matcher_logger.error(
                f"Error matching notes - Target: '{target}', Played: '{played}'. Error: {str(e)}",
                exc_info=True,
            )
            return False
