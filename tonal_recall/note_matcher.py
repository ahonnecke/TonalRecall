import re
from .logger import get_logger

# Get logger for this module
logger = get_logger(__name__)

# Compile regex to extract note name and octave
# This pattern matches:
# - Note name (A-G, case insensitive)
# - Optional accidental (# or b)
# - Optional octave number (0-9+)
NOTE_PATTERN = re.compile(r"^([A-Ga-g][#b]?)([0-9]*)", re.IGNORECASE)


class NoteMatcher:
    """
    Encapsulates logic for comparing detected notes to target notes,
    including normalization and enharmonic equivalence.
    """

    @staticmethod
    def normalize_to_sharp(note: str) -> str:
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
    def match(cls, target: str, played: str) -> bool:
        """
        Check if the played note matches the target note, ignoring octave.

        Args:
            target: The target note (e.g., 'A', 'A#', 'Bb')
            played: The played note (e.g., 'A4', 'A#3', 'Bb2')
        Returns:
            bool: True if the notes match (ignoring octave), False otherwise
        """
        # Log the initial matching attempt with more context
        logger.debug(
            f"🎵 MATCHER - Starting note matching"
            f"\n  • Target: '{target}' (type: {type(target)})"
            f"\n  • Played: '{played}' (type: {type(played)})"
        )

        # Ensure inputs are strings
        target = str(target).strip() if target is not None else ""
        played = str(played).strip() if played is not None else ""

        # Handle empty inputs
        if not target or not played:
            logger.warning(f"⚠️  EMPTY INPUT - Target: '{target}', Played: '{played}'")
            return False

        try:
            # Log the raw input before any processing
            logger.debug(f"🔍 INPUT - Raw target: '{target}', Raw played: '{played}'")

            # Normalize both notes to use sharps for consistency
            normalized_target = cls.normalize_to_sharp(target)
            normalized_played = cls.normalize_to_sharp(played)
            logger.debug(
                f"🔄 NORMALIZED"
                f"\n  • Target: '{target}' → '{normalized_target}'"
                f"\n  • Played: '{played}' → '{normalized_played}'"
            )

            # Log the raw regex matches for debugging
            target_match = NOTE_PATTERN.match(normalized_target)
            played_match = NOTE_PATTERN.match(normalized_played)

            logger.debug(
                f"🔍 REGEX MATCHES"
                f"\n  • Target: '{normalized_target}' → Groups: {target_match.groups() if target_match else 'No match'}"
                f"\n  • Played: '{normalized_played}' → Groups: {played_match.groups() if played_match else 'No match'}"
            )

            if not target_match or not played_match:
                logger.warning(
                    f"⚠️  INVALID NOTE FORMAT"
                    f"\n  • Target: '{normalized_target}' (match: {bool(target_match)})"
                    f"\n  • Played: '{normalized_played}' (match: {bool(played_match)})"
                )
                return False

            # Parse the notes into components
            target_match = NOTE_PATTERN.match(normalized_target)
            played_match = NOTE_PATTERN.match(normalized_played)

            # Validate note formats
            if not target_match:
                logger.warning(
                    f"Invalid target note format: '{target}' (normalized: '{normalized_target}')"
                )
                return False

            if not played_match:
                logger.debug(
                    f"Invalid played note format: '{played}' (normalized: '{normalized_played}')"
                )
                return False

            # Extract note names (ignoring case)
            target_note = (
                target_match.group(1).upper() if target_match.group(1) else None
            )
            played_note = (
                played_match.group(1).upper() if played_match.group(1) else None
            )

            # Extract octaves if they exist
            target_octave = target_match.group(2) or None
            played_octave = played_match.group(2) or None

            # Log the parsed components
            logger.debug(
                f"📝 PARSED COMPONENTS"
                f"\n  • Target: '{target_note}' (octave: {target_octave or 'None'}) - "
                f"Full match: '{target_match.group(0)}'"
                f"\n  • Played: '{played_note}' (octave: {played_octave or 'None'}) - "
                f"Full match: '{played_match.group(0)}'"
            )

            # Validate we have valid note names
            if not target_note or not played_note:
                logger.error(
                    f"❌ INVALID NOTE NAMES"
                    f"\n  • Target note: '{target_note}' (from '{target}')"
                    f"\n  • Played note: '{played_note}' (from '{played}')"
                )
                return False

            logger.debug(
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

            # Log the enharmonic map for debugging
            logger.debug(f"♭♯ ENHARMONIC MAP: {enharmonic_map}")

            # Check for direct match
            direct_match = target_note == played_note
            if direct_match:
                logger.info(f"✅ DIRECT MATCH: '{target_note}' == '{played_note}'")
                return True
            else:
                logger.debug(f"🔍 No direct match: '{played_note}' != '{target_note}'")

            # Check for enharmonic equivalents
            target_in_map = target_note in enharmonic_map
            played_in_map = played_note in enharmonic_map

            logger.debug(
                f"🔍 ENHARMONIC CHECK"
                f"\n  • Target '{target_note}' in map: {target_in_map}"
                f"\n  • Played '{played_note}' in map: {played_in_map}"
            )

            # Case 1: Target note has an enharmonic equivalent that matches played note
            if target_in_map and enharmonic_map[target_note] == played_note:
                logger.info(
                    f"✅ ENHARMONIC MATCH: '{target_note}' matches '{played_note}' "
                    f"via '{enharmonic_map[target_note]}'"
                )
                return True

            # Case 2: Played note has an enharmonic equivalent that matches target note
            if played_in_map and enharmonic_map[played_note] == target_note:
                logger.info(
                    f"✅ ENHARMONIC MATCH: '{played_note}' matches '{target_note}' "
                    f"via '{enharmonic_map[played_note]}'"
                )
                return True

            # If we get here, no match was found
            logger.debug(
                f"❌ NO MATCH FOUND"
                f"\n  • Target: '{target_note}' (from '{target}')"
                f"\n  • Played: '{played_note}' (from '{played}')"
                f"\n  • Enharmonic check failed - No equivalent found in map"
            )

            # Log all possible enharmonic matches for debugging
            if target_in_map:
                logger.debug(
                    f"  • Target '{target_note}' maps to: '{enharmonic_map[target_note]}'"
                )
            if played_in_map:
                logger.debug(
                    f"  • Played '{played_note}' maps to: '{enharmonic_map[played_note]}'"
                )

            return False

        except Exception as e:
            logger.error(
                f"❌ ERROR in NoteMatcher.match: {e}\n"
                f"Target: '{target}' (type: {type(target)}), "
                f"Played: '{played}' (type: {type(played)})\n"
                f"{str(e)}\n"
                f"{type(e).__name__}: {e}"
            )
            return False
