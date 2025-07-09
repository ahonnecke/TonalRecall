import time
import random
import queue
from typing import Optional
from .note_detector import NoteDetector
from .note_matcher import NOTE_PATTERN, NoteMatcher
from .logger import get_logger
from .note_types import DetectedNote

# Get logger for this module
logger = get_logger(__name__)


class NoteGame:
    note_sets = {
        0: ["F"],  # Single note (test mode), will be replaced by TEST_NOTE in __init__
        1: ["E", "A", "D", "G"],  # Open strings
        2: ["A", "B", "C", "D", "E", "F", "G"],  # Whole notes
        3: [
            "A",
            "A#",
            "B",
            "C",
            "C#",
            "D",
            "D#",
            "E",
            "F",
            "F#",
            "G",
            "G#",
        ],
        4: [
            "E1",
            "F1",
            "F#1",
            "G1",
            "G#1",
            "A2",
            "A#2",
            "B2",
            "C2",
            "C#2",
            "D2",
            "D#2",
            "E2",
            "B1",
            "C2",
        ],
        5: [
            "E1",
            "F1",
            "F#1",
            "G1",
            "G#1",
            "A2",
            "A#2",
            "B2",
            "C2",
            "C#2",
            "D2",
            "D#2",
            "E2",
            "F2",
            "F#2",
            "G2",
            "G#2",
            "A3",
            "A#3",
            "B3",
            "C3",
            "C#3",
            "D3",
            "D#3",
            "E3",
            "F3",
            "F#3",
            "G3",
            "G#3",
        ],
        6: [
            "E1",
            "F1",
            "F#1",
            "G1",
            "G#1",
            "A2",
            "A#2",
            "B2",
            "C2",
            "C#2",
            "D2",
            "D#2",
            "E2",
            "F2",
            "F#2",
            "G2",
            "G#2",
            "A3",
            "A#3",
            "B3",
            "C3",
            "C#3",
            "D3",
            "D#3",
            "E3",
            "F3",
            "F#3",
            "G3",
            "G#3",
            "B4",
            "C4",
            "C#4",
            "D4",
            "D#4",
        ],
    }
    """A simple game to practice playing notes on a guitar or bass"""

    def __init__(
        self,
        note_detector: Optional[NoteDetector] = None,
        difficulty: int = 3,
        target_note: Optional[str] = None,
    ) -> None:
        """Initialize the game.

        Args:
            note_detector: An optional NoteDetector-like instance.
            difficulty: Game difficulty level (0-6).
            target_note: A specific note to target for testing. Overrides difficulty.
        """
        # Initialize note detector and matcher
        self.detector = note_detector if note_detector is not None else NoteDetector()
        self.note_matcher = NoteMatcher()

        # Game state
        self.running = False
        self.current_target = None
        self.current_note = None
        self.start_time = 0
        self.last_note_change_time = 0
        self.game_start_time = 0
        self.time_remaining = 0
        self.event_queue = queue.Queue()
        self.matched_notes = []
        self.stats = {
            "total_notes": 0,
            "correct_notes": 0,
            "times": [],
            "notes_played": {},
            "notes_per_second": 0.0,
            "high_score_nps": 0.0,
        }

        # Game settings
        self.difficulty = difficulty
        self.test_mode = bool(target_note)
        self.test_note = target_note
        self.last_match_was_correct = None

        if self.test_mode:
            # In test mode, difficulty is 0 and we only use the target note.
            self.difficulty = 0
            self.available_notes = [self.test_note]
        else:
            # In normal mode, select notes based on difficulty.
            self.available_notes = self.note_sets.get(self.difficulty, [])

        # Fallback if no notes are available for the selected mode/difficulty.
        if not self.available_notes:
            logger.warning(
                f"No notes available for difficulty {self.difficulty}. Defaulting to level 1."
            )
            self.difficulty = 1
            self.available_notes = self.note_sets[1]

        logger.debug(
            "NoteGame initialized with %d available notes", len(self.available_notes)
        )

    def note_detected_callback(
        self, note: DetectedNote, signal_strength: float
    ) -> None:
        """Callback for when a note is detected. Puts stable notes on the event queue."""
        if not self.running:
            return

        # Only queue stable notes for processing
        if getattr(note, "is_stable", False):
            self.event_queue.put(note)

    def process_events(self) -> bool:
        """Process note events from the queue. Returns True if a note was matched."""
        note_matched = False
        try:
            while not self.event_queue.empty():
                note = self.event_queue.get_nowait()
                if self._handle_stable_note(note):
                    note_matched = True
        except queue.Empty:
            pass  # This is expected
        return note_matched

    def _handle_stable_note(self, note: DetectedNote) -> bool:
        """Handles the game logic for a stable note. Returns True if a match occurred."""
        if not self.running:
            return False

        played_note_full = note.note_name
        note_match = NOTE_PATTERN.match(played_note_full)
        if not note_match:
            logger.warning(f"Could not parse played note: {played_note_full}")
            return False

        played_note_class = note_match.group(1).upper()
        self.stats["notes_played"][played_note_class] = (
            self.stats["notes_played"].get(played_note_class, 0) + 1
        )

        logger.info("STABLE NOTE DETECTED: %s", played_note_full)

        target_note = self.current_target
        # In test mode, always match the octave exactly.
        match_octave = self.test_mode or self.difficulty >= 4
        match_result = self.note_matcher.match(
            target_note, played_note_full, match_octave=match_octave
        )
        self.last_match_was_correct = match_result

        logger.debug(
            "Matching - Target: '%s' vs Played: '%s' -> %s",
            target_note,
            played_note_full,
            "MATCH" if match_result else "NO MATCH",
        )

        if match_result:
            elapsed = time.time() - self.last_note_change_time
            self.stats["times"].append(elapsed)
            self.stats["correct_notes"] += 1
            self.matched_notes.append((target_note, elapsed))
            logger.info(
                "NOTE MATCHED! '%s' matches target '%s' in %.2f seconds",
                played_note_full,
                target_note,
                elapsed,
            )
            self.pick_new_target()
            return True

        return False

    def pick_new_target(self) -> str:
        """Pick a new target note from available notes based on current difficulty"""
        old_target = self.current_target

        if self.test_mode and self.test_note:
            self.current_target = self.test_note
        elif not self.available_notes:
            logger.error("No available notes to pick from. Halting game.")
            self.stop_game()
            return None
        else:
            self.current_target = random.choice(self.available_notes)

        # Reset hint state and update timestamp
        self.last_note_change_time = time.time()

        if self.current_target != old_target:
            logger.debug(
                "New target note: %s (was: %s) [Difficulty: %d]",
                self.current_target,
                old_target,
                self.difficulty,
            )
        return self.current_target

    def stop_game(self) -> None:
        """Stop the game and clean up resources"""
        self.running = False
        logger.info(
            "Game stopped. Final score: %d/%d",
            self.stats["correct_notes"],
            self.stats["total_notes"],
        )
        if self.detector and self.detector.is_running():
            self.detector.stop()

    def start(self):
        """Initializes the game state for a new game and starts the detector."""
        if self.running:
            logger.warning("Game is already running. Resetting.")

        self.running = True
        self.game_start_time = time.time()
        self.stats = {
            "total_notes": 0,
            "correct_notes": 0,
            "times": [],
            "notes_played": {},
            "notes_per_second": 0.0,
        }
        self.matched_notes = []
        self.event_queue = queue.Queue()

        self.pick_new_target()
        self.last_note_change_time = time.time()

        if self.detector:
            self.detector.start(callback=self.note_detected_callback)

        logger.info("Game started.")
