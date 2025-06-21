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
        self, note_detector: Optional[NoteDetector] = None, difficulty: int = 3
    ) -> None:
        """Initialize the game.

        Args:
            note_detector: Optional, a NoteDetector-like instance for dependency injection/testing
            difficulty: Game difficulty level (0-4). Default is 3 (half notes).
                       0: Single note (test mode)
                       1: Open strings (E, A, D, G)
                       2: Whole notes (A, B, C, D, E, F, G)
                       3: Half notes (chromatic scale with sharps)
                       4: String Master - Whole notes with string specification (e.g., "B, S0")
        """
        # Initialize note detector and matcher
        self.detector = note_detector if note_detector is not None else NoteDetector()
        self.note_matcher = NoteMatcher()

        # Game state
        self.running = False
        self.test_mode = False  # Flag for test mode
        self.test_note = None  # Note to test in test mode
        self.test_duration = 5  # Default test duration in seconds
        self.current_target = None
        self.current_note = None
        self.start_time = 0
        self.last_note_change_time = (
            0  # Track when the current note was first displayed
        )
        self.game_start_time = 0  # Track when the game started
        self.time_remaining = 0
        self.ui = None
        self.event_queue = queue.Queue()
        self.stats = {
            "total_notes": 0,
            "correct_notes": 0,
            "times": [],
            "notes_played": {},
            "notes_per_second": 0.0,
            "high_score_nps": 0.0,
        }
        self.TEST_NOTE = "F"

        # Set difficulty level
        self.difficulty = max(0, min(max(self.note_sets.keys()), int(difficulty)))

        self.note_sets[0] = [self.TEST_NOTE]

        # Set available notes based on difficulty
        self.available_notes = self.note_sets.get(
            self.difficulty, self.note_sets[3]
        )  # Default to level 3

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

    def process_events(self) -> None:
        """Process note events from the queue. Should be called from the main game loop."""
        try:
            # Process all available notes in the queue without blocking
            while not self.event_queue.empty():
                note = self.event_queue.get_nowait()
                self._handle_stable_note(note)
        except queue.Empty:
            # This is expected if the queue is empty, no-op.
            pass

    def _handle_stable_note(self, note: DetectedNote) -> None:
        """Handles the game logic for a stable note dequeued by process_events."""
        if not self.running:
            return

        played_note_full = note.note_name

        # Parse the note name to get the note class (e.g., 'G#' from 'G#4')
        note_match = NOTE_PATTERN.match(played_note_full)
        if not note_match:
            logger.warning(f"Could not parse played note: {played_note_full}")
            return

        played_note_class = note_match.group(1).upper()

        # Update stats for notes played, keyed by the note class
        self.stats["notes_played"][played_note_class] = (
            self.stats["notes_played"].get(played_note_class, 0) + 1
        )

        logger.info("STABLE NOTE DETECTED: %s", played_note_full)

        # Use NoteMatcher to check if the played note matches the target
        target_note = self.current_target
        match_result = self.note_matcher.match(
            target_note, played_note_full, match_octave=self.difficulty >= 4
        )

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
            logger.info(
                "NOTE MATCHED! '%s' matches target '%s' in %.2f seconds",
                played_note_full,
                target_note,
                elapsed,
            )
            self.pick_new_target()

    def pick_new_target(self) -> str:
        """Pick a new target note from available notes based on current difficulty"""
        old_target = self.current_target
        if self.test_mode and self.test_note:
            # In test mode, always return the test note
            self.current_target = self.test_note
        else:
            # In normal mode, pick a random note from available notes for current difficulty
            self.current_target = random.choice(self.available_notes)

        # Update the timestamp when the target note changes
        self.last_note_change_time = time.time()

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

        # Don't stop the detector here - let the UI handle cleanup
        # to avoid threading issues

    def update_display(self) -> None:
        """Update the game display"""
        # Only safe to call from main thread! (CursesUI: always, PygameUI: only from main loop)
        if self.ui:
            self.ui.update_display(self, show_octave=(self.difficulty >= 4))

    def start_game(self, duration=60):
        """Start the game with the specified duration in seconds"""
        if not hasattr(self, "ui") or self.ui is None:
            raise ValueError(
                "UI instance not set. Please set game.ui before starting the game."
            )

        try:
            # Initialize the UI
            if hasattr(self.ui, "init_screen"):
                self.screen = self.ui.init_screen()

            # Initialize game state
            self.running = True
            self.game_start_time = time.time()
            self.start_time = time.time()
            self.time_remaining = duration
            self.stats = {
                "total_notes": 0,
                "correct_notes": 0,
                "times": [],
                "notes_played": {},
            }

            # Start with a target note and set the initial time
            self.pick_new_target()
            self.last_note_change_time = time.time()  # Initialize the note change time

            # Update the display with initial state
            self.update_display()

            # Start the note detector
            if not self.detector.start(callback=self.note_detected_callback):
                raise RuntimeError("Failed to start note detector")

            if self.test_note:
                logger.info(f"Test started: {self.test_note} for {duration} seconds")

            if duration:
                logger.info(f"Game started with {duration} second duration")

        except Exception as e:
            logger.error(f"Error starting game: {e}")
            self.stop_game()
            raise

    def show_stats(self) -> None:
        """Show game statistics"""
        if self.ui:
            self.ui.show_stats(self)
