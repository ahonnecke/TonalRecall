import time
import random
from .note_detector import NoteDetector
from .note_matcher import NoteMatcher
from .logger import get_logger

# Get logger for this module
logger = get_logger(__name__)


class NoteGame:
    """A simple game to practice playing notes on a guitar or bass"""

    def __init__(self, note_detector=None, difficulty=3):
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
        self.time_remaining = 0
        self.ui = None
        self.stats = {
            "total_notes": 0,
            "correct_notes": 0,
            "times": [],
            "notes_played": {},
        }
        self.TEST_NOTE = "F#"

        # Set difficulty level (0-4)
        self.difficulty = max(0, min(4, int(difficulty)))  # Clamp to 0-4

        # Define note sets for each difficulty level
        self.note_sets = {
            0: [self.TEST_NOTE],  # Single note (test mode)
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
            ],  # Half notes (chromatic)
        }

        # Set available notes based on difficulty
        self.available_notes = self.note_sets.get(
            self.difficulty, self.note_sets[3]
        )  # Default to level 3

        logger.debug(
            "NoteGame initialized with %d available notes", len(self.available_notes)
        )

    def note_detected_callback(self, note, signal_strength):
        """Callback for when a note is detected

        Args:
            note: The detected note object with 'name' attribute
            signal_strength: The strength of the signal (not currently used)
        """
        if not self.running or not self.current_target:
            return

        # Store the entire DetectedNote object
        self.current_note = note
        played_note = note.name

        # Track note in stats
        if played_note in self.stats["notes_played"]:
            self.stats["notes_played"][played_note] += 1
        else:
            self.stats["notes_played"][played_note] = 1

        # Only check for matches on stable notes
        is_stable = getattr(note, "is_stable", False)
        logger.debug("Note detected: %s (stable: %s)", played_note, is_stable)

        if is_stable:
            logger.info("STABLE NOTE DETECTED: %s", played_note)

            # Use NoteMatcher to check if the played note matches the target
            target_note = str(self.current_target)
            match_result = self.note_matcher.match(target_note, played_note)

            logger.debug(
                "Matching - Target: '%s' vs Played: '%s' -> %s",
                target_note,
                played_note,
                "MATCH" if match_result else "NO MATCH",
            )

            if match_result:
                # Record successful match
                elapsed = time.time() - self.start_time
                self.stats["times"].append(elapsed)
                self.stats["correct_notes"] += 1
                logger.info(
                    "NOTE MATCHED! '%s' matches target '%s'", played_note, target_note
                )

                # Get a new target note
                self.pick_new_target()

                # Trigger UI update
                if self.ui:
                    self.ui.update_display(self)

    def set_difficulty(self, level):
        """Set the game difficulty level (0-4).

        Args:
            level: Difficulty level (0-4)
                  0: Single note (test mode)
                  1: Open strings (E, A, D, G)
                  2: Whole notes (A, B, C, D, E, F, G)
                  3: Half notes (chromatic scale with sharps)
                  4: String Master - Whole notes with string specification (e.g., "B, S0")
        """
        self.difficulty = max(0, min(4, int(level)))  # Clamp to 0-4
        self.available_notes = self.note_sets.get(self.difficulty, self.note_sets[3])
        logger.info(f"Difficulty set to level {self.difficulty}")

        # If game is running, pick a new target from the new note set
        if self.running and not self.test_mode:
            self.pick_new_target()

    def pick_new_target(self):
        """Pick a new target note from available notes based on current difficulty"""
        old_target = self.current_target
        if self.test_mode and self.test_note:
            # In test mode, always return the test note
            self.current_target = self.test_note
        else:
            # In normal mode, pick a random note from available notes for current difficulty
            self.current_target = random.choice(self.available_notes)
        logger.debug(
            "New target note: %s (was: %s) [Difficulty: %d]",
            self.current_target,
            old_target,
            self.difficulty,
        )
        return self.current_target

    def start_game(self, duration=60):
        """Start the game with the specified duration in seconds"""
        self.running = True
        self.start_time = time.time()
        self.time_remaining = duration
        self.stats = {
            "total_notes": 0,
            "correct_notes": 0,
            "times": [],
            "notes_played": {},
        }

        # Pick the first target note
        self.pick_new_target()

        logger.info("Game started! Target note: %s", self.current_target)

        # Start the detector if it's not already running
        if not self.detector.is_running():
            self.detector.start()

    def stop_game(self):
        """Stop the game and clean up resources"""
        self.running = False
        logger.info(
            "Game stopped. Final score: %d/%d",
            self.stats["correct_notes"],
            self.stats["total_notes"],
        )

        # Don't stop the detector here - let the UI handle cleanup
        # to avoid threading issues

    def get_stats(self):
        """Return game statistics"""
        return {
            "total_notes": self.stats["total_notes"],
            "correct_notes": self.stats["correct_notes"],
            "accuracy": (
                self.stats["correct_notes"] / self.stats["total_notes"] * 100
                if self.stats["total_notes"] > 0
                else 0
            ),
            "times": self.stats["times"],
            "notes_played": self.stats["notes_played"],
        }

    def update_display(self):
        """Update the game display"""
        # Only safe to call from main thread! (CursesUI: always, PygameUI: only from main loop)
        if self.ui:
            self.ui.update_display(self)

    def start_test_mode(self, test_note, duration=5):
        """Start the game in test mode with a single note

        Args:
            test_note: The single note to test (e.g., 'C', 'G#')
            duration: Test duration in seconds (default: 5)
        """
        self.test_note = test_note
        self.test_duration = duration
        logger.info(f"Starting test mode with note {test_note} for {duration} seconds")
        # Set the duration on the instance so UI can access it
        self.duration = duration
        self.start_game(duration)

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
            self.start_time = time.time()
            self.time_remaining = duration
            self.stats = {
                "total_notes": 0,
                "correct_notes": 0,
                "times": [],
                "notes_played": {},
            }

            # Start with a target note
            self.pick_new_target()

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

    def cleanup_curses(self):
        """Clean up curses settings"""
        if self.ui:
            self.ui.cleanup()

    def show_stats(self):
        """Show game statistics"""
        if self.ui:
            self.ui.show_stats(self)
