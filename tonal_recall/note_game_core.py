import time
import random
import logging
from tonal_recall.note_detector import NoteDetector
import signal
from tonal_recall.ui import CursesUI
from tonal_recall.note_matcher import NoteMatcher

game_core_logger = logging.getLogger("tonal_recall.note_game_core")
game_core_logger.setLevel(logging.INFO)  # Set to DEBUG for verbose output


class NoteGame:
    """A simple game to practice playing notes on a guitar or bass"""

    def __init__(self, debug=False, level=1, note_detector=None):
        """Initialize the game. Optionally inject a note_detector for testability.

        Args:
            debug: Whether to show debug information
            level: The game level (affects possible notes)
            note_detector: Optional, a NoteDetector-like instance for dependency injection/testing
        """
        self.debug = debug
        self.detector = (
            note_detector if note_detector is not None else NoteDetector(debug=debug)
        )
        self.running = False
        self.current_target = None
        self.current_note = None  # Track the current note being played
        self.start_time = 0
        self.time_remaining = 0
        self.ui = None  # UI abstraction
        self.screen = None  # Curses screen
        self.stats = {
            "total_notes": 0,
            "correct_notes": 0,
            "times": [],
            "notes_played": {},
        }
        self.level = level
        # Define notes per level
        self.level_notes = {
            1: ["E", "A", "D", "G"],  # Open strings only
            2: ["A", "B", "C", "D", "E", "F", "G"],  # All basic notes
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
            ],  # Chromatic scale with sharps and flats
            4: None,  # Level 4: specific note on a specific string (implemented below)
            # Level 5: specific note at a specific fret (future)
            # Level 6: ask for enharmonic equivalents (future)
            # Level 7: add timing/tempo constraints (future)
            # Level 8: chord tones or intervals (future)
        }
        # For level 4, we'll define the available notes as (note, string) tuples
        self.guitar_strings = [
            "E",
            "A",
            "D",
            "G",
        ]  # Could be expanded for 6-string or bass
        self.chromatic_notes = [
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
        ]
        if self.level == 4:
            self.available_notes = [
                (note, string)
                for note in self.chromatic_notes
                for string in self.guitar_strings
            ]
        else:
            self.available_notes = self.level_notes.get(
                self.level, ["A", "B", "C", "D", "E", "F", "G"]
            )
        # Set available notes based on level (default to all notes if level not mapped)
        self.available_notes = self.level_notes.get(
            self.level, ["A", "B", "C", "D", "E", "F", "G"]
        )

    def note_detected_callback(self, note, signal_strength):
        """Callback for when a note is detected

        Args:
            note: The detected note
            signal_strength: The strength of the signal
        """
        if not self.running or not self.current_target:
            return
        # For level 4, check both note and string
        if self.level == 4:
            # Level 4: current_target is a (note, string) tuple
            played_note = note.name
            played_string = getattr(note, "string", None)
            key = (played_note, played_string)
            if key in self.stats["notes_played"]:
                self.stats["notes_played"][key] += 1
            else:
                self.stats["notes_played"][key] = 1
            self.current_note = (
                f"{played_note} on {played_string}" if played_string else played_note
            )
            self._needs_update = True
            target_note, target_string = self.current_target
            if played_note == target_note and played_string == target_string:
                elapsed = time.time() - self.start_time
                self.stats["times"].append(elapsed)
                self.stats["correct_notes"] += 1
                self.pick_new_target()
                self._needs_update = True
        else:
            # Other levels: current_target is just a note, any string is valid
            def extract_note_base(note_name):
                # Extracts note letter and accidental (e.g. 'F#') from 'F#1'
                if len(note_name) > 1 and note_name[1] in ["#", "b"]:
                    return note_name[:2]
                return note_name[:1]

            note_base = extract_note_base(note.name)
            target_base = extract_note_base(self.current_target)
            game_core_logger.debug(
                f"[MATCH DEBUG] Target: {self.current_target}, Detected: {note.name}, Note base: {note_base}, Target base: {target_base}"
            )
            if note_base in self.stats["notes_played"]:
                self.stats["notes_played"][note_base] += 1
            else:
                self.stats["notes_played"][note_base] = 1
            self.current_note = note.name
            self._needs_update = True

            if NoteMatcher.match(self.current_target, note.name):
                elapsed = time.time() - self.start_time
                self.stats["times"].append(elapsed)
                self.stats["correct_notes"] += 1
                self.pick_new_target()

    def pick_new_target(self):
        """Pick a new target note (or note+string for level 4)"""
        old_target = self.current_target
        if self.level == 4:
            while self.current_target == old_target:
                self.current_target = random.choice(self.available_notes)
            # current_target is (note, string)
        else:
            while self.current_target == old_target:
                self.current_target = random.choice(self.available_notes)
        self.stats["total_notes"] += 1
        self.start_time = time.time()
        # Only update display immediately if using CursesUI
        if self.ui and isinstance(self.ui, CursesUI):
            self.update_display()

    def update_display(self):
        """Update the game display"""
        # Only safe to call from main thread! (CursesUI: always, PygameUI: only from main loop)
        if self.ui:
            self.ui.update_display(self)

    def start_game(self, duration=60):
        """Start the game

        Args:
            duration: How long to play the game for (in seconds)
        """
        # Initialize curses
        self.ui = CursesUI()
        self.screen = self.ui.init_screen()

        # Set up signal handler to restore terminal on exit
        def cleanup(sig, frame):
            if self.ui:
                self.ui.cleanup()
            exit(0)

        signal.signal(signal.SIGINT, cleanup)

        try:
            # Start the note detector
            if not self.detector.start(callback=self.note_detected_callback):
                self.screen.addstr(0, 0, "Failed to start note detector!")
                self.screen.refresh()
                time.sleep(2)
                self.cleanup_curses()
                return

            self.running = True
            self.stats = {
                "total_notes": 0,
                "correct_notes": 0,
                "times": [],
                "notes_played": {},
            }

            # Countdown
            self.screen.clear()
            self.screen.addstr(0, 0, "Get ready!")
            self.screen.refresh()
            for i in range(3, 0, -1):
                self.screen.addstr(1, 0, f"{i}...")
                self.screen.refresh()
                time.sleep(1)
            self.screen.addstr(1, 0, "GO!")
            self.screen.refresh()
            time.sleep(1)

            # Clear screen for game
            self.screen.clear()

            # Pick the first target note
            self.time_remaining = duration
            self.pick_new_target()

            # Run for the specified duration
            end_time = time.time() + duration
            last_second = int(duration)

            while time.time() < end_time and self.running:
                # Update time remaining
                self.time_remaining = max(0, end_time - time.time())
                current_second = int(self.time_remaining)

                # Update display if the second changed
                if current_second != last_second:
                    self.update_display()
                    last_second = current_second

                time.sleep(0.05)  # Small sleep to prevent CPU hogging

        except Exception as e:
            import traceback

            tb = traceback.format_exc()
            self.screen.addstr(10, 0, f"Error: {e}")
            self.screen.addstr(11, 0, tb)
            self.screen.refresh()
            time.sleep(4)
        finally:
            self.running = False
            self.detector.stop()
            if self.ui:
                self.ui.show_stats(self)
                self.ui.cleanup()

    def cleanup_curses(self):
        """Clean up curses settings"""
        if self.ui:
            self.ui.cleanup()

    def show_stats(self):
        """Show game statistics"""
        if self.ui:
            self.ui.show_stats(self)
