#!/usr/bin/env python3

import time
import random
import click
import signal
from typing import List, Dict
from note_detector import NoteDetector
from ui import NoteGameUI, CursesUI, PygameUI
import pyfiglet


class NoteGameUI:
    def update_display(self, game):
        raise NotImplementedError

    def show_stats(self, game):
        raise NotImplementedError

    def cleanup(self):
        pass


import pygame


class PygameUI(NoteGameUI):
    def __init__(self):
        self.screen = None
        self.width = 800
        self.height = 600
        self.bg_color = (30, 30, 30)
        self.text_color = (255, 255, 0)
        self.font = None
        self.timer_font = None
        self.note_font = None
        self.initialized = False

    def init_screen(self):
        pygame.init()
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Tonal Recall - Pygame UI")
        self.font = pygame.font.SysFont(None, 120)
        self.timer_font = pygame.font.SysFont(None, 48)
        self.note_font = pygame.font.SysFont(None, 64)
        self.initialized = True
        return self.screen

    def update_display(self, game):
        if not self.initialized or not self.screen:
            return
        self.screen.fill(self.bg_color)
        # Timer at the top
        timer_str = f"Time remaining: {int(game.time_remaining)}s"
        timer_surface = self.timer_font.render(timer_str, True, (200, 200, 255))
        timer_rect = timer_surface.get_rect(center=(self.width // 2, 40))
        self.screen.blit(timer_surface, timer_rect)
        # Target note in the center
        note = game.current_target
        if note:
            if game.level == 4 and isinstance(note, tuple):
                note_str = f"{note[0]} on {note[1]}"
            else:
                note_str = str(note)
            text_surface = self.font.render(note_str, True, self.text_color)
            text_rect = text_surface.get_rect(
                center=(self.width // 2, self.height // 2)
            )
            self.screen.blit(text_surface, text_rect)
        # Last detected note at the bottom
        if getattr(game, "current_note", None):
            played_str = f"You played: {game.current_note}"
            played_surface = self.note_font.render(played_str, True, (180, 255, 180))
            played_rect = played_surface.get_rect(
                center=(self.width // 2, self.height - 60)
            )
            self.screen.blit(played_surface, played_rect)
        pygame.display.flip()

    def show_stats(self, game):
        # Display stats in the pygame window until the user closes it
        if not self.initialized or not self.screen:
            return
        self.screen.fill((20, 20, 20))
        stats = game.stats
        lines = [
            "===== Game Statistics =====",
            f"Notes attempted: {stats['total_notes']}",
            f"Notes completed: {stats['correct_notes']}",
        ]
        if stats["times"]:
            avg_time = sum(stats["times"]) / len(stats["times"])
            min_time = min(stats["times"]) if stats["times"] else 0
            max_time = max(stats["times"]) if stats["times"] else 0
            lines.append(f"Average time per note: {avg_time:.2f} seconds")
            lines.append(f"Fastest note: {min_time:.2f} seconds")
            lines.append(f"Slowest note: {max_time:.2f} seconds")
        lines.append("")
        lines.append("")
        lines.append("Thank you for playing!")
        # Render lines
        font = pygame.font.SysFont(None, 48)
        y = 40
        for line in lines:
            surf = font.render(line, True, (255, 255, 255))
            rect = surf.get_rect(center=(self.width // 2, y))
            self.screen.blit(surf, rect)
            y += 50
        pygame.display.flip()
        # Wait for user to close window
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
            pygame.time.wait(100)
        self.cleanup()

    def cleanup(self):
        if self.initialized:
            pygame.quit()
            self.initialized = False


class CursesUI(NoteGameUI):
    def __init__(self):
        self.screen = None

    def init_screen(self):
        self.screen = curses.initscr()
        curses.noecho()
        curses.cbreak()
        self.screen.keypad(True)
        self.screen.clear()
        return self.screen

    def update_display(self, game):
        screen = self.screen
        if not screen:
            return
        screen.clear()
        screen.addstr(0, 0, f"Time remaining: {int(game.time_remaining)}s")
        if game.current_target:
            if game.level == 4 and isinstance(game.current_target, tuple):
                note_str = f"{game.current_target[0]} on {game.current_target[1]}"
            else:
                note_str = str(game.current_target)
            screen.addstr(2, 0, "Play this note:")
            figlet_text = pyfiglet.figlet_format(note_str)
            lines = figlet_text.splitlines()
            height, width = screen.getmaxyx()
            start_y = max(4, (height // 2) - (len(lines) // 2))
            for i, line in enumerate(lines):
                start_x = max(0, (width // 2) - (len(line) // 2))
                if start_y + i < height:
                    try:
                        screen.addstr(start_y + i, start_x, line)
                    except curses.error:
                        pass
        if game.current_note:
            screen.addstr(height - 4, 0, f"You played: {game.current_note}")
        screen.addstr(
            height - 2,
            0,
            f"Correct: {game.stats['correct_notes']} / {game.stats['total_notes']}",
        )
        screen.refresh()

    def show_stats(self, game):
        self.cleanup()
        print("\n===== Game Statistics =====")
        print(f"Notes attempted: {game.stats['total_notes']}")
        print(f"Notes completed: {game.stats['correct_notes']}")
        if game.stats["times"]:
            avg_time = sum(game.stats["times"]) / len(game.stats["times"])
            min_time = min(game.stats["times"]) if game.stats["times"] else 0
            max_time = max(game.stats["times"]) if game.stats["times"] else 0
            print(f"Average time per note: {avg_time:.2f} seconds")
            print(f"Fastest note: {min_time:.2f} seconds")
            print(f"Slowest note: {max_time:.2f} seconds")
        print("\nThank you for playing!")

    def cleanup(self):
        if self.screen:
            self.screen.keypad(False)
            curses.nocbreak()
            curses.echo()
            curses.endwin()
            self.screen = None


class NoteGame:
    """A simple game to practice playing notes on a guitar or bass"""

    def __init__(self, debug=False, level=1):
        """Initialize the game

        Args:
            debug: Whether to show debug information
            level: The game level (affects possible notes)
        """
        self.debug = debug
        self.detector = NoteDetector(debug=debug)
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
                "Bb",
                "B",
                "C",
                "C#",
                "Db",
                "D",
                "D#",
                "Eb",
                "E",
                "F",
                "F#",
                "Gb",
                "G",
                "G#",
                "Ab",
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
            "Bb",
            "B",
            "C",
            "C#",
            "Db",
            "D",
            "D#",
            "Eb",
            "E",
            "F",
            "F#",
            "Gb",
            "G",
            "G#",
            "Ab",
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
        if not self.running or not self.current_target or not self.screen:
            return
        # For level 4, check both note and string
        if self.level == 4:
            # Assume note.name is something like 'A', 'A#', etc.
            # and note.string is the string name (e.g., 'E', 'A', etc.)
            # If note.string is not available, you may need to adapt this logic
            played_note = note.name
            played_string = getattr(note, "string", None)
            # Record that this note was played
            key = (played_note, played_string)
            if key in self.stats["notes_played"]:
                self.stats["notes_played"][key] += 1
            else:
                self.stats["notes_played"][key] = 1
            self.current_note = (
                f"{played_note} on {played_string}" if played_string else played_note
            )
            self.update_display()
            # Check if this is the target note/string
            target_note, target_string = self.current_target
            if played_note == target_note and played_string == target_string:
                elapsed = time.time() - self.start_time
                self.stats["times"].append(elapsed)
                self.stats["correct_notes"] += 1
                # Show success message
                self.screen.addstr(
                    5,
                    0,
                    f" Correct! {target_note} on {target_string} detected in {elapsed:.2f} seconds",
                )
                self.screen.refresh()
                time.sleep(1)
                self.screen.addstr(5, 0, " " * 50)
                self.pick_new_target()
        else:
            # Extract just the note letter (A, B, C, etc.)
            simple_note = note.name[0]
            # Record that this note was played
            if simple_note in self.stats["notes_played"]:
                self.stats["notes_played"][simple_note] += 1
            else:
                self.stats["notes_played"][simple_note] = 1
            self.current_note = note.name
            self.update_display()
            # Check if this is the target note
            if simple_note == self.current_target:
                elapsed = time.time() - self.start_time
                self.stats["times"].append(elapsed)
                self.stats["correct_notes"] += 1
                # Show success message
                self.screen.addstr(
                    5,
                    0,
                    f" Correct! {self.current_target} detected in {elapsed:.2f} seconds",
                )
                self.screen.refresh()
                time.sleep(1)
                self.screen.addstr(5, 0, " " * 50)
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
            self.screen.addstr(10, 0, f"Error: {e}")
            self.screen.refresh()
            time.sleep(2)
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


@click.command()
@click.option("--debug", is_flag=True, help="Show debug information")
@click.option("--duration", "-t", default=60, help="Game duration in seconds")
@click.option("--level", "-l", default=1, help="Game level (1=open strings only)")
@click.option(
    "--ui",
    type=click.Choice(["curses", "pygame"]),
    default="curses",
    help="UI backend to use",
)
def main(debug, duration, level, ui):
    """Start the note guessing game"""
    import os

    prev_duration = None
    duration_file = os.path.join(os.path.dirname(__file__), ".last_game_duration")
    if os.path.exists(duration_file):
        try:
            with open(duration_file, "r") as f:
                prev_duration = float(f.read().strip())
        except Exception:
            prev_duration = None
    if prev_duration is not None:
        print(f"Previous game duration: {prev_duration:.2f} seconds")
    game = None
    try:
        game = NoteGame(debug=debug, level=level)
        # Select UI backend
        if ui == "curses":
            game.ui = CursesUI()
            game.screen = game.ui.init_screen()
            start_time = time.time()
            game.start_game(duration=duration)
            end_time = time.time()
        elif ui == "pygame":
            game.ui = PygameUI()
            game.ui.init_screen()
            duration_secs = duration
            game.stats = {
                "total_notes": 0,
                "correct_notes": 0,
                "times": [],
                "notes_played": {},
            }
            game.running = True
            game.time_remaining = duration_secs
            game.pick_new_target()
            game.ui.update_display(game)

            # Start note detector with callback
            def pygame_note_callback(note, signal_strength):
                # Only update game state, never call pygame UI methods from this thread!
                if not game.running or not game.current_target:
                    return
                needs_update = False
                if game.level == 4:
                    played_note = note.name
                    played_string = getattr(note, "string", None)
                    key = (played_note, played_string)
                    game.stats["notes_played"][key] = (
                        game.stats["notes_played"].get(key, 0) + 1
                    )
                    game.current_note = (
                        f"{played_note} on {played_string}"
                        if played_string
                        else played_note
                    )
                    target_note, target_string = game.current_target
                    if played_note == target_note and played_string == target_string:
                        elapsed = time.time() - game.start_time
                        game.stats["times"].append(elapsed)
                        game.stats["correct_notes"] += 1
                        game.pick_new_target()
                        needs_update = True
                else:
                    simple_note = note.name[0]
                    game.stats["notes_played"][simple_note] = (
                        game.stats["notes_played"].get(simple_note, 0) + 1
                    )
                    game.current_note = note.name
                    if simple_note == game.current_target:
                        elapsed = time.time() - game.start_time
                        game.stats["times"].append(elapsed)
                        game.stats["correct_notes"] += 1
                        game.pick_new_target()
                        needs_update = True
                # Always update display to show last played note
                game._needs_update = True

            if not game.detector.start(callback=pygame_note_callback):
                print("Failed to start note detector!")
                return
            start_time = time.time()
            end_time = start_time + duration_secs
            last_second = int(duration_secs)
            try:
                while time.time() < end_time and game.running:
                    game.time_remaining = max(0, end_time - time.time())
                    current_second = int(game.time_remaining)
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            game.running = False
                            break
                    if (
                        getattr(game, "_needs_update", False)
                        or current_second != last_second
                    ):
                        game.ui.update_display(game)
                        last_second = current_second
                        game._needs_update = False
                    pygame.time.wait(50)
            except KeyboardInterrupt:
                pass
            game.running = False
            game.detector.stop()
            game.ui.show_stats(game)
            game.ui.cleanup()
            end_time = time.time()
        played_duration = end_time - start_time
        # Save this duration for next time
        try:
            with open(duration_file, "w") as f:
                f.write(str(played_duration))
        except Exception as e:
            print(f"Warning: Could not save last game duration: {e}")
    except Exception as e:
        # Make sure UI is restored if there's an error
        if game is not None and getattr(game, "ui", None) is not None:
            game.ui.cleanup()
        print(f"Error: {e}")


if __name__ == "__main__":
    main()
