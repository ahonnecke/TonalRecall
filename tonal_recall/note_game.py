#!/usr/bin/env python3

import time
import random
import click
import curses
import signal
from typing import List, Dict
from note_detector import NoteDetector
import pyfiglet

class NoteGameUI:
    def update_display(self, game):
        raise NotImplementedError
    def show_stats(self, game):
        raise NotImplementedError
    def cleanup(self):
        pass

class PygameUI(NoteGameUI):
    def __init__(self):
        pass
    def init_screen(self):
        print("[PygameUI] Pygame UI not implemented yet. This is a stub.")
        return None
    def update_display(self, game):
        pass
    def show_stats(self, game):
        print("[PygameUI] show_stats called (stub)")
    def cleanup(self):
        pass

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
        screen.addstr(height - 2, 0, f"Correct: {game.stats['correct_notes']} / {game.stats['total_notes']}")
        screen.refresh()
    def show_stats(self, game):
        self.cleanup()
        print("\n===== Game Statistics =====")
        print(f"Notes attempted: {game.stats['total_notes']}")
        print(f"Notes completed: {game.stats['correct_notes']}")
        if game.stats['times']:
            avg_time = sum(game.stats['times']) / len(game.stats['times'])
            min_time = min(game.stats['times']) if game.stats['times'] else 0
            max_time = max(game.stats['times']) if game.stats['times'] else 0
            print(f"Average time per note: {avg_time:.2f} seconds")
            print(f"Fastest note: {min_time:.2f} seconds")
            print(f"Slowest note: {max_time:.2f} seconds")
        print("\nNotes played:")
        for note, count in sorted(game.stats['notes_played'].items()):
            print(f"  {note}: {count} times")
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
            'total_notes': 0,
            'correct_notes': 0,
            'times': [],
            'notes_played': {}
        }
        self.level = level
        # Define notes per level
        self.level_notes = {
            1: ['E', 'A', 'D', 'G'],  # Open strings only
            2: ['A', 'B', 'C', 'D', 'E', 'F', 'G'],  # All basic notes
            3: [
                'A', 'A#', 'Bb',
                'B',
                'C', 'C#', 'Db',
                'D', 'D#', 'Eb',
                'E',
                'F', 'F#', 'Gb',
                'G', 'G#', 'Ab'
            ],  # Chromatic scale with sharps and flats
            4: None,  # Level 4: specific note on a specific string (implemented below)
            # Level 5: specific note at a specific fret (future)
            # Level 6: ask for enharmonic equivalents (future)
            # Level 7: add timing/tempo constraints (future)
            # Level 8: chord tones or intervals (future)
        }
        # For level 4, we'll define the available notes as (note, string) tuples
        self.guitar_strings = ['E', 'A', 'D', 'G']  # Could be expanded for 6-string or bass
        self.chromatic_notes = [
            'A', 'A#', 'Bb', 'B', 'C', 'C#', 'Db', 'D', 'D#', 'Eb', 'E', 'F', 'F#', 'Gb', 'G', 'G#', 'Ab'
        ]
        if self.level == 4:
            self.available_notes = [(note, string) for note in self.chromatic_notes for string in self.guitar_strings]
        else:
            self.available_notes = self.level_notes.get(self.level, ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
        # Set available notes based on level (default to all notes if level not mapped)
        self.available_notes = self.level_notes.get(self.level, ['A', 'B', 'C', 'D', 'E', 'F', 'G'])
    
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
            played_string = getattr(note, 'string', None)
            # Record that this note was played
            key = (played_note, played_string)
            if key in self.stats['notes_played']:
                self.stats['notes_played'][key] += 1
            else:
                self.stats['notes_played'][key] = 1
            self.current_note = f"{played_note} on {played_string}" if played_string else played_note
            self.update_display()
            # Check if this is the target note/string
            target_note, target_string = self.current_target
            if played_note == target_note and played_string == target_string:
                elapsed = time.time() - self.start_time
                self.stats['times'].append(elapsed)
                self.stats['correct_notes'] += 1
                # Show success message
                self.screen.addstr(5, 0, f" Correct! {target_note} on {target_string} detected in {elapsed:.2f} seconds")
                self.screen.refresh()
                time.sleep(1)
                self.screen.addstr(5, 0, " " * 50)
                self.pick_new_target()
        else:
            # Extract just the note letter (A, B, C, etc.)
            simple_note = note.name[0]
            # Record that this note was played
            if simple_note in self.stats['notes_played']:
                self.stats['notes_played'][simple_note] += 1
            else:
                self.stats['notes_played'][simple_note] = 1
            self.current_note = note.name
            self.update_display()
            # Check if this is the target note
            if simple_note == self.current_target:
                elapsed = time.time() - self.start_time
                self.stats['times'].append(elapsed)
                self.stats['correct_notes'] += 1
                # Show success message
                self.screen.addstr(5, 0, f" Correct! {self.current_target} detected in {elapsed:.2f} seconds")
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
        self.stats['total_notes'] += 1
        self.start_time = time.time()
        self.update_display()
    
    def update_display(self):
        """Update the game display"""
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
                'total_notes': 0,
                'correct_notes': 0,
                'times': [],
                'notes_played': {}
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
@click.option('--debug', is_flag=True, help='Show debug information')
@click.option('--duration', '-t', default=60, help='Game duration in seconds')
@click.option('--level', '-l', default=1, help='Game level (1=open strings only)')
@click.option('--ui', type=click.Choice(['curses', 'pygame']), default='curses', help='UI backend to use')
def main(debug, duration, level, ui):
    """Start the note guessing game"""
    import os
    prev_duration = None
    duration_file = os.path.join(os.path.dirname(__file__), '.last_game_duration')
    if os.path.exists(duration_file):
        try:
            with open(duration_file, 'r') as f:
                prev_duration = float(f.read().strip())
        except Exception:
            prev_duration = None
    if prev_duration is not None:
        print(f"Previous game duration: {prev_duration:.2f} seconds")
    game = None
    try:
        game = NoteGame(debug=debug, level=level)
        # Select UI backend
        if ui == 'curses':
            game.ui = CursesUI()
            game.screen = game.ui.init_screen()
        elif ui == 'pygame':
            game.ui = PygameUI()
            game.ui.init_screen()
            print("Pygame UI selected. Stub only. Exiting.")
            return
        start_time = time.time()
        game.start_game(duration=duration)
        end_time = time.time()
        played_duration = end_time - start_time
        # Save this duration for next time
        try:
            with open(duration_file, 'w') as f:
                f.write(str(played_duration))
        except Exception as e:
            print(f"Warning: Could not save last game duration: {e}")
    except Exception as e:
        # Make sure UI is restored if there's an error
        if game is not None and getattr(game, 'ui', None) is not None:
            game.ui.cleanup()
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
