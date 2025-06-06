#!/usr/bin/env python3

import time
import random
import click
import curses
import signal
from typing import List, Dict
from note_detector import NoteDetector

class NoteGame:
    """A simple game to practice playing notes on a guitar or bass"""
    
    def __init__(self, debug=False):
        """Initialize the game
        
        Args:
            debug: Whether to show debug information
        """
        self.debug = debug
        self.detector = NoteDetector(debug=debug)
        self.running = False
        self.current_target = None
        self.current_note = None  # Track the current note being played
        self.start_time = 0
        self.time_remaining = 0
        self.screen = None  # Curses screen
        self.stats = {
            'total_notes': 0,
            'correct_notes': 0,
            'times': [],
            'notes_played': {}
        }
        
        # Available notes to play (just the basic notes, no sharps/flats)
        self.available_notes = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    
    def note_detected_callback(self, note, signal_strength):
        """Callback for when a note is detected
        
        Args:
            note: The detected note
            signal_strength: The strength of the signal
        """
        if not self.running or not self.current_target or not self.screen:
            return
            
        # Extract just the note letter (A, B, C, etc.)
        simple_note = note.name[0]
        
        # Record that this note was played
        if simple_note in self.stats['notes_played']:
            self.stats['notes_played'][simple_note] += 1
        else:
            self.stats['notes_played'][simple_note] = 1
        
        # Update the current note being played
        self.current_note = note.name
        self.update_display()
        
        # Check if this is the target note
        if simple_note == self.current_target:
            elapsed = time.time() - self.start_time
            self.stats['times'].append(elapsed)
            self.stats['correct_notes'] += 1
            
            # Show success message
            self.screen.addstr(5, 0, f"✓ Correct! {self.current_target} detected in {elapsed:.2f} seconds")
            self.screen.refresh()
            time.sleep(1)  # Pause briefly to show the success message
            self.screen.addstr(5, 0, " " * 50)  # Clear the success message
            
            # Pick a new target note
            self.pick_new_target()
    
    def pick_new_target(self):
        """Pick a new target note"""
        # Don't pick the same note twice in a row
        old_target = self.current_target
        while self.current_target == old_target:
            self.current_target = random.choice(self.available_notes)
        
        self.stats['total_notes'] += 1
        self.start_time = time.time()
        self.update_display()
    
    def update_display(self):
        """Update the game display"""
        if not self.screen:
            return
            
        # Clear specific lines
        for i in range(4):
            self.screen.addstr(i, 0, " " * 50)
            
        # Update display with current game state
        self.screen.addstr(0, 0, f"TARGET: {self.current_target or '---'}")
        self.screen.addstr(1, 0, f"PLAYING: {self.current_note or '---'}")
        self.screen.addstr(2, 0, f"TIME REMAINING: {int(self.time_remaining)} seconds")
        self.screen.addstr(3, 0, "----------------------------------------")
        self.screen.refresh()
    
    def start_game(self, duration=60):
        """Start the game
        
        Args:
            duration: How long to play the game for (in seconds)
        """
        # Initialize curses
        self.screen = curses.initscr()
        curses.noecho()  # Don't echo keypresses
        curses.cbreak()  # React to keys instantly
        self.screen.keypad(True)  # Enable special keys
        self.screen.clear()
        
        # Set up signal handler to restore terminal on exit
        def cleanup(sig, frame):
            self.cleanup_curses()
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
            self.show_stats()
            self.cleanup_curses()
    
    def cleanup_curses(self):
        """Clean up curses settings"""
        if self.screen:
            self.screen.keypad(False)
            curses.nocbreak()
            curses.echo()
            curses.endwin()
            self.screen = None
    
    def show_stats(self):
        """Show game statistics"""
        # Clean up curses first
        self.cleanup_curses()
        
        print("\n===== Game Statistics =====")
        print(f"Notes attempted: {self.stats['total_notes']}")
        print(f"Notes completed: {self.stats['correct_notes']}")
        
        if self.stats['times']:
            avg_time = sum(self.stats['times']) / len(self.stats['times'])
            min_time = min(self.stats['times']) if self.stats['times'] else 0
            max_time = max(self.stats['times']) if self.stats['times'] else 0
            print(f"Average time per note: {avg_time:.2f} seconds")
            print(f"Fastest note: {min_time:.2f} seconds")
            print(f"Slowest note: {max_time:.2f} seconds")
        
        print("\nNotes played:")
        for note, count in sorted(self.stats['notes_played'].items()):
            print(f"  {note}: {count} times")
        
        print("\nThank you for playing!")

@click.command()
@click.option('--debug', is_flag=True, help='Show debug information')
@click.option('--time', '-t', default=60, help='Game duration in seconds')
def main(debug, time):
    """Start the note guessing game"""
    try:
        game = NoteGame(debug=debug)
        game.start_game(duration=time)
    except Exception as e:
        # Make sure terminal is restored if there's an error
        curses.nocbreak()
        curses.echo()
        curses.endwin()
        print(f"Error: {e}")

if __name__ == '__main__':
    main()
