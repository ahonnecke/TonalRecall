#!/usr/bin/env python3

import time
import click
from tonal_recall.ui import NoteGameUI, CursesUI, PygameUI
import pyfiglet
from tonal_recall.note_game_core import NoteGame
from tonal_recall.stats import update_stats, load_stats


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
        # --- Current Note Display ---
        # Try to get the latest stable note (if available)
        current_note = getattr(game, "detector", None)
        stable_note = None
        if current_note and hasattr(current_note, "stable_note"):
            stable_note = current_note.stable_note
        # Fallback to game.current_note if set
        if stable_note is not None:
            note_display = (
                f"Current Note: {stable_note.name} ({stable_note.frequency:.1f} Hz)"
            )
        elif getattr(game, "current_note", None):
            note_display = f"Current Note: {game.current_note}"
        else:
            note_display = "No note detected"
        note_surface = self.note_font.render(note_display, True, (255, 220, 100))
        note_rect = note_surface.get_rect(center=(self.width // 2, self.height - 120))
        self.screen.blit(note_surface, note_rect)
        # Last detected note at the bottom
        if getattr(game, "current_note", None):
            played_str = f"You played: {game.current_note}"
            played_surface = self.note_font.render(played_str, True, (180, 255, 180))
            played_rect = played_surface.get_rect(
                center=(self.width // 2, self.height - 60)
            )
            self.screen.blit(played_surface, played_rect)
        pygame.display.flip()

    def show_stats(self, game, persistent_stats=None):
        # Display stats in the pygame window until the user closes it
        if not self.initialized or not self.screen:
            return
        self.screen.fill((20, 20, 20))
        stats = game.stats
        lines = [
            f"Notes completed: {stats['correct_notes']}",
        ]
        avg_time = None
        min_time = None
        max_time = None
        if stats["times"]:
            avg_time = sum(stats["times"]) / len(stats["times"])
            min_time = min(stats["times"]) if stats["times"] else 0
            max_time = max(stats["times"]) if stats["times"] else 0
            lines.append(f"Average time per note: {avg_time:.2f} seconds")
            lines.append(f"Fastest note: {min_time:.2f} seconds")
            lines.append(f"Slowest note: {max_time:.2f} seconds")
        # Add persistent stats
        if persistent_stats:
            lines.append("--- All-Time Stats ---")
            lines.append(
                f"High Score (Notes/sec): {persistent_stats.get('high_score_nps', 0):.2f}"
            )
            fastest = persistent_stats.get("fastest_note")
            if fastest is not None:
                lines.append(f"Fastest Note Ever: {fastest:.2f} seconds")
            if persistent_stats.get("history"):
                lines.append("Recent Sessions:")
                for entry in persistent_stats["history"][-5:]:
                    lines.append(
                        f"  NPS: {entry['nps']:.2f}, Fastest: {entry['fastest']:.2f} s"
                    )
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
        # --- Current Note Display ---
        current_note = getattr(game, "detector", None)
        stable_note = None
        if current_note and hasattr(current_note, "stable_note"):
            stable_note = current_note.stable_note
        # Fallback to game.current_note if set
        if stable_note is not None:
            note_display = (
                f"Current Note: {stable_note.name} ({stable_note.frequency:.1f} Hz)"
            )
        elif getattr(game, "current_note", None):
            note_display = f"Current Note: {game.current_note}"
        else:
            note_display = "No note detected"
        # Place above the last played note
        try:
            screen.addstr(height - 6, 0, note_display)
        except Exception:
            pass
        if game.current_note:
            screen.addstr(height - 4, 0, f"You played: {game.current_note}")
        screen.addstr(
            height - 2,
            0,
            f"Correct: {game.stats['correct_notes']} / {game.stats['total_notes']}",
        )
        screen.refresh()

    def show_stats(self, game, persistent_stats=None):
        self.cleanup()
        print(f"Notes completed: {game.stats['correct_notes']}")
        if game.stats["times"]:
            avg_time = sum(game.stats["times"]) / len(game.stats["times"]) if game.stats["times"] else None
            min_time = min(game.stats["times"]) if game.stats["times"] else None
            max_time = max(game.stats["times"]) if game.stats["times"] else None
            print(f"Average time per note: {avg_time:.2f} seconds" if avg_time is not None else "Average time per note: N/A")
            print(f"Fastest note: {min_time:.2f} seconds" if min_time is not None else "Fastest note: N/A")
            print(f"Slowest note: {max_time:.2f} seconds" if max_time is not None else "Slowest note: N/A")
        print("\nThank you for playing!")

    def cleanup(self):
        if self.screen:
            self.screen.keypad(False)
            curses.nocbreak()
            curses.echo()
            curses.endwin()
            self.screen = None


@click.command()
@click.option(
    "--debug", default=0, type=int, help="Debug level (0=off, 1=basic, 2=audio data)"
)
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
            end_time = time.time()
            played_duration = end_time - start_time
            # Compute stats
            correct_notes = game.stats.get("correct_notes", 0)
            nps = correct_notes / played_duration if played_duration > 0 else 0
            fastest = min(game.stats["times"]) if game.stats["times"] else None
            # Update persistent stats
            persistent_stats, _ = update_stats(nps, fastest)
            # Pass persistent stats to UI
            if hasattr(game.ui, "show_stats"):
                game.ui.show_stats(game, persistent_stats)
            game.ui.cleanup()
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
