#!/usr/bin/env python3

import time
import click
from tonal_recall.ui import NoteGameUI, CursesUI, PygameUI
import pyfiglet
from tonal_recall.note_game_core import NoteGame
from tonal_recall.stats import update_stats, load_stats
import pygame


class NoteGameUI:
    def update_display(self, game):
        raise NotImplementedError

    def show_stats(self, game):
        raise NotImplementedError


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
            # Define the note detector callback
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

            # Run the event/game loop via the UI abstraction
            game.ui.run_game_loop(game, duration_secs, pygame_note_callback)
            # After loop, compute stats and show results
            played_duration = duration_secs - game.time_remaining
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
