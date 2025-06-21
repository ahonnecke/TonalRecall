#!/usr/bin/env python3

import time
import argparse
import pygame
from tonal_recall.logger import get_logger
from tonal_recall.logging_config import setup_logging
from tonal_recall.note_game_core import NoteGame
from tonal_recall.ui import PygameUI
from tonal_recall.stats import update_stats


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Tonal Recall - A musical note training game"
    )
    parser.add_argument(
        "--ui",
        type=str,
        default="pygame",
        choices=["pygame", "curses"],
        help="UI to use (default: pygame)",
    )

    # Dynamically set the choices for difficulty from the NoteGame class
    difficulty_levels = sorted(NoteGame.note_sets.keys())
    parser.add_argument(
        "--difficulty",
        type=int,
        default=3,
        choices=difficulty_levels,
        help=f"Game difficulty level from {min(difficulty_levels)} to {max(difficulty_levels)}. (default: 3)",
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Game duration in seconds (default: 60)",
    )
    return parser.parse_args()


def setup_game_and_ui(args):
    # Initialize game with specified difficulty
    game = NoteGame(difficulty=args.difficulty)

    # Initialize UI based on command line argument
    if args.ui.lower() == "curses":
        from tonal_recall.ui import CursesUI

        ui = CursesUI()
    else:
        ui = PygameUI()
        ui.init_screen()

    # Set up game state
    game.ui = ui
    game.stats = {
        "total_notes": 0,
        "correct_notes": 0,
        "times": [],
        "notes_played": {},
    }
    return game, ui


def main():
    # Parse command line arguments
    args = parse_arguments()

    # Configure logging
    setup_logging()
    logger = get_logger(__name__)

    try:
        game, ui = setup_game_and_ui(args)

        # Run the game
        game_duration = args.duration  # Use duration from command line
        if hasattr(game, "test_mode") and game.test_mode:
            game_duration = game.test_duration

        game.start_game(duration=game_duration)
        ui.run_game_loop(game, game_duration)

        # Show final stats
        played_duration = game_duration - game.time_remaining
        correct_notes = game.stats["correct_notes"]
        nps = correct_notes / played_duration if played_duration > 0 else 0
        fastest = min(game.stats["times"]) if game.stats["times"] else None

        # Update and show persistent stats
        persistent_stats, _ = update_stats(nps, fastest)
        ui.show_stats(game, persistent_stats)

        # Keep the window open until closed
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
            time.sleep(0.1)

    except Exception:
        logger.exception("Error in main game loop")
        raise
    finally:
        if "ui" in locals():
            ui.cleanup()


if __name__ == "__main__":
    main()
