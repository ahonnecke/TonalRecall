#!/usr/bin/env python3

import sys
import time
import pygame
from tonal_recall.logger import get_logger
from tonal_recall.logging_config import setup_logging
from tonal_recall.note_game_core import NoteGame
from tonal_recall.ui import PygameUI
from tonal_recall.stats import update_stats

def main():
    # Configure logging
    setup_logging()
    logger = get_logger(__name__)
    
    try:
        # Initialize game
        game = NoteGame()
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
        
        # Set up note detection callback
        def on_note_detected(note, signal_strength):
            if game.running and game.current_target:
                game.note_detected_callback(note, signal_strength)
                ui.update_display(game)
        
        # Run the game
        game.start_game(duration=60)  # 60 second game by default
        ui.run_game_loop(game, 60, on_note_detected)
        
        # Show final stats
        played_duration = 60 - game.time_remaining
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
            
    except Exception as e:
        logger.exception("Error in main game loop")
        raise
    finally:
        if 'ui' in locals():
            ui.cleanup()

if __name__ == "__main__":
    main()
