import curses
import pyfiglet
import pygame
import time
from .logger import get_logger

# Get logger for this module
logger = get_logger(__name__)


class NoteGameUI:
    def update_display(self, game):
        raise NotImplementedError

    def show_stats(self, game):
        raise NotImplementedError

    def cleanup(self):
        pass


class CursesUI(NoteGameUI):
    def __init__(self):
        self.screen = None
        ui_logger.debug("Initializing CursesUI")

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
            played_note = game.current_note

            # Handle both string and DetectedNote objects
            if hasattr(played_note, "name"):  # It's a DetectedNote
                position_info = (
                    f" at {played_note.position}"
                    if hasattr(played_note, "position") and played_note.position
                    else ""
                )
                note_name = played_note.name
            else:  # It's a string
                position_info = ""
                note_name = str(played_note)

            screen.addstr(height - 5, 0, f"You played: {note_name}{position_info}")

        # Display stats
        screen.addstr(
            height - 3,
            0,
            f"Correct: {game.stats['correct_notes']} / {game.stats['total_notes']}",
        )
        screen.refresh()

    def show_stats(self, game, persistent_stats=None):
        self.cleanup()
        print(f"Notes completed: {game.stats['correct_notes']}")
        if game.stats["times"]:
            # Ensure times are in seconds and not None
            valid_times = [t for t in game.stats["times"] if t is not None]
            if valid_times:
                avg_time = sum(valid_times) / len(valid_times)
                min_time = min(valid_times)
                max_time = max(valid_times)
                print(f"Average time per note: {avg_time:.2f} seconds")
                print(f"Fastest note: {min_time:.2f} seconds")
                print(f"Slowest note: {max_time:.2f} seconds")
            else:
                print("Average time per note: N/A")
                print("Fastest note: N/A")
                print("Slowest note: N/A")
        if persistent_stats:
            print("\n--- All-Time Stats ---")
            high_score = persistent_stats.get("high_score_nps")
            print(
                f"High Score (Notes/sec): {high_score:.2f}"
                if high_score is not None
                else "High Score (Notes/sec): N/A"
            )
            fastest = persistent_stats.get("fastest_note")
            print(
                f"Fastest Note Ever: {fastest:.2f} seconds"
                if fastest is not None
                else "Fastest Note Ever: N/A"
            )
            if persistent_stats.get("history"):
                print("Recent Sessions:")
                for entry in persistent_stats["history"][-5:]:
                    nps = entry.get("nps")
                    fastest_val = entry.get("fastest")
                    nps_str = f"{nps:.2f}" if nps is not None else "N/A"
                    fastest_str = (
                        f"{fastest_val:.2f}" if fastest_val is not None else "N/A"
                    )
                    print(f"  NPS: {nps_str}, Fastest: {fastest_str} s")
        print("\nThank you for playing!")

    def cleanup(self):
        if self.screen:
            self.screen.keypad(False)
            curses.nocbreak()
            curses.echo()
            curses.endwin()
            self.screen = None


class PygameUI:
    """Pygame-based UI for Tonal Recall"""

    def __init__(self):
        """Initialize the Pygame UI"""
        self.screen = None
        self.width = 1024
        self.height = 768
        self.bg_color = (20, 20, 30)
        self.text_color = (255, 255, 0)
        self.secondary_color = (180, 255, 180)
        self.initialized = False
        self.clock = None

        # Fonts
        self.title_font = None
        self.large_font = None
        self.medium_font = None
        self.small_font = None
        self.start = time.perf_counter()

        logger.debug("Initializing PygameUI")

    def init_screen(self):
        """Initialize the Pygame screen and resources"""
        try:
            pygame.init()
            self.screen = pygame.display.set_mode((self.width, self.height))
            pygame.display.set_caption("Tonal Recall")

            # Initialize fonts
            self.title_font = pygame.font.SysFont("Arial", 48, bold=True)
            self.large_font = pygame.font.SysFont("Arial", 120, bold=True)
            self.medium_font = pygame.font.SysFont("Arial", 36)
            self.small_font = pygame.font.SysFont("Arial", 24)

            self.clock = pygame.time.Clock()
            self.initialized = True
            logger.info("Pygame UI initialized successfully")
            return self.screen

        except Exception as e:
            logger.error(f"Failed to initialize Pygame: {e}")
            self.cleanup()
            raise

    def update_display(self, game):
        """Update the game display

        Args:
            game: The game instance
        """
        if not self.initialized or not self.screen:
            return

        # Clear screen
        self.screen.fill(self.bg_color)

        # Draw header with time remaining
        header_rect = pygame.Rect(0, 0, self.width, 80)
        pygame.draw.rect(self.screen, (30, 30, 40), header_rect)

        # Draw time remaining
        time_str = f"Time: {int(game.time_remaining)}s"
        time_surface = self.medium_font.render(time_str, True, (200, 200, 255))
        time_rect = time_surface.get_rect(midtop=(self.width // 2, 20))
        self.screen.blit(time_surface, time_rect)

        # Draw score
        score_str = f"Score: {game.stats['correct_notes']}"
        score_surface = self.medium_font.render(score_str, True, (200, 255, 200))
        score_rect = score_surface.get_rect(topright=(self.width - 20, 20))
        self.screen.blit(score_surface, score_rect)

        # Draw target note
        if game.current_target:
            # Large target note in the center
            target_surface = self.large_font.render(
                str(game.current_target), True, self.text_color
            )
            target_rect = target_surface.get_rect(
                center=(self.width // 2, self.height // 2 - 40)
            )
            self.screen.blit(target_surface, target_rect)

            # Draw "Play this note" text above
            prompt_surface = self.medium_font.render(
                "Play this note:", True, (200, 200, 255)
            )
            prompt_rect = prompt_surface.get_rect(
                midbottom=(self.width // 2, self.height // 2 - 80)
            )
            self.screen.blit(prompt_surface, prompt_rect)

        # Draw current note being played
        if hasattr(game, "current_note") and game.current_note:
            played_note = game.current_note

            # Handle both string and DetectedNote objects
            if hasattr(played_note, "name"):  # It's a DetectedNote
                position_info = (
                    f" at {played_note.position}"
                    if hasattr(played_note, "position") and played_note.position
                    else ""
                )
                note_name = played_note.name
            else:  # It's a string
                position_info = ""
                note_name = str(played_note)

            note_str = f"You played: {note_name}{position_info}"
            note_surface = self.medium_font.render(note_str, True, self.secondary_color)
            note_rect = note_surface.get_rect(
                midtop=(self.width // 2, self.height // 2 + 80)
            )
            self.screen.blit(note_surface, note_rect)

        # Draw debug info
        if hasattr(game, "detector") and hasattr(game.detector, "current_note"):
            debug_y = self.height - 80
            debug_text = f"Detected: {game.detector.current_note.name if game.detector.current_note else 'None'}"
            debug_surface = self.small_font.render(debug_text, True, (150, 150, 150))
            self.screen.blit(debug_surface, (20, debug_y))

        pygame.display.flip()

    def show_stats(self, game, persistent_stats=None):
        """Display game statistics

        Args:
            game: The game instance
            persistent_stats: Optional persistent statistics
        """
        if not self.initialized or not self.screen:
            return

        # Dark background
        self.screen.fill((10, 10, 15))

        # Title
        title_surface = self.title_font.render(
            "Game Over - Statistics", True, (255, 255, 255)
        )
        title_rect = title_surface.get_rect(center=(self.width // 2, 60))
        self.screen.blit(title_surface, title_rect)

        # Game stats
        stats = game.stats
        y_pos = 150
        line_height = 50

        # Session stats
        stats_lines = [
            f"Notes Matched: {stats['correct_notes']}",
        ]

        # Add timing stats if available
        if stats["times"]:
            # Ensure times are in seconds and not None
            valid_times = [t for t in stats["times"] if t is not None]
            if valid_times:
                avg_time = sum(valid_times) / len(valid_times)
                min_time = min(valid_times)
                max_time = max(valid_times)

                # Format times to show at most 2 decimal places
                stats_lines.extend(
                    [
                        f"Average Time: {avg_time:.2f}s",
                        f"Fastest Match: {min_time:.2f}s",
                        f"Slowest Match: {max_time:.2f}s",
                    ]
                )

        # Add persistent stats if available
        if persistent_stats:
            stats_lines.append("")
            stats_lines.append("--- All-Time Stats ---")

            if "high_score_nps" in persistent_stats:
                stats_lines.append(
                    f"Best Score: {persistent_stats['high_score_nps']:.2f} notes/sec"
                )
            if "fastest_note" in persistent_stats:
                stats_lines.append(
                    f"Fastest Note: {persistent_stats['fastest_note']:.2f}s"
                )
            if "history" in persistent_stats:
                stats_lines.append(f"Games Played: {len(persistent_stats['history'])}")

        # Render all stats lines
        for i, line in enumerate(stats_lines):
            if not line.strip():
                y_pos += 20  # Extra space for section breaks
                continue

            line_surface = self.medium_font.render(line, True, (255, 255, 255))
            line_rect = line_surface.get_rect(
                midleft=(self.width // 3, y_pos + i * line_height)
            )
            self.screen.blit(line_surface, line_rect)

        # Instructions to exit
        exit_text = "Click the window close button to exit"
        exit_surface = self.small_font.render(exit_text, True, (200, 200, 200))
        exit_rect = exit_surface.get_rect(center=(self.width // 2, self.height - 50))
        self.screen.blit(exit_surface, exit_rect)

        pygame.display.flip()

        # Wait for window close
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
            self.clock.tick(30)

    def run_game_loop(self, game, duration_secs, note_callback):
        """Run the main game loop

        Args:
            game: The game instance
            duration_secs: Duration of the game in seconds
            note_callback: Callback function for note detection
        """
        if not self.initialized or not self.screen:
            logger.error("Cannot run game loop: UI not initialized")
            return

        logger.info("Starting game loop")

        # Calculate end time
        start_time = time.time()
        end_time = start_time + duration_secs

        # Main game loop
        running = True
        while running and time.time() < end_time and game.running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    game.running = False
                    break

            # Update game state
            current_time = time.time()
            game.time_remaining = max(0, end_time - current_time)

            # Check for note detection
            if hasattr(game, "detector") and game.detector:
                note = game.detector.get_current_note()
                if note and hasattr(note, "is_stable") and note.is_stable:
                    note_callback(note, 1.0)  # signal_strength not currently used

            # Calculate and display notes per second
            if hasattr(game, "game_start_time") and game.stats["correct_notes"] > 0:
                game_duration = time.time() - game.game_start_time
                if game_duration > 0:
                    game.stats["notes_per_second"] = (
                        game.stats["correct_notes"] / game_duration
                    )

            # Update display
            self.update_display(game)

            # Cap the frame rate
            self.clock.tick(30)

        # Game over
        logger.info("Game loop ended")
        game.running = False

        # Stop the detector if it's running
        if hasattr(game, "detector") and game.detector:
            game.detector.stop()

    def cleanup(self):
        """Clean up Pygame resources"""
        if self.initialized:
            logger.debug("Cleaning up Pygame resources")
            try:
                pygame.quit()
            except Exception as e:
                logger.error(f"Error during Pygame cleanup: {e}")
            self.initialized = False
