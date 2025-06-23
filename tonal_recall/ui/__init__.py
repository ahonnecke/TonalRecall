import pygame
import time
from ..logger import get_logger
from ..stats import load_stats, save_stats

# Get logger for this module
logger = get_logger(__name__)


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
        self.button_color = (0, 122, 255)
        self.button_text_color = (255, 255, 255)
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
            self.title_font = pygame.font.SysFont("Arial", 72, bold=True)
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

    def draw_button(self, text, x, y, width, height):
        """Draw a button and return True if the mouse is hovering over it."""
        mouse = pygame.mouse.get_pos()
        is_hovering = x + width > mouse[0] > x and y + height > mouse[1] > y

        if is_hovering:
            pygame.draw.rect(self.screen, self.button_color, (x, y, width, height))
        else:
            pygame.draw.rect(self.screen, self.button_color, (x, y, width, height), 2)

        text_surf = self.medium_font.render(text, True, self.button_text_color)
        text_rect = text_surf.get_rect(center=((x + (width / 2)), (y + (height / 2))))
        self.screen.blit(text_surf, text_rect)
        return is_hovering

    def show_start_screen(self):
        """Display the start screen and wait for user action."""
        all_time_stats = load_stats()

        while True:
            self.screen.fill(self.bg_color)

            # Title
            title_surf = self.title_font.render(
                "Tonal Recall", True, self.text_color
            )
            title_rect = title_surf.get_rect(center=(self.width / 2, 100))
            self.screen.blit(title_surf, title_rect)

            # All-Time Stats
            stats_title_surf = self.medium_font.render(
                "All-Time Stats", True, self.secondary_color
            )
            stats_title_rect = stats_title_surf.get_rect(center=(self.width / 2, 250))
            self.screen.blit(stats_title_surf, stats_title_rect)

            if all_time_stats:
                high_score = all_time_stats.get("high_score_nps", 0)
                fastest_note = all_time_stats.get("fastest_note", "N/A")
                games_played = len(all_time_stats.get("history", []))

                hs_text = f"Best Score: {high_score:.2f} notes/sec"
                fn_text = (
                    f"Fastest Note: {fastest_note:.2f}s"
                    if isinstance(fastest_note, float)
                    else "Fastest Note: N/A"
                )
                gp_text = f"Games Played: {games_played}"

                hs_surf = self.small_font.render(hs_text, True, (255, 255, 255))
                fn_surf = self.small_font.render(fn_text, True, (255, 255, 255))
                gp_surf = self.small_font.render(gp_text, True, (255, 255, 255))

                self.screen.blit(hs_surf, (self.width / 2 - 150, 320))
                self.screen.blit(fn_surf, (self.width / 2 - 150, 360))
                self.screen.blit(gp_surf, (self.width / 2 - 150, 400))

            # Start Button
            start_button_hover = self.draw_button(
                "Start Game", self.width / 2 - 100, 500, 200, 50
            )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False  # Quit
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if start_button_hover:
                        return True  # Start

            pygame.display.flip()
            self.clock.tick(30)

    def show_countdown(self):
        """Display a 3-second countdown before the game starts."""
        for i in range(3, 0, -1):
            self.screen.fill(self.bg_color)
            count_surf = self.large_font.render(str(i), True, self.text_color)
            count_rect = count_surf.get_rect(
                center=(self.width / 2, self.height / 2)
            )
            self.screen.blit(count_surf, count_rect)
            pygame.display.flip()
            time.sleep(1)

    def show_end_screen(self, game, persistent_stats):
        """Display the end game screen with stats and options."""
        while True:
            self.screen.fill(self.bg_color)
            self.show_stats(game, persistent_stats, is_end_screen=True)

            play_again_hover = self.draw_button(
                "Play Again", self.width / 2 - 220, self.height - 100, 200, 50
            )
            quit_hover = self.draw_button(
                "Quit", self.width / 2 + 20, self.height - 100, 200, 50
            )

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False  # Quit
                if event.type == pygame.MOUSEBUTTONDOWN:
                    if play_again_hover:
                        return True  # Play Again
                    if quit_hover:
                        return False  # Quit

            pygame.display.flip()
            self.clock.tick(30)

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
                note_name = (
                    played_note.note_name
                    if hasattr(played_note, "note_name")
                    else str(played_note)
                )

            # Display detected note
            played_surface = self.medium_font.render(
                f"Detected: {note_name}{position_info}", True, self.secondary_color
            )
            played_rect = played_surface.get_rect(
                center=(self.width // 2, self.height // 2 + 100)
            )
            self.screen.blit(played_surface, played_rect)

        # Draw recent matched notes
        if game.matched_notes:
            y_pos = self.height - 40
            for note, duration in reversed(game.matched_notes[-5:]):
                match_text = f"Matched: {note} in {duration:.2f}s"
                match_surface = self.small_font.render(
                    match_text, True, (200, 255, 200)
                )
                match_rect = match_surface.get_rect(bottomright=(self.width - 20, y_pos))
                self.screen.blit(match_surface, match_rect)
                y_pos -= 30

        pygame.display.flip()

    def show_stats(self, game, persistent_stats=None, is_end_screen=False):
        """Display game statistics

        Args:
            game: The game instance
            persistent_stats: Optional persistent statistics
            is_end_screen: Flag to adjust layout for the end screen
        """
        if not self.initialized or not self.screen:
            return

        if not is_end_screen:
            self.screen.fill(self.bg_color)

        title_surface = self.title_font.render("Game Over", True, (255, 69, 0))
        title_rect = title_surface.get_rect(center=(self.width // 2, 80))
        self.screen.blit(title_surface, title_rect)

        # Stats calculation
        correct_notes = game.stats.get("correct_notes", 0)
        total_time = sum(t for _, t in game.matched_notes)
        avg_time = total_time / correct_notes if correct_notes > 0 else 0
        min_time = min(t for _, t in game.matched_notes) if game.matched_notes else 0
        max_time = max(t for _, t in game.matched_notes) if game.matched_notes else 0
        nps = game.stats.get("notes_per_second", 0)

        # Display stats
        y_pos = 180
        line_height = 40
        stats_lines = [
            f"Final Score: {correct_notes} notes",
            f"Notes per Second: {nps:.2f}",
            "",
        ]

        if correct_notes > 0:
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
                fastest_note_val = persistent_stats['fastest_note']
                fn_text = f"Fastest Note: {fastest_note_val:.2f}s" if isinstance(fastest_note_val, float) else "Fastest Note: N/A"
                stats_lines.append(fn_text)
            if "history" in persistent_stats:
                stats_lines.append(f"Games Played: {len(persistent_stats['history'])}")

        # Render all stats lines
        current_y = y_pos
        for line in stats_lines:
            if not line.strip():
                current_y += 20  # Extra space for section breaks
                continue

            line_surface = self.medium_font.render(line, True, (255, 255, 255))
            line_rect = line_surface.get_rect(midleft=(self.width // 3, current_y))
            self.screen.blit(line_surface, line_rect)
            current_y += line_height

        if not is_end_screen:
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

    def _run_game_loop(self, game, duration_secs):
        """Run the main game loop.

        Args:
            game: The game instance.
            duration_secs: Duration of the game in seconds.
        """
        if not self.initialized or not self.screen:
            logger.error("Cannot run game loop: UI not initialized")
            return

        logger.info("Starting game loop")

        start_time = time.time()
        end_time = start_time + duration_secs

        running = True
        while running and time.time() < end_time and game.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    game.running = False
                    break

            # Process note events from the queue
            game.process_events()

            # Update game state
            current_time = time.time()
            game.time_remaining = max(0, end_time - current_time)

            # Calculate and display notes per second
            if hasattr(game, "game_start_time") and game.stats["correct_notes"] > 0:
                game_duration = time.time() - game.game_start_time
                if game_duration > 0:
                    game.stats["notes_per_second"] = (
                        game.stats["correct_notes"] / game_duration
                    )

            # Update the current note for display
            if hasattr(game, "detector"):
                game.current_note = game.detector.get_current_note()

            self.update_display(game)
            self.clock.tick(30)

        # Game over
        logger.info("Game loop ended")
        game.running = False

        # Stop the detector if it's running
        if hasattr(game, "detector") and game.detector:
            game.detector.stop()

    def run(self, game_factory, duration_secs=60):
        """Run the entire UI flow: start screen, game, end screen.

        Args:
            game_factory: A function that creates a new game instance.
            duration_secs: The duration of each game round in seconds.
        """
        if not self.initialized:
            self.init_screen()

        try:
            while True:
                if not self.show_start_screen():
                    break  # User chose to quit

                # Create a new game instance for each round
                game = game_factory()

                self.show_countdown()

                # Start the game logic (and note detection)
                game.start()

                # Run the main game loop
                self._run_game_loop(game, duration_secs)

                # Stop the note detector
                if game.detector:
                    game.detector.stop()

                # Load persistent stats to show on the end screen
                persistent_stats = save_stats(game.stats)

                if not self.show_end_screen(game, persistent_stats):
                    break  # User chose to quit

        except Exception as e:
            logger.error(f"An error occurred during the UI run loop: {e}", exc_info=True)
        finally:
            self.cleanup()

    def cleanup(self):
        """Clean up Pygame resources"""
        if self.initialized:
            logger.info("Cleaning up Pygame UI")
            pygame.quit()
            self.initialized = False
