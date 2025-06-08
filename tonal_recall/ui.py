import curses
import pyfiglet
import pygame


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
        if persistent_stats:
            print("\n--- All-Time Stats ---")
            high_score = persistent_stats.get('high_score_nps')
            print(f"High Score (Notes/sec): {high_score:.2f}" if high_score is not None else "High Score (Notes/sec): N/A")
            fastest = persistent_stats.get("fastest_note")
            print(f"Fastest Note Ever: {fastest:.2f} seconds" if fastest is not None else "Fastest Note Ever: N/A")
            if persistent_stats.get("history"):
                print("Recent Sessions:")
                for entry in persistent_stats["history"][-5:]:
                    nps = entry.get('nps')
                    fastest_val = entry.get('fastest')
                    nps_str = f"{nps:.2f}" if nps is not None else "N/A"
                    fastest_str = f"{fastest_val:.2f}" if fastest_val is not None else "N/A"
                    print(f"  NPS: {nps_str}, Fastest: {fastest_str} s")
        print("\nThank you for playing!")

    def cleanup(self):
        if self.screen:
            self.screen.keypad(False)
            curses.nocbreak()
            curses.echo()
            curses.endwin()
            self.screen = None


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
        timer_str = f"Time remaining: {int(game.time_remaining)}s"
        timer_surface = self.timer_font.render(timer_str, True, (200, 200, 255))
        timer_rect = timer_surface.get_rect(center=(self.width // 2, 40))
        self.screen.blit(timer_surface, timer_rect)
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
        if getattr(game, "current_note", None):
            played_str = f"You played: {game.current_note}"
            played_surface = self.note_font.render(played_str, True, (180, 255, 180))
            played_rect = played_surface.get_rect(
                center=(self.width // 2, self.height - 60)
            )
            self.screen.blit(played_surface, played_rect)
        pygame.display.flip()

    def show_stats(self, game, persistent_stats=None):
        if not self.initialized or not self.screen:
            return
        self.screen.fill((20, 20, 20))
        stats = game.stats
        lines = [
            "===== Game Statistics =====",
            f"Notes completed: {stats['correct_notes']}",
        ]
        avg_time = None
        min_time = None
        max_time = None
        if stats["times"]:
            avg_time = sum(stats["times"]) / len(stats["times"]) if stats["times"] else None
            min_time = min(stats["times"]) if stats["times"] else None
            max_time = max(stats["times"]) if stats["times"] else None
            lines.append(f"Fastest note: {min_time:.2f} seconds" if min_time is not None else "Fastest note: N/A")
            lines.append(f"Slowest note: {max_time:.2f} seconds" if max_time is not None else "Slowest note: N/A")
        # Add persistent stats
        if persistent_stats:
            lines.append("--- All-Time Stats ---")
            high_score = persistent_stats.get('high_score_nps')
            lines.append(f"High Score (Notes/sec): {high_score:.2f}" if high_score is not None else "High Score (Notes/sec): N/A")
            fastest = persistent_stats.get("fastest_note")
            lines.append(f"Fastest Note Ever: {fastest:.2f} seconds" if fastest is not None else "Fastest Note Ever: N/A")
            if persistent_stats.get("history"):
                lines.append("Recent Sessions:")
                for entry in persistent_stats["history"][-5:]:
                    nps = entry.get('nps')
                    fastest_val = entry.get('fastest')
                    nps_str = f"{nps:.2f}" if nps is not None else "N/A"
                    fastest_str = f"{fastest_val:.2f}" if fastest_val is not None else "N/A"
                    lines.append(f"  NPS: {nps_str}, Fastest: {fastest_str} s")
        lines.append("")
        lines.append("Thank you for playing!")
        # Render lines
        font = pygame.font.SysFont(None, 48)
        y = 40
        for i, line in enumerate(lines):
            surf = font.render(line, True, (255, 255, 255))
            rect = surf.get_rect(center=(self.width // 2, y))
            self.screen.blit(surf, rect)
            y += 50
        # Render average time per note in large, bold font
        big_font = pygame.font.SysFont(None, 96, bold=True)
        if avg_time is not None:
            avg_str = f"Average time per note: {avg_time:.2f} s"
        else:
            avg_str = "Average time per note: N/A"
        avg_surf = big_font.render(avg_str, True, (255, 255, 0))
        avg_rect = avg_surf.get_rect(center=(self.width // 2, self.height // 2))
        self.screen.blit(avg_surf, avg_rect)
        pygame.display.flip()
        # Wait for user to close window
        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
        self.cleanup()

    def run_game_loop(self, game, duration_secs, note_callback):
        """Run the main Pygame event/game loop for the given duration."""
        import time
        game.running = True
        game.time_remaining = duration_secs
        game.pick_new_target()
        self.update_display(game)
        if not game.detector.start(callback=note_callback):
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
                    self.update_display(game)
                    last_second = current_second
                    game._needs_update = False
                pygame.time.wait(50)
        except KeyboardInterrupt:
            pass
        game.running = False
        game.detector.stop()
        # The rest of the stats and cleanup logic remains in main.py

    def cleanup(self):
        if self.initialized:
            pygame.quit()
            self.initialized = False
