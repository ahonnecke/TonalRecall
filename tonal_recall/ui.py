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
        timer_rect = timer_surface.get_rect(center=(self.width//2, 40))
        self.screen.blit(timer_surface, timer_rect)
        note = game.current_target
        if note:
            if game.level == 4 and isinstance(note, tuple):
                note_str = f"{note[0]} on {note[1]}"
            else:
                note_str = str(note)
            text_surface = self.font.render(note_str, True, self.text_color)
            text_rect = text_surface.get_rect(center=(self.width//2, self.height//2))
            self.screen.blit(text_surface, text_rect)
        if getattr(game, 'current_note', None):
            played_str = f"You played: {game.current_note}"
            played_surface = self.note_font.render(played_str, True, (180, 255, 180))
            played_rect = played_surface.get_rect(center=(self.width//2, self.height - 60))
            self.screen.blit(played_surface, played_rect)
        pygame.display.flip()
    def show_stats(self, game):
        if not self.initialized or not self.screen:
            return
        self.screen.fill((20, 20, 20))
        stats = game.stats
        lines = [
            "===== Game Statistics =====",
            f"Notes attempted: {stats['total_notes']}",
            f"Notes completed: {stats['correct_notes']}",
        ]
        if stats['times']:
            avg_time = sum(stats['times']) / len(stats['times'])
            min_time = min(stats['times']) if stats['times'] else 0
            max_time = max(stats['times']) if stats['times'] else 0
            lines.append(f"Average time per note: {avg_time:.2f} seconds")
            lines.append(f"Fastest note: {min_time:.2f} seconds")
            lines.append(f"Slowest note: {max_time:.2f} seconds")
        lines.append("")
        lines.append("Notes played:")
        for note, count in sorted(stats['notes_played'].items()):
            lines.append(f"  {note}: {count} times")
        lines.append("")
        lines.append("Thank you for playing!")
        font = pygame.font.SysFont(None, 48)
        y = 40
        for line in lines:
            surf = font.render(line, True, (255,255,255))
            rect = surf.get_rect(center=(self.width//2, y))
            self.screen.blit(surf, rect)
            y += 50
        pygame.display.flip()
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
