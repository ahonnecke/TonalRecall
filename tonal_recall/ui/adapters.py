"""Adapters for connecting UI components to the note detection backend."""

from abc import ABC, abstractmethod
from typing import Optional, Callable, Dict, Any, List

from ..logger import get_logger
from ..note_types import DetectedNote
from ..core.interfaces import INoteDetectionService
from ..core.events import NoteDetectionEvents

logger = get_logger(__name__)


class UIAdapter(ABC):
    """Base class for UI adapters that connect to the note detection backend."""
    
    def __init__(self, note_detection_service: INoteDetectionService):
        """Initialize the UI adapter.
        
        Args:
            note_detection_service: Note detection service to use
        """
        self._note_detection_service = note_detection_service
        self._events = NoteDetectionEvents()
        self._running = False
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the UI.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    def update(self, delta_time: float) -> bool:
        """Update the UI.
        
        Args:
            delta_time: Time elapsed since last update in seconds
            
        Returns:
            True to continue running, False to exit
        """
        pass
    
    @abstractmethod
    def render(self) -> None:
        """Render the UI."""
        pass
    
    @abstractmethod
    def cleanup(self) -> None:
        """Clean up resources used by the UI."""
        pass
    
    def start(self) -> bool:
        """Start the UI and note detection.
        
        Returns:
            True if started successfully, False otherwise
        """
        if self._running:
            logger.warning("UI already running")
            return True
        
        # Initialize the UI
        if not self.initialize():
            logger.error("Failed to initialize UI")
            return False
        
        # Set up event handlers
        self._setup_event_handlers()
        
        # Start note detection
        try:
            self._note_detection_service.start(self._on_note_detected)
            self._running = True
            logger.info("UI started")
            return True
        except Exception as e:
            logger.error(f"Failed to start note detection: {e}")
            self.cleanup()
            return False
    
    def stop(self) -> None:
        """Stop the UI and note detection."""
        if not self._running:
            return
        
        # Stop note detection
        self._note_detection_service.stop()
        
        # Clean up UI resources
        self.cleanup()
        
        self._running = False
        logger.info("UI stopped")
    
    def is_running(self) -> bool:
        """Check if the UI is running.
        
        Returns:
            True if the UI is running, False otherwise
        """
        return self._running
    
    def _setup_event_handlers(self) -> None:
        """Set up event handlers for note detection events."""
        # Override in subclasses to add specific event handlers
        pass
    
    def _on_note_detected(self, note: DetectedNote, timestamp: float) -> None:
        """Handle note detection events.
        
        Args:
            note: The detected note
            timestamp: The timestamp when the note was detected
        """
        # Emit the note detected event for UI components to handle
        self._events.emit_note_detected(note, timestamp)


class PygameAdapter(UIAdapter):
    """Adapter for Pygame UI."""
    
    def __init__(self, note_detection_service: INoteDetectionService, config: Dict[str, Any] = None):
        """Initialize the Pygame adapter.
        
        Args:
            note_detection_service: Note detection service to use
            config: Configuration options
        """
        super().__init__(note_detection_service)
        self._config = config or {}
        self._pygame = None
        self._clock = None
        self._screen = None
        self._font = None
        self._note_callbacks: List[Callable[[DetectedNote, float], None]] = []
    
    def initialize(self) -> bool:
        """Initialize the Pygame UI.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Import pygame here to avoid dependency if not used
            import pygame
            self._pygame = pygame
            
            pygame.init()
            
            # Set up the display
            width = self._config.get("width", 800)
            height = self._config.get("height", 600)
            self._screen = pygame.display.set_mode((width, height))
            pygame.display.set_caption(self._config.get("title", "Tonal Recall"))
            
            # Set up the clock
            self._clock = pygame.time.Clock()
            
            # Set up the font
            font_size = self._config.get("font_size", 36)
            try:
                self._font = pygame.font.Font(None, font_size)
            except:
                self._font = pygame.font.SysFont("Arial", font_size)
            
            logger.info("Pygame UI initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Pygame UI: {e}")
            return False
    
    def update(self, delta_time: float) -> bool:
        """Update the Pygame UI.
        
        Args:
            delta_time: Time elapsed since last update in seconds
            
        Returns:
            True to continue running, False to exit
        """
        if not self._pygame:
            return False
        
        # Handle events
        for event in self._pygame.event.get():
            if event.type == self._pygame.QUIT:
                return False
            elif event.type == self._pygame.KEYDOWN:
                if event.key == self._pygame.K_ESCAPE:
                    return False
        
        # Update the clock
        self._clock.tick(60)
        
        return True
    
    def render(self) -> None:
        """Render the Pygame UI."""
        if not self._pygame or not self._screen:
            return
        
        # Clear the screen
        self._screen.fill((0, 0, 0))
        
        # Get the current note
        note = self._note_detection_service.get_current_note()
        
        # Render the note
        if note:
            text = self._font.render(f"Note: {note.name}", True, (255, 255, 255))
            text_rect = text.get_rect(center=(self._screen.get_width() // 2, self._screen.get_height() // 2))
            self._screen.blit(text, text_rect)
            
            # Render the frequency
            freq_text = self._font.render(f"Frequency: {note.frequency:.1f} Hz", True, (200, 200, 200))
            freq_rect = freq_text.get_rect(center=(self._screen.get_width() // 2, self._screen.get_height() // 2 + 50))
            self._screen.blit(freq_text, freq_rect)
            
            # Render the confidence
            conf_text = self._font.render(f"Confidence: {note.confidence:.2f}", True, (200, 200, 200))
            conf_rect = conf_text.get_rect(center=(self._screen.get_width() // 2, self._screen.get_height() // 2 + 100))
            self._screen.blit(conf_text, conf_rect)
        else:
            text = self._font.render("No note detected", True, (150, 150, 150))
            text_rect = text.get_rect(center=(self._screen.get_width() // 2, self._screen.get_height() // 2))
            self._screen.blit(text, text_rect)
        
        # Update the display
        self._pygame.display.flip()
    
    def cleanup(self) -> None:
        """Clean up Pygame resources."""
        if self._pygame:
            self._pygame.quit()
            logger.info("Pygame UI cleaned up")
    
    def _setup_event_handlers(self) -> None:
        """Set up event handlers for note detection events."""
        self._events.on_note_detected(self._handle_note_detected)
    
    def _handle_note_detected(self, note: DetectedNote, timestamp: float) -> None:
        """Handle note detection events.
        
        Args:
            note: The detected note
            timestamp: The timestamp when the note was detected
        """
        # Call all registered callbacks
        for callback in self._note_callbacks:
            try:
                callback(note, timestamp)
            except Exception as e:
                logger.error(f"Error in note callback: {e}")


class CursesAdapter(UIAdapter):
    """Adapter for Curses (terminal) UI."""
    
    def __init__(self, note_detection_service: INoteDetectionService, config: Dict[str, Any] = None):
        """Initialize the Curses adapter.
        
        Args:
            note_detection_service: Note detection service to use
            config: Configuration options
        """
        super().__init__(note_detection_service)
        self._config = config or {}
        self._curses = None
        self._stdscr = None
        self._last_note = None
        self._note_history: List[DetectedNote] = []
        self._max_history = self._config.get("max_history", 10)
    
    def initialize(self) -> bool:
        """Initialize the Curses UI.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Import curses here to avoid dependency if not used
            import curses
            self._curses = curses
            
            # Initialize curses
            self._stdscr = curses.initscr()
            curses.noecho()
            curses.cbreak()
            curses.curs_set(0)
            self._stdscr.keypad(True)
            self._stdscr.timeout(100)  # Non-blocking getch
            
            # Check if terminal supports colors
            if curses.has_colors():
                curses.start_color()
                curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)
                curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)
                curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
                curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)
            
            logger.info("Curses UI initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Curses UI: {e}")
            if self._curses:
                self._cleanup_curses()
            return False
    
    def update(self, delta_time: float) -> bool:
        """Update the Curses UI.
        
        Args:
            delta_time: Time elapsed since last update in seconds
            
        Returns:
            True to continue running, False to exit
        """
        if not self._curses or not self._stdscr:
            return False
        
        # Handle keyboard input
        try:
            key = self._stdscr.getch()
            if key == self._curses.KEY_ESCAPE or key == ord('q'):
                return False
        except:
            pass
        
        return True
    
    def render(self) -> None:
        """Render the Curses UI."""
        if not self._curses or not self._stdscr:
            return
        
        try:
            # Clear the screen
            self._stdscr.clear()
            
            # Get terminal dimensions
            height, width = self._stdscr.getmaxyx()
            
            # Draw header
            header = "Tonal Recall - Note Detection"
            self._stdscr.addstr(0, (width - len(header)) // 2, header, self._curses.A_BOLD)
            
            # Draw instructions
            self._stdscr.addstr(height - 1, 0, "Press 'q' or ESC to exit", self._curses.A_DIM)
            
            # Get the current note
            note = self._note_detection_service.get_current_note()
            
            # Draw the current note
            if note:
                note_text = f"Current Note: {note.name}"
                self._stdscr.addstr(2, (width - len(note_text)) // 2, note_text, self._curses.color_pair(1))
                
                freq_text = f"Frequency: {note.frequency:.1f} Hz"
                self._stdscr.addstr(3, (width - len(freq_text)) // 2, freq_text)
                
                conf_text = f"Confidence: {note.confidence:.2f}"
                self._stdscr.addstr(4, (width - len(conf_text)) // 2, conf_text)
                
                signal_text = f"Signal: {note.signal:.4f}"
                self._stdscr.addstr(5, (width - len(signal_text)) // 2, signal_text)
                
                # Update note history
                if self._last_note is None or self._last_note.name != note.name:
                    self._note_history.insert(0, note)
                    if len(self._note_history) > self._max_history:
                        self._note_history.pop()
                self._last_note = note
            else:
                note_text = "No note detected"
                self._stdscr.addstr(2, (width - len(note_text)) // 2, note_text, self._curses.color_pair(3))
            
            # Draw note history
            if self._note_history:
                history_header = "Note History:"
                self._stdscr.addstr(7, 2, history_header, self._curses.A_BOLD)
                
                for i, hist_note in enumerate(self._note_history):
                    if i >= height - 10:  # Limit by available space
                        break
                    hist_text = f"{i+1}. {hist_note.name} ({hist_note.frequency:.1f} Hz)"
                    self._stdscr.addstr(8 + i, 4, hist_text, self._curses.color_pair(4))
            
            # Refresh the screen
            self._stdscr.refresh()
            
        except Exception as e:
            logger.error(f"Error rendering Curses UI: {e}")
    
    def cleanup(self) -> None:
        """Clean up Curses resources."""
        if self._curses:
            self._cleanup_curses()
            logger.info("Curses UI cleaned up")
    
    def _cleanup_curses(self) -> None:
        """Clean up Curses resources safely."""
        try:
            if self._stdscr:
                self._stdscr.keypad(False)
            self._curses.nocbreak()
            self._curses.echo()
            self._curses.endwin()
        except:
            pass
    
    def _setup_event_handlers(self) -> None:
        """Set up event handlers for note detection events."""
        self._events.on_note_detected(self._handle_note_detected)
    
    def _handle_note_detected(self, note: DetectedNote, timestamp: float) -> None:
        """Handle note detection events.
        
        Args:
            note: The detected note
            timestamp: The timestamp when the note was detected
        """
        # Update the last note
        self._last_note = note
