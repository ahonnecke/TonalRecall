class MockNoteDetector:
    """A mock detector for unit tests. Allows manual triggering of note events."""
    def __init__(self):
        self.callback = None
        self.is_running = False

    def start(self, callback):
        self.callback = callback
        self.is_running = True
        return True

    def stop(self):
        self.is_running = False

    def simulate_note_event(self, note, signal_strength=1.0):
        if self.callback and self.is_running:
            self.callback(note, signal_strength)
