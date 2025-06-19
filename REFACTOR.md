# Refactor

This game, currently works with legacy not detection at this entrypoint:
uv run python ./tonal_recall/main.py --ui pygame --difficulty 3 --duration 60

it succesfully ingests audio, detects notes and allows the user to play a
"flashcard" game.

The current goal is to retain the functionality, but to refactor such that the
detection module is seperated from the flashcard frontend.

That has been started here:
There is a newly structured version here: 
python -m tonal_recall test --duration 100 --min-confidence 0.5 --min-signal 0.001

Long term the goal is for this backend note detection to be able to be re-used
for different frontend games.

You absolutely cannot just jump into this and go bashing about changing things,
there needs to be a meticulous and safe approach.

Because testing with an instrument is time consuming, craft a plan for very
carefully and safely extracting the note detection.

## Refactoring Plan

### Phase 1: Analysis and Preparation (No Code Changes)

1. **Map Current Architecture**
   - Document all components and their interactions
   - Identify dependencies between note detection and UI components
   - Create a visual diagram of the current architecture

2. **Define Clear Interfaces**
   - Design the interface between note detection backend and frontends
   - Document all data structures that cross the boundary
   - Define events and callbacks needed for communication

3. **Create Test Harness**
   - Develop automated tests for note detection functionality
   - Create a simple CLI test tool that exercises core functionality
   - Document baseline behavior for comparison after refactoring

### Phase 2: Separation of Concerns

1. **Extract Core Note Detection**
   - Move `NoteDetector` class to dedicated module without changing functionality
   - Ensure all dependencies are properly imported
   - Verify with test harness that functionality is preserved

2. **Create Facade Service**
   - Implement `NoteDetectionService` as a facade for audio input and note detection
   - Add configuration management for all detection parameters
   - Implement event-based communication system

3. **Update Audio Input**
   - Refactor audio input to be pluggable and independent
   - Create clear interface for different audio input implementations
   - Ensure backward compatibility with existing code

### Phase 3: Frontend Adaptation

1. **Create Adapter for Existing UI**
   - Implement adapter that connects note detection service to existing UI
   - Ensure all existing functionality works through the adapter
   - Minimize changes to UI code

2. **Update Main Entry Point**
   - Modify main.py to use the new architecture
   - Maintain all command-line options and behavior
   - Add new options for configuration if needed

3. **Verify Full Functionality**
   - Test complete system with real instruments
   - Compare behavior to baseline documentation
   - Fix any regressions or issues

### Phase 4: Enhancements and Documentation

1. **Add Plugin System**
   - Implement plugin architecture for extensions
   - Create sample plugins (e.g., visualization, recording)
   - Document plugin API

2. **Improve Error Handling**
   - Add robust error handling throughout the system
   - Implement graceful degradation for common failures
   - Add diagnostic logging

3. **Complete Documentation**
   - Create comprehensive API documentation
   - Add examples for creating new frontends
   - Document configuration options

## Implementation Details

### Core Components

```
tonal_recall/
├── core/
│   ├── note_detector.py       # Core pitch detection and note identification
│   ├── note_types.py          # Data structures for notes and detection results
│   └── events.py              # Event system for communication
├── audio/
│   ├── audio_input.py         # Abstract base class for audio input
│   ├── sounddevice_input.py   # Implementation using sounddevice
│   └── audio_utils.py         # Audio processing utilities
├── services/
│   └── note_detection_service.py  # Facade integrating audio and detection
├── config/
│   ├── config_manager.py      # Configuration management
│   └── default_config.py      # Default configuration values
├── ui/
│   ├── adapters/
│   │   └── pygame_adapter.py  # Adapter for pygame UI
│   └── pygame/                # Existing pygame UI
└── plugins/
    ├── plugin_base.py         # Base class for plugins
    └── visualizer_plugin.py   # Example visualization plugin
```

### Key Interfaces

```python
# Core interface for note detection
class INoteDetector:
    def process_audio(self, audio_data, timestamp) -> Optional[DetectedNote]: ...
    def get_stable_note(self) -> Optional[DetectedNote]: ...
    def set_sample_rate(self, sample_rate: int) -> None: ...

# Interface for audio input
class IAudioInput:
    def start(self, callback) -> None: ...
    def stop(self) -> None: ...
    def is_running(self) -> bool: ...

# Service facade
class INoteDetectionService:
    def start(self, callback) -> None: ...
    def stop(self) -> None: ...
    def get_current_note(self) -> Optional[DetectedNote]: ...
    def configure(self, **params) -> None: ...
```

### Testing Strategy

1. **Unit Tests**: For individual components
2. **Integration Tests**: For component interactions
3. **CLI Test Tool**: For manual testing with real instruments
4. **Regression Tests**: To ensure existing functionality is preserved

## Migration Path

To ensure a smooth transition and minimize risk:

1. Start with the core note detection components that were recently fixed
2. Create the service layer without changing existing functionality
3. Adapt the existing UI to use the new service
4. Gradually enhance with new features

## Success Criteria

- All existing functionality works without regression
- Note detection is completely separated from UI
- New frontends can be created without modifying backend code
- Configuration is flexible and well-documented
- Testing is comprehensive and automated where possible
