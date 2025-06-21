# Refactoring Tonal Recall: A Safe, Parallel Approach

This document outlines a stable-first strategy to refactor the Tonal Recall application. The primary goal is to decouple the core note detection logic from the UI, allowing it to be reused for different frontends.

To avoid the instability of a large, intensive refactoring effort, this plan focuses on building and verifying a new, parallel system before touching the existing, working application.

**Guiding Principle:** Build the new system in parallel with the old one, and only switch over once it's proven to be stable.

---

### **Revised and Improved Refactoring Plan**

#### **Phase 1: Build the Safety Net (Test-First)**

The goal of this phase is to create the tools we need to objectively verify that our refactored code works correctly, without relying on time-consuming manual testing with an instrument.

1.  **Establish a "Ground Truth" Test Suite:**
    *   Utilize the pre-recorded `.wav` files of single notes located in `/home/ahonnecke/src/tonal_recall/recordings/`.
    *   For each `.wav` file, create a corresponding `.json` file in the same directory that contains the expected sequence of detected notes (the "ground truth").
    *   This provides a repeatable, objective way to measure the accuracy of our note detector.

2.  **Develop the Test Harness:**
    *   Implement a test harness script that:
        *   Loads a `.wav` file and its corresponding "ground truth" `.json`.
        *   Uses a `WavFileAudioProvider` to feed audio data to the note detection logic.
        *   Compares the actual detected notes against the ground truth.
        *   Generates a clear report with accuracy metrics (e.g., "Test A: 95%% notes correct").

#### **Phase 2: Build the New Service in Isolation**

Now, we build the new components completely separately from the existing game logic.

1.  **Define Service Contracts (Interfaces):**
    *   Formalize `IAudioProvider` and `INoteDetectionService` abstract base classes (ABCs) in a new file, e.g., `tonal_recall/services/interfaces.py`. This defines the blueprint for our new components.

2.  **Implement `AudioProvider`s:**
    *   Create `LiveAudioProvider` which wraps the existing `sounddevice` logic for real-time input.
    *   Create `WavFileAudioProvider` which reads `.wav` files for the test harness.

3.  **Create the `NoteDetectionService`:**
    *   Create a new `NoteDetectionService` in a new file (`tonal_recall/services/note_detection_service.py`).
    *   This service will encapsulate the core note detection logic. It will take an `IAudioProvider` in its constructor, immediately decoupling it from the audio source.
    *   Run the test harness against this new service. The goal is to make the new service pass all the tests with the same or better accuracy as the original `NoteDetector`.

#### **Phase 3: Create and Verify the New Entrypoints**

Instead of modifying `main.py`, we create new, separate scripts to run our new service.

1.  **Create a Simple CLI Entrypoint for Live Testing:**
    *   Create a new script, e.g., `tonal_recall/cli_detect.py`.
    *   This script will initialize the `NoteDetectionService` with the `LiveAudioProvider` and simply print the detected notes to the console in real-time.
    *   This provides a quick, simple way to verify that the entire new stack works with a live instrument, without any of the complexity of the game UI.

2.  **Create a New Game Entrypoint (`game_v2.py`):**
    *   Create a new file, `tonal_recall/game_v2.py`.
    *   This file will be the entrypoint for the refactored game. It will:
        *   Initialize the `NoteDetectionService` with the `LiveAudioProvider`.
        *   Initialize the existing `pygame` UI from `tonal_recall/ui/pygame_ui.py`.
        *   Create a simple "Adapter" class that listens for events from the `NoteDetectionService` and calls the appropriate methods on the `pygame` UI instance.
    *   Now we can run `python -m tonal_recall.game_v2` to play the full game using the completely new, decoupled backend.

#### **Phase 4: Deprecation and Cleanup (Future)**

Only after the new `game_v2.py` entrypoint is confirmed to be stable and working correctly would we consider modifying the original code.

1.  **Update Documentation:** Change the `README` and other documentation to point to the new `game_v2.py` entrypoint.
2.  **Deprecate:** Mark `main.py` as deprecated.
3.  **Remove:** After a period of time, the old `main.py` and its direct dependencies can be safely removed.
