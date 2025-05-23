import sounddevice as sd
import numpy as np
import aubio
from collections import deque
from dataclasses import dataclass
from typing import Optional

@dataclass
class DetectedNote:
    name: str
    frequency: float
    confidence: float

def main():
    # Audio settings
    sample_rate = 48000
    buffer_size = 512  # Smaller buffer for lower latency
    
    # Initialize pitch detection
    pitch_detector = aubio.pitch(
        method="yin",
        buf_size=buffer_size,
        hop_size=buffer_size,
        samplerate=sample_rate
    )
    pitch_detector.set_unit("Hz")
    pitch_detector.set_tolerance(0.85)

    # Note history for smoothing
    history_size = 8  # Keep last 8 readings
    note_history = deque(maxlen=history_size)
    current_note: Optional[DetectedNote] = None
    
    def get_note_name(freq):
        if freq <= 0: return "---"
        notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
        half_steps = round(12 * np.log2(freq / 440.0))
        octave = 4 + (half_steps + 9) // 12
        note_idx = (half_steps % 12 + 12) % 12
        return f"{notes[note_idx]}{octave}"

    def get_stable_note() -> Optional[DetectedNote]:
        if len(note_history) < history_size // 2:
            return None
            
        # Count occurrences of each note name
        note_counts = {}
        for note in note_history:
            note_counts[note.name] = note_counts.get(note.name, 0) + 1
        
        # Find most common note
        most_common = max(note_counts.items(), key=lambda x: x[1])
        note_name, count = most_common
        
        # Only consider it stable if it appears in majority of history
        if count < history_size // 2:
            return None
            
        # Get average frequency and confidence for this note
        matching_notes = [n for n in note_history if n.name == note_name]
        avg_freq = sum(n.frequency for n in matching_notes) / len(matching_notes)
        avg_conf = sum(n.confidence for n in matching_notes) / len(matching_notes)
        
        return DetectedNote(note_name, avg_freq, avg_conf)

    def audio_callback(indata, frames, time, status):
        nonlocal current_note
        
        if status:
            if status.input_overflow:
                print("\rInput overflow                                            \r", end="")
            return
            
        # Get mono audio data
        audio_data = indata[:, 0] if indata.shape[1] > 0 else indata.flatten()
        
        pitch = pitch_detector(audio_data.astype(np.float32))[0]
        confidence = pitch_detector.get_confidence()
        
        if pitch > 0 and confidence > 0.3:
            note_name = get_note_name(pitch)
            detected = DetectedNote(note_name, pitch, confidence)
            note_history.append(detected)
            
            # Get stable note
            stable_note = get_stable_note()
            if stable_note:
                current_note = stable_note
                print(f"\r● {stable_note.name:4} | {stable_note.frequency:6.1f} Hz | {stable_note.confidence:4.2f}    \r", end="", flush=True)
            elif current_note:
                print(f"\r○ {current_note.name:4} | {current_note.frequency:6.1f} Hz | {current_note.confidence:4.2f}    \r", end="", flush=True)
        elif not note_history:  # Only show waiting when history is empty
            current_note = None
            print("\rWaiting for input...                                         \r", end="", flush=True)

    # List available devices
    print("\nAvailable audio input devices:")
    print("-" * 30)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{i}: {device['name']} (inputs: {device['max_input_channels']}, rate: {device['default_samplerate']})")

    device_id = 16  # Rocksmith Guitar Adapter
    device_info = sd.query_devices(device_id)
    
    print(f"\nUsing device: {device_info['name']}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Buffer size: {buffer_size} samples (~{buffer_size/sample_rate*1000:.1f}ms)")
    print("\nStarting audio stream...")
    print("● = stable note, ○ = changing note")
    print("Note format: G#3 means G-sharp in octave 3 (Middle C is C4)")
    print("Common bass notes: E1=41Hz, A1=55Hz, D2=73Hz, G2=98Hz")
    print("Common guitar notes: E2=82Hz, A2=110Hz, D3=147Hz, G3=196Hz")
    print("\nWaiting for input...")
    
    try:
        with sd.InputStream(
            device=device_id,
            channels=1,
            samplerate=sample_rate,
            blocksize=buffer_size,
            callback=audio_callback,
            latency='low'
        ):
            input()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
