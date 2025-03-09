import sounddevice as sd
import numpy as np
import aubio
import time
from collections import deque
from typing import NamedTuple

class Note(NamedTuple):
    frequency: float
    name: str
    confidence: float

def main():
    # Audio settings
    sample_rate = 48000
    buffer_size = 512  # Smaller buffer for lower latency
    
    # Initialize pitch detection
    pitch_detector = aubio.pitch(
        method="yin",
        buf_size=buffer_size,
        hop_size=buffer_size,  # Process entire buffer at once
        samplerate=sample_rate
    )
    pitch_detector.set_unit("Hz")
    pitch_detector.set_tolerance(0.85)

    # Note stability tracking
    history_size = 5  # Number of samples to keep for stability check
    note_history = deque(maxlen=history_size)
    stability_threshold = 1.0  # Hz - maximum difference for note to be considered stable

    def is_note_stable(note: Note) -> bool:
        if len(note_history) < history_size:
            return False
        
        # Check if all recent frequencies are within threshold
        frequencies = [n.frequency for n in note_history]
        max_diff = max(frequencies) - min(frequencies)
        return max_diff <= stability_threshold

    def get_note_name(freq):
        if freq <= 0: return "---"
        notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
        half_steps = round(12 * np.log2(freq / 440.0))
        octave = 4 + (half_steps + 9) // 12
        note_idx = (half_steps % 12 + 12) % 12
        return f"{notes[note_idx]}{octave}"

    def audio_callback(indata, frames, time, status):
        if status:
            # Only print overflow messages once in a while
            if status.input_overflow:
                print("\rInput overflow", end="")
            return
            
        # Get mono audio data and process it directly
        audio_data = indata[:, 0] if indata.shape[1] > 0 else indata.flatten()
        pitch = pitch_detector(audio_data.astype(np.float32))[0]
        confidence = pitch_detector.get_confidence()
        
        if pitch > 0 and confidence > 0.3:
            note_name = get_note_name(pitch)
            current_note = Note(pitch, note_name, confidence)
            note_history.append(current_note)
            
            # Check stability and update display
            stable = is_note_stable(current_note)
            stability_indicator = "●" if stable else "○"
            print(f"\r{stability_indicator} {note_name} ({pitch:.1f} Hz) | Confidence: {confidence:.2f}", end="", flush=True)

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
    print("Play your guitar - checking pitch detection")
    print("● = stable note, ○ = unstable note")
    
    try:
        with sd.InputStream(
            device=device_id,
            channels=1,
            samplerate=sample_rate,
            blocksize=buffer_size,
            callback=audio_callback,
            latency='low'  # Request low latency
        ):
            input()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
