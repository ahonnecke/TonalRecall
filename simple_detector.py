import sounddevice as sd
import numpy as np
import aubio
import time

def main():
    # Initialize with Blue Snowball settings
    device_id = 19  # Blue Snowball
    sample_rate = 48000
    chunk_size = 1024
    
    # Initialize pitch detection with basic settings
    pitch_detector = aubio.pitch(
        method="default",
        buf_size=chunk_size,
        hop_size=chunk_size,
        samplerate=sample_rate
    )

    def audio_callback(indata, frames, time, status):
        if status:
            print(f"\nStatus: {status}")
        
        # Get the audio data
        audio_data = indata[:, 0]
        signal_strength = np.sqrt(np.mean(audio_data**2))
        
        # Process all audio
        pitch = pitch_detector(audio_data.astype(np.float32))[0]
        if pitch > 0:  # Only filter out zero/negative frequencies
            note = get_note_name(pitch)
            print(f"\rSignal: {signal_strength:.6f} | Frequency: {pitch:.1f} Hz | Note: {note}", end="", flush=True)

    def get_note_name(freq):
        # A4 = 440Hz
        if freq <= 0: return "---"
        notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
        # Calculate how many half steps away from A4
        half_steps = round(12 * np.log2(freq / 440.0))
        # Calculate the octave and note
        octave = 4 + (half_steps + 9) // 12
        note_idx = (half_steps % 12 + 12) % 12
        return f"{notes[note_idx]}{octave}"

    print("\nStarting audio stream...")
    print("Showing all detected frequencies (Press Ctrl+C to stop)")
    
    try:
        with sd.InputStream(
            device=device_id,
            channels=1,
            samplerate=sample_rate,
            blocksize=chunk_size,
            callback=audio_callback
        ):
            input()
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
