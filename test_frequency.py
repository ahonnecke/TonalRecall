import sounddevice as sd
import numpy as np
import aubio
import time as time_module  # Rename to avoid conflict with callback parameter
import click
from collections import deque
from dataclasses import dataclass
from typing import Optional

# Use time_module instead of time to avoid conflict with callback parameter
time = time_module

@dataclass
class DetectedNote:
    name: str        # Note name (e.g., 'A4', 'C#3')
    frequency: float  # Frequency in Hz
    confidence: float # Detection confidence (0-1)
    is_stable: bool   # Whether this is a stable note

def get_note_name(freq):
    """Convert frequency to note name
    
    Args:
        freq: Frequency in Hz
        
    Returns:
        Note name with octave (e.g., 'A4', 'C#3')
    """
    if freq <= 0: return "---"
    
    # Standard reference: A4 = 440Hz
    # Calculate half steps from A4
    half_steps = round(12 * np.log2(freq / 440.0))
    
    # Calculate octave (A4 is in octave 4)
    octave = 4 + (half_steps + 9) // 12
    
    # Get note name (0 = A, 1 = A#, etc.)
    notes = ["A", "A#", "B", "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#"]
    note_idx = (half_steps % 12 + 12) % 12  # Ensure positive index
    
    return f"{notes[note_idx]}{octave}"

def find_rocksmith_adapter():
    """Find the Rocksmith USB Guitar Adapter in the device list"""
    try:
        devices = sd.query_devices()
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0 and 'rocksmith' in device['name'].lower():
                return i, device
        return None, None
    except Exception as e:
        print(f"Error listing audio devices: {e}")
        return None, None

@click.command()
@click.option('--debug', is_flag=True, help='Enable debug mode with detailed output')
def main(debug):
    # List available devices
    print("\nAvailable audio input devices:")
    print("-" * 30)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{i}: {device['name']} (inputs: {device['max_input_channels']}, rate: {device['default_samplerate']}Hz)")
    
    # Try to find Rocksmith adapter automatically
    rocksmith_id, rocksmith_device = find_rocksmith_adapter()
    
    # Get device selection
    try:
        if rocksmith_id is not None:
            print(f"\nAutomatically selected Rocksmith USB Guitar Adapter: {rocksmith_device['name']} (ID: {rocksmith_id})")
            device_id = rocksmith_id
            device_info = rocksmith_device
        else:
            print("\nRocksmith USB Guitar Adapter not found. Please select a device manually.")
            device_id = int(input("Select input device number: "))
            device_info = sd.query_devices(device_id)
            
        if device_info['max_input_channels'] == 0:
            raise ValueError("Selected device has no input channels")
            
        # Use device's preferred sample rate
        sample_rate = int(device_info['default_samplerate'])
        buffer_size = 512  # Smaller buffer for lower latency
        hop_size = buffer_size  # Match exactly to avoid buffer size mismatch errors
    except Exception as e:
        print(f"Error selecting device: {e}")
        return
    
    print(f"\nUsing device: {device_info['name']}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Buffer size: {buffer_size} samples")
    print(f"Host API: {sd.query_hostapis(device_info['hostapi'])['name']}")
    
    # Based on the memory about aubio, we need a smaller buffer size for lower latency
    # For bass guitar, we need to detect lower frequencies (E1 = 41.2 Hz)
    # For lower frequencies, we need a larger buffer size to capture enough cycles
    # Let's try a larger buffer size for better low frequency detection
    buffer_size = 2048  # Increase buffer size for better low frequency detection
    
    # Initialize pitch detection
    # Try different pitch detection methods to find the best one
    # - yinfft: Good for guitar/bass, best frequency precision
    # - yin: Good for monophonic instruments, faster than yinfft
    # - fcomb: Fast, works well for voice
    # - mcomb: Good for noisy signals
    # - schmitt: Simple, works for pure tones
    pitch_methods = ["yinfft", "yin", "fcomb", "mcomb", "schmitt"]
    pitch_method = pitch_methods[0]  # Start with yinfft which is often best for guitar/bass
    
    print(f"\nInitializing pitch detection with method: {pitch_method}")
    print(f"Buffer size: {buffer_size}, Sample rate: {sample_rate}")
    print(f"Expected low E (bass): 41.2 Hz, low E (guitar): 82.4 Hz")
    
    # Create pitch detector with larger buffer for better low frequency detection
    pitch_detector = aubio.pitch(
        method=pitch_method,
        buf_size=buffer_size,
        hop_size=buffer_size,
        samplerate=sample_rate
    )
    pitch_detector.set_unit("Hz")
    pitch_detector.set_silence(-90)  # Lower silence threshold to detect quieter sounds
    pitch_detector.set_tolerance(0.3)  # Lower tolerance for more responsive detection
    
    # Keep track of note history for stability detection
    note_history = deque(maxlen=10)  # Store last 10 detected notes
    current_note = None
    stable_note = None
    
    def get_stable_note() -> Optional[DetectedNote]:
        """Determine if there's a stable note in the history
        
        Returns:
            A DetectedNote if a stable note is found, None otherwise
        """
        nonlocal stable_note  # Access the current stable note for hysteresis
        
        # Filter out zero frequencies from history first
        valid_notes = [n for n in note_history if n.frequency > 0]
        
        if len(valid_notes) < 3:  # Need at least 3 valid readings for stability
            return stable_note  # Return the current stable note to maintain stability
            
        # Standard bass/guitar frequencies
        standard_notes = {
            'E1': 41.2,  # Bass low E
            'A1': 55.0,  # Bass A
            'D2': 73.4,  # Bass D
            'G2': 98.0,  # Bass G
            'E2': 82.4,  # Guitar low E
            'A2': 110.0, # Guitar A
            'D3': 146.8, # Guitar D
            'G3': 196.0, # Guitar G
            'B3': 246.9, # Guitar B
            'E4': 329.6  # Guitar high E
        }
        
        # Group frequencies that are close to each other (within 5Hz)
        freq_groups = []
        for note in valid_notes:
            # Check if this frequency fits in an existing group
            found_group = False
            for group in freq_groups:
                group_avg = sum(n.frequency for n in group) / len(group)
                if abs(note.frequency - group_avg) < 5:  # Within 5Hz
                    group.append(note)
                    found_group = True
                    break
            
            # If no matching group, create a new one
            if not found_group:
                freq_groups.append([note])
        
        # Find the largest group
        if not freq_groups:
            return stable_note  # Return the current stable note to maintain stability
            
        largest_group = max(freq_groups, key=len)
        
        # Calculate average frequency for this group
        avg_freq = sum(n.frequency for n in largest_group) / len(largest_group)
        avg_conf = sum(n.confidence for n in largest_group) / len(largest_group)
        
        # Find the closest standard note
        closest_note = min(standard_notes.items(), key=lambda x: abs(x[1] - avg_freq))
        note_name_std, freq_std = closest_note
        
        # If we're within 5% of a standard note, snap to that frequency
        if abs(avg_freq - freq_std) / freq_std < 0.05:
            avg_freq = freq_std
            note_name = note_name_std
        else:
            # Get the note name from the average frequency
            note_name = get_note_name(avg_freq)
        
        # Apply hysteresis - if we have a stable note already, require a stronger consensus to change
        if stable_note:
            # If the new note is different from the current stable note
            if note_name != stable_note.name:
                # Require a stronger consensus (70% instead of 50%) to change notes
                if len(largest_group) < len(valid_notes) * 0.7:
                    return stable_note  # Keep the current stable note
            # If it's the same note, just update the confidence
            else:
                return DetectedNote(stable_note.name, stable_note.frequency, avg_conf, True)
        else:
            # No current stable note, use normal threshold
            if len(largest_group) < len(valid_notes) * 0.5:  # 50% threshold
                return None
        
        return DetectedNote(note_name, avg_freq, avg_conf, True)
    
    # Callback to process audio
    def audio_callback(indata, frames, stream_time, status):
        nonlocal current_note, stable_note
        
        if status:
            if status.input_overflow:
                print("\rInput overflow                                            \r", end="")
                return  # Skip processing this buffer on overflow
            else:
                print(f"\rStatus: {status}                                      \r", end="")
        
        # Get the audio data and check levels
        audio_data = indata[:, 0] if indata.shape[1] > 0 else indata.flatten()
        signal_max = np.max(np.abs(audio_data))
        
        # Calculate RMS to get a better signal level measurement
        rms = np.sqrt(np.mean(audio_data**2))
        db = 20 * np.log10(rms) if rms > 0 else -100  # Convert to dB
        
        # Increase noise gate threshold to 0.01 as requested
        if signal_max > 0.01:  # Basic noise gate
            try:
                # Process the ENTIRE buffer - must match the pitch detector's buf_size exactly
                raw_pitch = pitch_detector(audio_data.astype(np.float32))
                pitch = raw_pitch[0]  # The frequency in Hz
                confidence = pitch_detector.get_confidence()
                
                # Apply a window function to reduce spectral leakage
                window = np.hanning(len(audio_data))
                windowed_data = audio_data * window
                
                # Calculate FFT with zero-padding for better frequency resolution
                n_fft = 4 * len(audio_data)  # Zero-padding for better frequency resolution
                fft = np.fft.rfft(windowed_data, n=n_fft)
                fft_freqs = np.fft.rfftfreq(n_fft, 1.0/sample_rate)
                
                # Get the magnitude spectrum
                magnitude = np.abs(fft)
                
                # Filter to focus on bass/guitar frequency range (30-500 Hz)
                bass_range = (fft_freqs >= 30) & (fft_freqs <= 500)
                bass_freqs = fft_freqs[bass_range]
                bass_magnitude = magnitude[bass_range]
                
                # Find the dominant frequency in the bass range
                if len(bass_magnitude) > 0:
                    # Find peaks in the spectrum
                    # We'll look for the strongest peak in the bass range
                    max_idx = np.argmax(bass_magnitude)
                    dom_freq = bass_freqs[max_idx]
                    
                    # Standard bass guitar frequencies (E1, A1, D2, G2)
                    bass_notes = {
                        'E1': 41.2,
                        'A1': 55.0,
                        'D2': 73.4,
                        'G2': 98.0
                    }
                    
                    # Find the closest bass note
                    closest_note = min(bass_notes.items(), key=lambda x: abs(x[1] - dom_freq))
                    note_name, note_freq = closest_note
                    
                    # If we're within 5% of a standard bass note, snap to that frequency
                    if abs(dom_freq - note_freq) / note_freq < 0.05:
                        dom_freq = note_freq
                else:
                    dom_freq = 0
                
                # Print output based on debug mode
                current_time = time.strftime("%H:%M:%S")
                
                if debug:
                    # Detailed debug output on new lines for historical tracking
                    if pitch > 0:
                        note_name = get_note_name(pitch)
                        dom_note = get_note_name(dom_freq) if dom_freq > 0 else "---"
                        print(f"{current_time} | Aubio: {pitch:.1f} Hz ({note_name}) | FFT: {dom_freq:.1f} Hz ({dom_note}) | Conf: {confidence:.2f} | Sig: {signal_max:.3f}")
                    else:
                        dom_note = get_note_name(dom_freq) if dom_freq > 0 else "---"
                        print(f"{current_time} | Aubio: {pitch:.1f} Hz | FFT: {dom_freq:.1f} Hz ({dom_note}) | Conf: {confidence:.2f} | Sig: {signal_max:.3f}")
                else:
                    # Simple output on same line for normal mode
                    if dom_freq > 0:
                        dom_note = get_note_name(dom_freq)
                        print(f"\rFrequency: {dom_freq:.1f} Hz | Note: {dom_note} | Signal: {signal_max:.3f}                 \r", end="", flush=True)
                
                # Bass guitar lowest note (E1) is ~41Hz, guitar lowest (E2) is ~82Hz
                # For bass guitar, prioritize FFT analysis which is more reliable for low frequencies
                # For higher frequencies, aubio might be more accurate
                detected_freq = dom_freq if 30 < dom_freq < 200 else pitch
                
                # If aubio returns 0 or a very high frequency, use the FFT frequency
                if pitch == 0 or pitch > 1000:
                    if 30 < dom_freq < 1000:
                        detected_freq = dom_freq
                        
                # If signal is weak (below 0.15), maintain the current stable note
                # This prevents jumping between notes during decay
                if signal_max < 0.15 and stable_note:
                    detected_freq = stable_note.frequency
                
                # Filter out unreasonable frequencies
                if detected_freq > 0 and 30 < detected_freq < 1000:  # Reasonable range for guitar/bass
                    # Convert frequency to note name
                    note_name = get_note_name(detected_freq)
                    detected = DetectedNote(note_name, detected_freq, confidence, False)
                    note_history.append(detected)
                    
                    # Get stable note
                    new_stable_note = get_stable_note()
                    if new_stable_note:
                        stable_note = new_stable_note
                        if debug:
                            print(f"● STABLE: {stable_note.name:4} | {stable_note.frequency:6.1f} Hz | {stable_note.confidence:4.2f}")
                        else:
                            print(f"\r● STABLE: {stable_note.name:4} | {stable_note.frequency:6.1f} Hz                       \r", end="", flush=True)
                    elif stable_note:
                        # Still show the last stable note but with a different indicator
                        if debug:
                            print(f"○ RECENT: {stable_note.name:4} | {stable_note.frequency:6.1f} Hz | {stable_note.confidence:4.2f}")
                        else:
                            print(f"\r○ RECENT: {stable_note.name:4} | {stable_note.frequency:6.1f} Hz                       \r", end="", flush=True)
                    else:
                        # Just show the current frequency and note
                        current_note = detected
                        if debug:
                            print(f"  CURRENT: {current_note.name:4} | {current_note.frequency:6.1f} Hz | {current_note.confidence:4.2f}")
                        else:
                            print(f"\r  CURRENT: {current_note.name:4} | {current_note.frequency:6.1f} Hz                       \r", end="", flush=True)
            except ValueError as e:
                # Handle buffer size mismatch error
                if debug:
                    current_time = time.strftime("%H:%M:%S")
                    print(f"{current_time} | Error: {e}")
                else:
                    print(f"\rError: {e}                                           \r", end="")
        else:
            # Signal too low, show waiting
            if debug:
                current_time = time.strftime("%H:%M:%S")
                print(f"{current_time} | Waiting for input... | Signal: {signal_max:.4f} | dB: {db:.1f}")
            else:
                print("\rWaiting for input...                                      \r", end="", flush=True)
    
    # Start streaming
    try:
        print(f"\nStarting audio stream on device {device_id}: {device_info['name']}")
        print(f"Sample rate: {sample_rate} Hz, Buffer size: {buffer_size}")
        print(f"Host API: {sd.query_hostapis(device_info['hostapi'])['name']}")
        print("\nGuitar/Bass note detection:")
        print("● = stable note, ○ = previously stable note, no symbol = current reading")
        print("Note format: A4 means A in octave 4 (Middle C is C4)")
        print("Common bass notes: E1=41Hz, A1=55Hz, D2=73Hz, G2=98Hz")
        print("Common guitar notes: E2=82Hz, A2=110Hz, D3=147Hz, G3=196Hz")
        
        if debug:
            print("\nDEBUG MODE: Showing detailed diagnostic output")
            print("Dual detection: Using both aubio pitch detection and FFT analysis")
            print("This helps identify when one method fails but the other succeeds")
        print("\nMake some noise with your instrument...")
        print("Press Ctrl+C to stop")
        
        with sd.InputStream(
            device=device_id,
            channels=1,
            samplerate=sample_rate,
            blocksize=buffer_size,
            callback=audio_callback,
            dtype='float32',  # Explicitly use float32 to match aubio expectations
            latency='high'  # Try high latency to reduce overflow issues
        ):
            # Add keyboard control to switch pitch detection methods
            print("Press 1-5 to switch pitch detection methods, Ctrl+C to exit")
            
            # Keep the stream running until interrupted
            while True:
                key = input()
                if key.isdigit() and 1 <= int(key) <= len(pitch_methods):
                    method_idx = int(key) - 1
                    pitch_method = pitch_methods[method_idx]
                    print(f"\nSwitching to pitch detection method: {pitch_method}")
                    
                    # Create new pitch detector with selected method
                    pitch_detector = aubio.pitch(
                        method=pitch_method,
                        buf_size=buffer_size,
                        hop_size=buffer_size,
                        samplerate=sample_rate
                    )
                    pitch_detector.set_unit("Hz")
                    pitch_detector.set_silence(-70)
                    pitch_detector.set_tolerance(0.8)
                    
                    # Clear history when changing methods
                    note_history.clear()
                    current_note = None
                    stable_note = None
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()  # Click will automatically handle the command-line arguments
