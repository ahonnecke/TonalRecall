import sounddevice as sd
import numpy as np
import aubio

def main():
    # List available devices
    print("\nAvailable audio input devices:")
    print("-" * 30)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            print(f"{i}: {device['name']} (inputs: {device['max_input_channels']}, rate: {device['default_samplerate']})")
    
    # Get device selection
    try:
        device_id = int(input("\nSelect input device number: "))
        device_info = sd.query_devices(device_id)
        if device_info['max_input_channels'] == 0:
            raise ValueError("Selected device has no input channels")
            
        # Use device's preferred sample rate
        sample_rate = int(device_info['default_samplerate'])
        buffer_size = 512  # Smaller buffer for lower latency
        hop_size = buffer_size // 2  # Process with overlap for better responsiveness
    except Exception as e:
        print(f"Error selecting device: {e}")
        return
    
    print(f"\nUsing device: {device_info['name']}")
    print(f"Sample rate: {sample_rate} Hz")
    print(f"Buffer size: {buffer_size} samples")
    print(f"Host API: {sd.query_hostapis(device_info['hostapi'])['name']}")
    
    # Initialize pitch detection
    pitch_detector = aubio.pitch(
        method="yinfft",
        buf_size=buffer_size,
        hop_size=hop_size,
        samplerate=sample_rate
    )
    pitch_detector.set_tolerance(0.85)
    
    # Keep track of last few frequencies for smoothing
    freq_history = []
    max_history = 3
    
    # Callback to process audio
    def audio_callback(indata, frames, time, status):
        if status:
            print(f"Status: {status}")
        
        # Get the audio data and check levels
        audio_data = indata[:, 0]
        signal_max = np.max(np.abs(audio_data))
        
        if signal_max > 0.01:  # Basic noise gate
            # Get pitch
            pitch = pitch_detector(audio_data.astype(np.float32))[0]
            confidence = pitch_detector.get_confidence()
            
            if confidence > 0.85:
                # Add to history and get average
                freq_history.append(pitch)
                if len(freq_history) > max_history:
                    freq_history.pop(0)
                avg_pitch = sum(freq_history) / len(freq_history)
                
                print(f"\rFrequency: {avg_pitch:.1f} Hz (conf: {confidence:.2f})", end="", flush=True)
    
    # Start streaming
    try:
        print("\nStarting audio stream... (Press Ctrl+C to stop)")
        print("Listening...")
        with sd.InputStream(
            device=device_id,
            channels=1,
            samplerate=sample_rate,
            blocksize=buffer_size,
            callback=audio_callback,
            latency='low'
        ):
            input()  # Wait for user to press Enter
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
