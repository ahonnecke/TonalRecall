import sounddevice as sd
import numpy as np
import aubio
import time

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

def main():
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
    
    # Initialize pitch detection
    # Make sure buf_size matches EXACTLY the size of data we'll feed to the detector
    pitch_detector = aubio.pitch(
        method="yin",  # yin method is often more reliable for guitar
        buf_size=buffer_size,  # Use full buffer_size to match audio callback data size
        hop_size=buffer_size,  # Match exactly to avoid buffer size mismatch errors
        samplerate=sample_rate
    )
    pitch_detector.set_unit("Hz")
    pitch_detector.set_tolerance(0.85)
    
    # Keep track of last few frequencies for smoothing
    freq_history = []
    max_history = 3
    
    # Callback to process audio
    def audio_callback(indata, frames, time, status):
        if status:
            if status.input_overflow:
                print("\rInput overflow                                            \r", end="")
                return  # Skip processing this buffer on overflow
            else:
                print(f"\rStatus: {status}                                      \r", end="")
        
        # Get the audio data and check levels
        audio_data = indata[:, 0] if indata.shape[1] > 0 else indata.flatten()
        signal_max = np.max(np.abs(audio_data))
        
        # Print signal level for debugging
        print(f"\rSignal level: {signal_max:.4f}                             \r", end="", flush=True)
        
        if signal_max > 0.01:  # Basic noise gate
            try:
                # Process the ENTIRE buffer - must match the pitch detector's buf_size exactly
                pitch = pitch_detector(audio_data.astype(np.float32))[0]
                confidence = pitch_detector.get_confidence()
                
                if confidence > 0.3:  # Lower threshold to see more readings
                    # Add to history and get average
                    freq_history.append(pitch)
                    if len(freq_history) > max_history:
                        freq_history.pop(0)
                    avg_pitch = sum(freq_history) / len(freq_history)
                    
                    print(f"\rFrequency: {avg_pitch:.1f} Hz (conf: {confidence:.2f})                \r", end="", flush=True)
            except ValueError as e:
                # Handle buffer size mismatch error
                print(f"\rError: {e}                                           \r", end="")
    
    # Start streaming
    try:
        print(f"\nStarting audio stream on device {device_id}: {device_info['name']}")
        print(f"Sample rate: {sample_rate} Hz, Buffer size: {buffer_size}")
        print(f"Host API: {sd.query_hostapis(device_info['hostapi'])['name']}")
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
            # Keep the stream running until interrupted
            while True:
                time.sleep(0.1)
    
    except KeyboardInterrupt:
        print("\nStopped by user")
    except Exception as e:
        print(f"\nError: {e}")

if __name__ == "__main__":
    main()
