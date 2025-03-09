import sounddevice as sd
import numpy as np
import aubio
import threading
import queue
import time

def list_audio_devices():
    """List all available audio input devices"""
    try:
        devices = sd.query_devices()
        input_devices = []
        for i, device in enumerate(devices):
            if device['max_input_channels'] > 0:
                input_devices.append((i, device['name']))
        return input_devices
    except Exception as e:
        print(f"Error listing audio devices: {e}")
        return []

class BassAudioDetector:
    def __init__(self, sample_rate=44100, buffer_size=1024, device=None):
        self.sample_rate = sample_rate
        self.buffer_size = buffer_size
        self.audio_queue = queue.Queue()
        self.running = False
        self.error = None
        
        # Set input device with error handling
        try:
            self.device = device
            if device is None:
                # Try to find Rocksmith cable
                devices = sd.query_devices()
                for i, dev in enumerate(devices):
                    if dev['max_input_channels'] > 0 and 'rocksmith' in dev['name'].lower():
                        self.device = i
                        print(f"Found Rocksmith cable: {dev['name']}")
                        break
            
            # Validate device
            if self.device is not None:
                device_info = sd.query_devices(self.device)
                if device_info['max_input_channels'] == 0:
                    raise ValueError(f"Selected device {self.device} has no input channels")
                
        except Exception as e:
            self.error = f"Error initializing audio device: {e}"
            print(self.error)
            return
        
        try:
            # Initialize aubio pitch detection
            self.pitch_detector = aubio.pitch(
                method="yinfft",
                buf_size=self.buffer_size,
                hop_size=self.buffer_size,
                samplerate=self.sample_rate
            )
            
            # Set pitch detection tolerance
            self.pitch_detector.set_tolerance(0.8)
            
        except Exception as e:
            self.error = f"Error initializing pitch detector: {e}"
            print(self.error)
            return
            
        # Current detected frequency
        self.current_frequency = 0.0
        self.confidence = 0.0

    def audio_callback(self, indata, frames, time, status):
        """Callback for sounddevice to process audio chunks"""
        if status:
            print(f"Status: {status}")
        if self.running:
            # Only process if we're running
            self.audio_queue.put(indata[:, 0])  # Only take first channel

    def process_audio(self):
        """Process audio chunks from the queue"""
        while self.running:
            try:
                audio_data = self.audio_queue.get(timeout=1.0)
                pitch = self.pitch_detector(audio_data.astype(np.float32))[0]
                confidence = self.pitch_detector.get_confidence()
                
                # Only update if we have a confident pitch detection
                if confidence > 0.8:
                    self.current_frequency = float(pitch)
                    self.confidence = confidence
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error processing audio: {e}")
                self.error = str(e)
                self.running = False
                break

    def start(self):
        """Start audio detection"""
        if self.error:
            print(f"Cannot start due to initialization error: {self.error}")
            return False
            
        try:
            self.running = True
            # Start audio processing thread
            self.process_thread = threading.Thread(target=self.process_audio)
            self.process_thread.daemon = True
            self.process_thread.start()
            
            # Start audio input stream with timeout
            self.stream = sd.InputStream(
                device=self.device,
                channels=1,
                samplerate=self.sample_rate,
                blocksize=self.buffer_size,
                callback=self.audio_callback
            )
            self.stream.start()
            print("Audio detection started")
            return True
            
        except Exception as e:
            self.error = f"Error starting audio detection: {e}"
            print(self.error)
            self.running = False
            return False

    def stop(self):
        """Stop audio detection"""
        self.running = False
        if hasattr(self, 'stream'):
            try:
                self.stream.stop()
                self.stream.close()
            except Exception as e:
                print(f"Error stopping stream: {e}")
        if hasattr(self, 'process_thread'):
            self.process_thread.join(timeout=1.0)

    def get_frequency(self):
        """Get the current detected frequency and confidence"""
        if self.error:
            return 0.0, 0.0
        return self.current_frequency, self.confidence

    def get_error(self):
        """Get any error that occurred during initialization or processing"""
        return self.error


if __name__ == "__main__":
    # Simple test to verify the detector works
    detector = BassAudioDetector()
    if detector.start():
        try:
            print("Listening for bass frequencies... Press Ctrl+C to stop")
            while True:
                freq, conf = detector.get_frequency()
                if conf > 0.8:  # Only print confident detections
                    print(f"Frequency: {freq:.1f} Hz (Confidence: {conf:.2f})")
                time.sleep(0.1)
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            detector.stop()
    else:
        print("Failed to start audio detection")
