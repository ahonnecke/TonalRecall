#  MEMORY.md - Project Development Notes

## Project Overview
This project is a **real-time frequency-based game control system** that:
- Captures **audio input from a USB device** (e.g., Rocksmith Real Tone Cable or a microphone).
- Analyzes **incoming audio frequencies in near real-time**.
- Uses **Rust for low-latency frequency analysis**.
- Uses **Python for game logic**.
- Uses **Godot as the game engine** for rendering and visual response.

## Technology Stack
| Component        | Technology Used |
|-----------------|----------------|
| **Audio Input** | `sounddevice` (Python) |
| **Real-Time Frequency Analysis** | `Rust` (Sliding DFT / Goertzel algorithm) |
| **Python-Rust Bridge** | `PyO3` |
| **Game Engine** | `Godot 4` (with Python scripting) |

## 1. Project Structure
Here's a **copy-paste-ready `MEMORY.md`** formatted as a **text file** for your project plan.

```
# MEMORY.md - Project Development Notes

## Project Overview
This project is a **real-time frequency-based game control system** that:
- Captures **audio input from a USB device** (e.g., Rocksmith Real Tone Cable or a microphone).
- Analyzes **incoming audio frequencies in near real-time**.
- Uses **Rust for low-latency frequency analysis**.
- Uses **Python for game logic**.
- Uses **Godot as the game engine** for rendering and visual response.

## Technology Stack
| Component        | Technology Used |
|-----------------|----------------|
| **Audio Input** | `sounddevice` (Python) |
| **Real-Time Frequency Analysis** | `Rust` (Sliding DFT / Goertzel algorithm) |
| **Python-Rust Bridge** | `PyO3` |
| **Game Engine** | `Godot 4` (with Python scripting) |

## 1. Project Structure
```
/audio-frequency-game
│── /audio_analyzer    # Rust module for real-time frequency detection
│   ├── Cargo.toml     # Rust dependencies
│   ├── src/lib.rs     # Rust implementation (Sliding DFT)
│── /python_app        # Python side of the project
│   ├── main.py        # Captures audio & calls Rust module
│── /game              # Game logic (Godot project)
│   ├── game.tscn      # Godot scene
│   ├── game.py        # Game script (Python)
│── README.md          # Project documentation
│── MEMORY.md          # Developer notes (this file)
```

## 2. Audio Processing Details
The core requirement is **low-latency frequency analysis**.
Instead of **FFT**, we use **Sliding DFT (Goertzel’s Algorithm)**, which:
- Computes frequency components **incrementally** instead of processing entire buffers.
- Reduces latency **from ~23ms (FFT) to ~11ms (Sliding DFT)** at a 44.1kHz sample rate.
- Is **faster for tracking a few key frequencies** (ideal for real-time game mechanics).

### Rust Processing Pipeline
1. **Python streams real-time audio** (via `sounddevice`).
2. The **Rust module receives raw PCM samples**.
3. Rust **runs Goertzel’s Algorithm** over a moving buffer to determine the dominant frequency.
4. **Rust returns the frequency to Python**, minimizing inter-process communication overhead.

## 3. Rust Implementation (`audio_analyzer`)
### Rust Dependencies (`Cargo.toml`)
```toml
[lib]
name = "audio_analyzer"
crate-type = ["cdylib"]

[dependencies]
pyo3 = { version = "0.20", features = ["extension-module"] }
```

### Rust Core Processing (`src/lib.rs`)
```rust
use pyo3::prelude::*;

fn goertzel(samples: &[f32], sample_rate: usize, target_freq: f32) -> f32 {
    let k = (0.5 + (samples.len() as f32 * target_freq / sample_rate as f32)).floor();
    let omega = (2.0 * std::f32::consts::PI * k / samples.len() as f32).cos();

    let mut s_prev = 0.0;
    let mut s_prev2 = 0.0;

    for &sample in samples {
        let s = sample + omega * s_prev - s_prev2;
        s_prev2 = s_prev;
        s_prev = s;
    }

    (s_prev2.powi(2) + s_prev.powi(2) - omega * s_prev * s_prev2).sqrt()
}

#[pyfunction]
fn analyze_audio(samples: Vec<f32>, sample_rate: usize) -> PyResult<f32> {
    let mut max_power = 0.0;
    let mut dominant_freq = 0.0;

    let freq_resolution = sample_rate as f32 / samples.len() as f32;
    for i in 1..samples.len() / 2 {
        let freq = i as f32 * freq_resolution;
        let power = goertzel(&samples, sample_rate, freq);

        if power > max_power {
            max_power = power;
            dominant_freq = freq;
        }
    }

    Ok(dominant_freq)
}

#[pymodule]
fn audio_analyzer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(analyze_audio, m)?)?;
    Ok(())
}
```

### Build the Rust Module
```sh
maturin develop
```

## 4. Python Integration (`python_app`)
### Python Dependencies
```sh
pip install sounddevice numpy maturin
```

### Real-Time Audio Capture (`main.py`)
```python
import sounddevice as sd
import numpy as np
import audio_analyzer

SAMPLE_RATE = 44100
CHUNK_SIZE = 512  # Small buffer for lower latency

def callback(indata, frames, time, status):
    if status:
        print(status)
    samples = indata[:, 0]  # Mono audio
    frequency = audio_analyzer.analyze_audio(samples.tolist(), SAMPLE_RATE)
    print(f"Dominant Frequency: {frequency:.2f} Hz")

with sd.InputStream(callback=callback, samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE):
    print("Listening... (Press Ctrl+C to stop)")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nStopping.")
```

## 5. Godot Integration (`game`)
### Game Logic (`game.py`)
```python
extends Node2D
import audio_analyzer
import sounddevice as sd
import numpy as np

SAMPLE_RATE = 44100
CHUNK_SIZE = 512
current_freq = 0.0

def _ready():
    print("Game started. Listening for frequency changes...")

def update_visuals():
    global current_freq
    node = get_node("/root/Game/VisualElement")

    if current_freq < 300:
        node.modulate = Color(1, 0, 0)  # Red for low frequencies
    elif current_freq < 600:
        node.modulate = Color(0, 1, 0)  # Green for mid-range frequencies
    else:
        node.modulate = Color(0, 0, 1)  # Blue for high frequencies

def audio_callback(indata, frames, time, status):
    global current_freq
    if status:
        print(status)
    samples = indata[:, 0]  # Mono audio
    current_freq = audio_analyzer.analyze_audio(samples.tolist(), SAMPLE_RATE)
    print(f"Detected Frequency: {current_freq:.2f} Hz")

with sd.InputStream(callback=audio_callback, samplerate=SAMPLE_RATE, channels=1, blocksize=CHUNK_SIZE):
    print("Listening... (Press Ctrl+C to stop)")
    try:
        while True:
            pass
    except KeyboardInterrupt:
        print("\nStopping.")
```

## 6. Audio Processing with Aubio
### Buffer Size Considerations
When using aubio for pitch detection, it's critical to understand the relationship between buffer sizes:

1. **Audio Stream Buffer**
   - The buffer size used by `sounddevice` for capturing audio
   - Must match the buffer size expected by aubio's pitch detector
   - Smaller buffers (e.g., 512 samples) provide lower latency but may be more prone to overflow

2. **Pitch Detector Configuration**
   - Aubio's pitch detector expects its input size to exactly match its configured buffer size
   - The input data length must match `buf_size` parameter exactly
   - Common error pattern: if input size is X, but buf_size is Y, you'll get "ValueError: input size of pitch should be Y, not X"

### Best Practices
1. **Buffer Alignment**
   ```python
   buffer_size = 512  # Choose a power of 2
   pitch_detector = aubio.pitch(
       method="yin",
       buf_size=buffer_size,  # Must match the audio stream's blocksize
       hop_size=buffer_size,  # Process entire buffer at once
       samplerate=sample_rate
   )
   ```

2. **Direct Processing**
   - Process audio data directly as it comes from the audio callback
   - Avoid complex buffer management when possible
   - Use mono audio to simplify processing

3. **Latency Optimization**
   ```python
   with sd.InputStream(
       blocksize=buffer_size,
       callback=audio_callback,
       latency='low'  # Request low latency mode
   ):
   ```

### Example Implementation
```python
def audio_callback(indata, frames, time, status):
    if status and status.input_overflow:
        print("\rInput overflow", end="")
        return

    # Get mono audio data and process it directly
    audio_data = indata[:, 0] if indata.shape[1] > 0 else indata.flatten()
    pitch = pitch_detector(audio_data.astype(np.float32))[0]
    confidence = pitch_detector.get_confidence()

    if pitch > 0 and confidence > 0.3:
        note = get_note_name(pitch)
        print(f"\rFrequency: {pitch:.1f} Hz | Note: {note}", end="")
```

This approach provides reliable real-time pitch detection with minimal latency, suitable for interactive applications like music games.

## 7. Future Improvements
- Optimize Further with Rust
