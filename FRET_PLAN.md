# Bass Guitar Fret Detection Implementation Plan

This document outlines the technical approach for implementing fret detection in the Tonal Recall application.

## 1. Audio Processing Pipeline

### Input
- Capture audio from the bass guitar using Web Audio API
- Sample rate: 44.1kHz (CD quality)
- Bit depth: 16-bit

### Pre-processing
- High-pass filter (cutoff ~80Hz) to remove handling noise
- Normalize the signal to -1.0 to 1.0 range
- Noise gate to eliminate background noise when not playing
- Optional: AGC (Automatic Gain Control) for consistent input levels

## 2. Frequency Analysis

### FFT Configuration
- Window size: 4096 samples (≈93ms at 44.1kHz)
- Window function: Blackman-Harris (good frequency resolution)
- Overlap: 50% for smooth detection
- Frequency resolution: ~10.8Hz

### Peak Detection
1. Apply FFT to get frequency spectrum
2. Find local maxima in the frequency domain
3. Identify fundamental frequency (F0) using:
   - Harmonic product spectrum
   - Or YIN algorithm for better low-frequency accuracy
4. Calculate confidence score based on:
   - Peak prominence
   - Harmonic consistency
   - Signal-to-noise ratio

## 3. Note and Fret Mapping

### Frequency to Note Conversion
- Use formula: `n = 12 × log2(f / f_ref) + 69` (MIDI note number)
- Where f_ref is A4 (440Hz)
- Convert MIDI number to note name and octave

### Fret Calculation
For standard bass tuning (E1, A1, D2, G2):

1. For each string (E1=41.2Hz, A1=55Hz, D2=73.42Hz, G2=98Hz):
   - Calculate expected frequency for each fret: `f = f0 × 2^(n/12)`
   - Where n is the fret number (0 = open string)
   - Find the fret with closest frequency match
   - Calculate confidence based on:
     - Frequency deviation from tempered scale
     - String's frequency range
     - Playing position (higher frets have smaller frequency differences)

## 4. Implementation Steps

### 1. Audio Context Setup
```javascript
const audioContext = new (window.AudioContext || window.webkitAudioContext)();
const analyser = audioContext.createAnalyser();
analyser.fftSize = 4096;
analyser.smoothingTimeConstant = 0.8; // Smoothing factor (0-1)
const bufferLength = analyser.frequencyBinCount;
const dataArray = new Float32Array(bufferLength);
```

### 2. Frequency Analysis Loop
1. Get frequency data: `analyser.getFloatFrequencyData(dataArray)`
2. Find peak frequency
3. Apply parabolic interpolation for better accuracy
4. Validate peak against noise floor

### 3. Note Detection
- Convert frequency to MIDI note number
- Map to note name and octave
- Calculate cents deviation from tempered scale

### 4. Fret Detection
- For each string:
  - Calculate expected note for each fret
  - Find best match considering:
    - Frequency proximity
    - Playability (hand position)
    - Musical context (current key/chord)

## 5. Testing and Calibration

### Test Cases
- Open strings
- Each fret position (1-24)
- Hammer-ons and pull-offs
- Bends and vibrato
- Harmonics

### Calibration
- Reference tone generation
- String tuning detection
- Intonation adjustment

## 6. Performance Considerations

### Optimizations
- Web Workers for FFT processing
- Adaptive window sizing
- Downsampling for higher frets
- Caching of frequency calculations

### Memory Management
- Reuse typed arrays
- Efficient garbage collection
- Throttle UI updates

## 7. Integration Points

### Input
- Microphone/line-in selection
- Audio routing
- Gain staging

### Output
- Note detection events
- Fret position
- Confidence metrics
- Visualization data

## 8. Future Enhancements

### Improved Accuracy
- Machine learning model for note classification
- String harmonic analysis
- Playing technique detection

### User Experience
- Visual fretboard feedback
- Intonation guidance
- Practice statistics

### Advanced Features
- Chord recognition
- Scale suggestions
- Real-time performance analysis

## 9. Dependencies
- Web Audio API
- (Optional) WebAssembly for DSP
- (Optional) Machine learning libraries for advanced features
