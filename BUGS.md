# Known Issues and Bugs

## Note Detection Inaccuracy

### Description
The note detector is incorrectly identifying G#0 when the actual note being played is G0. This occurs because the frequency grouping threshold is too wide, causing adjacent notes to be grouped together.

### Symptoms
- System reports G#0 when G0 is played
- Incorrect note detection in the lower octaves
- Inconsistent note identification for notes that are close in frequency

### Root Cause Analysis
1. **Frequency Grouping Threshold**: The current 10Hz grouping threshold is too wide for accurate note distinction in lower octaves.
2. **Note Proximity**: The frequency difference between G0 (~48.99Hz) and G#0 (~51.91Hz) is approximately 3Hz, which is within the current 10Hz threshold.
3. **Confidence Handling**: The system sometimes gives higher confidence to incorrect detections (G#0) even when most detections are for the correct note (G0).

### Logs
```
[DEBUG] Grouped G0 (48.8Hz) with group avg 48.8Hz (diff: 0.0 < 10.0Hz)
[DEBUG] Grouped G#0 (51.7Hz) with group avg 48.5Hz (diff: 3.2 < 10.0Hz)
[DEBUG] Ignoring small frequency change: G#0 -> G0 (diff: 3.0Hz < 10.0Hz)
```

### Impact
- Incorrect note detection affects the accuracy of the note recognition system
- Particularly problematic for bass notes where frequency differences between adjacent notes are smaller
- May cause frustration for users when the system doesn't correctly identify played notes

### Proposed Solutions
1. **Dynamic Thresholding**:
   - Implement frequency-dependent thresholding (smaller thresholds for lower frequencies)
   - Use a percentage-based threshold relative to the note frequency instead of a fixed Hz value

2. **Musical Interval Awareness**:
   - Consider musical intervals when grouping notes
   - Enforce that grouped notes should be musically related (e.g., octaves, fifths)

3. **Confidence Weighting**:
   - Give more weight to higher confidence detections when determining the final note
   - Implement a voting system that considers the frequency of each note detection

4. **Temporary Workaround**:
   - Adjust the `_group_hz` parameter to a smaller value (e.g., 5Hz) for better note separation
   - This can be done as a quick fix but a more robust solution is recommended

### Related Components
- `NoteDetector` class
- `_group_hz` parameter
- Note grouping logic in the detection pipeline

### Priority
**Medium-High** - Affects core functionality and user experience

### Status
**Open** - Needs implementation of one of the proposed solutions
