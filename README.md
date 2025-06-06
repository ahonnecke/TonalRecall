ahonnecke@antonym:~/src/guitar_learning_game$ uv run python test_frequency.py 
warning: No `requires-python` value found in the workspace. Defaulting to `>=3.11`.

Available audio input devices:
------------------------------
0: HDA Intel PCH: ALC3234 Analog (hw:0,0) (inputs: 2, rate: 44100.0Hz)
10: sysdefault (inputs: 128, rate: 48000.0Hz)
11: samplerate (inputs: 128, rate: 44100.0Hz)
12: speexrate (inputs: 128, rate: 44100.0Hz)
13: jack (inputs: 2, rate: 48000.0Hz)
14: pipewire (inputs: 64, rate: 44100.0Hz)
15: pulse (inputs: 32, rate: 44100.0Hz)
16: upmix (inputs: 8, rate: 44100.0Hz)
17: vdownmix (inputs: 6, rate: 44100.0Hz)
18: default (inputs: 32, rate: 44100.0Hz)
19: USB  Live camera Analog Stereo (inputs: 2, rate: 48000.0Hz)
20: Blue Snowball Mono (inputs: 1, rate: 48000.0Hz)
21: USB Audio Mono (inputs: 1, rate: 48000.0Hz)
22: Rocksmith Guitar Adapter Mono (inputs: 1, rate: 48000.0Hz)
23: Rhythmbox (inputs: 2, rate: 48000.0Hz)
24: Firefox (inputs: 4, rate: 48000.0Hz)
25: speech-dispatcher-dummy (inputs: 2, rate: 48000.0Hz)
27: Built-in Audio Analog Stereo Monitor (inputs: 2, rate: 48000.0Hz)
29: USB Audio Analog Stereo Monitor (inputs: 2, rate: 48000.0Hz)
33: PulseAudio Volume Control Monitor (inputs: 19, rate: 48000.0Hz)

Automatically selected Rocksmith USB Guitar Adapter: Rocksmith Guitar Adapter Mono (ID: 22)

Using device: Rocksmith Guitar Adapter Mono
Sample rate: 48000 Hz
Buffer size: 512 samples
Host API: JACK Audio Connection Kit

Initializing pitch detection with method: yinfft
Buffer size: 512, Sample rate: 48000

Starting audio stream on device 22: Rocksmith Guitar Adapter Mono
Sample rate: 48000 Hz, Buffer size: 512
Host API: JACK Audio Connection Kit

Guitar/Bass note detection:
● = stable note, ○ = previously stable note, no symbol = current reading
Note format: A4 means A in octave 4 (Middle C is C4)
Common bass notes: E1=41Hz, A1=55Hz, D2=73Hz, G2=98Hz
Common guitar notes: E2=82Hz, A2=110Hz, D3=147Hz, G3=196Hz

Make some noise with your instrument...
Press Ctrl+C to stop
Press 1-5 to switch pitch detection methods, Ctrl+C to exit
^Citing for input...                                       
Stopped by user
