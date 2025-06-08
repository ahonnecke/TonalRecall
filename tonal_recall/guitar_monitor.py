import sounddevice as sd


class GuitarMonitor:
    """
    Simple real-time audio pass-through (input to output) for guitar monitoring.
    Prints device info at startup for debugging.
    """

    def __init__(
        self,
        input_device=None,
        output_device=None,
        samplerate=44100,
        blocksize=1024,
        channels=1,
    ):
        print("[GuitarMonitor] Constructor called")
        self.input_device = input_device
        self.output_device = output_device
        self.samplerate = samplerate
        self.blocksize = blocksize
        self.channels = channels
        self.stream = None
        self.running = False

        # Print available devices and defaults for debugging
        print("\n=== SoundDevice Device List ===")
        try:
            devices = sd.query_devices()
            for idx, dev in enumerate(devices):
                print(
                    f"[{idx}] {dev['name']} (inputs: {dev['max_input_channels']}, outputs: {dev['max_output_channels']})"
                )
        except Exception as e:
            print(f"Could not query devices: {e}")
        print(f"Default input device: {sd.default.device[0]}")
        print(f"Default output device: {sd.default.device[1]}")
        print(
            f"Using input_device={self.input_device}, output_device={self.output_device}"
        )

    def start(self):
        print("[GuitarMonitor] start() called")
        if self.running:
            print("[GuitarMonitor] Already running")
            return

        def callback(indata, outdata, frames, time, status):
            if status:
                print(f"Monitor status: {status}")
            outdata[:] = indata

        try:
            self.stream = sd.Stream(
                device=(self.input_device, self.output_device),
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                channels=self.channels,
                dtype="float32",
                callback=callback,
                latency="low",
            )
            self.stream.start()
            self.running = True
            print("[GuitarMonitor] Stream started successfully")
        except Exception as e:
            print(f"[GuitarMonitor] Failed to start stream: {e}")
            self.running = False

    def stop(self):
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None
        self.running = False
