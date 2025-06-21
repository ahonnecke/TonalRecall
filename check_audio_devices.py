#!/usr/bin/env python3
"""
Script to check available audio devices and supported sample rates.
"""

import sounddevice as sd


def main():
    """Print information about audio devices."""
    print("Available audio devices:")
    print("-" * 70)

    devices = sd.query_devices()
    for i, device in enumerate(devices):
        print(f"Device {i}: {device['name']}")
        print(f"  Max input channels: {device['max_input_channels']}")
        print(f"  Max output channels: {device['max_output_channels']}")
        print(f"  Default sample rate: {device['default_samplerate']} Hz")

        # Try to determine supported sample rates
        if device["max_input_channels"] > 0:
            print("  Testing supported input sample rates...")
            for rate in [8000, 16000, 22050, 44100, 48000, 96000]:
                try:
                    sd.check_input_settings(device=i, samplerate=rate, channels=1)
                    print(f"    {rate} Hz: Supported")
                except Exception as e:
                    print(f"    {rate} Hz: Not supported ({str(e)})")

        print()

    print(f"Default input device: {sd.default.device[0]}")
    print(f"Default output device: {sd.default.device[1]}")
    print(f"Default sample rate: {sd.default.samplerate} Hz")


if __name__ == "__main__":
    main()
