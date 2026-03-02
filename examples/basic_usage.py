"""
Basic usage example for canary-fastrtc.

This example creates a simple sine wave audio sample and transcribes it.
"""

import numpy as np
from canary_fastrtc import get_stt_model


def main():
    # Create a simple sine wave as test audio
    sample_rate = 16000
    duration = 3  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)

    # Create a 440 Hz sine wave
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)

    print("Loading NVIDIA Canary model (this may take a moment the first time)...")

    # Initialize model with CPU for compatibility
    # For faster inference, you can use:
    # model = get_stt_model(device="cuda") if you have a GPU
    model = get_stt_model(device="cpu")

    print("Transcribing audio...")

    # Run transcription
    result = model.stt((sample_rate, audio))

    print(f"Transcription result: {result}")
    print("Note: A sine wave may not produce meaningful transcription.")


if __name__ == "__main__":
    main()
