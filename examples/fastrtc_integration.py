"""
Example showing how to integrate canary-fastrtc with FastRTC.

This example demonstrates how the get_stt_model function from canary_fastrtc
would be used with FastRTC, without requiring FastRTC to be installed.
"""

import numpy as np
from canary_fastrtc import get_stt_model


def test_stt_model():
    """
    Test the get_stt_model function from canary_fastrtc.
    """
    # Initialize the STT model using the get_stt_model function
    print("Loading NVIDIA Canary STT model...")
    stt_model = get_stt_model(device="cpu")

    # Create a simple audio sample for testing
    sample_rate = 16000
    duration = 2  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_data = 0.5 * np.sin(2 * np.pi * 440 * t)

    # Test the STT model
    print("Transcribing audio...")
    transcription = stt_model.stt((sample_rate, audio_data))

    print(f"Transcription: {transcription}")
    print("Note: A sine wave may not produce meaningful transcription.")

    return stt_model


def simulate_fastrtc_integration(stt_model):
    """
    Create a running FastRTC app that uses the STT model.

    This sets up a FastAPI server with FastRTC integration that can be
    accessed through a web browser for testing the STT model.
    """
    print("\nCreating FastRTC integration app...")

    # Import FastRTC components
    from fastrtc import Stream, StreamHandler
    from fastapi import FastAPI
    import uvicorn

    # Create a FastAPI app
    app = FastAPI(title="Canary-1B-v2 FastRTC Demo")

    # Create a custom stream handler that transcribes audio
    class TranscriptionHandler(StreamHandler):
        def __init__(self, stt_model):
            super().__init__()
            self.stt_model = stt_model

        def receive(self, audio):
            # Transcribe the audio using our STT model
            text = self.stt_model.stt(audio)
            print(f"Transcribed: {text}")

            # Send the transcription back to the client
            self.send_message_sync(text)

        def copy(self):
            # Create a copy of this handler
            return TranscriptionHandler(self.stt_model)

        def emit(self):
            # This method is required but not used in our case
            pass

    # Set up a Stream with our custom handler
    stream = Stream(
        TranscriptionHandler(stt_model),
        mode="send",  # Send mode for real-time communication
        modality="audio",  # Audio modality for speech input
    )

    # Mount the stream to the app
    stream.mount(app)

    # Add a simple home page
    @app.get("/")
    def home():
        return {
            "message": "Canary-1B-v2 FastRTC Demo",
            "instructions": "Open /stream in your browser to test the STT model",
        }

    print("\nStarting FastAPI server...")
    print("Open http://localhost:8000/stream in your browser to test the STT model")
    print("Speak into your microphone to see live transcription")
    print("Press Ctrl+C to stop the server")

    # Run the server
    # uvicorn.run(app, host="0.0.0.0", port=8000)
    stream.ui.launch(server_port=7860)


if __name__ == "__main__":
    print("Testing canary-fastrtc integration with FastRTC")

    # First test the STT model
    stt_model = test_stt_model()

    # Then run the FastRTC integration app
    simulate_fastrtc_integration(stt_model)
