from pathlib import Path
from typing import List, Dict
from canary_fastrtc import get_stt_model
from dotenv import load_dotenv
from fastrtc import (
    get_tts_model,
    Stream,
    ReplyOnPause,
)
from smolagents import CodeAgent, DuckDuckGoSearchTool, OpenAIServerModel

# Load environment variables
load_dotenv()

# Initialize file paths
curr_dir = Path(__file__).parent

# Initialize models
print("About to initialize Canary STT model...", flush=True)
stt_model = get_stt_model(verbose=True)
print("Canary STT model initialized!", flush=True)
tts_model = get_tts_model()

# Conversation state to maintain history
conversation_state: List[Dict[str, str]] = []

# System prompt for agent
system_prompt = "You are a helpful assistant with a snarky attitude."
model = OpenAIServerModel(model_id="gpt-4o")

agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
    ],
    model=model,
    max_steps=2,
    verbosity_level=2,
    description="Websearch agent",
)


def process_response(audio):
    """Process audio input and generate LLM response with TTS"""
    # Convert speech to text using Canary STT model
    text = stt_model.stt(audio)
    if not text.strip():
        return

    input_text = f"{system_prompt}\n\n{text}"
    # Get response from agent
    response_content = agent.run(input_text)

    # Convert response to audio using TTS model
    for audio_chunk in tts_model.stream_tts_sync(response_content or ""):
        # Yield the audio chunk
        yield audio_chunk


stream = Stream(
    handler=ReplyOnPause(process_response, input_sample_rate=16000),
    modality="audio",
    mode="send-receive",
    ui_args={
        "pulse_color": "rgb(255, 255, 255)",
        "icon_button_color": "rgb(255, 255, 255)",
        "title": "🧑‍💻 Canary Voice Agent",
    },
)

if __name__ == "__main__":
    stream.ui.launch(server_port=7860)
