# Canary-1B-v2 for FastRTC

A PyPI package that wraps NVIDIA's Canary-1B-v2 model (via NeMo toolkit) for speech-to-text (STT) transcription, compatible with the FastRTC STTModel protocol.

Canary is a multilingual ASR model supporting **25 European languages**, built on NVIDIA's FastConformer-Transformer encoder-decoder architecture.

## Installation

```bash
pip install canary-fastrtc
```

For audio file loading capabilities, install with the audio extras:

```bash
pip install canary-fastrtc[audio]
```

For development:

```bash
pip install canary-fastrtc[dev]
```

### Prerequisites

This package requires the **NVIDIA NeMo toolkit** (installed from the main branch). For GPU inference, ensure you have CUDA-compatible PyTorch installed:

```bash
# Example: Install PyTorch with CUDA 12.4
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124
```

## Usage

### Basic Usage

```python
from canary_fastrtc import get_stt_model
import numpy as np

# Create the model (downloads from HF/NGC if not cached)
model = get_stt_model()

# Example: Create a sample audio array (actual audio would come from a file or mic)
sample_rate = 16000
audio_data = np.zeros(16000, dtype=np.float32)  # 1 second of silence

# Transcribe
text = model.stt((sample_rate, audio_data))
print(f"Transcription: {text}")
```

### Loading Audio Files

If you've installed with the audio extras:

```python
from canary_fastrtc import get_stt_model, load_audio

# Load model
model = get_stt_model()

# Load audio file (automatically resamples to 16kHz)
audio = load_audio("path/to/audio.wav")

# Transcribe
text = model.stt(audio)
print(f"Transcription: {text}")
```

### Using with FastRTC

```python
from canary_fastrtc import get_stt_model

# Create the model
canary_model = get_stt_model()

# Use within FastRTC applications
# (Follow FastRTC documentation for integration details)
```

### Multilingual Transcription

Canary supports multiple languages. Specify the source language during initialization:

```python
from canary_fastrtc import CanarySTT

# Transcribe German audio
model = CanarySTT(language="de")
text = model.stt((sample_rate, german_audio))

# Transcribe Italian audio
model = CanarySTT(language="it")
text = model.stt((sample_rate, italian_audio))

# Transcribe French audio
model = CanarySTT(language="fr")
text = model.stt((sample_rate, french_audio))
```

Supported languages:

| Code | Language | Code | Language | Code | Language |
|------|----------|------|----------|------|----------|
| `bg` | Bulgarian | `hu` | Hungarian | `pt` | Portuguese |
| `hr` | Croatian | `it` | Italian | `ro` | Romanian |
| `cs` | Czech | `lv` | Latvian | `sk` | Slovak |
| `da` | Danish | `lt` | Lithuanian | `sl` | Slovenian |
| `nl` | Dutch | `mt` | Maltese | `es` | Spanish |
| `en` | English | `pl` | Polish | `sv` | Swedish |
| `et` | Estonian | `ru` | Russian | `uk` | Ukrainian |
| `fi` | Finnish | `de` | German | | |
| `fr` | French | `el` | Greek | | |

## Available Models

- `nvidia/canary-1b-v2` (default, multilingual, 1B parameters)
- `nvidia/canary-1b` (original version)

Example:

```python
from canary_fastrtc import get_stt_model

# Use the original Canary model
model = get_stt_model("nvidia/canary-1b")
```

## Advanced Configuration

```python
from canary_fastrtc import CanarySTT

# Configure with specific device, precision, and language
model = CanarySTT(
    model="nvidia/canary-1b-v2",
    device="cuda",          # Use GPU (recommended)
    dtype="float16",        # Half precision for faster inference
    language="en",          # Source language
)
```

## Requirements

- Python 3.10+
- PyTorch 2.0+ (with CUDA recommended for GPU inference)
- nemo_toolkit[asr] (installed from NeMo main branch)
- numpy 1.22+
- librosa 0.9+ (optional, for audio file loading)

## Development

Clone the repository and install in development mode:

```bash
git clone https://github.com/Codeblockz/canary-fastrtc.git
cd canary-fastrtc
pip install -e ".[dev,audio]"
```

Run tests:

```bash
pytest tests/
```

## License

MIT
