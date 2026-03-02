import numpy as np
import torch
from typing import Optional, Tuple


def detect_device() -> str:
    """Detect the best available device for inference."""
    if torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def load_audio(
    file_path: str,
    target_sr: int = 16000,
) -> Tuple[int, np.ndarray]:
    """
    Load an audio file and return it in the format expected by the STT model.

    Args:
        file_path: Path to the audio file
        target_sr: Target sample rate (Canary expects 16kHz)

    Returns:
        Tuple of (sample_rate, audio_data)
    """
    try:
        import librosa

        audio, sr = librosa.load(file_path, sr=target_sr)
        return (sr, audio)
    except ImportError:
        raise ImportError(
            "librosa is required for loading audio files. "
            "Install it with `pip install librosa`."
        )


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int = 16000,
) -> np.ndarray:
    """
    Resample audio to the target sample rate.

    Canary models expect 16kHz audio. Use this when your audio
    source provides a different sample rate.

    Args:
        audio: Audio data as numpy array
        orig_sr: Original sample rate
        target_sr: Target sample rate (default 16kHz)

    Returns:
        Resampled audio data
    """
    if orig_sr == target_sr:
        return audio

    try:
        import librosa

        return librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
    except ImportError:
        # Fallback: simple linear interpolation resampling
        duration = len(audio) / orig_sr
        target_length = int(duration * target_sr)
        indices = np.linspace(0, len(audio) - 1, target_length)
        return np.interp(indices, np.arange(len(audio)), audio).astype(audio.dtype)
