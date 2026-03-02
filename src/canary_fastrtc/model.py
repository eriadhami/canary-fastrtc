from typing import Literal, Optional, Protocol
from functools import lru_cache
import os
import sys
import wave
import tempfile
from pathlib import Path
import click
import torch
import numpy as np
from numpy.typing import NDArray


class STTModel(Protocol):
    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str: ...


class CanarySTT:
    """
    A Speech-to-Text model using NVIDIA's Canary-1B-v2 via the NeMo toolkit.
    Implements the FastRTC STTModel protocol.

    Canary is a multilingual ASR model supporting 25 European languages.
    It uses a FastConformer-Transformer encoder-decoder architecture.

    Supported languages:
        bg (Bulgarian), hr (Croatian), cs (Czech), da (Danish), nl (Dutch),
        en (English), et (Estonian), fi (Finnish), fr (French), de (German),
        el (Greek), hu (Hungarian), it (Italian), lv (Latvian), lt (Lithuanian),
        mt (Maltese), pl (Polish), pt (Portuguese), ro (Romanian), sk (Slovak),
        sl (Slovenian), es (Spanish), sv (Swedish), ru (Russian), uk (Ukrainian)

    Attributes:
        model_id: The NVIDIA NeMo model ID
        device: The device to run inference on ('cpu', 'cuda')
        dtype: Data type for model weights (float16, float32, bfloat16)
        language: Source language for transcription
    """

    MODEL_OPTIONS = Literal[
        "nvidia/canary-1b-v2",
        "nvidia/canary-1b",
    ]

    SUPPORTED_LANGUAGES = (
        "bg", "hr", "cs", "da", "nl",
        "en", "et", "fi", "fr", "de",
        "el", "hu", "it", "lv", "lt",
        "mt", "pl", "pt", "ro", "sk",
        "sl", "es", "sv", "ru", "uk",
    )

    def __init__(
        self,
        model: MODEL_OPTIONS = "nvidia/canary-1b-v2",
        device: Optional[str] = None,
        dtype: Literal["float16", "float32", "bfloat16"] = "float16",
        language: str = "en",
    ):
        """
        Initialize the NVIDIA Canary STT model.

        Args:
            model: Model ID to use (nvidia/canary-1b-v2 or nvidia/canary-1b)
            device: Device to use for inference (auto-detected if None)
            dtype: Model precision (float16 recommended for GPU inference)
            language: Source language code (one of 25 supported languages, e.g. en, it, de, fr, es)
        """
        self.model_id = model
        self.language = language

        # Auto-detect device if not specified
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device

        if dtype == "float16":
            self.dtype = torch.float16
        elif dtype == "bfloat16":
            self.dtype = torch.bfloat16
        else:
            self.dtype = torch.float32

        # Load the model
        self._load_model()

    def _load_model(self):
        """Load the Canary model via NVIDIA NeMo toolkit."""
        try:
            import nemo.collections.asr as nemo_asr
        except ImportError:
            raise ImportError(
                "NeMo toolkit is required for NVIDIA Canary models. "
                "Install it with: pip install nemo_toolkit[asr]"
            )

        self.asr_model = nemo_asr.models.ASRModel.from_pretrained(
            model_name=self.model_id
        )
        self.asr_model.eval()

        # Move to specified device
        if self.device != "cpu":
            self.asr_model = self.asr_model.to(self.device)

        # Set dtype for GPU inference
        if self.device != "cpu":
            if self.dtype == torch.float16:
                self.asr_model = self.asr_model.half()
            elif self.dtype == torch.bfloat16:
                self.asr_model = self.asr_model.bfloat16()

    def _write_temp_wav(self, audio_np: np.ndarray, sample_rate: int) -> str:
        """
        Write audio data to a temporary WAV file for NeMo transcription.

        Args:
            audio_np: Audio data as numpy array (float32 normalized to [-1, 1])
            sample_rate: Sample rate of the audio

        Returns:
            Path to the temporary WAV file
        """
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        # Ensure audio is float32 normalized to [-1, 1]
        if audio_np.dtype == np.int16:
            audio_float = audio_np.astype(np.float32) / 32768.0
        else:
            audio_float = audio_np.astype(np.float32)

        # Clamp to [-1, 1]
        audio_float = np.clip(audio_float, -1.0, 1.0)

        # Convert to int16 for WAV writing
        audio_int16 = (audio_float * 32767).astype(np.int16)

        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)  # 16-bit
            wf.setframerate(sample_rate)
            wf.writeframes(audio_int16.tobytes())

        return tmp_path

    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        """
        Transcribe audio to text using NVIDIA Canary.

        Args:
            audio: Tuple of (sample_rate, audio_data)
                  where audio_data is a numpy array of int16 or float32

        Returns:
            Transcribed text as string
        """
        sample_rate, audio_np = audio
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        # Handle different audio formats
        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0

        # Write to temporary WAV file for NeMo transcription
        tmp_path = self._write_temp_wav(audio_np, sample_rate)

        try:
            # Transcribe using NeMo
            transcriptions = self.asr_model.transcribe(
                audio=[tmp_path],
                batch_size=1,
                source_lang=self.language,
                target_lang=self.language,
            )

            # Handle different return types from NeMo
            if isinstance(transcriptions, list) and len(transcriptions) > 0:
                result = transcriptions[0]
                # Handle Hypothesis objects (NeMo can return these with beam search)
                if hasattr(result, "text"):
                    return result.text.strip()
                return str(result).strip()
            return ""
        finally:
            # Clean up temp file
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# For simpler imports
@lru_cache
def get_stt_model(
    model_name: str = "nvidia/canary-1b-v2",
    verbose: bool = True,
    **kwargs,
) -> STTModel:
    """
    Helper function to easily get a Canary STT model instance with warm-up.

    Args:
        model_name: Name of the model to use
        verbose: Whether to print status messages
        **kwargs: Additional arguments to pass to the model constructor

    Returns:
        A warmed-up STTModel instance
    """
    # Set environment variable for tokenizers
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Create the model - remove verbose from kwargs to avoid TypeError
    filtered_kwargs = {k: v for k, v in kwargs.items() if k != "verbose"}
    m = CanarySTT(model=model_name, **filtered_kwargs)

    # Warm up the model with 1 second of silence
    sample_rate = 16000
    audio = np.zeros(sample_rate, dtype=np.float32)

    # Print only to stderr with green styling
    if verbose:
        msg = click.style("INFO", fg="green") + ":\t  Warming up Canary STT model.\n"
        sys.stderr.write(msg)
        sys.stderr.flush()

    # Warm up the model
    m.stt((sample_rate, audio))

    if verbose:
        msg = (
            click.style("INFO", fg="green") + ":\t  Canary STT model warmed up.\n"
        )
        sys.stderr.write(msg)
        sys.stderr.flush()

    return m
