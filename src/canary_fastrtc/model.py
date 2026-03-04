"""NVIDIA Canary-1B-v2 wrapper for the FastRTC STTModel protocol.

Follows the exact NeMo ASR API used by the working reference implementation:
    model = ASRModel.from_pretrained("nvidia/canary-1b-v2")
    model.eval()
    output = model.transcribe([path], source_lang=..., target_lang=...)
    text = output[0].text
"""

from typing import Literal, Optional, Protocol
from functools import lru_cache
import logging
import os
import sys
import wave
import tempfile

import numpy as np
from numpy.typing import NDArray
import torch
# NOTE: NeMo is imported lazily inside _load_model() to avoid blocking
# the main thread for minutes at import time.  Do NOT add a top-level
# ``from nemo.collections.asr.models import ASRModel`` here.

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FastRTC STTModel protocol
# ---------------------------------------------------------------------------

class STTModel(Protocol):
    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str: ...


# ---------------------------------------------------------------------------
# Supported languages (Canary-1B-v2 - 25 European languages)
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES = (
    "bg", "hr", "cs", "da", "nl",
    "en", "et", "fi", "fr", "de",
    "el", "hu", "it", "lv", "lt",
    "mt", "pl", "pt", "ro", "sk",
    "sl", "es", "sv", "ru", "uk",
)


# ---------------------------------------------------------------------------
# CanarySTT - FastRTC-compatible wrapper
# ---------------------------------------------------------------------------

class CanarySTT:
    """Speech-to-Text using NVIDIA Canary-1B-v2 via NeMo.

    Implements the FastRTC ``STTModel`` protocol::

        def stt(self, audio: tuple[int, NDArray]) -> str: ...

    Canary-1B-v2 supports 25 European languages for ASR and speech
    translation.  This wrapper restricts itself to **transcription**
    (source_lang == target_lang) to satisfy the FastRTC STT contract.
    """

    MODEL_OPTIONS = Literal[
        "nvidia/canary-1b-v2",
        "nvidia/canary-1b",
    ]

    SUPPORTED_LANGUAGES = SUPPORTED_LANGUAGES

    def __init__(
        self,
        model: MODEL_OPTIONS = "nvidia/canary-1b-v2",
        device: Optional[str] = None,
        dtype: Literal["float16", "float32", "bfloat16"] = "float16",
        language: str = "en",
    ):
        """
        Args:
            model: NeMo model name (must be a Canary variant).
            device: ``"cuda"`` or ``"cpu"`` (auto-detected when *None*).
            dtype: Weight precision - ``"float16"`` recommended for GPU.
            language: ISO-639-1 source language code (e.g. ``"it"``).
        """
        self.model_id = model
        self.language = language

        # --- device ----------------------------------------------------------
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # --- dtype -----------------------------------------------------------
        _dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        self.dtype = _dtype_map.get(dtype, torch.float32)

        # --- load model (same pattern as the working reference) --------------
        self._load_model()

    # ------------------------------------------------------------------ #
    # Model loading - mirrors the working NVIDIA demo exactly            #
    # ------------------------------------------------------------------ #

    def _load_model(self):
        """Load the Canary model following the proven NeMo pattern.

        Always loads weights on **CPU** first (avoids CUDA-init hangs on
        cloud platforms like HF Spaces where the GPU driver may not be
        fully ready at import time), then moves to the requested device.
        """
        import time as _time

        logger.info("Loading model '%s' (target device=%s) …", self.model_id, self.device)

        # ---- 0. Lazy-import NeMo (takes 2-5 min on cold start) -------------
        # We do this in sub-steps with logging so hangs can be diagnosed.
        t0 = _time.monotonic()

        # Heartbeat: background thread prints every 30s so logs show life
        import threading as _th
        _hb_stop = _th.Event()

        def _hb():
            while not _hb_stop.wait(30):
                sys.stderr.write(
                    f"[canary-stt] … still loading NeMo "
                    f"({_time.monotonic() - t0:.0f}s elapsed)\n"
                )
                sys.stderr.flush()

        _hb_thread = _th.Thread(target=_hb, daemon=True, name="nemo-heartbeat")
        _hb_thread.start()

        try:
            logger.info("[0a/5] import nemo …")
            import nemo  # noqa: F811  — base package, light
            logger.info("[0a/5] nemo base imported (%.1fs)", _time.monotonic() - t0)

            logger.info("[0b/5] import nemo.collections.asr …")
            import nemo.collections.asr  # triggers ASR-specific init
            logger.info("[0b/5] nemo.collections.asr imported (%.1fs)", _time.monotonic() - t0)

            logger.info("[0c/5] import ASRModel …")
            from nemo.collections.asr.models import ASRModel  # full model registry
            logger.info("[0c/5] ASRModel imported (%.1fs)", _time.monotonic() - t0)
        finally:
            _hb_stop.set()  # stop heartbeat regardless of success/failure

        # 1. from_pretrained — always on CPU first
        t1 = _time.monotonic()
        logger.info("[1/5] ASRModel.from_pretrained (CPU) …")
        self.asr_model = ASRModel.from_pretrained(model_name=self.model_id)
        logger.info("[2/5] from_pretrained done (%.1fs)", _time.monotonic() - t1)

        self.asr_model.eval()
        logger.info("[3/5] model.eval() done")

        # 2. Move to target device
        if self.device == "cuda":
            t1 = _time.monotonic()
            logger.info("[4/5] Moving to CUDA …")
            self.asr_model = self.asr_model.to(self.device)
            logger.info("[4/5] .to('cuda') done (%.1fs)", _time.monotonic() - t1)

        # 3. Precision
        if self.device != "cpu":
            if self.dtype == torch.float16:
                self.asr_model = self.asr_model.half()
                logger.info("[4/5] .half() applied")
            elif self.dtype == torch.bfloat16:
                self.asr_model = self.asr_model.bfloat16()
                logger.info("[4/5] .bfloat16() applied")

        # 4. Disable dither for deterministic inference (from working reference)
        if hasattr(self.asr_model, "preprocessor") and hasattr(
            self.asr_model.preprocessor, "featurizer"
        ):
            self.asr_model.preprocessor.featurizer.dither = 0.0

        logger.info(
            "[5/5] Model '%s' ready on %s (total %.1fs)",
            self.model_id, self.device, _time.monotonic() - t0,
        )

    # ------------------------------------------------------------------ #
    # WAV helper                                                         #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _write_temp_wav(audio_np: np.ndarray, sample_rate: int) -> str:
        """Write a float32 audio array to a 16-bit mono WAV temp file."""
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp_path = tmp.name
        tmp.close()

        audio_f32 = audio_np.astype(np.float32)
        audio_f32 = np.clip(audio_f32, -1.0, 1.0)
        pcm16 = (audio_f32 * 32767).astype(np.int16)

        with wave.open(tmp_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm16.tobytes())

        return tmp_path

    # ------------------------------------------------------------------ #
    # FastRTC STTModel.stt()                                             #
    # ------------------------------------------------------------------ #

    def stt(self, audio: tuple[int, NDArray[np.int16 | np.float32]]) -> str:
        """Transcribe audio to text.

        Args:
            audio: ``(sample_rate, audio_data)`` - the FastRTC convention.

        Returns:
            Transcribed text (empty string on failure).
        """
        sample_rate, audio_np = audio

        # --- pre-process -----------------------------------------------------
        if audio_np.ndim > 1:
            audio_np = audio_np.squeeze()

        if audio_np.dtype == np.int16:
            audio_np = audio_np.astype(np.float32) / 32768.0

        # --- write temp WAV (NeMo transcribes from file paths) ---------------
        tmp_path = self._write_temp_wav(audio_np, sample_rate)

        try:
            # Transcribe - exact same API as the working NVIDIA demo:
            #   output = model.transcribe([path], source_lang=..., target_lang=...)
            #   text = output[0].text
            output = self.asr_model.transcribe(
                [tmp_path],
                source_lang=self.language,
                target_lang=self.language,
            )

            # output is a list; output[0] has a .text attribute
            if output and len(output) > 0:
                result = output[0]
                if hasattr(result, "text"):
                    return result.text.strip()
                return str(result).strip()
            return ""
        except Exception:
            logger.exception("Transcription failed")
            return ""
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


# --------------------------------------------------------------------------- #
# Convenience constructor (with warm-up)                                      #
# --------------------------------------------------------------------------- #

@lru_cache
def get_stt_model(
    model_name: str = "nvidia/canary-1b-v2",
    verbose: bool = True,
    **kwargs,
) -> STTModel:
    """Create a CanarySTT and warm it up with silence.

    Args:
        model_name: NeMo model identifier.
        verbose: Print status to stderr.
        **kwargs: Forwarded to CanarySTT.

    Returns:
        A ready-to-use STTModel.
    """
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    filtered_kwargs = {k: v for k, v in kwargs.items() if k != "verbose"}
    m = CanarySTT(model=model_name, **filtered_kwargs)

    # Warm up with 1 s of silence
    if verbose:
        sys.stderr.write("INFO:\t  Warming up Canary STT model.\n")
        sys.stderr.flush()

    sample_rate = 16000
    m.stt((sample_rate, np.zeros(sample_rate, dtype=np.float32)))

    if verbose:
        sys.stderr.write("INFO:\t  Canary STT model warmed up.\n")
        sys.stderr.flush()

    return m
