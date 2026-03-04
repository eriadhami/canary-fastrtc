"""canary-fastrtc - NVIDIA Canary-1B-v2 for FastRTC speech-to-text."""

import os as _os


def _sanitize_omp_env():
    """Fix OMP_NUM_THREADS before any C extensions or heavy imports.

    Kubernetes / HF Spaces can set OMP_NUM_THREADS to millicore format
    (e.g. '3500m') which crashes libgomp and numexpr.
    Must run before importing torch, numpy-heavy libs, NeMo, etc.
    """
    omp = _os.environ.get("OMP_NUM_THREADS", "")
    if omp and not omp.isdigit():
        if omp.endswith("m") and omp[:-1].isdigit():
            millicores = int(omp[:-1])
            threads = max(1, (millicores + 999) // 1000)
        else:
            threads = max(1, _os.cpu_count() or 1)
        _os.environ["OMP_NUM_THREADS"] = str(threads)


_sanitize_omp_env()

from .model import CanarySTT, get_stt_model, STTModel, SUPPORTED_LANGUAGES  # noqa: E402
from .utils import detect_device, load_audio, resample_audio  # noqa: E402

__version__ = "0.1.0"

__all__ = [
    "CanarySTT",
    "get_stt_model",
    "STTModel",
    "SUPPORTED_LANGUAGES",
    "detect_device",
    "load_audio",
    "resample_audio",
]
