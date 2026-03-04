import numpy as np
import os
import pytest
from unittest.mock import patch, MagicMock

from canary_fastrtc import SUPPORTED_LANGUAGES


# ---------------------------------------------------------------------------
# SUPPORTED_LANGUAGES (module-level, no model loading required)
# ---------------------------------------------------------------------------


def test_supported_languages_count():
    """Test that all 25 supported languages are present."""
    assert len(SUPPORTED_LANGUAGES) == 25


def test_supported_languages_contains_expected():
    """Test that all expected languages are in the tuple."""
    expected = {
        "bg", "hr", "cs", "da", "nl",
        "en", "et", "fi", "fr", "de",
        "el", "hu", "it", "lv", "lt",
        "mt", "pl", "pt", "ro", "sk",
        "sl", "es", "sv", "ru", "uk",
    }
    assert set(SUPPORTED_LANGUAGES) == expected


def test_supported_languages_class_attribute():
    """Test that CanarySTT.SUPPORTED_LANGUAGES equals the module-level tuple."""
    from canary_fastrtc.model import CanarySTT
    assert CanarySTT.SUPPORTED_LANGUAGES == SUPPORTED_LANGUAGES


# ---------------------------------------------------------------------------
# OMP_NUM_THREADS sanitisation (import-time helper)
# ---------------------------------------------------------------------------


def test_sanitize_omp_millicore():
    """Test that Kubernetes millicore OMP_NUM_THREADS is sanitised."""
    from canary_fastrtc import _sanitize_omp_env

    original = os.environ.get("OMP_NUM_THREADS")
    try:
        os.environ["OMP_NUM_THREADS"] = "3500m"
        _sanitize_omp_env()
        assert os.environ["OMP_NUM_THREADS"] == "4"

        os.environ["OMP_NUM_THREADS"] = "1000m"
        _sanitize_omp_env()
        assert os.environ["OMP_NUM_THREADS"] == "1"

        os.environ["OMP_NUM_THREADS"] = "500m"
        _sanitize_omp_env()
        assert os.environ["OMP_NUM_THREADS"] == "1"
    finally:
        if original is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = original


def test_sanitize_omp_valid_integer():
    """Test that a valid integer OMP_NUM_THREADS is left unchanged."""
    from canary_fastrtc import _sanitize_omp_env

    original = os.environ.get("OMP_NUM_THREADS")
    try:
        os.environ["OMP_NUM_THREADS"] = "4"
        _sanitize_omp_env()
        assert os.environ["OMP_NUM_THREADS"] == "4"
    finally:
        if original is None:
            os.environ.pop("OMP_NUM_THREADS", None)
        else:
            os.environ["OMP_NUM_THREADS"] = original


def test_sanitize_omp_not_set():
    """Test that missing OMP_NUM_THREADS is left alone."""
    from canary_fastrtc import _sanitize_omp_env

    original = os.environ.get("OMP_NUM_THREADS")
    try:
        os.environ.pop("OMP_NUM_THREADS", None)
        _sanitize_omp_env()
        assert "OMP_NUM_THREADS" not in os.environ
    finally:
        if original is not None:
            os.environ["OMP_NUM_THREADS"] = original


# ---------------------------------------------------------------------------
# CanarySTT initialisation (mocked - no real model download)
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_nemo():
    """Patch NeMo and torch so CanarySTT can be instantiated without weights."""
    mock_model = MagicMock()
    mock_model.eval.return_value = mock_model
    mock_model.to.return_value = mock_model
    mock_model.half.return_value = mock_model
    mock_model.bfloat16.return_value = mock_model
    mock_model.preprocessor.featurizer.dither = 1.0

    with patch("canary_fastrtc.model.ASRModel") as mock_asr_cls:
        mock_asr_cls.from_pretrained.return_value = mock_model
        yield mock_asr_cls, mock_model


def _make_model(mock_nemo, **kwargs):
    """Helper: import and create a CanarySTT with mocked NeMo."""
    from canary_fastrtc.model import CanarySTT
    return CanarySTT(**kwargs)


def test_default_initialisation(mock_nemo):
    """Test defaults: model id, language, device."""
    model = _make_model(mock_nemo, device="cpu")
    assert model.model_id == "nvidia/canary-1b-v2"
    assert model.language == "en"
    assert model.device == "cpu"


def test_custom_language(mock_nemo):
    """Test that a custom language is stored."""
    model = _make_model(mock_nemo, device="cpu", language="de")
    assert model.language == "de"


def test_italian_language(mock_nemo):
    """Test Italian language."""
    model = _make_model(mock_nemo, device="cpu", language="it")
    assert model.language == "it"


def test_model_eval_called(mock_nemo):
    """Test that .eval() is called during loading."""
    mock_asr_cls, mock_model = mock_nemo
    _make_model(mock_nemo, device="cpu")
    mock_model.eval.assert_called_once()


def test_dither_disabled(mock_nemo):
    """Test that dither is set to 0.0 for deterministic inference."""
    mock_asr_cls, mock_model = mock_nemo
    _make_model(mock_nemo, device="cpu")
    assert mock_model.preprocessor.featurizer.dither == 0.0


# ---------------------------------------------------------------------------
# get_stt_model convenience function
# ---------------------------------------------------------------------------


def test_get_stt_model(mock_nemo):
    """Test that get_stt_model returns a CanarySTT instance."""
    from canary_fastrtc.model import CanarySTT, get_stt_model

    # Clear lru_cache from prior runs
    get_stt_model.cache_clear()

    mock_asr_cls, mock_model = mock_nemo
    # Mock transcribe for warm-up call
    mock_result = MagicMock()
    mock_result.text = ""
    mock_model.transcribe.return_value = [mock_result]

    m = get_stt_model(device="cpu")
    assert isinstance(m, CanarySTT)


# ---------------------------------------------------------------------------
# stt() method (mocked transcribe)
# ---------------------------------------------------------------------------


def test_stt_float32_audio(mock_nemo):
    """Test stt() with float32 audio."""
    mock_asr_cls, mock_model = mock_nemo
    mock_result = MagicMock()
    mock_result.text = "hello world"
    mock_model.transcribe.return_value = [mock_result]

    model = _make_model(mock_nemo, device="cpu")
    audio = np.zeros(16000, dtype=np.float32)
    text = model.stt((16000, audio))
    assert text == "hello world"


def test_stt_int16_audio(mock_nemo):
    """Test stt() converts int16 to float32 before writing WAV."""
    mock_asr_cls, mock_model = mock_nemo
    mock_result = MagicMock()
    mock_result.text = "test"
    mock_model.transcribe.return_value = [mock_result]

    model = _make_model(mock_nemo, device="cpu")
    audio = np.zeros(16000, dtype=np.int16)
    text = model.stt((16000, audio))
    assert text == "test"


def test_stt_stereo_squeezed(mock_nemo):
    """Test stt() squeezes stereo to mono."""
    mock_asr_cls, mock_model = mock_nemo
    mock_result = MagicMock()
    mock_result.text = "mono"
    mock_model.transcribe.return_value = [mock_result]

    model = _make_model(mock_nemo, device="cpu")
    mono = np.zeros(16000, dtype=np.float32)
    stereo = np.stack([mono, mono])
    text = model.stt((16000, stereo))
    assert text == "mono"


def test_stt_transcribe_called_with_lang(mock_nemo):
    """Test that transcribe is called with source_lang and target_lang."""
    mock_asr_cls, mock_model = mock_nemo
    mock_result = MagicMock()
    mock_result.text = "ciao"
    mock_model.transcribe.return_value = [mock_result]

    model = _make_model(mock_nemo, device="cpu", language="it")
    audio = np.zeros(16000, dtype=np.float32)
    model.stt((16000, audio))

    call_kwargs = mock_model.transcribe.call_args
    assert call_kwargs[1]["source_lang"] == "it"
    assert call_kwargs[1]["target_lang"] == "it"


def test_stt_returns_empty_on_error(mock_nemo):
    """Test stt() returns empty string when transcribe raises."""
    mock_asr_cls, mock_model = mock_nemo
    mock_model.transcribe.side_effect = RuntimeError("boom")

    model = _make_model(mock_nemo, device="cpu")
    audio = np.zeros(16000, dtype=np.float32)
    text = model.stt((16000, audio))
    assert text == ""
