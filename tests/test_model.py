import numpy as np
import pytest
from canary_fastrtc import CanarySTT, get_stt_model


def test_model_initialization():
    """Test that the model can be initialized."""
    model = CanarySTT(device="cpu")
    assert model.device == "cpu"
    assert model.model_id == "nvidia/canary-1b-v2"


def test_model_language_default():
    """Test that the default language is English."""
    model = CanarySTT(device="cpu")
    assert model.language == "en"


def test_model_custom_language():
    """Test that a custom language can be set."""
    model = CanarySTT(device="cpu", language="de")
    assert model.language == "de"


def test_model_italian_language():
    """Test that Italian language can be set."""
    model = CanarySTT(device="cpu", language="it")
    assert model.language == "it"


def test_all_supported_languages():
    """Test that all 25 supported languages are recognized."""
    expected_languages = (
        "bg", "hr", "cs", "da", "nl",
        "en", "et", "fi", "fr", "de",
        "el", "hu", "it", "lv", "lt",
        "mt", "pl", "pt", "ro", "sk",
        "sl", "es", "sv", "ru", "uk",
    )
    assert CanarySTT.SUPPORTED_LANGUAGES == expected_languages
    assert len(CanarySTT.SUPPORTED_LANGUAGES) == 25


def test_helper_function():
    """Test the helper function."""
    model = get_stt_model(device="cpu")
    assert isinstance(model, CanarySTT)


@pytest.mark.skip(reason="Requires downloading model weights")
def test_transcription():
    """Test transcription with a simple audio sample."""
    # Create a simple sine wave as test audio
    sample_rate = 16000
    duration = 1  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    # Initialize model
    model = get_stt_model(device="cpu")

    # Run transcription
    result = model.stt((sample_rate, audio))

    # We don't expect meaningful transcription from a sine wave,
    # but the function should run without errors
    assert isinstance(result, str)


@pytest.mark.skip(reason="Requires downloading model weights")
def test_int16_audio():
    """Test that int16 audio is handled correctly."""
    sample_rate = 16000
    duration = 1
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_float = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio_int16 = (audio_float * 32767).astype(np.int16)

    model = get_stt_model(device="cpu")
    result = model.stt((sample_rate, audio_int16))
    assert isinstance(result, str)


@pytest.mark.skip(reason="Requires downloading model weights")
def test_stereo_audio():
    """Test that stereo audio is squeezed to mono."""
    sample_rate = 16000
    duration = 1
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    audio_mono = 0.5 * np.sin(2 * np.pi * 440 * t)
    audio_stereo = np.stack([audio_mono, audio_mono], axis=0)

    model = get_stt_model(device="cpu")
    result = model.stt((sample_rate, audio_stereo))
    assert isinstance(result, str)
