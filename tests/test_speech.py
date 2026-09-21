from types import SimpleNamespace

import pytest

from app.speech import azure_speech
from app.speech.azure_speech import (
    SpeechConfigurationError,
    SpeechInputError,
    SpeechRecognitionError,
    resolve_tts_profile,
    synthesize_speech,
    transcribe_audio,
)


class FakeFuture:
    def __init__(self, result):
        self.result = result

    def get(self):
        return self.result


def settings():
    return SimpleNamespace(
        azure_speech_key="azure-secret-value",
        azure_speech_region="test-region",
        azure_speech_en_locale="en-US",
        azure_speech_ar_locale="ar-SA",
        azure_speech_en_voice="en-US-TestVoice",
        azure_speech_ar_voice="ar-SA-ZariyahNeural",
    )


def fake_sdk(*, recognition_result=None, synthesis_result=None, voices=None):
    reasons = SimpleNamespace(
        NoMatch="no-match",
        Canceled="canceled",
        RecognizedSpeech="recognized",
        SynthesizingAudioCompleted="synthesized",
    )

    class SpeechConfig:
        def __init__(self, subscription, region):
            self.subscription = subscription
            self.region = region
            self.speech_synthesis_voice_name = None

        def set_speech_synthesis_output_format(self, output_format):
            self.output_format = output_format

    class SpeechRecognizer:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def recognize_once_async(self):
            return FakeFuture(recognition_result)

    class SpeechSynthesizer:
        def __init__(self, *, speech_config, audio_config):
            del audio_config
            if voices is not None:
                voices.append(speech_config.speech_synthesis_voice_name)

        def speak_text_async(self, text):
            if voices is not None:
                voices.append(text)
            return FakeFuture(synthesis_result)

    return SimpleNamespace(
        ResultReason=reasons,
        SpeechConfig=SpeechConfig,
        SpeechRecognizer=SpeechRecognizer,
        SpeechSynthesizer=SpeechSynthesizer,
        SpeechSynthesisOutputFormat=SimpleNamespace(Riff24Khz16BitMonoPcm="wav"),
        languageconfig=SimpleNamespace(
            AutoDetectSourceLanguageConfig=lambda languages: SimpleNamespace(
                languages=languages
            )
        ),
        audio=SimpleNamespace(
            AudioConfig=lambda filename: SimpleNamespace(filename=filename)
        ),
        AutoDetectSourceLanguageResult=lambda result: SimpleNamespace(
            language=result.locale
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("locale", "language", "text"),
    [
        ("en-US", "en", "Show my notifications"),
        ("ar-SA", "ar", "أرني تنبيهاتي"),
        ("ar-SA", "en", "How was my attendance this week?"),
        ("en-US", "ar", "أبغى أعرف حضوري هذا الأسبوع"),
        ("en-US", "ar", "أبغى أشوف attendance حقي هذا الأسبوع"),
    ],
)
async def test_stt_detects_and_normalizes_supported_languages(
    monkeypatch,
    locale: str,
    language: str,
    text: str,
) -> None:
    result = SimpleNamespace(reason="recognized", text=text, locale=locale)
    monkeypatch.setattr(azure_speech, "get_settings", settings)
    monkeypatch.setattr(
        azure_speech,
        "speechsdk",
        fake_sdk(recognition_result=result),
    )

    recognized = await transcribe_audio(b"RIFFdemo", content_type="audio/wav")

    assert recognized.transcript == text
    assert recognized.detected_locale == locale
    assert recognized.detected_language == language
    assert recognized.resolved_locale == ("ar-SA" if language == "ar" else "en-US")
    assert recognized.language_resolution_source.startswith("transcript_")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason", "error_type"),
    [
        ("no-match", SpeechInputError),
        ("canceled", SpeechRecognitionError),
    ],
)
async def test_stt_handles_no_match_and_cancellation(
    monkeypatch,
    reason: str,
    error_type: type[Exception],
) -> None:
    result = SimpleNamespace(reason=reason, text="", locale="en-US")
    monkeypatch.setattr(azure_speech, "get_settings", settings)
    monkeypatch.setattr(
        azure_speech,
        "speechsdk",
        fake_sdk(recognition_result=result),
    )

    with pytest.raises(error_type):
        await transcribe_audio(b"RIFFdemo", content_type="audio/wav")


@pytest.mark.asyncio
async def test_stt_rejects_empty_audio() -> None:
    with pytest.raises(SpeechInputError, match="empty"):
        await transcribe_audio(b"", content_type="audio/wav")


@pytest.mark.asyncio
async def test_stt_hides_azure_failure_and_key(monkeypatch, caplog) -> None:
    class BrokenRecognizer:
        def __init__(self, **kwargs):
            raise RuntimeError("azure-secret-value")

    sdk = fake_sdk(
        recognition_result=SimpleNamespace(
            reason="recognized",
            text="unused",
            locale="en-US",
        )
    )
    sdk.SpeechRecognizer = BrokenRecognizer
    monkeypatch.setattr(azure_speech, "get_settings", settings)
    monkeypatch.setattr(azure_speech, "speechsdk", sdk)

    with pytest.raises(SpeechRecognitionError, match="temporarily unavailable"):
        await transcribe_audio(b"RIFFdemo", content_type="audio/wav")

    assert "azure-secret-value" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("language", "expected_voice"),
    [("en", "en-US-TestVoice"), ("ar", "ar-SA-ZariyahNeural")],
)
async def test_tts_selects_configured_voice(
    monkeypatch,
    language: str,
    expected_voice: str,
) -> None:
    observed: list[str] = []
    result = SimpleNamespace(reason="synthesized", audio_data=b"RIFFaudio")
    monkeypatch.setattr(azure_speech, "get_settings", settings)
    monkeypatch.setattr(
        azure_speech,
        "speechsdk",
        fake_sdk(synthesis_result=result, voices=observed),
    )

    audio = await synthesize_speech("User-facing response", language=language)

    assert observed == [expected_voice, "User-facing response"]
    assert audio.data == b"RIFFaudio"
    assert audio.mime_type == "audio/wav"
    assert audio.locale == ("ar-SA" if language == "ar" else "en-US")
    assert audio.voice_name == expected_voice


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("transcript", "raw_locale", "expected_voice"),
    [
        ("How was my attendance this week?", "ar-SA", "en-US-TestVoice"),
        ("أبغى أعرف حضوري هذا الأسبوع", "en-US", "ar-SA-ZariyahNeural"),
    ],
)
async def test_resolved_transcript_language_selects_tts_voice(
    monkeypatch,
    transcript: str,
    raw_locale: str,
    expected_voice: str,
) -> None:
    observed: list[str] = []
    recognition = SimpleNamespace(
        reason="recognized",
        text=transcript,
        locale=raw_locale,
    )
    synthesis = SimpleNamespace(reason="synthesized", audio_data=b"RIFFaudio")
    sdk = fake_sdk(
        recognition_result=recognition,
        synthesis_result=synthesis,
        voices=observed,
    )
    monkeypatch.setattr(azure_speech, "get_settings", settings)
    monkeypatch.setattr(azure_speech, "speechsdk", sdk)

    recognized = await transcribe_audio(b"RIFFdemo", content_type="audio/wav")
    await synthesize_speech("Assistant answer", language=recognized.detected_language)

    assert observed == [expected_voice, "Assistant answer"]


@pytest.mark.parametrize(
    "voice",
    ["ar-SA-ZariyahNeural", "ar-SA-HamedNeural"],
)
def test_arabic_tts_allows_only_explicit_saudi_voices(voice: str) -> None:
    configured = settings()
    configured.azure_speech_ar_voice = voice

    profile = resolve_tts_profile("ar", configured)

    assert profile.locale == "ar-SA"
    assert profile.voice_name == voice


def test_arabic_tts_never_falls_back_to_english_voice() -> None:
    configured = settings()
    configured.azure_speech_ar_voice = configured.azure_speech_en_voice

    with pytest.raises(SpeechConfigurationError, match="approved ar-SA Saudi voice"):
        resolve_tts_profile("ar", configured)


def test_english_tts_profile_is_unchanged() -> None:
    profile = resolve_tts_profile("en", settings())

    assert profile.locale == "en-US"
    assert profile.voice_name == "en-US-TestVoice"
