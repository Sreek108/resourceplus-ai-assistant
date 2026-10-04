from types import SimpleNamespace

import pytest

from app.speech import azure_speech
from app.speech.azure_speech import (
    RecognitionCandidate,
    SpeechConfigurationError,
    SpeechInputError,
    SpeechRecognitionError,
    resolve_tts_profile,
    recognize_audio_candidates,
    speech_result_confidence,
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
    assert recognized.language_resolution_source == "candidate_recognition"


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
async def test_stt_preserves_confidence_when_provider_exposes_it(monkeypatch) -> None:
    result = SimpleNamespace(
        reason="recognized",
        text="Show my attendance",
        locale="en-US",
        confidence=0.91,
    )
    monkeypatch.setattr(azure_speech, "get_settings", settings)
    monkeypatch.setattr(
        azure_speech,
        "speechsdk",
        fake_sdk(recognition_result=result),
    )

    recognized = await transcribe_audio(b"RIFFdemo", content_type="audio/wav")

    assert recognized.stt_confidence == 0.91


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

    response_text = "رد للمستخدم" if language == "ar" else "User-facing response"
    audio = await synthesize_speech(response_text, language=language)

    assert observed == [expected_voice, response_text]
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
    answer = (
        "هذا رد المساعد"
        if recognized.detected_language == "ar"
        else "Assistant answer"
    )
    await synthesize_speech(answer, language=recognized.detected_language)

    assert observed == [expected_voice, answer]


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


@pytest.mark.asyncio
async def test_first_turn_candidates_recover_english_hello_from_arabic_autodetect(
    monkeypatch,
) -> None:
    candidates = {
        None: RecognitionCandidate(
            "هالو", "ar-SA", 0.82, True, "recognized", "primarily_arabic"
        ),
        "en-US": RecognitionCandidate(
            "Hello", "en-US", 0.94, True, "recognized", "primarily_latin"
        ),
        "ar-SA": RecognitionCandidate(
            "هالو", "ar-SA", 0.84, True, "recognized", "primarily_arabic"
        ),
    }
    monkeypatch.setattr(
        azure_speech,
        "_recognize_candidate_sync",
        lambda audio, *, locale: candidates[locale],
    )
    monkeypatch.setattr(azure_speech, "get_settings", settings)

    result = await recognize_audio_candidates(b"same buffered audio")

    assert result.selected.transcript == "Hello"
    assert result.selected.language == "en"
    assert result.fallback_used is True
    assert result.selection_reason == "primary_outscored_fallback"


@pytest.mark.asyncio
async def test_confident_established_primary_skips_fixed_locale_fallback(
    monkeypatch,
) -> None:
    calls: list[str | None] = []

    def recognize(audio, *, locale):
        calls.append(locale)
        if locale is None:
            return RecognitionCandidate(
                "هالو", "ar-SA", 0.7, True, "recognized", "primarily_arabic"
            )
        return RecognitionCandidate(
            "Hi", "en-US", 0.93, True, "recognized", "primarily_latin"
        )

    monkeypatch.setattr(azure_speech, "_recognize_candidate_sync", recognize)
    monkeypatch.setattr(azure_speech, "get_settings", settings)

    result = await recognize_audio_candidates(
        b"same buffered audio",
        established_language="en",
    )

    assert result.selected.transcript == "Hi"
    assert result.fallback_used is False
    assert calls == [None, "en-US"]


@pytest.mark.asyncio
async def test_confident_established_arabic_primary_survives_english_autodetect(
    monkeypatch,
) -> None:
    candidates = {
        None: RecognitionCandidate(
            "Hello", "en-US", 0.71, True, "recognized", "primarily_latin"
        ),
        "ar-SA": RecognitionCandidate(
            "هلا", "ar-SA", 0.91, True, "recognized", "primarily_arabic"
        ),
    }
    monkeypatch.setattr(
        azure_speech,
        "_recognize_candidate_sync",
        lambda audio, *, locale: candidates[locale],
    )
    monkeypatch.setattr(azure_speech, "get_settings", settings)

    result = await recognize_audio_candidates(
        b"same buffered audio",
        established_language="ar",
    )

    assert result.selected.transcript == "هلا"
    assert result.selected.language == "ar"
    assert result.fallback_used is False


@pytest.mark.asyncio
async def test_mixed_primary_transcript_triggers_alternate_locale_candidate(
    monkeypatch,
) -> None:
    candidates = {
        None: RecognitionCandidate(
            "Hay resource بلس.", "ar-SA", 0.72, True, "recognized", "mixed"
        ),
        "en-US": RecognitionCandidate(
            "Hay resource بلس.", "en-US", 0.7, True, "recognized", "mixed"
        ),
        "ar-SA": RecognitionCandidate(
            "هاي ريسورس بلس.", "ar-SA", 0.66, True, "recognized", "primarily_arabic"
        ),
    }
    monkeypatch.setattr(
        azure_speech,
        "_recognize_candidate_sync",
        lambda audio, *, locale: candidates[locale],
    )
    monkeypatch.setattr(azure_speech, "get_settings", settings)

    result = await recognize_audio_candidates(
        b"same buffered audio",
        established_language="en",
    )

    assert result.fallback_used is True
    assert result.arabic is not None
    assert result.auto.script_class == "mixed"


def test_detailed_nbest_confidence_uses_sdk_json_property() -> None:
    property_id = object()
    sdk = SimpleNamespace(
        PropertyId=SimpleNamespace(SpeechServiceResponse_JsonResult=property_id)
    )
    result = SimpleNamespace(
        properties={property_id: '{"NBest":[{"Confidence":0.87}]}'},
    )

    assert speech_result_confidence(result, sdk) == 0.87


def test_phrase_hints_combine_generic_hr_terms_with_live_day_types() -> None:
    captured: list[str] = []

    class Grammar:
        @classmethod
        def from_recognizer(cls, recognizer):
            assert recognizer == "recognizer"
            return cls()

        def addPhrase(self, phrase):
            captured.append(phrase)

    sdk = SimpleNamespace(PhraseListGrammar=Grammar)
    azure_speech._apply_phrase_hints(
        sdk,
        "recognizer",
        ("Dynamic Volunteer Day", "إجازة تطوع"),
    )

    assert "attendance" in captured
    assert "missing hours" in captured
    assert "Dynamic Volunteer Day" in captured
    assert "إجازة تطوع" in captured
    assert all("employee" not in phrase.casefold() for phrase in captured)


@pytest.mark.asyncio
async def test_tts_rejects_clear_content_voice_mismatch(monkeypatch) -> None:
    monkeypatch.setattr(azure_speech, "get_settings", settings)
    monkeypatch.setattr(
        azure_speech,
        "speechsdk",
        fake_sdk(
            synthesis_result=SimpleNamespace(
                reason="synthesized", audio_data=b"must-not-be-used"
            )
        ),
    )

    with pytest.raises(azure_speech.SpeechSynthesisError):
        await synthesize_speech("هذا رد عربي", language="en")
