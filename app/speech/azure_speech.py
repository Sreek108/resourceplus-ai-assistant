import asyncio
import logging
import os
import tempfile
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.observability import measure_stage
from app.speech.language import resolve_spoken_language


logger = logging.getLogger(__name__)
AUDIO_MIME_TYPE = "audio/wav"
SUPPORTED_INPUT_TYPES = {"audio/wav", "audio/x-wav", "application/octet-stream"}
ARABIC_TTS_LOCALE = "ar-SA"
ALLOWED_ARABIC_TTS_VOICES = frozenset(
    {"ar-SA-ZariyahNeural", "ar-SA-HamedNeural"}
)

try:
    import azure.cognitiveservices.speech as speechsdk
except ImportError:  # pragma: no cover - exercised through the safe config path
    speechsdk = None


class SpeechServiceError(Exception):
    """Base class for safe speech-service failures."""

    default_safe_category = "unknown_safe_category"

    def __init__(self, message: str, *, safe_category: str | None = None) -> None:
        super().__init__(message)
        self.safe_category = safe_category or self.default_safe_category


class SpeechInputError(SpeechServiceError):
    """The uploaded audio cannot be recognized safely."""

    default_safe_category = "fallback_invalid_audio"


class SpeechConfigurationError(SpeechServiceError):
    """Azure Speech configuration or SDK is unavailable."""

    default_safe_category = "voice_unavailable"


class SpeechRecognitionError(SpeechServiceError):
    """Azure Speech failed to recognize the upload."""

    default_safe_category = "speech_recognition_failed"


class SpeechSynthesisError(SpeechServiceError):
    """Azure Speech failed to synthesize a response."""

    default_safe_category = "speech_synthesis_failed"


@dataclass(frozen=True)
class SpeechTranscript:
    transcript: str
    detected_locale: str
    detected_language: str
    resolved_locale: str | None = None
    language_resolution_source: str | None = None


@dataclass(frozen=True)
class SpeechAudio:
    data: bytes
    mime_type: str = AUDIO_MIME_TYPE
    locale: str | None = None
    voice_name: str | None = None


@dataclass(frozen=True)
class TTSProfile:
    locale: str
    voice_name: str


def _require_sdk_and_settings() -> tuple[Any, Any]:
    settings = get_settings()
    if speechsdk is None:
        raise SpeechConfigurationError("Azure Speech support is not installed.")
    if not settings.azure_speech_key or not settings.azure_speech_region:
        raise SpeechConfigurationError("Azure Speech is not configured.")
    return speechsdk, settings


def _recognize_sync(audio: bytes) -> tuple[str, str, str, str]:
    sdk, settings = _require_sdk_and_settings()
    descriptor, path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(descriptor, "wb") as audio_file:
            audio_file.write(audio)
        speech_config = sdk.SpeechConfig(
            subscription=settings.azure_speech_key,
            region=settings.azure_speech_region,
        )
        auto_detect = sdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=[
                settings.azure_speech_en_locale,
                settings.azure_speech_ar_locale,
            ]
        )
        recognizer = sdk.SpeechRecognizer(
            speech_config=speech_config,
            auto_detect_source_language_config=auto_detect,
            audio_config=sdk.audio.AudioConfig(filename=path),
        )
        result = recognizer.recognize_once_async().get()
        if result.reason == sdk.ResultReason.NoMatch:
            raise SpeechInputError(
                "No speech could be recognized from the audio.",
                safe_category="fallback_stt_no_match",
            )
        if result.reason == sdk.ResultReason.Canceled:
            raise SpeechRecognitionError(
                "Speech recognition is temporarily unavailable. Please try again.",
                safe_category="azure_canceled",
            )
        if result.reason != sdk.ResultReason.RecognizedSpeech:
            raise SpeechRecognitionError(
                "Speech recognition did not complete. Please try again."
            )
        transcript = (result.text or "").strip()
        if not transcript:
            raise SpeechInputError(
                "No speech could be recognized from the audio.",
                safe_category="fallback_stt_no_match",
            )
        locale = sdk.AutoDetectSourceLanguageResult(result).language
        return (
            transcript,
            str(locale),
            settings.azure_speech_en_locale,
            settings.azure_speech_ar_locale,
        )
    except SpeechServiceError:
        raise
    except Exception as exc:
        logger.warning("Azure speech recognition failed: %s", type(exc).__name__)
        raise SpeechRecognitionError(
            "Speech recognition is temporarily unavailable. Please try again."
        ) from exc
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


async def transcribe_audio(
    audio: bytes,
    *,
    content_type: str | None = AUDIO_MIME_TYPE,
) -> SpeechTranscript:
    if not audio:
        raise SpeechInputError(
            "The uploaded audio is empty.",
            safe_category="fallback_invalid_audio",
        )
    normalized_type = (content_type or "").split(";", 1)[0].strip().casefold()
    if normalized_type not in SUPPORTED_INPUT_TYPES:
        raise SpeechInputError(
            "The demo currently accepts WAV audio only.",
            safe_category="fallback_invalid_audio",
        )
    with measure_stage("stt"):
        transcript, locale, english_locale, arabic_locale = await asyncio.to_thread(
            _recognize_sync,
            audio,
        )
    with measure_stage("language_resolution"):
        resolution = resolve_spoken_language(
            transcript,
            locale,
            english_locale=english_locale,
            arabic_locale=arabic_locale,
        )
    return SpeechTranscript(
        transcript=transcript,
        detected_locale=locale,
        detected_language=resolution.language,
        resolved_locale=resolution.locale,
        language_resolution_source=resolution.source,
    )


def resolve_tts_profile(language: str, settings: Any | None = None) -> TTSProfile:
    settings = settings or get_settings()
    if language == "ar":
        locale = ARABIC_TTS_LOCALE
        voice = settings.azure_speech_ar_voice
        if voice not in ALLOWED_ARABIC_TTS_VOICES:
            raise SpeechConfigurationError(
                "Arabic text-to-speech must use an approved ar-SA Saudi voice."
            )
    elif language == "en":
        locale = settings.azure_speech_en_locale
        voice = settings.azure_speech_en_voice
    else:
        raise SpeechInputError("The response language is not supported for speech.")
    if not voice:
        raise SpeechConfigurationError(
            f"Azure {locale} text-to-speech voice is not configured."
        )
    return TTSProfile(locale=locale, voice_name=voice)


def _synthesize_sync(text: str, language: str) -> SpeechAudio:
    sdk, settings = _require_sdk_and_settings()
    profile = resolve_tts_profile(language, settings)

    try:
        speech_config = sdk.SpeechConfig(
            subscription=settings.azure_speech_key,
            region=settings.azure_speech_region,
        )
        speech_config.speech_synthesis_voice_name = profile.voice_name
        speech_config.set_speech_synthesis_output_format(
            sdk.SpeechSynthesisOutputFormat.Riff24Khz16BitMonoPcm
        )
        synthesizer = sdk.SpeechSynthesizer(
            speech_config=speech_config,
            audio_config=None,
        )
        result = synthesizer.speak_text_async(text).get()
        if result.reason != sdk.ResultReason.SynthesizingAudioCompleted:
            raise SpeechSynthesisError(
                "Speech synthesis is temporarily unavailable. Please try again."
            )
        audio_data = bytes(result.audio_data)
        if not audio_data:
            raise SpeechSynthesisError(
                "Speech synthesis returned no audio. Please try again."
            )
        return SpeechAudio(
            audio_data,
            locale=profile.locale,
            voice_name=profile.voice_name,
        )
    except SpeechServiceError:
        raise
    except Exception as exc:
        logger.warning("Azure speech synthesis failed: %s", type(exc).__name__)
        raise SpeechSynthesisError(
            "Speech synthesis is temporarily unavailable. Please try again."
        ) from exc


async def synthesize_speech(text: str, *, language: str) -> SpeechAudio:
    normalized = text.strip()
    if not normalized:
        raise SpeechInputError("The assistant response is empty.")
    return await asyncio.to_thread(_synthesize_sync, normalized, language)
    default_safe_category = "voice_unavailable"
    default_safe_category = "speech_recognition_failed"
    default_safe_category = "speech_synthesis_failed"
