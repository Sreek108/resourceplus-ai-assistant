import asyncio
import inspect
import json
import logging
import os
import re
import tempfile
from dataclasses import dataclass
from typing import Any

from app.config import get_settings
from app.observability import measure_stage
from app.speech.language import (
    analyze_script,
    content_matches_language,
    explicit_language_request,
)


logger = logging.getLogger(__name__)
AUDIO_MIME_TYPE = "audio/wav"
SUPPORTED_INPUT_TYPES = {"audio/wav", "audio/x-wav", "application/octet-stream"}
ARABIC_TTS_LOCALE = "ar-SA"
ALLOWED_ARABIC_TTS_VOICES = frozenset(
    {"ar-SA-ZariyahNeural", "ar-SA-HamedNeural"}
)
GENERIC_HR_PHRASES = (
    "leave",
    "attendance",
    "approval",
    "request",
    "balance",
    "manager",
    "correction",
    "punch",
    "vacation",
    "short hours",
    "missing hours",
    "إجازة",
    "حضور",
    "موافقة",
    "طلب",
    "رصيد",
    "مدير",
    "تصحيح",
    "بصمة",
    "ساعات ناقصة",
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
    stt_confidence: float | None = None
    stt_primary_locale: str | None = None
    stt_primary_confidence: float | None = None
    stt_fallback_used: bool = False
    stt_fallback_locale: str | None = None
    stt_fallback_confidence: float | None = None
    stt_selection_reason: str | None = None
    transcript_script_class: str | None = None


@dataclass(frozen=True)
class RecognitionCandidate:
    transcript: str
    locale: str
    confidence: float | None
    success: bool
    result_status: str
    script_class: str

    @property
    def language(self) -> str:
        return "ar" if self.locale.casefold().startswith("ar") else "en"


@dataclass(frozen=True)
class CandidateRecognition:
    selected: RecognitionCandidate
    auto: RecognitionCandidate
    english: RecognitionCandidate | None
    arabic: RecognitionCandidate | None
    primary_locale: str
    fallback_used: bool
    fallback_locale: str | None
    selection_reason: str


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


def speech_result_confidence(result: object, sdk: object | None = None) -> float | None:
    """Read Azure confidence when detailed result metadata is available."""

    direct = getattr(result, "confidence", None)
    if isinstance(direct, (int, float)) and 0 <= float(direct) <= 1:
        return float(direct)
    try:
        property_id = getattr(
            getattr(sdk, "PropertyId"),
            "SpeechServiceResponse_JsonResult",
        )
        raw = result.properties.get(property_id)
        payload = json.loads(raw)
        confidence = payload["NBest"][0]["Confidence"]
    except (AttributeError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None
    return float(confidence) if isinstance(confidence, (int, float)) else None


def _enable_detailed_output(sdk: object, speech_config: object) -> None:
    detailed = getattr(getattr(sdk, "OutputFormat", None), "Detailed", None)
    if detailed is not None and hasattr(speech_config, "output_format"):
        speech_config.output_format = detailed


def _apply_phrase_hints(
    sdk: object,
    recognizer: object,
    phrase_hints: tuple[str, ...] = (),
) -> None:
    """Bias recognition with generic HR terms and current live DayTypes."""

    grammar_type = getattr(sdk, "PhraseListGrammar", None)
    if grammar_type is None or not hasattr(grammar_type, "from_recognizer"):
        return
    grammar = grammar_type.from_recognizer(recognizer)
    for phrase in dict.fromkeys((*GENERIC_HR_PHRASES, *phrase_hints)):
        cleaned = " ".join(str(phrase).split())
        if cleaned:
            grammar.addPhrase(cleaned)


def _require_sdk_and_settings() -> tuple[Any, Any]:
    settings = get_settings()
    if speechsdk is None:
        raise SpeechConfigurationError("Azure Speech support is not installed.")
    if not settings.azure_speech_key or not settings.azure_speech_region:
        raise SpeechConfigurationError("Azure Speech is not configured.")
    return speechsdk, settings


def _recognize_candidate_sync(
    audio: bytes,
    *,
    locale: str | None,
    phrase_hints: tuple[str, ...] = (),
) -> RecognitionCandidate:
    sdk, settings = _require_sdk_and_settings()
    descriptor, path = tempfile.mkstemp(suffix=".wav")
    try:
        with os.fdopen(descriptor, "wb") as audio_file:
            audio_file.write(audio)
        speech_config = sdk.SpeechConfig(
            subscription=settings.azure_speech_key,
            region=settings.azure_speech_region,
        )
        _enable_detailed_output(sdk, speech_config)
        audio_config = sdk.audio.AudioConfig(filename=path)
        if locale is None:
            auto_detect = sdk.languageconfig.AutoDetectSourceLanguageConfig(
                languages=[
                    settings.azure_speech_en_locale,
                    settings.azure_speech_ar_locale,
                ]
            )
            recognizer = sdk.SpeechRecognizer(
                speech_config=speech_config,
                auto_detect_source_language_config=auto_detect,
                audio_config=audio_config,
            )
        else:
            speech_config.speech_recognition_language = locale
            recognizer = sdk.SpeechRecognizer(
                speech_config=speech_config,
                audio_config=audio_config,
            )
        _apply_phrase_hints(sdk, recognizer, phrase_hints)
        result = recognizer.recognize_once_async().get()
        if result.reason == sdk.ResultReason.NoMatch:
            return RecognitionCandidate(
                "", locale or "und", None, False, "no_match", "script_neutral"
            )
        if result.reason == sdk.ResultReason.Canceled:
            return RecognitionCandidate(
                "", locale or "und", None, False, "cancelled", "script_neutral"
            )
        if result.reason != sdk.ResultReason.RecognizedSpeech:
            return RecognitionCandidate(
                "", locale or "und", None, False, "failed", "script_neutral"
            )
        transcript = (result.text or "").strip()
        if not transcript:
            return RecognitionCandidate(
                "", locale or "und", None, False, "empty", "script_neutral"
            )
        resolved_locale = locale
        if resolved_locale is None:
            resolved_locale = str(sdk.AutoDetectSourceLanguageResult(result).language)
        return RecognitionCandidate(
            transcript,
            str(resolved_locale),
            speech_result_confidence(result, sdk),
            True,
            "recognized",
            analyze_script(transcript).classification,
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


def _invoke_candidate_sync(
    audio: bytes,
    *,
    locale: str | None,
    phrase_hints: tuple[str, ...],
) -> RecognitionCandidate:
    """Keep injected recognizers/test doubles compatible with phrase hints."""

    parameters = inspect.signature(_recognize_candidate_sync).parameters.values()
    supports_hints = any(
        parameter.name == "phrase_hints"
        or parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )
    if supports_hints:
        return _recognize_candidate_sync(
            audio,
            locale=locale,
            phrase_hints=phrase_hints,
        )
    return _recognize_candidate_sync(audio, locale=locale)


def _script_consistent(candidate: RecognitionCandidate) -> bool:
    expected = "primarily_arabic" if candidate.language == "ar" else "primarily_latin"
    return candidate.script_class == expected


def _strong_candidate(candidate: RecognitionCandidate) -> bool:
    if not candidate.success or not _script_consistent(candidate):
        return False
    if candidate.confidence is not None:
        return candidate.confidence >= 0.65
    letters = analyze_script(candidate.transcript)
    return letters.arabic_letters + letters.latin_letters >= 2


def _candidate_score(
    candidate: RecognitionCandidate,
    *,
    established_language: str | None,
    auto_locale: str,
) -> float:
    if not candidate.success:
        return -100.0
    score = 10.0
    if candidate.confidence is not None:
        score += candidate.confidence * 5
    if _script_consistent(candidate):
        score += 3
    elif candidate.script_class == "mixed":
        score -= 1
        script = analyze_script(candidate.transcript)
        if (
            candidate.language == "ar" and script.arabic_letters > script.latin_letters
        ) or (
            candidate.language == "en" and script.latin_letters > script.arabic_letters
        ):
            score += 2
    elif candidate.script_class != "script_neutral":
        score -= 4
    requested = explicit_language_request(candidate.transcript)
    if requested == candidate.language:
        score += 8
    elif requested is not None:
        score -= 6
    if established_language == candidate.language:
        score += 1.5
    if auto_locale.casefold().startswith(candidate.language):
        score += 0.25
    normalized = " ".join(candidate.transcript.casefold().strip(" .?!،؟").split())
    if candidate.language == "en" and normalized in {"hello", "hi", "hey", "hello there"}:
        score += 2
    if candidate.language == "ar" and normalized in {"هالو", "هالي", "هاي"}:
        score -= 1
    return score


async def recognize_audio_candidates(
    audio: bytes,
    *,
    established_language: str | None = None,
    auto_candidate: RecognitionCandidate | None = None,
    force_fallback: bool = False,
    phrase_hints: tuple[str, ...] = (),
) -> CandidateRecognition:
    """Select a transcript from fixed-locale candidates for the same audio."""

    settings = get_settings()
    en_locale = settings.azure_speech_en_locale
    ar_locale = settings.azure_speech_ar_locale
    primary_locale = ar_locale if established_language == "ar" else en_locale
    alternate_locale = ar_locale if primary_locale == en_locale else en_locale
    fallback: RecognitionCandidate | None = None
    first_turn = established_language is None
    if auto_candidate is None and (first_turn or force_fallback):
        auto_candidate, primary, fallback = await asyncio.gather(
            asyncio.to_thread(
                _invoke_candidate_sync, audio, locale=None, phrase_hints=phrase_hints
            ),
            asyncio.to_thread(
                _invoke_candidate_sync,
                audio,
                locale=primary_locale,
                phrase_hints=phrase_hints,
            ),
            asyncio.to_thread(
                _invoke_candidate_sync,
                audio,
                locale=alternate_locale,
                phrase_hints=phrase_hints,
            ),
        )
    elif auto_candidate is None:
        auto_candidate, primary = await asyncio.gather(
            asyncio.to_thread(
                _invoke_candidate_sync, audio, locale=None, phrase_hints=phrase_hints
            ),
            asyncio.to_thread(
                _invoke_candidate_sync,
                audio,
                locale=primary_locale,
                phrase_hints=phrase_hints,
            ),
        )
    elif first_turn or force_fallback:
        primary, fallback = await asyncio.gather(
            asyncio.to_thread(
                _invoke_candidate_sync,
                audio,
                locale=primary_locale,
                phrase_hints=phrase_hints,
            ),
            asyncio.to_thread(
                _invoke_candidate_sync,
                audio,
                locale=alternate_locale,
                phrase_hints=phrase_hints,
            ),
        )
    else:
        primary = await asyncio.to_thread(
            _invoke_candidate_sync,
            audio,
            locale=primary_locale,
            phrase_hints=phrase_hints,
        )
    auto_locale = auto_candidate.locale
    explicit_other = any(
        requested not in {None, primary.language}
        for requested in (
            explicit_language_request(auto_candidate.transcript),
            explicit_language_request(primary.transcript),
        )
    )
    fallback_needed = (
        force_fallback
        or established_language is None
        or not _strong_candidate(primary)
        or explicit_other
        or primary.script_class == "mixed"
    )
    if fallback_needed and fallback is None:
        fallback_locale = ar_locale if primary.language == "en" else en_locale
        fallback = await asyncio.to_thread(
            _invoke_candidate_sync,
            audio,
            locale=fallback_locale,
            phrase_hints=phrase_hints,
        )
    candidates = [
        candidate for candidate in (primary, fallback) if candidate is not None
    ]
    if not any(candidate.success for candidate in candidates):
        if auto_candidate.success:
            selected = auto_candidate
            reason = "fixed_candidates_failed_auto_selected"
        else:
            if any(
                candidate.result_status == "cancelled"
                for candidate in (auto_candidate, *candidates)
            ):
                raise SpeechRecognitionError(
                    "Speech recognition is temporarily unavailable. Please try again.",
                    safe_category="azure_canceled",
                )
            raise SpeechInputError(
                "No speech could be recognized from the audio.",
                safe_category="fallback_stt_no_match",
            )
    else:
        selected = max(
            candidates,
            key=lambda item: _candidate_score(
                item,
                established_language=established_language,
                auto_locale=auto_locale,
            ),
        )
        if fallback is None:
            reason = "confident_session_primary"
        elif selected is primary:
            reason = "primary_outscored_fallback"
        else:
            reason = "fallback_outscored_primary"
    english = primary if primary.language == "en" else fallback
    arabic = primary if primary.language == "ar" else fallback
    return CandidateRecognition(
        selected=selected,
        auto=auto_candidate,
        english=english,
        arabic=arabic,
        primary_locale=primary_locale,
        fallback_used=fallback is not None,
        fallback_locale=fallback.locale if fallback is not None else None,
        selection_reason=reason,
    )


async def transcribe_audio(
    audio: bytes,
    *,
    content_type: str | None = AUDIO_MIME_TYPE,
    established_language: str | None = None,
    phrase_hints: tuple[str, ...] = (),
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
        recognition = await recognize_audio_candidates(
            audio,
            established_language=established_language,
            phrase_hints=phrase_hints,
        )
    selected = recognition.selected
    fallback = (
        recognition.arabic
        if recognition.primary_locale.casefold().startswith("en")
        else recognition.english
    )
    return SpeechTranscript(
        transcript=selected.transcript,
        detected_locale=recognition.auto.locale,
        detected_language=selected.language,
        resolved_locale=selected.locale,
        language_resolution_source="candidate_recognition",
        stt_confidence=selected.confidence,
        stt_primary_locale=recognition.primary_locale,
        stt_primary_confidence=(
            recognition.english.confidence
            if recognition.primary_locale.casefold().startswith("en")
            and recognition.english is not None
            else recognition.arabic.confidence if recognition.arabic is not None else None
        ),
        stt_fallback_used=recognition.fallback_used,
        stt_fallback_locale=recognition.fallback_locale,
        stt_fallback_confidence=fallback.confidence if fallback is not None else None,
        stt_selection_reason=recognition.selection_reason,
        transcript_script_class=selected.script_class,
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
    if not content_matches_language(text, language):
        raise SpeechSynthesisError(
            "The response text does not match the selected speech language."
        )
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
            category = "cancelled"
            safe_code = "unavailable"
            try:
                details = sdk.SpeechSynthesisCancellationDetails.from_result(result)
                raw_code = str(getattr(details, "error_code", "")).casefold()
                raw_reason = str(getattr(details, "reason", "")).casefold()
                safe_code = re.sub(r"[^a-z0-9_]+", "_", raw_code).strip("_")[:64] or "unavailable"
                if "authentication" in raw_code or "forbidden" in raw_code:
                    category = "auth_config"
                elif "timeout" in raw_code or "timeout" in raw_reason:
                    category = "timeout"
                elif any(cue in raw_code for cue in ("connection", "network")):
                    category = "network"
                elif any(cue in raw_code for cue in ("service", "too_many", "quota")):
                    category = "service_unavailable"
            except Exception:
                pass
            logger.warning(
                "Azure speech synthesis cancelled: category=%s code=%s",
                category,
                safe_code,
            )
            raise SpeechSynthesisError(
                "Speech synthesis is temporarily unavailable. Please try again.",
                safe_category=category,
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
    settings = get_settings()
    attempts = 2 if getattr(settings, "azure_tts_transient_retry", True) else 1
    transient = {"timeout", "network", "service_unavailable"}
    for attempt in range(attempts):
        try:
            return await asyncio.wait_for(
                asyncio.to_thread(_synthesize_sync, normalized, language),
                timeout=getattr(settings, "azure_tts_timeout_seconds", 20.0),
            )
        except asyncio.TimeoutError as exc:
            error = SpeechSynthesisError(
                "Speech synthesis timed out.",
                safe_category="timeout",
            )
            logger.warning("Azure speech synthesis failed: category=timeout")
            if attempt + 1 >= attempts:
                raise error from exc
        except SpeechSynthesisError as exc:
            logger.warning(
                "Azure speech synthesis failed: category=%s",
                exc.safe_category,
            )
            if exc.safe_category not in transient or attempt + 1 >= attempts:
                raise
    raise SpeechSynthesisError("Speech synthesis is temporarily unavailable.")
    default_safe_category = "voice_unavailable"
    default_safe_category = "speech_recognition_failed"
    default_safe_category = "speech_synthesis_failed"
