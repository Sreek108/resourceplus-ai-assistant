import asyncio
import logging
import threading
from typing import Any

from app.config import get_settings
from app.observability import measure_stage
from app.speech.azure_speech import (
    SpeechConfigurationError,
    SpeechInputError,
    SpeechRecognitionError,
    SpeechTranscript,
    speechsdk,
)
from app.speech.language import resolve_spoken_language


logger = logging.getLogger(__name__)
STREAM_SAMPLE_RATE = 16_000
STREAM_BITS_PER_SAMPLE = 16
STREAM_CHANNELS = 1
FINALIZATION_TIMEOUT_SECONDS = 12


class StreamingSpeechRecognizer:
    """One Azure continuous-recognition session backed by raw PCM push audio."""

    def __init__(self) -> None:
        settings = get_settings()
        if speechsdk is None:
            raise SpeechConfigurationError("Azure Speech support is not installed.")
        if not settings.azure_speech_key or not settings.azure_speech_region:
            raise SpeechConfigurationError("Azure Speech is not configured.")

        self._settings = settings
        self._segments: list[str] = []
        self._locale: str | None = None
        self._error: str | None = None
        self._error_category: str | None = None
        self._stopped = threading.Event()
        self._lock = threading.Lock()
        stream_format = speechsdk.audio.AudioStreamFormat(
            samples_per_second=STREAM_SAMPLE_RATE,
            bits_per_sample=STREAM_BITS_PER_SAMPLE,
            channels=STREAM_CHANNELS,
        )
        self._stream = speechsdk.audio.PushAudioInputStream(
            stream_format=stream_format
        )
        speech_config = speechsdk.SpeechConfig(
            subscription=settings.azure_speech_key,
            region=settings.azure_speech_region,
        )
        auto_detect = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(
            languages=[
                settings.azure_speech_en_locale,
                settings.azure_speech_ar_locale,
            ]
        )
        self._recognizer = speechsdk.SpeechRecognizer(
            speech_config=speech_config,
            auto_detect_source_language_config=auto_detect,
            audio_config=speechsdk.audio.AudioConfig(stream=self._stream),
        )
        self._recognizer.recognized.connect(self._on_recognized)
        self._recognizer.canceled.connect(self._on_canceled)
        self._recognizer.session_stopped.connect(self._on_stopped)
        self._started = False
        self._closed = False

    def _on_recognized(self, event: Any) -> None:
        result = event.result
        if result.reason != speechsdk.ResultReason.RecognizedSpeech:
            return
        text = (result.text or "").strip()
        if not text:
            return
        try:
            locale = speechsdk.AutoDetectSourceLanguageResult(result).language
        except Exception:
            locale = None
        with self._lock:
            self._segments.append(text)
            if locale:
                self._locale = str(locale)

    def _on_canceled(self, event: Any) -> None:
        if getattr(event, "reason", None) == speechsdk.CancellationReason.Error:
            self._error = "Speech recognition was canceled."
            self._error_category = "azure_canceled"
            reason = str(getattr(event, "reason", "unknown")).rsplit(".", 1)[-1]
            logger.warning("Azure streaming recognition canceled: reason=%s", reason[:64])
        self._stopped.set()

    def _on_stopped(self, event: Any) -> None:
        del event
        self._stopped.set()

    async def start(self) -> None:
        try:
            await asyncio.to_thread(
                self._recognizer.start_continuous_recognition_async().get
            )
            self._started = True
        except Exception as exc:
            logger.warning("Azure streaming recognition start failed: %s", type(exc).__name__)
            raise SpeechRecognitionError(
                "Streaming speech recognition is temporarily unavailable.",
                safe_category="speech_recognition_failed",
            ) from exc

    def write(self, pcm_chunk: bytes) -> None:
        if not self._started or self._closed:
            raise SpeechInputError(
                "The streaming speech session is not active.",
                safe_category="invalid_stream_state",
            )
        if not pcm_chunk or len(pcm_chunk) % 2:
            raise SpeechInputError(
                "The PCM audio chunk is malformed.",
                safe_category="invalid_stream_state",
            )
        self._stream.write(pcm_chunk)

    async def finish(self) -> SpeechTranscript:
        if not self._started or self._closed:
            raise SpeechInputError(
                "The streaming speech session is not active.",
                safe_category="invalid_stream_state",
            )
        self._closed = True
        try:
            self._stream.close()
            stopped = await asyncio.to_thread(
                self._stopped.wait,
                FINALIZATION_TIMEOUT_SECONDS,
            )
            await asyncio.to_thread(
                self._recognizer.stop_continuous_recognition_async().get
            )
        except Exception as exc:
            logger.warning(
                "Azure streaming recognition finalization failed: %s",
                type(exc).__name__,
            )
            raise SpeechRecognitionError(
                "Streaming speech recognition did not finish successfully.",
                safe_category="speech_recognition_failed",
            ) from exc
        if not stopped:
            raise SpeechRecognitionError(
                "Streaming speech recognition took too long to finalize.",
                safe_category="stream_finalize_timeout",
            )
        if self._error:
            raise SpeechRecognitionError(
                "Streaming speech recognition is temporarily unavailable.",
                safe_category=self._error_category or "azure_canceled",
            )
        with self._lock:
            transcript = " ".join(self._segments).strip()
            locale = self._locale
        if not transcript:
            raise SpeechInputError(
                "No speech could be recognized from the audio.",
                safe_category="no_recognized_speech",
            )
        raw_locale = locale or self._settings.azure_speech_en_locale
        with measure_stage("language_resolution"):
            resolution = resolve_spoken_language(
                transcript,
                raw_locale,
                english_locale=self._settings.azure_speech_en_locale,
                arabic_locale=self._settings.azure_speech_ar_locale,
            )
        return SpeechTranscript(
            transcript=transcript,
            detected_locale=raw_locale,
            detected_language=resolution.language,
            resolved_locale=resolution.locale,
            language_resolution_source=resolution.source,
        )

    async def cancel(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._stream.close()
            if self._started:
                await asyncio.to_thread(
                    self._recognizer.stop_continuous_recognition_async().get
                )
        except Exception as exc:
            logger.warning("Azure streaming recognition cleanup failed: %s", type(exc).__name__)
