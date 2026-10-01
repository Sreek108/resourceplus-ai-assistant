import logging
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Iterator

from app.audit import (
    current_interaction_audit,
    record_latency,
    record_model_request,
    record_safe_error,
    safe_error_category,
)
from app.telemetry import (
    classify_error,
    emit_structured_event,
    record_model_metrics,
    record_speech_metrics,
)


logger = logging.getLogger(__name__)

SAFE_VOICE_ERROR_CATEGORIES = frozenset(
    {
        "none",
        "no_audio",
        "no_recognized_speech",
        "azure_canceled",
        "websocket_disconnected",
        "invalid_stream_state",
        "stream_finalize_timeout",
        "client_aborted",
        "fallback_invalid_audio",
        "fallback_stt_no_match",
        "speech_recognition_failed",
        "speech_synthesis_failed",
        "timeout",
        "cancelled",
        "network",
        "auth_config",
        "service_unavailable",
        "voice_unavailable",
        "unknown_safe_category",
    }
)

STAGE_NAMES = frozenset(
    {
        "audio_receive",
        "stt",
        "language_resolution",
        "agent",
        "router",
        "openai_main",
        "follow_up_classifier",
        "confirmation_classifier",
        "response_renderer",
        "resourceplus",
        "voice_summary",
        "speech_normalization",
        "tts",
        "response_build",
        "response_serialization",
        "post_release_stt_finalize",
        "response_send",
        "audit_persist",
    }
)
TOP_LEVEL_STAGES = (
    "audio_receive",
    "stt",
    "language_resolution",
    "agent",
    "voice_summary",
    "speech_normalization",
    "tts",
    "response_build",
    "response_serialization",
    "audit_persist",
)


@dataclass
class VoiceLatencyTrace:
    """Request-local duration accumulator containing no request or employee data."""

    started_at: float = field(default_factory=time.perf_counter)
    durations: dict[str, float] = field(default_factory=dict)
    model_requests: int = 0
    route_completed_at: float | None = None
    first_audio_at: float | None = None
    released_at: float | None = None
    response_sent_at: float | None = None
    error_category: str = "none"

    def add_duration(self, stage: str, duration: float) -> None:
        if stage not in STAGE_NAMES:
            raise ValueError(f"Unsupported voice latency stage: {stage}")
        self.durations[stage] = self.durations.get(stage, 0.0) + max(duration, 0.0)

    def mark_route_complete(self) -> None:
        self.route_completed_at = time.perf_counter()

    def finish_response_serialization(self) -> None:
        if self.route_completed_at is not None:
            self.add_duration(
                "response_serialization",
                time.perf_counter() - self.route_completed_at,
            )

    def mark_audio_chunk(self) -> None:
        if self.first_audio_at is None:
            self.first_audio_at = time.perf_counter()

    def mark_release(self) -> None:
        now = time.perf_counter()
        self.released_at = now
        if self.first_audio_at is not None:
            self.durations["audio_stream_duration"] = max(
                now - self.first_audio_at,
                0.0,
            )

    def mark_response_sent(self) -> None:
        self.response_sent_at = time.perf_counter()

    def set_error_category(self, category: str | None) -> None:
        self.error_category = (
            category if category in SAFE_VOICE_ERROR_CATEGORIES else "unknown_safe_category"
        )

    def format_summary(self, *, status_code: int) -> str:
        total = max(time.perf_counter() - self.started_at, 0.0)
        accounted = sum(self.durations.get(stage, 0.0) for stage in TOP_LEVEL_STAGES)
        other = max(total - accounted, 0.0)
        fields = [
            "VOICE_LATENCY",
            f"status={status_code}",
            f"stream_error_category={self.error_category}",
            f"total={total:.3f}s",
            *(f"{stage}={self.durations.get(stage, 0.0):.3f}s" for stage in TOP_LEVEL_STAGES),
            f"openai_main={self.durations.get('openai_main', 0.0):.3f}s",
            f"follow_up_classifier={self.durations.get('follow_up_classifier', 0.0):.3f}s",
            f"confirmation_classifier={self.durations.get('confirmation_classifier', 0.0):.3f}s",
            f"response_renderer={self.durations.get('response_renderer', 0.0):.3f}s",
            f"resourceplus={self.durations.get('resourceplus', 0.0):.3f}s",
            f"model_requests={self.model_requests}",
            f"stream_session_total={total:.3f}s",
            f"audio_stream_duration={self.durations.get('audio_stream_duration', 0.0):.3f}s",
            f"post_release_stt_finalize={self.durations.get('post_release_stt_finalize', 0.0):.3f}s",
            f"response_send={self.durations.get('response_send', 0.0):.3f}s",
            f"audit_persist={self.durations.get('audit_persist', 0.0):.3f}s",
            f"total_after_release={max((self.response_sent_at or time.perf_counter()) - self.released_at, 0.0) if self.released_at is not None else 0.0:.3f}s",
            f"other={other:.3f}s",
        ]
        return " ".join(fields)

    def snapshot(self) -> dict[str, float]:
        snapshot = {key: round(value, 6) for key, value in self.durations.items()}
        snapshot["total"] = round(max(time.perf_counter() - self.started_at, 0.0), 6)
        if self.released_at is not None:
            snapshot["total_after_release"] = round(
                max(
                    (self.response_sent_at or time.perf_counter()) - self.released_at,
                    0.0,
                ),
                6,
            )
        return snapshot


_current_voice_trace: ContextVar[VoiceLatencyTrace | None] = ContextVar(
    "current_voice_latency_trace",
    default=None,
)


def start_voice_trace() -> tuple[VoiceLatencyTrace, Token[VoiceLatencyTrace | None]]:
    trace = VoiceLatencyTrace()
    return trace, _current_voice_trace.set(trace)


def reset_voice_trace(token: Token[VoiceLatencyTrace | None]) -> None:
    _current_voice_trace.reset(token)


def current_voice_trace() -> VoiceLatencyTrace | None:
    return _current_voice_trace.get()


@contextmanager
def measure_stage(stage: str) -> Iterator[None]:
    if stage not in STAGE_NAMES:
        raise ValueError(f"Unsupported voice latency stage: {stage}")
    trace = current_voice_trace()
    audit = current_interaction_audit()
    if trace is None and audit is None:
        yield
        return
    started_at = time.perf_counter()
    succeeded = False
    error_category = None
    try:
        yield
        succeeded = True
    except BaseException as exc:
        if isinstance(exc, Exception):
            error_category = safe_error_category(exc)
        else:
            error_category = "unknown_safe_category"
        if audit is not None and audit.error_category is None:
            record_safe_error(error_category, stage=stage)
        raise
    finally:
        duration = time.perf_counter() - started_at
        if trace is not None:
            trace.add_duration(stage, duration)
        record_latency(stage, duration)
        owner, error_stage = classify_error(error_category, stage=stage)
        emit_structured_event(
            component="latency",
            event=f"{stage}_completed",
            duration_ms=duration * 1000,
            status="success" if succeeded else "failure",
            error_category=error_category,
            error_owner=None if succeeded else owner,
            error_stage=None if succeeded else error_stage,
            retryable=None if succeeded else True,
        )
        if stage in {"stt", "post_release_stt_finalize"}:
            record_speech_metrics(
                service="stt",
                duration_ms=duration * 1000,
                success=succeeded,
            )
        elif stage == "tts":
            record_speech_metrics(
                service="tts",
                duration_ms=duration * 1000,
                success=succeeded,
            )


@contextmanager
def measure_model_call(stage: str) -> Iterator[None]:
    if stage not in {
        "openai_main",
        "follow_up_classifier",
        "confirmation_classifier",
        "response_renderer",
    }:
        raise ValueError(f"Unsupported OpenAI latency stage: {stage}")
    trace = current_voice_trace()
    if trace is not None:
        trace.model_requests += 1
    record_model_request()
    started_at = time.perf_counter()
    succeeded = False
    try:
        with measure_stage(stage):
            yield
        succeeded = True
    finally:
        duration_ms = (time.perf_counter() - started_at) * 1000
        record_model_metrics(stage=stage, duration_ms=duration_ms, success=succeeded)
        emit_structured_event(
            component="openai",
            event="model_request_completed",
            duration_ms=duration_ms,
            status="success" if succeeded else "failure",
            error_category=None if succeeded else "openai_error",
            error_owner=None if succeeded else "openai",
            error_stage=None if succeeded else stage,
            retryable=None if succeeded else True,
        )


def mark_voice_route_complete() -> None:
    trace = current_voice_trace()
    if trace is not None:
        trace.mark_route_complete()
