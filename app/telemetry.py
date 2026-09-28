"""Safe, fail-open correlation, structured logging, and in-process metrics.

This module deliberately contains no conversation content.  Its context and metric
shapes are compatible with later OpenTelemetry/Prometheus exporters without making
an external observability service part of the request path.
"""

from __future__ import annotations

import json
import logging
import re
import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Lock
from typing import Any, Mapping
from uuid import uuid4

from app.config import get_settings


logger = logging.getLogger("app.structured")

TRACE_HEADER = "X-Trace-ID"
CORRELATION_HEADER = "X-Correlation-ID"
TRACE_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{7,63}$")
SAFE_VALUE_PATTERN = re.compile(r"^[A-Za-z0-9_./:-]{1,128}$")

ERROR_OWNERS = frozenset(
    {
        "frontend",
        "network",
        "ai_backend",
        "openai",
        "azure_stt",
        "azure_tts",
        "resourceplus_api",
        "configuration",
        "validation",
        "transaction",
        "unknown",
    }
)

ERROR_STAGES = frozenset(
    {
        "http_request",
        "websocket_request",
        "audio_receive",
        "audio_stream",
        "stt",
        "post_release_stt_finalize",
        "language_resolution",
        "openai_main",
        "follow_up_classifier",
        "confirmation_classifier",
        "response_renderer",
        "agent",
        "resourceplus",
        "speech_normalization",
        "tts",
        "response_send",
        "audit_persist",
        "frontend",
        "configuration",
        "transaction",
        "validation",
        "unknown",
    }
)

FRONTEND_EVENTS = frozenset(
    {
        "request_started",
        "response_received",
        "websocket_opened",
        "websocket_closed",
        "microphone_started",
        "microphone_released",
        "audio_play_started",
        "audio_play_completed",
        "audio_play_failed",
        "network_error",
        "http_error",
        "render_error",
    }
)

ERROR_OWNER_BY_CATEGORY = {
    "no_speech": "validation",
    "no_audio": "validation",
    "no_recognized_speech": "azure_stt",
    "fallback_stt_no_match": "azure_stt",
    "fallback_invalid_audio": "validation",
    "azure_canceled": "azure_stt",
    "speech_recognition_failed": "azure_stt",
    "speech_synthesis_failed": "azure_tts",
    "voice_unavailable": "configuration",
    "invalid_stream_state": "validation",
    "stream_finalize_timeout": "azure_stt",
    "client_aborted": "frontend",
    "websocket_disconnected": "network",
    "resourceplus_error": "resourceplus_api",
    "resourceplus_timeout": "resourceplus_api",
    "no_resourceplus_suggestion": "transaction",
    "openai_error": "openai",
    "tts_error": "azure_tts",
    "validation_error": "validation",
    "confirmation_expired": "transaction",
    "access_denied": "transaction",
    "configuration_error": "configuration",
    "network_error": "network",
    "unknown_safe_category": "unknown",
}

ERROR_STAGE_BY_CATEGORY = {
    "no_speech": "stt",
    "no_audio": "audio_receive",
    "no_recognized_speech": "stt",
    "fallback_stt_no_match": "stt",
    "fallback_invalid_audio": "audio_receive",
    "azure_canceled": "stt",
    "speech_recognition_failed": "stt",
    "speech_synthesis_failed": "tts",
    "voice_unavailable": "configuration",
    "invalid_stream_state": "audio_stream",
    "stream_finalize_timeout": "post_release_stt_finalize",
    "client_aborted": "websocket_request",
    "websocket_disconnected": "websocket_request",
    "resourceplus_error": "resourceplus",
    "resourceplus_timeout": "resourceplus",
    "no_resourceplus_suggestion": "resourceplus",
    "openai_error": "openai_main",
    "tts_error": "tts",
    "validation_error": "validation",
    "confirmation_expired": "transaction",
    "access_denied": "transaction",
    "configuration_error": "configuration",
    "network_error": "http_request",
    "unknown_safe_category": "unknown",
}

_current_trace_id: ContextVar[str | None] = ContextVar(
    "current_observability_trace_id",
    default=None,
)


def validate_trace_id(value: str) -> str:
    """Trim and validate an opaque correlation ID without trying to interpret it."""

    normalized = value.strip()
    if not TRACE_ID_PATTERN.fullmatch(normalized):
        raise ValueError(
            "X-Trace-ID must be 8-64 safe ASCII letters, digits, '.', '_', ':', or '-'."
        )
    return normalized


def new_trace_id() -> str:
    return str(uuid4())


def start_trace(incoming: str | None = None) -> tuple[str, Token[str | None]]:
    trace_id = validate_trace_id(incoming) if incoming is not None else new_trace_id()
    return trace_id, _current_trace_id.set(trace_id)


def reset_trace(token: Token[str | None]) -> None:
    _current_trace_id.reset(token)


def current_trace_id() -> str | None:
    return _current_trace_id.get()


def classify_error(
    category: str | None,
    *,
    stage: str | None = None,
    owner: str | None = None,
) -> tuple[str, str]:
    safe_owner = owner if owner in ERROR_OWNERS else ERROR_OWNER_BY_CATEGORY.get(
        category or "",
        "unknown",
    )
    candidate_stage = stage or ERROR_STAGE_BY_CATEGORY.get(category or "", "unknown")
    safe_stage = candidate_stage if candidate_stage in ERROR_STAGES else "unknown"
    return safe_owner, safe_stage


def _deployment_metadata() -> dict[str, str]:
    try:
        settings = get_settings()
        return {
            "environment": settings.app_environment,
            "deployment_id": settings.deployment_id,
            "app_version": settings.app_version,
        }
    except Exception:
        return {
            "environment": "unknown",
            "deployment_id": "unknown",
            "app_version": "unknown",
        }


def _safe_scalar(value: Any, *, limit: int = 128) -> str | int | float | bool | None:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    normalized = str(value)[:limit]
    lowered = normalized.casefold()
    if "@" in normalized or any(
        marker in lowered
        for marker in ("authorization", "bearer", "password", "api_key", "apikey", "secret", "token")
    ):
        return "[redacted]"
    return normalized if SAFE_VALUE_PATTERN.fullmatch(normalized) else None


def _safe_diagnostic_text(value: Any, *, limit: int = 240) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = " ".join(value.split())
    lowered = normalized.casefold()
    if (
        not normalized
        or len(normalized) > limit
        or "@" in normalized
        or "http://" in lowered
        or "https://" in lowered
        or any(
            marker in lowered
            for marker in (
                "authorization",
                "bearer",
                "password",
                "api_key",
                "apikey",
                "secret",
                "token",
            )
        )
    ):
        return None
    return normalized


def emit_structured_event(
    *,
    level: str = "INFO",
    component: str,
    event: str,
    duration_ms: float | None = None,
    status: str | int | None = None,
    error_category: str | None = None,
    error_owner: str | None = None,
    error_stage: str | None = None,
    retryable: bool | None = None,
    endpoint: str | None = None,
    upstream_error_code: str | None = None,
    upstream_error_message: str | None = None,
    upstream_correlation_id: str | None = None,
    validation_fields: tuple[str, ...] = (),
) -> None:
    """Emit only allowlisted scalar fields; logging failures are swallowed."""

    try:
        metadata = _deployment_metadata()
        safe_error_category = (
            error_category
            if error_category in ERROR_OWNER_BY_CATEGORY
            else "unknown_safe_category"
            if error_category is not None
            else None
        )
        interaction_id = None
        session_reference = None
        try:
            from app.audit import current_interaction_audit

            audit = current_interaction_audit()
            if audit is not None:
                interaction_id = audit.interaction_id
                session_reference = audit.session_reference
        except Exception:
            pass
        payload: dict[str, Any] = {
            "timestamp": datetime.now(UTC).isoformat(),
            "level": level.upper() if level.upper() in {"DEBUG", "INFO", "WARNING", "ERROR"} else "INFO",
            "service": "resourceplus-ai-assistant",
            **metadata,
            "trace_id": current_trace_id(),
            "interaction_id": interaction_id,
            "session_reference": session_reference,
            "component": _safe_scalar(component),
            "event": _safe_scalar(event),
            "duration_ms": None if duration_ms is None else round(max(float(duration_ms), 0.0), 3),
            "status": _safe_scalar(status),
            "error_category": safe_error_category,
            "error_owner": error_owner if error_owner in ERROR_OWNERS else None,
            "error_stage": error_stage if error_stage in ERROR_STAGES else None,
            "retryable": retryable,
            "endpoint": _safe_scalar(endpoint),
            "upstream_error_code": _safe_scalar(upstream_error_code),
            "upstream_error_message": _safe_diagnostic_text(
                upstream_error_message
            ),
            "upstream_correlation_id": _safe_scalar(upstream_correlation_id),
            "validation_fields": [
                safe_field
                for field in validation_fields[:12]
                if (safe_field := _safe_scalar(field)) is not None
            ]
            or None,
        }
        clean = {key: value for key, value in payload.items() if value is not None}
        logger.log(
            getattr(logging, clean["level"], logging.INFO),
            json.dumps(clean, ensure_ascii=True, separators=(",", ":")),
        )
    except Exception:
        # Observability is never allowed to affect the user path.
        return


@dataclass
class _Histogram:
    buckets: tuple[float, ...]
    counts: list[int] = field(init=False)
    count: int = 0
    total: float = 0.0

    def __post_init__(self) -> None:
        self.counts = [0 for _ in self.buckets]

    def observe(self, value: float) -> None:
        safe_value = max(float(value), 0.0)
        self.count += 1
        self.total += safe_value
        for index, boundary in enumerate(self.buckets):
            if safe_value <= boundary:
                self.counts[index] += 1


class MetricsRegistry:
    """Bounded Prometheus text registry with fixed, low-cardinality labels."""

    HISTOGRAM_BUCKETS_MS = (
        5.0,
        10.0,
        25.0,
        50.0,
        100.0,
        250.0,
        500.0,
        1_000.0,
        2_000.0,
        5_000.0,
        10_000.0,
        30_000.0,
        60_000.0,
    )
    COUNTER_LABELS: Mapping[str, tuple[str, ...]] = {
        "resourceplus_assistant_requests_total": ("mode", "status"),
        "resourceplus_assistant_openai_requests_total": ("stage", "status"),
        "resourceplus_assistant_resourceplus_requests_total": ("method", "status"),
        "resourceplus_assistant_azure_stt_requests_total": ("status",),
        "resourceplus_assistant_azure_tts_requests_total": ("status",),
        "resourceplus_assistant_websocket_failures_total": ("category",),
        "resourceplus_assistant_pending_action_events_total": ("state",),
        "resourceplus_assistant_frontend_events_total": ("event", "status"),
    }
    HISTOGRAM_LABELS: Mapping[str, tuple[str, ...]] = {
        "resourceplus_assistant_http_request_duration_ms": ("method", "route", "status"),
        "resourceplus_assistant_voice_after_release_duration_ms": ("status",),
        "resourceplus_assistant_openai_request_duration_ms": ("stage", "status"),
        "resourceplus_assistant_resourceplus_request_duration_ms": ("method", "status"),
        "resourceplus_assistant_azure_stt_duration_ms": ("status",),
        "resourceplus_assistant_azure_tts_duration_ms": ("status",),
    }
    ALLOWED_LABEL_VALUES: Mapping[str, frozenset[str]] = {
        "mode": frozenset({"text", "voice", "other"}),
        "status": frozenset({"success", "failure", "timeout", "aborted", "no_speech"}),
        "method": frozenset({"GET", "POST", "OPTIONS", "OTHER"}),
        "route": frozenset(
            {
                "chat",
                "voice_chat",
                "frontend_telemetry",
                "health",
                "ready",
                "metrics",
                "ops_diagnostics",
                "static",
                "other",
            }
        ),
        "stage": frozenset(
            {
                "openai_main",
                "follow_up_classifier",
                "confirmation_classifier",
                "response_renderer",
            }
        ),
        "category": frozenset(
            {
                "no_audio",
                "no_recognized_speech",
                "azure_canceled",
                "websocket_disconnected",
                "invalid_stream_state",
                "stream_finalize_timeout",
                "client_aborted",
                "unknown_safe_category",
            }
        ),
        "state": frozenset(
            {"prepared", "confirmed", "rejected", "expired", "executed", "failed"}
        ),
        "event": FRONTEND_EVENTS,
    }

    def __init__(self) -> None:
        self._lock = Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], _Histogram] = {}

    def _labels(
        self,
        name: str,
        supplied: Mapping[str, str],
        definitions: Mapping[str, tuple[str, ...]],
    ) -> tuple[tuple[str, str], ...]:
        expected = definitions.get(name)
        if expected is None or set(supplied) != set(expected):
            raise ValueError("Unsupported metric or label set")
        normalized: list[tuple[str, str]] = []
        for key in expected:
            value = str(supplied[key])
            if value not in self.ALLOWED_LABEL_VALUES[key]:
                raise ValueError("Unsupported metric label value")
            normalized.append((key, value))
        return tuple(normalized)

    def increment(self, name: str, **labels: str) -> None:
        try:
            label_tuple = self._labels(name, labels, self.COUNTER_LABELS)
            with self._lock:
                key = (name, label_tuple)
                self._counters[key] = self._counters.get(key, 0) + 1
        except Exception:
            return

    def observe(self, name: str, value: float, **labels: str) -> None:
        try:
            label_tuple = self._labels(name, labels, self.HISTOGRAM_LABELS)
            with self._lock:
                key = (name, label_tuple)
                histogram = self._histograms.get(key)
                if histogram is None:
                    histogram = _Histogram(self.HISTOGRAM_BUCKETS_MS)
                    self._histograms[key] = histogram
                histogram.observe(value)
        except Exception:
            return

    @staticmethod
    def _format_labels(labels: tuple[tuple[str, str], ...], extra: tuple[str, str] | None = None) -> str:
        items = [*labels]
        if extra is not None:
            items.append(extra)
        if not items:
            return ""
        return "{" + ",".join(f'{key}="{value}"' for key, value in items) + "}"

    def render(self) -> str:
        try:
            with self._lock:
                counters = list(self._counters.items())
                histograms = list(self._histograms.items())
            lines: list[str] = []
            seen_types: set[str] = set()
            for (name, labels), value in sorted(counters):
                if name not in seen_types:
                    lines.extend((f"# TYPE {name} counter",))
                    seen_types.add(name)
                lines.append(f"{name}{self._format_labels(labels)} {value}")
            for (name, labels), histogram in sorted(histograms):
                if name not in seen_types:
                    lines.append(f"# TYPE {name} histogram")
                    seen_types.add(name)
                for boundary, count in zip(histogram.buckets, histogram.counts, strict=True):
                    lines.append(
                        f"{name}_bucket{self._format_labels(labels, ('le', f'{boundary:g}'))} {count}"
                    )
                lines.append(
                    f"{name}_bucket{self._format_labels(labels, ('le', '+Inf'))} {histogram.count}"
                )
                lines.append(f"{name}_sum{self._format_labels(labels)} {histogram.total:.3f}")
                lines.append(f"{name}_count{self._format_labels(labels)} {histogram.count}")
            return "\n".join(lines) + ("\n" if lines else "")
        except Exception:
            return ""

    def reset(self) -> None:
        with self._lock:
            self._counters.clear()
            self._histograms.clear()


metrics = MetricsRegistry()


def route_label(path: str) -> str:
    return {
        "/api/chat": "chat",
        "/api/voice/chat": "voice_chat",
        "/api/telemetry/frontend": "frontend_telemetry",
        "/health": "health",
        "/ready": "ready",
        "/metrics": "metrics",
        "/api/ops/diagnostics": "ops_diagnostics",
        "/": "static",
    }.get(path, "static" if not path.startswith("/api/") else "other")


def request_mode(path: str) -> str:
    if path == "/api/chat":
        return "text"
    if path.startswith("/api/voice/"):
        return "voice"
    return "other"


def status_label(status_code: int) -> str:
    return "success" if status_code < 400 else "failure"


def record_http_metrics(*, method: str, path: str, status_code: int, duration_ms: float) -> None:
    try:
        safe_method = method if method in {"GET", "POST", "OPTIONS"} else "OTHER"
        safe_status = status_label(status_code)
        metrics.increment(
            "resourceplus_assistant_requests_total",
            mode=request_mode(path),
            status=safe_status,
        )
        metrics.observe(
            "resourceplus_assistant_http_request_duration_ms",
            duration_ms,
            method=safe_method,
            route=route_label(path),
            status=safe_status,
        )
    except Exception:
        return


def record_model_metrics(*, stage: str, duration_ms: float, success: bool) -> None:
    try:
        status = "success" if success else "failure"
        metrics.increment("resourceplus_assistant_openai_requests_total", stage=stage, status=status)
        metrics.observe(
            "resourceplus_assistant_openai_request_duration_ms",
            duration_ms,
            stage=stage,
            status=status,
        )
    except Exception:
        return


def record_resourceplus_metrics(*, method: str, status: str | int, duration_ms: float) -> None:
    try:
        normalized_method = method.upper() if method.upper() in {"GET", "POST"} else "OTHER"
        if status == "timeout":
            normalized_status = "timeout"
        elif isinstance(status, int) and status < 400:
            normalized_status = "success"
        else:
            normalized_status = "failure"
        metrics.increment(
            "resourceplus_assistant_resourceplus_requests_total",
            method=normalized_method,
            status=normalized_status,
        )
        metrics.observe(
            "resourceplus_assistant_resourceplus_request_duration_ms",
            duration_ms,
            method=normalized_method,
            status=normalized_status,
        )
    except Exception:
        return


def record_speech_metrics(*, service: str, duration_ms: float, success: bool) -> None:
    try:
        if service not in {"stt", "tts"}:
            return
        status = "success" if success else "failure"
        prefix = "azure_stt" if service == "stt" else "azure_tts"
        metrics.increment(f"resourceplus_assistant_{prefix}_requests_total", status=status)
        metrics.observe(
            f"resourceplus_assistant_{prefix}_duration_ms",
            duration_ms,
            status=status,
        )
    except Exception:
        return


def record_action_metric(state: str) -> None:
    try:
        if state == "pending_confirmation":
            state = "prepared"
        metrics.increment("resourceplus_assistant_pending_action_events_total", state=state)
    except Exception:
        return


def record_frontend_metric(event: str, *, success: bool) -> None:
    try:
        metrics.increment(
            "resourceplus_assistant_frontend_events_total",
            event=event,
            status="success" if success else "failure",
        )
    except Exception:
        return


def record_websocket_failure(category: str) -> None:
    try:
        safe = category if category in MetricsRegistry.ALLOWED_LABEL_VALUES["category"] else "unknown_safe_category"
        metrics.increment("resourceplus_assistant_websocket_failures_total", category=safe)
    except Exception:
        return


def elapsed_ms(started_at: float) -> float:
    return max((time.perf_counter() - started_at) * 1000, 0.0)


class ResponseSendTelemetryMiddleware:
    """Measure ASGI response transmission without buffering or changing the body."""

    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope.get("type") != "http":
            await self.app(scope, receive, send)
            return
        send_started: float | None = None
        response_trace_id: str | None = None
        response_status: int | None = None

        async def observed_send(message: dict[str, Any]) -> None:
            nonlocal send_started, response_trace_id, response_status
            if message.get("type") == "http.response.start":
                send_started = time.perf_counter()
                response_status = message.get("status")
                for name, value in message.get("headers", []):
                    if name.lower() == TRACE_HEADER.lower().encode("ascii"):
                        try:
                            response_trace_id = validate_trace_id(value.decode("ascii"))
                        except (UnicodeDecodeError, ValueError):
                            response_trace_id = None
                        break
            await send(message)
            if (
                message.get("type") == "http.response.body"
                and not message.get("more_body", False)
                and send_started is not None
            ):
                token = None
                try:
                    if response_trace_id and current_trace_id() != response_trace_id:
                        _, token = start_trace(response_trace_id)
                    emit_structured_event(
                        component="response",
                        event="response_send_completed",
                        duration_ms=elapsed_ms(send_started),
                        status=response_status,
                    )
                finally:
                    if token is not None:
                        reset_trace(token)

        await self.app(scope, receive, observed_send)
