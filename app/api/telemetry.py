"""Bounded, allowlisted browser telemetry using the existing audit database."""

from __future__ import annotations

import logging
import time
from collections import deque
from threading import Lock
from typing import Literal

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.audit import SAFE_ERROR_CATEGORIES, get_audit_store
from app.config import get_settings
from app.telemetry import (
    classify_error,
    current_trace_id,
    emit_structured_event,
    record_frontend_metric,
    reset_trace,
    start_trace,
    validate_trace_id,
)


logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/telemetry", tags=["telemetry"])
_event_times: deque[float] = deque()
_rate_lock = Lock()


class FrontendTelemetryEvent(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trace_id: str | None = Field(default=None, min_length=8, max_length=64)
    event: Literal[
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
    ]
    duration_ms: float | None = Field(default=None, ge=0, le=600_000)
    error_category: str | None = Field(default=None, max_length=64)
    http_status: int | None = Field(default=None, ge=100, le=599)

    @field_validator("trace_id")
    @classmethod
    def safe_trace_id(cls, value: str | None) -> str | None:
        return validate_trace_id(value) if value is not None else None

    @field_validator("error_category")
    @classmethod
    def safe_error_category(cls, value: str | None) -> str | None:
        if value is not None and value not in SAFE_ERROR_CATEGORIES:
            raise ValueError("Unsupported safe error category.")
        return value


def _within_rate_limit(limit: int) -> bool:
    now = time.monotonic()
    cutoff = now - 60
    with _rate_lock:
        while _event_times and _event_times[0] < cutoff:
            _event_times.popleft()
        if len(_event_times) >= limit:
            return False
        _event_times.append(now)
        return True


async def _persist_frontend_event_safely(event: FrontendTelemetryEvent, trace_id: str) -> None:
    settings = get_settings()
    if not settings.ai_audit_enabled:
        return
    try:
        store = get_audit_store(
            settings.ai_audit_db_path,
            settings.ai_audit_retention_days,
        )
        await store.write_frontend_event(
            trace_id=trace_id,
            event=event.event,
            duration_ms=event.duration_ms,
            error_category=event.error_category,
            http_status=event.http_status,
        )
    except Exception as exc:
        logger.warning("Frontend telemetry persistence failed: %s", type(exc).__name__)


@router.post("/frontend", status_code=202)
async def frontend_telemetry(
    event: FrontendTelemetryEvent,
    background_tasks: BackgroundTasks,
) -> dict[str, bool]:
    settings = get_settings()
    if not _within_rate_limit(settings.frontend_telemetry_rate_limit_per_minute):
        raise HTTPException(status_code=429, detail="Frontend telemetry rate limit exceeded.")

    resolved_trace_id = event.trace_id or current_trace_id()
    if resolved_trace_id is None:  # Defensive: normal HTTP middleware always sets one.
        raise HTTPException(status_code=400, detail="A trace ID is required.")
    token = None
    if resolved_trace_id != current_trace_id():
        _, token = start_trace(resolved_trace_id)
    try:
        owner, error_stage = classify_error(
            event.error_category,
            stage="frontend",
            owner="frontend",
        )
        record_frontend_metric(
            event.event,
            success=event.event not in {"network_error", "http_error", "render_error", "audio_play_failed"},
        )
        emit_structured_event(
            component="frontend",
            event=event.event,
            duration_ms=event.duration_ms,
            status=event.http_status,
            error_category=event.error_category,
            error_owner=owner if event.error_category else None,
            error_stage=error_stage if event.error_category else None,
            retryable=True if event.event in {"network_error", "http_error"} else None,
        )
        background_tasks.add_task(
            _persist_frontend_event_safely,
            event,
            resolved_trace_id,
        )
    finally:
        if token is not None:
            reset_trace(token)
    return {"accepted": True}
