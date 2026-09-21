import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from app.config import get_settings
from app.telemetry import (
    classify_error,
    current_trace_id,
    emit_structured_event,
    record_action_metric,
)


logger = logging.getLogger(__name__)

SAFE_ERROR_CATEGORIES = frozenset(
    {
        "no_speech",
        "no_audio",
        "no_recognized_speech",
        "fallback_stt_no_match",
        "fallback_invalid_audio",
        "azure_canceled",
        "speech_recognition_failed",
        "speech_synthesis_failed",
        "voice_unavailable",
        "invalid_stream_state",
        "stream_finalize_timeout",
        "client_aborted",
        "websocket_disconnected",
        "resourceplus_error",
        "resourceplus_timeout",
        "openai_error",
        "tts_error",
        "validation_error",
        "configuration_error",
        "network_error",
        "confirmation_expired",
        "access_denied",
        "unknown_safe_category",
    }
)
SAFE_ACTION_STATES = frozenset(
    {
        "none",
        "prepared",
        "pending_confirmation",
        "rejected",
        "expired",
        "executed",
        "failed",
    }
)
SAFE_ACTION_RESULTS = frozenset(
    {
        "submitted_for_approval",
        "cancelled",
        "approved",
        "updated",
        "succeeded",
        "failed",
    }
)
SAFE_ACTION_TYPES = frozenset(
    {
        "book_day_type",
        "create_exceptional_entry",
        "cancel_day_type_request",
        "approve_supervisor_request",
        "approve_all_requests",
        "update_notification_read_status",
    }
)
AUDIT_SCHEMA_VERSION = 4


@dataclass
class InteractionAudit:
    input_mode: str
    trace_id: str | None = None
    app_version: str | None = None
    git_commit: str | None = None
    environment: str | None = None
    deployment_id: str | None = None
    input_source: str | None = None
    user_text: str | None = None
    session_reference: str | None = None
    interaction_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    raw_detected_locale: str | None = None
    resolved_language: str | None = None
    response_language: str | None = None
    display_message: str | None = None
    speech_message: str | None = None
    tts_text: str | None = None
    tts_requested: bool | None = None
    tts_generated: bool | None = None
    tts_locale: str | None = None
    tts_voice: str | None = None
    autoplay_result: str | None = None
    tools_used: list[str] = field(default_factory=list)
    resourceplus_calls: list[dict[str, Any]] = field(default_factory=list)
    model_requests: int = 0
    # Persisted/exported values are milliseconds. Operational VOICE_LATENCY logs
    # intentionally remain seconds-based for backward-compatible production logs.
    latencies: dict[str, float] = field(default_factory=dict)
    success: bool = False
    result_status: str = "unknown"
    error_category: str | None = None
    error_owner: str | None = None
    error_stage: str | None = None
    confirmation_required: bool = False
    confirmed: bool | None = None
    action_type: str | None = None
    action_state: str = "none"
    action_result: str | None = None
    _started_at: float = field(default_factory=time.perf_counter, repr=False, compare=False)


_current_audit: ContextVar[InteractionAudit | None] = ContextVar(
    "current_interaction_audit",
    default=None,
)


def safe_session_reference(session_id: str | None) -> str | None:
    if not session_id:
        return None
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]


def start_interaction_audit(
    *,
    input_mode: str,
    input_source: str | None = None,
    user_text: str | None = None,
    session_id: str | None = None,
) -> tuple[InteractionAudit, Token[InteractionAudit | None]]:
    settings = get_settings()
    audit = InteractionAudit(
        input_mode=input_mode,
        trace_id=current_trace_id(),
        app_version=getattr(settings, "app_version", None),
        git_commit=getattr(settings, "git_commit", None),
        environment=getattr(settings, "app_environment", None),
        deployment_id=getattr(settings, "deployment_id", None),
        input_source=input_source or ("typed" if input_mode == "text" else "stt"),
        user_text=user_text,
        session_reference=safe_session_reference(session_id),
    )
    return audit, _current_audit.set(audit)


def reset_interaction_audit(token: Token[InteractionAudit | None]) -> None:
    _current_audit.reset(token)


def current_interaction_audit() -> InteractionAudit | None:
    return _current_audit.get()


def record_model_request() -> None:
    audit = current_interaction_audit()
    if audit is not None:
        audit.model_requests += 1


def record_latency(stage: str, duration_seconds: float) -> None:
    audit = current_interaction_audit()
    if audit is not None:
        duration_ms = max(duration_seconds, 0.0) * 1000
        audit.latencies[stage] = round(
            audit.latencies.get(stage, 0.0) + duration_ms,
            3,
        )


def record_tool_usage(name: str) -> None:
    audit = current_interaction_audit()
    if audit is not None and name not in audit.tools_used:
        audit.tools_used.append(name)


def record_safe_error(
    category: str | None,
    *,
    owner: str | None = None,
    stage: str | None = None,
) -> None:
    audit = current_interaction_audit()
    if audit is not None:
        audit.error_category = (
            category if category in SAFE_ERROR_CATEGORIES else "unknown_safe_category"
        )
        audit.error_owner, audit.error_stage = classify_error(
            audit.error_category,
            owner=owner,
            stage=stage,
        )


def safe_error_category(exc: Exception) -> str:
    category = getattr(exc, "safe_category", None)
    if isinstance(category, str) and category in SAFE_ERROR_CATEGORIES:
        return category
    name = type(exc).__name__
    if name in {"AIConfigurationError", "SpeechConfigurationError", "ResourcePlusConfigurationError"}:
        return "configuration_error"
    if name == "ResourcePlusTimeoutError":
        return "resourceplus_timeout"
    if name.startswith("ResourcePlus"):
        return "resourceplus_error"
    if name in {"OpenAIServiceError", "OpenAIError"}:
        return "openai_error"
    if name == "SpeechSynthesisError":
        return "tts_error"
    if name in {"ValidationError", "ValueError", "TypeError", "KeyError"}:
        return "validation_error"
    return "unknown_safe_category"


def record_action_state(
    *,
    action_type: str | None = None,
    state: str,
    confirmation_required: bool | None = None,
    confirmed: bool | None = None,
    result: str | None = None,
) -> None:
    audit = current_interaction_audit()
    if audit is None:
        return
    if action_type is not None:
        audit.action_type = (
            action_type if action_type in SAFE_ACTION_TYPES else "unknown_action"
        )
    audit.action_state = state if state in SAFE_ACTION_STATES else "failed"
    if confirmation_required is not None:
        audit.confirmation_required = confirmation_required
    if confirmed is not None:
        audit.confirmed = confirmed
    if result is not None:
        audit.action_result = result if result in SAFE_ACTION_RESULTS else "failed"
    metric_state = (
        "confirmed"
        if audit.action_state == "prepared" and confirmed is True
        else audit.action_state
    )
    record_action_metric(metric_state)


def record_resourceplus_call(
    *,
    method: str,
    endpoint: str,
    status: str | int,
    duration: float,
) -> None:
    audit = current_interaction_audit()
    if audit is not None:
        duration_ms = round(max(duration, 0.0) * 1000, 3)
        audit.resourceplus_calls.append(
            {
                "method": method.upper(),
                "endpoint": endpoint,
                "status": status,
                "duration_ms": duration_ms,
            }
        )
        audit.latencies["resourceplus"] = round(
            audit.latencies.get("resourceplus", 0.0) + duration_ms,
            3,
        )


def _normalize_resourceplus_calls(calls: object) -> list[dict[str, Any]]:
    if not isinstance(calls, list):
        return []
    normalized: list[dict[str, Any]] = []
    for call in calls:
        if not isinstance(call, dict):
            continue
        item = {
            "method": call.get("method"),
            "endpoint": call.get("endpoint"),
            "status": call.get("status"),
        }
        if "duration_ms" in call:
            item["duration_ms"] = call.get("duration_ms")
        elif isinstance(call.get("duration"), (int, float)):
            item["duration_ms"] = round(float(call["duration"]) * 1000, 3)
        normalized.append(item)
    return normalized


class AuditStore:
    _MIGRATION_COLUMNS = {
        "trace_id": "TEXT",
        "app_version": "TEXT",
        "git_commit": "TEXT",
        "environment": "TEXT",
        "deployment_id": "TEXT",
        "error_owner": "TEXT",
        "error_stage": "TEXT",
        "input_source": "TEXT",
        "response_language": "TEXT",
        "tts_requested": "INTEGER",
        "tts_generated": "INTEGER",
        "tts_locale": "TEXT",
        "tts_voice": "TEXT",
        "autoplay_result": "TEXT",
        "result_status": "TEXT NOT NULL DEFAULT 'unknown'",
        "action_type": "TEXT",
        "action_state": "TEXT NOT NULL DEFAULT 'none'",
        "confirmed": "INTEGER",
        "action_result": "TEXT",
        "latency_unit": "TEXT NOT NULL DEFAULT 'seconds'",
        "schema_version": "INTEGER NOT NULL DEFAULT 1",
    }

    def __init__(self, path: str, *, retention_days: int) -> None:
        self.path = Path(path)
        self.retention_days = retention_days
        self._initialized = False
        self._initialize_lock = Lock()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize_sync(self) -> None:
        if self._initialized:
            return
        with self._initialize_lock:
            if self._initialized:
                return
            with self._connect() as connection:
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS assistant_audit (
                        interaction_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        trace_id TEXT,
                        app_version TEXT,
                        git_commit TEXT,
                        environment TEXT,
                        deployment_id TEXT,
                        session_reference TEXT,
                        input_mode TEXT NOT NULL,
                        user_text TEXT,
                        raw_detected_locale TEXT,
                        resolved_language TEXT,
                        display_message TEXT,
                        speech_message TEXT,
                        tts_text TEXT,
                        tools_json TEXT NOT NULL,
                        resourceplus_json TEXT NOT NULL,
                        model_requests INTEGER NOT NULL,
                        latencies_json TEXT NOT NULL,
                        success INTEGER NOT NULL,
                        error_category TEXT,
                        error_owner TEXT,
                        error_stage TEXT,
                        confirmation_required INTEGER NOT NULL,
                        input_source TEXT,
                        response_language TEXT,
                        tts_requested INTEGER,
                        tts_generated INTEGER,
                        tts_locale TEXT,
                        tts_voice TEXT,
                        autoplay_result TEXT,
                        result_status TEXT NOT NULL DEFAULT 'unknown',
                        action_type TEXT,
                        action_state TEXT NOT NULL DEFAULT 'none',
                        confirmed INTEGER,
                        action_result TEXT,
                        latency_unit TEXT NOT NULL DEFAULT 'milliseconds',
                        schema_version INTEGER NOT NULL DEFAULT 4
                    )
                    """
                )
                existing = {
                    row["name"]
                    for row in connection.execute("PRAGMA table_info(assistant_audit)")
                }
                for name, definition in self._MIGRATION_COLUMNS.items():
                    if name not in existing:
                        connection.execute(
                            f"ALTER TABLE assistant_audit ADD COLUMN {name} {definition}"
                        )
                connection.execute(
                    """
                    CREATE TABLE IF NOT EXISTS frontend_telemetry (
                        event_id TEXT PRIMARY KEY,
                        timestamp TEXT NOT NULL,
                        trace_id TEXT NOT NULL,
                        event TEXT NOT NULL,
                        duration_ms REAL,
                        error_category TEXT,
                        http_status INTEGER
                    )
                    """
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_audit_trace_id ON assistant_audit(trace_id)"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS idx_frontend_trace_id ON frontend_telemetry(trace_id)"
                )
                cutoff = (
                    datetime.now(UTC) - timedelta(days=self.retention_days)
                ).isoformat()
                connection.execute(
                    "DELETE FROM assistant_audit WHERE timestamp < ?",
                    (cutoff,),
                )
                connection.execute(
                    "DELETE FROM frontend_telemetry WHERE timestamp < ?",
                    (cutoff,),
                )
            self._initialized = True

    def _write_sync(self, audit: InteractionAudit, *, store_content: bool) -> None:
        persist_started = time.perf_counter()
        self._initialize_sync()
        content = (
            (audit.user_text, audit.display_message, audit.speech_message, audit.tts_text)
            if store_content
            else (None, None, None, None)
        )
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO assistant_audit (
                    interaction_id, timestamp, trace_id, app_version, git_commit,
                    environment, deployment_id, session_reference, input_mode,
                    user_text, raw_detected_locale, resolved_language,
                    display_message, speech_message, tts_text, tools_json,
                    resourceplus_json, model_requests, latencies_json, success,
                    error_category, error_owner, error_stage,
                    confirmation_required, input_source,
                    response_language, tts_requested, tts_generated,
                    tts_locale, tts_voice, autoplay_result, result_status,
                    action_type, action_state,
                    confirmed, action_result, latency_unit, schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    audit.interaction_id,
                    audit.timestamp,
                    audit.trace_id,
                    audit.app_version,
                    audit.git_commit,
                    audit.environment,
                    audit.deployment_id,
                    audit.session_reference,
                    audit.input_mode,
                    *content[:1],
                    audit.raw_detected_locale,
                    audit.resolved_language,
                    *content[1:],
                    json.dumps(audit.tools_used, ensure_ascii=False),
                    json.dumps(audit.resourceplus_calls, ensure_ascii=False),
                    audit.model_requests,
                    json.dumps(audit.latencies, ensure_ascii=False),
                    int(audit.success),
                    audit.error_category,
                    audit.error_owner,
                    audit.error_stage,
                    int(audit.confirmation_required),
                    audit.input_source,
                    audit.response_language,
                    None if audit.tts_requested is None else int(audit.tts_requested),
                    None if audit.tts_generated is None else int(audit.tts_generated),
                    audit.tts_locale,
                    audit.tts_voice,
                    audit.autoplay_result,
                    audit.result_status,
                    audit.action_type,
                    audit.action_state,
                    None if audit.confirmed is None else int(audit.confirmed),
                    audit.action_result,
                    "milliseconds",
                    AUDIT_SCHEMA_VERSION,
                ),
            )
            audit.latencies["audit_persist"] = round(
                audit.latencies.get("audit_persist", 0.0)
                + (time.perf_counter() - persist_started) * 1000,
                3,
            )
            audit.latencies["total"] = round(
                max(
                    audit.latencies.get("total", 0.0),
                    (time.perf_counter() - audit._started_at) * 1000,
                ),
                3,
            )
            connection.execute(
                "UPDATE assistant_audit SET latencies_json = ? WHERE interaction_id = ?",
                (
                    json.dumps(audit.latencies, ensure_ascii=False),
                    audit.interaction_id,
                ),
            )

    async def write(self, audit: InteractionAudit, *, store_content: bool) -> None:
        await asyncio.to_thread(self._write_sync, audit, store_content=store_content)

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> dict[str, Any]:
        item = dict(row)
        item["tools"] = json.loads(item.pop("tools_json"))
        calls = json.loads(item.pop("resourceplus_json"))
        item["resourceplus_calls"] = _normalize_resourceplus_calls(calls)
        latencies = json.loads(item.pop("latencies_json"))
        if item.get("latency_unit") != "milliseconds":
            latencies = {
                key: round(float(value) * 1000, 3)
                for key, value in latencies.items()
                if isinstance(value, (int, float))
            }
        item["latencies_ms"] = latencies
        # Compatibility alias for the existing debug endpoint and callers.
        item["latencies"] = latencies
        item["success"] = bool(item["success"])
        item["confirmation_required"] = bool(item["confirmation_required"])
        for nullable_bool in ("tts_requested", "tts_generated", "confirmed"):
            if item.get(nullable_bool) is not None:
                item[nullable_bool] = bool(item[nullable_bool])
        return item

    def _recent_sync(self, limit: int) -> list[dict[str, Any]]:
        self._initialize_sync()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM assistant_audit ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    async def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._recent_sync, limit)

    def _query_sync(
        self,
        *,
        from_timestamp: str | None = None,
        to_timestamp: str | None = None,
        language: str | None = None,
        mode: str | None = None,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        self._initialize_sync()
        where: list[str] = []
        values: list[object] = []
        if from_timestamp:
            where.append("timestamp >= ?")
            values.append(from_timestamp)
        if to_timestamp:
            where.append("timestamp < ?")
            values.append(to_timestamp)
        if language:
            where.append("resolved_language = ?")
            values.append(language)
        if mode:
            where.append("input_mode = ?")
            values.append(mode)
        if status:
            where.append("result_status = ?")
            values.append(status)
        clause = f" WHERE {' AND '.join(where)}" if where else ""
        with self._connect() as connection:
            rows = connection.execute(
                f"SELECT * FROM assistant_audit{clause} ORDER BY timestamp ASC",
                values,
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    async def query(self, **filters: str | None) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._query_sync, **filters)

    def _trace_sync(self, trace_id: str) -> list[dict[str, Any]]:
        self._initialize_sync()
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM assistant_audit WHERE trace_id = ? ORDER BY timestamp ASC",
                (trace_id,),
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    async def by_trace_id(self, trace_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._trace_sync, trace_id)

    def _write_frontend_event_sync(
        self,
        *,
        trace_id: str,
        event: str,
        duration_ms: float | None,
        error_category: str | None,
        http_status: int | None,
    ) -> None:
        self._initialize_sync()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO frontend_telemetry (
                    event_id, timestamp, trace_id, event, duration_ms,
                    error_category, http_status
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(uuid4()),
                    datetime.now(UTC).isoformat(),
                    trace_id,
                    event,
                    duration_ms,
                    error_category,
                    http_status,
                ),
            )

    async def write_frontend_event(self, **event: Any) -> None:
        await asyncio.to_thread(self._write_frontend_event_sync, **event)

    def _frontend_events_sync(self, trace_id: str) -> list[dict[str, Any]]:
        self._initialize_sync()
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT timestamp, trace_id, event, duration_ms,
                       error_category, http_status
                FROM frontend_telemetry
                WHERE trace_id = ?
                ORDER BY timestamp ASC
                """,
                (trace_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    async def frontend_events(self, trace_id: str) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._frontend_events_sync, trace_id)

    def _cleanup_sync(self, *, now: datetime | None = None) -> int:
        self._initialize_sync()
        cutoff = ((now or datetime.now(UTC)) - timedelta(days=self.retention_days)).isoformat()
        with self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM assistant_audit WHERE timestamp < ?",
                (cutoff,),
            )
            connection.execute(
                "DELETE FROM frontend_telemetry WHERE timestamp < ?",
                (cutoff,),
            )
            return cursor.rowcount

    async def cleanup(self, *, now: datetime | None = None) -> int:
        return await asyncio.to_thread(self._cleanup_sync, now=now)


async def persist_audit(audit: InteractionAudit) -> None:
    settings = get_settings()
    if not settings.ai_audit_enabled:
        return
    try:
        if audit.trace_id is None:
            audit.trace_id = current_trace_id()
        if audit.error_category and (not audit.error_owner or not audit.error_stage):
            audit.error_owner, audit.error_stage = classify_error(audit.error_category)
        store = get_audit_store(
            settings.ai_audit_db_path,
            settings.ai_audit_retention_days,
        )
        await store.write(audit, store_content=settings.ai_audit_store_content)
        emit_structured_event(
            component="audit",
            event="interaction_persisted",
            duration_ms=audit.latencies.get("audit_persist"),
            status="success",
        )
    except Exception as exc:  # Audit must never break the assistant response.
        logger.warning("Assistant audit persistence failed: %s", type(exc).__name__)
        emit_structured_event(
            component="audit",
            event="interaction_persist_failed",
            status="failure",
            error_category="unknown_safe_category",
            error_owner="ai_backend",
            error_stage="audit_persist",
            retryable=True,
        )


@lru_cache(maxsize=8)
def get_audit_store(path: str, retention_days: int) -> AuditStore:
    return AuditStore(path, retention_days=retention_days)


def audit_as_safe_dict(audit: InteractionAudit) -> dict[str, Any]:
    result = asdict(audit)
    result.pop("_started_at", None)
    return result
