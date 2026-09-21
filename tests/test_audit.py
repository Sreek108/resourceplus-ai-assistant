import asyncio
import csv
import io
import json
import sqlite3
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app import audit as audit_module
from app.api import chat as chat_module
from app.api import voice as voice_module
from app.audit import AuditStore, InteractionAudit, persist_audit
from app.audit import (
    reset_interaction_audit,
    start_interaction_audit,
)
from app.resourceplus.client import ResourcePlusClient
from app.main import app
from app.models.schemas import ChatResponse
from app.ai.sessions import session_store
from app.observability import measure_model_call, measure_stage
from app.speech import SpeechAudio, SpeechTranscript
from scripts import export_audit


client = TestClient(app)


@pytest.fixture
def audit_tmp_path():
    path = Path(".test-artifacts") / f"audit-{uuid4()}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def audit_settings(path, *, store_content=True, debug=False):
    return SimpleNamespace(
        ai_audit_enabled=True,
        ai_audit_store_content=store_content,
        ai_audit_retention_days=7,
        ai_audit_db_path=str(path),
        ai_audit_debug_endpoint_enabled=debug,
    )


def test_text_turn_creates_one_content_audit_record(monkeypatch, audit_tmp_path) -> None:
    database = audit_tmp_path / "audit.db"
    monkeypatch.setattr(
        audit_module,
        "get_settings",
        lambda: audit_settings(database),
    )

    async def process(request):
        return ChatResponse(
            success=True,
            message="### Balance\n\nYou have **12 days**.",
            language="en",
            tools_used=["get_home_data"],
            session_id="private-session-id",
        ).set_speech_message("You have twelve vacation days.")

    monkeypatch.setattr(chat_module, "process_chat", process)
    response = client.post("/api/chat", json={"message": "What is my balance?"})

    assert response.status_code == 200
    rows = asyncio.run(AuditStore(str(database), retention_days=7).recent())
    assert len(rows) == 1
    record = rows[0]
    assert record["user_text"] == "What is my balance?"
    assert record["input_source"] == "typed"
    assert record["display_message"] == "### Balance\n\nYou have **12 days**."
    assert record["speech_message"] == "You have twelve vacation days."
    assert record["tts_text"] is None
    assert record["resolved_language"] == "en"
    assert record["response_language"] == "en"
    assert record["tools"] == ["get_home_data"]
    assert record["session_reference"] != "private-session-id"
    assert record["confirmation_required"] is False
    assert record["tts_requested"] is None
    assert record["tts_generated"] is None
    assert record["result_status"] == "success"
    assert record["latency_unit"] == "milliseconds"
    assert record["schema_version"] == 4
    assert record["latencies_ms"]["agent"] >= 0


def test_voice_audit_preserves_exact_display_speech_and_tts(monkeypatch, audit_tmp_path) -> None:
    database = audit_tmp_path / "voice-audit.db"
    monkeypatch.setattr(
        audit_module,
        "get_settings",
        lambda: audit_settings(database),
    )

    async def transcribe(*args, **kwargs):
        return SpeechTranscript("Show my attendance", "en-US", "en")

    async def process(request, *, detected_language):
        return ChatResponse(
            success=True,
            message="### Attendance\n\n- Monday: **Present**",
            language=detected_language,
            tools_used=["get_attendance_summary"],
            session_id="voice-private-session",
        ).set_speech_message("You were present on Monday.")

    async def synthesize(text, *, language):
        assert text == "You were present on Monday."
        return SpeechAudio(
            b"audio-not-persisted",
            locale="en-US",
            voice_name="en-US-AvaNeural",
        )

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    monkeypatch.setattr(voice_module, "process_chat", process)
    monkeypatch.setattr(voice_module, "synthesize_speech", synthesize)

    response = client.post(
        "/api/voice/chat",
        files={"audio": ("voice.wav", b"RIFFinput", "audio/wav")},
        data={"confirmation_id": "secret-confirmation-id"},
    )

    assert response.status_code == 200
    rows = asyncio.run(AuditStore(str(database), retention_days=7).recent())
    assert len(rows) == 1
    record = rows[0]
    assert record["user_text"] == "Show my attendance"
    assert record["input_source"] == "stt"
    assert record["raw_detected_locale"] == "en-US"
    assert record["resolved_language"] == "en"
    assert record["response_language"] == "en"
    assert record["display_message"] == "### Attendance\n\n- Monday: **Present**"
    assert record["speech_message"] == "You were present on Monday."
    assert record["tts_text"] == "You were present on Monday."
    assert record["tts_requested"] is True
    assert record["tts_generated"] is True
    assert record["tts_locale"] == "en-US"
    assert record["tts_voice"] == "en-US-AvaNeural"
    assert record["autoplay_result"] is None
    assert "tts" in record["latencies"]
    database_bytes = database.read_bytes()
    assert b"secret-confirmation-id" not in database_bytes
    assert b"audio-not-persisted" not in database_bytes
    assert b"RIFFinput" not in database_bytes


@pytest.mark.asyncio
async def test_content_disabled_stores_metadata_only(audit_tmp_path) -> None:
    store = AuditStore(str(audit_tmp_path / "metadata.db"), retention_days=7)
    record = InteractionAudit(
        input_mode="voice",
        user_text="employee HR content",
        display_message="private display",
        speech_message="private speech",
        tts_text="private tts",
        tts_requested=True,
        tts_generated=True,
        tts_locale="ar-SA",
        tts_voice="ar-SA-ZariyahNeural",
        resolved_language="ar",
        tools_used=["get_home_data"],
        action_type="book_day_type",
        action_state="pending_confirmation",
        confirmation_required=True,
        confirmed=False,
        success=True,
    )

    await store.write(record, store_content=False)
    saved = (await store.recent())[0]

    assert saved["user_text"] is None
    assert saved["display_message"] is None
    assert saved["speech_message"] is None
    assert saved["tts_text"] is None
    assert saved["resolved_language"] == "ar"
    assert saved["tts_locale"] == "ar-SA"
    assert saved["tts_voice"] == "ar-SA-ZariyahNeural"
    assert saved["tools"] == ["get_home_data"]
    assert saved["action_type"] == "book_day_type"
    assert saved["action_state"] == "pending_confirmation"
    assert saved["confirmation_required"] is True


@pytest.mark.asyncio
async def test_model_count_and_available_latency_stages_are_persisted_in_ms(
    audit_tmp_path,
) -> None:
    audit, token = start_interaction_audit(input_mode="text", user_text="hello")
    try:
        with measure_stage("agent"):
            with measure_model_call("openai_main"):
                pass
            with measure_model_call("confirmation_classifier"):
                pass
    finally:
        reset_interaction_audit(token)
    audit.success = True
    audit.result_status = "success"
    store = AuditStore(str(audit_tmp_path / "timings.db"), retention_days=7)

    await store.write(audit, store_content=True)
    saved = (await store.recent())[0]

    assert saved["model_requests"] == 2
    assert saved["latencies_ms"]["agent"] >= 0
    assert saved["latencies_ms"]["openai_main"] >= 0
    assert saved["latencies_ms"]["confirmation_classifier"] >= 0
    assert saved["latencies_ms"]["audit_persist"] >= 0
    assert saved["latencies_ms"]["total"] >= 0


@pytest.mark.asyncio
async def test_audit_failure_does_not_fail_request_path(monkeypatch, audit_tmp_path, caplog) -> None:
    settings = audit_settings(audit_tmp_path / "broken.db")
    monkeypatch.setattr(audit_module, "get_settings", lambda: settings)

    async def broken_write(*args, **kwargs):
        raise OSError("secret filesystem detail")

    store = AuditStore(str(audit_tmp_path / "broken.db"), retention_days=7)
    monkeypatch.setattr(store, "write", broken_write)
    monkeypatch.setattr(audit_module, "get_audit_store", lambda *args: store)

    await persist_audit(InteractionAudit(input_mode="text"))

    assert "Assistant audit persistence failed: OSError" in caplog.text
    assert "secret filesystem detail" not in caplog.text


@pytest.mark.asyncio
async def test_retention_cleanup_removes_only_expired_rows(audit_tmp_path) -> None:
    store = AuditStore(str(audit_tmp_path / "retention.db"), retention_days=7)
    now = datetime.now(UTC)
    expired = InteractionAudit(
        input_mode="text",
        timestamp=(now - timedelta(days=8)).isoformat(),
    )
    current = InteractionAudit(
        input_mode="text",
        timestamp=(now - timedelta(days=1)).isoformat(),
    )
    await store.write(current, store_content=False)
    await store.write(expired, store_content=False)

    removed = await store.cleanup(now=now)
    rows = await store.recent()

    assert removed == 1
    assert [row["interaction_id"] for row in rows] == [current.interaction_id]


def test_debug_audit_endpoint_is_hidden_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.audit_debug.get_settings",
        lambda: SimpleNamespace(
            ai_audit_enabled=False,
            ai_audit_debug_endpoint_enabled=False,
        ),
    )

    response = client.get("/api/debug/audit/recent")

    assert response.status_code == 404


def test_debug_audit_endpoint_is_forced_hidden_during_public_uat(monkeypatch) -> None:
    monkeypatch.setattr(
        "app.api.audit_debug.get_settings",
        lambda: SimpleNamespace(
            ai_audit_enabled=True,
            ai_audit_debug_endpoint_enabled=True,
            uat_allowed_origins="https://lead-uat.ngrok-free.app",
        ),
    )

    response = client.get("/api/debug/audit/recent")

    assert response.status_code == 404


@pytest.mark.asyncio
async def test_resourceplus_audit_contains_only_safe_endpoint_and_status(
    monkeypatch,
    audit_tmp_path,
) -> None:
    import httpx

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    resourceplus = ResourcePlusClient(
        base_url="https://example.test/Mobile/",
        transport=httpx.MockTransport(handler),
    )
    audit, token = start_interaction_audit(input_mode="text")
    try:
        await resourceplus.get(
            "api/Client/GetHomeData",
            params={
                "Usremail": "private@example.com",
                "token": "secret-token",
                "Authorization": "Bearer backend-secret",
            },
        )
    finally:
        reset_interaction_audit(token)

    assert audit.resourceplus_calls[0]["endpoint"] == "api/Client/GetHomeData"
    assert audit.resourceplus_calls[0]["status"] == 200
    assert audit.resourceplus_calls[0]["duration_ms"] >= 0
    serialized = str(audit.resourceplus_calls)
    assert "private@example.com" not in serialized
    assert "secret-token" not in serialized
    assert "backend-secret" not in serialized
    database = audit_tmp_path / "resourceplus-trace.db"
    await AuditStore(str(database), retention_days=7).write(audit, store_content=False)
    database_bytes = database.read_bytes()
    assert b"private@example.com" not in database_bytes
    assert b"secret-token" not in database_bytes
    assert b"backend-secret" not in database_bytes


@pytest.mark.asyncio
async def test_additive_migration_preserves_legacy_rows(audit_tmp_path) -> None:
    database = audit_tmp_path / "legacy.db"
    timestamp = datetime.now(UTC).isoformat()
    with sqlite3.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE assistant_audit (
                interaction_id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
                session_reference TEXT, input_mode TEXT NOT NULL, user_text TEXT,
                raw_detected_locale TEXT, resolved_language TEXT,
                display_message TEXT, speech_message TEXT, tts_text TEXT,
                tools_json TEXT NOT NULL, resourceplus_json TEXT NOT NULL,
                model_requests INTEGER NOT NULL, latencies_json TEXT NOT NULL,
                success INTEGER NOT NULL, error_category TEXT,
                confirmation_required INTEGER NOT NULL
            )
            """
        )
        connection.execute(
            "INSERT INTO assistant_audit VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-interaction",
                timestamp,
                "legacy-session-ref",
                "voice",
                "legacy transcript",
                "en-US",
                "en",
                "legacy display",
                "legacy speech",
                "legacy tts",
                '["get_home_data"]',
                '[{"method":"GET","endpoint":"api/Client/GetHomeData","status":200,"duration":0.125}]',
                1,
                '{"total":0.25}',
                1,
                None,
                0,
            ),
        )

    rows = await AuditStore(str(database), retention_days=7).recent()

    assert len(rows) == 1
    assert rows[0]["interaction_id"] == "legacy-interaction"
    assert rows[0]["schema_version"] == 1
    assert rows[0]["trace_id"] is None
    assert rows[0]["app_version"] is None
    assert rows[0]["git_commit"] is None
    assert rows[0]["environment"] is None
    assert rows[0]["deployment_id"] is None
    assert rows[0]["error_owner"] is None
    assert rows[0]["error_stage"] is None
    assert rows[0]["tts_locale"] is None
    assert rows[0]["tts_voice"] is None
    assert rows[0]["latencies_ms"]["total"] == 250.0
    assert rows[0]["resourceplus_calls"][0]["duration_ms"] == 125.0
    assert rows[0]["action_state"] == "none"


def test_action_audit_records_safe_state_without_pending_internal_ids(
    monkeypatch,
    audit_tmp_path,
) -> None:
    database = audit_tmp_path / "action.db"
    monkeypatch.setattr(
        audit_module,
        "get_settings",
        lambda: audit_settings(database),
    )
    session_store.clear()
    session_id = session_store.ensure_session("private-session-token")
    pending = session_store.create_pending_action(
        session_id,
        action_type="create_exceptional_entry",
        validated_arguments={
            "entry_time": "2026-09-10T17:00:00",
            "entry_type": 2,
            "reason_id": "private-reason-id",
            "mapping_id": "private-mapping-id",
        },
        summary="Submit the exceptional entry?",
        language="en",
    )

    response = client.post(
        "/api/chat",
        json={
            "message": "No",
            "session_id": session_id,
            "confirmation_id": pending.confirmation_id,
        },
    )

    assert response.status_code == 200
    record = asyncio.run(AuditStore(str(database), retention_days=7).recent())[0]
    assert record["action_type"] == "create_exceptional_entry"
    assert record["action_state"] == "rejected"
    assert record["action_result"] == "cancelled"
    assert record["confirmed"] is False
    assert record["confirmation_required"] is False
    raw = database.read_bytes()
    assert pending.confirmation_id.encode() not in raw
    assert b"private-reason-id" not in raw
    assert b"private-mapping-id" not in raw
    assert b"private-session-token" not in raw


def test_confirmed_action_audit_records_safe_result_only(
    monkeypatch,
    audit_tmp_path,
) -> None:
    from app.services import chat as chat_service

    database = audit_tmp_path / "executed-action.db"
    monkeypatch.setattr(
        audit_module,
        "get_settings",
        lambda: audit_settings(database),
    )
    session_store.clear()
    session_id = session_store.ensure_session("execution-session-token")
    pending = session_store.create_pending_action(
        session_id,
        action_type="book_day_type",
        validated_arguments={
            "date_from": "2026-09-22",
            "date_to": "2026-09-22",
            "day_type_id": 99,
        },
        summary="Submit Business Travel?",
        language="en",
    )

    async def execute(action_type, arguments):
        assert action_type == "book_day_type"
        assert arguments["day_type_id"] == 99
        return {"success": True, "message": "Request submitted for approval"}

    async def render(facts, **kwargs):
        return facts

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    monkeypatch.setattr(chat_service, "render_user_message", render)
    response = client.post(
        "/api/chat",
        json={
            "message": "Yes",
            "session_id": session_id,
            "confirmation_id": pending.confirmation_id,
        },
    )

    assert response.status_code == 200
    record = asyncio.run(AuditStore(str(database), retention_days=7).recent())[0]
    assert record["action_type"] == "book_day_type"
    assert record["action_state"] == "executed"
    assert record["action_result"] == "submitted_for_approval"
    assert record["confirmed"] is True
    raw = database.read_bytes()
    assert pending.confirmation_id.encode() not in raw
    assert b"execution-session-token" not in raw
    assert b"day_type_id" not in raw


def test_export_csv_and_json_preserve_safe_structured_trace(
    monkeypatch,
    audit_tmp_path,
) -> None:
    database = audit_tmp_path / "export.db"
    store = AuditStore(str(database), retention_days=7)
    audit = InteractionAudit(
        input_mode="voice",
        input_source="stt",
        user_text="Show my attendance",
        raw_detected_locale="en-US",
        resolved_language="en",
        response_language="en",
        display_message="Attendance is shown.",
        speech_message="Your attendance is shown.",
        tts_text="Your attendance is shown.",
        tts_requested=True,
        tts_generated=True,
        tts_locale="en-US",
        tts_voice="en-US-AvaNeural",
        tools_used=["get_attendance_summary"],
        resourceplus_calls=[
            {
                "method": "GET",
                "endpoint": "api/AI/AttendanceSummary",
                "status": 200,
                "duration_ms": 335.0,
            }
        ],
        model_requests=2,
        latencies={
            "resourceplus": 335.0,
            "openai_main": 420.0,
            "post_release_stt_finalize": 110.0,
            "tts": 240.0,
            "total_after_release": 1200.0,
            "total": 2800.0,
        },
        success=True,
        result_status="success",
    )
    asyncio.run(store.write(audit, store_content=True))
    monkeypatch.setattr(
        export_audit,
        "get_settings",
        lambda: audit_settings(database),
    )
    csv_output = audit_tmp_path / "trace.csv"
    json_output = audit_tmp_path / "trace.json"

    assert export_audit.main(["--format", "csv", "--output", str(csv_output)]) == 0
    assert export_audit.main(["--format", "json", "--output", str(json_output)]) == 0

    csv_rows = list(csv.DictReader(io.StringIO(csv_output.read_text(encoding="utf-8"))))
    assert csv_rows[0]["mode"] == "voice"
    assert csv_rows[0]["detected_locale"] == "en-US"
    assert csv_rows[0]["tts_locale"] == "en-US"
    assert csv_rows[0]["tts_voice"] == "en-US-AvaNeural"
    assert csv_rows[0]["rp_endpoints"] == "api/AI/AttendanceSummary"
    assert csv_rows[0]["resourceplus_ms"] == "335.0"
    assert csv_rows[0]["openai_ms"] == "420.0"
    assert csv_rows[0]["stt_finalize_ms"] == "110.0"
    json_rows = json.loads(json_output.read_text(encoding="utf-8"))
    assert json_rows[0]["resourceplus_calls"] == [
        {
            "method": "GET",
            "endpoint": "api/AI/AttendanceSummary",
            "status": 200,
            "duration_ms": 335.0,
        }
    ]
    assert json_rows[0]["latencies_ms"]["tts"] == 240.0
    assert json_rows[0]["tts_locale"] == "en-US"
    assert json_rows[0]["tts_voice"] == "en-US-AvaNeural"
    assert "latencies" not in json_rows[0]
