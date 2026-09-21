import asyncio
import json
import logging
import shutil
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import pytest
from fastapi.testclient import TestClient

from app import audit as audit_module
from app.ai.agent import OpenAIServiceError
from app.api import chat as chat_module
from app.api import health as health_module
from app.api import operations as operations_module
from app.api import telemetry as frontend_telemetry_module
from app.api import voice as voice_module
from app.audit import (
    AuditStore,
    InteractionAudit,
    reset_interaction_audit,
    start_interaction_audit,
)
from app.main import app
from app.models.schemas import ChatResponse
from app.observability import measure_model_call, measure_stage
from app.resourceplus.client import ResourcePlusClient
from app.speech import SpeechAudio, SpeechTranscript
from app.telemetry import (
    TRACE_HEADER,
    classify_error,
    emit_structured_event,
    metrics,
    record_http_metrics,
    record_action_metric,
    record_resourceplus_metrics,
    record_websocket_failure,
    reset_trace,
    start_trace,
)
from scripts import diagnose_trace


client = TestClient(app)


@pytest.fixture
def observability_tmp_path():
    path = Path(".test-artifacts") / f"observability-{uuid4()}"
    path.mkdir(parents=True)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


def _audit_settings(path: Path, *, enabled: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        ai_audit_enabled=enabled,
        ai_audit_store_content=True,
        ai_audit_retention_days=7,
        ai_audit_db_path=str(path),
        frontend_telemetry_rate_limit_per_minute=120,
        app_version="1.2.3",
        git_commit="abc123",
        deployment_id="uat-1",
        app_environment="uat",
    )


def test_trace_header_is_propagated_and_generated() -> None:
    supplied = "uat-trace-123456"
    propagated = client.get("/health", headers={TRACE_HEADER: supplied})
    generated = client.get("/health")

    assert propagated.status_code == 200
    assert propagated.headers[TRACE_HEADER] == supplied
    assert len(generated.headers[TRACE_HEADER]) == 36
    assert generated.headers[TRACE_HEADER] != supplied


def test_invalid_trace_header_is_rejected_safely() -> None:
    response = client.get(
        "/health",
        headers={TRACE_HEADER: "employee@example.com?token=secret"},
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "Invalid X-Trace-ID header."}
    assert len(response.headers[TRACE_HEADER]) == 36
    assert "example.com" not in response.headers[TRACE_HEADER]


def test_cors_allows_and_exposes_trace_header() -> None:
    response = client.options(
        "/api/chat",
        headers={
            "Origin": "http://127.0.0.1:5173",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type,x-trace-id",
        },
    )
    assert response.status_code == 200
    assert "x-trace-id" in response.headers["access-control-allow-headers"].casefold()
    actual = client.get(
        "/health",
        headers={"Origin": "http://127.0.0.1:5173"},
    )
    assert "x-trace-id" in actual.headers["access-control-expose-headers"].casefold()


def test_interaction_audit_captures_trace_and_deployment_metadata(
    monkeypatch,
    observability_tmp_path,
) -> None:
    database = observability_tmp_path / "trace-audit.db"
    settings = _audit_settings(database)
    monkeypatch.setattr(audit_module, "get_settings", lambda: settings)

    async def process(request):
        return ChatResponse(
            success=True,
            message="Hello",
            language="en",
            session_id="private-session",
        ).set_speech_message("Hello")

    monkeypatch.setattr(chat_module, "process_chat", process)
    response = client.post(
        "/api/chat",
        headers={TRACE_HEADER: "audit-trace-1234"},
        json={"message": "Hello"},
    )

    assert response.status_code == 200
    record = asyncio.run(AuditStore(str(database), retention_days=7).recent())[0]
    assert record["trace_id"] == "audit-trace-1234"
    assert record["app_version"] == "1.2.3"
    assert record["git_commit"] == "abc123"
    assert record["environment"] == "uat"
    assert record["deployment_id"] == "uat-1"


def test_interaction_audit_persists_normalized_error_owner_and_stage(
    monkeypatch,
    observability_tmp_path,
) -> None:
    database = observability_tmp_path / "error-owner.db"
    monkeypatch.setattr(
        audit_module,
        "get_settings",
        lambda: _audit_settings(database),
    )

    async def fail(request):
        with measure_model_call("openai_main"):
            raise OpenAIServiceError("Safe assistant failure")

    monkeypatch.setattr(chat_module, "process_chat", fail)
    response = client.post("/api/chat", json={"message": "Hello"})
    assert response.status_code == 502
    record = asyncio.run(AuditStore(str(database), retention_days=7).recent())[0]
    assert record["error_category"] == "openai_error"
    assert record["error_owner"] == "openai"
    assert record["error_stage"] == "openai_main"


def test_resourceplus_receives_only_safe_correlation_header() -> None:
    seen = {}

    async def handler(request: httpx.Request) -> httpx.Response:
        seen.update(request.headers)
        return httpx.Response(200, json={"success": False, "message": "Duplicate booking"})

    resourceplus = ResourcePlusClient(
        base_url="https://example.test/Mobile/",
        transport=httpx.MockTransport(handler),
    )
    _, token = start_trace("resourceplus-trace-1")
    try:
        result = asyncio.run(
            resourceplus.get(
                "api/AI/Test",
                params={"usrEmail": "private@example.com", "token": "private-token"},
            )
        )
    finally:
        reset_trace(token)

    assert result["success"] is False
    assert seen["x-correlation-id"] == "resourceplus-trace-1"
    assert "usrEmail" not in str(seen)
    assert "private@example.com" not in str(seen)


def test_websocket_trace_reaches_voice_audit(monkeypatch) -> None:
    class Recognizer:
        async def start(self):
            return None

        def write(self, chunk):
            assert chunk

        async def finish(self):
            return SpeechTranscript("Hello", "en-US", "en")

        async def cancel(self):
            return None

    captured = []

    async def process(request, *, detected_language):
        return ChatResponse(
            success=True,
            message="Hello",
            language=detected_language,
            session_id="session",
        ).set_speech_message("Hello")

    async def synthesize(text, *, language):
        return SpeechAudio(b"audio", locale="en-US", voice_name="en-US-AvaNeural")

    async def persist(audit):
        captured.append(audit)

    monkeypatch.setattr(voice_module, "StreamingSpeechRecognizer", Recognizer)
    monkeypatch.setattr(voice_module, "process_chat", process)
    monkeypatch.setattr(voice_module, "synthesize_speech", synthesize)
    monkeypatch.setattr(voice_module, "persist_audit", persist)

    with client.websocket_connect(
        "/api/voice/stream",
        headers={TRACE_HEADER: "voice-trace-1234"},
    ) as websocket:
        websocket.send_json({"type": "start", "sample_rate": 16_000})
        assert websocket.receive_json() == {"type": "ready"}
        websocket.send_bytes(b"\x00\x00" * 320)
        websocket.send_json({"type": "end"})
        assert websocket.receive_json()["type"] == "final"

    assert len(captured) == 1
    assert captured[0].trace_id == "voice-trace-1234"


def test_structured_logging_redacts_sensitive_values(caplog) -> None:
    _, token = start_trace("structured-trace-1")
    caplog.set_level(logging.INFO, logger="app.structured")
    try:
        emit_structured_event(
            component="http",
            event="Bearer-private-secret-token",
            endpoint="api/AI/Test?usrEmail=private@example.com",
            status=500,
            error_category="resourceplus_error",
            error_owner="resourceplus_api",
        )
    finally:
        reset_trace(token)

    message = caplog.records[-1].getMessage()
    parsed = json.loads(message)
    assert parsed["trace_id"] == "structured-trace-1"
    assert parsed["event"] == "[redacted]"
    assert parsed["endpoint"] == "[redacted]"
    assert "private" not in message
    assert "example.com" not in message


def test_prometheus_metrics_are_bounded_and_contain_no_high_cardinality_labels() -> None:
    metrics.reset()
    record_http_metrics(method="POST", path="/api/chat", status_code=200, duration_ms=42)
    record_resourceplus_metrics(method="GET", status=200, duration_ms=20)
    rendered = metrics.render()

    assert 'resourceplus_assistant_requests_total{mode="text",status="success"} 1' in rendered
    assert "resourceplus_assistant_http_request_duration_ms_bucket" in rendered
    assert 'resourceplus_assistant_resourceplus_requests_total{method="GET",status="success"} 1' in rendered
    for prohibited in ("trace_id", "session_id", "employee", "usrEmail", "resourceplus-trace"):
        assert prohibited not in rendered

    response = client.get("/metrics", headers={TRACE_HEADER: "metrics-trace-123"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")


def test_provider_voice_websocket_and_pending_action_metrics_are_exported() -> None:
    metrics.reset()
    audit, audit_token = start_interaction_audit(input_mode="voice")
    try:
        with measure_stage("stt"):
            pass
        with measure_model_call("openai_main"):
            pass
        with measure_stage("tts"):
            pass
    finally:
        reset_interaction_audit(audit_token)
    assert audit.model_requests == 1
    record_websocket_failure("websocket_disconnected")
    for state in ("prepared", "confirmed", "rejected", "expired", "executed", "failed"):
        record_action_metric(state)

    rendered = metrics.render()
    assert "resourceplus_assistant_openai_request_duration_ms_bucket" in rendered
    assert "resourceplus_assistant_azure_stt_duration_ms_bucket" in rendered
    assert "resourceplus_assistant_azure_tts_duration_ms_bucket" in rendered
    assert 'resourceplus_assistant_websocket_failures_total{category="websocket_disconnected"} 1' in rendered
    for state in ("prepared", "confirmed", "rejected", "expired", "executed", "failed"):
        assert f'state="{state}"' in rendered


def test_health_and_readiness_are_safe_local_checks(monkeypatch) -> None:
    settings = SimpleNamespace(
        openai_api_key="secret-openai-key",
        openai_model="model",
        rp_base_url="https://example.test/Mobile/",
        rp_instance="Universal",
        rp_default_email="private@example.com",
        azure_speech_key=None,
        azure_speech_region=None,
    )
    monkeypatch.setattr(health_module, "get_settings", lambda: settings)

    health = client.get("/health")
    ready = client.get("/ready")

    assert health.json() == {"status": "ok"}
    assert ready.status_code == 200
    assert ready.json() == {
        "status": "ready",
        "components": {"openai": True, "resourceplus": True, "voice": False},
    }
    assert "secret-openai-key" not in ready.text
    assert "private@example.com" not in ready.text


def test_readiness_reports_missing_core_configuration_without_provider_calls(monkeypatch) -> None:
    monkeypatch.setattr(
        health_module,
        "get_settings",
        lambda: SimpleNamespace(
            openai_api_key=None,
            openai_model=None,
            rp_base_url="https://example.test/",
            rp_instance="Universal",
            rp_default_email="configured",
            azure_speech_key=None,
            azure_speech_region=None,
        ),
    )
    response = client.get("/ready")
    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_operations_diagnostics_is_disabled_by_default_and_protected(monkeypatch) -> None:
    disabled = SimpleNamespace(ops_diagnostics_enabled=False, ops_diagnostics_token=None)
    monkeypatch.setattr(operations_module, "get_settings", lambda: disabled)
    assert client.get("/api/ops/diagnostics").status_code == 404

    enabled = SimpleNamespace(
        ops_diagnostics_enabled=True,
        ops_diagnostics_token="private-ops-key",
        app_environment="uat",
        deployment_id="uat-1",
        app_version="1.2.3",
        git_commit="abc123",
        ai_audit_enabled=True,
        ai_audit_store_content=False,
        observability_exporter="none",
        openai_api_key="private-openai-key",
        openai_model="model",
        rp_base_url="https://example.test/",
        rp_instance="Universal",
        rp_default_email="private@example.com",
        azure_speech_key="private-azure-key",
        azure_speech_region="region",
    )
    monkeypatch.setattr(operations_module, "get_settings", lambda: enabled)
    assert client.get("/api/ops/diagnostics").status_code == 401
    response = client.get(
        "/api/ops/diagnostics",
        headers={"X-Ops-Key": "private-ops-key"},
    )
    assert response.status_code == 200
    assert response.json()["deployment_id"] == "uat-1"
    for secret in ("private-ops-key", "private-openai-key", "private-azure-key", "private@example.com"):
        assert secret not in response.text


def test_frontend_telemetry_is_allowlisted_and_uses_existing_audit_database(
    monkeypatch,
    observability_tmp_path,
) -> None:
    database = observability_tmp_path / "frontend.db"
    settings = _audit_settings(database)
    monkeypatch.setattr(frontend_telemetry_module, "get_settings", lambda: settings)
    response = client.post(
        "/api/telemetry/frontend",
        headers={TRACE_HEADER: "frontend-trace-1"},
        json={
            "trace_id": "frontend-trace-1",
            "event": "response_received",
            "duration_ms": 215.5,
            "http_status": 200,
        },
    )

    assert response.status_code == 202
    assert response.json() == {"accepted": True}
    events = asyncio.run(AuditStore(str(database), retention_days=7).frontend_events("frontend-trace-1"))
    assert events == [
        {
            "timestamp": events[0]["timestamp"],
            "trace_id": "frontend-trace-1",
            "event": "response_received",
            "duration_ms": 215.5,
            "error_category": None,
            "http_status": 200,
        }
    ]


def test_invalid_or_arbitrary_frontend_telemetry_is_rejected() -> None:
    arbitrary = client.post(
        "/api/telemetry/frontend",
        json={"event": "console_log", "stack": "secret stack trace"},
    )
    secret_category = client.post(
        "/api/telemetry/frontend",
        json={"event": "network_error", "error_category": "Bearer secret"},
    )

    assert arbitrary.status_code == 422
    assert secret_category.status_code == 422

    oversized = client.post(
        "/api/telemetry/frontend",
        content=json.dumps({"event": "render_error", "padding": "x" * 3_000}),
        headers={"Content-Type": "application/json"},
    )
    assert oversized.status_code == 413


def test_error_owner_taxonomy_and_business_result_classification() -> None:
    assert classify_error("openai_error") == ("openai", "openai_main")
    assert classify_error("azure_canceled") == ("azure_stt", "stt")
    assert classify_error("resourceplus_error") == ("resourceplus_api", "resourceplus")
    assert classify_error("confirmation_expired") == ("transaction", "transaction")

    metrics.reset()
    # A valid HTTP 200 business-rule response is provider success, regardless of
    # the business payload's own success flag.
    record_resourceplus_metrics(method="POST", status=200, duration_ms=10)
    rendered = metrics.render()
    assert 'method="POST",status="success"' in rendered
    assert 'method="POST",status="failure"' not in rendered


def test_diagnostic_cli_prints_safe_timeline_without_content_or_internal_ids(
    monkeypatch,
    observability_tmp_path,
    capsys,
) -> None:
    database = observability_tmp_path / "diagnostic.db"
    store = AuditStore(str(database), retention_days=7)
    audit = InteractionAudit(
        input_mode="voice",
        trace_id="diagnostic-trace-1",
        user_text="private employee content",
        session_reference="hashed-session",
        resolved_language="ar",
        resourceplus_calls=[
            {"method": "GET", "endpoint": "api/AI/AttendanceSummary", "status": 200, "duration_ms": 25.0}
        ],
        latencies={"stt": 12.0, "openai_main": 30.0, "tts": 14.0, "total": 90.0},
        result_status="success",
        action_type="book_day_type",
        action_state="rejected",
    )
    asyncio.run(store.write(audit, store_content=True))
    asyncio.run(
        store.write_frontend_event(
            trace_id="diagnostic-trace-1",
            event="microphone_released",
            duration_ms=800.0,
            error_category=None,
            http_status=None,
        )
    )
    monkeypatch.setattr(
        diagnose_trace,
        "get_settings",
        lambda: _audit_settings(database),
    )

    assert diagnose_trace.main(["--trace-id", "diagnostic-trace-1", "--json"]) == 0
    output = capsys.readouterr().out
    parsed = json.loads(output)
    assert parsed["trace_id"] == "diagnostic-trace-1"
    assert any(item.get("endpoint") == "api/AI/AttendanceSummary" for item in parsed["timeline"])
    assert "private employee content" not in output
    assert "hashed-session" not in output
    assert "confirmation_id" not in output


def test_observability_helpers_fail_open(monkeypatch) -> None:
    def fail(*args, **kwargs):
        raise RuntimeError("telemetry unavailable")

    monkeypatch.setattr(metrics, "increment", fail)
    monkeypatch.setattr(metrics, "observe", fail)
    record_http_metrics(method="GET", path="/health", status_code=200, duration_ms=1)
    record_resourceplus_metrics(method="GET", status=200, duration_ms=1)
