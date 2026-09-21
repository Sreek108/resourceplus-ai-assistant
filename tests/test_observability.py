import logging
import time
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient

from app.api import voice as voice_module
from app.main import app
from app.models.schemas import ChatResponse
from app.observability import (
    VoiceLatencyTrace,
    measure_model_call,
    measure_stage,
    reset_voice_trace,
    start_voice_trace,
)
from app.resourceplus.client import ResourcePlusClient
from app.speech import SpeechAudio, SpeechRecognitionError, SpeechTranscript


client = TestClient(app)


def _latency_messages(caplog) -> list[str]:
    return [
        record.getMessage()
        for record in caplog.records
        if record.name == "app.voice_latency" and "VOICE_LATENCY" in record.getMessage()
    ]


def test_timing_stages_accumulate_independently() -> None:
    trace, token = start_voice_trace()
    try:
        with measure_stage("stt"):
            time.sleep(0.002)
        with measure_stage("agent"):
            time.sleep(0.003)
        with measure_stage("tts"):
            time.sleep(0.004)
        with measure_model_call("openai_main"):
            time.sleep(0.001)
        with measure_model_call("openai_main"):
            time.sleep(0.001)
        summary = trace.format_summary(status_code=200)
    finally:
        reset_voice_trace(token)

    assert set(trace.durations) == {"stt", "agent", "tts", "openai_main"}
    assert trace.durations["stt"] >= 0.002
    assert trace.durations["agent"] >= 0.003
    assert trace.durations["tts"] >= 0.004
    assert "total=" in summary
    assert "stt=" in summary
    assert "agent=" in summary
    assert "tts=" in summary
    assert trace.model_requests == 2
    assert "model_requests=2" in summary
    assert "stream_error_category=none" in summary


def test_voice_timing_preserves_response_and_does_not_log_sensitive_values(
    monkeypatch,
    caplog,
) -> None:
    secret_session = "session-secret-value"
    secret_confirmation = "confirmation-secret-value"

    async def transcribe(*args, **kwargs):
        with measure_stage("stt"):
            pass
        with measure_stage("language_resolution"):
            pass
        return SpeechTranscript("Show my notifications", "en-US", "en")

    async def process(request, *, detected_language):
        return ChatResponse(
            success=True,
            message="You have two notifications.",
            language=detected_language,
            session_id=request.session_id,
        ).set_speech_message("You have two notifications.")

    async def synthesize(text, *, language):
        del text, language
        return SpeechAudio(b"RIFFresponse")

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    monkeypatch.setattr(voice_module, "process_chat", process)
    monkeypatch.setattr(voice_module, "synthesize_speech", synthesize)
    caplog.set_level(logging.INFO, logger="app.voice_latency")

    response = client.post(
        "/api/voice/chat",
        files={"audio": ("utterance.wav", b"RIFFinput", "audio/wav")},
        data={
            "session_id": secret_session,
            "confirmation_id": secret_confirmation,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["message"] == "You have two notifications."
    assert body["session_id"] == secret_session
    assert not any(key.startswith("timing") for key in body)
    messages = _latency_messages(caplog)
    assert len(messages) == 1
    assert "status=200" in messages[0]
    assert "stt=" in messages[0]
    assert "language_resolution=" in messages[0]
    assert "agent=" in messages[0]
    assert "voice_summary=" in messages[0]
    assert "voice_summary=0.000s" in messages[0]
    assert "follow_up_classifier=0.000s" in messages[0]
    assert "tts=" in messages[0]
    assert secret_session not in messages[0]
    assert secret_confirmation not in messages[0]


def test_failed_voice_stage_still_logs_timing(monkeypatch, caplog) -> None:
    async def transcribe(*args, **kwargs):
        with measure_stage("stt"):
            raise SpeechRecognitionError("Speech recognition failed safely.")

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    caplog.set_level(logging.INFO, logger="app.voice_latency")

    response = client.post(
        "/api/voice/chat",
        files={"audio": ("utterance.wav", b"RIFFinput", "audio/wav")},
    )

    assert response.status_code == 502
    messages = _latency_messages(caplog)
    assert len(messages) == 1
    assert "status=502" in messages[0]
    assert "stt=" in messages[0]
    assert "total=" in messages[0]


@pytest.mark.asyncio
async def test_resourceplus_timing_logs_only_safe_path(monkeypatch, caplog) -> None:
    secret_email = "employee-secret@example.com"
    secret_token = "backend-secret-token"

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"ok": True})

    monkeypatch.setattr(
        "app.resourceplus.client.get_settings",
        lambda: SimpleNamespace(
            rp_base_url="https://example.test/Mobile/",
            rp_timeout_seconds=10,
        ),
    )
    resourceplus = ResourcePlusClient(transport=httpx.MockTransport(handler))
    caplog.set_level(logging.INFO, logger="app.resourceplus.client")

    result = await resourceplus.get(
        "api/Client/GetHomeData",
        params={"Usremail": secret_email, "token": secret_token},
    )

    assert result == {"ok": True}
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == "app.resourceplus.client" and "RP_API" in record.getMessage()
    ]
    assert len(messages) == 1
    assert "method=GET" in messages[0]
    assert "endpoint=api/Client/GetHomeData" in messages[0]
    assert "status=200" in messages[0]
    assert secret_email not in messages[0]
    assert secret_token not in messages[0]


def test_summary_format_cannot_accept_secret_named_fields() -> None:
    trace = VoiceLatencyTrace()

    with pytest.raises(ValueError, match="Unsupported voice latency stage"):
        trace.add_duration("authorization=backend-secret", 1.0)

    summary = trace.format_summary(status_code=500)
    assert "backend-secret" not in summary


def test_voice_error_category_is_restricted_to_safe_allowlist() -> None:
    trace = VoiceLatencyTrace()
    trace.set_error_category("authorization=backend-secret")

    summary = trace.format_summary(status_code=500)

    assert "stream_error_category=unknown_safe_category" in summary
    assert "backend-secret" not in summary
