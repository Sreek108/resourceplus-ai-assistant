import json
from datetime import date

import pytest
from fastapi.testclient import TestClient

from app.ai import actions, conversation
from app.ai.agent import AgentResult
from app.ai.sessions import session_store
from app.ai.tools import execute_tool
from app.api import voice as voice_module
from app.identity import (
    RequestIdentity,
    bind_request_identity,
    reset_request_identity,
)
from app.main import app
from app.speech import SpeechAudio, SpeechTranscript
from app.services import chat as chat_service
from app.services import fast_reads


client = TestClient(app)
TEST_IDENTITY = RequestIdentity("voice.employee@example.com", "VoiceTest")


@pytest.fixture(autouse=True)
def clear_shared_voice_sessions():
    session_store.clear()
    yield
    session_store.clear()


def _seed_out_reason_draft(session_id: str) -> None:
    token = bind_request_identity(TEST_IDENTITY)
    try:
        session_store.ensure_session(session_id)
        session_store.create_exceptional_entry_draft(
            session_id,
            attendance_date="2026-09-29",
            entry_type="OUT",
            suggested_entry_time="29/09/2026 17:00",
            language="en",
            reason_options=["Outside Work", "Family Circumstances", "Other"],
        )
        session_store.append_history(
            session_id,
            "user",
            "I forgot to punch out today.",
        )
        session_store.append_history(
            session_id,
            "assistant",
            "What was the reason?",
        )
    finally:
        reset_request_identity(token)


def _session_state(session_id: str):
    token = bind_request_identity(TEST_IDENTITY)
    try:
        return (
            session_store.get_exceptional_entry_draft(session_id),
            session_store.get_pending_action(session_id)[0],
            session_store.get_history(session_id),
        )
    finally:
        reset_request_identity(token)


def _identity_fields() -> dict[str, str]:
    return {
        "email": TEST_IDENTITY.email,
        "instance": TEST_IDENTITY.instance,
    }


async def _spoken_audio(*args, **kwargs):
    return SpeechAudio(b"voice-response")


def test_http_voice_buffer_transcript_uses_deterministic_balance(monkeypatch) -> None:
    calls = []

    async def transcribe(*args, **kwargs):
        return SpeechTranscript(
            "How much buffer time do I have?",
            "en-US",
            "en",
        )

    async def balance(target_date):
        calls.append(target_date)
        return {"hasPolicy": True, "limitType": 2, "remaining": 120}

    async def forbidden_write(*args, **kwargs):
        raise AssertionError("a voice balance read must never execute a write")

    def no_model(*args, **kwargs):
        raise AssertionError("an exact voice balance transcript must not call OpenAI")

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    monkeypatch.setattr(voice_module, "synthesize_speech", _spoken_audio)
    monkeypatch.setattr(fast_reads, "resourceplus_today", lambda: date(2026, 10, 1))
    monkeypatch.setattr(fast_reads, "get_exceptional_entry_balance", balance)
    monkeypatch.setattr(chat_service, "execute_pending_action", forbidden_write)
    monkeypatch.setattr(__import__("app.ai.agent", fromlist=["AsyncOpenAI"]), "AsyncOpenAI", no_model)

    response = client.post(
        "/api/voice/chat",
        files={"audio": ("buffer.wav", b"RIFFvoice", "audio/wav")},
        data={"session_id": "voice-buffer", **_identity_fields()},
    )
    body = response.json()

    assert response.status_code == 200
    assert calls == ["2026-10-01"]
    assert body["transcript"] == "How much buffer time do I have?"
    assert body["message"] == "You have 120 minutes of buffer time left."
    assert body["tools_used"] == ["get_exceptional_entry_balance"]


@pytest.mark.parametrize(
    ("transcript", "locale", "language", "reply"),
    [
        ("Hello how are you", "en-US", "en", "I'm doing well, thanks!"),
        ("السلام عليكم", "ar-SA", "ar", "وعليكم السلام! كيف أقدر أساعدك؟"),
    ],
)
def test_http_voice_casual_turn_retains_reason_draft_without_resourceplus(
    monkeypatch,
    transcript: str,
    locale: str,
    language: str,
    reply: str,
) -> None:
    session_id = f"http-casual-{language}"
    _seed_out_reason_draft(session_id)

    async def transcribe(*args, **kwargs):
        return SpeechTranscript(transcript, locale, language)

    async def natural_agent(message, *args, **kwargs):
        assert message == transcript
        assert kwargs["history"] == []
        return AgentResult(reply, [], speech_message=reply)

    async def no_resourceplus(*args, **kwargs):
        raise AssertionError("A casual voice turn must not call ResourcePlus")

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    monkeypatch.setattr(voice_module, "synthesize_speech", _spoken_audio)
    monkeypatch.setattr(chat_service, "run_agent", natural_agent)
    monkeypatch.setattr(actions, "get_missing_punch_suggestions", no_resourceplus)
    monkeypatch.setattr(actions, "get_exception_reasons", no_resourceplus)
    monkeypatch.setattr(actions, "create_exceptional_entry", no_resourceplus)

    response = client.post(
        "/api/voice/chat",
        files={"audio": ("casual.wav", b"RIFFvoice", "audio/wav")},
        data={"session_id": session_id, **_identity_fields()},
    )

    assert response.status_code == 200
    assert response.json()["message"] == reply
    draft, pending, _ = _session_state(session_id)
    assert draft is not None
    assert pending is None


class _StreamingRecognizer:
    def __init__(self, transcript: SpeechTranscript) -> None:
        self.transcript = transcript

    async def start(self):
        return None

    def write(self, chunk):
        assert chunk

    async def finish(self):
        return self.transcript

    async def cancel(self):
        return None


def test_websocket_voice_casual_turn_retains_reason_draft_without_resourceplus(
    monkeypatch,
) -> None:
    session_id = "websocket-casual-en"
    _seed_out_reason_draft(session_id)
    recognizer = _StreamingRecognizer(
        SpeechTranscript("Hello how are you", "en-US", "en")
    )

    async def natural_agent(message, *args, **kwargs):
        assert message == "Hello how are you"
        assert kwargs["history"] == []
        return AgentResult(
            "I'm doing well, thanks!",
            [],
            speech_message="I'm doing well, thanks!",
        )

    async def no_resourceplus(*args, **kwargs):
        raise AssertionError("A casual streaming turn must not call ResourcePlus")

    async def no_audit(*args, **kwargs):
        return None

    monkeypatch.setattr(voice_module, "StreamingSpeechRecognizer", lambda: recognizer)
    monkeypatch.setattr(voice_module, "synthesize_speech", _spoken_audio)
    monkeypatch.setattr(voice_module, "persist_audit", no_audit)
    monkeypatch.setattr(chat_service, "run_agent", natural_agent)
    monkeypatch.setattr(actions, "get_missing_punch_suggestions", no_resourceplus)
    monkeypatch.setattr(actions, "get_exception_reasons", no_resourceplus)
    monkeypatch.setattr(actions, "create_exceptional_entry", no_resourceplus)

    with client.websocket_connect("/api/voice/stream") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "sample_rate": 16_000,
                "session_id": session_id,
                **_identity_fields(),
            }
        )
        assert websocket.receive_json() == {"type": "ready"}
        websocket.send_bytes(b"\x00\x00" * 320)
        websocket.send_json({"type": "end"})
        result = websocket.receive_json()

    assert result["type"] == "final"
    assert result["message"] == "I'm doing well, thanks!"
    draft, pending, _ = _session_state(session_id)
    assert draft is not None
    assert pending is None


def test_http_voice_valid_reason_continues_retained_draft(monkeypatch) -> None:
    session_id = "http-valid-voice-reason"
    _seed_out_reason_draft(session_id)
    calls = {"suggestions": 0, "reasons": 0, "writes": 0}

    async def transcribe(*args, **kwargs):
        return SpeechTranscript("Outside Work", "en-US", "en")

    async def suggestions(*args, **kwargs):
        calls["suggestions"] += 1
        return [
            {
                "attDate": "29/09/2026",
                "suggestedEntryTime": "29/09/2026 17:00",
                "entryType": "OUT",
            }
        ]

    async def reasons(*args, **kwargs):
        calls["reasons"] += 1
        return [{"reasonID": "outside-live-id", "reasonName": "Outside Work"}]

    async def no_write(**kwargs):
        calls["writes"] += 1
        raise AssertionError("Reason selection must not execute the write")

    async def no_agent(*args, **kwargs):
        raise AssertionError("A live reason must use the deterministic draft flow")

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    monkeypatch.setattr(voice_module, "synthesize_speech", _spoken_audio)
    monkeypatch.setattr(chat_service, "run_agent", no_agent)
    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    monkeypatch.setattr(actions, "create_exceptional_entry", no_write)

    response = client.post(
        "/api/voice/chat",
        files={"audio": ("reason.wav", b"RIFFvoice", "audio/wav")},
        data={"session_id": session_id, **_identity_fields()},
    )

    assert response.status_code == 200
    assert response.json()["requires_confirmation"] is True
    draft, pending, _ = _session_state(session_id)
    assert draft is None
    assert pending is not None
    assert pending.validated_arguments["entry_type"] == 2
    assert pending.validated_arguments["reason_id"] == "outside-live-id"
    assert calls == {"suggestions": 1, "reasons": 1, "writes": 0}


def test_http_voice_new_hr_intent_permanently_clears_old_draft(monkeypatch) -> None:
    session_id = "http-new-profile-intent"
    _seed_out_reason_draft(session_id)

    async def transcribe(*args, **kwargs):
        return SpeechTranscript("Show my profile", "en-US", "en")

    async def profile_agent(message, *args, **kwargs):
        assert message == "Show my profile"
        assert kwargs["history"] == []
        return AgentResult(
            "Your profile is ready.",
            ["get_profile_data"],
            speech_message="Your profile is ready.",
        )

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    monkeypatch.setattr(voice_module, "synthesize_speech", _spoken_audio)
    monkeypatch.setattr(chat_service, "run_agent", profile_agent)

    response = client.post(
        "/api/voice/chat",
        files={"audio": ("profile.wav", b"RIFFvoice", "audio/wav")},
        data={"session_id": session_id, **_identity_fields()},
    )

    assert response.status_code == 200
    draft, pending, history = _session_state(session_id)
    assert draft is None
    assert pending is None
    assert all("forgot to punch out" not in item["content"] for item in history)


@pytest.mark.parametrize(
    ("transcript", "locale", "language"),
    [
        ("I forgot to Punjab Today.", "en-US", "en"),
        ("نسيت أسجل اليوم", "ar-SA", "ar"),
    ],
)
def test_http_voice_ambiguous_transcript_cannot_start_transaction(
    monkeypatch,
    transcript: str,
    locale: str,
    language: str,
) -> None:
    session_id = f"ambiguous-voice-{language}"

    async def transcribe(*args, **kwargs):
        return SpeechTranscript(transcript, locale, language)

    async def forbidden(*args, **kwargs):
        raise AssertionError("Ambiguous STT must not reach model or ResourcePlus")

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    monkeypatch.setattr(voice_module, "synthesize_speech", _spoken_audio)
    monkeypatch.setattr(chat_service, "run_agent", forbidden)
    monkeypatch.setattr(actions, "get_missing_punch_suggestions", forbidden)
    monkeypatch.setattr(actions, "get_exception_reasons", forbidden)
    monkeypatch.setattr(actions, "create_exceptional_entry", forbidden)

    response = client.post(
        "/api/voice/chat",
        files={"audio": ("unclear.wav", b"RIFFvoice", "audio/wav")},
        data={"session_id": session_id, **_identity_fields()},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["requires_confirmation"] is False
    assert body["tools_used"] == []
    draft, pending, _ = _session_state(session_id)
    assert draft is None
    assert pending is None


@pytest.mark.parametrize(
    ("transcript", "locale", "language", "expected_direction"),
    [
        ("I forgot to punch in today", "en-US", "en", "IN"),
        ("I forgot to punch out today", "en-US", "en", "OUT"),
        ("نسيت أسجل دخول اليوم", "ar-SA", "ar", "IN"),
        ("نسيت أسجل خروج اليوم", "ar-SA", "ar", "OUT"),
    ],
)
def test_http_voice_explicit_direction_remains_transaction_eligible(
    monkeypatch,
    transcript: str,
    locale: str,
    language: str,
    expected_direction: str,
) -> None:
    session_id = f"voice-direction-{language}-{expected_direction}"
    reads = {"attendance": 0, "reasons": 0, "writes": 0}

    async def transcribe(*args, **kwargs):
        return SpeechTranscript(transcript, locale, language)

    async def attendance(*args, **kwargs):
        reads["attendance"] += 1
        return {"Days": [{
            "AttDate": "29/09/2026",
            "DayType": "Regular",
            "CheckIN": "09:10",
            "CheckOut": "17:00",
            "NetHrs": "07:50",
            "LessHrs": "00:10",
        }]}

    async def reasons(*args, **kwargs):
        reads["reasons"] += 1
        return [{"reasonID": "outside-live-id", "reasonName": "Outside Work"}]

    async def balance(*args, **kwargs):
        return {"hasPolicy": True, "remaining": 120}

    async def no_write(**kwargs):
        reads["writes"] += 1
        raise AssertionError("Preparing a draft must not execute a write")

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    monkeypatch.setattr(voice_module, "synthesize_speech", _spoken_audio)
    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 29))
    monkeypatch.setattr(actions, "get_attendance_summary", attendance)
    monkeypatch.setattr(conversation, "cached_exception_reasons", reasons)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)
    monkeypatch.setattr(actions, "create_exceptional_entry_from_summary", no_write)

    response = client.post(
        "/api/voice/chat",
        files={"audio": ("direction.wav", b"RIFFvoice", "audio/wav")},
        data={"session_id": session_id, **_identity_fields()},
    )

    assert response.status_code == 200
    token = bind_request_identity(TEST_IDENTITY)
    try:
        draft = session_store.get_conversation_draft(session_id)
        pending = session_store.get_pending_action(session_id)[0]
    finally:
        reset_request_identity(token)
    assert draft is not None and draft.intent == "less_hours_correction"
    assert draft.slots["entry_type"] == ("1" if expected_direction == "IN" else "2")
    assert pending is None
    assert reads == {"attendance": 1, "reasons": 1, "writes": 0}
