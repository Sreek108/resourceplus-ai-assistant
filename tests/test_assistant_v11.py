from __future__ import annotations

import asyncio
import json
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from app.ai import actions, conversation
from app.ai.agent import AgentResult
from app.ai.conversation import extract_date
from app.ai.sessions import InMemorySessionStore
from app.identity import RequestIdentity, bind_request_identity, reset_request_identity
from app.main import app
from app.models.schemas import ChatRequest
from app.resourceplus import reference_cache as cache_module
from app.resourceplus.reference_cache import ReferenceDataCache, reference_data_cache
from app.services import chat as chat_service
from app.services import fast_reads
from app.api import voice as voice_module
from app.models.schemas import ChatResponse
from app.speech import SpeechAudio, SpeechSynthesisError, SpeechTranscript
from tests.test_voice_stream import FakeStreamingRecognizer


client = TestClient(app)


@pytest.fixture
def identity():
    token = bind_request_identity(RequestIdentity("employee@example.com", "Universal"))
    try:
        yield
    finally:
        reset_request_identity(token)


@pytest.mark.asyncio
async def test_missing_punch_slots_continue_without_reasking_direction(monkeypatch) -> None:
    store = InMemorySessionStore()

    async def suggestions(start, end, **kwargs):
        assert (start, end) == ("2026-09-06", "2026-09-06")
        return [{
            "attDate": "06/09/2026",
            "suggestedEntryTime": "06/09/2026 09:00",
            "entryType": "IN",
            "shift": "General",
            "isNightShift": 0,
        }]

    async def reasons(*args, **kwargs):
        return [{"reasonID": "outside-live", "reasonName": "Outside Work"}]

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)

    first = await chat_service.process_chat(
        ChatRequest(message="I want to correct a missing punch"), store=store
    )
    second = await chat_service.process_chat(
        ChatRequest(message="IN on 6 September", session_id=first.session_id), store=store
    )

    assert "date" in first.message.lower() and "in or out" in first.message.lower()
    assert "reason" in second.message.lower()
    assert "in or out" not in second.message.lower()
    assert second.needs_reason is True
    assert second.blocks[0].type == "actions"


@pytest.mark.asyncio
async def test_reason_follow_up_creates_confirmation_not_write(monkeypatch) -> None:
    store = InMemorySessionStore()
    writes = []

    async def suggestions(*args, **kwargs):
        return [{
            "attDate": "06/09/2026",
            "suggestedEntryTime": "06/09/2026 09:00",
            "entryType": "IN",
            "shift": "General",
            "isNightShift": 0,
        }]

    async def reasons(*args, **kwargs):
        return [{"reasonID": "outside-live", "reasonName": "Outside Work"}]

    async def forbidden_write(**kwargs):
        writes.append(kwargs)

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    monkeypatch.setattr(actions, "create_exceptional_entry", forbidden_write)

    first = await chat_service.process_chat(
        ChatRequest(message="Fix my missing IN punch on 6 September"), store=store
    )
    prepared = await chat_service.process_chat(
        ChatRequest(message="Outside Work", session_id=first.session_id), store=store
    )

    assert prepared.requires_confirmation is True
    assert prepared.confirmation_id
    assert prepared.blocks[0].type == "confirmation"
    assert writes == []


@pytest.mark.asyncio
async def test_compensatory_leave_tomorrow_resolves_all_slots(monkeypatch) -> None:
    reference_data_cache.clear()
    day_types = [{"dayID": 30, "dayType": "Compensatory Leave", "group": "Leave"}]

    async def cached_types(*args, **kwargs):
        return day_types

    monkeypatch.setattr(cache_module, "get_day_types", cached_types)
    monkeypatch.setattr(actions, "get_day_types", cached_types)
    response = await chat_service.process_chat(
        ChatRequest(message="I need compensatory leave tomorrow"),
        store=InMemorySessionStore(),
    )

    assert response.requires_confirmation is True
    assert "Compensatory Leave" in response.message
    assert "Which" not in response.message
    assert response.blocks[0].type == "confirmation"


@pytest.mark.asyncio
async def test_casual_interruption_preserves_non_executable_draft() -> None:
    store = InMemorySessionStore()
    first = await chat_service.process_chat(
        ChatRequest(message="I want to correct a missing punch"), store=store
    )
    before = store.get_conversation_draft(first.session_id)
    reply = await chat_service.process_chat(
        ChatRequest(message="Thanks", session_id=first.session_id), store=store
    )
    after = store.get_conversation_draft(first.session_id)
    assert reply.message == "You're welcome."
    assert before is not None and after is not None
    assert after.intent == before.intent and after.slots == before.slots


@pytest.mark.asyncio
async def test_new_hr_intent_clears_old_non_executable_draft(monkeypatch) -> None:
    store = InMemorySessionStore()
    first = await chat_service.process_chat(
        ChatRequest(message="I want to correct a missing punch"), store=store
    )

    async def profile(*args, **kwargs):
        return {"EmployeeName": "Demo Employee"}

    monkeypatch.setattr(fast_reads, "get_profile_data", profile)
    response = await chat_service.process_chat(
        ChatRequest(message="Show my profile", session_id=first.session_id), store=store
    )
    assert response.blocks[0].type == "key_value"
    assert store.get_conversation_draft(first.session_id) is None


@pytest.mark.asyncio
async def test_profile_fast_path_uses_one_read_and_no_model(monkeypatch) -> None:
    reads = 0

    async def profile(*args, **kwargs):
        nonlocal reads
        reads += 1
        return {"EmployeeName": "Demo Employee", "Department": "HR"}

    async def no_model(*args, **kwargs):
        raise AssertionError("high-confidence profile read must not invoke the model")

    monkeypatch.setattr(fast_reads, "get_profile_data", profile)
    monkeypatch.setattr(chat_service, "run_agent", no_model)
    # The service-level monkeypatch deliberately exercises the existing injection
    # seam, so call the real agent for this routing assertion.
    result = await __import__("app.ai.agent", fromlist=["run_agent"]).run_agent(
        "Show my profile", lang=1, session_id="profile-fast", history=[]
    )
    assert reads == 1
    assert result.blocks and result.blocks[0].type == "key_value"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "How much buffer time do I have?",
        "How much cover time do I have?",
        "How much allowance time do I have?",
        "How many allowance minutes do I have?",
        "Show my buffer balance",
        "How many excuse minutes do I have?",
        "Show my remaining attendance allowance",
        "What is my exceptional-entry allowance?",
        "How much allowance is left this week?",
    ],
)
async def test_balance_fast_path_uses_zero_model_calls(monkeypatch, message) -> None:
    calls = []

    async def balance(target_date):
        calls.append(target_date)
        return {
            "hasPolicy": True,
            "policyName": "Attendance allowance",
            "limitType": 2,
            "limitValue": 120,
            "used": 20,
            "remaining": 100.0,
            "periodStart": "2026-09-28",
            "periodEnd": "2026-10-04",
            "resetsOn": "2026-10-05",
        }

    def no_model(*args, **kwargs):
        raise AssertionError("clear balance reads must not construct an OpenAI client")

    monkeypatch.setattr(fast_reads, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(fast_reads, "get_exceptional_entry_balance", balance)
    monkeypatch.setattr(__import__("app.ai.agent", fromlist=["AsyncOpenAI"]), "AsyncOpenAI", no_model)

    result = await __import__("app.ai.agent", fromlist=["run_agent"]).run_agent(
        message, lang=1, session_id="balance-fast", history=[]
    )

    assert calls == ["2026-09-30"]
    assert result.tools_used == ["get_exceptional_entry_balance"]
    assert result.message == "You have 100 minutes remaining."
    assert result.speech_message == result.message
    assert result.blocks and result.blocks[0].type == "key_value"
    assert {item.label: item.value for item in result.blocks[0].items}["Unit"] == "Minutes"


@pytest.mark.asyncio
async def test_noisy_buffer_follow_up_refreshes_recent_trusted_balance(monkeypatch) -> None:
    store = InMemorySessionStore()
    calls = []

    async def balance(target_date):
        calls.append(target_date)
        return {"hasPolicy": True, "limitType": 2, "remaining": 120}

    async def forbidden_write(*args, **kwargs):
        raise AssertionError("buffer reads must never execute a ResourcePlus write")

    def no_model(*args, **kwargs):
        raise AssertionError("recovered buffer transcripts must not construct a model client")

    monkeypatch.setattr(fast_reads, "resourceplus_today", lambda: date(2026, 10, 1))
    monkeypatch.setattr(fast_reads, "get_exceptional_entry_balance", balance)
    monkeypatch.setattr(chat_service, "execute_pending_action", forbidden_write)
    monkeypatch.setattr(__import__("app.ai.agent", fromlist=["AsyncOpenAI"]), "AsyncOpenAI", no_model)

    first = await chat_service.process_chat(
        ChatRequest(message="How much buffer time do I have?"),
        store=store,
    )
    refreshed = await chat_service.process_chat(
        ChatRequest(message="Buffet time. Time.", session_id=first.session_id),
        store=store,
    )

    assert calls == ["2026-10-01", "2026-10-01"]
    assert refreshed.tools_used == ["get_exceptional_entry_balance"]
    assert refreshed.message == "You have 120 minutes remaining."
    assert refreshed.speech_message == refreshed.message


@pytest.mark.asyncio
async def test_noisy_buffer_without_trusted_context_asks_clarification(monkeypatch) -> None:
    store = InMemorySessionStore()

    async def forbidden_balance(*args, **kwargs):
        raise AssertionError("ambiguous text without context must not fabricate a balance")

    async def forbidden_agent(*args, **kwargs):
        raise AssertionError("ambiguous degraded buffer text must not reach the model")

    monkeypatch.setattr(fast_reads, "get_exceptional_entry_balance", forbidden_balance)
    monkeypatch.setattr(chat_service, "run_agent", forbidden_agent)

    response = await chat_service.process_chat(
        ChatRequest(message="Buffet time. Time."),
        store=store,
    )

    assert response.message == "Do you mean your remaining buffer time?"
    assert response.tools_used == []


@pytest.mark.asyncio
async def test_generic_model_cannot_retract_recent_trusted_balance(monkeypatch) -> None:
    store = InMemorySessionStore()
    reads = []

    async def balance(target_date):
        reads.append(target_date)
        return {"hasPolicy": True, "limitType": 2, "remaining": 120}

    monkeypatch.setattr(fast_reads, "resourceplus_today", lambda: date(2026, 10, 1))
    monkeypatch.setattr(fast_reads, "get_exceptional_entry_balance", balance)
    first = await chat_service.process_chat(
        ChatRequest(message="How much buffer time do I have?"),
        store=store,
    )

    async def contradict(*args, **kwargs):
        return AgentResult(
            "The earlier 120-minute figure wasn't supported.",
            [],
            speech_message="The earlier 120-minute figure wasn't supported.",
        )

    monkeypatch.setattr(chat_service, "run_agent", contradict)
    guarded = await chat_service.process_chat(
        ChatRequest(message="Can you explain the earlier answer?", session_id=first.session_id),
        store=store,
    )

    assert reads == ["2026-10-01"]
    assert "wasn't supported" not in guarded.message
    assert guarded.message == (
        "Your previous balance came from a completed HR balance check. "
        "I can refresh it if you'd like."
    )


@pytest.mark.asyncio
async def test_previous_month_reuses_trusted_attendance_topic_without_model(monkeypatch) -> None:
    store = InMemorySessionStore()
    calls = []

    async def attendance(start, end, **kwargs):
        calls.append((start, end))
        return {"Days": []}

    def no_model(*args, **kwargs):
        raise AssertionError("attendance month follow-ups must remain deterministic")

    monkeypatch.setattr(fast_reads, "resourceplus_today", lambda: date(2026, 10, 1))
    monkeypatch.setattr(fast_reads, "get_attendance_summary", attendance)
    monkeypatch.setattr(__import__("app.ai.agent", fromlist=["AsyncOpenAI"]), "AsyncOpenAI", no_model)

    first = await chat_service.process_chat(
        ChatRequest(message="Show me my attendance this month."),
        store=store,
    )
    previous = await chat_service.process_chat(
        ChatRequest(message="Previous month.", session_id=first.session_id),
        store=store,
    )

    assert calls == [("2026-10-01", "2026-10-01"), ("2026-09-01", "2026-09-30")]
    assert previous.tools_used == ["get_attendance_summary"]
    assert previous.message == "You have no attendance records for this period."


@pytest.mark.asyncio
async def test_balance_fast_path_handles_no_policy_and_clear_arabic(monkeypatch) -> None:
    calls = []

    async def balance(target_date):
        calls.append(target_date)
        return {"hasPolicy": False}

    monkeypatch.setattr(fast_reads, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(fast_reads, "get_exceptional_entry_balance", balance)
    result = await __import__("app.ai.agent", fromlist=["run_agent"]).run_agent(
        "كم عندي دقائق استئذان؟",
        lang=1,
        session_id="balance-fast-ar",
        history=[],
        response_language="ar",
    )

    assert calls == ["2026-09-30"]
    assert result.tools_used == ["get_exceptional_entry_balance"]
    assert result.blocks and result.blocks[0].type == "notice"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Show my attendance this month",
        "Could you list my attendance for this month?",
        "Uh, show me my attendance over this month please",
    ],
)
async def test_attendance_fast_path_builds_grounded_table(monkeypatch, message) -> None:
    async def attendance(start, end, **kwargs):
        return {"Days": [{
            "AttDate": "30/09/2026",
            "DayType": "Regular",
            "CheckIN": "09:00",
            "CheckOut": "18:00",
            "NetHrs": "09:00",
            "LessHrs": "00:00",
        }]}

    monkeypatch.setattr(fast_reads, "get_attendance_summary", attendance)
    result = await __import__("app.ai.agent", fromlist=["run_agent"]).run_agent(
        message, lang=1, session_id="attendance-fast", history=[]
    )
    assert result.message.startswith("You have 1 attendance record for this period.")
    assert result.speech_message == "You have 1 attendance record for this period."
    table = next(block for block in result.blocks or [] if block.type == "table")
    assert table.rows == [{
        "date": "30/09/2026", "status": "Regular", "in": "09:00",
        "out": "18:00", "worked": "09:00", "shortfall": "00:00",
    }]


@pytest.mark.asyncio
async def test_independent_fast_reads_run_concurrently(monkeypatch) -> None:
    active = 0
    peak = 0

    async def tracked(value):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        active -= 1
        return value

    monkeypatch.setattr(fast_reads, "get_attendance_summary", lambda *a, **k: tracked({"Days": []}))
    monkeypatch.setattr(fast_reads, "get_missing_punch_suggestions", lambda *a, **k: tracked([]))
    result = await fast_reads.try_fast_read(
        "Show my attendance and missing punches", lang=1, response_language="en"
    )
    assert result is not None
    assert peak == 2


@pytest.mark.asyncio
async def test_reference_cache_ttl_and_copy_isolation() -> None:
    cache = ReferenceDataCache(ttl_seconds=60, max_entries=2)
    loads = 0

    async def loader():
        nonlocal loads
        loads += 1
        return [{"dayType": "Annual Leave"}]

    first = await cache.get_or_load(("day_types", "universal", 1), loader)
    first[0]["dayType"] = "Changed"
    second = await cache.get_or_load(("day_types", "universal", 1), loader)
    assert loads == 1
    assert second[0]["dayType"] == "Annual Leave"


def test_date_grounding_for_relative_explicit_and_ordinal_dates() -> None:
    today = date(2026, 9, 30)
    assert extract_date("today", today=today) == today
    assert extract_date("yesterday", today=today) == date(2026, 9, 29)
    assert extract_date("IN on 6 September", today=today) == date(2026, 9, 6)
    assert extract_date("the 29th", today=today) == date(2026, 9, 29)


def test_chat_stream_emits_backward_compatible_complete_response(monkeypatch) -> None:
    async def process(request):
        return chat_service.ChatResponse(
            success=True,
            message="Hello",
            language="en",
            session_id="stream-session",
        ).set_speech_message("Hello")

    from app.api import chat as chat_api
    monkeypatch.setattr(chat_api, "process_chat", process)
    response = client.post("/api/chat/stream", json={"message": "Hello"})
    assert response.status_code == 200
    assert "event: accepted" in response.text
    assert "event: text_delta" in response.text
    assert "event: complete" in response.text
    assert '"response_schema_version": 2' in response.text


@pytest.mark.asyncio
async def test_chat_stream_early_close_resets_audit_before_processing(
    monkeypatch,
) -> None:
    from app.api import chat as chat_api
    from app.audit import current_interaction_audit

    process_calls = []

    async def process(request):
        process_calls.append(request)
        raise AssertionError("an early-closed stream must not start processing")

    monkeypatch.setattr(chat_api, "process_chat", process)
    response = await chat_api.chat_stream(ChatRequest(message="Hello"))
    iterator = response.body_iterator

    assert current_interaction_audit() is None
    assert (await anext(iterator)).startswith("event: accepted")
    assert current_interaction_audit() is not None
    await iterator.aclose()

    assert current_interaction_audit() is None
    assert process_calls == []


def test_progressive_voice_sends_text_before_final_audio(monkeypatch) -> None:
    recognizer = FakeStreamingRecognizer(
        SpeechTranscript("Show my profile", "en-US", "en")
    )

    async def process(request, *, detected_language):
        return ChatResponse(
            success=True,
            message="Your profile is ready.",
            language="en",
            session_id="progressive-session",
        ).set_speech_message("Your profile is ready.")

    async def synthesize(*args, **kwargs):
        return SpeechAudio(b"audio")

    async def persist(*args, **kwargs):
        return None

    monkeypatch.setattr(voice_module, "StreamingSpeechRecognizer", lambda: recognizer)
    monkeypatch.setattr(voice_module, "process_chat", process)
    monkeypatch.setattr(voice_module, "synthesize_speech", synthesize)
    monkeypatch.setattr(voice_module, "persist_audit", persist)

    with client.websocket_connect("/api/voice/stream") as websocket:
        websocket.send_json({
            "type": "start", "sample_rate": 16_000, "progressive_events": True,
        })
        assert websocket.receive_json()["type"] == "ready"
        assert websocket.receive_json()["type"] == "listening"
        websocket.send_bytes(b"\x01\x00" * 320)
        websocket.send_json({"type": "end"})
        event_types = [websocket.receive_json()["type"] for _ in range(4)]

    assert event_types == ["transcript_final", "processing", "assistant_text", "final"]


def test_progressive_voice_tts_failure_keeps_text_and_final(monkeypatch) -> None:
    recognizer = FakeStreamingRecognizer(
        SpeechTranscript("Show my profile", "en-US", "en")
    )

    async def process(request, *, detected_language):
        return ChatResponse(
            success=True,
            message="Your profile is ready.",
            language="en",
            session_id="progressive-tts-failure",
        ).set_speech_message("Your profile is ready.")

    async def synthesize(*args, **kwargs):
        raise SpeechSynthesisError("temporary", safe_category="service_unavailable")

    async def persist(*args, **kwargs):
        return None

    monkeypatch.setattr(voice_module, "StreamingSpeechRecognizer", lambda: recognizer)
    monkeypatch.setattr(voice_module, "process_chat", process)
    monkeypatch.setattr(voice_module, "synthesize_speech", synthesize)
    monkeypatch.setattr(voice_module, "persist_audit", persist)

    with client.websocket_connect("/api/voice/stream") as websocket:
        websocket.send_json({
            "type": "start", "sample_rate": 16_000, "progressive_events": True,
        })
        websocket.receive_json()
        websocket.receive_json()
        websocket.send_bytes(b"\x01\x00" * 320)
        websocket.send_json({"type": "end"})
        transcript = websocket.receive_json()
        processing = websocket.receive_json()
        text_event = websocket.receive_json()
        tts_error = websocket.receive_json()
        final = websocket.receive_json()

    assert transcript["type"] == "transcript_final"
    assert processing["type"] == "processing"
    assert text_event["type"] == "assistant_text"
    assert text_event["message"] == "Your profile is ready."
    assert tts_error["type"] == "error" and tts_error["scope"] == "tts"
    assert final["type"] == "final"
    assert final["message"] == "Your profile is ready."
    assert final["audio_base64"] == ""


@pytest.mark.asyncio
async def test_exceptional_read_interrupts_less_hours_reason_draft(monkeypatch) -> None:
    store = InMemorySessionStore()

    async def attendance(*args, **kwargs):
        return {"Days": [{
            "AttDate": "2026-09-29",
            "DayType": "Regular",
            "CheckIN": "09:20",
            "CheckOut": "17:00",
            "NetHrs": "07:40",
            "LessHrs": "00:20",
        }]}

    async def reasons(*args, **kwargs):
        return [{"reasonID": "traffic", "reasonName": "Traffic"}]

    async def balance(*args, **kwargs):
        return {"hasPolicy": True, "remaining": 100, "limitType": 2}

    calls = []

    async def exceptional(start, end, **kwargs):
        calls.append((start, end))
        return [{
            "exceptionalID": "secret-read-id",
            "entryTime": "2026-09-22T08:45:00",
            "entryType": 1,
            "reason": "Traffic",
            "status": "Pending",
        }]

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(fast_reads, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(actions, "get_attendance_summary", attendance)
    monkeypatch.setattr(conversation, "cached_exception_reasons", reasons)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)
    monkeypatch.setattr(fast_reads, "get_exceptional_entry_requests", exceptional)

    first = await chat_service.process_chat(
        ChatRequest(message="Correct my late arrival yesterday"), store=store
    )
    assert first.needs_reason is True
    assert store.get_conversation_draft(first.session_id) is not None

    response = await chat_service.process_chat(
        ChatRequest(
            message="Can you show my exceptional entries this month?",
            session_id=first.session_id,
        ),
        store=store,
    )

    assert response.tools_used == ["get_exceptional_entries"]
    assert response.blocks[0].type == "table"
    assert response.blocks[0].rows[0]["date"] == "2026-09-22"
    assert "couldn't match that reason" not in response.message.casefold()
    assert response.needs_reason is False
    assert "secret-read-id" not in response.model_dump_json()
    assert calls == [("2026-09-01", "2026-09-30")]
    assert store.get_conversation_draft(first.session_id) is None
    assert store.get_pending_action(first.session_id)[0] is None


@pytest.mark.asyncio
async def test_new_read_does_not_clear_immutable_pending_action(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    pending = store.create_pending_action(
        session_id,
        action_type="cancel_exceptional_entry",
        validated_arguments={"exceptional_id": "backend-only", "display": "22 September"},
        summary="Confirm cancellation",
        language="en",
    )

    async def other(*args, **kwargs):
        return "OTHER"

    monkeypatch.setattr(chat_service, "_confirmation_decision", other)
    response = await chat_service.process_chat(
        ChatRequest(message="Show my exceptional entries this month", session_id=session_id),
        store=store,
    )
    preserved, _ = store.get_pending_action(session_id)
    assert response.requires_confirmation is True
    assert response.confirmation_id == pending.confirmation_id
    assert preserved is not None and preserved.confirmation_id == pending.confirmation_id


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Show my exceptional entries this month",
        "Show my exceptional entry requests",
        "Show my exception requests this week",
        "What exceptional entries do I have?",
        "Show my pending exceptional entries",
        "Show my exceptional energies this month",
    ],
)
async def test_exceptional_entry_read_fast_path_is_specific_and_structured(
    monkeypatch,
    message,
) -> None:
    calls = []

    async def exceptional(start, end, **kwargs):
        calls.append((start, end))
        return [
            {
                "exceptionalID": "never-visible-one",
                "entryTime": "2026-09-10T09:15:00",
                "entryType": 1,
                "reason": "Traffic",
                "status": "Pending",
            },
            {
                "exceptionalID": "never-visible-two",
                "entryTime": "22/09/2026 16:30",
                "entryType": 2,
                "reason": "Family Circumstances",
                "status": "Approved",
            },
        ]

    async def forbidden_broad(*args, **kwargs):
        raise AssertionError("specific exceptional reads must not aggregate MyRequests")

    def forbidden_model(*args, **kwargs):
        raise AssertionError("specific exceptional reads must not call OpenAI")

    monkeypatch.setattr(fast_reads, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(fast_reads, "get_exceptional_entry_requests", exceptional)
    monkeypatch.setattr(fast_reads, "get_my_request_status", forbidden_broad)
    monkeypatch.setattr(__import__("app.ai.agent", fromlist=["AsyncOpenAI"]), "AsyncOpenAI", forbidden_model)

    result = await __import__("app.ai.agent", fromlist=["run_agent"]).run_agent(
        message, lang=1, session_id="exceptional-read", history=[]
    )

    assert result.tools_used == ["get_exceptional_entries"]
    assert calls
    if "this week" not in message.casefold():
        assert calls == [("2026-09-01", "2026-09-30")]
    table = result.blocks[0]
    assert [column.label for column in table.columns] == ["Date", "Type", "Reason", "Status"]
    assert table.rows[0] == {
        "date": "2026-09-10",
        "type": "Late Arrival",
        "reason": "Traffic",
        "status": "Pending",
    }
    assert table.rows[1]["type"] == "Early Departure"
    assert "exceptionalID" not in table.model_dump_json()
    assert "never-visible" not in result.message
    assert result.message.startswith(
        "You have 2 exceptional-entry requests for this period."
    )
    assert result.speech_message == (
        "You have 2 exceptional-entry requests for this period."
    )


def _pending_exception_rows() -> list[dict[str, object]]:
    return [
        {
            "exceptionalID": "secret-first",
            "entryTime": "2026-09-20T09:10:00",
            "entryType": 1,
            "reason": "Traffic",
            "status": "Pending",
        },
        {
            "exceptionalID": "secret-second",
            "entryTime": "2026-09-22T16:20:00",
            "entryType": 2,
            "reason": "Family Circumstances",
            "status": "Not Approved",
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selection", "expected_id"),
    [
        ("Family Circumstances", "secret-second"),
        ("Family circumstance", "secret-second"),
        ("22 September", "secret-second"),
        ("The early departure one", "secret-second"),
        ("Number 2", "secret-second"),
    ],
)
async def test_cancellation_candidate_continuation_resolves_without_write(
    monkeypatch,
    selection,
    expected_id,
) -> None:
    store = InMemorySessionStore()
    writes = []

    async def exceptional(*args, **kwargs):
        return _pending_exception_rows()

    async def forbidden_write(*args, **kwargs):
        writes.append((args, kwargs))

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(actions, "get_exceptional_entry_requests", exceptional)
    monkeypatch.setattr(actions, "cancel_exceptional_entry", forbidden_write)

    discovered = await chat_service.process_chat(
        ChatRequest(message="Cancel my pending exceptional entry"), store=store
    )
    table = next(block for block in discovered.blocks if block.type == "table")
    action_block = next(block for block in discovered.blocks if block.type == "actions")
    assert table.rows[0]["date"] == "2026-09-20"
    assert table.rows[0]["type"] == "Late Arrival"
    assert table.rows[1]["type"] == "Early Departure"
    assert "Minutes" not in [column.label for column in table.columns]
    assert all("secret-" not in action.label for action in action_block.actions)
    assert all(action.value.startswith("Select exceptional entry ce-") for action in action_block.actions)

    prepared = await chat_service.process_chat(
        ChatRequest(message=selection, session_id=discovered.session_id), store=store
    )
    pending, _ = store.get_pending_action(discovered.session_id)
    assert prepared.requires_confirmation is True
    assert pending is not None
    assert pending.validated_arguments["exceptional_id"] == expected_id
    assert "secret-" not in prepared.model_dump_json()
    assert writes == []


@pytest.mark.asyncio
async def test_ambiguous_cancellation_selection_keeps_non_executable_draft(monkeypatch) -> None:
    store = InMemorySessionStore()
    rows = _pending_exception_rows()
    rows[1]["entryType"] = 1

    async def exceptional(*args, **kwargs):
        return rows

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(actions, "get_exceptional_entry_requests", exceptional)
    discovered = await chat_service.process_chat(
        ChatRequest(message="Cancel my pending exceptional entry"), store=store
    )
    response = await chat_service.process_chat(
        ChatRequest(message="The late arrival one", session_id=discovered.session_id),
        store=store,
    )
    assert "more than one" in response.message
    assert response.requires_confirmation is False
    assert store.get_pending_action(discovered.session_id)[0] is None
    assert store.get_conversation_draft(discovered.session_id) is not None


@pytest.mark.asyncio
async def test_cancellation_no_and_replay_never_post(monkeypatch) -> None:
    store = InMemorySessionStore()
    writes = []

    async def exceptional(*args, **kwargs):
        return [_pending_exception_rows()[0]]

    async def forbidden_write(*args, **kwargs):
        writes.append((args, kwargs))

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(actions, "get_exceptional_entry_requests", exceptional)
    monkeypatch.setattr(actions, "cancel_exceptional_entry", forbidden_write)
    prepared = await chat_service.process_chat(
        ChatRequest(message="Cancel my pending exceptional entry"), store=store
    )
    rejected = await chat_service.process_chat(
        ChatRequest(
            message="No",
            session_id=prepared.session_id,
            confirmation_id=prepared.confirmation_id,
        ),
        store=store,
    )
    replay = await chat_service.process_chat(
        ChatRequest(
            message="No",
            session_id=prepared.session_id,
            confirmation_id=prepared.confirmation_id,
        ),
        store=store,
    )
    assert rejected.success is True
    assert replay.success is False
    assert writes == []


def test_scoped_voice_exceptional_normalization() -> None:
    assert conversation._is_cancel_exception_request(
        "Cancel my pending exceptional interest."
    )
    assert fast_reads.classify_fast_read(
        "Show my exceptional energies this month."
    ) == ["exceptional_entries"]
    assert fast_reads.classify_fast_read("I study renewable energies") == []
    assert not conversation._is_cancel_exception_request("I have an interest in HR")


@pytest.mark.asyncio
async def test_cancellation_selection_state_is_session_and_language_isolated(monkeypatch) -> None:
    store = InMemorySessionStore()

    async def exceptional(*args, **kwargs):
        return _pending_exception_rows()

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(actions, "get_exceptional_entry_requests", exceptional)
    english = await chat_service.process_chat(
        ChatRequest(message="Cancel my pending exceptional entry"), store=store
    )
    arabic_session = store.ensure_session()
    store.save_conversation_draft(
        arabic_session,
        intent="cancel_exceptional_entry",
        slots={
            "cancellation_candidates": json.dumps([
                {
                    "key": "ce-arabic-only",
                    "exceptional_id": "arabic-secret",
                    "date": "2026-09-25",
                    "type": "Late Arrival",
                    "reason": "سبب عربي",
                    "status": "Pending",
                }
            ], ensure_ascii=False),
        },
        language="ar",
    )
    prepared = await chat_service.process_chat(
        ChatRequest(message="Number 2", session_id=english.session_id), store=store
    )
    assert prepared.requires_confirmation is True
    assert store.get_conversation_draft(arabic_session) is not None
    assert store.get_pending_action(arabic_session)[0] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Cancel my pending exceptional entry",
        "Cancel my pending exceptional entries",
        "Cancel my pending exceptional interest",
    ],
)
async def test_cancellation_discovery_variants_are_deterministic(monkeypatch, message) -> None:
    store = InMemorySessionStore()
    reads = []
    writes = []

    async def exceptional(start, end, **kwargs):
        reads.append((start, end))
        return [_pending_exception_rows()[0]]

    async def forbidden_write(*args, **kwargs):
        writes.append((args, kwargs))

    def forbidden_model(*args, **kwargs):
        raise AssertionError("clear cancellation discovery must not call OpenAI")

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(actions, "get_exceptional_entry_requests", exceptional)
    monkeypatch.setattr(actions, "cancel_exceptional_entry", forbidden_write)
    monkeypatch.setattr(__import__("app.ai.agent", fromlist=["AsyncOpenAI"]), "AsyncOpenAI", forbidden_model)

    response = await chat_service.process_chat(ChatRequest(message=message), store=store)
    assert reads == [(date(2026, 9, 1), date(2026, 9, 30))]
    assert response.requires_confirmation is True
    assert response.tools_used == [
        "get_exceptional_entries",
        "prepare_cancel_exceptional_entry",
    ]
    assert writes == []


@pytest.mark.asyncio
async def test_opaque_candidate_selection_and_stale_replay_are_safe(monkeypatch) -> None:
    store = InMemorySessionStore()
    writes = []

    async def exceptional(*args, **kwargs):
        return _pending_exception_rows()

    async def forbidden_write(*args, **kwargs):
        writes.append((args, kwargs))

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(actions, "get_exceptional_entry_requests", exceptional)
    monkeypatch.setattr(actions, "cancel_exceptional_entry", forbidden_write)

    discovered = await chat_service.process_chat(
        ChatRequest(message="Cancel my pending exceptional entries"), store=store
    )
    action = next(block for block in discovered.blocks if block.type == "actions").actions[1]
    assert "ce-" not in action.label
    assert action.value.startswith("Select exceptional entry ce-")

    prepared = await chat_service.process_chat(
        ChatRequest(message=action.value, session_id=discovered.session_id), store=store
    )
    pending, _ = store.get_pending_action(discovered.session_id)
    assert prepared.requires_confirmation is True
    assert pending is not None
    assert pending.validated_arguments["exceptional_id"] == "secret-second"
    assert writes == []

    rejected = await chat_service.process_chat(
        ChatRequest(
            message="No",
            session_id=discovered.session_id,
            confirmation_id=prepared.confirmation_id,
        ),
        store=store,
    )
    stale = await chat_service.process_chat(
        ChatRequest(message=action.value, session_id=discovered.session_id), store=store
    )
    assert rejected.success is True
    assert stale.message == (
        "That selection is no longer active. Please open your pending exceptional entries again."
    )
    assert stale.requires_confirmation is False
    assert store.get_pending_action(discovered.session_id)[0] is None
    assert store.get_conversation_draft(discovered.session_id) is None
    assert writes == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("new_request", "expected_minutes", "expected_entry_type"),
    [
        ("Only correct 10 minutes of my late arrival on 10 September", 10, 1),
        ("Correct only 10 minutes on 10 September", 10, None),
        ("Only correct my late arrival on 10 September", None, 1),
        ("Correct my early departure on 10 September", None, 2),
    ],
)
async def test_new_correction_replaces_reason_draft_and_preserves_scope(
    monkeypatch,
    new_request,
    expected_minutes,
    expected_entry_type,
) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    store.save_conversation_draft(
        session_id,
        intent="less_hours_correction",
        slots={
            "date": "2026-09-29",
            "less_hours": "00:30",
            "reason_options": "Traffic\x1fOutside Work",
        },
        validated_slots=("date",),
        language="en",
    )

    async def attendance(*args, **kwargs):
        return {"Days": [{
            "AttDate": "2026-09-10",
            "DayType": "Regular",
            "CheckIN": "09:30",
            "CheckOut": "17:00",
            "NetHrs": "07:30",
            "LessHrs": "00:30",
        }]}

    async def reasons(*args, **kwargs):
        return [
            {"reasonID": "traffic-id", "reasonName": "Traffic"},
            {"reasonID": "outside-id", "reasonName": "Outside Work"},
        ]

    async def balance(*args, **kwargs):
        return {"hasPolicy": True, "remaining": 100, "limitType": 2}

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(actions, "get_attendance_summary", attendance)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    monkeypatch.setattr(conversation, "cached_exception_reasons", reasons)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)

    reason_prompt = await chat_service.process_chat(
        ChatRequest(message=new_request, session_id=session_id), store=store
    )
    replacement = store.get_conversation_draft(session_id)
    assert reason_prompt.needs_reason is True
    assert "couldn't match that reason" not in reason_prompt.message.casefold()
    assert replacement is not None
    assert replacement.slots["date"] == "2026-09-10"
    assert replacement.slots.get("minutes") == (
        str(expected_minutes) if expected_minutes is not None else None
    )
    assert replacement.slots.get("entry_type") == (
        str(expected_entry_type) if expected_entry_type is not None else None
    )

    prepared = await chat_service.process_chat(
        ChatRequest(message="Traffic", session_id=session_id), store=store
    )
    pending, _ = store.get_pending_action(session_id)
    assert prepared.requires_confirmation is True
    assert pending is not None
    assert pending.validated_arguments.get("minutes") == expected_minutes
    assert pending.validated_arguments.get("entry_type") == expected_entry_type
    assert "Date: 2026-09-10" in prepared.message
    if expected_minutes is None:
        assert "Requested minutes:" not in prepared.message
    else:
        assert f"Requested minutes: {expected_minutes}" in prepared.message
        assert "Requested minutes: 30" not in prepared.message
    if expected_entry_type == 1:
        assert "Scope: late IN only" in prepared.message
    elif expected_entry_type == 2:
        assert "Scope: early OUT only" in prepared.message
    else:
        assert "Scope:" not in prepared.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "read_request",
    [
        "Show my exceptional entries this month",
        "Show my exceptional energies this month",
    ],
)
async def test_exceptional_read_interrupts_cancellation_selection_draft(
    monkeypatch,
    read_request,
) -> None:
    store = InMemorySessionStore()
    discovery_reads = []
    read_calls = []

    async def discovery(*args, **kwargs):
        discovery_reads.append(1)
        return _pending_exception_rows()

    async def exceptional_read(start, end, **kwargs):
        read_calls.append((start, end))
        return _pending_exception_rows()

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(fast_reads, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(actions, "get_exceptional_entry_requests", discovery)
    monkeypatch.setattr(fast_reads, "get_exceptional_entry_requests", exceptional_read)

    discovered = await chat_service.process_chat(
        ChatRequest(message="Cancel my pending exceptional entries"), store=store
    )
    assert store.get_conversation_draft(discovered.session_id) is not None
    response = await chat_service.process_chat(
        ChatRequest(message=read_request, session_id=discovered.session_id), store=store
    )
    assert discovery_reads == [1]
    assert read_calls == [("2026-09-01", "2026-09-30")]
    assert response.tools_used == ["get_exceptional_entries"]
    assert response.blocks[0].title == "Exceptional entries"
    assert store.get_conversation_draft(discovered.session_id) is None
    assert store.get_pending_action(discovered.session_id)[0] is None


@pytest.mark.asyncio
async def test_english_leave_read_interrupts_arabic_cancellation_draft(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()
    writes = []

    async def exceptional(*args, **kwargs):
        return _pending_exception_rows()

    async def requests(start, end, **kwargs):
        assert (start, end) == ("2026-09-01", "2026-09-30")
        return {"merged_requests": []}

    async def forbidden_write(*args, **kwargs):
        writes.append((args, kwargs))

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(fast_reads, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(actions, "get_exceptional_entry_requests", exceptional)
    monkeypatch.setattr(fast_reads, "get_my_request_status", requests)
    monkeypatch.setattr(actions, "cancel_exceptional_entry", forbidden_write)

    discovered = await chat_service.process_chat(
        ChatRequest(message="ألغِ طلبات الاستثناء المعلقة"), store=store
    )
    response = await chat_service.process_chat(
        ChatRequest(message="Show my leave request", session_id=discovered.session_id),
        store=store,
    )

    assert response.tools_used == ["get_my_request_status"]
    assert response.blocks[0].title == "My requests"
    assert store.get_conversation_draft(discovered.session_id) is None
    assert store.get_pending_action(discovered.session_id)[0] is None
    assert writes == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("intent", "slots"),
    [
        ("correct_missing_punch", {"direction": "IN"}),
        (
            "book_day_type",
            {"date_from": "2026-10-01", "date_to": "2026-10-01"},
        ),
    ],
)
async def test_profile_read_interrupts_date_or_day_type_draft(
    monkeypatch,
    intent,
    slots,
) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    store.save_conversation_draft(
        session_id,
        intent=intent,
        slots=slots,
        language="en",
    )

    async def profile(*args, **kwargs):
        return {"EmployeeName": "Test Employee"}

    monkeypatch.setattr(fast_reads, "get_profile_data", profile)
    response = await chat_service.process_chat(
        ChatRequest(message="Show my profile", session_id=session_id),
        store=store,
    )

    assert response.tools_used == ["get_profile_data"]
    assert response.blocks[0].title == "Profile"
    assert store.get_conversation_draft(session_id) is None
    assert store.get_pending_action(session_id)[0] is None
