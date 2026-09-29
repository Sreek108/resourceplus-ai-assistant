from __future__ import annotations

import json

import pytest

from app.ai import actions, tools
from app.ai.attendance_intent import (
    is_existing_late_punch,
    is_prospective_late_arrival,
)
from app.ai.sessions import InMemorySessionStore
from app.models.schemas import ChatRequest
from app.services import chat as chat_service


@pytest.mark.parametrize(
    "message",
    [
        "I'm stuck in traffic and I'll be 30 minutes late today.",
        "I'll reach around 9:30 today.",
        "I'll punch in 20 minutes late today.",
        "I may arrive half an hour late.",
        "I will be late today because of traffic.",
    ],
)
def test_prospective_late_arrival_is_distinct_from_punch_correction(message) -> None:
    assert is_prospective_late_arrival(message) is True


@pytest.mark.parametrize(
    "message",
    [
        "I forgot to punch in today.",
        "My OUT punch is missing.",
        "Correct my missing punch for today.",
        "I forgot to punch out yesterday.",
        "Fix my attendance punch for September 22.",
        "I punched in 20 minutes late today.",
    ],
)
def test_existing_or_explicit_correction_is_not_classified_as_future_lateness(
    message,
) -> None:
    assert is_prospective_late_arrival(message) is False


@pytest.mark.parametrize(
    "message",
    [
        "I punched in late.",
        "I punched in 20 minutes late today.",
        "I reached late and already punched in.",
    ],
)
def test_completed_late_punch_is_recognized_without_becoming_missing(message) -> None:
    assert is_existing_late_punch(message) is True
    assert is_prospective_late_arrival(message) is False


@pytest.mark.parametrize(
    "message",
    [
        "I forgot to punch in today.",
        "My OUT punch is missing.",
        "Correct my missing punch for today.",
    ],
)
def test_explicit_missing_punch_overrides_completed_punch_routing(message) -> None:
    assert is_existing_late_punch(message) is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "I'm stuck in traffic and I'll be 30 minutes late today.",
        "I'll reach around 9:30 today.",
        "I'll punch in 20 minutes late today.",
    ],
)
async def test_late_arrival_returns_unsupported_without_attendance_workflow(
    monkeypatch,
    message: str,
) -> None:
    store = InMemorySessionStore()
    calls: list[str] = []

    async def should_not_run_agent(*args, **kwargs):
        calls.append("agent")
        raise AssertionError("Prospective lateness must be routed before the model")

    async def should_not_read(*args, **kwargs):
        calls.append("missing-punch-read")
        raise AssertionError("Prospective lateness must not query missing punches")

    async def should_not_write(*args, **kwargs):
        calls.append("resourceplus-write")
        raise AssertionError("Prospective lateness must not write to ResourcePlus")

    monkeypatch.setattr(chat_service, "run_agent", should_not_run_agent)
    monkeypatch.setattr(actions, "get_missing_punch_suggestions", should_not_read)
    monkeypatch.setattr(actions, "create_exceptional_entry", should_not_write)

    response = await chat_service.process_chat(
        ChatRequest(message=message),
        store=store,
    )

    assert response.success is True
    assert "late-arrival or buffer request service isn't connected" in response.message
    assert response.tools_used == []
    assert calls == []
    assert store.get_exceptional_entry_draft(response.session_id) is None
    assert store.get_pending_action(response.session_id) == (None, False)


@pytest.mark.asyncio
async def test_arabic_late_arrival_does_not_enter_missing_punch_flow(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()

    async def should_not_run(*args, **kwargs):
        raise AssertionError("Arabic prospective lateness must not enter the agent")

    monkeypatch.setattr(chat_service, "run_agent", should_not_run)
    response = await chat_service.process_chat(
        ChatRequest(message="راح أتأخر اليوم بسبب الزحمة ويمكن أوصل بعد نص ساعة"),
        detected_language="ar",
        store=store,
    )

    assert response.success is True
    assert response.language == "ar"
    assert "غير مرتبطة حاليًا" in response.message
    assert response.tools_used == []
    assert store.get_exceptional_entry_draft(response.session_id) is None
    assert store.get_pending_action(response.session_id) == (None, False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "I punched in late.",
        "I punched in 20 minutes late today.",
        "I reached late and already punched in.",
    ],
)
async def test_completed_late_punch_has_safe_unknown_policy_response(
    monkeypatch,
    message: str,
) -> None:
    store = InMemorySessionStore()
    calls: list[str] = []

    async def should_not_run(*args, **kwargs):
        calls.append("unexpected-call")
        raise AssertionError("A completed late punch must not enter missing-punch flow")

    monkeypatch.setattr(chat_service, "run_agent", should_not_run)
    monkeypatch.setattr(actions, "get_missing_punch_suggestions", should_not_run)
    monkeypatch.setattr(actions, "create_exceptional_entry", should_not_run)

    response = await chat_service.process_chat(
        ChatRequest(message=message),
        store=store,
    )

    assert response.success is True
    assert "already punched in late" in response.message
    assert "isn't a missing-punch correction" in response.message
    assert "can't check whether any adjustment or approval is required" in response.message
    assert "no correction needed" not in response.message.casefold()
    assert response.tools_used == []
    assert calls == []
    assert store.get_exceptional_entry_draft(response.session_id) is None
    assert store.get_pending_action(response.session_id) == (None, False)


@pytest.mark.asyncio
async def test_arabic_completed_late_punch_has_safe_unknown_policy_response(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()

    async def should_not_run(*args, **kwargs):
        raise AssertionError("Arabic completed late punch must not enter correction flow")

    monkeypatch.setattr(chat_service, "run_agent", should_not_run)
    response = await chat_service.process_chat(
        ChatRequest(message="وصلت متأخر وبصمت دخول"),
        detected_language="ar",
        store=store,
    )

    assert response.success is True
    assert "بصمت دخول متأخر" in response.message
    assert "ليست حالة بصمة مفقودة" in response.message
    assert "يلزم تعديل أو موافقة" in response.message
    assert response.tools_used == []
    assert store.get_exceptional_entry_draft(response.session_id) is None
    assert store.get_pending_action(response.session_id) == (None, False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "direction"),
    [
        ("I forgot to punch in today.", "IN"),
        ("My OUT punch is missing.", "OUT"),
        ("Correct my missing punch for today.", "IN"),
    ],
)
async def test_explicit_missing_punch_requests_still_reach_existing_flow(
    monkeypatch,
    message: str,
    direction: str,
) -> None:
    suggestion_calls: list[tuple[object, object]] = []

    async def suggestions(start, end, **kwargs):
        suggestion_calls.append((start, end))
        return [
            {
                "attDate": "29/09/2026",
                "suggestedEntryTime": "29/09/2026 08:00",
                "entryType": direction,
                "shift": "General Shift",
                "isNightShift": 0,
            }
        ]

    async def reasons(*args, **kwargs):
        return [{"reasonID": "reason-live", "reasonName": "Traffic Delay"}]

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    store = InMemorySessionStore()
    session_id = store.ensure_session()

    result = await tools.execute_tool(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-29",
            "punch_direction": direction,
            "reason_name": None,
            "remarks": None,
        },
        lang=1,
        session_id=session_id,
        response_language="en",
        source_user_message=message,
        store=store,
    )

    payload = json.loads(result.output)
    assert suggestion_calls == [("2026-09-29", "2026-09-29")]
    assert payload["needs_reason"] is True
    assert result.pending_action is None
    assert store.get_exceptional_entry_draft(session_id) is not None
    assert store.get_pending_action(session_id) == (None, False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool_name",
    ["get_missing_punch_suggestions", "prepare_exceptional_entry"],
)
@pytest.mark.parametrize(
    "source_message",
    [
        "Traffic is bad, so I'll probably punch in 20 minutes late.",
        "I punched in 20 minutes late today.",
    ],
)
async def test_tool_boundary_blocks_late_arrival_substitution(
    monkeypatch,
    tool_name: str,
    source_message: str,
) -> None:
    calls: list[str] = []

    async def should_not_read(*args, **kwargs):
        calls.append("read")
        raise AssertionError("Late arrival must not query MissingPunchSuggestions")

    async def should_not_prepare(*args, **kwargs):
        calls.append("prepare")
        raise AssertionError("Late arrival must not prepare an exceptional entry")

    monkeypatch.setattr(tools, "get_missing_punch_suggestions", should_not_read)
    monkeypatch.setattr(tools, "prepare_write_action", should_not_prepare)
    store = InMemorySessionStore()
    session_id = store.ensure_session()

    result = await tools.execute_tool(
        tool_name,
        (
            {"from_date": "2026-09-29", "to_date": "2026-09-29"}
            if tool_name == "get_missing_punch_suggestions"
            else {
                "target_date": "2026-09-29",
                "punch_direction": None,
                "reason_name": None,
                "remarks": None,
            }
        ),
        lang=1,
        session_id=session_id,
        response_language="en",
        source_user_message=source_message,
        store=store,
    )

    payload = json.loads(result.output)
    assert payload["prepared"] is False
    assert payload["requires_confirmation"] is False
    assert result.pending_action is None
    assert "isn't connected" in result.terminal_message
    if source_message.startswith("I punched"):
        assert "can't check whether any adjustment or approval is required" in (
            result.terminal_message
        )
    assert calls == []
    assert store.get_exceptional_entry_draft(session_id) is None
    assert store.get_pending_action(session_id) == (None, False)
