from __future__ import annotations

import json

import pytest

from app.ai import actions
from app.ai.attendance_intent import (
    explicit_missing_punch_direction,
    is_explicit_missing_punch_correction,
)
from app.ai.sessions import InMemorySessionStore
from app.ai.tools import execute_tool
from app.models.schemas import ChatRequest
from app.services import chat as chat_service


def suggestion(direction: str) -> dict[str, object]:
    return {
        "attDate": "29/09/2026",
        "suggestedEntryTime": (
            "29/09/2026 08:00" if direction == "IN" else "29/09/2026 17:00"
        ),
        "entryType": direction,
        "shift": "General Shift",
        "isNightShift": 0,
    }


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("I forgot to punch in today.", "IN"),
        ("I forgot to punch out yesterday.", "OUT"),
        ("My IN punch is missing.", "IN"),
        ("My OUT punch is missing.", "OUT"),
        ("نسيت أبصم دخول اليوم.", "IN"),
        ("نسيت أبصم خروج اليوم.", "OUT"),
    ],
)
def test_explicit_direction_is_grounded_from_user_message(
    message: str,
    expected: str,
) -> None:
    assert is_explicit_missing_punch_correction(message) is True
    assert explicit_missing_punch_direction(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "My punch is missing today.",
        "I missed my morning punch.",
        "I missed my evening punch.",
    ],
)
def test_ambiguous_or_time_of_day_wording_does_not_guess_direction(message) -> None:
    assert is_explicit_missing_punch_correction(message) is True
    assert explicit_missing_punch_direction(message) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("I forgot to punch in today.", "IN"),
        ("I forgot to punch out today.", "OUT"),
        ("My IN punch is missing.", "IN"),
        ("My OUT punch is missing.", "OUT"),
        ("نسيت أبصم دخول اليوم.", "IN"),
        ("نسيت أبصم خروج اليوم.", "OUT"),
    ],
)
async def test_both_live_suggestions_are_filtered_by_grounded_direction(
    monkeypatch,
    message: str,
    expected: str,
) -> None:
    async def suggestions(*args, **kwargs):
        return [suggestion("IN"), suggestion("OUT")]

    async def reasons(*args, **kwargs):
        return [{"reasonID": "reason-live", "reasonName": "Family Circumstances"}]

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    store = InMemorySessionStore()
    session_id = store.ensure_session()

    result = await execute_tool(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-29",
            # Deliberately conflict with the user's wording. Backend grounding wins.
            "punch_direction": "OUT" if expected == "IN" else "IN",
            "reason_name": None,
            "remarks": None,
        },
        lang=1,
        session_id=session_id,
        response_language="ar" if "بصم" in message else "en",
        source_user_message=message,
        store=store,
    )

    assert json.loads(result.output)["needs_reason"] is True
    draft = store.get_exceptional_entry_draft(session_id)
    assert draft is not None
    assert draft.entry_type == expected
    assert result.pending_action is None
    assert store.get_pending_action(session_id) == (None, False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "requested", "available"),
    [
        ("I forgot to punch in today.", "IN", "OUT"),
        ("I forgot to punch out today.", "OUT", "IN"),
    ],
)
async def test_opposite_only_suggestion_stops_without_reasons_or_state(
    monkeypatch,
    message: str,
    requested: str,
    available: str,
) -> None:
    calls: list[str] = []

    async def suggestions(*args, **kwargs):
        calls.append("suggestions")
        return [suggestion(available)]

    async def reasons(*args, **kwargs):
        calls.append("reasons")
        raise AssertionError("Opposite direction must stop before Reasons GET")

    async def write(*args, **kwargs):
        calls.append("write")
        raise AssertionError("Opposite direction must never write")

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    monkeypatch.setattr(actions, "create_exceptional_entry", write)
    store = InMemorySessionStore()
    session_id = store.ensure_session()

    result = await execute_tool(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-29",
            "punch_direction": available,
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
    assert f"missing {requested} punch" in payload["message"]
    assert calls == ["suggestions"]
    assert result.pending_action is None
    assert store.get_exceptional_entry_draft(session_id) is None
    assert store.get_pending_action(session_id) == (None, False)


@pytest.mark.asyncio
async def test_ambiguous_missing_punch_overrides_model_guess_and_asks_direction(
    monkeypatch,
) -> None:
    async def suggestions(*args, **kwargs):
        return [suggestion("IN"), suggestion("OUT")]

    async def reasons(*args, **kwargs):
        raise AssertionError("Ambiguous direction must stop before Reasons GET")

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    store = InMemorySessionStore()
    session_id = store.ensure_session()

    result = await execute_tool(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-29",
            # The model guessed IN, but the user did not select a direction.
            "punch_direction": "IN",
            "reason_name": None,
            "remarks": None,
        },
        lang=1,
        session_id=session_id,
        response_language="en",
        source_user_message="My punch is missing today.",
        store=store,
    )

    payload = json.loads(result.output)
    assert "both a missing IN and a missing OUT punch" in payload["message"]
    assert "Which one do you want to correct?" in payload["message"]
    assert store.get_exceptional_entry_draft(session_id) is None
    assert store.get_pending_action(session_id) == (None, False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("direction", "source_message", "entry_type"),
    [
        ("IN", "I forgot to punch in today.", 1),
        ("OUT", "I forgot to punch out today.", 2),
    ],
)
async def test_direction_survives_draft_reason_pending_confirmation_and_post(
    monkeypatch,
    direction: str,
    source_message: str,
    entry_type: int,
) -> None:
    async def suggestions(*args, **kwargs):
        return [suggestion("IN"), suggestion("OUT")]

    async def reasons(*args, **kwargs):
        return [{"reasonID": "reason-family-live", "reasonName": "Family Circumstances"}]

    posts: list[dict[str, object]] = []

    async def post(**kwargs):
        posts.append(kwargs)
        return {"success": True, "message": "Submitted for approval"}

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    monkeypatch.setattr(actions, "create_exceptional_entry", post)
    store = InMemorySessionStore()
    session_id = store.ensure_session()

    prepared = await execute_tool(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-29",
            # Deliberately wrong to prove the source message remains authoritative.
            "punch_direction": "OUT" if direction == "IN" else "IN",
            "reason_name": None,
            "remarks": None,
        },
        lang=1,
        session_id=session_id,
        response_language="en",
        source_user_message=source_message,
        store=store,
    )
    assert prepared.pending_action is None
    draft = store.get_exceptional_entry_draft(session_id)
    assert draft is not None and draft.entry_type == direction

    reason_response = await chat_service.process_chat(
        ChatRequest(
            message="Family Circumstances",
            session_id=session_id,
        ),
        store=store,
    )
    assert reason_response.requires_confirmation is True
    pending, expired = store.get_pending_action(session_id)
    assert expired is False
    assert pending is not None
    assert pending.validated_arguments["entry_type"] == entry_type
    assert pending.validated_arguments["reason_id"] == "reason-family-live"

    confirmed = await chat_service.process_chat(
        ChatRequest(
            message="Yes",
            session_id=session_id,
            confirmation_id=pending.confirmation_id,
        ),
        store=store,
    )

    assert confirmed.success is True
    assert len(posts) == 1
    assert posts[0]["entry_type"] == entry_type
    assert posts[0]["reason_id"] == "reason-family-live"
