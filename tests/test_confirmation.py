from datetime import datetime, timedelta, timezone

import pytest
from fastapi.testclient import TestClient

from app.ai.sessions import (
    InMemorySessionStore,
    PendingActionExpired,
    session_store,
)
from app.ai.agent import AgentResult
from app.main import app
from app.models.schemas import ChatRequest
from app.observability import reset_voice_trace, start_voice_trace
from app.services import chat as chat_service


client = TestClient(app)


@pytest.fixture(autouse=True)
def clear_sessions(monkeypatch) -> None:
    session_store.clear()

    async def render(facts, **kwargs):
        return facts

    async def classify(*args, **kwargs):
        return "OTHER"

    monkeypatch.setattr(chat_service, "render_user_message", render)
    monkeypatch.setattr(chat_service, "classify_confirmation_intent", classify)


def seed_pending(arguments: dict[str, object] | None = None):
    session_id = session_store.ensure_session()
    action = session_store.create_pending_action(
        session_id,
        action_type="book_day_type",
        validated_arguments=arguments
        or {
            "date_from": "2026-09-22",
            "date_to": "2026-09-24",
            "day_type_id": 20,
            "day_type_name": "Annual Leave",
        },
        summary="Submit Annual Leave from 22 through 24 September?",
        language="en",
    )
    return session_id, action


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "detected_language", "tool_name"),
    [
        ("What is my vacation balance?", "en", "get_home_data"),
        ("How was my attendance this week?", "en", "get_attendance_summary"),
        ("Show my notifications", "en", "get_notifications"),
        ("أبغى أعرف رصيد إجازتي", "ar", "get_home_data"),
    ],
)
async def test_read_only_turn_never_invokes_confirmation_classifier(
    monkeypatch,
    message: str,
    detected_language: str,
    tool_name: str,
) -> None:
    store = InMemorySessionStore()

    async def classify(*args, **kwargs):
        raise AssertionError("Read-only turns must not invoke confirmation classification")

    async def run(*args, **kwargs):
        return AgentResult(message="Current ResourcePlus result.", tools_used=[tool_name])

    monkeypatch.setattr(chat_service, "classify_confirmation_intent", classify)
    monkeypatch.setattr(chat_service, "run_agent", run)
    trace, token = start_voice_trace()
    try:
        response = await chat_service.process_chat(
            ChatRequest(message=message),
            detected_language=detected_language,
            store=store,
        )
    finally:
        reset_voice_trace(token)

    assert response.success is True
    assert response.language == detected_language
    assert response.tools_used == [tool_name]
    assert trace.durations.get("confirmation_classifier", 0.0) == 0.0


@pytest.mark.asyncio
async def test_natural_english_confirmation_classifies_only_with_pending_action(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session("natural-confirmation")
    expected_arguments = {
        "date_from": "2026-09-22",
        "date_to": "2026-09-24",
        "day_type_id": 20,
        "day_type_name": "Annual Leave",
    }
    action = store.create_pending_action(
        session_id,
        action_type="book_day_type",
        validated_arguments=expected_arguments,
        summary="Submit Annual Leave from 22 through 24 September?",
        language="en",
    )
    # Mutating the caller-visible clone must not alter the store's authoritative
    # validated arguments.
    action.validated_arguments["date_to"] = "2099-01-01"
    classifications = []
    executions = []

    async def classify(message, *, pending_summary):
        classifications.append((message, pending_summary))
        return "CONFIRM"

    async def execute(action_type, arguments):
        executions.append((action_type, arguments))
        return {"success": True, "message": "Request submitted for approval"}

    async def render(facts, **kwargs):
        return facts

    monkeypatch.setattr(chat_service, "classify_confirmation_intent", classify)
    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    monkeypatch.setattr(chat_service, "render_user_message", render)

    response = await chat_service.process_chat(
        ChatRequest(
            message="Yeah, go ahead.",
            session_id=session_id,
            confirmation_id=action.confirmation_id,
        ),
        detected_language="en",
        store=store,
    )

    assert response.success is True
    assert classifications == [
        ("Yeah, go ahead.", "Submit Annual Leave from 22 through 24 September?")
    ]
    assert executions == [
        (
            "book_day_type",
            {
                "date_from": "2026-09-22",
                "date_to": "2026-09-24",
                "day_type_id": 20,
                "day_type_name": "Annual Leave",
            },
        )
    ]


def test_confirmation_executes_only_stored_action(monkeypatch) -> None:
    session_id, action = seed_pending()
    calls: list[tuple[str, dict[str, object]]] = []

    async def execute(action_type, arguments):
        calls.append((action_type, arguments))
        return {"success": True, "message": "Request submitted for approval"}

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    response = client.post(
        "/api/chat",
        json={
            "message": "Yes, please",
            "session_id": session_id,
            "confirmation_id": action.confirmation_id,
        },
    )
    assert response.status_code == 200
    assert response.json()["success"] is True
    assert response.json()["language"] == "en"
    assert response.json()["requires_confirmation"] is False
    assert calls == [
        (
            "book_day_type",
            {
                "date_from": "2026-09-22",
                "date_to": "2026-09-24",
                "day_type_id": 20,
                "day_type_name": "Annual Leave",
            },
        )
    ]


def test_no_confirmation_post_occurs_before_yes(monkeypatch) -> None:
    session_id, action = seed_pending()
    calls: list[object] = []

    async def execute(*args, **kwargs):
        calls.append((args, kwargs))
        return {"success": True, "message": "Unexpected"}

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    response = client.post(
        "/api/chat",
        json={"message": "Maybe", "session_id": session_id},
    )
    assert response.json()["requires_confirmation"] is True
    assert response.json()["confirmation_id"] == action.confirmation_id
    assert calls == []


def test_modifying_request_does_not_change_stored_operation(monkeypatch) -> None:
    session_id, action = seed_pending()
    calls: list[dict[str, object]] = []

    async def execute(action_type, arguments):
        calls.append(arguments)
        return {"success": True, "message": "Submitted"}

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    changed = client.post(
        "/api/chat",
        json={
            "message": "Change it to September 30",
            "session_id": session_id,
        },
    )
    assert changed.json()["requires_confirmation"] is True
    assert changed.json()["confirmation_id"] == action.confirmation_id
    assert calls == []

    confirmed = client.post(
        "/api/chat",
        json={"message": "Yes", "session_id": session_id},
    )
    assert confirmed.json()["success"] is True
    assert calls[0]["date_to"] == "2026-09-24"


def test_rejection_discards_pending_action(monkeypatch) -> None:
    session_id, _ = seed_pending()

    async def execute(*args, **kwargs):
        raise AssertionError("A rejected action must not execute")

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    response = client.post(
        "/api/chat",
        json={"message": "Don't do it", "session_id": session_id},
    )
    assert response.json()["success"] is True
    assert response.json()["requires_confirmation"] is False
    assert session_store.get_pending_action(session_id)[0] is None


@pytest.mark.asyncio
async def test_english_pending_no_uses_deterministic_english_rejection(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session("english-rejection")
    store.create_pending_action(
        session_id,
        action_type="book_day_type",
        validated_arguments={"date_from": "2026-09-22", "date_to": "2026-09-22"},
        summary="Submit Business Travel for 22 September?",
        language="en",
    )

    async def execute(*args, **kwargs):
        raise AssertionError("A rejected action must never execute")

    async def wrong_language_renderer(*args, **kwargs):
        return "تم إلغاء الإجراء المعلق بنجاح."

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    monkeypatch.setattr(chat_service, "render_user_message", wrong_language_renderer)
    response = await chat_service.process_chat(
        ChatRequest(message="No", session_id=session_id),
        detected_language="en",
        store=store,
    )

    assert response.language == "en"
    assert response.message == "The pending action was cancelled. Nothing was submitted."
    assert store.get_pending_action(session_id)[0] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "detected_language"),
    [("لا", "ar"), ("No", "en")],
)
async def test_arabic_pending_short_rejection_keeps_arabic_flow_language(
    monkeypatch,
    reply: str,
    detected_language: str,
) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session("arabic-rejection")
    store.create_pending_action(
        session_id,
        action_type="book_day_type",
        validated_arguments={"date_from": "2026-09-22", "date_to": "2026-09-22"},
        summary="هل تريد تأكيد طلب رحلة العمل؟",
        language="ar",
    )

    async def classify(*args, **kwargs):
        return "REJECT"

    async def execute(*args, **kwargs):
        raise AssertionError("A rejected action must never execute")

    monkeypatch.setattr(chat_service, "classify_confirmation_intent", classify)
    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    response = await chat_service.process_chat(
        ChatRequest(message=reply, session_id=session_id),
        detected_language=detected_language,
        store=store,
    )

    assert response.language == "ar"
    assert response.message.startswith("تم إلغاء الإجراء")
    assert store.get_pending_action(session_id)[0] is None


@pytest.mark.asyncio
async def test_meaningful_language_switch_changes_acknowledgement_not_safety(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session("language-switch")
    store.create_pending_action(
        session_id,
        action_type="book_day_type",
        validated_arguments={"date_from": "2026-09-22", "date_to": "2026-09-22"},
        summary="Submit Business Travel for 22 September?",
        language="en",
    )

    async def classify(*args, **kwargs):
        return "REJECT"

    async def execute(*args, **kwargs):
        raise AssertionError("Language switching must not execute a rejection")

    monkeypatch.setattr(chat_service, "classify_confirmation_intent", classify)
    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    response = await chat_service.process_chat(
        ChatRequest(message="لا، لا أريد تنفيذ هذا الطلب", session_id=session_id),
        detected_language="ar",
        store=store,
    )

    assert response.language == "ar"
    assert response.message.startswith("تم إلغاء الإجراء")


@pytest.mark.asyncio
async def test_expired_pending_action_uses_stored_language_without_execution(
    monkeypatch,
) -> None:
    current = [datetime(2026, 9, 19, tzinfo=timezone.utc)]
    store = InMemorySessionStore(
        confirmation_ttl_seconds=1,
        now=lambda: current[0],
    )
    session_id = store.ensure_session("expired-language")
    store.create_pending_action(
        session_id,
        action_type="book_day_type",
        validated_arguments={"date_from": "2026-09-22", "date_to": "2026-09-22"},
        summary="Submit Business Travel?",
        language="en",
    )
    current[0] += timedelta(seconds=2)

    async def execute(*args, **kwargs):
        raise AssertionError("An expired action must never execute")

    async def classify(*args, **kwargs):
        raise AssertionError("An expired action must not invoke confirmation classification")

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    monkeypatch.setattr(chat_service, "classify_confirmation_intent", classify)
    response = await chat_service.process_chat(
        ChatRequest(message="Yes", session_id=session_id),
        detected_language="en",
        store=store,
    )

    assert response.success is False
    assert response.language == "en"
    assert response.message.startswith("The pending confirmation expired")
    assert response.requires_confirmation is False


@pytest.mark.parametrize("message", ["Yes", "No"])
def test_yes_or_no_without_pending_action_does_nothing(monkeypatch, message: str) -> None:
    calls: list[object] = []

    async def execute(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    response = client.post("/api/chat", json={"message": message})
    assert response.status_code == 200
    assert response.json()["success"] is False
    assert "no pending action" in response.json()["message"].lower()
    assert response.json()["session_id"]
    assert calls == []


def test_wrong_confirmation_id_does_not_execute(monkeypatch) -> None:
    session_id, action = seed_pending()
    calls: list[object] = []

    async def execute(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    response = client.post(
        "/api/chat",
        json={
            "message": "Confirm",
            "session_id": session_id,
            "confirmation_id": "wrong-id",
        },
    )
    assert response.json()["success"] is False
    assert response.json()["requires_confirmation"] is True
    assert response.json()["confirmation_id"] == action.confirmation_id
    assert response.json()["language"] == "en"
    assert "does not match" in response.json()["message"].lower()
    assert calls == []


def test_expired_confirmation_cannot_execute() -> None:
    current = [datetime(2026, 9, 19, tzinfo=timezone.utc)]
    store = InMemorySessionStore(
        confirmation_ttl_seconds=300,
        now=lambda: current[0],
    )
    session_id = store.ensure_session()
    store.create_pending_action(
        session_id,
        action_type="book_day_type",
        validated_arguments={"day_type_id": 20},
        summary="Confirm",
        language="en",
    )
    current[0] += timedelta(minutes=6)
    with pytest.raises(PendingActionExpired):
        store.consume_pending_action(session_id)


def test_history_is_bounded_and_isolated_by_session() -> None:
    store = InMemorySessionStore(history_limit=3)
    first = store.ensure_session()
    second = store.ensure_session()
    for index in range(5):
        store.append_history(first, "user", str(index))
    store.append_history(second, "user", "other")
    assert [item["content"] for item in store.get_history(first)] == ["2", "3", "4"]
    assert store.get_history(second) == [{"role": "user", "content": "other"}]
