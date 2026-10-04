import json
from datetime import date, datetime, timedelta, timezone

import pytest

from app.ai import actions, conversation
from app.ai.agent import AgentResult
from app.ai.sessions import InMemorySessionStore
from app.ai.tools import execute_tool
from app.audit import reset_interaction_audit, start_interaction_audit
from app.models.schemas import ChatRequest
from app.resourceplus.client import ResourcePlusHTTPError
from app.services import chat as chat_service


def _install_exceptional_entry_reads(monkeypatch, write_calls: list[dict]) -> None:
    async def suggestions(*args, **kwargs):
        return [
            {
                "attDate": "01/09/2026",
                "suggestedEntryTime": "01/09/2026 08:00",
                "entryType": "IN",
            }
        ]

    async def reasons(*args, **kwargs):
        return [
            {
                "reasonID": "exception-generation-live-id",
                "reasonName": "Exception Generation",
            },
            {
                "reasonID": "embassy-live-id",
                "reasonName": "Embassy Purposes",
            },
            {
                "reasonID": "family-live-id",
                "reasonName": "Family Circumstances",
            },
            {"reasonID": "other-live-id", "reasonName": "Other"},
        ]

    async def submit(**kwargs):
        write_calls.append(kwargs)
        return {"success": True, "message": "Request submitted for approval"}

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    monkeypatch.setattr(actions, "create_exceptional_entry", submit)


@pytest.mark.asyncio
@pytest.mark.parametrize("model_reason", [None, "Exception Generation"])
async def test_missing_reason_lists_live_reasons_without_pending_or_default(
    monkeypatch,
    model_reason: str | None,
) -> None:
    write_calls: list[dict] = []
    _install_exceptional_entry_reads(monkeypatch, write_calls)
    store = InMemorySessionStore()
    session_id = store.ensure_session(f"missing-reason-{model_reason}")
    audit, token = start_interaction_audit(
        input_mode="text",
        user_text="I want to correct the IN punch on September 1",
        session_id=session_id,
    )
    try:
        result = await execute_tool(
            "prepare_exceptional_entry",
            {
                "target_date": "2026-09-01",
                "punch_direction": "IN",
                # Cover both the correct null tool argument and a model trying to
                # fill the first live reason even though the employee did not say it.
                "reason_name": model_reason,
                "remarks": None,
            },
            lang=1,
            session_id=session_id,
            response_language="en",
            source_user_message="I want to correct the IN punch on September 1",
            store=store,
        )
    finally:
        reset_interaction_audit(token)

    body = json.loads(result.output)
    assert body["needs_reason"] is True
    assert body["reason_options"] == [
        {"label": "Exception Generation", "value": "Exception Generation"},
        {"label": "Embassy Purposes", "value": "Embassy Purposes"},
        {"label": "Family Circumstances", "value": "Family Circumstances"},
        {"label": "Other", "value": "Other"},
    ]
    assert body["requires_confirmation"] is False
    assert body["requires_clarification"] is True
    assert "01 September 2026" in body["message"]
    assert "IN punch" in body["message"]
    assert "8:00 AM" in body["message"]
    assert "Exception Generation" in body["message"]
    assert "Family Circumstances" in body["message"]
    assert "Other" in body["message"]
    assert "exception-generation-live-id" not in result.output
    assert "family-live-id" not in result.output
    assert "reasonID" not in result.output
    assert result.pending_action is None
    assert store.get_pending_action(session_id)[0] is None
    draft = store.get_exceptional_entry_draft(session_id)
    assert draft is not None
    assert draft.attendance_date == "2026-09-01"
    assert draft.entry_type == "IN"
    assert draft.suggested_entry_time == "01/09/2026 08:00"
    assert draft.language == "en"
    assert draft.reason_options == (
        "Exception Generation",
        "Embassy Purposes",
        "Family Circumstances",
        "Other",
    )
    assert audit.action_type is None
    assert audit.action_state == "none"
    assert audit.confirmation_required is False
    assert write_calls == []


@pytest.mark.asyncio
async def test_less_hours_without_actionable_suggestion_never_offers_reasons(
    monkeypatch,
) -> None:
    calls = {"suggestions": 0, "reasons": 0, "posts": 0}

    async def no_suggestions(*args, **kwargs):
        calls["suggestions"] += 1
        return []

    async def forbidden_reasons(*args, **kwargs):
        calls["reasons"] += 1
        raise AssertionError("Reasons require an actionable suggestion")

    async def forbidden_post(**kwargs):
        calls["posts"] += 1
        raise AssertionError("A non-actionable less-hours record cannot be posted")

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", no_suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", forbidden_reasons)
    monkeypatch.setattr(actions, "create_exceptional_entry", forbidden_post)
    store = InMemorySessionStore()
    session_id = store.ensure_session("less-hours-with-both-punches")

    # AttendanceSummary may report IN=15:34, OUT=16:14 and LessHrs=07:20.
    # That fact is deliberately not converted into a missing direction here.
    result = await execute_tool(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-10",
            "punch_direction": None,
            "reason_name": None,
            "remarks": None,
        },
        lang=1,
        session_id=session_id,
        response_language="en",
        source_user_message="correct my less hour on Sep 10",
        store=store,
    )

    body = json.loads(result.output)
    assert calls == {"suggestions": 1, "reasons": 0, "posts": 0}
    assert body["needs_reason"] is False
    assert body["reason_options"] == []
    assert "valid suggested punch correction" in body["message"]
    assert "Available reasons" not in body["message"]
    assert "missing IN" not in body["message"]
    assert "missing OUT" not in body["message"]
    assert store.get_exceptional_entry_draft(session_id) is None
    assert store.get_pending_action(session_id)[0] is None


def _install_counted_reason_follow_up(monkeypatch, reason_rows=None):
    calls = {"suggestions": 0, "reasons": 0, "posts": 0}

    async def suggestions(*args, **kwargs):
        calls["suggestions"] += 1
        return [
            {
                "attDate": "01/09/2026",
                "suggestedEntryTime": "01/09/2026 08:00",
                "entryType": "IN",
            }
        ]

    async def reasons(*args, **kwargs):
        calls["reasons"] += 1
        return reason_rows or [
            {"reasonID": "embassy-live-id", "reasonName": "Embassy Purposes"},
            {
                "reasonID": "family-live-id",
                "reasonName": "Family Circumstances",
            },
            {"reasonID": "other-live-id", "reasonName": "Other"},
        ]

    async def submit(**kwargs):
        calls["posts"] += 1
        raise AssertionError("Reason selection must never execute a live write")

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    monkeypatch.setattr(actions, "create_exceptional_entry", submit)
    return calls


def _store_reason_draft(
    store: InMemorySessionStore,
    session_id: str,
    *,
    language: str = "en",
    entry_type: str = "IN",
    reason_options: list[str] | None = None,
) -> None:
    store.create_exceptional_entry_draft(
        session_id,
        attendance_date="2026-09-01",
        entry_type=entry_type,
        suggested_entry_time=(
            "01/09/2026 08:00" if entry_type == "IN" else "01/09/2026 17:00"
        ),
        language=language,
        reason_options=reason_options
        or ["Embassy Purposes", "Family Circumstances", "Other"],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("user_reason", "expected_reason", "expected_reason_id"),
    [
        ("embassy purpose", "Embassy Purposes", "embassy-live-id"),
        ("family circumstance", "Family Circumstances", "family-live-id"),
    ],
)
async def test_draft_reason_follow_up_is_deterministic_and_bounded(
    monkeypatch,
    user_reason: str,
    expected_reason: str,
    expected_reason_id: str,
) -> None:
    calls = _install_counted_reason_follow_up(monkeypatch)
    store = InMemorySessionStore()
    session_id = store.ensure_session(f"bounded-{expected_reason_id}")
    _store_reason_draft(store, session_id)

    async def no_agent(*args, **kwargs):
        raise AssertionError("A draft reason must bypass the generic OpenAI tool loop")

    monkeypatch.setattr(chat_service, "run_agent", no_agent)
    audit, token = start_interaction_audit(
        input_mode="text",
        user_text=user_reason,
        session_id=session_id,
    )
    try:
        response = await chat_service.process_chat(
            ChatRequest(message=user_reason, session_id=session_id),
            detected_language="en",
            store=store,
        )
    finally:
        reset_interaction_audit(token)

    pending, expired = store.get_pending_action(session_id)
    assert expired is False
    assert pending is not None
    assert response.requires_confirmation is True
    assert response.confirmation_id == pending.confirmation_id
    assert response.language == "en"
    assert pending.validated_arguments["reason_id"] == expected_reason_id
    assert pending.validated_arguments["reason_name"] == expected_reason
    assert pending.validated_arguments["entry_time"] == "01/09/2026 08:00"
    assert "01 September 2026" in response.message
    assert "IN punch" in response.message
    assert "8:00 AM" in response.message
    assert expected_reason in response.message
    assert store.get_exceptional_entry_draft(session_id) is None
    assert calls == {"suggestions": 1, "reasons": 1, "posts": 0}
    assert audit.model_requests == 0
    assert audit.error_category is None
    assert audit.action_type == "create_exceptional_entry"
    assert audit.action_state == "pending_confirmation"
    assert audit.confirmation_required is True


@pytest.mark.asyncio
async def test_repeated_reason_selection_cannot_create_multiple_pending_actions(
    monkeypatch,
) -> None:
    calls = _install_counted_reason_follow_up(monkeypatch)
    store = InMemorySessionStore()
    session_id = store.ensure_session("repeated-reason-selection")
    _store_reason_draft(store, session_id)

    first = await chat_service.process_chat(
        ChatRequest(message="Embassy Purposes", session_id=session_id),
        detected_language="en",
        store=store,
    )
    pending, _ = store.get_pending_action(session_id)
    assert pending is not None

    async def classify(*args, **kwargs):
        return "OTHER"

    async def render(facts, **kwargs):
        return facts

    monkeypatch.setattr(chat_service, "classify_confirmation_intent", classify)
    monkeypatch.setattr(chat_service, "render_user_message", render)
    repeated = await chat_service.process_chat(
        ChatRequest(message="Embassy Purposes", session_id=session_id),
        detected_language="en",
        store=store,
    )
    still_pending, _ = store.get_pending_action(session_id)

    assert first.confirmation_id == pending.confirmation_id
    assert repeated.confirmation_id == pending.confirmation_id
    assert still_pending is not None
    assert still_pending.confirmation_id == pending.confirmation_id
    assert calls == {"suggestions": 1, "reasons": 1, "posts": 0}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reason_text", "reason_rows", "expected_fragment"),
    [
        (
            "medical appointment",
            [
                {"reasonID": "embassy-id", "reasonName": "Embassy Purposes"},
                {"reasonID": "family-id", "reasonName": "Family Circumstances"},
            ],
            "couldn't match",
        ),
        (
            "family",
            [
                {"reasonID": "family-one", "reasonName": "Family Circumstances"},
                {"reasonID": "family-two", "reasonName": "Family Emergency"},
            ],
            "more than one",
        ),
    ],
)
async def test_unknown_or_ambiguous_draft_reason_stops_without_loop(
    monkeypatch,
    reason_text: str,
    reason_rows: list[dict[str, str]],
    expected_fragment: str,
) -> None:
    calls = _install_counted_reason_follow_up(monkeypatch, reason_rows)
    store = InMemorySessionStore()
    session_id = store.ensure_session(f"clarify-{reason_text}")
    _store_reason_draft(store, session_id)

    async def no_agent(*args, **kwargs):
        raise AssertionError("Clarification must terminate without the agent loop")

    monkeypatch.setattr(chat_service, "run_agent", no_agent)
    response = await chat_service.process_chat(
        ChatRequest(message=reason_text, session_id=session_id),
        detected_language="en",
        store=store,
    )

    assert response.requires_confirmation is False
    assert expected_fragment in response.message
    assert store.get_pending_action(session_id)[0] is None
    assert store.get_exceptional_entry_draft(session_id) is not None
    assert calls == {"suggestions": 1, "reasons": 1, "posts": 0}


@pytest.mark.asyncio
async def test_greeting_retains_draft_and_uses_normal_conversation(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session("draft-greeting")
    _store_reason_draft(store, session_id)
    store.append_history(session_id, "user", "I forgot to punch in today.")
    store.append_history(session_id, "assistant", "What was the reason?")
    agent_calls = 0

    async def normal_agent(*args, **kwargs):
        nonlocal agent_calls
        agent_calls += 1
        assert kwargs["history"] == []
        return AgentResult("Hello! How can I help?", [])

    async def no_draft_reads(*args, **kwargs):
        raise AssertionError("A greeting must not re-run exceptional-entry reads")

    monkeypatch.setattr(chat_service, "run_agent", normal_agent)
    monkeypatch.setattr(
        chat_service,
        "prepare_exceptional_entry_reason_follow_up",
        no_draft_reads,
    )
    response = await chat_service.process_chat(
        ChatRequest(message="Hello, how are you?", session_id=session_id),
        detected_language="en",
        store=store,
    )

    assert response.message == "Hello! How can I help?"
    assert agent_calls == 1
    assert store.get_exceptional_entry_draft(session_id) is not None
    assert store.get_pending_action(session_id)[0] is None
    assert store.get_history(session_id)[0] == {
        "role": "user",
        "content": "I forgot to punch in today.",
    }


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "agent_message", "tool_name"),
    [
        ("My punch is missing today.", "Which punch is missing?", "prepare_exceptional_entry"),
        ("Show my attendance.", "Your attendance is ready.", "get_attendance_summary"),
        ("Show my profile.", "Your profile is ready.", "get_profile_data"),
    ],
)
async def test_supported_hr_intent_abandons_reason_draft_and_routes_normally(
    monkeypatch,
    message: str,
    agent_message: str,
    tool_name: str,
) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session(f"draft-new-intent-{tool_name}")
    _store_reason_draft(store, session_id, entry_type="OUT")
    store.append_history(session_id, "user", "I forgot to punch out today.")
    store.append_history(session_id, "assistant", "What was the reason?")
    agent_calls = 0

    async def normal_agent(user_message, *args, **kwargs):
        nonlocal agent_calls
        agent_calls += 1
        assert user_message == message
        assert kwargs["history"] == []
        assert store.get_exceptional_entry_draft(session_id) is None
        return AgentResult(agent_message, [tool_name], speech_message=agent_message)

    async def no_reason_follow_up(*args, **kwargs):
        raise AssertionError("A new HR intent must not be consumed as a reason")

    async def no_resourceplus_write(**kwargs):
        raise AssertionError("The abandoned draft must never cause a ResourcePlus write")

    monkeypatch.setattr(chat_service, "run_agent", normal_agent)
    monkeypatch.setattr(
        chat_service,
        "prepare_exceptional_entry_reason_follow_up",
        no_reason_follow_up,
    )
    monkeypatch.setattr(actions, "create_exceptional_entry", no_resourceplus_write)
    if tool_name == "prepare_exceptional_entry":
        day = actions.LessHoursDay(
            date(2026, 10, 1), "Regular", "09:10", "17:00",
            "07:50", "00:10", "eligible", {},
        )

        async def inspect(*args, **kwargs):
            return actions.LessHoursInspection((day,), (day,))

        async def reasons(*args, **kwargs):
            return [{"reasonID": "live-id", "reasonName": "Traffic"}]

        async def balance(*args, **kwargs):
            return {"hasPolicy": True, "remaining": 120}

        monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
        monkeypatch.setattr(conversation, "cached_exception_reasons", reasons)
        monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)
    response = await chat_service.process_chat(
        ChatRequest(message=message, session_id=session_id),
        detected_language="en",
        store=store,
    )

    if tool_name == "prepare_exceptional_entry":
        assert response.needs_reason is True
        assert "reason" in response.message.lower()
        assert agent_calls == 0
    else:
        assert response.message == agent_message
        assert agent_calls == 1
    assert "couldn't match that reason" not in response.message
    assert store.get_exceptional_entry_draft(session_id) is None
    assert store.get_pending_action(session_id)[0] is None


@pytest.mark.asyncio
async def test_profile_intent_permanently_abandons_old_out_draft(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session("draft-profile-then-old-reason")
    _store_reason_draft(
        store,
        session_id,
        entry_type="OUT",
        reason_options=["Outside Work", "Family Circumstances", "Other"],
    )
    store.append_history(session_id, "user", "I forgot to punch out today.")
    store.append_history(
        session_id,
        "assistant",
        "What was the reason? Outside Work, Family Circumstances, or Other?",
    )
    agent_calls: list[str] = []
    write_calls = 0

    async def normal_agent(user_message, *args, **kwargs):
        agent_calls.append(user_message)
        if user_message == "Show my profile":
            assert kwargs["history"] == []
            return AgentResult("Your profile is ready.", ["get_profile_data"])
        assert user_message == "Outside Work"
        assert kwargs["history"] == [
            {"role": "user", "content": "Show my profile"},
            {"role": "assistant", "content": "Your profile is ready."},
        ]
        return AgentResult("How can I help with that?", [])

    async def no_reason_follow_up(*args, **kwargs):
        raise AssertionError("An abandoned draft must not resume reason selection")

    async def no_resourceplus_write(**kwargs):
        nonlocal write_calls
        write_calls += 1
        raise AssertionError("An abandoned draft must never write to ResourcePlus")

    monkeypatch.setattr(chat_service, "run_agent", normal_agent)
    monkeypatch.setattr(
        chat_service,
        "prepare_exceptional_entry_reason_follow_up",
        no_reason_follow_up,
    )
    monkeypatch.setattr(actions, "create_exceptional_entry", no_resourceplus_write)

    profile_response = await chat_service.process_chat(
        ChatRequest(message="Show my profile", session_id=session_id),
        detected_language="en",
        store=store,
    )
    assert profile_response.tools_used == ["get_profile_data"]
    assert store.get_exceptional_entry_draft(session_id) is None

    later_response = await chat_service.process_chat(
        ChatRequest(message="Outside Work", session_id=session_id),
        detected_language="en",
        store=store,
    )

    assert later_response.requires_confirmation is False
    assert "prepare_exceptional_entry" not in later_response.tools_used
    assert store.get_exceptional_entry_draft(session_id) is None
    assert store.get_pending_action(session_id)[0] is None
    assert write_calls == 0
    assert agent_calls == ["Show my profile", "Outside Work"]


@pytest.mark.asyncio
async def test_waiting_for_reason_routes_prospective_lateness_without_old_action(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session("draft-prospective-late")
    _store_reason_draft(store, session_id, entry_type="OUT")
    store.append_history(session_id, "user", "I forgot to punch out today.")
    store.append_history(session_id, "assistant", "What was the reason?")

    async def forbidden(*args, **kwargs):
        raise AssertionError("Late arrival must not enter the reason or agent tool flow")

    monkeypatch.setattr(chat_service, "run_agent", forbidden)
    monkeypatch.setattr(
        chat_service,
        "prepare_exceptional_entry_reason_follow_up",
        forbidden,
    )
    response = await chat_service.process_chat(
        ChatRequest(
            message="I'll be 20 minutes late today.",
            session_id=session_id,
        ),
        detected_language="en",
        store=store,
    )

    assert "isn't connected yet" in response.message
    assert store.get_exceptional_entry_draft(session_id) is None
    assert store.get_pending_action(session_id)[0] is None
    assert all(
        "forgot to punch out" not in item["content"]
        for item in store.get_history(session_id)
    )


@pytest.mark.asyncio
async def test_short_invalid_reason_like_text_still_lists_live_options(
    monkeypatch,
) -> None:
    calls = _install_counted_reason_follow_up(monkeypatch)
    store = InMemorySessionStore()
    session_id = store.ensure_session("draft-invalid-reason-like")
    _store_reason_draft(store, session_id)

    async def no_agent(*args, **kwargs):
        raise AssertionError("A reason-like reply must stay in deterministic reason flow")

    monkeypatch.setattr(chat_service, "run_agent", no_agent)
    response = await chat_service.process_chat(
        ChatRequest(message="Traffic jam", session_id=session_id),
        detected_language="en",
        store=store,
    )

    assert "couldn't match that reason" in response.message
    assert response.needs_reason is True
    assert response.reason_options
    assert calls == {"suggestions": 1, "reasons": 1, "posts": 0}
    assert store.get_exceptional_entry_draft(session_id) is not None
    assert store.get_pending_action(session_id)[0] is None


@pytest.mark.asyncio
async def test_arabic_missing_punch_intent_is_not_consumed_as_reason(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session("draft-arabic-new-intent")
    _store_reason_draft(store, session_id, language="ar", entry_type="OUT")
    message = "بصمتي ناقصة اليوم"

    async def normal_agent(user_message, *args, **kwargs):
        assert user_message == message
        assert kwargs["history"] == []
        return AgentResult("أي بصمة تريد تصحيحها؟", ["prepare_exceptional_entry"])

    async def no_reason_follow_up(*args, **kwargs):
        raise AssertionError("Arabic HR intent must not be consumed as a reason")

    monkeypatch.setattr(chat_service, "run_agent", normal_agent)
    monkeypatch.setattr(
        chat_service,
        "prepare_exceptional_entry_reason_follow_up",
        no_reason_follow_up,
    )
    day = actions.LessHoursDay(
        date(2026, 10, 1), "Regular", "09:10", "17:00",
        "07:50", "00:10", "eligible", {},
    )

    async def inspect(*args, **kwargs):
        return actions.LessHoursInspection((day,), (day,))

    async def reasons(*args, **kwargs):
        return [{"reasonID": "live-id", "reasonName": "Traffic"}]

    async def balance(*args, **kwargs):
        return {"hasPolicy": True, "remaining": 120}

    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    monkeypatch.setattr(conversation, "cached_exception_reasons", reasons)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)
    response = await chat_service.process_chat(
        ChatRequest(message=message, session_id=session_id),
        detected_language="ar",
        store=store,
    )

    assert response.needs_reason is True
    assert response.message.endswith("وش سبب التصحيح؟")
    assert store.get_exceptional_entry_draft(session_id) is None
    assert store.get_pending_action(session_id)[0] is None


@pytest.mark.asyncio
async def test_cancel_clears_non_executable_draft_without_post(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session("draft-cancel")
    _store_reason_draft(store, session_id)

    async def forbidden(*args, **kwargs):
        raise AssertionError("Draft cancellation must not invoke agents or writes")

    monkeypatch.setattr(chat_service, "run_agent", forbidden)
    monkeypatch.setattr(chat_service, "execute_pending_action", forbidden)
    response = await chat_service.process_chat(
        ChatRequest(message="cancel", session_id=session_id),
        detected_language="en",
        store=store,
    )

    assert response.success is True
    assert response.requires_confirmation is False
    assert response.message == "Okay, I won't submit it."
    assert store.get_exceptional_entry_draft(session_id) is None
    assert store.get_pending_action(session_id)[0] is None


@pytest.mark.asyncio
async def test_arabic_draft_preserves_transaction_language(monkeypatch) -> None:
    calls = _install_counted_reason_follow_up(
        monkeypatch,
        [{"reasonID": "family-ar-id", "reasonName": "ظروف عائلية"}],
    )
    store = InMemorySessionStore()
    session_id = store.ensure_session("arabic-draft")
    _store_reason_draft(store, session_id, language="ar")

    async def no_agent(*args, **kwargs):
        raise AssertionError("Arabic reason selection must stay deterministic")

    monkeypatch.setattr(chat_service, "run_agent", no_agent)
    response = await chat_service.process_chat(
        ChatRequest(message="ظروف عائلية", session_id=session_id),
        detected_language="ar",
        store=store,
    )

    assert response.language == "ar"
    assert response.requires_confirmation is True
    assert "طلب تصحيح البصمة" in response.message
    assert "ظروف عائلية" in response.message
    assert calls == {"suggestions": 1, "reasons": 1, "posts": 0}


def test_exceptional_entry_draft_expires_without_becoming_pending() -> None:
    now = [datetime(2026, 9, 22, tzinfo=timezone.utc)]
    store = InMemorySessionStore(
        confirmation_ttl_seconds=1,
        now=lambda: now[0],
    )
    session_id = store.ensure_session("expiring-draft")
    _store_reason_draft(store, session_id)

    now[0] += timedelta(seconds=2)

    assert store.get_exceptional_entry_draft(session_id) is None
    assert store.get_pending_action(session_id)[0] is None


@pytest.mark.asyncio
async def test_later_user_reason_is_revalidated_and_creates_pending_action(
    monkeypatch,
) -> None:
    write_calls: list[dict] = []
    _install_exceptional_entry_reads(monkeypatch, write_calls)
    store = InMemorySessionStore()
    session_id = store.ensure_session("later-reason")
    audit, token = start_interaction_audit(
        input_mode="text",
        user_text="Family Circumstances",
        session_id=session_id,
    )
    try:
        result = await execute_tool(
            "prepare_exceptional_entry",
            {
                "target_date": "2026-09-01",
                "punch_direction": "IN",
                "reason_name": "Family Circumstances",
                "remarks": None,
            },
            lang=1,
            session_id=session_id,
            response_language="en",
            source_user_message=(
                "Correct my IN punch because of Family Circumstances"
            ),
            store=store,
        )
    finally:
        reset_interaction_audit(token)

    pending = result.pending_action
    assert pending is not None
    assert pending.validated_arguments["reason_id"] == "family-live-id"
    assert pending.validated_arguments["reason_name"] == "Family Circumstances"
    assert pending.validated_arguments["entry_time"] == "01/09/2026 08:00"
    assert "01 September 2026" in pending.summary
    assert "IN punch" in pending.summary
    assert "8:00 AM" in pending.summary
    assert "Family Circumstances" in pending.summary
    assert audit.action_type == "create_exceptional_entry"
    assert audit.action_state == "pending_confirmation"
    assert audit.confirmation_required is True
    assert write_calls == []


@pytest.mark.asyncio
async def test_reason_in_initial_request_allows_direct_safe_preparation(monkeypatch) -> None:
    write_calls: list[dict] = []
    _install_exceptional_entry_reads(monkeypatch, write_calls)
    store = InMemorySessionStore()
    result = await execute_tool(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-01",
            "punch_direction": "IN",
            "reason_name": "Family Circumstances",
            "remarks": "Family Circumstances",
        },
        lang=1,
        session_id=store.ensure_session("initial-reason"),
        response_language="en",
        source_user_message=(
            "Correct my September 1 IN punch because of Family Circumstances"
        ),
        store=store,
    )

    assert result.pending_action is not None
    assert result.pending_action.validated_arguments["reason_id"] == "family-live-id"
    assert write_calls == []


def _pending_exceptional_entry(store: InMemorySessionStore, language: str):
    session_id = store.ensure_session(f"result-{language}")
    pending = store.create_pending_action(
        session_id,
        action_type="create_exceptional_entry",
        validated_arguments={
            "entry_time": "01/09/2026 08:00",
            "entry_type": 1,
            "reason_id": "family-live-id",
            "reason_name": "Family Circumstances",
            "remarks": "Family Circumstances",
            "attendance_date": "2026-09-01",
            "shift": None,
            "is_night_shift": None,
        },
        summary="Submit the exceptional-entry request?",
        language=language,
    )
    return session_id, pending


@pytest.mark.asyncio
async def test_english_transaction_ignores_arabic_resourceplus_success(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id, pending = _pending_exceptional_entry(store, "en")
    executions: list[dict] = []

    async def execute(action_type, arguments):
        executions.append(dict(arguments))
        return {"success": True, "message": "تم تسجيل الإدخال بنجاح."}

    async def no_renderer(*args, **kwargs):
        raise AssertionError("A known transaction result must not require the model")

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    monkeypatch.setattr(chat_service, "render_user_message", no_renderer)
    audit, token = start_interaction_audit(
        input_mode="text",
        user_text="Yes",
        session_id=session_id,
    )
    try:
        response = await chat_service.process_chat(
            ChatRequest(
                message="Yes",
                session_id=session_id,
                confirmation_id=pending.confirmation_id,
            ),
            detected_language="en",
            store=store,
        )
    finally:
        reset_interaction_audit(token)

    assert len(executions) == 1
    assert executions[0] == pending.validated_arguments
    assert response.language == "en"
    assert response.message == (
        "I've sent your attendance correction for approval."
    )
    assert response.speech_message == (
        "I've sent your attendance correction for approval."
    )
    assert "تم تسجيل" not in response.message
    assert audit.action_state == "executed"
    assert audit.confirmed is True
    assert audit.action_result == "submitted_for_approval"


@pytest.mark.asyncio
async def test_arabic_transaction_ignores_english_resourceplus_success(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id, pending = _pending_exceptional_entry(store, "ar")
    executions = 0

    async def execute(action_type, arguments):
        nonlocal executions
        executions += 1
        return {"success": True, "message": "Entry recorded successfully."}

    async def no_renderer(*args, **kwargs):
        raise AssertionError("A known transaction result must not require the model")

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    monkeypatch.setattr(chat_service, "render_user_message", no_renderer)
    response = await chat_service.process_chat(
        ChatRequest(
            message="Yes",
            session_id=session_id,
            confirmation_id=pending.confirmation_id,
        ),
        detected_language="en",
        store=store,
    )

    assert executions == 1
    assert response.language == "ar"
    assert response.message == "أرسلت تصحيح حضورك للموافقة."
    assert response.speech_message == "أرسلت تصحيح حضورك للموافقة."
    assert "تصحيح حضورك" in response.message
    assert "للموافقة" in response.message
    assert "Entry recorded" not in response.message


@pytest.mark.asyncio
async def test_short_arabic_confirmation_does_not_flip_english_transaction(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()
    session_id, pending = _pending_exceptional_entry(store, "en")

    async def classify(*args, **kwargs):
        return "CONFIRM"

    async def execute(*args, **kwargs):
        return {"success": True, "message": "تم تسجيل الإدخال بنجاح."}

    monkeypatch.setattr(chat_service, "classify_confirmation_intent", classify)
    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    response = await chat_service.process_chat(
        ChatRequest(
            message="نعم",
            session_id=session_id,
            confirmation_id=pending.confirmation_id,
        ),
        detected_language="ar",
        store=store,
    )

    assert response.language == "en"
    assert response.message.startswith("I've sent your attendance correction")


@pytest.mark.asyncio
async def test_known_failure_is_rendered_in_transaction_language(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id, pending = _pending_exceptional_entry(store, "en")

    async def execute(*args, **kwargs):
        return {"success": False, "message": "تعذر تسجيل الإدخال."}

    async def no_renderer(*args, **kwargs):
        raise AssertionError("A known transaction result must not require the model")

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    monkeypatch.setattr(chat_service, "render_user_message", no_renderer)
    response = await chat_service.process_chat(
        ChatRequest(
            message="Yes",
            session_id=session_id,
            confirmation_id=pending.confirmation_id,
        ),
        detected_language="en",
        store=store,
    )

    assert response.success is False
    assert response.language == "en"
    assert response.message == (
        "I couldn't send your attendance correction. Nothing was recorded."
    )
    assert response.speech_message == (
        "I couldn't send your attendance correction, so nothing was recorded."
    )


@pytest.mark.asyncio
async def test_upstream_500_is_failed_deterministic_and_not_replayable(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()
    session_id, pending = _pending_exceptional_entry(store, "en")
    executions = 0

    async def execute(*args, **kwargs):
        nonlocal executions
        executions += 1
        raise ResourcePlusHTTPError(
            500,
            endpoint="api/AI/ExceptionalEntries/Request",
        )

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    audit, token = start_interaction_audit(
        input_mode="text",
        user_text="Yes",
        session_id=session_id,
    )
    try:
        response = await chat_service.process_chat(
            ChatRequest(
                message="Yes",
                session_id=session_id,
                confirmation_id=pending.confirmation_id,
            ),
            detected_language="en",
            store=store,
        )
    finally:
        reset_interaction_audit(token)

    assert response.success is False
    assert response.message == (
        "I couldn't send your attendance correction. Nothing was recorded."
    )
    assert "success" not in response.message.casefold()
    assert response.display_message == response.message
    assert audit.action_state == "failed"
    assert audit.action_result == "failed"
    assert audit.confirmed is True
    assert audit.error_category == "resourceplus_error"
    assert store.get_pending_action(session_id)[0] is None

    replay = await chat_service.process_chat(
        ChatRequest(
            message="Yes",
            session_id=session_id,
            confirmation_id=pending.confirmation_id,
        ),
        detected_language="en",
        store=store,
    )
    assert replay.success is False
    assert replay.message == "There is no pending action to confirm or reject."
    assert executions == 1


@pytest.mark.asyncio
async def test_upstream_500_uses_arabic_transaction_failure_message(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id, pending = _pending_exceptional_entry(store, "ar")

    async def classify(*args, **kwargs):
        return "CONFIRM"

    async def execute(*args, **kwargs):
        raise ResourcePlusHTTPError(500)

    monkeypatch.setattr(chat_service, "classify_confirmation_intent", classify)
    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    response = await chat_service.process_chat(
        ChatRequest(
            message="نعم",
            session_id=session_id,
            confirmation_id=pending.confirmation_id,
        ),
        detected_language="ar",
        store=store,
    )

    assert response.success is False
    assert response.language == "ar"
    assert response.message == (
        "ما قدرت أرسل تصحيح حضورك. ما تسجّل أي طلب."
    )
    assert response.speech_message == (
        "ما قدرت أرسل تصحيح حضورك، وما تسجّل أي طلب."
    )
