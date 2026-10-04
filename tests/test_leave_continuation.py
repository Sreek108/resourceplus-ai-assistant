from __future__ import annotations

from datetime import date

import pytest

from app.ai import actions, agent, conversation
from app.ai.sessions import InMemorySessionStore
from app.models.schemas import ChatRequest
from app.services import chat as chat_service
from app.services import fast_reads


DAY_TYPES = [
    {"dayID": 11, "dayType": "Compassionate Leave", "group": "Leave"},
    {"dayID": 12, "dayType": "Sick Leave", "group": "Leave"},
    {"dayID": 13, "dayType": "Marriage Leave", "group": "Leave"},
    {"dayID": 21, "dayType": "Business Travel", "group": "Business Travel"},
    {"dayID": 31, "dayType": "Absent", "group": "Attendance"},
    {"dayID": 32, "dayType": "Holiday", "group": "Holiday"},
    {"dayID": 33, "dayType": "Week End", "group": "Week End"},
]


def _mock_day_types(monkeypatch, rows=DAY_TYPES):
    calls = []

    async def cached(*args, **kwargs):
        calls.append("cached")
        return rows

    async def fresh(*args, **kwargs):
        calls.append("fresh")
        return rows

    async def no_model(*args, **kwargs):
        raise AssertionError("supported leave selection must not call OpenAI")

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 10, 2))
    monkeypatch.setattr(conversation, "cached_day_types", cached)
    monkeypatch.setattr(chat_service, "cached_day_types", cached)
    monkeypatch.setattr(actions, "get_day_types", fresh)
    monkeypatch.setattr(chat_service, "run_agent", no_model)
    return calls


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("initial_message", "selection", "expected_id"),
    [
        ("apply leave for 10 sep", "Compassionate Leave", 11),
        ("then apply leave of 10 sep", "compassionate-leave", 11),
        ("I need leave on 10 September", "Sick Leave", 12),
    ],
)
async def test_bare_leave_type_uses_draft_date_and_live_id(
    monkeypatch, initial_message, selection, expected_id
) -> None:
    store = InMemorySessionStore()
    calls = _mock_day_types(monkeypatch)

    first = await chat_service.process_chat(ChatRequest(message=initial_message), store=store)
    draft = store.get_conversation_draft(first.session_id)
    assert first.requires_confirmation is False
    assert draft is not None and draft.intent == "book_day_type"
    assert draft.slots["date_from"] == "2026-09-10"
    assert draft.slots["date_to"] == "2026-09-10"
    assert draft.slots["booking_group"] == "leave"
    assert draft.slots["state"] == "awaiting_day_type"
    assert "Compassionate Leave" in draft.slots["day_type_options"]
    assert store.get_pending_action(first.session_id)[0] is None
    assert first.tools_used == ["get_day_types"]
    assert [row["type"] for row in first.blocks[0].rows] == [
        "Compassionate Leave", "Sick Leave", "Marriage Leave"
    ]

    prepared = await chat_service.process_chat(
        ChatRequest(message=selection, session_id=first.session_id), store=store
    )
    pending, _ = store.get_pending_action(first.session_id)
    assert prepared.requires_confirmation is True
    assert prepared.blocks[0].type == "confirmation"
    assert pending is not None and pending.action_type == "book_day_type"
    assert pending.validated_arguments["date_from"] == "2026-09-10"
    assert pending.validated_arguments["date_to"] == "2026-09-10"
    assert pending.validated_arguments["day_type_id"] == expected_id
    assert "10 September 2026" in prepared.message
    assert store.get_conversation_draft(first.session_id) is None
    assert calls == ["cached", "fresh"]


@pytest.mark.asyncio
async def test_unknown_and_ambiguous_day_types_keep_non_executable_draft(monkeypatch) -> None:
    store = InMemorySessionStore()
    _mock_day_types(monkeypatch, [
        *DAY_TYPES,
        {"dayID": 14, "dayType": "Sick Leave Extended", "group": "Leave"},
    ])
    first = await chat_service.process_chat(
        ChatRequest(message="apply leave for 10 sep"), store=store
    )
    unknown = await chat_service.process_chat(
        ChatRequest(message="Imaginary Leave", session_id=first.session_id), store=store
    )
    ambiguous = await chat_service.process_chat(
        ChatRequest(message="Sick", session_id=first.session_id), store=store
    )
    assert "couldn't match" in unknown.message
    assert "More than one" in ambiguous.message
    assert store.get_pending_action(first.session_id)[0] is None
    draft = store.get_conversation_draft(first.session_id)
    assert draft is not None and draft.slots["date_from"] == "2026-09-10"


@pytest.mark.asyncio
async def test_near_voice_match_uses_only_live_day_type_and_revalidates(monkeypatch) -> None:
    rows = [
        {"dayID": 77, "dayType": "Compensatory Leave", "group": "Leave"},
        {"dayID": 78, "dayType": "Personal Time", "group": "Leave"},
    ]
    store = InMemorySessionStore()
    calls = _mock_day_types(monkeypatch, rows)
    first = await chat_service.process_chat(
        ChatRequest(message="apply leave for 9 October"), store=store
    )

    response = await chat_service.process_chat(
        ChatRequest(message="Compensatory relief.", session_id=first.session_id),
        store=store,
    )

    pending, _ = store.get_pending_action(first.session_id)
    assert response.requires_confirmation is True
    assert pending is not None
    assert pending.validated_arguments["day_type_name"] == "Compensatory Leave"
    assert pending.validated_arguments["day_type_id"] == 77
    assert calls == ["cached", "fresh"]


@pytest.mark.asyncio
async def test_uncertain_voice_match_clarifies_live_day_type_without_action(monkeypatch) -> None:
    rows = [{"dayID": 77, "dayType": "Compensatory Leave", "group": "Leave"}]
    store = InMemorySessionStore()
    _mock_day_types(monkeypatch, rows)
    first = await chat_service.process_chat(
        ChatRequest(message="apply leave for 9 October"), store=store
    )

    response = await chat_service.process_chat(
        ChatRequest(message="Compensation relief.", session_id=first.session_id),
        store=store,
    )

    assert response.message == "Did you mean Compensatory Leave for Friday, Oct 9?"
    assert response.requires_confirmation is False
    assert store.get_pending_action(first.session_id)[0] is None
    assert store.get_conversation_draft(first.session_id) is not None


@pytest.mark.asyncio
async def test_business_travel_date_prepares_directly_from_live_type(monkeypatch) -> None:
    store = InMemorySessionStore()
    _mock_day_types(monkeypatch)
    response = await chat_service.process_chat(
        ChatRequest(message="business travel on 12 September"), store=store
    )
    pending, _ = store.get_pending_action(response.session_id)
    assert response.requires_confirmation is True
    assert pending is not None
    assert pending.validated_arguments["day_type_id"] == 21
    assert pending.validated_arguments["date_from"] == "2026-09-12"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options_message",
    ["Show available day types", "View Leave & Business Travel options"],
)
async def test_recovery_options_are_filtered_without_changing_general_day_types(
    monkeypatch, options_message
) -> None:
    store = InMemorySessionStore()
    _mock_day_types(monkeypatch)
    first = await chat_service.process_chat(
        ChatRequest(message="apply leave for 10 sep"), store=store
    )
    recovery = await chat_service.process_chat(
        ChatRequest(message=options_message, session_id=first.session_id),
        store=store,
    )
    assert [row["type"] for row in recovery.blocks[0].rows] == [
        "Compassionate Leave", "Sick Leave", "Marriage Leave"
    ]
    assert store.get_conversation_draft(first.session_id) is not None

    async def general(*args, **kwargs):
        return DAY_TYPES

    monkeypatch.setattr(fast_reads, "cached_day_types", general)
    listing = await fast_reads.try_fast_read(
        "Show available day types", lang=1, response_language="en"
    )
    assert listing is not None
    assert {row["type"] for row in listing.blocks[0].rows} == {
        row["dayType"] for row in DAY_TYPES
    }


@pytest.mark.asyncio
async def test_leave_draft_is_superseded_by_profile_read(monkeypatch) -> None:
    store = InMemorySessionStore()
    _mock_day_types(monkeypatch)
    monkeypatch.setattr(chat_service, "run_agent", agent.run_agent)

    async def profile(*args, **kwargs):
        return {"EmployeeName": "Test Employee"}

    monkeypatch.setattr(fast_reads, "get_profile_data", profile)
    first = await chat_service.process_chat(
        ChatRequest(message="apply leave for 10 sep"), store=store
    )
    read = await chat_service.process_chat(
        ChatRequest(message="Show my profile", session_id=first.session_id), store=store
    )
    assert read.tools_used == ["get_profile_data"]
    assert store.get_conversation_draft(first.session_id) is None
    assert store.get_pending_action(first.session_id)[0] is None


@pytest.mark.asyncio
async def test_pending_action_is_not_replaced_by_leave_selection(monkeypatch) -> None:
    store = InMemorySessionStore()
    _mock_day_types(monkeypatch)
    prepared = await chat_service.process_chat(
        ChatRequest(message="business travel on 12 September"), store=store
    )
    pending_before, _ = store.get_pending_action(prepared.session_id)
    repeat = await chat_service.process_chat(
        ChatRequest(message="apply leave for 10 sep", session_id=prepared.session_id),
        store=store,
    )
    pending_after, _ = store.get_pending_action(prepared.session_id)
    assert repeat.requires_confirmation is True
    assert pending_after == pending_before
    assert store.get_conversation_draft(prepared.session_id) is None


@pytest.mark.asyncio
async def test_leave_type_selection_and_no_never_write(monkeypatch) -> None:
    store = InMemorySessionStore()
    _mock_day_types(monkeypatch)
    writes: list[object] = []

    async def forbidden_write(*args, **kwargs):
        writes.append((args, kwargs))
        raise AssertionError("leave booking must not POST before confirmation")

    monkeypatch.setattr(actions, "book_day_type", forbidden_write)
    first = await chat_service.process_chat(
        ChatRequest(message="apply leave for 10 sep"), store=store
    )
    assert store.get_pending_action(first.session_id)[0] is None
    assert writes == []
    prepared = await chat_service.process_chat(
        ChatRequest(message="Compassionate Leave", session_id=first.session_id),
        store=store,
    )
    assert prepared.requires_confirmation is True
    assert writes == []
    rejected = await chat_service.process_chat(
        ChatRequest(message="No", session_id=first.session_id), store=store
    )
    assert rejected.requires_confirmation is False
    assert store.get_pending_action(first.session_id)[0] is None
    assert writes == []


@pytest.mark.asyncio
async def test_arabic_bare_leave_type_continues_without_model(monkeypatch) -> None:
    rows = [
        {"dayID": 41, "dayType": "إجازة مرضية", "group": "إجازة"},
        {"dayID": 42, "dayType": "إجازة زواج", "group": "إجازة"},
        {"dayID": 43, "dayType": "عطلة رسمية", "group": "عطلة"},
    ]
    store = InMemorySessionStore()
    _mock_day_types(monkeypatch, rows)
    first = await chat_service.process_chat(
        ChatRequest(message="أريد إجازة يوم 10 سبتمبر"), store=store
    )
    assert first.language == "ar"
    assert first.requires_confirmation is False
    assert [row["type"] for row in first.blocks[0].rows] == ["إجازة مرضية", "إجازة زواج"]
    prepared = await chat_service.process_chat(
        ChatRequest(message="إجازة مرضية", session_id=first.session_id), store=store
    )
    pending, _ = store.get_pending_action(first.session_id)
    assert prepared.requires_confirmation is True
    assert pending is not None
    assert pending.validated_arguments["day_type_id"] == 41
    assert pending.validated_arguments["date_from"] == "2026-09-10"
    assert "أنت على وشك" in prepared.message


@pytest.mark.asyncio
async def test_leave_date_range_survives_follow_up(monkeypatch) -> None:
    store = InMemorySessionStore()
    _mock_day_types(monkeypatch)
    session_id = store.ensure_session()
    store.save_conversation_draft(
        session_id,
        intent="book_day_type",
        slots={
            "date_from": "2026-09-10",
            "date_to": "2026-09-12",
            "booking_group": "leave",
            "state": "awaiting_day_type",
        },
        language="en",
    )
    response = await chat_service.process_chat(
        ChatRequest(message="Sick Leave", session_id=session_id), store=store
    )
    pending, _ = store.get_pending_action(session_id)
    assert response.requires_confirmation is True
    assert pending is not None
    assert pending.validated_arguments["date_from"] == "2026-09-10"
    assert pending.validated_arguments["date_to"] == "2026-09-12"


@pytest.mark.asyncio
async def test_bare_business_travel_uses_recovery_date(monkeypatch) -> None:
    store = InMemorySessionStore()
    _mock_day_types(monkeypatch)
    session_id = store.ensure_session()
    store.save_conversation_draft(
        session_id,
        intent="book_day_type",
        slots={
            "date_from": "2026-09-10",
            "date_to": "2026-09-10",
            "state": "awaiting_day_type",
        },
        language="en",
    )
    prepared = await chat_service.process_chat(
        ChatRequest(message="Business Travel", session_id=session_id), store=store
    )
    pending, _ = store.get_pending_action(session_id)
    assert prepared.requires_confirmation is True
    assert pending is not None
    assert pending.validated_arguments["day_type_id"] == 21
    assert pending.validated_arguments["date_from"] == "2026-09-10"


def test_next_friday_is_resolved_deterministically() -> None:
    today = date(2026, 10, 2)

    assert conversation.extract_date("I need leave next Friday", today=today) == date(
        2026, 10, 9
    )
    assert conversation.extract_date("أحتاج إجازة الجمعة الجاية", today=today) == date(
        2026, 10, 9
    )


@pytest.mark.asyncio
async def test_next_friday_keeps_date_and_asks_only_for_live_leave_type(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()
    _mock_day_types(monkeypatch)

    response = await chat_service.process_chat(
        ChatRequest(message="I need leave next Friday"),
        store=store,
    )

    draft = store.get_conversation_draft(response.session_id)
    assert draft is not None
    assert draft.slots["date_from"] == "2026-10-09"
    assert draft.slots["state"] == "awaiting_day_type"
    assert "Friday, Oct 9" in response.message
    assert "Which leave type" in response.message
    assert "What date" not in response.message
    assert response.blocks[-1].type == "actions"
    assert [action.label for action in response.blocks[-1].actions] == [
        "Compassionate Leave",
        "Sick Leave",
        "Marriage Leave",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected_date", "date_copy"),
    [
        ("I need a leave on next Monday", "2026-10-05", "Monday, Oct 5"),
        ("I need leave tomorrow", "2026-10-03", "Saturday, Oct 3"),
        ("I want to take leave tomorrow", "2026-10-03", "Saturday, Oct 3"),
    ],
)
async def test_generic_leave_with_date_asks_only_for_live_type(
    monkeypatch, message, expected_date, date_copy
) -> None:
    store = InMemorySessionStore()
    _mock_day_types(monkeypatch)

    response = await chat_service.process_chat(ChatRequest(message=message), store=store)

    draft = store.get_conversation_draft(response.session_id)
    assert draft is not None and draft.slots["date_from"] == expected_date
    assert draft.slots["state"] == "awaiting_day_type"
    assert date_copy in response.message
    assert "Which leave type" in response.message
    assert "couldn't match that day type" not in response.message
    assert [action.label for action in response.blocks[-1].actions] == [
        "Compassionate Leave", "Sick Leave", "Marriage Leave"
    ]


@pytest.mark.asyncio
async def test_generic_leave_replaces_prior_options_date_without_day_type_mismatch(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()
    _mock_day_types(monkeypatch)
    session_id = store.ensure_session()
    store.save_conversation_draft(
        session_id,
        intent="book_day_type",
        slots={
            "booking_group": "leave",
            "state": "leave_options_shown",
            "day_type_options": '["Compassionate Leave", "Sick Leave", "Marriage Leave"]',
        },
        language="en",
    )

    response = await chat_service.process_chat(
        ChatRequest(message="I need a leave on next Monday", session_id=session_id),
        store=store,
    )

    draft = store.get_conversation_draft(session_id)
    assert draft is not None and draft.slots["date_from"] == "2026-10-05"
    assert response.message == "Sure — Monday, Oct 5. Which leave type would you like to use?"
    assert "couldn't match" not in response.message


@pytest.mark.asyncio
async def test_arabic_generic_leave_tomorrow_asks_only_for_live_type(monkeypatch) -> None:
    rows = [
        {"dayID": 41, "dayType": "إجازة تطوع", "group": "إجازة"},
        {"dayID": 42, "dayType": "إجازة زواج", "group": "إجازة"},
    ]
    store = InMemorySessionStore()
    _mock_day_types(monkeypatch, rows)

    response = await chat_service.process_chat(
        ChatRequest(message="أحتاج إجازة غدًا"), store=store
    )

    draft = store.get_conversation_draft(response.session_id)
    assert response.language == "ar"
    assert draft is not None and draft.slots["date_from"] == "2026-10-03"
    assert draft.slots["state"] == "awaiting_day_type"
    assert [action.label for action in response.blocks[-1].actions] == [
        "إجازة تطوع", "إجازة زواج"
    ]


@pytest.mark.asyncio
async def test_explicit_live_day_type_and_date_prepares_directly(monkeypatch) -> None:
    rows = [
        {"dayID": 77, "dayType": "Compensatory Leave", "group": "Leave"},
        {"dayID": 78, "dayType": "Study Break", "group": "Leave"},
    ]
    store = InMemorySessionStore()
    _mock_day_types(monkeypatch, rows)

    response = await chat_service.process_chat(
        ChatRequest(message="I need Compensatory Leave on Oct 13"), store=store
    )

    pending, _ = store.get_pending_action(response.session_id)
    assert response.requires_confirmation is True
    assert pending is not None
    assert pending.validated_arguments["day_type_id"] == 77
    assert pending.validated_arguments["date_from"] == "2026-10-13"


@pytest.mark.asyncio
async def test_arabic_next_friday_keeps_date_and_uses_live_arabic_actions(
    monkeypatch,
) -> None:
    rows = [
        {"dayID": 41, "dayType": "إجازة تطوع", "group": "إجازة"},
        {"dayID": 42, "dayType": "إجازة زواج", "group": "إجازة"},
        {"dayID": 43, "dayType": "مهمة عمل", "group": "مهمة عمل"},
    ]
    store = InMemorySessionStore()
    _mock_day_types(monkeypatch, rows)

    response = await chat_service.process_chat(
        ChatRequest(message="أحتاج إجازة الجمعة الجاية"),
        store=store,
    )

    draft = store.get_conversation_draft(response.session_id)
    assert response.language == "ar"
    assert draft is not None and draft.slots["date_from"] == "2026-10-09"
    assert "الجمعة، 9 أكتوبر" in response.message
    assert "تاريخ" not in response.message
    assert [action.label for action in response.blocks[-1].actions] == [
        "إجازة تطوع",
        "إجازة زواج",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "language"),
    [
        ("What is my leave balance?", "en"),
        ("وش باقي لي من الإجازات؟", "ar"),
    ],
)
async def test_leave_balance_preserves_live_result_and_adds_only_live_type_options(
    monkeypatch,
    message: str,
    language: str,
) -> None:
    rows = [
        {"dayID": 51, "dayType": "Volunteer Leave", "group": "Leave"},
        {"dayID": 52, "dayType": "Study Break", "group": "Leave"},
        {"dayID": 53, "dayType": "Field Assignment", "group": "Business Travel"},
    ]
    store = InMemorySessionStore()
    _mock_day_types(monkeypatch, rows)
    authoritative = (
        "رصيد إجازتك المستحق حاليًا هو 8 يومًا."
        if language == "ar"
        else "Your current eligible leave balance is 8 days."
    )

    async def home(*args, **kwargs):
        return {"EligibleVacation": 8}

    monkeypatch.setattr(chat_service, "get_home_data", home)
    response = await chat_service.process_chat(
        ChatRequest(message=message),
        detected_language=language,
        store=store,
    )

    assert response.message == authoritative
    assert response.tools_used == ["get_home_data", "get_day_types"]
    assert response.blocks[-1].type == "actions"
    labels = [action.label for action in response.blocks[-1].actions]
    assert labels == ["Volunteer Leave", "Study Break"]
    assert "Annual Leave" not in labels
    assert "Sick Leave" not in labels
    options = next(block for block in response.blocks if block.type == "table")
    assert options.rows == [
        {"type": "Volunteer Leave", "group": "Leave"},
        {"type": "Study Break", "group": "Leave"},
    ]
    assert all("balance" not in row for row in options.rows)


@pytest.mark.asyncio
async def test_leave_balance_options_support_direct_follow_up_without_reasking_date(
    monkeypatch,
) -> None:
    rows = [
        {"dayID": 61, "dayType": "Annual Leave", "group": "Leave"},
        {"dayID": 62, "dayType": "Personal Leave", "group": "Leave"},
    ]
    store = InMemorySessionStore()
    calls = _mock_day_types(monkeypatch, rows)

    async def home(*args, **kwargs):
        return {"EligibleVacation": 8}

    monkeypatch.setattr(chat_service, "get_home_data", home)
    balance = await chat_service.process_chat(
        ChatRequest(message="Show my leave balance"),
        store=store,
    )
    prepared = await chat_service.process_chat(
        ChatRequest(
            message="use annual leave next Friday",
            session_id=balance.session_id,
        ),
        store=store,
    )

    pending, _ = store.get_pending_action(balance.session_id)
    assert prepared.requires_confirmation is True
    assert "Which date" not in prepared.message
    assert pending is not None
    assert pending.validated_arguments["date_from"] == "2026-10-09"
    assert pending.validated_arguments["date_to"] == "2026-10-09"
    assert pending.validated_arguments["day_type_id"] == 61
    assert pending.validated_arguments["day_type_name"] == "Annual Leave"
    assert calls == ["cached", "fresh"]
