import json
from datetime import date, datetime, timedelta, timezone

import pytest

from app.ai import actions, tools as ai_tools
from app.ai.actions import ActionIntent, execute_pending_action, prepare_write_action
from app.ai.sessions import InMemorySessionStore
from app.ai.tools import ALLOWED_TOOL_NAMES, TOOL_DEFINITIONS, DateRange, execute_tool
from app.identity import (
    RequestIdentity,
    bind_request_identity,
    current_request_identity,
    reset_request_identity,
)
from app.models.schemas import ChatRequest
from app.services import chat as chat_service


@pytest.mark.asyncio
async def test_attendance_tool_explains_net_and_less_hours_for_speech(monkeypatch) -> None:
    async def attendance(*args, **kwargs):
        return {
            "Days": [
                {
                    "AttDate": "22/09/2026",
                    "NetHrs": "00:40",
                    "LessHrs": "07:20",
                }
            ]
        }

    monkeypatch.setattr(ai_tools, "get_attendance_summary", attendance)
    result = await execute_tool(
        "get_attendance_summary",
        {"from_date": "2026-09-22", "to_date": "2026-09-22"},
        lang=1,
        session_id="attendance-semantics",
        response_language="en",
    )

    payload = json.loads(result.output)
    assert payload["data"]["Days"][0]["NetHrs"] == "00:40"
    assert payload["data"]["Days"][0]["LessHrs"] == "07:20"
    assert payload["attendance_semantics"] == {
        "NetHrs": "time actually worked",
        "LessHrs": "shortfall from required working hours",
        "speech_rule": (
            "State actual worked time and the shortfall as separate quantities; "
            "never describe NetHrs as the amount worked less than expected."
        ),
    }


@pytest.mark.asyncio
async def test_missing_punch_is_validated_and_not_posted_before_confirmation(
    monkeypatch,
) -> None:
    write_calls: list[dict] = []
    read_calls: list[tuple[str, object, object]] = []

    async def suggestions(start, end, **kwargs):
        read_calls.append(("suggestions", start, end))
        return [
            {
                "attDate": "16/09/2026",
                "suggestedEntryTime": "16/09/2026 17:00",
                "entryType": "OUT",
                "shift": "General Shift (09:00 : 18:00)",
                "isNightShift": 0,
            }
        ]

    async def reasons(*args, **kwargs):
        read_calls.append(("reasons", None, None))
        return [
            {"reasonID": "reason-family-live", "reasonName": "Family Circumstances"},
            {"reasonID": "reason-other-live", "reasonName": "Other"},
        ]

    async def submit(**kwargs):
        write_calls.append(kwargs)
        return {"success": True, "message": "Exceptional entry submitted"}

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    monkeypatch.setattr(actions, "create_exceptional_entry", submit)

    intent = await prepare_write_action(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-16",
            "reason_name": "family circumstance",
            "remarks": "Family circumstance",
            "_user_message": "Family circumstance",
        },
        lang=1,
        response_language="en",
    )
    assert write_calls == []
    assert read_calls == [
        ("suggestions", "2026-09-16", "2026-09-16"),
        ("reasons", None, None),
    ]
    assert intent.validated_arguments == {
        "entry_time": "16/09/2026 17:00",
        "entry_type": 2,
        "reason_id": "reason-family-live",
        "reason_name": "Family Circumstances",
        "remarks": "Family circumstance",
        "attendance_date": "2026-09-16",
        "shift": "General Shift (09:00 : 18:00)",
        "is_night_shift": 0,
    }
    assert "OUT punch" in intent.summary
    assert "5:00 PM" in intent.summary
    assert "Family Circumstances" in intent.summary

    result = await execute_pending_action(
        intent.action_type,
        intent.validated_arguments,
    )
    assert result["success"] is True
    assert write_calls == [
        {
            "entry_time": "16/09/2026 17:00",
            "entry_type": 2,
            "reason_id": "reason-family-live",
            "remarks": "Family circumstance",
        }
    ]


@pytest.mark.asyncio
async def test_exceptional_entry_ignores_model_supplied_time_type_and_ids(
    monkeypatch,
) -> None:
    async def suggestions(*args, **kwargs):
        return [
            {
                "attDate": "16/09/2026",
                "suggestedEntryTime": "16/09/2026 17:30",
                "entryType": "OUT",
            }
        ]

    async def reasons(*args, **kwargs):
        return [{"reasonID": "reason-live", "reasonName": "Traffic"}]

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    intent = await prepare_write_action(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-16",
            "reason_name": "Traffic",
            "remarks": "Traffic",
            "_user_message": "Traffic",
            "entry_time": "2099-01-01T08:30:00",
            "entry_type": 1,
            "reason_id": "invented-id",
        },
        lang=1,
        response_language="en",
    )

    assert intent.validated_arguments["entry_time"] == "16/09/2026 17:30"
    assert intent.validated_arguments["entry_type"] == 2
    assert intent.validated_arguments["reason_id"] == "reason-live"


@pytest.mark.asyncio
async def test_exceptional_entry_without_suggestion_is_clear_and_never_prepared(
    monkeypatch,
    caplog,
) -> None:
    writes = []

    async def suggestions(*args, **kwargs):
        return []

    async def reasons(*args, **kwargs):
        raise AssertionError("Reasons are unnecessary when ResourcePlus has no suggestion")

    async def submit(**kwargs):
        writes.append(kwargs)

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    monkeypatch.setattr(actions, "create_exceptional_entry", submit)
    caplog.set_level("INFO", logger="app.ai.tools")
    store = InMemorySessionStore()
    session_id = store.ensure_session("no-suggestion")

    result = await execute_tool(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-10",
            "reason_name": "Family circumstance",
            "remarks": "Family circumstance",
        },
        lang=1,
        session_id=session_id,
        response_language="en",
        source_user_message="Traffic caused the missing punch",
        store=store,
    )

    body = json.loads(result.output)
    assert result.failed is False
    assert result.pending_action is None
    assert store.get_pending_action(session_id)[0] is None
    assert body["prepared"] is False
    assert body["requires_clarification"] is False
    assert "does not currently provide a valid suggested punch correction" in body[
        "message"
    ]
    assert "selected punch time" not in body["message"]
    assert writes == []
    assert "category=no_resourceplus_suggestion" in caplog.text


@pytest.mark.asyncio
async def test_multiple_suggestion_dates_require_date_selection(monkeypatch) -> None:
    async def suggestions(*args, **kwargs):
        return [
            {
                "attDate": "10/09/2026",
                "suggestedEntryTime": "10/09/2026 09:00",
                "entryType": "IN",
            },
            {
                "attDate": "16/09/2026",
                "suggestedEntryTime": "16/09/2026 17:00",
                "entryType": "OUT",
            },
        ]

    async def reasons(*args, **kwargs):
        raise AssertionError("Reasons require a clearly selected suggestion")

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    store = InMemorySessionStore()
    session_id = store.ensure_session("multiple-suggestions")

    result = await execute_tool(
        "prepare_exceptional_entry",
        {
            "target_date": None,
            "reason_name": "Family circumstance",
            "remarks": "Family circumstance",
        },
        lang=1,
        session_id=session_id,
        response_language="en",
        resolved_range=DateRange(
            "this_month",
            date(2026, 9, 1),
            date(2026, 9, 21),
        ),
        store=store,
    )

    body = json.loads(result.output)
    assert result.pending_action is None
    assert store.get_pending_action(session_id)[0] is None
    assert body["requires_clarification"] is True
    assert "10 September 2026" in body["message"]
    assert "16 September 2026" in body["message"]
    assert "Which date" in body["message"]


@pytest.mark.asyncio
async def test_live_suggestion_is_authoritative_without_attendance_prefilter(
    monkeypatch,
) -> None:
    async def suggestions(*args, **kwargs):
        return [
            {
                "attDate": "10/09/2026",
                "suggestedEntryTime": "10/09/2026 09:07",
                "entryType": "IN",
                "shift": "Night Shift",
                "isNightShift": 1,
            }
        ]

    async def reasons(*args, **kwargs):
        return [{"reasonID": "traffic-live", "reasonName": "Traffic Delay"}]

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    store = InMemorySessionStore()
    session_id = store.ensure_session("authoritative-suggestion")

    result = await execute_tool(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-10",
            "reason_name": "traffic",
            "remarks": "Traffic delay",
        },
        lang=1,
        session_id=session_id,
        response_language="en",
        source_user_message="Correct my missing punch because of traffic",
        store=store,
    )

    assert result.pending_action is not None
    pending, expired = store.get_pending_action(session_id)
    assert expired is False
    assert pending is not None
    assert pending.validated_arguments == {
        "entry_time": "10/09/2026 09:07",
        "entry_type": 1,
        "reason_id": "traffic-live",
        "reason_name": "Traffic Delay",
        "remarks": "Traffic delay",
        "attendance_date": "2026-09-10",
        "shift": "Night Shift",
        "is_night_shift": 1,
    }


@pytest.mark.asyncio
async def test_selected_date_uses_only_its_resourceplus_suggestion(monkeypatch) -> None:
    async def suggestions(*args, **kwargs):
        return [
            {
                "attDate": "09/09/2026",
                "suggestedEntryTime": "09/09/2026 09:15",
                "entryType": "IN",
            },
            {
                "attDate": "10/09/2026",
                "suggestedEntryTime": "10/09/2026 17:11",
                "entryType": "OUT",
            },
        ]

    async def reasons(*args, **kwargs):
        return [{"reasonID": "client-live", "reasonName": "Client Meeting"}]

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)

    intent = await prepare_write_action(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-10",
            "reason_name": "client meeting",
            "remarks": "Client meeting",
            "_user_message": "Client meeting",
        },
        lang=1,
        response_language="en",
    )

    assert intent.validated_arguments["attendance_date"] == "2026-09-10"
    assert intent.validated_arguments["entry_time"] == "10/09/2026 17:11"
    assert intent.validated_arguments["entry_type"] == 2
    assert intent.validated_arguments["reason_id"] == "client-live"


@pytest.mark.asyncio
async def test_unknown_live_reason_requires_clarification_without_pending_action(
    monkeypatch,
) -> None:
    async def suggestions(*args, **kwargs):
        return [
            {
                "attDate": "10/09/2026",
                "suggestedEntryTime": "10/09/2026 17:00",
                "entryType": "OUT",
            }
        ]

    async def reasons(*args, **kwargs):
        return [{"reasonID": "traffic-secret", "reasonName": "Traffic Delay"}]

    async def no_match(*args, **kwargs):
        return None

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    monkeypatch.setattr(actions, "match_live_reason", no_match)
    store = InMemorySessionStore()
    session_id = store.ensure_session("unknown-reason")

    result = await execute_tool(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-10",
            "reason_name": "unsupported reason",
            "remarks": "Unsupported reason",
        },
        lang=1,
        session_id=session_id,
        response_language="en",
        source_user_message=(
            "Correct my missing punch because of an unsupported reason"
        ),
        store=store,
    )

    body = json.loads(result.output)
    assert result.pending_action is None
    assert store.get_pending_action(session_id)[0] is None
    assert body["requires_clarification"] is True
    assert "Traffic Delay" in body["message"]
    assert "traffic-secret" not in result.output


@pytest.mark.asyncio
async def test_ambiguous_live_reason_asks_user_without_exposing_ids(monkeypatch) -> None:
    async def suggestions(*args, **kwargs):
        return [
            {
                "attDate": "10/09/2026",
                "suggestedEntryTime": "10/09/2026 17:00",
                "entryType": "OUT",
            }
        ]

    async def reasons(*args, **kwargs):
        return [
            {"reasonID": "secret-one", "reasonName": "Family Circumstances"},
            {"reasonID": "secret-two", "reasonName": "Family Emergency"},
        ]

    async def ambiguous(*args, **kwargs):
        return None

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    monkeypatch.setattr(actions, "match_live_reason", ambiguous)
    store = InMemorySessionStore()
    session_id = store.ensure_session("ambiguous-reason")

    result = await execute_tool(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-10",
            "reason_name": "family issue",
            "remarks": "Family issue",
        },
        lang=1,
        session_id=session_id,
        response_language="en",
        source_user_message="Correct my missing punch because of a family issue",
        store=store,
    )

    body = json.loads(result.output)
    assert result.pending_action is None
    assert body["requires_clarification"] is True
    assert "Which reason should I use?" in body["message"]
    assert "Family Circumstances" in body["message"]
    assert "secret-one" not in result.output
    assert "secret-two" not in result.output


@pytest.mark.asyncio
async def test_exceptional_entry_yes_executes_exact_pending_arguments(
    monkeypatch,
) -> None:
    async def suggestions(*args, **kwargs):
        return [
            {
                "attDate": "10/09/2026",
                "suggestedEntryTime": "10/09/2026 17:00",
                "entryType": "OUT",
            }
        ]

    async def reasons(*args, **kwargs):
        return [{"reasonID": "family-live-id", "reasonName": "Family Circumstances"}]

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    store = InMemorySessionStore()
    session_id = store.ensure_session("exceptional-confirm")
    prepared = await execute_tool(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-10",
            "reason_name": "family circumstance",
            "remarks": "Family circumstance",
        },
        lang=1,
        session_id=session_id,
        response_language="en",
        source_user_message=(
            "Correct my missing punch because of a family circumstance"
        ),
        store=store,
    )
    pending = prepared.pending_action
    assert pending is not None

    executions = []

    async def execute(action_type, arguments):
        executions.append((action_type, arguments))
        return {"success": True, "message": "Request submitted for approval"}

    async def render(facts, **kwargs):
        return facts

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    monkeypatch.setattr(chat_service, "render_user_message", render)
    response = await chat_service.process_chat(
        ChatRequest(
            message="Yes",
            session_id=session_id,
            confirmation_id=pending.confirmation_id,
        ),
        detected_language="en",
        store=store,
    )

    assert response.success is True
    assert executions == [
        (
            "create_exceptional_entry",
            {
                "entry_time": "10/09/2026 17:00",
                "entry_type": 2,
                "reason_id": "family-live-id",
                "reason_name": "Family Circumstances",
                "remarks": "Family circumstance",
                "attendance_date": "2026-09-10",
                "shift": None,
                "is_night_shift": None,
            },
        )
    ]


@pytest.mark.asyncio
async def test_exceptional_entry_no_discards_without_write(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session("exceptional-reject")
    pending = store.create_pending_action(
        session_id,
        action_type="create_exceptional_entry",
        validated_arguments={
            "entry_time": "10/09/2026 17:00",
            "entry_type": 2,
            "reason_id": "family-live-id",
            "reason_name": "Family Circumstances",
            "remarks": "Family circumstance",
        },
        summary="Confirm exceptional entry",
        language="en",
    )

    async def execute(*args, **kwargs):
        raise AssertionError("No must never execute the ResourcePlus write")

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    response = await chat_service.process_chat(
        ChatRequest(
            message="No",
            session_id=session_id,
            confirmation_id=pending.confirmation_id,
        ),
        detected_language="en",
        store=store,
    )

    assert response.success is True
    assert response.language == "en"
    assert store.get_pending_action(session_id)[0] is None


@pytest.mark.asyncio
async def test_expired_exceptional_entry_never_posts(monkeypatch) -> None:
    now = [datetime(2026, 9, 22, tzinfo=timezone.utc)]
    store = InMemorySessionStore(
        confirmation_ttl_seconds=1,
        now=lambda: now[0],
    )
    session_id = store.ensure_session("exceptional-expired")
    store.create_pending_action(
        session_id,
        action_type="create_exceptional_entry",
        validated_arguments={
            "entry_time": "10/09/2026 17:00",
            "entry_type": 2,
            "reason_id": "family-live-id",
            "reason_name": "Family Circumstances",
            "remarks": "Family circumstance",
            "attendance_date": "10/09/2026",
            "shift": None,
            "is_night_shift": 0,
        },
        summary="Confirm exceptional entry",
        language="en",
    )
    now[0] += timedelta(seconds=2)

    async def execute(*args, **kwargs):
        raise AssertionError("An expired exceptional entry must never be posted")

    async def classify(*args, **kwargs):
        raise AssertionError("An expired action must not be classified")

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    monkeypatch.setattr(chat_service, "classify_confirmation_intent", classify)
    response = await chat_service.process_chat(
        ChatRequest(message="Yes", session_id=session_id),
        detected_language="en",
        store=store,
    )

    assert response.success is False
    assert response.requires_confirmation is False
    assert "expired" in response.message.lower()


@pytest.mark.asyncio
async def test_leave_resolves_real_day_type_and_surfaces_conflict(monkeypatch) -> None:
    writes: list[dict] = []

    async def day_types(*args, **kwargs):
        return [{"dayID": 20, "dayType": "Annual Leave", "group": "Leave"}]

    async def book(**kwargs):
        writes.append(kwargs)
        return {
            "success": False,
            "message": "A request already exists for an overlapping date range",
        }

    monkeypatch.setattr(actions, "get_day_types", day_types)
    monkeypatch.setattr(actions, "book_day_type", book)
    intent = await prepare_write_action(
        "prepare_book_day_type",
        {
            "date_from": "2026-09-22",
            "date_to": "2026-09-24",
            "day_type_name": "annual leave",
        },
        lang=1,
        response_language="en",
    )
    assert writes == []
    assert intent.validated_arguments["day_type_id"] == 20
    result = await execute_pending_action(intent.action_type, intent.validated_arguments)
    assert result == {
        "success": False,
        "message": "A request already exists for an overlapping date range",
    }
    assert writes[0]["day_type_id"] == 20


@pytest.mark.asyncio
async def test_cancellation_requires_real_pending_request_and_mapping_id(monkeypatch) -> None:
    writes: list[str] = []

    async def my_requests(*args, **kwargs):
        return [
            {
                "mappingID": "mapping-real",
                "dayType": "Annual Leave",
                "dateFrom": "22/09/2026",
                "dateTo": "24/09/2026",
                "status": "Pending",
            }
        ]

    async def cancel(mapping_id):
        writes.append(mapping_id)
        return {"success": True, "message": "Request cancelled"}

    monkeypatch.setattr(actions, "get_my_day_type_requests", my_requests)
    monkeypatch.setattr(actions, "cancel_day_type_request", cancel)
    intent = await prepare_write_action(
        "prepare_cancel_day_type_request",
        {
            "day_type_name": "Annual Leave",
            "date_from": "2026-09-22",
            "date_to": "2026-09-24",
            "mapping_id": "model-invented",
        },
        lang=1,
        response_language="en",
    )
    assert writes == []
    assert intent.validated_arguments["mapping_id"] == "mapping-real"
    await execute_pending_action(intent.action_type, intent.validated_arguments)
    assert writes == ["mapping-real"]


@pytest.mark.asyncio
async def test_cancel_matches_live_resourceplus_dd_mm_yyyy_date(monkeypatch) -> None:
    async def my_requests(*args, **kwargs):
        return [
            {
                "mappingID": "mapping-live-format",
                "dayType": "Business Travel",
                "dateFrom": "22-09-2026",
                "dateTo": "22-09-2026",
                "status": "Pending",
            }
        ]

    monkeypatch.setattr(actions, "get_my_day_type_requests", my_requests)
    intent = await prepare_write_action(
        "prepare_cancel_day_type_request",
        {
            "day_type_name": "business travel",
            "date_from": "2026-09-22",
            "date_to": None,
        },
        lang=1,
        response_language="en",
    )

    assert intent.validated_arguments["mapping_id"] == "mapping-live-format"


@pytest.mark.asyncio
async def test_cancel_matches_single_day_with_equal_start_and_end(monkeypatch) -> None:
    async def my_requests(*args, **kwargs):
        return [
            {
                "mappingID": "mapping-single-day",
                "dayType": "Business Travel",
                "dateFrom": "22/09/2026",
                "dateTo": "22/09/2026",
                "status": "Pending",
            }
        ]

    monkeypatch.setattr(actions, "get_my_day_type_requests", my_requests)
    intent = await prepare_write_action(
        "prepare_cancel_day_type_request",
        {
            "day_type_name": "Business Travel",
            "date_from": "2026-09-22",
            "date_to": "2026-09-22",
        },
        lang=1,
        response_language="en",
    )

    assert intent.validated_arguments["mapping_id"] == "mapping-single-day"


@pytest.mark.asyncio
async def test_cancel_matches_requested_date_inside_multi_day_request(monkeypatch) -> None:
    async def my_requests(*args, **kwargs):
        return [
            {
                "mappingID": "mapping-multi-day",
                "dayType": "Business Travel",
                "dateFrom": "20-09-2026",
                "dateTo": "24-09-2026",
                "status": "Pending",
            }
        ]

    monkeypatch.setattr(actions, "get_my_day_type_requests", my_requests)
    intent = await prepare_write_action(
        "prepare_cancel_day_type_request",
        {
            "day_type_name": "Business Travel",
            "date_from": "2026-09-22",
            "date_to": None,
        },
        lang=1,
        response_language="en",
    )

    assert intent.validated_arguments["mapping_id"] == "mapping-multi-day"


@pytest.mark.asyncio
async def test_cancel_treats_blank_resourceplus_end_as_single_day(monkeypatch) -> None:
    async def my_requests(*args, **kwargs):
        return [
            {
                "mappingID": "mapping-blank-end",
                "dayType": "Business Travel",
                "dateFrom": "22-09-2026",
                "dateTo": None,
                "status": "Pending",
            }
        ]

    monkeypatch.setattr(actions, "get_my_day_type_requests", my_requests)
    intent = await prepare_write_action(
        "prepare_cancel_day_type_request",
        {
            "day_type_name": "Business Travel",
            "date_from": "2026-09-22",
            "date_to": None,
        },
        lang=1,
        response_language="en",
    )

    assert intent.validated_arguments["mapping_id"] == "mapping-blank-end"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "record_override",
    [
        {"dayType": "Annual Leave"},
        {"status": "Approved"},
    ],
)
async def test_cancel_rejects_wrong_day_type_or_non_pending_request(
    monkeypatch,
    record_override: dict[str, str],
) -> None:
    record = {
        "mappingID": "mapping-ineligible",
        "dayType": "Business Travel",
        "dateFrom": "22-09-2026",
        "dateTo": "22-09-2026",
        "status": "Pending",
    }
    record.update(record_override)

    async def my_requests(*args, **kwargs):
        return [record]

    monkeypatch.setattr(actions, "get_my_day_type_requests", my_requests)
    with pytest.raises(ValueError, match="No matching pending absence request"):
        await prepare_write_action(
            "prepare_cancel_day_type_request",
            {
                "day_type_name": "Business Travel",
                "date_from": "2026-09-22",
                "date_to": None,
            },
            lang=1,
            response_language="en",
        )


@pytest.mark.asyncio
async def test_cancel_rejects_ambiguous_pending_requests(monkeypatch) -> None:
    async def my_requests(*args, **kwargs):
        return [
            {
                "mappingID": mapping_id,
                "dayType": "Business Travel",
                "dateFrom": "22-09-2026",
                "dateTo": "22-09-2026",
                "status": "Pending",
            }
            for mapping_id in ("mapping-one", "mapping-two")
        ]

    monkeypatch.setattr(actions, "get_my_day_type_requests", my_requests)
    with pytest.raises(ValueError, match="More than one matching pending"):
        await prepare_write_action(
            "prepare_cancel_day_type_request",
            {
                "day_type_name": "Business Travel",
                "date_from": "2026-09-22",
                "date_to": None,
            },
            lang=1,
            response_language="en",
        )


@pytest.mark.asyncio
async def test_cancel_prepare_stores_resourceplus_id_without_posting(monkeypatch) -> None:
    writes: list[str] = []

    async def my_requests(*args, **kwargs):
        return [
            {
                "mappingID": "mapping-from-resourceplus",
                "dayType": "Business Travel",
                "dateFrom": "22-09-2026",
                "dateTo": "22-09-2026",
                "status": "Pending",
            }
        ]

    async def cancel(mapping_id):
        writes.append(mapping_id)
        return {"success": True}

    monkeypatch.setattr(actions, "get_my_day_type_requests", my_requests)
    monkeypatch.setattr(actions, "cancel_day_type_request", cancel)
    store = InMemorySessionStore()
    session_id = store.ensure_session("cancel-prepare-test")
    result = await execute_tool(
        "prepare_cancel_day_type_request",
        {
            "day_type_name": "Business Travel",
            "date_from": "2026-09-22",
            "date_to": None,
            "mapping_id": "model-invented",
        },
        lang=1,
        session_id=session_id,
        response_language="en",
        store=store,
    )

    pending, expired = store.get_pending_action(session_id)
    assert result.failed is False
    assert result.pending_action is not None
    assert pending is not None
    assert expired is False
    assert pending.validated_arguments["mapping_id"] == "mapping-from-resourceplus"
    assert writes == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "expected_status"),
    [("approve", 1), ("reject", 2)],
)
async def test_individual_supervisor_action_preserves_request_type(
    monkeypatch,
    decision: str,
    expected_status: int,
) -> None:
    writes: list[dict] = []

    async def pending(*args, **kwargs):
        return [
            {
                "requestId": "request-real",
                "requestType": "ExceptionEntry",
                "employeeName": "Reem",
                "detail": "Missing punch",
            }
        ]

    async def approve(**kwargs):
        writes.append(kwargs)
        return {"success": True, "message": "Updated"}

    monkeypatch.setattr(actions, "get_pending_approvals", pending)
    monkeypatch.setattr(actions, "approve_supervisor_request", approve)
    intent = await prepare_write_action(
        "prepare_supervisor_request",
        {
            "employee_name": "Reem",
            "detail": "Missing punch",
            "decision": decision,
            "request_id": "invented",
            "request_type": "Absence",
        },
        lang=1,
        response_language="en",
    )
    assert writes == []
    assert intent.validated_arguments["request_id"] == "request-real"
    assert intent.validated_arguments["request_type"] == "ExceptionEntry"
    await execute_pending_action(intent.action_type, intent.validated_arguments)
    assert writes[0]["request_type"] == "ExceptionEntry"
    assert writes[0]["status"] == expected_status


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("verification_rows", "expected_result", "expected_message"),
    [
        ([], "approved", "Done — Talal Sabbagh's Work From Home request has been approved."),
        (
            [
                {
                    "requestId": "request-real",
                    "requestType": "ExceptionEntry",
                    "employeeName": "Talal Sabbagh",
                    "detail": "Work From Home",
                }
            ],
            "approval_pending_verification",
            (
                "The approval was accepted, but the request is still showing as "
                "pending. I won't submit it again."
            ),
        ),
    ],
)
async def test_supervisor_approval_is_verified_once_with_bound_identity(
    monkeypatch,
    verification_rows,
    expected_result,
    expected_message,
) -> None:
    identity = RequestIdentity("hana.haddad@example.com", "portalv21")
    pending_reads: list[RequestIdentity] = []
    writes: list[tuple[dict, RequestIdentity]] = []
    action_states: list[dict] = []

    async def pending(*args, **kwargs):
        pending_reads.append(current_request_identity())
        return verification_rows

    async def approve(**kwargs):
        writes.append((kwargs, current_request_identity()))
        return {"success": True, "message": "Updated"}

    monkeypatch.setattr(actions, "get_pending_approvals", pending)
    monkeypatch.setattr(actions, "approve_supervisor_request", approve)
    monkeypatch.setattr(
        chat_service,
        "record_action_state",
        lambda **kwargs: action_states.append(kwargs),
    )

    store = InMemorySessionStore()
    token = bind_request_identity(identity)
    try:
        session_id = store.ensure_session("verified-supervisor-approval")
        action = store.create_pending_action(
            session_id,
            action_type="approve_supervisor_request",
            validated_arguments={
                "request_id": "request-real",
                "request_type": "ExceptionEntry",
                "status": 1,
                "employee_name": "Talal Sabbagh",
                "detail": "Work From Home",
            },
            summary="Approve Talal's Work From Home request?",
            language="en",
        )
    finally:
        reset_request_identity(token)

    response = await chat_service.process_chat(
        ChatRequest(
            message="Yes",
            session_id=session_id,
            confirmation_id=action.confirmation_id,
            email=identity.email,
            instance=identity.instance,
        ),
        store=store,
    )

    assert response.success is True
    assert response.message == expected_message
    assert len(writes) == 1
    assert writes[0][0] == {
        "request_id": "request-real",
        "request_type": "ExceptionEntry",
        "status": 1,
    }
    assert writes[0][1] == identity
    assert pending_reads == [identity]
    assert action_states[-1]["state"] == "executed"
    assert action_states[-1]["result"] == expected_result


@pytest.mark.asyncio
async def test_supervisor_approval_identity_mismatch_does_not_write_or_verify(
    monkeypatch,
) -> None:
    writes: list[dict] = []
    reads: list[bool] = []

    async def approve(**kwargs):
        writes.append(kwargs)
        return {"success": True}

    async def pending(*args, **kwargs):
        reads.append(True)
        return []

    monkeypatch.setattr(actions, "approve_supervisor_request", approve)
    monkeypatch.setattr(actions, "get_pending_approvals", pending)
    store = InMemorySessionStore()
    owner = RequestIdentity("hana.haddad@example.com", "portalv21")
    token = bind_request_identity(owner)
    try:
        session_id = store.ensure_session("approval-owner-mismatch")
        action = store.create_pending_action(
            session_id,
            action_type="approve_supervisor_request",
            validated_arguments={
                "request_id": "request-real",
                "request_type": "ExceptionEntry",
                "status": 1,
            },
            summary="Approve request?",
            language="en",
        )
    finally:
        reset_request_identity(token)

    response = await chat_service.process_chat(
        ChatRequest(
            message="Yes",
            session_id=session_id,
            confirmation_id=action.confirmation_id,
            email=owner.email,
            instance="Universal",
        ),
        store=store,
    )

    assert response.success is False
    assert writes == []
    assert reads == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("decision", "expected_status"),
    [("approve", 1), ("reject", 2)],
)
async def test_bulk_supervisor_action_counts_scope(
    monkeypatch,
    decision: str,
    expected_status: int,
) -> None:
    writes: list[dict] = []

    async def pending(*args, **kwargs):
        return [
            {"requestId": "1", "requestType": "Absence"},
            {"requestId": "2", "requestType": "Absence"},
            {"requestId": "3", "requestType": "ExceptionEntry"},
        ]

    async def approve_all(**kwargs):
        writes.append(kwargs)
        return {"success": True, "message": "Updated"}

    monkeypatch.setattr(actions, "get_pending_approvals", pending)
    monkeypatch.setattr(actions, "approve_all_requests", approve_all)
    intent = await prepare_write_action(
        "prepare_approve_all_requests",
        {"decision": decision, "request_type": "Absence"},
        lang=1,
        response_language="en",
    )
    assert writes == []
    assert intent.validated_arguments["count"] == 2
    assert "2" in intent.summary
    await execute_pending_action(intent.action_type, intent.validated_arguments)
    assert writes == [{"status": expected_status, "request_type": "Absence"}]


def test_openai_tools_cannot_choose_identity_or_resourceplus_ids() -> None:
    forbidden_properties = {
        "usrEmail",
        "usr_email",
        "manager_email",
        "reasonID",
        "reason_id",
        "dayTypeID",
        "day_type_id",
        "mappingID",
        "mapping_id",
        "requestId",
        "request_id",
        "requestType",
        "request_type_id",
    }
    for tool in TOOL_DEFINITIONS:
        properties = set(tool["parameters"].get("properties", {}))
        assert properties.isdisjoint(forbidden_properties)

    exceptional = next(
        tool for tool in TOOL_DEFINITIONS if tool["name"] == "prepare_exceptional_entry"
    )
    assert set(exceptional["parameters"]["properties"]) == {
        "target_date",
        "punch_direction",
        "reason_name",
        "remarks",
    }
    assert "entry_time" not in exceptional["parameters"]["properties"]
    assert "entry_type" not in exceptional["parameters"]["properties"]
    assert exceptional["parameters"]["properties"]["reason_name"]["type"] == [
        "string",
        "null",
    ]

    assert "create_exceptional_entry" not in ALLOWED_TOOL_NAMES
    assert "book_day_type" not in ALLOWED_TOOL_NAMES
    assert "cancel_day_type_request" not in ALLOWED_TOOL_NAMES
    assert "approve_supervisor_request" not in ALLOWED_TOOL_NAMES
    assert "approve_all_requests" not in ALLOWED_TOOL_NAMES


@pytest.mark.asyncio
async def test_write_intent_tool_creates_pending_confirmation(monkeypatch) -> None:
    async def prepared(*args, **kwargs):
        return ActionIntent(
            action_type="book_day_type",
            validated_arguments={
                "date_from": "2026-09-22",
                "date_to": "2026-09-24",
                "day_type_id": 20,
                "day_type_name": "Annual Leave",
            },
            summary="Confirm Annual Leave",
            language="en",
        )

    monkeypatch.setattr("app.ai.tools.prepare_write_action", prepared)
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    result = await execute_tool(
        "prepare_book_day_type",
        {
            "date_from": "2026-09-22",
            "date_to": "2026-09-24",
            "day_type_name": "Annual Leave",
        },
        lang=1,
        session_id=session_id,
        response_language="en",
        store=store,
    )
    pending, expired = store.get_pending_action(session_id)
    assert expired is False
    assert result.pending_action is not None
    assert pending is not None
    assert pending.validated_arguments["day_type_id"] == 20


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        (
            "prepare_exceptional_entry",
            {
                "target_date": "2026-09-29",
                "punch_direction": "OUT",
                "reason_name": None,
                "remarks": None,
            },
        ),
        (
            "prepare_book_day_type",
            {
                "date_from": "2026-09-29",
                "date_to": "2026-09-29",
                "day_type_name": "Annual Leave",
            },
        ),
        (
            "prepare_cancel_day_type_request",
            {
                "day_type_name": "Annual Leave",
                "date_from": "2026-09-29",
                "date_to": None,
            },
        ),
        (
            "prepare_supervisor_request",
            {"employee_name": "Employee", "detail": None, "decision": "approve"},
        ),
        (
            "prepare_approve_all_requests",
            {"decision": "approve", "request_type": "all"},
        ),
        (
            "prepare_notification_read_status",
            {"target": "all", "notification_title": None, "read_status": 1},
        ),
    ],
)
async def test_ambiguous_source_cannot_start_any_write_preparation(
    monkeypatch,
    tool_name: str,
    arguments: dict,
) -> None:
    async def forbidden(*args, **kwargs):
        raise AssertionError("An ungrounded model guess must not prepare a write")

    monkeypatch.setattr(ai_tools, "prepare_write_action", forbidden)
    store = InMemorySessionStore()
    session_id = store.ensure_session(f"ungrounded-{tool_name}")

    result = await execute_tool(
        tool_name,
        arguments,
        lang=1,
        session_id=session_id,
        response_language="en",
        source_user_message="I forgot to Punjab Today.",
        store=store,
    )

    payload = json.loads(result.output)
    assert payload["prepared"] is False
    assert payload["requires_confirmation"] is False
    assert result.tool_used is False
    assert result.pending_action is None
    assert store.get_pending_action(session_id)[0] is None
