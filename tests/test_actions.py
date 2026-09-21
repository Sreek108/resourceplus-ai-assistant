import json
from datetime import date

import pytest

from app.ai import actions
from app.ai.actions import ActionIntent, execute_pending_action, prepare_write_action
from app.ai.sessions import InMemorySessionStore
from app.ai.tools import ALLOWED_TOOL_NAMES, TOOL_DEFINITIONS, DateRange, execute_tool
from app.models.schemas import ChatRequest
from app.services import chat as chat_service


@pytest.mark.asyncio
async def test_missing_punch_is_validated_and_not_posted_before_confirmation(
    monkeypatch,
) -> None:
    write_calls: list[dict] = []
    read_calls: list[tuple[str, object, object]] = []

    async def attendance(start, end, **kwargs):
        read_calls.append(("attendance", start, end))
        return {
            "Days": [
                {
                    "Date": "16/09/2026",
                    "LessHrs": "01:20",
                    "DayType": "Present",
                }
            ]
        }

    async def suggestions(start, end, **kwargs):
        read_calls.append(("suggestions", start, end))
        return [
            {
                "attDate": "16/09/2026",
                "suggestedEntryTime": "16/09/2026 17:00",
                "entryType": "OUT",
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

    monkeypatch.setattr(actions, "get_attendance_summary", attendance)
    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    monkeypatch.setattr(actions, "create_exceptional_entry", submit)

    intent = await prepare_write_action(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-16",
            "reason_name": "family circumstance",
            "remarks": "Family circumstance",
        },
        lang=1,
        response_language="en",
    )
    assert write_calls == []
    assert read_calls == [
        ("attendance", "2026-09-16", "2026-09-16"),
        ("suggestions", "2026-09-16", "2026-09-16"),
        ("reasons", None, None),
    ]
    assert intent.validated_arguments == {
        "entry_time": "2026-09-16T17:00:00",
        "entry_type": 2,
        "reason_id": "reason-family-live",
        "reason_name": "Family Circumstances",
        "remarks": "Family circumstance",
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
            "entry_time": "2026-09-16T17:00:00",
            "entry_type": 2,
            "reason_id": "reason-family-live",
            "remarks": "Family circumstance",
        }
    ]


@pytest.mark.asyncio
async def test_exceptional_entry_ignores_model_supplied_time_type_and_ids(
    monkeypatch,
) -> None:
    async def attendance(*args, **kwargs):
        return {"Days": [{"Date": "16/09/2026", "LessHrs": "00:45"}]}

    async def suggestions(*args, **kwargs):
        return [
            {
                "suggestedEntryTime": "16/09/2026 17:30",
                "entryType": "OUT",
            }
        ]

    async def reasons(*args, **kwargs):
        return [{"reasonID": "reason-live", "reasonName": "Traffic"}]

    monkeypatch.setattr(actions, "get_attendance_summary", attendance)
    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    intent = await prepare_write_action(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-16",
            "reason_name": "Traffic",
            "remarks": "Traffic",
            "entry_time": "2099-01-01T08:30:00",
            "entry_type": 1,
            "reason_id": "invented-id",
        },
        lang=1,
        response_language="en",
    )

    assert intent.validated_arguments["entry_time"] == "2026-09-16T17:30:00"
    assert intent.validated_arguments["entry_type"] == 2
    assert intent.validated_arguments["reason_id"] == "reason-live"


@pytest.mark.asyncio
async def test_exceptional_entry_without_suggestion_is_clear_and_never_prepared(
    monkeypatch,
    caplog,
) -> None:
    writes = []

    async def attendance(*args, **kwargs):
        return {"Days": [{"Date": "10/09/2026", "LessHrs": "01:20"}]}

    async def suggestions(*args, **kwargs):
        return []

    async def reasons(*args, **kwargs):
        raise AssertionError("Reasons are unnecessary when ResourcePlus has no suggestion")

    async def submit(**kwargs):
        writes.append(kwargs)

    monkeypatch.setattr(actions, "get_attendance_summary", attendance)
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
        store=store,
    )

    body = json.loads(result.output)
    assert result.failed is False
    assert result.pending_action is None
    assert store.get_pending_action(session_id)[0] is None
    assert body["prepared"] is False
    assert body["requires_clarification"] is False
    assert "doesn't currently provide a suggested punch correction" in body["message"]
    assert "selected punch time" not in body["message"]
    assert writes == []
    assert "category=no_resourceplus_suggestion" in caplog.text


@pytest.mark.asyncio
async def test_multiple_less_hours_days_require_date_selection(monkeypatch) -> None:
    async def attendance(*args, **kwargs):
        return {
            "Days": [
                {"Date": "10/09/2026", "LessHrs": "01:20"},
                {"Date": "16/09/2026", "LessHrs": "00:45"},
            ]
        }

    async def suggestions(*args, **kwargs):
        raise AssertionError("A date must be selected before requesting suggestions")

    monkeypatch.setattr(actions, "get_attendance_summary", attendance)
    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    store = InMemorySessionStore()
    session_id = store.ensure_session("multiple-less-hours")

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
    assert "10 September 2026 — 1h 20m" in body["message"]
    assert "16 September 2026 — 45m" in body["message"]
    assert "Which date" in body["message"]


@pytest.mark.asyncio
async def test_non_less_hours_date_never_requests_a_suggestion(monkeypatch) -> None:
    async def attendance(*args, **kwargs):
        return {"Days": [{"Date": "10/09/2026", "LessHrs": "00:00"}]}

    async def suggestions(*args, **kwargs):
        raise AssertionError("An inapplicable attendance day must stop before suggestions")

    monkeypatch.setattr(actions, "get_attendance_summary", attendance)
    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    store = InMemorySessionStore()
    session_id = store.ensure_session("not-applicable")

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
        store=store,
    )

    body = json.loads(result.output)
    assert result.pending_action is None
    assert "couldn't find an applicable less-hours record" in body["message"]
    assert store.get_pending_action(session_id)[0] is None


@pytest.mark.asyncio
async def test_ambiguous_live_reason_asks_user_without_exposing_ids(monkeypatch) -> None:
    async def attendance(*args, **kwargs):
        return {"Days": [{"Date": "10/09/2026", "LessHrs": "01:20"}]}

    async def suggestions(*args, **kwargs):
        return [{"suggestedEntryTime": "10/09/2026 17:00", "entryType": "OUT"}]

    async def reasons(*args, **kwargs):
        return [
            {"reasonID": "secret-one", "reasonName": "Family Circumstances"},
            {"reasonID": "secret-two", "reasonName": "Family Emergency"},
        ]

    async def ambiguous(*args, **kwargs):
        return None

    monkeypatch.setattr(actions, "get_attendance_summary", attendance)
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
    async def attendance(*args, **kwargs):
        return {"Days": [{"Date": "10/09/2026", "LessHrs": "01:20"}]}

    async def suggestions(*args, **kwargs):
        return [
            {
                "suggestedEntryTime": "10/09/2026 17:00",
                "entryType": "OUT",
            }
        ]

    async def reasons(*args, **kwargs):
        return [{"reasonID": "family-live-id", "reasonName": "Family Circumstances"}]

    monkeypatch.setattr(actions, "get_attendance_summary", attendance)
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
                "entry_time": "2026-09-10T17:00:00",
                "entry_type": 2,
                "reason_id": "family-live-id",
                "reason_name": "Family Circumstances",
                "remarks": "Family circumstance",
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
            "entry_time": "2026-09-10T17:00:00",
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
        "reason_name",
        "remarks",
    }
    assert "entry_time" not in exceptional["parameters"]["properties"]
    assert "entry_type" not in exceptional["parameters"]["properties"]

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
