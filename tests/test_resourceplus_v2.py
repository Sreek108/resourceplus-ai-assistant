from __future__ import annotations

from datetime import date

import httpx
import pytest

from app.ai import actions
from app.ai import conversation
from app.ai.actions import (
    ActionResolutionRequired,
    execute_pending_action,
    find_pending_exceptional_entries,
    inspect_less_hours_period,
    prepare_write_action,
)
from app.ai.conversation import (
    _explicit_entry_type,
    _explicit_minutes,
    _is_cancel_exception_request,
    is_less_hours_correction_request,
    is_less_hours_request,
)
from app.ai.sessions import InMemorySessionStore
from app.audit import SAFE_ACTION_TYPES
from app.identity import RequestIdentity, bind_request_identity, reset_request_identity
from app.models.schemas import ChatRequest
from app.resourceplus.client import ResourcePlusClient
from app.resourceplus.exceptional import (
    cancel_exceptional_entry,
    create_exceptional_entry_from_summary,
    get_exceptional_entry_balance,
)
from app.services import chat as chat_service
from app.services.response_blocks import (
    cancellable_exceptions_block,
    exceptional_balance_block,
    exceptional_submission_blocks,
    attendance_blocks,
    less_hours_block,
    markdown_table,
)


def attendance_row(
    *,
    day_type: str = "Present",
    less: str = "00:17",
    check_in: str | None = "09:17",
    check_out: str | None = "17:00",
    att_date: str = "2026-09-29",
) -> dict[str, object]:
    return {
        "AttDate": att_date,
        "DayType": day_type,
        "CheckIN": check_in,
        "CheckOut": check_out,
        "NetHrs": "07:43",
        "LessHrs": less,
    }


@pytest.mark.asyncio
async def test_balance_wrapper_uses_identity_and_contract() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/Mobile/api/AI/ExceptionalEntries/Balance"
        assert dict(request.url.params) == {
            "usrEmail": "employee@example.com",
            "date": "2026-09-29",
            "instanceName": "Universal",
        }
        return httpx.Response(200, json={"hasPolicy": True, "remaining": 120})

    client = ResourcePlusClient(
        base_url="https://example.test/Mobile",
        transport=httpx.MockTransport(handler),
    )
    result = await get_exceptional_entry_balance(
        date(2026, 9, 29),
        "employee@example.com",
        "Universal",
        client=client,
    )
    assert result["remaining"] == 120


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entry_type", "minutes", "expected_optional"),
    [
        (None, None, {}),
        (1, None, {"entryType": 1}),
        (2, None, {"entryType": 2}),
        (None, 10, {"minutes": 10}),
        (1, 10, {"entryType": 1, "minutes": 10}),
    ],
)
async def test_from_summary_wrapper_optional_fields_are_explicit_only(
    entry_type: int | None,
    minutes: int | None,
    expected_optional: dict[str, int],
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/Mobile/api/AI/ExceptionalEntries/FromSummary"
        assert dict(request.url.params) == {"instanceName": "Universal"}
        body = __import__("json").loads(request.content)
        assert body == {
            "usrEmail": "employee@example.com",
            "attDate": "2026-09-29",
            "reasonID": "live-reason",
            "remarks": "Traffic",
            **expected_optional,
        }
        return httpx.Response(200, json={"isAutoApproved": False, "message": "Submitted"})

    client = ResourcePlusClient(
        base_url="https://example.test/Mobile",
        transport=httpx.MockTransport(handler),
    )
    await create_exceptional_entry_from_summary(
        "2026-09-29",
        "live-reason",
        "Traffic",
        entry_type=entry_type,
        minutes=minutes,
        usr_email="employee@example.com",
        instance_name="Universal",
        client=client,
    )


@pytest.mark.asyncio
async def test_cancel_wrapper_uses_identity_and_resolved_id() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/Mobile/api/AI/ExceptionalEntries/Cancel"
        assert dict(request.url.params) == {"instanceName": "Universal"}
        assert __import__("json").loads(request.content) == {
            "usrEmail": "employee@example.com",
            "exceptionalID": "resolved-id",
        }
        return httpx.Response(200, json={"success": True})

    client = ResourcePlusClient(
        base_url="https://example.test/Mobile",
        transport=httpx.MockTransport(handler),
    )
    await cancel_exceptional_entry(
        "resolved-id",
        "employee@example.com",
        "Universal",
        client=client,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("row", "expected"),
    [
        (attendance_row(), "eligible"),
        (attendance_row(day_type="Absent", check_in=None, check_out=None), "no_punches"),
        (attendance_row(day_type="Week End", check_in=None, check_out=None), "week_end"),
        (attendance_row(day_type="Holiday", check_in=None, check_out=None), "holiday"),
        (attendance_row(day_type="Annual Leave", check_in=None, check_out=None), "leave"),
        (attendance_row(day_type="Business Travel", check_in=None, check_out=None), "business_travel"),
        (attendance_row(less="00:00"), "no_missing_hours"),
    ],
)
async def test_authoritative_attendance_classification(monkeypatch, row, expected) -> None:
    async def summary(*args, **kwargs):
        return {"Days": [row]}

    monkeypatch.setattr(actions, "get_attendance_summary", summary)
    inspection = await inspect_less_hours_period("2026-09-29", "2026-09-29", lang=1)
    assert inspection.days[0].eligibility == expected
    assert bool(inspection.eligible_days) is (expected == "eligible")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("entry_type", "minutes", "expected_keys"),
    [
        (None, None, set()),
        (1, None, {"entry_type"}),
        (2, None, {"entry_type"}),
        (None, 10, {"minutes"}),
        (2, 10, {"entry_type", "minutes"}),
    ],
)
async def test_prepare_from_summary_stores_only_explicit_optional_fields(
    monkeypatch,
    entry_type,
    minutes,
    expected_keys,
) -> None:
    async def summary(*args, **kwargs):
        return {"Days": [attendance_row()]}

    async def reasons(*args, **kwargs):
        return [{"reasonID": "live-guid", "reasonName": "Vehicular Accident"}]

    monkeypatch.setattr(actions, "get_attendance_summary", summary)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    intent = await prepare_write_action(
        "prepare_less_hours_correction",
        {
            "target_date": "2026-09-29",
            "reason_name": "Vehicular Accident",
            "remarks": "Vehicular Accident",
            "entry_type": entry_type,
            "minutes": minutes,
        },
        lang=1,
        response_language="en",
    )
    present = {key for key in ("entry_type", "minutes") if key in intent.validated_arguments}
    assert present == expected_keys
    assert intent.validated_arguments["reason_id"] == "live-guid"
    assert "live-guid" not in intent.summary


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "row",
    [
        attendance_row(day_type="Absent", check_in=None, check_out=None),
        attendance_row(day_type="Week End", check_in=None, check_out=None),
        attendance_row(day_type="Holiday", check_in=None, check_out=None),
        attendance_row(day_type="Annual Leave", check_in=None, check_out=None),
        attendance_row(day_type="Business Travel", check_in=None, check_out=None),
        attendance_row(less="00:00"),
    ],
)
async def test_ineligible_day_never_prepares_from_summary(monkeypatch, row) -> None:
    async def summary(*args, **kwargs):
        return {"Days": [row]}

    async def reasons(*args, **kwargs):
        raise AssertionError("Reasons must not be fetched for an ineligible day")

    monkeypatch.setattr(actions, "get_attendance_summary", summary)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    with pytest.raises(ActionResolutionRequired):
        await prepare_write_action(
            "prepare_less_hours_correction",
            {"target_date": "2026-09-29", "reason_name": "Anything"},
            lang=1,
            response_language="en",
        )


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Correct my less hours yesterday", True),
        ("Only correct my late arrival yesterday", True),
        ("صحح الساعات الناقصة أمس", True),
        ("Show my profile", False),
    ],
)
def test_bilingual_less_hours_routing(message, expected) -> None:
    assert is_less_hours_request(message) is expected


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Show my less hours this month", False),
        ("Do I have any less hours?", False),
        ("Which days did I have short hours?", False),
        ("Correct my less hours", True),
        ("Fix my less hours from yesterday", True),
        ("Use my buffer for the 10th", True),
        ("Regularize the 10 September entry", True),
        ("Apply my excuse time to the 10th", True),
    ],
)
def test_less_hours_read_and_correction_intents_are_separate(message, expected) -> None:
    assert is_less_hours_correction_request(message) is expected


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Only correct my late arrival yesterday", 1),
        ("Only correct my early departure yesterday", 2),
        ("صحح تأخر الدخول فقط", 1),
        ("صحح الخروج المبكر فقط", 2),
        ("Correct my less hours", None),
    ],
)
def test_explicit_side_extraction(message, expected) -> None:
    assert _explicit_entry_type(message) == expected


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Correct only 10 minutes yesterday", 10),
        ("صحح فقط 15 دقيقة أمس", 15),
        ("Correct my less hours yesterday", None),
    ],
)
def test_explicit_partial_minutes_extraction(message, expected) -> None:
    assert _explicit_minutes(message) == expected


@pytest.mark.parametrize(
    "message",
    [
        "Cancel my pending exceptional entry",
        "Cancel my pending exceptional entries",
        "Cancel my pending exception entry",
        "Cancel my pending exception entries",
        "Cancel my pending exceptional-entry request",
        "Cancel my pending exceptional-entry requests",
        "Cancel my pending exceptional interest",
        "Cancel yesterday's exception",
        "إلغاء الإدخال الاستثنائي المعلق",
    ],
)
def test_bilingual_cancellation_routing(message) -> None:
    assert _is_cancel_exception_request(message)


@pytest.mark.asyncio
async def test_pending_exception_discovery_filters_status(monkeypatch) -> None:
    async def requests(*args, **kwargs):
        return [
            {"exceptionalID": "one", "status": "Pending", "attDate": "2026-09-29"},
            {"exceptionalID": "two", "status": "Approved", "attDate": "2026-09-28"},
            {"exceptionalID": "three", "isCancellable": True, "status": "Other"},
        ]

    monkeypatch.setattr(actions, "get_exceptional_entry_requests", requests)
    result = await find_pending_exceptional_entries("2026-09-01", "2026-09-30", lang=1)
    assert [row["exceptionalID"] for row in result] == ["one", "three"]


@pytest.mark.parametrize(
    ("row", "expected"),
    [
        ({"status": "Not Approved"}, True),
        ({"status": "Approved"}, False),
        ({"status": "Rejected"}, False),
        ({"status": "Not Approved", "isCancellable": False}, False),
        ({"status": "Approved", "isCancellable": True}, True),
        ({"status": "Pending Approval"}, True),
    ],
)
def test_cancellation_status_and_explicit_flag_precedence(row, expected) -> None:
    assert actions._is_cancellable_exception(row) is expected


@pytest.mark.asyncio
async def test_ambiguous_cancellation_never_prepares(monkeypatch) -> None:
    async def requests(*args, **kwargs):
        return [
            {"exceptionalID": "one", "status": "Pending", "attDate": "2026-09-28"},
            {"exceptionalID": "two", "status": "Pending", "attDate": "2026-09-29"},
        ]

    monkeypatch.setattr(actions, "get_exceptional_entry_requests", requests)
    with pytest.raises(ActionResolutionRequired, match="more than one"):
        await prepare_write_action(
            "prepare_cancel_exceptional_entry",
            {"date_from": "2026-09-01", "date_to": "2026-09-30"},
            lang=1,
            response_language="en",
        )


@pytest.mark.asyncio
async def test_cancellation_resolves_id_but_never_displays_it(monkeypatch) -> None:
    async def requests(*args, **kwargs):
        return [{
            "exceptionalID": "secret-guid",
            "status": "Pending",
            "attDate": "2026-09-29",
            "reasonName": "Traffic",
        }]

    monkeypatch.setattr(actions, "get_exceptional_entry_requests", requests)
    intent = await prepare_write_action(
        "prepare_cancel_exceptional_entry",
        {"date_from": "2026-09-01", "date_to": "2026-09-30"},
        lang=1,
        response_language="en",
    )
    assert intent.validated_arguments["exceptional_id"] == "secret-guid"
    assert "secret-guid" not in intent.summary


@pytest.mark.asyncio
async def test_execute_from_summary_passes_stored_arguments_exactly(monkeypatch) -> None:
    calls = []

    async def submit(*args, **kwargs):
        calls.append((args, kwargs))
        return {"isAutoApproved": False}

    monkeypatch.setattr(actions, "create_exceptional_entry_from_summary", submit)
    await execute_pending_action(
        "create_exceptional_entry_from_summary",
        {
            "att_date": "2026-09-29",
            "reason_id": "live-guid",
            "reason_name": "Traffic",
            "remarks": "Traffic",
            "less_hours": "00:17",
            "minutes": 10,
        },
    )
    assert calls[0][1] == {
        "att_date": "2026-09-29",
        "reason_id": "live-guid",
        "remarks": "Traffic",
        "entry_type": None,
        "minutes": 10,
    }


@pytest.mark.parametrize("has_policy", [True, False])
def test_balance_block_never_invents_policy(has_policy) -> None:
    payload = {
        "hasPolicy": has_policy,
        "policyName": "Buffer minutes",
        "remaining": 120,
        "resetsOn": "2026-09-28",
    }
    block = exceptional_balance_block(payload)
    assert (block is not None) is has_policy


@pytest.mark.parametrize(
    ("limit_type", "expected_unit"),
    [
        (1, "Exception count"),
        (2, "Minutes"),
        (3, None),
        (None, None),
    ],
)
def test_balance_block_uses_only_documented_limit_type_units(
    limit_type,
    expected_unit,
) -> None:
    block = exceptional_balance_block(
        {
            "hasPolicy": True,
            "policyName": "Allowance",
            "limitType": limit_type,
            "limitValue": 10,
            "used": 2,
            "remaining": 8,
        }
    )
    assert block is not None
    units = [item.value for item in block.items if item.label == "Unit"]
    assert units == ([] if expected_unit is None else [expected_unit])


def test_less_hours_block_has_v2_columns() -> None:
    day = actions.LessHoursDay(
        date(2026, 9, 29), "Present", "09:17", "17:00", "07:43", "00:17", "eligible", {}
    )
    block = less_hours_block([day])
    assert [column.label for column in block.columns] == [
        "Date", "Day Type", "Check In", "Check Out", "Worked Hours", "Less Hours", "Action"
    ]
    assert block.rows[0]["action"] == "Correction candidate"
    assert "Eligible" not in markdown_table(block)


def test_attendance_blocks_map_realistic_resourceplus_v2_rows() -> None:
    payload = {
        "Days": [
            {
                "AttDate": f"2026-09-{day:02d}",
                "DayType": day_type,
                "CheckIN": check_in,
                "CheckOut": check_out,
                "NetHrs": worked,
                "LessHrs": less,
            }
            for day, day_type, check_in, check_out, worked, less in [
                (1, "Regular", "09:01", "18:03", "09:02", "00:00"),
                (2, "Absent", None, None, "00:00", "08:00"),
                (3, "Week End", None, None, "00:00", "00:00"),
                (4, "Leave", None, None, "00:00", "00:00"),
                (5, "Business Travel", "08:45", "17:30", "08:45", "00:00"),
            ]
        ]
    }

    table = next(block for block in attendance_blocks(payload) if block.type == "table")

    assert table.title == "Attendance"
    assert [row["status"] for row in table.rows] == [
        "Regular", "Absent", "Week End", "Leave", "Business Travel"
    ]
    assert table.rows[0] == {
        "date": "2026-09-01",
        "status": "Regular",
        "in": "09:01",
        "out": "18:03",
        "worked": "09:02",
        "shortfall": "00:00",
    }
    assert table.rows[4]["in"] == "08:45"
    assert table.rows[4]["out"] == "17:30"
    assert table.rows[4]["worked"] == "08:45"


@pytest.mark.parametrize("auto_approved", [True, False])
def test_submission_blocks_render_result_warning_and_split_entries(auto_approved) -> None:
    blocks = exceptional_submission_blocks(
        {
            "isAutoApproved": auto_approved,
            "message": "ResourcePlus result",
            "warning": "One split part failed",
            "entries": [{"entry": "Late IN", "minutes": 10, "status": "Created"}],
        },
        language="en",
    )
    assert [block.type for block in blocks] == ["notice", "notice", "table"]
    assert blocks[1].level == "warning"


@pytest.mark.parametrize(
    ("payload", "expected"),
    [
        ({"success": True, "isAutoApproved": True}, True),
        ({"success": True, "isAutoApproved": False}, True),
        ({"success": False}, False),
        (
            {
                "success": False,
                "isAutoApproved": True,
                "remaining": 100,
                "entries": [{"status": "Approved"}],
            },
            False,
        ),
        ({"isAutoApproved": True}, True),
        ({"isAutoApproved": False}, True),
        ({"message": "No documented success marker"}, False),
    ],
)
def test_from_summary_success_contract_and_narrow_compatibility(payload, expected) -> None:
    assert chat_service._from_summary_operation_success(payload) is expected


@pytest.mark.parametrize("success", [True, False])
def test_from_summary_warning_is_rendered_for_success_and_failure(success) -> None:
    blocks = exceptional_submission_blocks(
        {
            "success": success,
            "isAutoApproved": False,
            "message": "Authoritative result",
            "warning": "Authoritative warning",
        },
        language="en",
        success=success,
    )
    assert blocks[0].level == ("success" if success else "error")
    assert blocks[1].level == "warning"
    assert blocks[1].message == "Authoritative warning"


def test_cancellation_block_does_not_expose_id() -> None:
    block = cancellable_exceptions_block([{
        "exceptionalID": "hidden",
        "entryTime": "2026-09-29T09:17:00",
        "entryType": 1,
        "reason": "Traffic",
        "status": "Pending",
    }])
    assert [column.label for column in block.columns] == [
        "Date", "Type", "Reason", "Status",
    ]
    assert block.rows == [{
        "date": "2026-09-29",
        "type": "Late Arrival",
        "reason": "Traffic",
        "status": "Pending",
    }]
    assert "Minutes" not in [column.label for column in block.columns]
    assert "hidden" not in block.model_dump_json()


def test_cancellation_block_maps_early_departure() -> None:
    block = cancellable_exceptions_block([{
        "exceptionalID": "hidden",
        "entryTime": "22/09/2026 16:30",
        "entryType": 2,
        "reason": "Family Circumstances",
        "status": "Not Approved",
    }])
    assert block.rows[0]["date"] == "2026-09-22"
    assert block.rows[0]["type"] == "Early Departure"


@pytest.mark.asyncio
@pytest.mark.parametrize("auto_approved", [True, False])
async def test_confirmation_executes_from_summary_once_and_surfaces_result(
    monkeypatch,
    auto_approved,
) -> None:
    store = InMemorySessionStore()
    identity = RequestIdentity("employee@example.com", "Universal")
    token = bind_request_identity(identity)
    try:
        session_id = store.ensure_session()
        pending = store.create_pending_action(
            session_id,
            action_type="create_exceptional_entry_from_summary",
            validated_arguments={
                "att_date": "2026-09-29",
                "reason_id": "live-guid",
                "reason_name": "Traffic",
                "remarks": "Traffic",
                "less_hours": "00:17",
            },
            summary="Confirm correction",
            language="en",
        )
    finally:
        reset_request_identity(token)

    writes = []

    async def submit(**kwargs):
        writes.append(kwargs)
        return {
            "success": True,
            "isAutoApproved": auto_approved,
            "message": "Your allowance was updated.",
            "warning": "Check the split.",
            "requestedMinutes": 17,
            "remaining": 103,
            "resetsOn": "2026-10-05",
            "entries": [{"entry": "Late IN", "minutes": 17, "status": "Approved"}],
        }

    monkeypatch.setattr(actions, "create_exceptional_entry_from_summary", submit)
    request = ChatRequest(
        message="yes",
        session_id=session_id,
        confirmation_id=pending.confirmation_id,
        email=identity.email,
        instance=identity.instance,
    )
    response = await chat_service.process_chat(request, store=store)
    replay = await chat_service.process_chat(request, store=store)
    assert response.success is True
    assert len(writes) == 1
    expected_lead = (
        "Your attendance correction was auto-approved."
        if auto_approved
        else "Your attendance correction was submitted and is waiting for manager approval."
    )
    assert response.message.startswith(expected_lead)
    assert response.speech_message == response.message
    assert "ResourcePlus" not in response.message
    assert "Your allowance was updated." in response.message
    assert "Warning: Check the split." in response.message
    assert "103" in response.message and "2026-10-05" in response.message
    assert [block.type for block in response.blocks] == ["notice", "notice", "table"]
    assert replay.success is False
    assert len(writes) == 1


@pytest.mark.asyncio
async def test_documented_from_summary_failure_wins_and_surfaces_message_warning(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()
    identity = RequestIdentity("employee@example.com", "Universal")
    token = bind_request_identity(identity)
    try:
        session_id = store.ensure_session()
        pending = store.create_pending_action(
            session_id,
            action_type="create_exceptional_entry_from_summary",
            validated_arguments={
                "att_date": "2026-09-29",
                "reason_id": "live-guid",
                "reason_name": "Traffic",
                "remarks": "Traffic",
                "less_hours": "00:17",
            },
            summary="Confirm correction",
            language="en",
        )
    finally:
        reset_request_identity(token)

    writes = []

    async def rejected(**kwargs):
        writes.append(kwargs)
        return {
            "success": False,
            "isAutoApproved": True,
            "message": "Missing time is more than 4 hours. Apply leave instead.",
            "warning": "No entry was created.",
            "remaining": 120,
            "entries": [{"status": "Approved"}],
        }

    monkeypatch.setattr(actions, "create_exceptional_entry_from_summary", rejected)
    response = await chat_service.process_chat(
        ChatRequest(
            message="yes",
            session_id=session_id,
            confirmation_id=pending.confirmation_id,
            email=identity.email,
            instance=identity.instance,
        ),
        store=store,
    )
    replay = await chat_service.process_chat(
        ChatRequest(
            message="yes",
            session_id=session_id,
            confirmation_id=pending.confirmation_id,
            email=identity.email,
            instance=identity.instance,
        ),
        store=store,
    )
    assert response.success is False
    assert response.message.startswith(
        "Your attendance correction could not be completed."
    )
    assert "Missing time is more than 4 hours. Apply leave instead." in response.message
    assert "Warning: No entry was created." in response.message
    assert "remaining" not in response.message.casefold()
    assert response.blocks[0].level == "error"
    assert response.blocks[1].level == "warning"
    assert [block.type for block in response.blocks] == ["notice", "notice", "actions"]
    assert response.blocks[2].actions[0].value == "Show available day types"
    assert all(block.type not in {"table", "key_value"} for block in response.blocks)
    assert replay.success is False
    assert len(writes) == 1


@pytest.mark.asyncio
async def test_confirmation_identity_mismatch_makes_zero_writes(monkeypatch) -> None:
    store = InMemorySessionStore()
    token = bind_request_identity(RequestIdentity("owner@example.com", "Universal"))
    try:
        session_id = store.ensure_session()
        pending = store.create_pending_action(
            session_id,
            action_type="cancel_exceptional_entry",
            validated_arguments={"exceptional_id": "hidden", "display": "29 September"},
            summary="Confirm cancellation",
            language="en",
        )
    finally:
        reset_request_identity(token)

    writes = []

    async def cancel(*args, **kwargs):
        writes.append(kwargs)

    monkeypatch.setattr(actions, "cancel_exceptional_entry", cancel)
    response = await chat_service.process_chat(
        ChatRequest(
            message="yes",
            session_id=session_id,
            confirmation_id=pending.confirmation_id,
            email="other@example.com",
            instance="Universal",
        ),
        store=store,
    )
    assert response.success is False
    assert writes == []


def test_new_write_types_are_audit_allowlisted() -> None:
    assert {
        "create_exceptional_entry_from_summary",
        "cancel_exceptional_entry",
    } <= SAFE_ACTION_TYPES


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Show my less hours this month",
        "Show my less hours",
        "Do I have any less hours?",
        "Which days did I have short hours?",
        "List my less-hour entries",
        "How many less-hour days do I have?",
        "Show my less hours on September 29",
    ],
)
async def test_less_hours_reads_create_no_write_preparation_state(
    monkeypatch,
    message,
) -> None:
    store = InMemorySessionStore()
    balance_dates = []
    day = actions.LessHoursDay(
        date(2026, 9, 29), "Regular", "09:17", "17:00", "07:43", "00:17", "eligible", {}
    )

    async def inspect(*args, **kwargs):
        return actions.LessHoursInspection((day,), (day,))

    async def balance(target_date, *args, **kwargs):
        balance_dates.append(str(target_date))
        return {"hasPolicy": True, "remaining": 120, "resetsOn": "2026-10-05"}

    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)

    response = await chat_service.process_chat(ChatRequest(message=message), store=store)
    pending, _ = store.get_pending_action(response.session_id)

    assert response.needs_reason is False
    assert response.requires_confirmation is False
    assert response.message == (
        "You have 1 less-hours entry eligible for correction during this period."
    )
    assert response.speech_message == response.message
    assert pending is None
    assert store.get_conversation_draft(response.session_id) is None
    assert [block.type for block in response.blocks] == ["table", "key_value", "actions"]
    assert response.blocks[0].rows[0]["less"] == "00:17"
    assert response.blocks[0].rows[0]["action"] == "Correction candidate"
    assert response.blocks[2].actions[0].value.startswith("Correct my less hours")
    assert balance_dates == ["2026-09-29"]


@pytest.mark.asyncio
async def test_less_hours_read_with_zero_candidates_skips_balance_reasons_and_state(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()
    day = actions.LessHoursDay(
        date(2026, 9, 29), "Regular", "09:00", "18:00", "09:00", "00:00", "no_missing_hours", {}
    )

    async def inspect(*args, **kwargs):
        return actions.LessHoursInspection((day,), ())

    async def forbidden(*args, **kwargs):
        raise AssertionError("zero candidates must not fetch Balance or Reasons")

    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", forbidden)
    monkeypatch.setattr(conversation, "cached_exception_reasons", forbidden)
    response = await chat_service.process_chat(
        ChatRequest(message="Do I have any less hours yesterday?"),
        store=store,
    )
    pending, _ = store.get_pending_action(response.session_id)

    assert response.tools_used == ["get_attendance_summary"]
    assert response.message == (
        "You have no less-hours entries eligible for correction during this period."
    )
    assert response.speech_message == response.message
    assert response.blocks[0].rows == []
    assert store.get_conversation_draft(response.session_id) is None
    assert pending is None


@pytest.mark.asyncio
async def test_less_hours_read_with_multiple_candidates_omits_generic_balance(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()
    days = tuple(
        actions.LessHoursDay(day, "Regular", "09:10", "17:00", "07:50", "00:10", "eligible", {})
        for day in (date(2026, 9, 10), date(2026, 9, 24))
    )

    async def inspect(*args, **kwargs):
        return actions.LessHoursInspection(days, days)

    async def forbidden_balance(*args, **kwargs):
        raise AssertionError("multiple candidates must not fetch a generic Balance")

    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", forbidden_balance)
    response = await chat_service.process_chat(
        ChatRequest(message="Show my less hours this month"),
        store=store,
    )

    assert response.tools_used == ["get_attendance_summary"]
    assert [block.type for block in response.blocks] == ["table", "actions"]
    assert len(response.blocks[0].rows) == 2
    assert all(row["action"] == "Correction candidate" for row in response.blocks[0].rows)


@pytest.mark.asyncio
async def test_less_hours_conversation_reason_continuation_prepares_without_write(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()
    balance_dates = []
    day = actions.LessHoursDay(
        date(2026, 9, 29), "Present", "09:17", "17:00", "07:43", "00:17", "eligible", {}
    )

    async def inspect(*args, **kwargs):
        return actions.LessHoursInspection((day,), (day,))

    async def balance(target_date, *args, **kwargs):
        balance_dates.append(str(target_date))
        return {"hasPolicy": True, "remaining": 120, "resetsOn": "2026-10-05"}

    async def cached_reasons(*args, **kwargs):
        return [{"reasonID": "cached-id", "reasonName": "Vehicular Accident"}]

    async def summary(*args, **kwargs):
        return {"Days": [attendance_row()]}

    async def live_reasons(*args, **kwargs):
        return [{"reasonID": "fresh-id", "reasonName": "Vehicular Accident"}]

    writes = []

    async def submit(**kwargs):
        writes.append(kwargs)

    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)
    monkeypatch.setattr(conversation, "cached_exception_reasons", cached_reasons)
    monkeypatch.setattr(actions, "get_attendance_summary", summary)
    monkeypatch.setattr(actions, "get_exception_reasons", live_reasons)
    monkeypatch.setattr(actions, "create_exceptional_entry_from_summary", submit)

    first = await chat_service.process_chat(
        ChatRequest(message="Correct my less hours yesterday"),
        store=store,
    )
    prepared = await chat_service.process_chat(
        ChatRequest(message="Vehicular Accident", session_id=first.session_id),
        store=store,
    )
    pending, _ = store.get_pending_action(first.session_id)
    assert first.needs_reason is True
    assert {block.type for block in first.blocks} == {"table", "key_value", "actions"}
    assert prepared.requires_confirmation is True
    assert pending is not None
    assert pending.validated_arguments["reason_id"] == "fresh-id"
    assert "entry_type" not in pending.validated_arguments
    assert "minutes" not in pending.validated_arguments
    assert balance_dates == ["2026-09-29"]
    assert writes == []


@pytest.mark.asyncio
async def test_absent_conversation_offers_leave_and_preserves_date_for_day_type(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()
    day = actions.LessHoursDay(
        date(2026, 9, 29), "Absent", None, None, "00:00", "08:00", "no_punches", {}
    )

    async def inspect(*args, **kwargs):
        return actions.LessHoursInspection((day,), ())

    async def balance(*args, **kwargs):
        raise AssertionError("ineligible correction must not fetch Balance")

    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)
    response = await chat_service.process_chat(
        ChatRequest(message="Correct my less hours yesterday"),
        store=store,
    )
    draft = store.get_conversation_draft(response.session_id)
    assert "Leave or Business Travel" in response.message
    assert response.requires_confirmation is False
    assert draft is not None and draft.intent == "book_day_type"
    assert draft.slots["date_from"] == "2026-09-29"


@pytest.mark.asyncio
async def test_multiple_candidate_dates_ask_for_selection_without_balance(monkeypatch) -> None:
    store = InMemorySessionStore()
    days = tuple(
        actions.LessHoursDay(day, "Present", "09:10", "17:00", "07:50", "00:10", "eligible", {})
        for day in (date(2026, 9, 28), date(2026, 9, 29))
    )

    async def inspect(*args, **kwargs):
        return actions.LessHoursInspection(days, days)

    async def balance(*args, **kwargs):
        raise AssertionError("multiple candidates must not fetch a generic Balance")

    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)
    response = await chat_service.process_chat(
        ChatRequest(message="Correct my less hours this week"),
        store=store,
    )
    assert "more than one correction candidate" in response.message
    assert response.blocks[0].type == "table"
    assert len(response.blocks[0].rows) == 2
    assert [block.type for block in response.blocks] == ["table", "actions"]
    assert response.tools_used == ["get_attendance_summary"]
    assert response.requires_confirmation is False


@pytest.mark.asyncio
async def test_selecting_candidate_uses_selected_date_for_balance(monkeypatch) -> None:
    store = InMemorySessionStore()
    days = tuple(
        actions.LessHoursDay(day, "Present", "09:10", "17:00", "07:50", "00:10", "eligible", {})
        for day in (date(2026, 9, 10), date(2026, 9, 24))
    )
    balance_dates = []

    async def inspect(start, end, **kwargs):
        start_date = start if isinstance(start, date) else date.fromisoformat(str(start))
        end_date = end if isinstance(end, date) else date.fromisoformat(str(end))
        selected = tuple(day for day in days if start_date <= day.attendance_date <= end_date)
        return actions.LessHoursInspection(selected, selected)

    async def balance(target_date, *args, **kwargs):
        balance_dates.append(str(target_date))
        return {"hasPolicy": True, "remaining": 30}

    async def reasons(*args, **kwargs):
        return [{"reasonID": "cached-id", "reasonName": "Traffic"}]

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)
    monkeypatch.setattr(conversation, "cached_exception_reasons", reasons)

    first = await chat_service.process_chat(
        ChatRequest(message="Correct my less hours this month"),
        store=store,
    )
    selected = await chat_service.process_chat(
        ChatRequest(message="2026-09-24", session_id=first.session_id),
        store=store,
    )

    assert balance_dates == ["2026-09-24"]
    assert selected.needs_reason is True
    assert [block.type for block in selected.blocks] == ["table", "key_value", "actions"]


@pytest.mark.asyncio
async def test_cancel_conversation_discovers_then_prepares_confirmation(monkeypatch) -> None:
    store = InMemorySessionStore()
    row = {
        "exceptionalID": "resolved-only",
        "status": "Pending",
        "attDate": "2026-09-29",
        "reasonName": "Traffic",
    }

    async def find(*args, **kwargs):
        return [row]

    async def requests(*args, **kwargs):
        return [row]

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(conversation, "find_pending_exceptional_entries", find)
    monkeypatch.setattr(actions, "get_exceptional_entry_requests", requests)
    response = await chat_service.process_chat(
        ChatRequest(message="Cancel yesterday's exception"),
        store=store,
    )
    pending, _ = store.get_pending_action(response.session_id)
    assert response.requires_confirmation is True
    assert pending is not None
    assert pending.validated_arguments["exceptional_id"] == "resolved-only"
    assert "resolved-only" not in response.message