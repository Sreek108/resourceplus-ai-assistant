from __future__ import annotations

from datetime import date

import pytest

from app.ai.sessions import InMemorySessionStore
from app.models.schemas import ChatRequest
from app.services import chat as chat_service, fast_reads
from app.services.request_history import normalize_request_history
from app.services.response_blocks import request_history_block


DAY_TYPES = [
    {
        "mappingID": "leave-compassionate-secret",
        "dayType": "Compassionate Leave",
        "dateFrom": "2026-10-08",
        "status": "Approved",
    },
    {
        "mappingID": "leave-compensatory-secret",
        "dayType": "Compensatory Leave",
        "dateFrom": "2026-10-09",
        "status": "Approved",
    },
]
EXCEPTIONAL = [
    {
        "exceptionalID": "attendance-secret",
        "entryTime": "2026-09-02T09:03:00",
        "reasonName": "Outside Work",
        "status": "Not Approved",
    }
]


def _merged_payload(exceptional_status: str = "Not Approved"):
    exceptional = {**EXCEPTIONAL[0], "status": exceptional_status}
    return {
        "absence_requests": DAY_TYPES,
        "exceptional_entry_requests": [exceptional],
        "merged_requests": [
            *(
                {
                    "request_kind": "absence",
                    "raw_status": row["status"],
                    "record": row,
                }
                for row in DAY_TYPES
            ),
            {
                "request_kind": "exceptional_entry",
                "raw_status": exceptional_status,
                "record": exceptional,
            },
        ],
    }


def _session_with_correction_then_leave() -> tuple[InMemorySessionStore, str]:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    store.save_trusted_result(
        session_id,
        message="Correction submitted",
        tools_used=("create_exceptional_entry_from_summary",),
        language="en",
        recent_request_category="attendance_correction",
        recent_request_date="2026-09-02",
        recent_request_detail="Outside Work",
        recent_request_state="submitted_for_approval",
    )
    store.save_trusted_result(
        session_id,
        message="Leave approved",
        tools_used=("get_my_request_status",),
        language="en",
        recent_request_category="leave",
        recent_request_date="2026-10-09",
        recent_request_detail="Compensatory Leave",
        recent_request_state="approved",
    )
    return store, session_id


def _install_unified(monkeypatch, *, exceptional_status: str = "Not Approved"):
    calls: list[tuple[str, str]] = []

    async def requests(start, end, **kwargs):
        calls.append((start, end))
        return _merged_payload(exceptional_status)

    async def leave_requests(*args, **kwargs):
        return DAY_TYPES

    async def correction_requests(*args, **kwargs):
        return [{**EXCEPTIONAL[0], "status": exceptional_status}]

    async def no_model(*args, **kwargs):
        raise AssertionError("request-history reads must not call a model")

    monkeypatch.setattr(fast_reads, "resourceplus_today", lambda: date(2026, 10, 4))
    monkeypatch.setattr(fast_reads, "get_my_request_status", requests)
    monkeypatch.setattr(fast_reads, "get_my_day_type_requests", leave_requests)
    monkeypatch.setattr(fast_reads, "get_exceptional_entry_requests", correction_requests)
    monkeypatch.setattr(chat_service, "run_agent", no_model)
    return calls


@pytest.mark.asyncio
async def test_generic_request_feed_merges_leave_and_attendance_without_ids(monkeypatch) -> None:
    calls = _install_unified(monkeypatch)
    store, session_id = _session_with_correction_then_leave()
    response = await chat_service.process_chat(
        ChatRequest(message="show my requests", session_id=session_id), store=store
    )

    assert calls == [("2026-06-06", "2027-10-04")]
    assert response.tools_used == ["get_my_request_status"]
    assert len(response.blocks[0].rows) == 3
    assert [column.key for column in response.blocks[0].columns] == [
        "request", "date", "detail", "status"
    ]
    assert {row["request"] for row in response.blocks[0].rows} == {
        "Leave", "Attendance correction"
    }
    assert [row["status"] for row in response.blocks[0].rows] == [
        "Approved", "Approved", "Pending"
    ]
    serialized = response.model_dump_json()
    assert "attendance-secret" not in serialized
    assert "leave-compensatory-secret" not in serialized


@pytest.mark.asyncio
async def test_generic_pending_ignores_recent_leave_date_and_keeps_correction(monkeypatch) -> None:
    _install_unified(monkeypatch)
    store, session_id = _session_with_correction_then_leave()
    response = await chat_service.process_chat(
        ChatRequest(message="show my pending request", session_id=session_id), store=store
    )

    assert response.message == (
        "You have one request still waiting for approval: your Sep 2 Outside Work "
        "attendance correction."
    )
    assert response.blocks[0].rows == [{
        "request": "Attendance correction",
        "type": "Attendance correction",
        "date": "2026-09-02",
        "detail": "Outside Work",
        "status": "Pending",
    }]


@pytest.mark.asyncio
async def test_approved_filter_does_not_claim_those_are_the_only_requests(monkeypatch) -> None:
    _install_unified(monkeypatch)
    store, session_id = _session_with_correction_then_leave()
    response = await chat_service.process_chat(
        ChatRequest(message="which requests were approved", session_id=session_id), store=store
    )

    assert "in your recent request history" in response.message
    assert len(response.blocks[0].rows) == 2
    assert all(row["status"] == "Approved" for row in response.blocks[0].rows)


@pytest.mark.asyncio
async def test_category_scopes_use_only_the_required_live_endpoint(monkeypatch) -> None:
    leave_calls = []
    correction_calls = []

    async def leave(start, end, **kwargs):
        leave_calls.append((start, end))
        return DAY_TYPES

    async def corrections(start, end, **kwargs):
        correction_calls.append((start, end))
        return EXCEPTIONAL

    async def forbidden(*args, **kwargs):
        raise AssertionError("scoped reads must not call the combined endpoint")

    monkeypatch.setattr(fast_reads, "resourceplus_today", lambda: date(2026, 10, 4))
    monkeypatch.setattr(fast_reads, "get_my_day_type_requests", leave)
    monkeypatch.setattr(fast_reads, "get_exceptional_entry_requests", corrections)
    monkeypatch.setattr(fast_reads, "get_my_request_status", forbidden)
    store, session_id = _session_with_correction_then_leave()

    leave_response = await chat_service.process_chat(
        ChatRequest(message="show my leave requests", session_id=session_id), store=store
    )
    correction_response = await chat_service.process_chat(
        ChatRequest(
            message="show my attendance correction requests",
            session_id=session_id,
        ),
        store=store,
    )
    exceptional_response = await chat_service.process_chat(
        ChatRequest(
            message="show my exceptional entry requests",
            session_id=session_id,
        ),
        store=store,
    )

    assert leave_response.tools_used == ["get_my_day_type_requests"]
    assert correction_response.tools_used == ["get_exceptional_entries"]
    assert exceptional_response.tools_used == ["get_exceptional_entries"]
    assert all(row["request"] == "Leave" for row in leave_response.blocks[0].rows)
    assert all(
        row["request"] == "Attendance correction"
        for row in correction_response.blocks[0].rows
    )
    assert len(leave_calls) == 1
    assert len(correction_calls) == 2


@pytest.mark.asyncio
async def test_explicit_request_date_range_wins(monkeypatch) -> None:
    calls = _install_unified(monkeypatch)
    response = await chat_service.process_chat(
        ChatRequest(message="show my requests from 2026-10-08 to 2026-10-09"),
        store=InMemorySessionStore(),
    )

    assert calls == [("2026-10-08", "2026-10-09")]
    assert len(response.blocks[0].rows) == 2
    assert {row["detail"] for row in response.blocks[0].rows} == {
        "Compassionate Leave", "Compensatory Leave"
    }


@pytest.mark.asyncio
async def test_authoritative_approved_supersedes_session_pending(monkeypatch) -> None:
    _install_unified(monkeypatch, exceptional_status="Approved")
    async def approved_correction(*args, **kwargs):
        return [{**EXCEPTIONAL[0], "status": "Approved"}]

    monkeypatch.setattr(fast_reads, "get_exceptional_entry_requests", approved_correction)
    store, session_id = _session_with_correction_then_leave()
    response = await chat_service.process_chat(
        ChatRequest(
            message="show my attendance correction requests",
            session_id=session_id,
        ),
        store=store,
    )

    assert response.blocks[0].rows[0]["status"] == "Approved"
    assert "Awaiting" not in response.model_dump_json()


@pytest.mark.asyncio
async def test_arabic_pending_request_uses_the_same_unified_feed(monkeypatch) -> None:
    _install_unified(monkeypatch)
    store, session_id = _session_with_correction_then_leave()
    response = await chat_service.process_chat(
        ChatRequest(message="اعرض طلباتي المعلقة", session_id=session_id), store=store
    )

    assert response.language == "ar"
    assert len(response.blocks[0].rows) == 1
    assert response.blocks[0].rows[0]["request"] == "Attendance correction"
    assert response.blocks[0].rows[0]["status"] == "Pending"
    assert response.blocks[0].columns[0].label == "الطلب"


@pytest.mark.asyncio
async def test_arabic_all_requests_returns_every_category_and_status(monkeypatch) -> None:
    _install_unified(monkeypatch)
    response = await chat_service.process_chat(
        ChatRequest(message="اعرض طلباتي"), store=InMemorySessionStore()
    )

    assert response.language == "ar"
    assert len(response.blocks[0].rows) == 3
    assert {row["request"] for row in response.blocks[0].rows} == {
        "Leave", "Attendance correction"
    }
    assert {row["status"] for row in response.blocks[0].rows} == {
        "Approved", "Pending"
    }


def test_employee_status_normalization_keeps_raw_resourceplus_status() -> None:
    normalized = normalize_request_history({
        "merged_requests": [
            {
                "request_kind": "absence",
                "raw_status": status,
                "record": {
                    "dateFrom": f"2026-10-{day:02d}",
                    "dayType": "Live type",
                    "status": status,
                },
            }
            for day, status in enumerate(
                ("Approved", "Pending", "Not Approved", "Rejected"), start=1
            )
        ]
    })

    assert {item.raw_status: item.resolved_status for item in normalized} == {
        "Approved": "Approved",
        "Pending": "Pending",
        "Not Approved": "Pending",
        "Rejected": "Rejected",
    }


def _history_payload(*rows: tuple[str, dict]) -> dict:
    return {
        "merged_requests": [
            {
                "request_kind": kind,
                "raw_status": record["status"],
                "record": record,
            }
            for kind, record in rows
        ]
    }


def _attendance_record(
    *,
    reason: str = "Outside Work",
    status: str = "Not Approved",
    request_id: str = "attendance-one",
) -> dict:
    return {
        "exceptionalID": request_id,
        "entryTime": "2026-09-02T09:03:00",
        "reasonName": reason,
        "status": status,
    }


def test_duplicate_identical_attendance_correction_renders_once() -> None:
    duplicate = _attendance_record()
    requests = normalize_request_history(_history_payload(
        ("exceptional_entry", duplicate),
        ("exceptional_entry", dict(duplicate)),
    ))

    block = request_history_block(requests)

    assert len(requests) == 1
    assert len(block.rows) == 1
    assert block.rows[0] == {
        "request": "Attendance correction",
        "type": "Attendance correction",
        "date": "2026-09-02",
        "detail": "Outside Work",
        "status": "Pending",
    }
    assert requests[0].raw_status == "Not Approved"
    assert requests[0].source == "ExceptionalEntries"
    assert requests[0].source_reference == "attendance-one"
    assert requests[0].raw_status_history == ("Not Approved",)
    assert requests[0].source_history == ("ExceptionalEntries",)


def test_same_date_attendance_corrections_with_different_reasons_are_preserved() -> None:
    requests = normalize_request_history(_history_payload(
        ("exceptional_entry", _attendance_record()),
        (
            "exceptional_entry",
            _attendance_record(reason="Client Visit", request_id="attendance-two"),
        ),
    ))

    assert len(requests) == 2
    assert {item.detail for item in requests} == {"Outside Work", "Client Visit"}


def test_same_date_leave_and_attendance_correction_are_preserved() -> None:
    leave = {
        "mappingID": "leave-one",
        "dateFrom": "2026-09-02",
        "dayType": "Live leave type",
        "status": "Not Approved",
    }
    requests = normalize_request_history(_history_payload(
        ("absence", leave),
        ("exceptional_entry", _attendance_record()),
    ))

    assert len(requests) == 2
    assert {item.category for item in requests} == {
        "leave",
        "attendance_correction",
    }


def test_duplicate_approved_request_is_returned_once() -> None:
    duplicate = _attendance_record(status="Approved")
    requests = normalize_request_history(_history_payload(
        ("exceptional_entry", duplicate),
        ("exceptional_entry", dict(duplicate)),
    ))

    assert len(requests) == 1
    assert requests[0].resolved_status == "Approved"
    assert requests[0].raw_status == "Approved"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected_language", "count_fragment"),
    [
        ("show my pending requests", "en", "one request"),
        ("اعرض طلباتي المعلقة", "ar", "طلب واحد"),
    ],
)
async def test_duplicate_pending_request_counts_once_in_english_and_arabic(
    monkeypatch,
    message: str,
    expected_language: str,
    count_fragment: str,
) -> None:
    duplicate = _attendance_record()
    payload = _history_payload(
        ("exceptional_entry", duplicate),
        ("exceptional_entry", dict(duplicate)),
    )

    async def requests(*args, **kwargs):
        return payload

    async def no_model(*args, **kwargs):
        raise AssertionError("deduplicated request-history reads must not call a model")

    monkeypatch.setattr(fast_reads, "get_my_request_status", requests)
    monkeypatch.setattr(chat_service, "run_agent", no_model)
    response = await chat_service.process_chat(
        ChatRequest(message=message),
        store=InMemorySessionStore(),
    )

    assert response.language == expected_language
    assert count_fragment in response.message
    assert len(response.blocks[0].rows) == 1
    assert response.blocks[0].rows[0]["status"] == "Pending"


@pytest.mark.asyncio
@pytest.mark.parametrize("message", ["show my pending requests", "اعرض طلباتي المعلقة"])
async def test_pending_filter_includes_pending_and_not_approved_without_model(
    monkeypatch, message
) -> None:
    payload = {
        "merged_requests": [
            {
                "request_kind": "absence",
                "raw_status": "Pending",
                "record": {
                    "dateFrom": "2026-10-10",
                    "dayType": "Annual Leave",
                    "status": "Pending",
                },
            },
            {
                "request_kind": "exceptional_entry",
                "raw_status": "Not Approved",
                "record": {
                    "entryTime": "2026-09-02T09:03:00",
                    "reasonName": "Outside Work",
                    "status": "Not Approved",
                },
            },
            {
                "request_kind": "absence",
                "raw_status": "Approved",
                "record": {
                    "dateFrom": "2026-10-09",
                    "dayType": "Compensatory Leave",
                    "status": "Approved",
                },
            },
            {
                "request_kind": "absence",
                "raw_status": "Rejected",
                "record": {
                    "dateFrom": "2026-10-08",
                    "dayType": "Compassionate Leave",
                    "status": "Rejected",
                },
            },
        ]
    }

    async def requests(*args, **kwargs):
        return payload

    async def no_model(*args, **kwargs):
        raise AssertionError("request-history reads must not call a model")

    monkeypatch.setattr(fast_reads, "get_my_request_status", requests)
    monkeypatch.setattr(chat_service, "run_agent", no_model)
    response = await chat_service.process_chat(
        ChatRequest(message=message), store=InMemorySessionStore()
    )

    assert len(response.blocks[0].rows) == 2
    assert {row["detail"] for row in response.blocks[0].rows} == {
        "Annual Leave", "Outside Work"
    }
    assert {row["status"] for row in response.blocks[0].rows} == {"Pending"}


@pytest.mark.asyncio
async def test_arabic_attendance_request_scope_uses_exceptional_entries_only(monkeypatch) -> None:
    calls = []

    async def corrections(start, end, **kwargs):
        calls.append((start, end))
        return EXCEPTIONAL

    async def forbidden(*args, **kwargs):
        raise AssertionError("Arabic attendance request scope must not fetch leave")

    monkeypatch.setattr(fast_reads, "resourceplus_today", lambda: date(2026, 10, 4))
    monkeypatch.setattr(fast_reads, "get_exceptional_entry_requests", corrections)
    monkeypatch.setattr(fast_reads, "get_my_request_status", forbidden)
    monkeypatch.setattr(fast_reads, "get_my_day_type_requests", forbidden)
    store, session_id = _session_with_correction_then_leave()
    response = await chat_service.process_chat(
        ChatRequest(message="اعرض طلبات تصحيح الحضور", session_id=session_id), store=store
    )

    assert response.language == "ar"
    assert response.tools_used == ["get_exceptional_entries"]
    assert len(calls) == 1
    assert response.blocks[0].rows[0]["request"] == "Attendance correction"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected_detail"),
    [
        ("What happened to my latest request?", "Compensatory Leave"),
        ("What happened to my leave?", "Compensatory Leave"),
        ("What about my attendance correction?", "Outside Work"),
    ],
)
async def test_natural_request_follow_ups_resolve_across_the_unified_feed(
    monkeypatch, message, expected_detail
) -> None:
    _install_unified(monkeypatch)
    store, session_id = _session_with_correction_then_leave()
    response = await chat_service.process_chat(
        ChatRequest(message=message, session_id=session_id), store=store
    )

    assert response.blocks[0].rows[0]["detail"] == expected_detail
    assert len(response.blocks[0].rows) == 1


@pytest.mark.asyncio
async def test_all_approved_question_checks_every_request_category(monkeypatch) -> None:
    _install_unified(monkeypatch)
    store, session_id = _session_with_correction_then_leave()
    response = await chat_service.process_chat(
        ChatRequest(message="Are all my requests approved?", session_id=session_id),
        store=store,
    )

    assert response.message == "No — you have 1 request still waiting for approval."
    assert len(response.blocks[0].rows) == 3


@pytest.mark.asyncio
async def test_rejected_filter_requires_an_authoritative_rejected_status(monkeypatch) -> None:
    payload = _merged_payload()
    rejected = {
        "request_kind": "absence",
        "raw_status": "Rejected",
        "record": {
            "dayType": "Annual Leave",
            "dateFrom": "2026-10-12",
            "status": "Rejected",
        },
    }
    payload["merged_requests"].append(rejected)

    async def requests(*args, **kwargs):
        return payload

    monkeypatch.setattr(fast_reads, "get_my_request_status", requests)
    response = await chat_service.process_chat(
        ChatRequest(message="show rejected requests"), store=InMemorySessionStore()
    )

    assert len(response.blocks[0].rows) == 1
    assert response.blocks[0].rows[0]["detail"] == "Annual Leave"
    assert response.blocks[0].rows[0]["status"] == "Rejected"


@pytest.mark.asyncio
async def test_request_history_session_remains_identity_bound(monkeypatch) -> None:
    _install_unified(monkeypatch)
    store = InMemorySessionStore()
    first = await chat_service.process_chat(
        ChatRequest(
            message="show my requests",
            email="employee.one@example.com",
            instance="Universal",
        ),
        store=store,
    )
    response = await chat_service.process_chat(
        ChatRequest(
            message="show my requests",
            session_id=first.session_id,
            email="employee.two@example.com",
            instance="Universal",
        ),
        store=store,
    )

    assert response.success is False
    assert "current signed-in user" in response.message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "show my request list",
        "show my requests list",
        "show Mai request",
        "show my quick request list",
        "show request list",
    ],
)
async def test_voice_request_list_variants_use_unified_zero_model_path(
    monkeypatch, message
) -> None:
    calls = _install_unified(monkeypatch)

    response = await chat_service.process_chat(
        ChatRequest(message=message), store=InMemorySessionStore()
    )

    assert calls == [("2026-06-06", "2027-10-04")]
    assert response.tools_used == ["get_my_request_status"]
    assert len(response.blocks[0].rows) == 3
    assert {row["status"] for row in response.blocks[0].rows} == {
        "Approved", "Pending"
    }


@pytest.mark.asyncio
async def test_quick_list_uses_request_history_only_with_recent_request_context(
    monkeypatch,
) -> None:
    calls = _install_unified(monkeypatch)
    store = InMemorySessionStore()
    first = await chat_service.process_chat(
        ChatRequest(message="show my requests"), store=store
    )

    response = await chat_service.process_chat(
        ChatRequest(message="show my quick list", session_id=first.session_id),
        store=store,
    )

    assert len(calls) == 2
    assert response.tools_used == ["get_my_request_status"]
    assert len(response.blocks[0].rows) == 3
    assert "no pending requests" not in response.message.casefold()


@pytest.mark.asyncio
async def test_quick_list_without_request_context_clarifies_without_model_or_api(
    monkeypatch,
) -> None:
    async def forbidden(*args, **kwargs):
        raise AssertionError("ambiguous quick list must not call a model or ResourcePlus")

    monkeypatch.setattr(chat_service, "run_agent", forbidden)
    monkeypatch.setattr(fast_reads, "get_my_request_status", forbidden)
    response = await chat_service.process_chat(
        ChatRequest(message="show my quick list"), store=InMemorySessionStore()
    )

    assert response.message == "Do you mean your request list?"
    assert response.tools_used == []
    assert response.blocks == []


@pytest.mark.asyncio
async def test_two_structured_pending_requests_cannot_render_as_none(monkeypatch) -> None:
    payload = {
        "merged_requests": [
            {
                "request_kind": "absence",
                "raw_status": "Pending",
                "record": {
                    "dateFrom": "2026-10-15",
                    "dayType": "Live leave type",
                    "status": "Pending",
                },
            },
            {
                "request_kind": "exceptional_entry",
                "raw_status": "Not Approved",
                "record": {
                    "entryTime": "2026-09-02T09:03:00",
                    "reasonName": "Outside Work",
                    "status": "Not Approved",
                },
            },
        ]
    }

    async def requests(*args, **kwargs):
        return payload

    async def no_model(*args, **kwargs):
        raise AssertionError("structured request facts must bypass the model")

    monkeypatch.setattr(fast_reads, "get_my_request_status", requests)
    monkeypatch.setattr(chat_service, "run_agent", no_model)
    response = await chat_service.process_chat(
        ChatRequest(message="show my pending requests"),
        store=InMemorySessionStore(),
    )

    assert response.message == "You have 2 requests still waiting for approval."
    assert len(response.blocks[0].rows) == 2
    assert all(row["status"] == "Pending" for row in response.blocks[0].rows)
    assert "no pending requests" not in response.message.casefold()


@pytest.mark.asyncio
async def test_show_latest_request_returns_latest_structured_row(monkeypatch) -> None:
    _install_unified(monkeypatch)
    response = await chat_service.process_chat(
        ChatRequest(message="show my latest request"), store=InMemorySessionStore()
    )

    assert len(response.blocks[0].rows) == 1
    assert response.blocks[0].rows[0]["date"] == "2026-10-09"
    assert response.blocks[0].rows[0]["detail"] == "Compensatory Leave"
