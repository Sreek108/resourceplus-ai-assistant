from __future__ import annotations

from datetime import date

import pytest

from app.ai import actions, conversation
from app.ai.sessions import InMemorySessionStore
from app.audit import reset_interaction_audit, start_interaction_audit
from app.identity import RequestIdentity, bind_request_identity, reset_request_identity
from app.models.schemas import (
    ActionsBlock,
    BlockColumn,
    ChatRequest,
    ChatResponse,
    KeyValueBlock,
    ListBlock,
    StatCardsBlock,
    TableBlock,
)
from app.services import chat as chat_service, fast_reads
from app.services.fast_reads import _balance_message, _message_for


def _day(day_number: int) -> actions.LessHoursDay:
    return actions.LessHoursDay(
        date(2026, 9, day_number), "Present", "09:00", "17:58",
        "07:58", "00:02", "eligible", {},
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "language"),
    [
        ("Show my less hours", "en"),
        ("اعرض الساعات الناقصة", "ar"),
    ],
)
async def test_less_hours_copy_uses_only_resolved_rows_and_keeps_ui_facts(
    monkeypatch, message, language
) -> None:
    available = tuple(_day(number) for number in (2, 16, 23, 27))
    covered = tuple(_day(number) for number in (6, 8, 14))
    existing = _day(24)
    inspection = actions.LessHoursInspection(
        days=(*available, *covered, existing),
        eligible_days=available,
        duplicate_guards=(
            *(
                actions.ExceptionalEntryGuard(day.attendance_date, "Approved", "approved")
                for day in covered
            ),
            actions.ExceptionalEntryGuard(
                existing.attendance_date, "Not Approved", "ambiguous_not_approved"
            ),
            # An approved request with no short-hour attendance row is not narrated.
            actions.ExceptionalEntryGuard(date(2026, 9, 5), "Approved", "approved"),
        ),
    )

    async def inspect(*args, **kwargs):
        return inspection

    async def no_balance(*args, **kwargs):
        raise AssertionError("A multiple-date read must not fetch a generic balance")

    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", no_balance)
    response = await chat_service.process_chat(
        ChatRequest(message=message), store=InMemorySessionStore()
    )

    table = next(block for block in response.blocks if block.type == "table")
    action_block = next(block for block in response.blocks if block.type == "actions")
    assert response.language == language
    assert [row["date"] for row in table.rows] == [
        *[day.attendance_date.isoformat() for day in available],
        *[day.attendance_date.isoformat() for day in covered],
        existing.attendance_date.isoformat(),
    ]
    assert [row["less"] for row in table.rows] == ["00:02"] * 8
    assert [row["action"] for row in table.rows[4:7]] == [
        "Approved exception" if language == "en" else "استثناء معتمد"
    ] * 3
    assert len(action_block.actions) == 4
    assert [action.value.rsplit(" ", 1)[-1] for action in action_block.actions] == [
        day.attendance_date.isoformat() for day in available
    ]
    assert "2026-09-05" not in response.message
    assert "ResourcePlus already has" not in response.message
    assert "exceptional-entry" not in response.message
    assert len(response.speech_message) < len(response.message)
    if language == "en":
        assert response.message.startswith(
            "You have 4 September attendance gaps left to fix: Sep 2, 16, 23, and 27."
        )
        assert "3 other short-hour days are covered" in response.message
        assert "1 other short-hour day already has a request" in response.message
        assert response.speech_message.count(".") <= 2
    else:
        assert response.message.startswith("عندك 4 أيام حضور تحتاج تصحيح")
        assert "2026-09-02" in response.message
        assert "طلبات موجودة" in response.message


@pytest.mark.asyncio
async def test_fix_the_ordinal_continues_last_verified_less_hours_read(monkeypatch) -> None:
    days = (_day(2), _day(16))
    reads: list[tuple[date, date]] = []

    async def inspect(start, end, **kwargs):
        reads.append((start, end))
        selected = tuple(day for day in days if start <= day.attendance_date <= end)
        return actions.LessHoursInspection(selected, selected)

    async def balance(*args, **kwargs):
        return {"hasPolicy": False}

    async def reasons(*args, **kwargs):
        return [{"reasonID": "live-id", "reasonName": "Traffic"}]

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)
    monkeypatch.setattr(conversation, "cached_exception_reasons", reasons)
    store = InMemorySessionStore()

    read = await chat_service.process_chat(
        ChatRequest(message="Show my less hours"), store=store
    )
    assert store.get_pending_action(read.session_id)[0] is None
    assert store.get_conversation_draft(read.session_id) is None
    follow_up = await chat_service.process_chat(
        ChatRequest(message="fix the 16th", session_id=read.session_id), store=store
    )

    assert reads == [
        (date(2026, 9, 1), date(2026, 9, 30)),
        (date(2026, 9, 16), date(2026, 9, 16)),
    ]
    assert follow_up.needs_reason is True
    assert "00:02 short on Sep 16" in follow_up.message
    assert "What was the reason?" in follow_up.message
    assert store.get_pending_action(read.session_id)[0] is None


def test_no_pending_approvals_and_buffer_are_personal_and_grounded(monkeypatch) -> None:
    empty = TableBlock(
        title="Pending approvals",
        columns=[BlockColumn(key="status", label="Status")],
        rows=[],
    )
    display, speech = _message_for("approvals", [empty], "en")
    assert display == speech == "You're all caught up — there's nothing waiting for your approval."
    display_ar, speech_ar = _message_for("approvals", [empty], "ar")
    assert display_ar == speech_ar == "أمورك تمام — ما فيه طلبات تنتظر موافقتك."

    monkeypatch.setattr("app.services.fast_reads.resourceplus_today", lambda: date(2026, 9, 30))
    balance = {
        "hasPolicy": True, "limitType": 2, "remaining": 59,
        "periodStart": "2026-09-28", "periodEnd": "2026-10-04",
    }
    assert _balance_message(balance, "en") == (
        "You have 59 minutes of buffer time left this week.",
        "You have 59 minutes of buffer time left this week.",
    )
    assert _balance_message(balance, "ar") == (
        "باقي لك 59 دقيقة من وقت السماح هالأسبوع.",
        "باقي لك 59 دقيقة من وقت السماح هالأسبوع.",
    )


@pytest.mark.asyncio
async def test_existing_request_lookup_failure_does_not_claim_attendance_is_complete(
    monkeypatch,
) -> None:
    inspection = actions.LessHoursInspection(
        days=(_day(16), _day(23)),
        eligible_days=(),
        exceptional_entries_checked=False,
    )

    async def inspect(*args, **kwargs):
        return inspection

    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    response = await chat_service.process_chat(
        ChatRequest(message="Show my less hours"), store=InMemorySessionStore()
    )

    table = next(block for block in response.blocks if block.type == "table")
    assert len(table.rows) == 2
    assert {row["action"] for row in table.rows} == {"Correction unavailable"}
    assert all(block.type != "actions" for block in response.blocks)
    assert "couldn't check your existing requests" in response.message
    assert "no attendance gaps" not in response.message


@pytest.mark.parametrize("language", ["en", "ar"])
@pytest.mark.parametrize("auto_approved", [True, False])
def test_correction_completion_speech_uses_verified_state_only(
    language, auto_approved
) -> None:
    result = {
        "success": True,
        "isAutoApproved": auto_approved,
        "requestedMinutes": 17,
        "remaining": 103,
        "resetsOn": "2026-10-05",
        "message": "API update completed",
    }
    display = chat_service._from_summary_result_message(
        result, language, success=True
    )
    speech = chat_service._from_summary_speech_message(
        result, language, success=True
    )
    assert display is not None
    assert len(speech) < len(display)
    assert "API update completed" not in display + speech
    assert "103" in display and "2026-10-05" in display
    assert "103" not in speech
    if language == "en":
        assert ("approved automatically" in speech) is auto_approved
        assert ("manager for approval" in speech) is (not auto_approved)
    else:
        assert ("تلقائياً" in speech) is auto_approved
        assert ("لموافقة مديرك" in speech) is (not auto_approved)


@pytest.mark.asyncio
async def test_verified_hod_approval_uses_trusted_name_in_arabic(monkeypatch) -> None:
    store = InMemorySessionStore()
    identity = RequestIdentity("supervisor@example.com", "Universal")
    token = bind_request_identity(identity)
    try:
        session_id = store.ensure_session()
        pending = store.create_pending_action(
            session_id,
            action_type="approve_supervisor_request",
            validated_arguments={
                "request_id": "verified-id",
                "request_type": "Absence",
                "status": 1,
                "employee_name": "Talal",
                "detail": "Work From Home",
            },
            summary="Confirm approval",
            language="ar",
        )
    finally:
        reset_request_identity(token)

    async def verified(*args, **kwargs):
        return {"success": True, "_approval_verification": "verified"}

    monkeypatch.setattr(chat_service, "execute_pending_action", verified)
    response = await chat_service.process_chat(
        ChatRequest(
            message="نعم",
            session_id=session_id,
            confirmation_id=pending.confirmation_id,
            email=identity.email,
            instance=identity.instance,
        ),
        store=store,
    )
    assert response.success is True
    assert response.message == "تمت الموافقة على طلب Work From Home الخاص بـTalal."
    assert response.speech_message == response.message
    assert "verified-id" not in response.message


@pytest.mark.asyncio
async def test_accepted_hod_rejection_is_not_described_as_an_approval(monkeypatch) -> None:
    store = InMemorySessionStore()
    identity = RequestIdentity("supervisor@example.com", "Universal")
    token = bind_request_identity(identity)
    try:
        session_id = store.ensure_session()
        pending = store.create_pending_action(
            session_id,
            action_type="approve_supervisor_request",
            validated_arguments={
                "request_id": "request-id",
                "request_type": "Absence",
                "status": 2,
                "employee_name": "Talal",
                "detail": "Work From Home",
            },
            summary="Confirm rejection",
            language="en",
        )
    finally:
        reset_request_identity(token)

    async def accepted(*args, **kwargs):
        return {"success": True}

    monkeypatch.setattr(chat_service, "execute_pending_action", accepted)
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
    assert response.success is True
    assert response.message.startswith("The rejection was accepted")
    assert "approved" not in response.message


def test_pending_approval_verification_copy_never_claims_completed_approval() -> None:
    for language in ("en", "ar"):
        display, speech = chat_service._deterministic_action_result(
            "approve_supervisor_request", "approval_pending_verification", language
        )
        assert "ResourcePlus" not in display + speech
        if language == "en":
            assert "pending" in display
            assert "I won't submit it again" in speech
        else:
            assert "انتظار الموافقة" in display
            assert "ما راح أعيد إرساله" in speech


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "Do I have anything to fix?",
        "Show missing hours",
        "Show my less hours",
        "Show short hours",
        "Show attendance gaps",
        "Anything I need to correct?",
        "What attendance issues do I have?",
        "Show last month missing hours",
        "Show previous month less hours",
    ],
)
async def test_attendance_gap_phrases_share_resolved_deterministic_workflow(
    monkeypatch, message
) -> None:
    available = _day(16)
    approved = _day(6)
    existing = _day(24)
    inspection = actions.LessHoursInspection(
        days=(available, approved, existing),
        eligible_days=(available,),
        duplicate_guards=(
            actions.ExceptionalEntryGuard(approved.attendance_date, "Approved", "approved"),
            actions.ExceptionalEntryGuard(
                existing.attendance_date, "Not Approved", "ambiguous_not_approved"
            ),
        ),
    )
    calls = []

    async def inspect(*args, **kwargs):
        calls.append(args[:2])
        return inspection

    async def no_balance(*args, **kwargs):
        return {"hasPolicy": False}

    async def no_model(*args, **kwargs):
        raise AssertionError("attendance-gap reads must not use the model")

    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", no_balance)
    monkeypatch.setattr(chat_service, "run_agent", no_model)
    response = await chat_service.process_chat(
        ChatRequest(message=message), store=InMemorySessionStore()
    )

    table = next(block for block in response.blocks if block.type == "table")
    action_block = next(block for block in response.blocks if block.type == "actions")
    assert calls
    assert [row["action"] for row in table.rows] == [
        "Correction available", "Approved exception", "Existing request"
    ]
    assert [action.value for action in action_block.actions] == [
        "Correct my less hours on 2026-09-16"
    ]
    assert "1 September attendance gap left to fix" in response.message


@pytest.mark.asyncio
async def test_explicit_gap_read_overrides_awaiting_date_draft(monkeypatch) -> None:
    store = InMemorySessionStore()
    first = await chat_service.process_chat(
        ChatRequest(message="I need to correct my attendance."), store=store
    )
    assert "Which date" in first.message
    assert store.get_conversation_draft(first.session_id) is not None

    async def inspect(start, end, **kwargs):
        assert (start, end) == (date(2026, 9, 1), date(2026, 9, 30))
        day = _day(16)
        return actions.LessHoursInspection((day,), (day,))

    async def balance(*args, **kwargs):
        return {"hasPolicy": False}

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 10, 4))
    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)
    response = await chat_service.process_chat(
        ChatRequest(
            message="Show missing hours of last month.",
            session_id=first.session_id,
        ),
        store=store,
    )
    assert "Sep 16" in response.message
    assert "more than one correction candidate" not in response.message


@pytest.mark.asyncio
async def test_natural_date_correction_uses_guarded_less_hours_path(monkeypatch) -> None:
    approved = _day(6)
    inspection = actions.LessHoursInspection(
        days=(approved,),
        eligible_days=(),
        duplicate_guards=(
            actions.ExceptionalEntryGuard(approved.attendance_date, "Approved", "approved"),
        ),
    )

    async def inspect(start, end, **kwargs):
        assert (start, end) == (date(2026, 9, 6), date(2026, 9, 6))
        return inspection

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 10, 4))
    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    response = await chat_service.process_chat(
        ChatRequest(message="correct Sep 6"), store=InMemorySessionStore()
    )
    assert response.message == (
        "Sep 6 is already covered by an approved attendance correction, so "
        "there's nothing else to submit."
    )
    assert response.requires_confirmation is False
    assert all(block.type != "actions" for block in response.blocks)


@pytest.mark.asyncio
async def test_recent_submitted_request_context_overrides_raw_not_approved_copy(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    store.save_trusted_result(
        session_id,
        message="I've sent your attendance correction to your manager for approval.",
        tools_used=("create_exceptional_entry_from_summary",),
        language="en",
        recent_request_date="2026-09-16",
        recent_request_detail="Work From Home",
        recent_request_state="submitted_for_approval",
    )
    calls = []

    async def requests(start, end, **kwargs):
        calls.append((start, end))
        return {
            "merged_requests": [
                {
                    "request_kind": "ExceptionalEntry",
                    "raw_status": "Not Approved",
                    "record": {
                        "date": "2026-09-16",
                        "reasonName": "Work From Home",
                        "status": "Not Approved",
                    },
                }
            ]
        }

    monkeypatch.setattr(fast_reads, "get_my_request_status", requests)
    response = await chat_service.process_chat(
        ChatRequest(message="show my request", session_id=session_id), store=store
    )
    assert calls and calls[0][0] < "2026-09-16" < calls[0][1]
    assert response.message == (
        "Your Sep 16 Work From Home attendance correction is still awaiting "
        "manager approval."
    )
    assert "no pending requests" not in response.message.casefold()
    table = next(block for block in response.blocks if block.type == "table")
    assert table.rows[0]["status"] == "Pending"


@pytest.mark.asyncio
async def test_recent_submitted_request_accepts_newer_approved_state(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    store.save_trusted_result(
        session_id,
        message="Submitted",
        tools_used=("create_exceptional_entry_from_summary",),
        language="en",
        recent_request_date="2026-09-16",
        recent_request_detail="Work From Home",
        recent_request_state="submitted_for_approval",
    )

    async def requests(*args, **kwargs):
        return {
            "merged_requests": [{
                "request_kind": "ExceptionalEntry",
                "raw_status": "Approved",
                "record": {
                    "date": "2026-09-16",
                    "reasonName": "Work From Home",
                    "status": "Approved",
                },
            }]
        }

    monkeypatch.setattr(fast_reads, "get_my_request_status", requests)
    response = await chat_service.process_chat(
        ChatRequest(message="show my request", session_id=session_id), store=store
    )
    assert response.message == (
        "Your Sep 16 Work From Home attendance correction has been approved."
    )
    assert store.get_trusted_result(session_id).recent_request_state == "approved"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "follow_up",
    ["show my request", "what happened to my request"],
)
async def test_recent_leave_request_phrases_converge_without_model(
    monkeypatch, follow_up
) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    store.save_trusted_result(
        session_id,
        message="Submitted",
        tools_used=("book_day_type",),
        language="en",
        recent_request_category="leave",
        recent_request_date="2026-10-09",
        recent_request_detail="Compensatory Leave",
        recent_request_state="submitted_for_approval",
    )

    async def requests(start, end, **kwargs):
        return {
            "merged_requests": [{
                "request_kind": "absence",
                "raw_status": "Not Approved",
                "record": {"status": "Not Approved"},
            }]
        }

    async def no_model(*args, **kwargs):
        raise AssertionError("recent request reads must use the deterministic path")

    monkeypatch.setattr(fast_reads, "get_my_request_status", requests)
    monkeypatch.setattr(chat_service, "run_agent", no_model)
    response = await chat_service.process_chat(
        ChatRequest(message=follow_up, session_id=session_id), store=store
    )

    assert response.message == (
        "Your Oct 9 Compensatory Leave request is still awaiting manager approval."
    )
    table = next(block for block in response.blocks if block.type == "table")
    assert table.rows[0]["type"] == "Compensatory Leave"
    assert table.rows[0]["date"] == "2026-10-09"
    assert table.rows[0]["detail"] == "Compensatory Leave"


@pytest.mark.asyncio
async def test_recent_leave_approved_refresh_supersedes_submission_state(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    store.save_trusted_result(
        session_id,
        message="Submitted",
        tools_used=("book_day_type",),
        language="en",
        recent_request_category="leave",
        recent_request_date="2026-10-09",
        recent_request_detail="Dynamic Leave Type",
        recent_request_state="submitted_for_approval",
    )

    async def requests(*args, **kwargs):
        return {
            "merged_requests": [{
                "request_kind": "absence",
                "raw_status": "Approved",
                "record": {
                    "dateFrom": "2026-10-09",
                    "dayType": "Dynamic Leave Type",
                    "status": "Approved",
                },
            }]
        }

    monkeypatch.setattr(fast_reads, "get_my_request_status", requests)
    response = await chat_service.process_chat(
        ChatRequest(message="is my leave approved", session_id=session_id), store=store
    )
    assert response.message == (
        "Your Dynamic Leave Type request for Oct 9 has been approved."
    )
    assert store.get_trusted_result(session_id).recent_request_state == "approved"


@pytest.mark.asyncio
async def test_arabic_recent_leave_status_uses_same_authoritative_refresh(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    store.save_trusted_result(
        session_id,
        message="تم الإرسال",
        tools_used=("book_day_type",),
        language="ar",
        recent_request_category="leave",
        recent_request_date="2026-10-09",
        recent_request_detail="إجازة تطوع",
        recent_request_state="submitted_for_approval",
    )

    async def requests(*args, **kwargs):
        return {"merged_requests": []}

    async def no_model(*args, **kwargs):
        raise AssertionError("Arabic recent request reads must not call a model")

    monkeypatch.setattr(fast_reads, "get_my_request_status", requests)
    monkeypatch.setattr(chat_service, "run_agent", no_model)
    response = await chat_service.process_chat(
        ChatRequest(message="وش صار على طلبي", session_id=session_id), store=store
    )
    assert response.language == "ar"
    assert "إجازة تطوع" in response.message
    assert "2026-10-09" in response.message
    assert "بانتظار" in response.message


@pytest.mark.asyncio
async def test_confirmed_leave_submission_saves_safe_recent_context(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    pending = store.create_pending_action(
        session_id,
        action_type="book_day_type",
        validated_arguments={
            "date_from": "2026-10-09",
            "date_to": "2026-10-09",
            "day_type_id": 77,
            "day_type_name": "Compensatory Leave",
        },
        summary="Submit Compensatory Leave?",
        language="en",
    )

    async def execute(*args, **kwargs):
        return {"success": True, "message": "submitted"}

    async def revalidate(*args, **kwargs):
        return None

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    monkeypatch.setattr(chat_service, "revalidate_pending_action", revalidate)
    response = await chat_service.process_chat(
        ChatRequest(
            message="yes",
            session_id=session_id,
            confirmation_id=pending.confirmation_id,
        ),
        store=store,
    )
    trusted = store.get_trusted_result(session_id)
    assert response.message == (
        "Your Compensatory Leave request for Oct 9 has been sent to your manager "
        "for approval."
    )
    assert trusted is not None
    assert trusted.recent_request_category == "leave"
    assert trusted.recent_request_detail == "Compensatory Leave"
    assert trusted.recent_request_date == "2026-10-09"
    assert trusted.recent_request_state == "submitted_for_approval"
    assert "day_type_id" not in trusted.__dataclass_fields__
    assert trusted.recent_request_detail != "77"
    assert all(correlation.detail != "77" for correlation in trusted.request_correlations)


@pytest.mark.asyncio
async def test_gap_follow_up_reuses_remembered_attendance_period(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    store.save_trusted_result(
        session_id,
        message="September attendance",
        tools_used=("get_attendance_summary",),
        language="en",
        attendance_period=("2026-09-01", "2026-09-30"),
        attendance_period_label="September",
        attendance_period_source="explicit_user_period",
    )

    async def inspect(start, end, **kwargs):
        assert (start, end) == (date(2026, 9, 1), date(2026, 9, 30))
        day = _day(23)
        return actions.LessHoursInspection((day,), (day,))

    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    async def balance(*args, **kwargs):
        return None

    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)
    response = await chat_service.process_chat(
        ChatRequest(message="Do I have anything to fix?", session_id=session_id),
        store=store,
    )
    assert response.blocks[0].title == "Attendance gaps"
    assert response.blocks[0].rows[0]["date"] == "2026-09-23"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "expected_title"),
    [
        ("In my previous attendance, do I have an attendance gap?", "Attendance gaps"),
        ("في حضوري السابق، هل عندي فجوات حضور؟", "فجوات الحضور"),
    ],
)
async def test_previous_attendance_gap_wording_reuses_period_in_both_languages(
    monkeypatch, message, expected_title
) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    store.save_trusted_result(
        session_id,
        message="September attendance",
        tools_used=("get_attendance_summary",),
        language="en",
        attendance_period=("2026-09-01", "2026-09-30"),
        attendance_period_label="September",
        attendance_period_source="explicit_user_period",
    )

    async def inspect(start, end, **kwargs):
        assert (start, end) == (date(2026, 9, 1), date(2026, 9, 30))
        day = _day(23)
        return actions.LessHoursInspection((day,), (day,))

    async def balance(*args, **kwargs):
        return None

    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)
    response = await chat_service.process_chat(
        ChatRequest(message=message, session_id=session_id), store=store
    )
    assert response.blocks[0].title == expected_title


@pytest.mark.asyncio
async def test_explicit_named_month_overrides_remembered_attendance_period(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    store.save_trusted_result(
        session_id,
        message="September attendance",
        tools_used=("get_attendance_summary",),
        language="en",
        attendance_period=("2026-09-01", "2026-09-30"),
    )

    async def attendance(start, end, **kwargs):
        assert (start, end) == ("2026-10-01", "2026-10-04")
        return []

    monkeypatch.setattr(fast_reads, "get_attendance_summary", attendance)
    response = await chat_service.process_chat(
        ChatRequest(message="What about October?", session_id=session_id), store=store
    )
    assert response.tools_used == ["get_attendance_summary"]
    assert store.get_trusted_result(session_id).attendance_period == (
        "2026-10-01", "2026-10-04"
    )


@pytest.mark.asyncio
async def test_comma_date_correction_uses_less_hours_resolver(monkeypatch) -> None:
    selected = _day(23)

    async def inspect(start, end, **kwargs):
        assert (start, end) == (date(2026, 9, 23), date(2026, 9, 23))
        return actions.LessHoursInspection((selected,), (selected,))

    async def balance(*args, **kwargs):
        return {"hasPolicy": False}

    async def reasons(*args, **kwargs):
        return [{"reasonID": 1, "reasonName": "Traffic"}]

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 10, 4))
    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)
    monkeypatch.setattr(conversation, "cached_exception_reasons", reasons)
    response = await chat_service.process_chat(
        ChatRequest(message="Correct, September 23."), store=InMemorySessionStore()
    )
    assert response.needs_reason is True
    assert "reason" in response.message.casefold()


@pytest.mark.asyncio
async def test_single_approval_natural_voice_phrase_is_one_fresh_read_and_no_model(
    monkeypatch,
) -> None:
    rows = [{
        "requestId": "real-request",
        "requestType": "Absence",
        "employeeName": "Talal Sabbagh",
        "detail": "Compensatory Leave",
        "dateFrom": "2026-10-09",
        "status": "Pending",
    }]
    calls = []

    async def pending(*args, **kwargs):
        calls.append("read")
        return rows

    async def no_model(*args, **kwargs):
        raise AssertionError("single approval continuation must not call a model")

    monkeypatch.setattr(fast_reads, "get_pending_approvals", pending)
    monkeypatch.setattr(actions, "get_pending_approvals", pending)
    monkeypatch.setattr(chat_service, "run_agent", no_model)
    store = InMemorySessionStore()
    shown = await chat_service.process_chat(
        ChatRequest(message="show pending approvals"), store=store
    )
    audit, token = start_interaction_audit(input_mode="voice")
    try:
        response = await chat_service.process_chat(
            ChatRequest(
                message="approve the compensate review",
                session_id=shown.session_id,
            ),
            store=store,
        )
    finally:
        reset_interaction_audit(token)
    assert calls == ["read", "read"]
    assert response.message == (
        "Approve Talal Sabbagh's Compensatory Leave request for Oct 9?"
    )
    assert response.requires_confirmation is True
    assert audit.model_requests == 0
    assert audit.error_category is None


@pytest.mark.asyncio
async def test_one_shown_approval_plus_bare_approve_prepares_individual(monkeypatch) -> None:
    rows = [{
        "requestId": "real-request",
        "requestType": "Absence",
        "employeeName": "Talal",
        "detail": "Work From Home",
        "dateFrom": "2026-09-16",
        "status": "Pending",
    }]

    async def pending(*args, **kwargs):
        return rows

    monkeypatch.setattr(fast_reads, "get_pending_approvals", pending)
    monkeypatch.setattr(actions, "get_pending_approvals", pending)
    store = InMemorySessionStore()
    shown = await chat_service.process_chat(
        ChatRequest(message="show pending approvals"), store=store
    )
    response = await chat_service.process_chat(
        ChatRequest(message="approve", session_id=shown.session_id), store=store
    )
    pending_action, _ = store.get_pending_action(shown.session_id)
    assert pending_action is not None
    assert pending_action.action_type == "approve_supervisor_request"
    assert pending_action.validated_arguments["request_id"] == "real-request"
    assert response.message == "Approve Talal's Work From Home request for Sep 16?"
    assert response.requires_confirmation is True


@pytest.mark.asyncio
async def test_multiple_shown_approvals_plus_bare_approve_asks_which(monkeypatch) -> None:
    rows = [
        {"requestId": "1", "requestType": "Absence", "employeeName": "Talal", "detail": "WFH"},
        {"requestId": "2", "requestType": "ExceptionEntry", "employeeName": "Reem", "detail": "Traffic"},
    ]

    async def pending(*args, **kwargs):
        return rows

    monkeypatch.setattr(fast_reads, "get_pending_approvals", pending)
    monkeypatch.setattr(actions, "get_pending_approvals", pending)
    store = InMemorySessionStore()
    shown = await chat_service.process_chat(
        ChatRequest(message="show pending approvals"), store=store
    )
    response = await chat_service.process_chat(
        ChatRequest(message="approve", session_id=shown.session_id), store=store
    )
    assert "more than one matching pending request" in response.message
    assert response.requires_confirmation is False
    assert store.get_pending_action(shown.session_id)[0] is None


@pytest.mark.asyncio
async def test_approve_all_requires_explicit_bulk_language(monkeypatch) -> None:
    rows = [
        {"requestId": "1", "requestType": "Absence", "employeeName": "Talal", "detail": "WFH"},
        {"requestId": "2", "requestType": "ExceptionEntry", "employeeName": "Reem", "detail": "Traffic"},
    ]

    async def pending(*args, **kwargs):
        return rows

    monkeypatch.setattr(fast_reads, "get_pending_approvals", pending)
    monkeypatch.setattr(actions, "get_pending_approvals", pending)
    store = InMemorySessionStore()
    shown = await chat_service.process_chat(
        ChatRequest(message="show pending approvals"), store=store
    )
    response = await chat_service.process_chat(
        ChatRequest(message="approve all", session_id=shown.session_id), store=store
    )
    pending_action, _ = store.get_pending_action(shown.session_id)
    assert pending_action is not None
    assert pending_action.action_type == "approve_all_requests"
    assert pending_action.validated_arguments["count"] == 2
    assert response.requires_confirmation is True


def test_natural_cancellation_and_confirmation_speech_copy() -> None:
    assert chat_service._confirmation_message("cancelled", "en") == "Okay, I won't submit it."
    assert chat_service._confirmation_message("cancelled", "ar") == "حسنًا، ما راح أرسله."
    assert "ResourcePlus operation was approved" not in str(chat_service.ACTION_RESULT_MESSAGES)


@pytest.mark.asyncio
async def test_arabic_natural_gap_question_uses_resolved_state(monkeypatch) -> None:
    day = _day(16)

    async def inspect(*args, **kwargs):
        return actions.LessHoursInspection((day,), (day,))

    async def balance(*args, **kwargs):
        return {"hasPolicy": False}

    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)
    response = await chat_service.process_chat(
        ChatRequest(message="هل عندي شيء أصلحه؟"), store=InMemorySessionStore()
    )
    assert response.language == "ar"
    assert "2026-09-16" in response.message
    actions_block = next(block for block in response.blocks if block.type == "actions")
    assert len(actions_block.actions) == 1


@pytest.mark.asyncio
async def test_arabic_single_approval_follow_up_targets_individual(monkeypatch) -> None:
    rows = [{
        "requestId": "real-request",
        "requestType": "Absence",
        "employeeName": "Talal",
        "detail": "Work From Home",
        "dateFrom": "2026-09-16",
        "status": "Pending",
    }]

    async def pending(*args, **kwargs):
        return rows

    monkeypatch.setattr(fast_reads, "get_pending_approvals", pending)
    monkeypatch.setattr(actions, "get_pending_approvals", pending)
    store = InMemorySessionStore()
    shown = await chat_service.process_chat(
        ChatRequest(message="اعرض الموافقات المعلقة"), store=store
    )
    response = await chat_service.process_chat(
        ChatRequest(message="وافق", session_id=shown.session_id), store=store
    )
    pending_action, _ = store.get_pending_action(shown.session_id)
    assert pending_action is not None
    assert pending_action.action_type == "approve_supervisor_request"
    assert response.language == "ar"
    assert "Talal" in response.message
    assert "2026-09-16" in response.message or "Sep 16" in response.message


@pytest.mark.asyncio
async def test_bulk_approval_copy_requires_verified_completion(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    pending = store.create_pending_action(
        session_id,
        action_type="approve_all_requests",
        validated_arguments={
            "status": 1,
            "request_type": None,
            "scope": "all types",
            "count": 3,
        },
        summary="Approve all 3 pending requests?",
        language="en",
    )

    async def verified(*args, **kwargs):
        return {
            "success": True,
            "_bulk_approval_verification": "verified",
            "_verified_count": 3,
        }

    monkeypatch.setattr(chat_service, "execute_pending_action", verified)
    response = await chat_service.process_chat(
        ChatRequest(
            message="yes",
            session_id=session_id,
            confirmation_id=pending.confirmation_id,
        ),
        store=store,
    )
    assert response.message == "Done — I approved all 3 pending requests."
    assert "operation" not in response.message.casefold()


@pytest.mark.asyncio
async def test_verified_empty_gap_context_guides_correction_start_without_draft() -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    store.save_trusted_result(
        session_id,
        message="No missing hours.",
        tools_used=("get_attendance_summary", "get_exceptional_entry_requests"),
        language="en",
        correction_dates=(),
        attendance_period=("2026-10-01", "2026-10-04"),
    )

    outcome = await conversation.continue_conversation(
        "I need to correct my attendance",
        lang=1,
        session_id=session_id,
        language="en",
        store=store,
    )

    assert outcome is not None
    assert outcome.message == (
        "I don't see any attendance gaps for October so far. If you mean another "
        "month or a specific date, tell me which one."
    )
    assert store.get_conversation_draft(session_id) is None


@pytest.mark.asyncio
async def test_awaiting_date_and_reason_render_in_current_language() -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    store.save_conversation_draft(
        session_id,
        intent="less_hours_correction",
        slots={},
        language="en",
    )

    awaiting_date = await conversation.continue_conversation(
        "أحتاج إلى تصحيح حضوري",
        lang=1,
        session_id=session_id,
        language="ar",
        store=store,
    )
    assert awaiting_date is not None
    assert awaiting_date.message == "ما تاريخ الساعات الناقصة؟"

    store.save_conversation_draft(
        session_id,
        intent="less_hours_correction",
        slots={
            "date": "2026-09-16",
            "reason_options": "Traffic\x1fWork From Home",
        },
        language="en",
    )
    awaiting_reason = await conversation.continue_conversation(
        "سبب مختلف",
        lang=1,
        session_id=session_id,
        language="ar",
        store=store,
    )
    assert awaiting_reason is not None
    assert awaiting_reason.needs_reason is True
    assert awaiting_reason.message.startswith("اختر سبباً")


@pytest.mark.asyncio
async def test_pending_confirmation_rerenders_after_explicit_language_switch(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    pending = store.create_pending_action(
        session_id,
        action_type="book_day_type",
        validated_arguments={
            "date_from": "2026-10-06",
            "date_to": "2026-10-06",
            "day_type_name": "Work From Home",
        },
        summary="Submit Work From Home on 6 October 2026?",
        language="en",
    )

    async def other(*args, **kwargs):
        return "OTHER"

    monkeypatch.setattr(chat_service, "classify_confirmation_intent", other)
    response = await chat_service.process_chat(
        ChatRequest(message="خلينا نكمل بالعربي", session_id=session_id),
        detected_language="ar",
        store=store,
    )

    assert response.language == "ar"
    assert response.requires_confirmation is True
    assert response.confirmation_id == pending.confirmation_id
    assert "Work From Home" in response.message
    assert store.get_pending_action(session_id)[0] == pending


def test_empty_structured_blocks_are_removed_but_message_is_preserved() -> None:
    response = ChatResponse(
        success=True,
        message="There are no results.",
        language="en",
        session_id="empty-blocks",
        blocks=[
            TableBlock(title="Empty", columns=[], rows=[]),
            KeyValueBlock(title="Empty", items=[]),
            StatCardsBlock(title="Empty", items=[]),
            ListBlock(title="Empty", items=[]),
            ActionsBlock(title="Empty", actions=[]),
        ],
    )

    assert response.message == "There are no results."
    assert response.blocks == []


@pytest.mark.asyncio
async def test_zero_row_gap_read_has_no_empty_table(monkeypatch) -> None:
    async def inspect(*args, **kwargs):
        return actions.LessHoursInspection((), ())

    monkeypatch.setattr(conversation, "inspect_less_hours_period", inspect)
    response = await chat_service.process_chat(
        ChatRequest(message="show missing hours"),
        store=InMemorySessionStore(),
    )

    assert "no attendance gaps" in response.message.casefold()
    assert response.blocks == []


@pytest.mark.asyncio
async def test_zero_row_request_read_has_no_empty_table(monkeypatch) -> None:
    async def requests(*args, **kwargs):
        return {"merged_requests": []}

    monkeypatch.setattr(fast_reads, "get_my_request_status", requests)
    response = await chat_service.process_chat(
        ChatRequest(message="show my requests"),
        store=InMemorySessionStore(),
    )

    assert response.message
    assert response.blocks == []


@pytest.mark.asyncio
async def test_pending_cancellation_uses_explicitly_switched_language(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    store.create_pending_action(
        session_id,
        action_type="book_day_type",
        validated_arguments={
            "date_from": "2026-10-06",
            "date_to": "2026-10-06",
            "day_type_name": "Work From Home",
        },
        summary="Submit Work From Home on 6 October 2026?",
        language="en",
    )

    async def reject(*args, **kwargs):
        return "REJECT"

    monkeypatch.setattr(chat_service, "classify_confirmation_intent", reject)
    response = await chat_service.process_chat(
        ChatRequest(
            message="لا، ألغ الطلب وخلينا نكمل بالعربي",
            session_id=session_id,
        ),
        detected_language="ar",
        store=store,
    )

    assert response.language == "ar"
    assert response.message == "حسنًا، ما راح أرسله."
    assert store.get_pending_action(session_id)[0] is None
