from __future__ import annotations

from datetime import date

import pytest

from app.ai import actions, agent, conversation
from app.ai.sessions import InMemorySessionStore
from app.models.schemas import ChatRequest
from app.services import chat as chat_service
from app.services import fast_reads


def _attendance_row() -> dict[str, object]:
    return {
        "AttDate": "2026-09-10",
        "DayType": "Regular",
        "CheckIN": "09:30",
        "CheckOut": "17:00",
        "NetHrs": "07:30",
        "LessHrs": "00:30",
    }


def _exception_rows() -> list[dict[str, object]]:
    return [
        {
            "exceptionalID": "arabic-secret-one",
            "entryTime": "2026-09-20T09:10:00",
            "entryType": 1,
            "reason": "Traffic",
            "status": "Pending",
        },
        {
            "exceptionalID": "arabic-secret-two",
            "entryTime": "2026-09-22T16:20:00",
            "entryType": 2,
            "reason": "الظروف العائلية",
            "status": "Not Approved",
        },
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "كم بقي لي من دقائق السماح؟",
        "كم عندي من وقت السماح؟",
        "اعرض رصيد السماح",
        "كم متبقي لي من دقائق التعويض هذا الأسبوع؟",
    ],
)
async def test_arabic_balance_uses_balance_only_fast_path(monkeypatch, message) -> None:
    calls = []

    async def balance(target_date):
        calls.append(target_date)
        return {
            "hasPolicy": True,
            "policyName": "Monthly buffer",
            "limitType": 2,
            "limitValue": 120,
            "used": 20,
            "remaining": 100.0,
            "resetsOn": "2026-10-01",
        }

    async def forbidden_home(*args, **kwargs):
        raise AssertionError("Arabic balance must not call GetHomeData")

    def forbidden_model(*args, **kwargs):
        raise AssertionError("Arabic balance must not call OpenAI")

    monkeypatch.setattr(fast_reads, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(fast_reads, "get_exceptional_entry_balance", balance)
    monkeypatch.setattr(agent, "AsyncOpenAI", forbidden_model)
    monkeypatch.setattr(__import__("app.resourceplus.home", fromlist=["get_home_data"]), "get_home_data", forbidden_home)

    result = await agent.run_agent(
        message,
        lang=1,
        session_id="arabic-balance",
        history=[],
        response_language="ar",
    )

    assert len(calls) == 1
    assert result.tools_used == ["get_exceptional_entry_balance"]
    assert result.message == "باقي لك 100 دقيقة من وقت السماح."
    assert result.speech_message == result.message
    block = result.blocks[0]
    assert block.title == "رصيد السماح"
    assert {item.label: item.value for item in block.items}["المتبقي"] == 100
    assert {item.label: item.value for item in block.items}["الوحدة"] == "دقائق"


@pytest.mark.asyncio
async def test_arabic_less_hours_read_is_localized_and_structured(monkeypatch) -> None:
    store = InMemorySessionStore()

    async def attendance(start, end, **kwargs):
        assert (start, end) == ("2026-09-01", "2026-09-30")
        return {"Days": [_attendance_row()]}

    async def balance(*args, **kwargs):
        return {"hasPolicy": True, "limitType": 2, "remaining": 100}

    async def forbidden_agent(*args, **kwargs):
        raise AssertionError("Arabic less-hours read must not call the model path")

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(actions, "get_attendance_summary", attendance)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)
    monkeypatch.setattr(chat_service, "run_agent", forbidden_agent)

    response = await chat_service.process_chat(
        ChatRequest(message="اعرض الساعات الناقصة لهذا الشهر"),
        store=store,
    )

    assert response.language == "ar"
    assert response.message == "عندك يوم حضور واحد يحتاج تصحيح خلال هالفترة: 2026-09-10."
    assert response.speech_message == "عندك يوم حضور واحد يحتاج تصحيح خلال هالفترة."
    table = next(block for block in response.blocks if block.type == "table")
    assert table.title == "فجوات الحضور"
    assert table.rows[0]["date"] == "2026-09-10"
    assert table.rows[0]["less"] == "00:30"
    assert [column.label for column in table.columns][:3] == ["التاريخ", "نوع اليوم", "الدخول"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "minutes", "entry_type"),
    [
        ("صحح الساعات الناقصة يوم 10 سبتمبر", None, None),
        ("صحح فقط 10 دقائق يوم 10 سبتمبر", 10, None),
        ("صحح فقط 10 دقائق من التأخير يوم 10 سبتمبر", 10, 1),
        ("صحح التأخير فقط يوم 10 سبتمبر", None, 1),
        ("صحح الخروج المبكر فقط يوم 10 سبتمبر", None, 2),
    ],
)
async def test_arabic_less_hours_correction_slots_and_reason_continuation(
    monkeypatch,
    message,
    minutes,
    entry_type,
) -> None:
    store = InMemorySessionStore()
    writes = []

    async def attendance(*args, **kwargs):
        return {"Days": [_attendance_row()]}

    async def reasons(*args, **kwargs):
        return [{"reasonID": "traffic-live-id", "reasonName": "Traffic"}]

    async def balance(*args, **kwargs):
        return {"hasPolicy": True, "limitType": 2, "remaining": 100}

    async def forbidden_write(*args, **kwargs):
        writes.append((args, kwargs))

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(actions, "get_attendance_summary", attendance)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    monkeypatch.setattr(actions, "create_exceptional_entry_from_summary", forbidden_write)
    monkeypatch.setattr(conversation, "cached_exception_reasons", reasons)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)

    prompt = await chat_service.process_chat(ChatRequest(message=message), store=store)
    draft = store.get_conversation_draft(prompt.session_id)
    assert prompt.language == "ar"
    assert prompt.message.endswith("وش سبب التصحيح؟")
    assert prompt.needs_reason is True
    assert draft is not None and draft.slots["date"] == "2026-09-10"
    assert draft.slots.get("minutes") == (str(minutes) if minutes is not None else None)
    assert draft.slots.get("entry_type") == (
        str(entry_type) if entry_type is not None else None
    )
    assert prompt.blocks[-1].title == "اختر السبب"

    prepared = await chat_service.process_chat(
        ChatRequest(message="Traffic", session_id=prompt.session_id),
        store=store,
    )
    pending, _ = store.get_pending_action(prompt.session_id)
    assert prepared.requires_confirmation is True
    assert pending is not None
    assert pending.validated_arguments.get("minutes") == minutes
    assert pending.validated_arguments.get("entry_type") == entry_type
    assert prepared.message.startswith("للتأكيد: أرسل تصحيح حضور")
    assert "دقيقة" in prepared.speech_message
    if minutes is not None:
        assert f"الدقائق المطلوبة: {minutes}" in prepared.message
    if entry_type == 1:
        assert "النطاق: التأخر في الدخول فقط" in prepared.message
    elif entry_type == 2:
        assert "النطاق: الخروج المبكر فقط" in prepared.message
    assert writes == []

    rejected = await chat_service.process_chat(
        ChatRequest(
            message="لا",
            session_id=prompt.session_id,
            confirmation_id=prepared.confirmation_id,
        ),
        detected_language="ar",
        store=store,
    )
    assert rejected.message == "حسنًا، ما راح أرسله."
    assert writes == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "اعرض استثناءاتي هذا الشهر",
    ],
)
async def test_arabic_exceptional_read_is_specific_and_localized(monkeypatch, message) -> None:
    calls = []

    async def exceptional(start, end, **kwargs):
        calls.append((start, end))
        return _exception_rows()

    async def forbidden_broad(*args, **kwargs):
        raise AssertionError("Arabic exceptional reads must not call MyRequests")

    def forbidden_model(*args, **kwargs):
        raise AssertionError("Arabic exceptional reads must not call OpenAI")

    monkeypatch.setattr(fast_reads, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(fast_reads, "get_exceptional_entry_requests", exceptional)
    monkeypatch.setattr(fast_reads, "get_my_request_status", forbidden_broad)
    monkeypatch.setattr(agent, "AsyncOpenAI", forbidden_model)

    result = await agent.run_agent(
        message,
        lang=1,
        session_id="arabic-exception-read",
        history=[],
        response_language="ar",
    )

    assert result.tools_used == ["get_exceptional_entries"]
    assert calls
    table = result.blocks[0]
    assert table.title == "طلبات تصحيح الحضور"
    assert [column.label for column in table.columns] == ["التاريخ", "النوع", "السبب", "الحالة"]
    assert table.rows[0]["type"] == "وصول متأخر"
    assert table.rows[1]["type"] == "خروج مبكر"
    assert table.rows[1]["status"] == "غير موافق عليه"
    assert "arabic-secret" not in table.model_dump_json()
    assert result.message.startswith("عندك طلبين تصحيح حضور خلال هالفترة.")
    assert result.speech_message == "عندك طلبين تصحيح حضور خلال هالفترة."


@pytest.mark.parametrize(
    ("auto_approved", "expected"),
    [
        (True, "تم تصحيح حضورك واعتماده تلقائياً."),
        (False, "أرسلت تصحيح حضورك لموافقة مديرك."),
    ],
)
def test_arabic_correction_result_copy_is_employee_facing(
    auto_approved,
    expected,
) -> None:
    message = chat_service._from_summary_result_message(
        {"success": True, "isAutoApproved": auto_approved},
        "ar",
        success=True,
    )

    assert message == expected
    assert "ResourcePlus" not in message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("selection", "expected_id"),
    [
        ("الثاني", "arabic-secret-two"),
        ("طلب 22 سبتمبر", "arabic-secret-two"),
        ("طلب الخروج المبكر", "arabic-secret-two"),
        ("طلب الظروف العائلية", "arabic-secret-two"),
    ],
)
async def test_arabic_cancellation_discovery_and_selection(
    monkeypatch,
    selection,
    expected_id,
) -> None:
    store = InMemorySessionStore()
    writes = []

    async def exceptional(*args, **kwargs):
        return _exception_rows()

    async def forbidden_write(*args, **kwargs):
        writes.append((args, kwargs))

    def forbidden_model(*args, **kwargs):
        raise AssertionError("Arabic cancellation must not call OpenAI")

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(actions, "get_exceptional_entry_requests", exceptional)
    monkeypatch.setattr(actions, "cancel_exceptional_entry", forbidden_write)
    monkeypatch.setattr(agent, "AsyncOpenAI", forbidden_model)

    discovered = await chat_service.process_chat(
        ChatRequest(message="ألغِ طلبات الاستثناء المعلقة"),
        store=store,
    )
    assert discovered.language == "ar"
    assert discovered.message.startswith("وجدت أكثر من طلب قابل للإلغاء")
    table = next(block for block in discovered.blocks if block.type == "table")
    assert table.title == "طلبات تصحيح حضور قابلة للإلغاء"
    assert table.rows[1]["type"] == "خروج مبكر"

    prepared = await chat_service.process_chat(
        ChatRequest(message=selection, session_id=discovered.session_id),
        store=store,
    )
    pending, _ = store.get_pending_action(discovered.session_id)
    assert prepared.requires_confirmation is True
    assert pending is not None
    assert pending.validated_arguments["exceptional_id"] == expected_id
    assert pending.language == "ar"
    assert prepared.message.startswith("تبغى تلغي طلب تصحيح الحضور")
    assert any(
        block.type == "confirmation" for block in prepared.blocks
    )
    assert "arabic-secret" not in prepared.model_dump_json()
    assert writes == []

    rejected = await chat_service.process_chat(
        ChatRequest(
            message="لا",
            session_id=discovered.session_id,
            confirmation_id=prepared.confirmation_id,
        ),
        detected_language="ar",
        store=store,
    )
    assert rejected.success is True
    assert writes == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "message",
    [
        "ألغِ طلب الاستثناء المعلق",
        "الغِ طلب الاستثناء",
        "إلغاء طلب الاستثناء",
        "أريد إلغاء طلب استثناء",
        "ألغِ أحد طلبات الاستثناء",
    ],
)
async def test_arabic_single_cancellation_prepares_without_model(monkeypatch, message) -> None:
    store = InMemorySessionStore()

    async def exceptional(*args, **kwargs):
        return [_exception_rows()[0]]

    def forbidden_model(*args, **kwargs):
        raise AssertionError("Arabic cancellation must not call OpenAI")

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(actions, "get_exceptional_entry_requests", exceptional)
    monkeypatch.setattr(agent, "AsyncOpenAI", forbidden_model)
    response = await chat_service.process_chat(ChatRequest(message=message), store=store)
    assert response.requires_confirmation is True
    assert response.language == "ar"
    assert response.tools_used == ["get_exceptional_entries", "prepare_cancel_exceptional_entry"]


@pytest.mark.asyncio
async def test_arabic_new_intents_replace_only_non_executable_drafts(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    store.save_conversation_draft(
        session_id,
        intent="less_hours_correction",
        slots={
            "date": "2026-09-29",
            "reason_options": "Traffic\x1fOutside Work",
        },
        language="ar",
    )

    async def exceptional(*args, **kwargs):
        return _exception_rows()

    monkeypatch.setattr(fast_reads, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(fast_reads, "get_exceptional_entry_requests", exceptional)
    response = await chat_service.process_chat(
        ChatRequest(message="اعرض طلبات الاستثناء لهذا الشهر", session_id=session_id),
        store=store,
    )
    assert response.tools_used == ["get_exceptional_entries"]
    assert store.get_conversation_draft(session_id) is None

    pending = store.create_pending_action(
        session_id,
        action_type="cancel_exceptional_entry",
        validated_arguments={"exceptional_id": "protected", "display": "طلب معلّق"},
        summary="تأكيد إلغاء الطلب",
        language="ar",
    )

    async def other(*args, **kwargs):
        return "OTHER"

    monkeypatch.setattr(chat_service, "_confirmation_decision", other)
    protected = await chat_service.process_chat(
        ChatRequest(message="اعرض الساعات الناقصة لهذا الشهر", session_id=session_id),
        store=store,
    )
    still_pending, _ = store.get_pending_action(session_id)
    assert protected.requires_confirmation is True
    assert still_pending is not None
    assert still_pending.confirmation_id == pending.confirmation_id


@pytest.mark.asyncio
async def test_arabic_new_correction_replaces_old_reason_draft(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    store.save_conversation_draft(
        session_id,
        intent="less_hours_correction",
        slots={
            "date": "2026-09-29",
            "reason_options": "Traffic\x1fOutside Work",
        },
        language="ar",
    )

    async def attendance(*args, **kwargs):
        return {"Days": [_attendance_row()]}

    async def reasons(*args, **kwargs):
        return [{"reasonID": "traffic-live-id", "reasonName": "Traffic"}]

    async def balance(*args, **kwargs):
        return {"hasPolicy": True, "remaining": 100, "limitType": 2}

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(actions, "get_attendance_summary", attendance)
    monkeypatch.setattr(conversation, "cached_exception_reasons", reasons)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)
    response = await chat_service.process_chat(
        ChatRequest(
            message="صحح فقط 10 دقائق من التأخير يوم 10 سبتمبر",
            session_id=session_id,
        ),
        store=store,
    )
    replacement = store.get_conversation_draft(session_id)
    assert response.message.endswith("وش سبب التصحيح؟")
    assert replacement is not None
    assert replacement.slots["date"] == "2026-09-10"
    assert replacement.slots["minutes"] == "10"
    assert replacement.slots["entry_type"] == "1"


@pytest.mark.asyncio
async def test_arabic_less_hours_read_interrupts_cancellation_draft(monkeypatch) -> None:
    store = InMemorySessionStore()

    async def exceptional(*args, **kwargs):
        return _exception_rows()

    async def attendance(*args, **kwargs):
        return {"Days": [_attendance_row()]}

    async def balance(*args, **kwargs):
        return {"hasPolicy": True, "remaining": 100, "limitType": 2}

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(actions, "get_exceptional_entry_requests", exceptional)
    monkeypatch.setattr(actions, "get_attendance_summary", attendance)
    monkeypatch.setattr(conversation, "get_exceptional_entry_balance", balance)
    discovered = await chat_service.process_chat(
        ChatRequest(message="ألغِ طلبات الاستثناء المعلقة"),
        store=store,
    )
    assert store.get_conversation_draft(discovered.session_id) is not None
    read = await chat_service.process_chat(
        ChatRequest(
            message="اعرض الساعات الناقصة لهذا الشهر",
            session_id=discovered.session_id,
        ),
        store=store,
    )
    assert read.message == "عندك يوم حضور واحد يحتاج تصحيح خلال هالفترة: 2026-09-10."
    assert store.get_conversation_draft(discovered.session_id) is None
    assert store.get_pending_action(discovered.session_id)[0] is None


@pytest.mark.asyncio
async def test_arabic_balance_interrupts_cancellation_draft_without_write(
    monkeypatch,
) -> None:
    store = InMemorySessionStore()
    writes = []

    async def exceptional(*args, **kwargs):
        return _exception_rows()

    async def balance(target_date):
        assert target_date == "2026-09-30"
        return {
            "hasPolicy": True,
            "limitType": 2,
            "limitValue": 120,
            "used": 20,
            "remaining": 100,
            "resetsOn": "2026-10-01",
        }

    async def forbidden_write(*args, **kwargs):
        writes.append((args, kwargs))

    def forbidden_model(*args, **kwargs):
        raise AssertionError("draft interruption must remain deterministic")

    monkeypatch.setattr(conversation, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(fast_reads, "resourceplus_today", lambda: date(2026, 9, 30))
    monkeypatch.setattr(actions, "get_exceptional_entry_requests", exceptional)
    monkeypatch.setattr(fast_reads, "get_exceptional_entry_balance", balance)
    monkeypatch.setattr(actions, "cancel_exceptional_entry", forbidden_write)
    monkeypatch.setattr(agent, "AsyncOpenAI", forbidden_model)

    discovered = await chat_service.process_chat(
        ChatRequest(message="ألغِ طلبات الاستثناء المعلقة"),
        store=store,
    )
    assert store.get_conversation_draft(discovered.session_id) is not None

    response = await chat_service.process_chat(
        ChatRequest(
            message="كم بقي لي من دقائق السماح؟",
            session_id=discovered.session_id,
        ),
        store=store,
    )

    assert response.language == "ar"
    assert response.tools_used == ["get_exceptional_entry_balance"]
    assert "100" in response.message
    assert store.get_conversation_draft(discovered.session_id) is None
    assert store.get_pending_action(discovered.session_id)[0] is None
    assert writes == []


@pytest.mark.asyncio
@pytest.mark.parametrize("reply", ["نعم", "أجل"])
async def test_explicit_arabic_yes_executes_only_stored_pending_action(monkeypatch, reply) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session()
    pending = store.create_pending_action(
        session_id,
        action_type="cancel_exceptional_entry",
        validated_arguments={"exceptional_id": "stored-only", "display": "طلب معلّق"},
        summary="تأكيد إلغاء الطلب",
        language="ar",
    )
    executions = []

    async def execute(action_type, arguments):
        executions.append((action_type, arguments))
        return {"success": True, "message": "Cancelled"}

    async def forbidden_classifier(*args, **kwargs):
        raise AssertionError("Explicit Arabic confirmation must not call OpenAI")

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    monkeypatch.setattr(chat_service, "classify_confirmation_intent", forbidden_classifier)
    response = await chat_service.process_chat(
        ChatRequest(
            message=reply,
            session_id=session_id,
            confirmation_id=pending.confirmation_id,
        ),
        detected_language="ar",
        store=store,
    )
    assert response.success is True
    assert executions == [
        ("cancel_exceptional_entry", {"exceptional_id": "stored-only", "display": "طلب معلّق"})
    ]
