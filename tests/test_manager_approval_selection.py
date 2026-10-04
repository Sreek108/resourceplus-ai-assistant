from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.ai import actions
from app.ai.sessions import InMemorySessionStore
from app.api import voice as voice_module
from app.main import app
from app.models.schemas import ChatRequest
from app.services.approval_selection import approval_candidates, resolve_pending_approval
from app.services import chat as chat_service, fast_reads
from app.speech import SpeechAudio, SpeechTranscript


PENDING = [
    {
        "requestId": "leave-talal",
        "requestType": "Absence",
        "employeeName": "Talal Sabbagh",
        "detail": "Compensatory Leave",
        "dateFrom": "2026-10-09",
        "status": "Pending",
    },
    {
        "requestId": "correction-talal",
        "requestType": "ExceptionEntry",
        "employeeName": "Talal Sabbagh",
        "detail": "Outside Work",
        "AttDate": "2026-09-23",
        "status": "Pending",
    },
    {
        "requestId": "leave-ahmed",
        "requestType": "Absence",
        "employeeName": "Ahmed Ali",
        "detail": "Annual Leave",
        "dateFrom": "2026-10-12",
        "status": "Pending",
    },
    {
        "requestId": "correction-fatima",
        "requestType": "ExceptionEntry",
        "employeeName": "Fatima Noor",
        "detail": "Remote Site",
        "AttDate": "2026-10-02",
        "status": "Pending",
    },
]


client = TestClient(app)


def test_employee_history_dedupe_does_not_change_manager_approval_candidates() -> None:
    visually_identical = [
        {
            "requestId": request_id,
            "requestType": "ExceptionEntry",
            "employeeName": "Talal Sabbagh",
            "detail": "Outside Work",
            "AttDate": "2026-09-02",
            "status": "Pending",
        }
        for request_id in ("manager-request-one", "manager-request-two")
    ]

    candidates = approval_candidates(visually_identical)
    selected = resolve_pending_approval(candidates, "Approve request 2")

    assert len(candidates) == 2
    assert len(selected) == 1
    assert selected[0].request_id == "manager-request-two"


def _install_pending(monkeypatch, rows=PENDING):
    fast_calls: list[str] = []
    fresh_calls: list[str] = []

    async def shown(*args, **kwargs):
        fast_calls.append("shown")
        return rows

    async def fresh(*args, **kwargs):
        fresh_calls.append("fresh")
        return rows

    async def no_model(*args, **kwargs):
        raise AssertionError("deterministic approval selection must not call a model")

    monkeypatch.setattr(fast_reads, "get_pending_approvals", shown)
    monkeypatch.setattr(actions, "get_pending_approvals", fresh)
    monkeypatch.setattr(chat_service, "run_agent", no_model)
    return fast_calls, fresh_calls


async def _show(store: InMemorySessionStore):
    return await chat_service.process_chat(
        ChatRequest(message="show pending approvals"),
        store=store,
    )


@pytest.mark.asyncio
async def test_pending_block_is_friendly_actionable_and_hides_ids(monkeypatch) -> None:
    _install_pending(monkeypatch)
    response = await _show(InMemorySessionStore())
    block = response.blocks[0]

    assert [column.key for column in block.columns] == [
        "employee", "request", "date", "detail", "status", "action"
    ]
    assert block.rows[0] == {
        "employee": "Talal Sabbagh",
        "request": "Leave",
        "date": "2026-10-09",
        "detail": "Compensatory Leave",
        "status": "Pending",
        "action": "",
    }
    assert block.rows[1]["request"] == "Attendance correction"
    assert block.rows[1]["date"] == "2026-09-23"
    assert [action.label for action in block.row_actions[0]] == ["Approve", "Reject"]
    serialized = response.model_dump_json()
    assert "leave-talal" not in serialized
    assert "correction-talal" not in serialized
    assert '"ordinal":1' in serialized


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "expected_id", "expected_copy"),
    [
        (
            "Approve Talal's Compensatory Leave",
            "leave-talal",
            "Approve Talal Sabbagh's Compensatory Leave request for Oct 9?",
        ),
        (
            "Approve Talal's Outside Work request",
            "correction-talal",
            "Approve Talal Sabbagh's Outside Work attendance correction for Sep 23?",
        ),
        ("Approve the Oct 9 request", "leave-talal", None),
        ("Approve Talal's leave request", "leave-talal", None),
        ("Approve the first one", "leave-talal", None),
        ("Approve request 2", "correction-talal", None),
        ("Approve Ahmed's Annual Leave", "leave-ahmed", None),
        ("Approve Fatima's request", "correction-fatima", None),
    ],
)
async def test_text_and_voice_phrases_select_one_exact_request_without_model(
    monkeypatch, command, expected_id, expected_copy
) -> None:
    fast_calls, fresh_calls = _install_pending(monkeypatch)
    store = InMemorySessionStore()
    shown = await _show(store)
    response = await chat_service.process_chat(
        ChatRequest(message=command, session_id=shown.session_id),
        store=store,
    )
    pending, _ = store.get_pending_action(shown.session_id)

    assert pending is not None
    assert pending.action_type == "approve_supervisor_request"
    assert pending.validated_arguments["request_id"] == expected_id
    assert response.requires_confirmation is True
    if expected_copy:
        assert response.message == expected_copy
    assert fast_calls == ["shown"]
    assert fresh_calls == ["fresh"]


def test_natural_voice_uses_the_same_exact_request_resolver(monkeypatch) -> None:
    _install_pending(monkeypatch)

    async def transcribe(*args, **kwargs):
        return SpeechTranscript("Approve request 2", "en-US", "en")

    async def synthesize(text, *, language):
        assert language == "en"
        return SpeechAudio(b"RIFFvoice")

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    monkeypatch.setattr(voice_module, "synthesize_speech", synthesize)
    shown = client.post("/api/chat", json={"message": "show pending approvals"})
    response = client.post(
        "/api/voice/chat",
        files={"audio": ("utterance.wav", b"RIFFinput", "audio/wav")},
        data={"session_id": shown.json()["session_id"]},
    )

    assert response.status_code == 200
    assert response.json()["requires_confirmation"] is True
    assert response.json()["message"] == (
        "Approve Talal Sabbagh's Outside Work attendance correction for Sep 23?"
    )
    assert "correction-talal" not in response.text


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["Approve Talal's request", "Approve"])
async def test_ambiguous_request_never_guesses(monkeypatch, command) -> None:
    _install_pending(monkeypatch)
    store = InMemorySessionStore()
    shown = await _show(store)
    response = await chat_service.process_chat(
        ChatRequest(message=command, session_id=shown.session_id), store=store
    )

    assert "more than one matching" in response.message
    assert response.requires_confirmation is False
    assert store.get_pending_action(shown.session_id)[0] is None
    assert len(response.blocks[0].rows) == (2 if "Talal" in command else 4)


@pytest.mark.asyncio
async def test_explicit_nonmatching_category_or_date_returns_no_match(monkeypatch) -> None:
    _install_pending(monkeypatch)
    store = InMemorySessionStore()
    shown = await _show(store)

    wrong_category = await chat_service.process_chat(
        ChatRequest(
            message="Approve Ahmed's attendance correction",
            session_id=shown.session_id,
        ),
        store=store,
    )
    wrong_date = await chat_service.process_chat(
        ChatRequest(message="Approve the Oct 31 request", session_id=shown.session_id),
        store=store,
    )

    assert wrong_category.message == "I couldn't find that pending request."
    assert wrong_date.message == "I couldn't find that pending request."
    assert store.get_pending_action(shown.session_id)[0] is None


@pytest.mark.asyncio
async def test_ordinal_uses_the_most_recently_displayed_matching_snapshot(
    monkeypatch,
) -> None:
    _install_pending(monkeypatch)
    store = InMemorySessionStore()
    shown = await _show(store)
    narrowed = await chat_service.process_chat(
        ChatRequest(message="Approve leave", session_id=shown.session_id), store=store
    )

    assert [row["detail"] for row in narrowed.blocks[0].rows] == [
        "Compensatory Leave",
        "Annual Leave",
    ]
    response = await chat_service.process_chat(
        ChatRequest(message="Approve the second one", session_id=shown.session_id),
        store=store,
    )
    pending, _ = store.get_pending_action(shown.session_id)

    assert response.requires_confirmation is True
    assert pending is not None
    assert pending.validated_arguments["request_id"] == "leave-ahmed"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("command", "expected_id"),
    [
        ("Reject Talal's Compensatory Leave", "leave-talal"),
        ("Reject Talal's Outside Work request", "correction-talal"),
        ("Reject request 2", "correction-talal"),
    ],
)
async def test_natural_reject_uses_the_same_exact_resolver(
    monkeypatch, command, expected_id
) -> None:
    _install_pending(monkeypatch)
    store = InMemorySessionStore()
    shown = await _show(store)
    response = await chat_service.process_chat(
        ChatRequest(message=command, session_id=shown.session_id), store=store
    )
    pending, _ = store.get_pending_action(shown.session_id)

    assert response.requires_confirmation is True
    assert pending is not None
    assert pending.validated_arguments["request_id"] == expected_id
    assert pending.validated_arguments["status"] == 2


@pytest.mark.asyncio
async def test_button_payload_selects_exact_request_but_only_prepares(monkeypatch) -> None:
    _install_pending(monkeypatch)
    store = InMemorySessionStore()
    shown = await _show(store)
    action = shown.blocks[0].row_actions[1][1]

    response = await chat_service.process_chat(
        ChatRequest(
            message=action.value,
            session_id=shown.session_id,
            approval_selection=action.payload,
        ),
        store=store,
    )
    pending, _ = store.get_pending_action(shown.session_id)

    assert pending is not None
    assert pending.validated_arguments["request_id"] == "correction-talal"
    assert pending.validated_arguments["status"] == 2
    assert response.requires_confirmation is True
    assert response.message == (
        "Reject Talal Sabbagh's Outside Work attendance correction for Sep 23?"
    )


@pytest.mark.asyncio
async def test_request_disappearing_before_confirmation_never_writes(monkeypatch) -> None:
    _install_pending(monkeypatch)
    store = InMemorySessionStore()
    shown = await _show(store)
    prepared = await chat_service.process_chat(
        ChatRequest(
            message="Approve request 1",
            session_id=shown.session_id,
        ),
        store=store,
    )
    pending, _ = store.get_pending_action(shown.session_id)
    writes = []

    async def disappeared(*args, **kwargs):
        return PENDING[1:]

    async def forbidden_write(*args, **kwargs):
        writes.append((args, kwargs))
        raise AssertionError("a stale request must not be written")

    monkeypatch.setattr(actions, "get_pending_approvals", disappeared)
    monkeypatch.setattr(chat_service, "execute_pending_action", forbidden_write)
    response = await chat_service.process_chat(
        ChatRequest(
            message="yes",
            session_id=shown.session_id,
            confirmation_id=prepared.confirmation_id,
        ),
        store=store,
    )

    assert pending is not None
    assert response.success is False
    assert "no longer in your pending approvals" in response.message
    assert writes == []


@pytest.mark.asyncio
async def test_verified_rejection_uses_specific_result_copy(monkeypatch) -> None:
    store = InMemorySessionStore()
    calls = 0

    async def pending(*args, **kwargs):
        nonlocal calls
        calls += 1
        return [PENDING[0]] if calls == 1 else PENDING[1:]

    async def reject(**kwargs):
        assert kwargs == {
            "request_id": "leave-talal",
            "request_type": "Absence",
            "status": 2,
        }
        return {"success": True, "message": "Updated"}

    monkeypatch.setattr(actions, "get_pending_approvals", pending)
    monkeypatch.setattr(actions, "approve_supervisor_request", reject)
    session_id = store.ensure_session()
    action = store.create_pending_action(
        session_id,
        action_type="approve_supervisor_request",
        validated_arguments={
            "request_id": "leave-talal",
            "request_type": "Absence",
            "status": 2,
            "employee_name": "Talal Sabbagh",
            "detail": "Compensatory Leave",
            "category": "Leave",
            "request_date": "2026-10-09",
            "approval_snapshot_verified": True,
        },
        summary="Reject Talal's Compensatory Leave request?",
        language="en",
    )
    response = await chat_service.process_chat(
        ChatRequest(
            message="yes",
            session_id=session_id,
            confirmation_id=action.confirmation_id,
        ),
        store=store,
    )

    assert response.message == (
        "Done — Talal Sabbagh's Compensatory Leave request has been rejected."
    )
    assert response.blocks[0].title == "Pending approvals"
    assert len(response.blocks[0].rows) == 3
    assert all(row["detail"] != "Compensatory Leave" for row in response.blocks[0].rows)
    assert len(store.get_trusted_result(session_id).approval_candidates) == 3


@pytest.mark.asyncio
async def test_arabic_selection_and_rejection_are_deterministic(monkeypatch) -> None:
    rows = [{
        "requestId": "arabic-leave",
        "requestType": "Absence",
        "employeeName": "أحمد علي",
        "detail": "إجازة تعويض",
        "dateFrom": "2026-10-12",
        "status": "Pending",
    }]
    _install_pending(monkeypatch, rows)
    store = InMemorySessionStore()
    shown = await chat_service.process_chat(
        ChatRequest(message="اعرض الموافقات المعلقة"), store=store
    )
    response = await chat_service.process_chat(
        ChatRequest(message="ارفض طلب إجازة تعويض", session_id=shown.session_id),
        store=store,
    )
    pending, _ = store.get_pending_action(shown.session_id)

    assert response.language == "ar"
    assert response.requires_confirmation is True
    assert pending is not None
    assert pending.validated_arguments["request_id"] == "arabic-leave"
    assert pending.validated_arguments["status"] == 2


@pytest.mark.asyncio
async def test_approval_snapshot_is_identity_isolated(monkeypatch) -> None:
    _install_pending(monkeypatch)
    store = InMemorySessionStore()
    shown = await chat_service.process_chat(
        ChatRequest(
            message="show pending approvals",
            email="manager.one@example.com",
            instance="Universal",
        ),
        store=store,
    )
    response = await chat_service.process_chat(
        ChatRequest(
            message="Approve request 1",
            session_id=shown.session_id,
            email="manager.two@example.com",
            instance="Universal",
        ),
        store=store,
    )

    assert response.success is False
    assert "current signed-in user" in response.message
