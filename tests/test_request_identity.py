from __future__ import annotations

import asyncio
import logging
from contextlib import contextmanager
from dataclasses import FrozenInstanceError

import pytest
from fastapi.testclient import TestClient

from app import identity as identity_module
from app.ai.agent import AgentResult
from app.ai.sessions import InMemorySessionStore, SessionIdentityMismatch, session_store
from app.ai.tools import TOOL_DEFINITIONS
from app.api import voice as voice_module
from app.identity import (
    RequestIdentity,
    RequestIdentityError,
    bind_request_identity,
    current_request_identity,
    reset_request_identity,
    resolve_request_identity,
)
from app.main import app
from app.models.schemas import ChatRequest, ChatResponse
from app.resourceplus import exceptional as exceptional_module
from app.resourceplus.approvals import get_pending_approvals
from app.resourceplus.attendance import (
    get_attendance_summary,
    get_exception_reasons,
    get_missing_punch_suggestions,
)
from app.resourceplus.employee import get_profile_data
from app.resourceplus.exceptional import get_exceptional_entry_balance
from app.resourceplus.leave import get_my_day_type_requests
from app.resourceplus.requests import get_exceptional_entry_requests
from app.services import chat as chat_service
from app.speech import SpeechAudio, SpeechTranscript


client = TestClient(app)


@contextmanager
def identity_scope(email: str, instance: str):
    token = bind_request_identity(RequestIdentity(email=email, instance=instance))
    try:
        yield
    finally:
        reset_request_identity(token)


class RecordingResourcePlusClient:
    def __init__(self) -> None:
        self.gets: list[tuple[str, dict[str, object]]] = []
        self.posts: list[tuple[str, dict[str, object], dict[str, object]]] = []

    async def get(self, path: str, *, params: dict[str, object]):
        self.gets.append((path, params))
        return {"ok": True}

    async def post(
        self,
        path: str,
        *,
        params: dict[str, object],
        json_body: dict[str, object],
    ):
        self.posts.append((path, params, json_body))
        return {"success": True, "isAutoApproved": False}


def test_text_request_binds_supplied_identity(monkeypatch) -> None:
    session_store.clear()
    observed: list[RequestIdentity] = []

    async def run_agent(message, *, lang, session_id, history, response_language):
        observed.append(current_request_identity())
        return AgentResult(message="Current profile.", tools_used=["get_profile_data"])

    monkeypatch.setattr(chat_service, "run_agent", run_agent)
    response = client.post(
        "/api/chat",
        json={
            "message": "Show my profile",
            "session_id": "identity-text-session",
            "email": "employee@example.com",
            "instance": "Client02",
            "confirmation_id": None,
        },
    )

    assert response.status_code == 200
    assert observed == [RequestIdentity("employee@example.com", "Client02")]
    assert "employee@example.com" not in response.text


@pytest.mark.asyncio
async def test_profile_and_attendance_use_bound_request_identity() -> None:
    resourceplus = RecordingResourcePlusClient()

    with identity_scope("employee@example.com", "Client02"):
        await get_profile_data(client=resourceplus)
        await get_attendance_summary(
            "2026-09-01",
            "2026-09-30",
            client=resourceplus,
        )

    assert resourceplus.gets[0][1] == {
        "instanceName": "Client02",
        "Usremail": "employee@example.com",
        "Lang": 1,
    }
    assert resourceplus.gets[1][1] == {
        "usrEmail": "employee@example.com",
        "fromDate": "2026-09-01",
        "toDate": "2026-09-30",
        "instanceName": "Client02",
        "lang": 1,
    }


@pytest.mark.asyncio
async def test_all_deterministic_reads_use_explicit_portal_instance() -> None:
    resourceplus = RecordingResourcePlusClient()

    with identity_scope("talal.sabbagh@example.com", "portalv21"):
        await get_profile_data(client=resourceplus)
        await get_attendance_summary("2026-09-01", "2026-09-30", client=resourceplus)
        await get_missing_punch_suggestions(
            "2026-09-01", "2026-09-30", client=resourceplus
        )
        await get_exceptional_entry_balance("2026-09-14", client=resourceplus)
        await get_exception_reasons(client=resourceplus)
        await get_exceptional_entry_requests(
            "2026-09-01", "2026-09-30", client=resourceplus
        )
        await get_my_day_type_requests(
            "2026-09-01", "2026-09-30", client=resourceplus
        )
        await get_pending_approvals(client=resourceplus)

    assert [path for path, _ in resourceplus.gets] == [
        "api/Client/GetProfileData",
        "api/AI/AttendanceSummary",
        "api/AI/MissingPunchSuggestions",
        "api/AI/ExceptionalEntries/Balance",
        "api/AI/ExceptionalEntries/Reasons",
        "api/AI/ExceptionalEntries",
        "api/AI/DayTypeMapping/MyRequests",
        "api/AI/Supervisor/PendingApprovals",
    ]
    assert all(
        params["instanceName"] == "portalv21"
        for _, params in resourceplus.gets
    )


@pytest.mark.asyncio
async def test_confirmed_from_summary_uses_pending_owner_not_default_instance(
    monkeypatch,
) -> None:
    class LocalSettings:
        app_environment = "local"
        rp_default_email = "fallback@example.com"
        rp_instance = "Universal"

    class ContextDroppingStore(InMemorySessionStore):
        def consume_pending_action(self, session_id, confirmation_id=None):
            action = super().consume_pending_action(session_id, confirmation_id)
            # Reproduce the observed execution seam: the ambient request identity
            # is unavailable after the immutable action has been authorized.
            monkeypatch.setattr("app.identity.get_settings", lambda: LocalSettings())
            identity_module._request_identity.set(None)
            return action

    resourceplus = RecordingResourcePlusClient()
    monkeypatch.setattr(
        exceptional_module,
        "ResourcePlusClient",
        lambda: resourceplus,
    )
    store = ContextDroppingStore()
    owner = RequestIdentity("talal.sabbagh@example.com", "portalv21")
    with identity_scope(owner.email, owner.instance):
        session_id = store.ensure_session("talal-from-summary")
        pending = store.create_pending_action(
            session_id,
            action_type="create_exceptional_entry_from_summary",
            validated_arguments={
                "att_date": "2026-09-14",
                "reason_id": "live-reason-id",
                "reason_name": "Family Circumstances",
                "remarks": "Family Circumstances",
                "less_hours": "00:30",
            },
            summary="Submit attendance correction?",
            language="en",
        )

    request = ChatRequest(
        message="Yes",
        session_id=session_id,
        confirmation_id=pending.confirmation_id,
        email=owner.email,
        instance=owner.instance,
    )
    response = await chat_service.process_chat(request, store=store)
    replay = await chat_service.process_chat(request, store=store)

    assert response.success is True
    assert replay.success is False
    assert len(resourceplus.posts) == 1
    path, params, body = resourceplus.posts[0]
    assert path == "api/AI/ExceptionalEntries/FromSummary"
    assert params == {"instanceName": "portalv21"}
    assert body == {
        "usrEmail": "talal.sabbagh@example.com",
        "attDate": "2026-09-14",
        "reasonID": "live-reason-id",
        "remarks": "Family Circumstances",
    }


def test_voice_http_passes_supplied_identity_to_shared_chat(monkeypatch) -> None:
    seen: list[ChatRequest] = []

    async def transcribe(*args, **kwargs):
        return SpeechTranscript("Show my attendance", "en-US", "en")

    async def process(request, *, detected_language):
        seen.append(request)
        return ChatResponse(
            success=True,
            message="Attendance ready.",
            language=detected_language,
            session_id=request.session_id or "voice-session",
        ).set_speech_message("Attendance ready.")

    async def synthesize(*args, **kwargs):
        return SpeechAudio(b"RIFFaudio")

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    monkeypatch.setattr(voice_module, "process_chat", process)
    monkeypatch.setattr(voice_module, "synthesize_speech", synthesize)

    response = client.post(
        "/api/voice/chat",
        files={"audio": ("utterance.wav", b"RIFFinput", "audio/wav")},
        data={
            "session_id": "identity-voice-http",
            "email": "employee@example.com",
            "instance": "Client02",
        },
    )

    assert response.status_code == 200
    assert seen[0].email == "employee@example.com"
    assert seen[0].instance == "Client02"


def test_voice_websocket_start_passes_supplied_identity(monkeypatch) -> None:
    seen: list[ChatRequest] = []

    class Recognizer:
        async def start(self):
            return None

        def write(self, chunk):
            return None

        async def finish(self):
            return SpeechTranscript("Show my attendance", "en-US", "en")

        async def cancel(self):
            return None

    async def process(request, *, detected_language):
        seen.append(request)
        return ChatResponse(
            success=True,
            message="Attendance ready.",
            language=detected_language,
            session_id=request.session_id or "voice-session",
        ).set_speech_message("Attendance ready.")

    async def synthesize(*args, **kwargs):
        return SpeechAudio(b"RIFFaudio")

    async def persist(*args, **kwargs):
        return None

    monkeypatch.setattr(voice_module, "StreamingSpeechRecognizer", Recognizer)
    monkeypatch.setattr(voice_module, "process_chat", process)
    monkeypatch.setattr(voice_module, "synthesize_speech", synthesize)
    monkeypatch.setattr(voice_module, "persist_audit", persist)

    with client.websocket_connect("/api/voice/stream") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "sample_rate": 16_000,
                "session_id": "identity-voice-stream",
                "email": "employee@example.com",
                "instance": "Client02",
            }
        )
        assert websocket.receive_json() == {"type": "ready"}
        websocket.send_bytes(b"\x00\x00" * 320)
        websocket.send_json({"type": "end"})
        assert websocket.receive_json()["type"] == "final"

    assert seen[0].email == "employee@example.com"
    assert seen[0].instance == "Client02"


@pytest.mark.asyncio
async def test_same_session_and_identity_retain_conversation_history(monkeypatch) -> None:
    store = InMemorySessionStore()
    histories: list[list[dict[str, str]]] = []

    async def run_agent(message, *, lang, session_id, history, response_language):
        histories.append(history)
        return AgentResult(message=f"Reply {len(histories)}", tools_used=[])

    monkeypatch.setattr(chat_service, "run_agent", run_agent)
    identity = {"email": "employee@example.com", "instance": "Universal"}
    await chat_service.process_chat(
        ChatRequest(message="First", session_id="owned-session", **identity),
        store=store,
    )
    await chat_service.process_chat(
        ChatRequest(message="Second", session_id="owned-session", **identity),
        store=store,
    )

    assert histories[0] == []
    assert histories[1] == [
        {"role": "user", "content": "First"},
        {"role": "assistant", "content": "Reply 1"},
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "second_identity",
    [
        {"email": "manager@example.com", "instance": "Universal"},
        {"email": "employee@example.com", "instance": "Client02"},
    ],
)
async def test_same_session_with_different_identity_is_rejected(
    monkeypatch,
    second_identity: dict[str, str],
) -> None:
    store = InMemorySessionStore()
    calls: list[str] = []

    async def run_agent(message, *, lang, session_id, history, response_language):
        calls.append(message)
        return AgentResult(message="private first-user response", tools_used=[])

    monkeypatch.setattr(chat_service, "run_agent", run_agent)
    await chat_service.process_chat(
        ChatRequest(
            message="First",
            session_id="shared-browser-session",
            email="employee@example.com",
            instance="Universal",
        ),
        store=store,
    )
    response = await chat_service.process_chat(
        ChatRequest(
            message="Second",
            session_id="shared-browser-session",
            **second_identity,
        ),
        store=store,
    )

    assert response.success is False
    assert "Start a new conversation" in response.message
    assert "private first-user response" not in response.message
    assert calls == ["First"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "confirmer",
    [
        RequestIdentity("talal.sabbagh@example.com", "Universal"),
        RequestIdentity("other.employee@example.com", "portalv21"),
    ],
)
async def test_pending_action_cannot_be_confirmed_by_another_identity(
    monkeypatch,
    confirmer: RequestIdentity,
) -> None:
    store = InMemorySessionStore()
    with identity_scope("talal.sabbagh@example.com", "portalv21"):
        session_id = store.ensure_session("pending-owner-session")
        action = store.create_pending_action(
            session_id,
            action_type="book_day_type",
            validated_arguments={
                "date_from": "2026-09-29",
                "date_to": "2026-09-29",
                "day_type_id": 20,
                "day_type_name": "Annual Leave",
            },
            summary="Submit Annual Leave for 29 September?",
            language="en",
        )

    writes: list[object] = []

    async def execute(*args, **kwargs):
        writes.append((args, kwargs))
        return {"success": True}

    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    response = await chat_service.process_chat(
        ChatRequest(
            message="Yes",
            session_id=session_id,
            confirmation_id=action.confirmation_id,
            email=confirmer.email,
            instance=confirmer.instance,
        ),
        store=store,
    )

    assert response.success is False
    assert writes == []
    with identity_scope("talal.sabbagh@example.com", "portalv21"):
        remaining, expired = store.get_pending_action(session_id)
    assert expired is False
    assert remaining is not None
    assert remaining.confirmation_id == action.confirmation_id


def test_pending_action_owner_is_immutable() -> None:
    store = InMemorySessionStore()
    with identity_scope("employee@example.com", "Universal"):
        session_id = store.ensure_session("immutable-owner-session")
        action = store.create_pending_action(
            session_id,
            action_type="cancel_day_type",
            validated_arguments={"mapping_id": "request-1"},
            summary="Cancel request?",
            language="en",
        )

    with pytest.raises(FrozenInstanceError):
        action.owner = RequestIdentity("manager@example.com", "Universal")  # type: ignore[misc]


@pytest.mark.asyncio
async def test_simultaneous_requests_keep_identity_context_isolated(monkeypatch) -> None:
    observed: dict[str, list[RequestIdentity]] = {}

    async def run_agent(message, *, lang, session_id, history, response_language):
        first = current_request_identity()
        await asyncio.sleep(0)
        observed[message] = [first, current_request_identity()]
        return AgentResult(message="Done", tools_used=[])

    monkeypatch.setattr(chat_service, "run_agent", run_agent)
    await asyncio.gather(
        chat_service.process_chat(
            ChatRequest(
                message="employee",
                session_id="concurrent-employee",
                email="employee@example.com",
                instance="Universal",
            ),
            store=InMemorySessionStore(),
        ),
        chat_service.process_chat(
            ChatRequest(
                message="manager",
                session_id="concurrent-manager",
                email="manager@example.com",
                instance="Client02",
            ),
            store=InMemorySessionStore(),
        ),
    )

    assert observed == {
        "employee": [
            RequestIdentity("employee@example.com", "Universal"),
            RequestIdentity("employee@example.com", "Universal"),
        ],
        "manager": [
            RequestIdentity("manager@example.com", "Client02"),
            RequestIdentity("manager@example.com", "Client02"),
        ],
    }


@pytest.mark.parametrize(
    "payload",
    [
        {"email": "employee@example.com"},
        {"instance": "Universal"},
    ],
)
def test_text_rejects_partial_identity_without_exposing_values(payload) -> None:
    response = client.post("/api/chat", json={"message": "Hello", **payload})

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_identity"
    assert "employee@example.com" not in response.text


@pytest.mark.parametrize(
    "data",
    [
        {"email": "employee@example.com"},
        {"instance": "Universal"},
    ],
)
def test_voice_http_rejects_partial_identity_before_processing(monkeypatch, data) -> None:
    transcriptions: list[object] = []

    async def transcribe(*args, **kwargs):
        transcriptions.append((args, kwargs))
        raise AssertionError("Invalid identity must fail before speech processing")

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    response = client.post(
        "/api/voice/chat",
        files={"audio": ("utterance.wav", b"RIFFinput", "audio/wav")},
        data=data,
    )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "invalid_identity"
    assert transcriptions == []


@pytest.mark.parametrize(
    "identity",
    [
        {"email": "employee@example.com"},
        {"instance": "Universal"},
    ],
)
def test_voice_websocket_rejects_partial_identity(identity) -> None:
    with client.websocket_connect("/api/voice/stream") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "sample_rate": 16_000,
                **identity,
            }
        )
        response = websocket.receive_json()

    assert response["type"] == "error"
    assert response["code"] == "invalid_stream_state"
    assert "employee@example.com" not in str(response)


@pytest.mark.parametrize(
    ("default_email", "default_instance"),
    [
        ("fallback@example.com", ""),
        ("", "Universal"),
    ],
)
def test_fallback_identity_is_all_or_nothing(
    monkeypatch,
    default_email: str,
    default_instance: str,
) -> None:
    class IncompleteSettings:
        app_environment = "local"
        rp_default_email = default_email
        rp_instance = default_instance

    monkeypatch.setattr("app.identity.get_settings", lambda: IncompleteSettings())
    with pytest.raises(RequestIdentityError):
        resolve_request_identity(None, None)
    with pytest.raises(RequestIdentityError):
        resolve_request_identity("employee@example.com", None)

    assert resolve_request_identity("employee@example.com", "Client02") == RequestIdentity(
        "employee@example.com",
        "Client02",
    )


@pytest.mark.parametrize("environment", ["uat", "production"])
def test_deployed_environment_rejects_missing_request_identity(
    monkeypatch,
    environment: str,
) -> None:
    class DeployedSettings:
        app_environment = environment
        rp_default_email = "fallback@example.com"
        rp_instance = "Universal"

    monkeypatch.setattr("app.identity.get_settings", lambda: DeployedSettings())

    with pytest.raises(RequestIdentityError, match="required in this environment"):
        resolve_request_identity(None, None)


@pytest.mark.parametrize("environment", ["local", "test", " LOCAL ", "TEST"])
def test_local_and_test_allow_complete_paired_fallback(
    monkeypatch,
    environment: str,
) -> None:
    class LocalSettings:
        app_environment = environment
        rp_default_email = "Fallback@Example.com"
        rp_instance = "Universal"

    monkeypatch.setattr("app.identity.get_settings", lambda: LocalSettings())

    identity = resolve_request_identity(None, None)

    assert identity.email == "Fallback@Example.com"
    assert identity.instance == "Universal"


@pytest.mark.parametrize(
    ("email", "instance"),
    [
        ("employee@example.com", None),
        (None, "Client02"),
    ],
)
def test_partial_request_identity_never_uses_deployment_defaults(
    monkeypatch,
    email: str | None,
    instance: str | None,
) -> None:
    class LocalSettings:
        app_environment = "local"
        rp_default_email = "fallback@example.com"
        rp_instance = "Universal"

    settings_calls: list[bool] = []

    def settings():
        settings_calls.append(True)
        return LocalSettings()

    monkeypatch.setattr("app.identity.get_settings", settings)

    with pytest.raises(RequestIdentityError, match="must be supplied together"):
        resolve_request_identity(email, instance)
    assert settings_calls == []


def test_email_ownership_is_case_insensitive_without_altering_outbound_value() -> None:
    original = RequestIdentity(" Employee@Company.com ", " Universal ")
    same_owner = RequestIdentity("employee@company.com", "Universal")

    assert original == same_owner
    assert hash(original) == hash(same_owner)
    assert original.email == "Employee@Company.com"
    assert original.instance == "Universal"


def test_email_case_and_whitespace_reuse_the_same_session_owner() -> None:
    store = InMemorySessionStore()
    with identity_scope(" Employee@Company.com ", " Universal "):
        session_id = store.ensure_session("normalized-owner-session")
        store.append_history(session_id, "user", "Private history")

    with identity_scope("employee@company.com", "Universal"):
        assert store.ensure_session(session_id) == session_id
        assert store.get_history(session_id) == [
            {"role": "user", "content": "Private history"}
        ]


def test_whitespace_cannot_disguise_a_different_session_owner() -> None:
    store = InMemorySessionStore()
    with identity_scope("employee@company.com", "Universal"):
        session_id = store.ensure_session("whitespace-owner-session")

    with identity_scope(" manager@company.com ", " Universal "):
        with pytest.raises(SessionIdentityMismatch):
            store.ensure_session(session_id)


def test_pending_action_owner_accepts_only_normalized_same_email() -> None:
    store = InMemorySessionStore()
    with identity_scope(" Employee@Company.com ", " Universal "):
        session_id = store.ensure_session("normalized-pending-owner")
        action = store.create_pending_action(
            session_id,
            action_type="cancel_day_type",
            validated_arguments={"mapping_id": "request-1"},
            summary="Cancel request?",
            language="en",
        )

    with identity_scope(" manager@company.com ", " Universal "):
        with pytest.raises(SessionIdentityMismatch):
            store.consume_pending_action(session_id, action.confirmation_id)

    with identity_scope(" employee@company.com ", "Universal"):
        consumed = store.consume_pending_action(session_id, action.confirmation_id)

    assert consumed.confirmation_id == action.confirmation_id
    assert consumed.owner.email == "Employee@Company.com"


def test_llm_tool_contract_does_not_expose_identity_arguments() -> None:
    forbidden = {"email", "instance", "usrEmail", "instanceName"}
    for tool in TOOL_DEFINITIONS:
        properties = tool["parameters"].get("properties", {})
        assert forbidden.isdisjoint(properties)


def test_raw_email_is_not_added_to_structured_logs(monkeypatch, caplog) -> None:
    session_store.clear()

    async def run_agent(*args, **kwargs):
        return AgentResult(message="Done", tools_used=[])

    monkeypatch.setattr(chat_service, "run_agent", run_agent)
    with caplog.at_level(logging.INFO):
        response = client.post(
            "/api/chat",
            json={
                "message": "Hello",
                "session_id": "private-log-session",
                "email": "private.employee@example.com",
                "instance": "Universal",
            },
        )

    assert response.status_code == 200
    assert "private.employee@example.com" not in caplog.text
