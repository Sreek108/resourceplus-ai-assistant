from types import SimpleNamespace

import pytest

from app.ai import agent
from app.ai.agent import AgentResult, detect_language
from app.ai.sessions import InMemorySessionStore
from app.models.schemas import ChatRequest
from app.services import chat as chat_service


def test_language_detection_handles_natural_arabic_and_english() -> None:
    assert detect_language("وش باقي لي من الإجازات؟") == "ar"
    assert detect_language("How much leave do I have?") == "en"


def test_confirmation_fast_paths_are_not_an_arabic_phrase_map() -> None:
    configured = (
        chat_service.UNAMBIGUOUS_ENGLISH_CONFIRMATIONS
        | chat_service.UNAMBIGUOUS_ENGLISH_REJECTIONS
    )
    assert configured
    assert all(phrase.isascii() for phrase in configured)


@pytest.mark.asyncio
async def test_arabic_leave_balance_uses_deterministic_localized_path(monkeypatch) -> None:
    calls = []

    async def classify(*args, **kwargs):
        return "OTHER"

    async def run(message, *, lang, session_id, history, response_language):
        calls.append((message, lang, history, response_language))
        return AgentResult(message="لديك رصيد إجازة متاح.", tools_used=["get_home_data"])

    async def day_types(*args, **kwargs):
        return []

    async def home(*args, **kwargs):
        return {"EligibleVacation": 8}

    monkeypatch.setattr(chat_service, "classify_confirmation_intent", classify)
    monkeypatch.setattr(chat_service, "run_agent", run)
    monkeypatch.setattr(chat_service, "cached_day_types", day_types)
    monkeypatch.setattr(chat_service, "get_home_data", home)
    response = await chat_service.process_chat(
        ChatRequest(message="وش باقي لي من الإجازات؟"),
        store=InMemorySessionStore(),
    )

    assert calls == []
    assert response.language == "ar"
    assert response.tools_used == ["get_home_data", "get_day_types"]


@pytest.mark.asyncio
async def test_natural_follow_up_receives_existing_session_history(monkeypatch) -> None:
    calls = []
    store = InMemorySessionStore()

    async def classify(*args, **kwargs):
        return "OTHER"

    async def run(message, *, lang, session_id, history, response_language):
        calls.append((message, session_id, history, response_language))
        reply = "Attendance summary." if len(calls) == 1 else "Last week’s summary."
        return AgentResult(message=reply, tools_used=["get_attendance_summary"])

    monkeypatch.setattr(chat_service, "classify_confirmation_intent", classify)
    monkeypatch.setattr(chat_service, "run_agent", run)

    first = await chat_service.process_chat(
        ChatRequest(message="Could you check how my attendance looks this week?"),
        store=store,
    )
    second = await chat_service.process_chat(
        ChatRequest(message="And the week before?", session_id=first.session_id),
        store=store,
    )

    assert second.session_id == first.session_id
    assert calls[1][2] == [
        {
            "role": "user",
            "content": "Could you check how my attendance looks this week?",
        },
        {"role": "assistant", "content": "Attendance summary."},
    ]
    assert calls[0][3] == "en"
    assert calls[1][3] == "en"


@pytest.mark.asyncio
async def test_agent_requests_dynamic_arabic_output(monkeypatch) -> None:
    response = SimpleNamespace(output=[], output_text="هذه إجابة عربية ديناميكية.")

    class FakeResponses:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return response

    responses = FakeResponses()
    monkeypatch.setattr(
        agent,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key="test", openai_model="test-model"),
    )
    monkeypatch.setattr(
        agent,
        "AsyncOpenAI",
        lambda api_key: SimpleNamespace(responses=responses),
    )

    result = await agent.run_agent(
        "أرني أحدث تنبيهاتي",
        lang=1,
        session_id="arabic-agent",
        history=[],
    )

    assert result.message == "هذه إجابة عربية ديناميكية."
    assert "response language is Arabic" in responses.calls[0]["instructions"]


@pytest.mark.asyncio
async def test_arabic_voice_confirmation_executes_exact_stored_action(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session("voice-confirm")
    expected = {"notifcn_id": 41, "read_status": 1}
    action = store.create_pending_action(
        session_id,
        action_type="update_notification_read_status",
        validated_arguments=expected,
        summary="Mark the latest notification as read.",
        language="ar",
    )
    executions = []

    async def classify(*args, **kwargs):
        return "CONFIRM"

    async def execute(action_type, arguments):
        executions.append((action_type, arguments))
        return {"success": True, "message": "Updated"}

    async def render(*args, **kwargs):
        return "تم تحديث التنبيه."

    monkeypatch.setattr(chat_service, "classify_confirmation_intent", classify)
    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    monkeypatch.setattr(chat_service, "render_user_message", render)
    response = await chat_service.process_chat(
        ChatRequest(
            message="أكيد",
            session_id=session_id,
            confirmation_id=action.confirmation_id,
        ),
        detected_language="ar",
        store=store,
    )

    assert response.success is True
    assert response.language == "ar"
    assert executions == [("update_notification_read_status", expected)]


@pytest.mark.asyncio
async def test_arabic_voice_rejection_discards_without_execution(monkeypatch) -> None:
    store = InMemorySessionStore()
    session_id = store.ensure_session("voice-reject")
    store.create_pending_action(
        session_id,
        action_type="update_notification_read_status",
        validated_arguments={"notifcn_id": 41, "read_status": 1},
        summary="Mark the latest notification as read.",
        language="ar",
    )

    async def classify(*args, **kwargs):
        return "REJECT"

    async def execute(*args, **kwargs):
        raise AssertionError("Rejected actions must not execute")

    async def render(*args, **kwargs):
        return "تم إلغاء الإجراء."

    monkeypatch.setattr(chat_service, "classify_confirmation_intent", classify)
    monkeypatch.setattr(chat_service, "execute_pending_action", execute)
    monkeypatch.setattr(chat_service, "render_user_message", render)
    response = await chat_service.process_chat(
        ChatRequest(message="لا، خلاص", session_id=session_id),
        detected_language="ar",
        store=store,
    )

    assert response.success is True
    assert store.get_pending_action(session_id)[0] is None
