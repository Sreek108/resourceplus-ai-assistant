from types import SimpleNamespace

import pytest

from app.ai import agent
from app.ai.prompts import SYSTEM_PROMPT
from app.ai.tools import ToolExecutionResult


@pytest.mark.asyncio
async def test_existing_read_tool_loop_still_returns_final_answer(monkeypatch) -> None:
    first = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name="get_profile_data",
                arguments="{}",
                call_id="call-1",
            )
        ],
        output_text="",
    )
    second = SimpleNamespace(
        output=[],
        output_text=(
            '{"display_message":"### Profile\\n\\nYour profile is available.",'
            '"speech_message":"Your profile is available."}'
        ),
    )

    class FakeResponses:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return first if len(self.calls) == 1 else second

    responses = FakeResponses()
    fake_client = SimpleNamespace(responses=responses)
    monkeypatch.setattr(
        agent,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key="test", openai_model="test-model"),
    )
    monkeypatch.setattr(agent, "AsyncOpenAI", lambda api_key: fake_client)

    async def execute(*args, **kwargs):
        return ToolExecutionResult(
            output='{"success":true,"data":{"EmployeeName":"Demo"}}'
        )

    monkeypatch.setattr(agent, "execute_tool", execute)
    result = await agent.run_agent(
        "Show my profile",
        lang=1,
        session_id="session-1",
        history=[],
    )
    assert result.message == "### Profile\n\nYour profile is available."
    assert result.speech_message == "Your profile is available."
    assert result.tools_used == ["get_profile_data"]
    assert len(responses.calls) == 2
    assert responses.calls[0]["parallel_tool_calls"] is False
    assert responses.calls[0]["text"] == agent.ASSISTANT_MESSAGES_TEXT_CONFIG
    assert any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in responses.calls[1]["input"]
    )


@pytest.mark.asyncio
async def test_terminal_clarification_stops_tool_loop_after_one_model_call(
    monkeypatch,
) -> None:
    tool_call = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name="prepare_exceptional_entry",
                arguments=(
                    '{"target_date":"2026-09-01","punch_direction":"IN",'
                    '"reason_name":null,"remarks":null}'
                ),
                call_id="reason-required",
            )
        ],
        output_text="",
    )

    class FakeResponses:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            if len(self.calls) > 1:
                raise AssertionError("A terminal clarification must not re-enter OpenAI")
            return tool_call

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

    async def execute(*args, **kwargs):
        return ToolExecutionResult(
            output='{"success":true,"needs_reason":true}',
            terminal_message="What was the reason?",
            needs_reason=True,
            reason_options=["Embassy Purposes", "Family Circumstances"],
        )

    monkeypatch.setattr(agent, "execute_tool", execute)
    result = await agent.run_agent(
        "Correct my September 1 IN punch",
        lang=1,
        session_id="terminal-clarification",
        history=[],
    )

    assert result.message == "What was the reason?"
    assert result.speech_message == "What was the reason?"
    assert result.requires_confirmation is False
    assert result.needs_reason is True
    assert result.reason_options == ["Embassy Purposes", "Family Circumstances"]
    assert result.tools_used == ["prepare_exceptional_entry"]
    assert len(responses.calls) == 1


def test_system_prompt_prefers_conversation_without_weakening_authority() -> None:
    assert "Answer the user's direct question first" in SYSTEM_PROMPT
    assert "one to three short sentences" in SYSTEM_PROMPT
    assert "full profile" in SYSTEM_PROMPT
    assert "do not turn the answer into a report" in SYSTEM_PROMPT
    assert "conversation history" in SYSTEM_PROMPT
    assert "ResourcePlus reads" in SYSTEM_PROMPT
    assert "freshness rule is mandatory on every turn" in SYSTEM_PROMPT
    assert "substitute for the current lookup" in SYSTEM_PROMPT
    assert "remove any sentence whose only purpose is to offer" in SYSTEM_PROMPT
    assert "Do not ask an unrequested follow-up question" in SYSTEM_PROMPT
    assert "Do not append unrequested advice" in SYSTEM_PROMPT
    assert "without pretending to be a human employee" in SYSTEM_PROMPT
    assert "fixed Arabic command" in SYSTEM_PROMPT
    assert "professional Saudi conversational Arabic" in SYSTEM_PROMPT
    assert "less formal than written MSA" in SYSTEM_PROMPT
    assert "without exaggerated slang" in SYSTEM_PROMPT
    assert "English HR terms" in SYSTEM_PROMPT
    assert "must never change" in SYSTEM_PROMPT
    assert "Never ask the employee for an exceptional-entry reason before calling" in (
        SYSTEM_PROMPT
    )
    assert "AttendanceSummary LessHrs alone never proves a missing IN or OUT" in (
        SYSTEM_PROMPT
    )
    assert "NetHrs is the time actually worked" in SYSTEM_PROMPT
    assert "LessHrs is the" in SYSTEM_PROMPT
    assert "worked 40 minutes and was 7 hours 20 minutes short" in SYSTEM_PROMPT
    assert "only a prepare_* tool" in SYSTEM_PROMPT


@pytest.mark.asyncio
async def test_main_agent_produces_display_and_speech_in_one_model_request(
    monkeypatch,
) -> None:
    response = SimpleNamespace(
        output=[],
        output_text=(
            '{"display_message":"### Attendance\\n\\n- Monday: Absent\\n- Tuesday: Present",'
            '"speech_message":"You were absent on Monday. The breakdown is on screen."}'
        ),
    )

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
        "How was my attendance?",
        lang=1,
        session_id="one-turn",
        history=[],
    )

    assert result.message.startswith("### Attendance")
    assert result.speech_message == (
        "You were absent on Monday. The breakdown is on screen."
    )
    assert len(responses.calls) == 1
    instructions = responses.calls[0]["instructions"]
    assert "display_message" in instructions
    assert "speech_message" in instructions
    assert responses.calls[0]["text"] == agent.ASSISTANT_MESSAGES_TEXT_CONFIG


@pytest.mark.asyncio
async def test_arabic_display_and_saudi_speech_are_equivalent_and_preserve_code_switching(
    monkeypatch,
) -> None:
    display = "وفقًا للسجل، لديك 12 يومًا، وبيانات attendance لهذا الأسبوع مكتملة."
    speech = "حسب السجل، عندك 12 يوم، وبيانات attendance حق هالأسبوع مكتملة."
    response = SimpleNamespace(
        output=[],
        output_text=(
            '{"display_message":"'
            + display
            + '","speech_message":"'
            + speech
            + '"}'
        ),
    )

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
        "أبغى أشوف attendance حقي، وعندي كم يوم؟",
        lang=1,
        session_id="arabic-style",
        history=[],
        response_language="ar",
    )

    assert result.message == display
    assert result.speech_message == speech
    assert result.message != result.speech_message
    # Both representations retain the same authoritative quantity and status.
    assert "12" in result.message and "12" in result.speech_message
    assert "مكتملة" in result.message and "مكتملة" in result.speech_message
    assert "attendance" in result.message and "attendance" in result.speech_message
    instructions = responses.calls[0]["instructions"]
    assert "naturally Saudi conversational" in instructions
    assert "overly formal MSA" in instructions
    assert "exaggerated slang" in instructions
    assert "English HR terms" in instructions
    assert "factually and transactionally equivalent" in instructions


@pytest.mark.asyncio
async def test_employee_data_follow_up_forces_fresh_read_tool(monkeypatch) -> None:
    tool_call = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name="get_attendance_summary",
                arguments='{"date_from":"2026-09-13","date_to":"2026-09-19"}',
                call_id="follow-up-read",
            )
        ],
        output_text="",
    )
    final = SimpleNamespace(
        output=[],
        output_text=(
            '{"display_message":"Those days had missing hours.",'
            '"speech_message":"Those days had missing hours."}'
        ),
    )

    class FakeResponses:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return [tool_call, final][len(self.calls) - 1]

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

    executed = []

    async def execute(*args, **kwargs):
        executed.append((args, kwargs))
        return ToolExecutionResult(output='{"success":true,"data":{"Days":[]}}')

    monkeypatch.setattr(agent, "execute_tool", execute)
    result = await agent.run_agent(
        "What about last week?",
        lang=1,
        session_id="follow-up-session",
        history=[
            {"role": "user", "content": "How was my attendance this week?"},
            {"role": "assistant", "content": "You were absent on several days."},
        ],
    )

    assert result.tools_used == ["get_attendance_summary"]
    assert len(executed) == 1
    assert executed[0][0][0] == "get_attendance_summary"
    assert len(responses.calls) == 2
    assert "tool_choice" not in responses.calls[0]
    assert responses.calls[0]["input"][:2] == [
        {"role": "user", "content": "How was my attendance this week?"},
        {"role": "assistant", "content": "You were absent on several days."},
    ]
    assert any(
        isinstance(item, dict) and item.get("type") == "function_call_output"
        for item in responses.calls[1]["input"]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("message", "history", "expected_language"),
    [
        (
            "Any unread ones?",
            [
                {"role": "user", "content": "Show my notifications"},
                {"role": "assistant", "content": "You have three notifications."},
            ],
            "English",
        ),
        (
            "وش منها غير مقروء؟",
            [
                {"role": "user", "content": "أرني تنبيهاتي"},
                {"role": "assistant", "content": "لديك ثلاثة تنبيهات."},
            ],
            "Arabic",
        ),
    ],
)
async def test_notification_follow_up_uses_fresh_read_without_classifier(
    monkeypatch,
    message: str,
    history: list[dict[str, str]],
    expected_language: str,
) -> None:
    tool_call = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="function_call",
                name="get_notifications",
                arguments='{"top":10}',
                call_id="fresh-notifications",
            )
        ],
        output_text="",
    )
    final = SimpleNamespace(
        output=[],
        output_text=(
            '{"display_message":"Current unread notifications are on screen.",'
            '"speech_message":"Your current unread notifications are on screen."}'
        ),
    )

    class FakeResponses:
        def __init__(self):
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return [tool_call, final][len(self.calls) - 1]

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
    executed = []

    async def execute(*args, **kwargs):
        executed.append((args, kwargs))
        return ToolExecutionResult(output='{"success":true,"data":[]}')

    monkeypatch.setattr(agent, "execute_tool", execute)

    result = await agent.run_agent(
        message,
        lang=1,
        session_id="notification-follow-up",
        history=history,
    )

    assert result.tools_used == ["get_notifications"]
    assert len(executed) == 1
    assert executed[0][0][0] == "get_notifications"
    assert len(responses.calls) == 2
    assert responses.calls[0]["input"][:2] == history
    assert f"response language is {expected_language}" in responses.calls[0][
        "instructions"
    ]
