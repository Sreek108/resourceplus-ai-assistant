from __future__ import annotations

import pytest

from app.ai.sessions import InMemorySessionStore
from app.models.schemas import ChatRequest
from app.services import chat as chat_service
from app.services import fast_reads
from app.services.response_blocks import profile_block


@pytest.mark.parametrize(
    "payload",
    [
        {"Name": "Test Employee", "Position": "Engineer"},
        {"Data": {"Name": "Test Employee", "Position": "Engineer"}},
        {"Data": [{"Name": "Test Employee", "Position": "Engineer"}]},
        [{"Name": "Test Employee", "Position": "Engineer"}],
        {"Profile": [{"Name": "Test Employee", "Position": "Engineer"}]},
        {"Employee": {"Name": "Test Employee", "Position": "Engineer"}},
    ],
)
def test_profile_block_supports_direct_and_wrapped_shapes(payload) -> None:
    block = profile_block(payload)

    assert block is not None
    assert block.type == "key_value"
    assert block.title == "Profile"
    assert {item.label: item.value for item in block.items} == {
        "Name": "Test Employee",
        "Position": "Engineer",
    }


def test_profile_block_supports_actual_resourceplus_section_shape() -> None:
    block = profile_block(
        {
            "Contact information": {
                "Emp_Number": "TEST-001",
                "EmployeeName": "Test Employee",
                "Emp_Email": "test.employee@example.test",
                "Emp_Mobile": "0000000000",
            },
            "Work Information": [
                {
                    "DateOfJoin": "2020-01-01",
                    "PositionName": "Engineer",
                    "Company": "Test Company",
                    "Organization": "Test Organization",
                }
            ],
            "Skills": [],
            "Certifications": [],
            "StaticContents": [
                {"ContentType": "Heading", "ContentText": "Profile"}
            ],
        }
    )

    assert block is not None
    values = {item.label: item.value for item in block.items}
    assert set(values) == {
        "Employee code",
        "Name",
        "Email",
        "Mobile",
        "Joining date",
        "Position",
        "Company",
        "Organization",
    }
    assert "ContentType" not in values
    assert "ContentText" not in values


@pytest.mark.parametrize(
    "payload",
    [
        {"Data": []},
        {},
        None,
        "unexpected",
        ["unexpected"],
        {"Data": "unexpected"},
    ],
)
def test_profile_block_rejects_empty_or_malformed_payloads(payload) -> None:
    assert profile_block(payload) is None


@pytest.mark.asyncio
async def test_profile_fast_read_returns_profile_block_without_model(monkeypatch) -> None:
    store = InMemorySessionStore()

    async def profile(*args, **kwargs):
        return {
            "Data": [
                {
                    "EmployeeName": "Test Employee",
                    "PositionName": "Engineer",
                }
            ]
        }

    def no_model(*args, **kwargs):
        raise AssertionError("deterministic profile reads must not call OpenAI")

    monkeypatch.setattr(fast_reads, "get_profile_data", profile)
    monkeypatch.setattr(
        __import__("app.ai.agent", fromlist=["AsyncOpenAI"]),
        "AsyncOpenAI",
        no_model,
    )

    response = await chat_service.process_chat(
        ChatRequest(message="Show my profile"),
        store=store,
    )

    assert response.success is True
    assert response.tools_used == ["get_profile_data"]
    assert len(response.blocks) == 1
    assert response.blocks[0].type == "key_value"
    assert response.blocks[0].title == "Profile"
    assert response.message == "Here are your profile details."


@pytest.mark.asyncio
async def test_empty_profile_fast_read_does_not_claim_details_exist(monkeypatch) -> None:
    store = InMemorySessionStore()

    async def profile(*args, **kwargs):
        return {"Data": []}

    monkeypatch.setattr(fast_reads, "get_profile_data", profile)
    response = await chat_service.process_chat(
        ChatRequest(message="Show my profile"),
        store=store,
    )

    assert response.success is True
    assert response.tools_used == ["get_profile_data"]
    assert response.blocks == []
    assert response.message == "I couldn't find any available profile details."
    assert response.speech_message == response.message
