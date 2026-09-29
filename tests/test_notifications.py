import json

import httpx
import pytest

from app.ai import actions, tools
from app.ai.sessions import InMemorySessionStore
from app.ai.tools import execute_tool
from app.models.schemas import ChatRequest
from app.resourceplus.client import ResourcePlusClient
from app.resourceplus.notifications import (
    get_notifications,
    update_notification_read_status,
)
from app.services import chat as chat_service


def mock_client(handler) -> ResourcePlusClient:
    return ResourcePlusClient(
        base_url="https://example.test/Mobile/",
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_notification_service_uses_backend_identity_and_contract() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        assert request.url.params["usrEmail"] == "employee@example.com"
        assert request.url.params["instanceName"] == "Universal"
        assert request.url.params["lang"] == "1"
        if request.url.path.endswith("/GetNotifcnData"):
            return httpx.Response(
                200,
                json={
                    "Notifications": [
                        {
                            "NotifcnID": 41,
                            "NotifcnTitle": "Leave update",
                            "ReadStatus": 0,
                        }
                    ],
                    "StaticContents": [],
                    "CommonContents": [],
                },
            )
        assert request.url.path.endswith("/UpdateReadStatus")
        assert request.url.params["notifcnID"] == "41"
        assert request.url.params["readStatus"] == "1"
        return httpx.Response(200, json={"success": True, "message": "Updated"})

    client = mock_client(handler)
    records = await get_notifications(
        usr_email="employee@example.com",
        instance_name="Universal",
        client=client,
    )
    result = await update_notification_read_status(
        41,
        1,
        usr_email="employee@example.com",
        instance_name="Universal",
        client=client,
    )

    assert records[0]["ReadStatus"] == 0
    assert result["success"] is True
    assert len(requests) == 2


@pytest.mark.asyncio
async def test_notification_read_tool_preserves_status_and_hides_query_string(
    monkeypatch,
) -> None:
    async def notifications(*args, **kwargs):
        return [
            {
                "NotifcnID": 40,
                "NotifcnTitle": "Older alert",
                "NotifcnDate": "19/09/2026 09:00:00 AM",
                "ReadStatus": 1,
                "QueryString": "internal=40",
            },
            {
                "NotifcnID": 41,
                "NotifcnTitle": "Latest alert",
                "NotifcnDate": "20/09/2026 10:00:00 AM",
                "ReadStatus": 0,
                "QueryString": "internal=41",
            },
        ]

    monkeypatch.setattr(tools, "get_notifications", notifications)
    result = await execute_tool(
        "get_notifications",
        {},
        lang=1,
        session_id="notifications-read",
        response_language="en",
    )
    payload = json.loads(result.output)

    assert result.failed is False
    assert [item["ReadStatus"] for item in payload["data"]] == [1, 0]
    assert "QueryString" not in result.output


@pytest.mark.asyncio
async def test_prepare_latest_notification_stores_real_id_without_write(
    monkeypatch,
) -> None:
    writes: list[tuple[int, int]] = []

    async def notifications(*args, **kwargs):
        return [
            {
                "NotifcnID": 40,
                "NotifcnTitle": "Older alert",
                "NotifcnDate": "19-09-2026 09:00",
                "ReadStatus": 0,
            },
            {
                "NotifcnID": 41,
                "NotifcnTitle": "Latest alert",
                "NotifcnDate": "20-09-2026 10:00",
                "ReadStatus": 0,
            },
        ]

    async def update(notifcn_id, read_status):
        writes.append((notifcn_id, read_status))
        return {"success": True, "message": "Updated"}

    monkeypatch.setattr(actions, "get_notifications", notifications)
    monkeypatch.setattr(actions, "update_notification_read_status", update)
    store = InMemorySessionStore()
    session_id = store.ensure_session("notification-latest")
    prepared = await execute_tool(
        "prepare_notification_read_status",
        {
            "target": "latest",
            "notification_title": None,
            "read_status": 1,
            "notifcn_id": 999,
        },
        lang=1,
        session_id=session_id,
        response_language="en",
        store=store,
    )

    pending, _ = store.get_pending_action(session_id)
    assert prepared.pending_action is not None
    assert pending is not None
    assert pending.validated_arguments["notifcn_id"] == 41
    assert writes == []

    async def render(facts, **kwargs):
        return facts

    monkeypatch.setattr(chat_service, "render_user_message", render)
    confirmed = await chat_service.process_chat(
        ChatRequest(message="Yes", session_id=session_id),
        store=store,
    )

    assert confirmed.success is True
    assert writes == [(41, 1)]


@pytest.mark.asyncio
async def test_prepare_all_notifications_uses_documented_zero_id(monkeypatch) -> None:
    async def notifications(*args, **kwargs):
        return [
            {"NotifcnID": 40, "NotifcnTitle": "One"},
            {"NotifcnID": 41, "NotifcnTitle": "Two"},
        ]

    monkeypatch.setattr(actions, "get_notifications", notifications)
    intent = await actions.prepare_write_action(
        "prepare_notification_read_status",
        {
            "target": "all",
            "notification_title": None,
            "read_status": 1,
        },
        lang=1,
        response_language="en",
    )

    assert intent.validated_arguments["notifcn_id"] == 0
    assert intent.validated_arguments["count"] == 2


@pytest.mark.asyncio
async def test_prepare_one_notification_requires_exact_unique_title(monkeypatch) -> None:
    async def notifications(*args, **kwargs):
        return [
            {"NotifcnID": 40, "NotifcnTitle": "Leave update"},
            {"NotifcnID": 41, "NotifcnTitle": "Payroll alert"},
        ]

    monkeypatch.setattr(actions, "get_notifications", notifications)
    intent = await actions.prepare_write_action(
        "prepare_notification_read_status",
        {
            "target": "one",
            "notification_title": "leave update",
            "read_status": 0,
        },
        lang=1,
        response_language="en",
    )

    assert intent.validated_arguments["notifcn_id"] == 40
    assert intent.validated_arguments["read_status"] == 0
