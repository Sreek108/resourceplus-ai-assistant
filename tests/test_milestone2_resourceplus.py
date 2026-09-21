import json
import logging
from datetime import date
from types import SimpleNamespace

import httpx
import pytest

from app.ai import tools as ai_tools
from app.ai.tools import DateRange, execute_tool
from app.resourceplus import approvals
from app.resourceplus.approvals import (
    approve_all_requests,
    approve_supervisor_request,
    get_pending_approvals,
)
from app.resourceplus.attendance import (
    create_exceptional_entry,
    get_exception_reasons,
    get_missing_punch_suggestions,
)
from app.resourceplus.leave import (
    book_day_type,
    cancel_day_type_request,
    get_day_types,
    get_my_day_type_requests,
)
from app.resourceplus.requests import (
    ResourcePlusRequestStatusError,
    get_my_request_status,
)
from app.resourceplus.client import ResourcePlusClient, ResourcePlusConfigurationError


def mock_client(handler) -> ResourcePlusClient:
    return ResourcePlusClient(
        base_url="https://example.test/Mobile/",
        transport=httpx.MockTransport(handler),
    )


@pytest.mark.asyncio
async def test_missing_punch_reads_and_exceptional_entry_post_contract() -> None:
    requests: list[httpx.Request] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        if request.url.path.endswith("/MissingPunchSuggestions"):
            assert request.method == "GET"
            assert request.url.params["usrEmail"] == "employee@example.com"
            assert request.url.params["fromDate"] == "2026-09-16"
            return httpx.Response(
                200,
                json=[
                    {
                        "attDate": "16/09/2026",
                        "suggestedEntryTime": "16/09/2026 09:00",
                        "entryType": "IN",
                    }
                ],
            )
        if request.url.path.endswith("/ExceptionalEntries/Reasons"):
            assert request.method == "GET"
            assert "usrEmail" not in request.url.params
            return httpx.Response(
                200,
                json=[{"reasonID": "reason-real", "reasonName": "Traffic"}],
            )
        assert request.url.path.endswith("/ExceptionalEntries/Request")
        assert request.method == "POST"
        assert request.url.params["instanceName"] == "Universal"
        body = __import__("json").loads(request.content)
        assert body == {
            "usrEmail": "employee@example.com",
            "entryTime": "2026-09-16T09:00:00",
            "entryType": 1,
            "reasonID": "reason-real",
            "remarks": "Delayed due to traffic",
        }
        return httpx.Response(200, json={"success": True, "message": "Submitted"})

    client = mock_client(handler)
    suggestions = await get_missing_punch_suggestions(
        "2026-09-16",
        "2026-09-16",
        usr_email="employee@example.com",
        client=client,
    )
    reasons = await get_exception_reasons(client=client)
    result = await create_exceptional_entry(
        "2026-09-16T09:00:00",
        1,
        reasons[0]["reasonID"],
        "Delayed due to traffic",
        usr_email="employee@example.com",
        client=client,
    )
    assert suggestions[0]["entryType"] == "IN"
    assert result["success"] is True
    assert [request.method for request in requests] == ["GET", "GET", "POST"]


@pytest.mark.asyncio
async def test_leave_endpoints_preserve_contract_and_conflict_message() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/DayTypes"):
            return httpx.Response(
                200,
                json=[{"dayID": 20, "dayType": "Annual Leave", "group": "Leave"}],
            )
        if request.url.path.endswith("/DayTypeMapping/MyRequests"):
            assert request.url.params["dateFrom"] == "2026-09-22"
            assert request.url.params["dateTo"] == "2026-09-24"
            return httpx.Response(
                200,
                json=[
                    {
                        "mappingID": "mapping-real",
                        "dayType": "Annual Leave",
                        "dateFrom": "22/09/2026",
                        "dateTo": "24/09/2026",
                        "status": "Pending",
                    }
                ],
            )
        body = __import__("json").loads(request.content)
        if request.url.path.endswith("/DayTypeMapping/Book"):
            assert body["dayTypeID"] == 20
            assert body["usrEmail"] == "employee@example.com"
            return httpx.Response(
                200,
                json={
                    "success": False,
                    "message": "A request already exists for an overlapping date range",
                },
            )
        assert request.url.path.endswith("/DayTypeMapping/Cancel")
        assert body == {
            "usrEmail": "employee@example.com",
            "mappingID": "mapping-real",
        }
        return httpx.Response(200, json={"success": True, "message": "Cancelled"})

    client = mock_client(handler)
    day_types = await get_day_types(client=client)
    requests = await get_my_day_type_requests(
        "2026-09-22",
        "2026-09-24",
        usr_email="employee@example.com",
        client=client,
    )
    conflict = await book_day_type(
        "2026-09-22",
        "2026-09-24",
        day_types[0]["dayID"],
        usr_email="employee@example.com",
        client=client,
    )
    cancelled = await cancel_day_type_request(
        requests[0]["mappingID"],
        usr_email="employee@example.com",
        client=client,
    )
    assert conflict == {
        "success": False,
        "message": "A request already exists for an overlapping date range",
    }
    assert cancelled["success"] is True


@pytest.mark.asyncio
async def test_request_status_calls_both_apis_and_preserves_raw_status() -> None:
    called: set[str] = set()

    async def handler(request: httpx.Request) -> httpx.Response:
        called.add(request.url.path)
        if request.url.path.endswith("/DayTypeMapping/MyRequests"):
            assert set(request.url.params) == {
                "usrEmail",
                "instanceName",
                "lang",
                "dateFrom",
                "dateTo",
            }
            assert request.url.params["dateFrom"] == "2026-09-01"
            assert request.url.params["dateTo"] == "2026-09-30"
            return httpx.Response(
                200,
                json=[{"mappingID": "m1", "status": "Pending"}],
            )
        assert request.url.path.endswith("/ExceptionalEntries")
        assert set(request.url.params) == {
            "usrEmail",
            "fromDate",
            "toDate",
            "instanceName",
            "lang",
        }
        assert request.url.params["fromDate"] == "09/01/2026"
        assert request.url.params["toDate"] == "09/30/2026"
        return httpx.Response(
            200,
            json=[{"exceptionalID": "e1", "status": "Not Approved"}],
        )

    result = await get_my_request_status(
        "2026-09-01",
        "2026-09-30",
        client=mock_client(handler),
    )
    assert len(called) == 2
    assert result["absence_requests"][0]["status"] == "Pending"
    assert result["exceptional_entry_requests"][0]["status"] == "Not Approved"
    assert [item["raw_status"] for item in result["merged_requests"]] == [
        "Pending",
        "Not Approved",
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("failed_endpoint", "expected_log_endpoint"),
    [
        ("day_type", "api/AI/DayTypeMapping/MyRequests"),
        ("exceptional", "api/AI/ExceptionalEntries"),
    ],
)
async def test_request_status_logs_the_failing_sub_call_without_identity(
    failed_endpoint: str,
    expected_log_endpoint: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        is_day_type = request.url.path.endswith("/DayTypeMapping/MyRequests")
        should_fail = (failed_endpoint == "day_type" and is_day_type) or (
            failed_endpoint == "exceptional" and not is_day_type
        )
        if should_fail:
            return httpx.Response(500, json={"Message": "An error has occurred."})
        return httpx.Response(200, json=[])

    with caplog.at_level(logging.WARNING):
        with pytest.raises(
            ResourcePlusRequestStatusError,
            match="request status is temporarily unavailable",
        ) as exc_info:
            await get_my_request_status(
                "2026-09-01",
                "2026-09-30",
                client=mock_client(handler),
            )

    assert expected_log_endpoint in caplog.text
    assert "http_status=500" in caplog.text
    assert "employee@example.com" not in caplog.text
    assert "DayTypeMapping/MyRequests" not in str(exc_info.value)
    assert "ExceptionalEntries" not in str(exc_info.value)


@pytest.mark.asyncio
async def test_request_status_tool_returns_only_a_safe_error(monkeypatch) -> None:
    async def fail_request_status(*args, **kwargs):
        raise ResourcePlusRequestStatusError(
            "ResourcePlus request status is temporarily unavailable. Please try again."
        )

    monkeypatch.setattr(ai_tools, "get_my_request_status", fail_request_status)
    result = await execute_tool(
        "get_my_request_status",
        {"from_date": "2026-09-01", "to_date": "2026-09-30"},
        lang=1,
        session_id="request-status-test",
        response_language="en",
        resolved_range=DateRange(
            "this_month",
            date(2026, 9, 1),
            date(2026, 9, 30),
        ),
    )

    payload = json.loads(result.output)
    assert result.failed is True
    assert payload == {
        "success": False,
        "error": (
            "ResourcePlus request status is temporarily unavailable. Please try again."
        ),
    }
    assert "DayTypeMapping/MyRequests" not in result.output
    assert "ExceptionalEntries" not in result.output


@pytest.mark.asyncio
async def test_supervisor_contract_preserves_request_type(monkeypatch) -> None:
    monkeypatch.setattr(
        approvals,
        "get_settings",
        lambda: SimpleNamespace(
            rp_manager_email="manager@example.com",
            rp_instance="Universal",
        ),
    )
    posts: list[dict] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET":
            assert request.url.params["usrEmail"] == "manager@example.com"
            return httpx.Response(
                200,
                json=[
                    {
                        "requestId": "request-real",
                        "requestType": "ExceptionEntry",
                    }
                ],
            )
        body = __import__("json").loads(request.content)
        posts.append(body)
        return httpx.Response(200, json={"success": True, "message": "Done"})

    client = mock_client(handler)
    await get_pending_approvals(client=client)
    await approve_supervisor_request(
        "request-real",
        "ExceptionEntry",
        2,
        client=client,
    )
    await approve_all_requests(1, "Absence", client=client)
    assert posts[0]["requestType"] == "ExceptionEntry"
    assert posts[0]["status"] == 2
    assert posts[1]["requestType"] == "Absence"
    assert posts[1]["status"] == 1


@pytest.mark.asyncio
async def test_supervisor_requires_configured_manager(monkeypatch) -> None:
    monkeypatch.setattr(
        approvals,
        "get_settings",
        lambda: SimpleNamespace(rp_manager_email=None, rp_instance="Universal"),
    )
    with pytest.raises(ResourcePlusConfigurationError, match="RP_MANAGER_EMAIL"):
        await get_pending_approvals(client=mock_client(lambda request: None))
