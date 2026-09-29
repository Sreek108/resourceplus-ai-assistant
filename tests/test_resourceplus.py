import json
import logging
from types import SimpleNamespace

import httpx
import pytest

from app.resourceplus import client as resourceplus_client
from app.resourceplus.attendance import get_attendance_summary
from app.resourceplus.client import (
    ResourcePlusClient,
    ResourcePlusHTTPError,
    ResourcePlusInvalidResponseError,
)
from app.resourceplus.employee import get_profile_data
from app.resourceplus.home import get_home_data


@pytest.mark.asyncio
async def test_attendance_uses_expected_route_and_parameter_casing() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/Mobile/api/AI/AttendanceSummary"
        params = request.url.params
        assert params["usrEmail"] == "employee@example.com"
        assert params["fromDate"] == "2026-09-14"
        assert params["toDate"] == "2026-09-19"
        assert params["instanceName"] == "Universal"
        assert params["lang"] == "1"
        return httpx.Response(200, json={"Attendance Counts": []})

    client = ResourcePlusClient(
        base_url="https://example.test/Mobile",
        transport=httpx.MockTransport(handler),
    )
    result = await get_attendance_summary(
        "2026-09-14",
        "2026-09-19",
        usr_email="employee@example.com",
        instance_name="Universal",
        client=client,
    )
    assert result == {"Attendance Counts": []}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("getter", "expected_path"),
    [
        (get_home_data, "/Mobile/api/Client/GetHomeData"),
        (get_profile_data, "/Mobile/api/Client/GetProfileData"),
    ],
)
async def test_client_routes_preserve_documented_parameter_casing(
    getter,
    expected_path: str,
) -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path == expected_path
        assert request.url.params["instanceName"] == "Universal"
        assert request.url.params["Usremail"] == "employee@example.com"
        assert request.url.params["Lang"] == "1"
        assert "usrEmail" not in request.url.params
        return httpx.Response(200, json={"ok": True})

    client = ResourcePlusClient(
        base_url="https://example.test/Mobile/",
        transport=httpx.MockTransport(handler),
    )
    result = await getter(
        usr_email="employee@example.com",
        instance_name="Universal",
        client=client,
    )
    assert result == {"ok": True}


@pytest.mark.asyncio
async def test_invalid_json_is_reported() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="not json")

    client = ResourcePlusClient(
        base_url="https://example.test/Mobile/",
        transport=httpx.MockTransport(handler),
    )
    with pytest.raises(ResourcePlusInvalidResponseError):
        await client.get("api/Client/GetHomeData", params={})


@pytest.mark.asyncio
async def test_http_error_exposes_only_safe_allowlisted_upstream_diagnostics(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setattr(
        resourceplus_client,
        "get_settings",
        lambda: SimpleNamespace(app_environment="uat"),
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={
                "errorCode": "RP-VALIDATION-500",
                "message": "The exceptional entry could not be processed.",
                "errors": {
                    "entryTime": ["Invalid value"],
                    "reasonID": ["Unknown value"],
                },
                "ignored": "employee@example.com bearer secret-value",
            },
            headers={"X-Correlation-ID": "rp-correlation-123"},
        )

    client = ResourcePlusClient(
        base_url="https://example.test/Mobile/",
        timeout_seconds=1,
        transport=httpx.MockTransport(handler),
    )
    with caplog.at_level(logging.WARNING, logger="app.structured"):
        with pytest.raises(ResourcePlusHTTPError) as captured:
            await client.post(
                "api/AI/ExceptionalEntries/Request",
                params={"instanceName": "Universal"},
                json_body={"private": "body-must-never-be-logged"},
            )

    error = captured.value
    assert error.status_code == 500
    assert error.endpoint == "api/AI/ExceptionalEntries/Request"
    assert error.diagnostic.code == "RP-VALIDATION-500"
    assert error.diagnostic.message == "The exceptional entry could not be processed."
    assert error.diagnostic.correlation_id == "rp-correlation-123"
    assert error.diagnostic.validation_fields == ("entryTime", "reasonID")

    events = [
        json.loads(record.message)
        for record in caplog.records
        if record.name == "app.structured"
        and '"event":"upstream_http_error"' in record.message
    ]
    assert len(events) == 1
    event = events[0]
    assert event["status"] == 500
    assert event["endpoint"] == "api/AI/ExceptionalEntries/Request"
    assert event["upstream_error_code"] == "RP-VALIDATION-500"
    assert event["upstream_error_message"] == (
        "The exceptional entry could not be processed."
    )
    assert event["upstream_correlation_id"] == "rp-correlation-123"
    assert event["validation_fields"] == ["entryTime", "reasonID"]
    serialized = json.dumps(event)
    assert "employee@example.com" not in serialized
    assert "body-must-never-be-logged" not in serialized
    assert "secret-value" not in serialized


@pytest.mark.asyncio
async def test_production_structured_log_omits_upstream_message_details(
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setattr(
        resourceplus_client,
        "get_settings",
        lambda: SimpleNamespace(app_environment="production"),
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            500,
            json={
                "errorCode": "RP-500",
                "message": "Safe but local-only diagnostic.",
                "errors": {"entryTime": ["Invalid"]},
            },
            headers={"X-Correlation-ID": "rp-production-123"},
        )

    client = ResourcePlusClient(
        base_url="https://example.test/Mobile/",
        timeout_seconds=1,
        transport=httpx.MockTransport(handler),
    )
    with caplog.at_level(logging.WARNING, logger="app.structured"):
        with pytest.raises(ResourcePlusHTTPError):
            await client.post(
                "api/AI/ExceptionalEntries/Request",
                params={"instanceName": "Universal"},
                json_body={},
            )

    event = next(
        json.loads(record.message)
        for record in caplog.records
        if record.name == "app.structured"
        and '"event":"upstream_http_error"' in record.message
    )
    assert event["status"] == 500
    assert event["endpoint"] == "api/AI/ExceptionalEntries/Request"
    assert "upstream_error_code" not in event
    assert "upstream_error_message" not in event
    assert "upstream_correlation_id" not in event
    assert "validation_fields" not in event


def test_from_date_must_not_follow_to_date() -> None:
    async def unused_handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("HTTP must not be called")

    client = ResourcePlusClient(
        base_url="https://example.test/Mobile/",
        transport=httpx.MockTransport(unused_handler),
    )

    async def call() -> None:
        await get_attendance_summary(
            "2026-09-20",
            "2026-09-19",
            client=client,
        )

    with pytest.raises(ValueError, match="on or before"):
        import asyncio

        asyncio.run(call())


def test_absolute_resourceplus_route_is_rejected() -> None:
    client = ResourcePlusClient(base_url="https://example.test/Mobile/")
    with pytest.raises(ValueError, match="relative paths"):
        client._build_url("https://attacker.example/api")
