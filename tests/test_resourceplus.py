import httpx
import pytest

from app.resourceplus.attendance import get_attendance_summary
from app.resourceplus.client import (
    ResourcePlusClient,
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
    result = await getter(usr_email="employee@example.com", client=client)
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

