from datetime import date

import pytest

from app.ai import tools
from app.ai.tools import execute_tool, resolve_relative_date_range


TODAY = date(2026, 9, 19)


def test_today() -> None:
    resolved = resolve_relative_date_range("Show my attendance today", today=TODAY)
    assert resolved is not None
    assert resolved.from_date == TODAY
    assert resolved.to_date == TODAY


def test_yesterday() -> None:
    resolved = resolve_relative_date_range("attendance yesterday", today=TODAY)
    assert resolved is not None
    assert resolved.from_date == date(2026, 9, 18)
    assert resolved.to_date == date(2026, 9, 18)


def test_this_week_runs_monday_through_today() -> None:
    resolved = resolve_relative_date_range("Show my attendance this week", today=TODAY)
    assert resolved is not None
    assert resolved.from_date == date(2026, 9, 14)
    assert resolved.to_date == TODAY


def test_last_week_runs_monday_through_sunday() -> None:
    resolved = resolve_relative_date_range("Show last week", today=TODAY)
    assert resolved is not None
    assert resolved.from_date == date(2026, 9, 7)
    assert resolved.to_date == date(2026, 9, 13)


def test_this_month_runs_from_first_through_today() -> None:
    resolved = resolve_relative_date_range("this month", today=TODAY)
    assert resolved is not None
    assert resolved.from_date == date(2026, 9, 1)
    assert resolved.to_date == TODAY


def test_previous_month_is_complete_calendar_month() -> None:
    resolved = resolve_relative_date_range("Previous month", today=date(2026, 10, 1))
    assert resolved is not None
    assert resolved.label == "previous_month"
    assert resolved.from_date == date(2026, 9, 1)
    assert resolved.to_date == date(2026, 9, 30)


@pytest.mark.parametrize(
    ("message", "expected_start"),
    [
        ("اعرض حضوري اليوم", TODAY),
        ("اعرض حضوري هذا الأسبوع", date(2026, 9, 14)),
        ("اعرض حضوري هذا الشهر", date(2026, 9, 1)),
    ],
)
def test_supported_arabic_relative_period_cues(message, expected_start) -> None:
    resolved = resolve_relative_date_range(message, today=TODAY)
    assert resolved is not None
    assert resolved.from_date == expected_start
    assert resolved.to_date == TODAY


@pytest.mark.asyncio
async def test_request_status_this_month_uses_full_calendar_month(monkeypatch) -> None:
    calls: list[tuple[object, object]] = []

    async def request_status(start, end, **kwargs):
        calls.append((start, end))
        return {"merged_requests": []}

    monkeypatch.setattr(tools, "get_my_request_status", request_status)
    resolved = resolve_relative_date_range("Show my requests this month", today=TODAY)
    assert resolved is not None
    await execute_tool(
        "get_my_request_status",
        {"from_date": "ignored", "to_date": "ignored"},
        lang=1,
        session_id="request-month",
        response_language="en",
        resolved_range=resolved,
    )

    assert calls == [("2026-09-01", "2026-09-30")]


@pytest.mark.asyncio
async def test_attendance_this_month_stops_at_today(monkeypatch) -> None:
    calls: list[tuple[object, object]] = []

    async def attendance(start, end, **kwargs):
        calls.append((start, end))
        return {"Days": []}

    monkeypatch.setattr(tools, "get_attendance_summary", attendance)
    resolved = resolve_relative_date_range("Show attendance this month", today=TODAY)
    assert resolved is not None
    await execute_tool(
        "get_attendance_summary",
        {"from_date": "ignored", "to_date": "ignored"},
        lang=1,
        session_id="attendance-month",
        response_language="en",
        resolved_range=resolved,
    )

    assert calls == [("2026-09-01", "2026-09-19")]
