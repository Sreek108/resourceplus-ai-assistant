import json
from datetime import date

import pytest

from app.ai import actions, tools
from app.ai.actions import prepare_write_action
from app.ai.sessions import InMemorySessionStore
from app.ai.tools import execute_tool
from app.resourceplus.missing_punch import (
    missing_punch_tool_data,
    normalize_missing_punch_suggestions,
    parse_missing_punch_date,
)


SEPTEMBER_FIRST_IN = {
    "attDate": "01/09/2026",
    "suggestedEntryTime": "01/09/2026 09:00",
    "entryType": "IN",
    "shift": "General Shift(09:00 : 18:00)",
    "isNightShift": 0,
}


@pytest.mark.asyncio
async def test_informational_and_prepare_flows_share_one_valid_normalization(
    monkeypatch,
) -> None:
    calls: list[str] = []

    async def suggestions(*args, **kwargs):
        calls.append("suggestions")
        return [SEPTEMBER_FIRST_IN]

    async def reasons(*args, **kwargs):
        calls.append("reasons")
        return [{"reasonID": "traffic-live", "reasonName": "Traffic Delay"}]

    monkeypatch.setattr(tools, "get_missing_punch_suggestions", suggestions)
    informational = await execute_tool(
        "get_missing_punch_suggestions",
        {"from_date": "2026-09-01", "to_date": "2026-09-01"},
        lang=1,
        session_id="informational-september-first",
        response_language="en",
    )
    displayed = json.loads(informational.output)["data"]
    assert displayed["missing_punch_count"] == 1
    assert displayed["correctable_suggestion_count"] == 1
    missing = displayed["missing_punches_by_date"][0]
    shown = displayed["correctable_suggestions_by_date"][0]
    assert missing["att_date"] == shown["att_date"] == "2026-09-01"
    assert missing["missing_punches"][0]["entry_type"] == "IN"
    assert missing["missing_punches"][0]["is_missing_punch"] is True
    assert missing["missing_punches"][0]["is_correctable"] is True

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    intent = await prepare_write_action(
        "prepare_exceptional_entry",
        {
            "target_date": "September 1, 2026",
            "punch_direction": None,
            "reason_name": "traffic",
            "remarks": "Traffic delay",
            "_user_message": "traffic",
        },
        lang=1,
        response_language="en",
    )

    assert intent.validated_arguments["attendance_date"] == shown["att_date"]
    assert (
        intent.validated_arguments["entry_time"]
        == shown["suggestions"][0]["suggested_entry_time"]
    )
    assert intent.validated_arguments["entry_type"] == 1
    assert calls == ["suggestions", "suggestions", "reasons"]


@pytest.mark.asyncio
async def test_same_date_in_and_out_are_shown_then_require_direction(
    monkeypatch,
) -> None:
    raw = [
        SEPTEMBER_FIRST_IN,
        {
            **SEPTEMBER_FIRST_IN,
            "suggestedEntryTime": "01/09/2026 18:00",
            "entryType": "OUT",
        },
    ]
    calls: list[str] = []

    async def suggestions(*args, **kwargs):
        calls.append("suggestions")
        return raw

    async def reasons(*args, **kwargs):
        calls.append("reasons")
        return [{"reasonID": "traffic-live", "reasonName": "Traffic Delay"}]

    monkeypatch.setattr(tools, "get_missing_punch_suggestions", suggestions)
    informational = await execute_tool(
        "get_missing_punch_suggestions",
        {"from_date": "2026-09-01", "to_date": "2026-09-01"},
        lang=1,
        session_id="two-punches-read",
        response_language="en",
    )
    shown = json.loads(informational.output)["data"]
    assert [
        item["entry_type"]
        for item in shown["missing_punches_by_date"][0]["missing_punches"]
    ] == ["IN", "OUT"]
    assert [
        item["entry_type"]
        for item in shown["correctable_suggestions_by_date"][0]["suggestions"]
    ] == ["IN", "OUT"]

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    store = InMemorySessionStore()
    session_id = store.ensure_session("two-punches-prepare")
    clarification = await execute_tool(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-01",
            "punch_direction": None,
            "reason_name": "traffic",
            "remarks": "Traffic delay",
        },
        lang=1,
        session_id=session_id,
        response_language="en",
        source_user_message="traffic",
        store=store,
    )
    body = json.loads(clarification.output)
    assert body["requires_clarification"] is True
    assert "both a missing IN and a missing OUT" in body["message"]
    assert "no_resourceplus_suggestion" not in clarification.output
    assert clarification.pending_action is None
    assert calls == ["suggestions", "suggestions"]

    prepared = await execute_tool(
        "prepare_exceptional_entry",
        {
            "target_date": "01/09/2026",
            "punch_direction": "IN",
            "reason_name": "traffic",
            "remarks": "Traffic delay",
        },
        lang=1,
        session_id=session_id,
        response_language="en",
        source_user_message="traffic",
        store=store,
    )
    assert prepared.pending_action is not None
    assert prepared.pending_action.validated_arguments["entry_type"] == 1
    assert prepared.pending_action.validated_arguments["entry_time"] == (
        "01/09/2026 09:00"
    )
    assert calls == ["suggestions", "suggestions", "suggestions", "reasons"]


@pytest.mark.asyncio
async def test_empty_time_rows_remain_visible_but_are_not_correctable(
    monkeypatch,
) -> None:
    live_uat_shape = [
        {
            "attDate": "9/1/2026 12:00:00 AM",
            "suggestedEntryTime": "",
            "entryType": "IN",
            "shift": "General Shift(00:00 : 00:00)",
            "isNightShift": 0,
        },
        {
            "attDate": "9/1/2026 12:00:00 AM",
            "suggestedEntryTime": "",
            "entryType": "OUT",
            "shift": "General Shift(00:00 : 00:00)",
            "isNightShift": 0,
        },
    ]
    reasons_called = False

    async def suggestions(*args, **kwargs):
        return live_uat_shape

    async def reasons(*args, **kwargs):
        nonlocal reasons_called
        reasons_called = True
        return []

    monkeypatch.setattr(tools, "get_missing_punch_suggestions", suggestions)
    informational = await execute_tool(
        "get_missing_punch_suggestions",
        {"from_date": "2026-09-01", "to_date": "2026-09-01"},
        lang=1,
        session_id="uat-empty-time-read",
        response_language="en",
    )
    read_data = json.loads(informational.output)["data"]
    assert read_data["missing_punch_count"] == 2
    assert read_data["missing_punch_date_count"] == 1
    assert [
        item["entry_type"]
        for item in read_data["missing_punches_by_date"][0]["missing_punches"]
    ] == ["IN", "OUT"]
    assert all(
        item["is_missing_punch"] is True and item["is_correctable"] is False
        for item in read_data["missing_punches_by_date"][0]["missing_punches"]
    )
    assert read_data["correctable_suggestion_count"] == 0
    assert read_data["correctable_suggestions_by_date"] == []
    assert read_data["non_correctable_missing_punch_count"] == 2
    assert read_data["invalid_row_count"] == 0
    assert "has not provided valid suggested correction times" in read_data["message"]

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    store = InMemorySessionStore()
    result = await execute_tool(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-01",
            "punch_direction": None,
            "reason_name": "traffic",
            "remarks": "Traffic delay",
        },
        lang=1,
        session_id=store.ensure_session("uat-empty-time-prepare"),
        response_language="en",
        source_user_message="traffic",
        store=store,
    )
    assert result.pending_action is None
    assert "valid suggested punch correction" in json.loads(result.output)["message"]
    assert reasons_called is False


@pytest.mark.asyncio
async def test_mixed_in_correctable_out_informational_only(monkeypatch) -> None:
    raw = [
        SEPTEMBER_FIRST_IN,
        {
            **SEPTEMBER_FIRST_IN,
            "suggestedEntryTime": "",
            "entryType": "OUT",
        },
    ]
    calls: list[str] = []

    async def suggestions(*args, **kwargs):
        calls.append("suggestions")
        return raw

    async def reasons(*args, **kwargs):
        calls.append("reasons")
        return [{"reasonID": "traffic-live", "reasonName": "Traffic Delay"}]

    monkeypatch.setattr(tools, "get_missing_punch_suggestions", suggestions)
    informational = await execute_tool(
        "get_missing_punch_suggestions",
        {"from_date": "2026-09-01", "to_date": "2026-09-01"},
        lang=1,
        session_id="mixed-read",
        response_language="en",
    )
    data = json.loads(informational.output)["data"]
    assert [
        (item["entry_type"], item["is_correctable"])
        for item in data["missing_punches_by_date"][0]["missing_punches"]
    ] == [("IN", True), ("OUT", False)]
    assert [
        item["entry_type"]
        for item in data["correctable_suggestions_by_date"][0]["suggestions"]
    ] == ["IN"]

    monkeypatch.setattr(actions, "get_missing_punch_suggestions", suggestions)
    monkeypatch.setattr(actions, "get_exception_reasons", reasons)
    store = InMemorySessionStore()
    prepared = await execute_tool(
        "prepare_exceptional_entry",
        {
            "target_date": "2026-09-01",
            "punch_direction": None,
            "reason_name": "traffic",
            "remarks": "Traffic delay",
        },
        lang=1,
        session_id=store.ensure_session("mixed-prepare"),
        response_language="en",
        source_user_message="traffic",
        store=store,
    )
    assert prepared.pending_action is not None
    assert prepared.pending_action.validated_arguments["entry_type"] == 1
    assert calls == ["suggestions", "suggestions", "reasons"]


def test_shift_only_row_is_not_a_missing_punch() -> None:
    normalized = normalize_missing_punch_suggestions(
        [
            {
                "attDate": "18/09/2026",
                "suggestedEntryTime": "",
                "shift": "General Shift(00:00 : 00:00)",
                "isNightShift": 0,
            }
        ]
    )
    tool_data = missing_punch_tool_data(normalized)

    assert normalized.missing_punches == ()
    assert normalized.correctable_suggestions == ()
    assert normalized.invalid_row_count == 1
    assert tool_data["missing_punch_count"] == 0
    assert tool_data["missing_punches_by_date"] == []


@pytest.mark.parametrize(
    "value",
    ["01/09/2026", "September 1, 2026", "2026-09-01", "9/1/2026 12:00:00 AM"],
)
def test_missing_punch_dates_normalize_to_same_calendar_date(value: str) -> None:
    assert parse_missing_punch_date(value) == date(2026, 9, 1)


def test_normalizer_separates_missing_punches_from_correctable_suggestions() -> None:
    normalized = normalize_missing_punch_suggestions(
        [
            SEPTEMBER_FIRST_IN,
            {**SEPTEMBER_FIRST_IN, "entryType": "in"},
            {**SEPTEMBER_FIRST_IN, "entryType": "IN/OUT"},
            {**SEPTEMBER_FIRST_IN, "suggestedEntryTime": "   "},
            {
                **SEPTEMBER_FIRST_IN,
                "suggestedEntryTime": "2026-09-01T09:00:00",
            },
            {**SEPTEMBER_FIRST_IN, "attDate": "not-a-date"},
        ]
    )

    assert len(normalized.missing_punches) == 3
    assert len(normalized.correctable_suggestions) == 1
    assert normalized.correctable_suggestions[0].entry_type == "IN"
    assert all(
        row.is_correctable is False for row in normalized.missing_punches[1:]
    )
    assert normalized.invalid_row_count == 3
