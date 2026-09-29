import re
from datetime import date, datetime
from typing import Any

from app.config import DEFAULT_RESOURCEPLUS_LANG
from app.identity import resourceplus_identity
from app.resourceplus.client import ResourcePlusClient


ATTENDANCE_SUMMARY_ROUTE = "api/AI/AttendanceSummary"
MISSING_PUNCH_SUGGESTIONS_ROUTE = "api/AI/MissingPunchSuggestions"
EXCEPTION_REASONS_ROUTE = "api/AI/ExceptionalEntries/Reasons"
CREATE_EXCEPTIONAL_ENTRY_ROUTE = "api/AI/ExceptionalEntries/Request"


def _parse_iso_date(value: date | str, field_name: str) -> date:
    if isinstance(value, date):
        return value
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must use YYYY-MM-DD format.") from exc


async def get_attendance_summary(
    from_date: date | str,
    to_date: date | str,
    lang: int = DEFAULT_RESOURCEPLUS_LANG,
    usr_email: str | None = None,
    instance_name: str | None = None,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    start = _parse_iso_date(from_date, "from_date")
    end = _parse_iso_date(to_date, "to_date")
    if start > end:
        raise ValueError("from_date must be on or before to_date.")

    identity = resourceplus_identity(email=usr_email, instance=instance_name)
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.get(
        ATTENDANCE_SUMMARY_ROUTE,
        params={
            "usrEmail": identity.email,
            "fromDate": start.isoformat(),
            "toDate": end.isoformat(),
            "instanceName": identity.instance,
            "lang": lang,
        },
    )


async def get_missing_punch_suggestions(
    from_date: date | str,
    to_date: date | str,
    lang: int = DEFAULT_RESOURCEPLUS_LANG,
    usr_email: str | None = None,
    instance_name: str | None = None,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    start = _parse_iso_date(from_date, "from_date")
    end = _parse_iso_date(to_date, "to_date")
    if start > end:
        raise ValueError("from_date must be on or before to_date.")

    identity = resourceplus_identity(email=usr_email, instance=instance_name)
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.get(
        MISSING_PUNCH_SUGGESTIONS_ROUTE,
        params={
            "usrEmail": identity.email,
            "fromDate": start.isoformat(),
            "toDate": end.isoformat(),
            "instanceName": identity.instance,
            "lang": lang,
        },
    )


async def get_exception_reasons(
    lang: int = DEFAULT_RESOURCEPLUS_LANG,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    identity = resourceplus_identity()
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.get(
        EXCEPTION_REASONS_ROUTE,
        params={
            "instanceName": identity.instance,
            "lang": lang,
        },
    )


def _format_exceptional_entry_request_time(value: str) -> str:
    """Validate an authoritative suggestion and format it for the write API."""

    if not isinstance(value, str) or not value:
        raise ValueError("entry_time is required.")
    if not re.fullmatch(r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}", value):
        raise ValueError("entry_time must use DD/MM/YYYY HH:MM format.")
    try:
        parsed = datetime.strptime(value, "%d/%m/%Y %H:%M")
    except ValueError as exc:
        raise ValueError("entry_time must use DD/MM/YYYY HH:MM format.") from exc
    if parsed.strftime("%d/%m/%Y %H:%M") != value:
        raise ValueError("entry_time must use DD/MM/YYYY HH:MM format.")
    return parsed.strftime("%Y/%m/%d %H:%M")


async def create_exceptional_entry(
    entry_time: str,
    entry_type: int,
    reason_id: str,
    remarks: str,
    usr_email: str | None = None,
    instance_name: str | None = None,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    submit_entry_time = _format_exceptional_entry_request_time(entry_time)
    if entry_type not in {1, 2}:
        raise ValueError("entry_type must be 1 (IN) or 2 (OUT).")
    if not reason_id.strip():
        raise ValueError("reason_id is required.")
    if not remarks.strip():
        raise ValueError("remarks are required.")

    identity = resourceplus_identity(email=usr_email, instance=instance_name)
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.post(
        CREATE_EXCEPTIONAL_ENTRY_ROUTE,
        params={"instanceName": identity.instance},
        json_body={
            "usrEmail": identity.email,
            "entryTime": submit_entry_time,
            "entryType": entry_type,
            "reasonID": reason_id,
            "remarks": remarks.strip(),
        },
    )
