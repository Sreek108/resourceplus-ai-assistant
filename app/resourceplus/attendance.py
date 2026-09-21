from datetime import date, datetime
from typing import Any

from app.config import DEFAULT_RESOURCEPLUS_LANG, get_settings
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
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    start = _parse_iso_date(from_date, "from_date")
    end = _parse_iso_date(to_date, "to_date")
    if start > end:
        raise ValueError("from_date must be on or before to_date.")

    settings = get_settings()
    # TODO: Replace the configured POC email with authenticated_user.email.
    identity = usr_email or settings.rp_default_email
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.get(
        ATTENDANCE_SUMMARY_ROUTE,
        params={
            "usrEmail": identity,
            "fromDate": start.isoformat(),
            "toDate": end.isoformat(),
            "instanceName": settings.rp_instance,
            "lang": lang,
        },
    )


async def get_missing_punch_suggestions(
    from_date: date | str,
    to_date: date | str,
    lang: int = DEFAULT_RESOURCEPLUS_LANG,
    usr_email: str | None = None,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    start = _parse_iso_date(from_date, "from_date")
    end = _parse_iso_date(to_date, "to_date")
    if start > end:
        raise ValueError("from_date must be on or before to_date.")

    settings = get_settings()
    identity = usr_email or settings.rp_default_email
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.get(
        MISSING_PUNCH_SUGGESTIONS_ROUTE,
        params={
            "usrEmail": identity,
            "fromDate": start.isoformat(),
            "toDate": end.isoformat(),
            "instanceName": settings.rp_instance,
            "lang": lang,
        },
    )


async def get_exception_reasons(
    lang: int = DEFAULT_RESOURCEPLUS_LANG,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    settings = get_settings()
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.get(
        EXCEPTION_REASONS_ROUTE,
        params={
            "instanceName": settings.rp_instance,
            "lang": lang,
        },
    )


def _parse_iso_datetime(value: datetime | str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "entry_time must use YYYY-MM-DDTHH:MM:SS format."
            ) from exc
    if parsed.tzinfo is not None:
        raise ValueError("entry_time must be a local datetime without a timezone.")
    return parsed


async def create_exceptional_entry(
    entry_time: datetime | str,
    entry_type: int,
    reason_id: str,
    remarks: str,
    usr_email: str | None = None,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    parsed_time = _parse_iso_datetime(entry_time)
    if entry_type not in {1, 2}:
        raise ValueError("entry_type must be 1 (IN) or 2 (OUT).")
    if not reason_id.strip():
        raise ValueError("reason_id is required.")
    if not remarks.strip():
        raise ValueError("remarks are required.")

    settings = get_settings()
    identity = usr_email or settings.rp_default_email
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.post(
        CREATE_EXCEPTIONAL_ENTRY_ROUTE,
        params={"instanceName": settings.rp_instance},
        json_body={
            "usrEmail": identity,
            "entryTime": parsed_time.isoformat(timespec="seconds"),
            "entryType": entry_type,
            "reasonID": reason_id,
            "remarks": remarks.strip(),
        },
    )
