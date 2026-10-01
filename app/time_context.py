from datetime import datetime, timedelta, timezone as fixed_timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config import get_settings


def resourceplus_today():
    """Return the configured ResourcePlus/user-local calendar date."""

    timezone_name = get_settings().resourceplus_timezone
    try:
        timezone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError:
        # Windows installations may not include the IANA tzdata package. Riyadh
        # has no daylight-saving transitions, so its fixed UTC+03:00 offset is an
        # exact dependency-free fallback. Other invalid names fail closed to UTC.
        timezone = (
            fixed_timezone(timedelta(hours=3))
            if timezone_name == "Asia/Riyadh"
            else fixed_timezone.utc
        )
    return datetime.now(timezone).date()
