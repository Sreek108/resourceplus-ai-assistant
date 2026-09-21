from datetime import date
from typing import Any

from app.config import DEFAULT_RESOURCEPLUS_LANG, get_settings
from app.resourceplus.attendance import _parse_iso_date
from app.resourceplus.client import ResourcePlusClient


DAY_TYPES_ROUTE = "api/AI/DayTypes"
BOOK_DAY_TYPE_ROUTE = "api/AI/DayTypeMapping/Book"
MY_DAY_TYPE_REQUESTS_ROUTE = "api/AI/DayTypeMapping/MyRequests"
CANCEL_DAY_TYPE_REQUEST_ROUTE = "api/AI/DayTypeMapping/Cancel"


async def get_day_types(
    lang: int = DEFAULT_RESOURCEPLUS_LANG,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    settings = get_settings()
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.get(
        DAY_TYPES_ROUTE,
        params={"instanceName": settings.rp_instance, "lang": lang},
    )


async def book_day_type(
    date_from: date | str,
    date_to: date | str,
    day_type_id: int,
    usr_email: str | None = None,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    start = _parse_iso_date(date_from, "date_from")
    end = _parse_iso_date(date_to, "date_to")
    if start > end:
        raise ValueError("date_from must be on or before date_to.")
    if isinstance(day_type_id, bool) or not isinstance(day_type_id, int):
        raise ValueError("day_type_id must be an integer returned by ResourcePlus.")

    settings = get_settings()
    identity = usr_email or settings.rp_default_email
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.post(
        BOOK_DAY_TYPE_ROUTE,
        params={"instanceName": settings.rp_instance},
        json_body={
            "usrEmail": identity,
            "dateFrom": start.isoformat(),
            "dateTo": end.isoformat(),
            "dayTypeID": day_type_id,
        },
    )


async def get_my_day_type_requests(
    date_from: date | str,
    date_to: date | str,
    lang: int = DEFAULT_RESOURCEPLUS_LANG,
    usr_email: str | None = None,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    start = _parse_iso_date(date_from, "date_from")
    end = _parse_iso_date(date_to, "date_to")
    if start > end:
        raise ValueError("date_from must be on or before date_to.")

    settings = get_settings()
    identity = usr_email or settings.rp_default_email
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.get(
        MY_DAY_TYPE_REQUESTS_ROUTE,
        params={
            "usrEmail": identity,
            "instanceName": settings.rp_instance,
            "lang": lang,
            "dateFrom": start.isoformat(),
            "dateTo": end.isoformat(),
        },
    )


async def cancel_day_type_request(
    mapping_id: str,
    usr_email: str | None = None,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    if not mapping_id.strip():
        raise ValueError("mapping_id is required.")

    settings = get_settings()
    identity = usr_email or settings.rp_default_email
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.post(
        CANCEL_DAY_TYPE_REQUEST_ROUTE,
        params={"instanceName": settings.rp_instance},
        json_body={"usrEmail": identity, "mappingID": mapping_id},
    )
