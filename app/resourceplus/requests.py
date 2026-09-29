import asyncio
import logging
from datetime import date
from typing import Any, Awaitable

from app.config import DEFAULT_RESOURCEPLUS_LANG
from app.identity import resourceplus_identity
from app.resourceplus.attendance import _parse_iso_date
from app.resourceplus.client import (
    ResourcePlusClient,
    ResourcePlusError,
    ResourcePlusHTTPError,
)
from app.resourceplus.leave import (
    MY_DAY_TYPE_REQUESTS_ROUTE,
    get_my_day_type_requests,
)


EXCEPTIONAL_ENTRY_REQUESTS_ROUTE = "api/AI/ExceptionalEntries"
logger = logging.getLogger(__name__)

# TODO: Do not implement ExceptionalEntries/Cancel until the ResourcePlus backend
# team provides its request-body contract.


class ResourcePlusRequestStatusError(ResourcePlusError):
    """A request-status sub-call failed; the public message stays endpoint-neutral."""


async def _request_status_sub_call(
    endpoint: str,
    request: Awaitable[Any],
) -> Any:
    try:
        return await request
    except ResourcePlusError as exc:
        status_code = exc.status_code if isinstance(exc, ResourcePlusHTTPError) else None
        logger.warning(
            "ResourcePlus request-status sub-call failed: endpoint=%s "
            "error_type=%s http_status=%s",
            endpoint,
            type(exc).__name__,
            status_code if status_code is not None else "unavailable",
        )
        raise ResourcePlusRequestStatusError(
            "ResourcePlus request status is temporarily unavailable. Please try again."
        ) from exc


async def get_exceptional_entry_requests(
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
        EXCEPTIONAL_ENTRY_REQUESTS_ROUTE,
        params={
            "usrEmail": identity.email,
            "fromDate": start.strftime("%m/%d/%Y"),
            "toDate": end.strftime("%m/%d/%Y"),
            "instanceName": identity.instance,
            "lang": lang,
        },
    )


async def get_my_request_status(
    date_from: date | str,
    date_to: date | str,
    lang: int = DEFAULT_RESOURCEPLUS_LANG,
    *,
    client: ResourcePlusClient | None = None,
) -> dict[str, Any]:
    start = _parse_iso_date(date_from, "date_from")
    end = _parse_iso_date(date_to, "date_to")
    if start > end:
        raise ValueError("date_from must be on or before date_to.")

    resourceplus = client or ResourcePlusClient()
    absence_requests, exceptional_requests = await asyncio.gather(
        _request_status_sub_call(
            MY_DAY_TYPE_REQUESTS_ROUTE,
            get_my_day_type_requests(start, end, lang=lang, client=resourceplus),
        ),
        _request_status_sub_call(
            EXCEPTIONAL_ENTRY_REQUESTS_ROUTE,
            get_exceptional_entry_requests(start, end, lang=lang, client=resourceplus),
        ),
    )

    absence_items = absence_requests if isinstance(absence_requests, list) else []
    exceptional_items = (
        exceptional_requests if isinstance(exceptional_requests, list) else []
    )
    merged = [
        {
            "request_kind": "absence",
            "raw_status": item.get("status"),
            "record": item,
        }
        for item in absence_items
        if isinstance(item, dict)
    ]
    merged.extend(
        {
            "request_kind": "exceptional_entry",
            "raw_status": item.get("status"),
            "record": item,
        }
        for item in exceptional_items
        if isinstance(item, dict)
    )
    return {
        "absence_requests": absence_requests,
        "exceptional_entry_requests": exceptional_requests,
        "merged_requests": merged,
    }
