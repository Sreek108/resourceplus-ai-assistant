from datetime import date
from typing import Any

from app.identity import resourceplus_identity
from app.resourceplus.attendance import _parse_iso_date
from app.resourceplus.client import ResourcePlusClient


EXCEPTIONAL_BALANCE_ROUTE = "api/AI/ExceptionalEntries/Balance"
CREATE_FROM_SUMMARY_ROUTE = "api/AI/ExceptionalEntries/FromSummary"
CANCEL_EXCEPTIONAL_ENTRY_ROUTE = "api/AI/ExceptionalEntries/Cancel"


async def get_exceptional_entry_balance(
    target_date: date | str,
    usr_email: str | None = None,
    instance_name: str | None = None,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    selected_date = _parse_iso_date(target_date, "target_date")
    identity = resourceplus_identity(email=usr_email, instance=instance_name)
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.get(
        EXCEPTIONAL_BALANCE_ROUTE,
        params={
            "usrEmail": identity.email,
            "date": selected_date.isoformat(),
            "instanceName": identity.instance,
        },
    )


async def create_exceptional_entry_from_summary(
    att_date: date | str,
    reason_id: str,
    remarks: str = "",
    *,
    entry_type: int | None = None,
    minutes: int | None = None,
    usr_email: str | None = None,
    instance_name: str | None = None,
    client: ResourcePlusClient | None = None,
) -> Any:
    selected_date = _parse_iso_date(att_date, "att_date")
    if not isinstance(reason_id, str) or not reason_id.strip():
        raise ValueError("reason_id is required.")
    if not isinstance(remarks, str):
        raise ValueError("remarks must be a string.")
    if entry_type is not None and entry_type not in {1, 2}:
        raise ValueError("entry_type must be 1 (late IN), 2 (early OUT), or omitted.")
    if minutes is not None and (
        isinstance(minutes, bool) or not isinstance(minutes, int) or minutes <= 0
    ):
        raise ValueError("minutes must be a positive integer or omitted.")

    identity = resourceplus_identity(email=usr_email, instance=instance_name)
    body: dict[str, object] = {
        "usrEmail": identity.email,
        "attDate": selected_date.isoformat(),
        "reasonID": reason_id.strip(),
        "remarks": remarks.strip(),
    }
    if entry_type is not None:
        body["entryType"] = entry_type
    if minutes is not None:
        body["minutes"] = minutes
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.post(
        CREATE_FROM_SUMMARY_ROUTE,
        params={"instanceName": identity.instance},
        json_body=body,
    )


async def cancel_exceptional_entry(
    exceptional_id: str,
    usr_email: str | None = None,
    instance_name: str | None = None,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    if not isinstance(exceptional_id, str) or not exceptional_id.strip():
        raise ValueError("exceptional_id is required.")
    identity = resourceplus_identity(email=usr_email, instance=instance_name)
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.post(
        CANCEL_EXCEPTIONAL_ENTRY_ROUTE,
        params={"instanceName": identity.instance},
        json_body={
            "usrEmail": identity.email,
            "exceptionalID": exceptional_id.strip(),
        },
    )
