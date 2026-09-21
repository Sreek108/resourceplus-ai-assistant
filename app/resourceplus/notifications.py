from typing import Any

from app.config import DEFAULT_RESOURCEPLUS_LANG, get_settings
from app.resourceplus.client import ResourcePlusClient


NOTIFICATIONS_ROUTE = "api/Client/GetNotifcnData"
UPDATE_NOTIFICATION_READ_STATUS_ROUTE = "api/Client/UpdateReadStatus"


async def get_notifications(
    lang: int = DEFAULT_RESOURCEPLUS_LANG,
    usr_email: str | None = None,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    settings = get_settings()
    identity = usr_email or settings.rp_default_email
    resourceplus = client or ResourcePlusClient()
    response = await resourceplus.get(
        NOTIFICATIONS_ROUTE,
        params={
            "instanceName": settings.rp_instance,
            "usrEmail": identity,
            "lang": lang,
        },
    )
    if isinstance(response, dict) and isinstance(response.get("Notifications"), list):
        return response["Notifications"]
    return response


async def update_notification_read_status(
    notifcn_id: int,
    read_status: int,
    lang: int = DEFAULT_RESOURCEPLUS_LANG,
    usr_email: str | None = None,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    """Execute the legacy query-parameter mutation after backend confirmation."""

    if isinstance(notifcn_id, bool) or not isinstance(notifcn_id, int):
        raise ValueError("notifcn_id must be a ResourcePlus notification ID.")
    if notifcn_id < 0:
        raise ValueError("notifcn_id cannot be negative.")
    if read_status not in {0, 1}:
        raise ValueError("read_status must be 0 (unread) or 1 (read).")

    settings = get_settings()
    identity = usr_email or settings.rp_default_email
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.get(
        UPDATE_NOTIFICATION_READ_STATUS_ROUTE,
        params={
            "instanceName": settings.rp_instance,
            "usrEmail": identity,
            "lang": lang,
            "notifcnID": notifcn_id,
            "readStatus": read_status,
        },
    )
