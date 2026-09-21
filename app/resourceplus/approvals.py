from typing import Any

from app.config import DEFAULT_RESOURCEPLUS_LANG, get_settings
from app.resourceplus.client import (
    ResourcePlusClient,
    ResourcePlusConfigurationError,
)


PENDING_APPROVALS_ROUTE = "api/AI/Supervisor/PendingApprovals"
APPROVE_SUPERVISOR_REQUEST_ROUTE = "api/AI/Supervisor/Approve"
APPROVE_ALL_REQUESTS_ROUTE = "api/AI/Supervisor/ApproveAll"


def _manager_identity() -> str:
    manager_email = get_settings().rp_manager_email
    if not manager_email:
        raise ResourcePlusConfigurationError(
            "Manager demo identity is not configured. Set RP_MANAGER_EMAIL."
        )
    return manager_email


async def get_pending_approvals(
    lang: int = DEFAULT_RESOURCEPLUS_LANG,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    settings = get_settings()
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.get(
        PENDING_APPROVALS_ROUTE,
        params={
            "usrEmail": _manager_identity(),
            "instanceName": settings.rp_instance,
            "lang": lang,
        },
    )


async def approve_supervisor_request(
    request_id: str,
    request_type: str,
    status: int,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    if not request_id.strip() or not request_type.strip():
        raise ValueError("request_id and request_type are required.")
    if status not in {1, 2}:
        raise ValueError("status must be 1 (approve) or 2 (reject).")

    settings = get_settings()
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.post(
        APPROVE_SUPERVISOR_REQUEST_ROUTE,
        params={"instanceName": settings.rp_instance},
        json_body={
            "usrEmail": _manager_identity(),
            "requestId": request_id,
            "requestType": request_type,
            "status": status,
        },
    )


async def approve_all_requests(
    status: int,
    request_type: str | None = None,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    if status not in {1, 2}:
        raise ValueError("status must be 1 (approve) or 2 (reject).")
    if request_type not in {None, "Absence", "ExceptionEntry"}:
        raise ValueError("request_type must be Absence or ExceptionEntry.")

    settings = get_settings()
    body: dict[str, str | int] = {
        "usrEmail": _manager_identity(),
        "status": status,
    }
    if request_type is not None:
        body["requestType"] = request_type
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.post(
        APPROVE_ALL_REQUESTS_ROUTE,
        params={"instanceName": settings.rp_instance},
        json_body=body,
    )
