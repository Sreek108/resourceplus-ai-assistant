from typing import Any

from app.config import DEFAULT_RESOURCEPLUS_LANG
from app.identity import current_request_identity
from app.resourceplus.client import ResourcePlusClient


PENDING_APPROVALS_ROUTE = "api/AI/Supervisor/PendingApprovals"
APPROVE_SUPERVISOR_REQUEST_ROUTE = "api/AI/Supervisor/Approve"
APPROVE_ALL_REQUESTS_ROUTE = "api/AI/Supervisor/ApproveAll"


async def get_pending_approvals(
    lang: int = DEFAULT_RESOURCEPLUS_LANG,
    *,
    client: ResourcePlusClient | None = None,
) -> Any:
    identity = current_request_identity()
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.get(
        PENDING_APPROVALS_ROUTE,
        params={
            "usrEmail": identity.email,
            "instanceName": identity.instance,
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

    identity = current_request_identity()
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.post(
        APPROVE_SUPERVISOR_REQUEST_ROUTE,
        params={"instanceName": identity.instance},
        json_body={
            "usrEmail": identity.email,
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

    identity = current_request_identity()
    body: dict[str, str | int] = {
        "usrEmail": identity.email,
        "status": status,
    }
    if request_type is not None:
        body["requestType"] = request_type
    resourceplus = client or ResourcePlusClient()
    return await resourceplus.post(
        APPROVE_ALL_REQUESTS_ROUTE,
        params={"instanceName": identity.instance},
        json_body=body,
    )
