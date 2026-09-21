from fastapi import APIRouter, HTTPException, Query

from app.audit import get_audit_store
from app.config import get_settings


router = APIRouter(prefix="/api/debug/audit", tags=["audit-debug"])


@router.get("/recent", include_in_schema=False)
async def recent_audit_records(
    limit: int = Query(default=20, ge=1, le=100),
) -> list[dict]:
    settings = get_settings()
    if not (
        settings.ai_audit_enabled
        and settings.ai_audit_debug_endpoint_enabled
        and not getattr(settings, "uat_allowed_origins", "").strip()
    ):
        raise HTTPException(status_code=404, detail="Not found.")
    store = get_audit_store(
        settings.ai_audit_db_path,
        settings.ai_audit_retention_days,
    )
    return await store.recent(limit)
