"""Opt-in, protected, safe operational configuration diagnostics."""

from __future__ import annotations

import secrets

from fastapi import APIRouter, Header, HTTPException

from app.config import get_settings


router = APIRouter(prefix="/api/ops", tags=["operations"])


@router.get("/diagnostics")
async def diagnostics(x_ops_key: str | None = Header(default=None)) -> dict[str, object]:
    settings = get_settings()
    if not settings.ops_diagnostics_enabled:
        raise HTTPException(status_code=404, detail="Not Found")
    configured_token = settings.ops_diagnostics_token
    if not configured_token or not x_ops_key or not secrets.compare_digest(
        configured_token,
        x_ops_key,
    ):
        raise HTTPException(status_code=401, detail="Operations access denied.")
    return {
        "status": "ok",
        "environment": settings.app_environment,
        "deployment_id": settings.deployment_id,
        "app_version": settings.app_version,
        "git_commit": settings.git_commit,
        "audit_enabled": settings.ai_audit_enabled,
        "audit_content_enabled": settings.ai_audit_store_content,
        "observability_exporter": settings.observability_exporter,
        "components": {
            "openai_configured": bool(settings.openai_api_key and settings.openai_model),
            "resourceplus_configured": bool(
                settings.rp_base_url and settings.rp_instance and settings.rp_default_email
            ),
            "azure_speech_configured": bool(
                settings.azure_speech_key and settings.azure_speech_region
            ),
        },
    }
