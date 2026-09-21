from fastapi import APIRouter
from fastapi.responses import JSONResponse, PlainTextResponse

from app.config import get_settings
from app.telemetry import metrics


router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/ready")
async def ready() -> JSONResponse:
    """Check local configuration only; no provider is contacted."""

    settings = get_settings()
    components = {
        "openai": bool(settings.openai_api_key and settings.openai_model),
        "resourceplus": bool(
            settings.rp_base_url and settings.rp_instance and settings.rp_default_email
        ),
        "voice": bool(settings.azure_speech_key and settings.azure_speech_region),
    }
    core_ready = components["openai"] and components["resourceplus"]
    return JSONResponse(
        status_code=200 if core_ready else 503,
        content={
            "status": "ready" if core_ready else "not_ready",
            "components": components,
        },
    )


@router.get("/metrics", include_in_schema=False)
async def prometheus_metrics() -> PlainTextResponse:
    return PlainTextResponse(
        metrics.render(),
        media_type="text/plain; version=0.0.4; charset=utf-8",
    )
