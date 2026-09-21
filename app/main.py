import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse

from app.ai.agent import AIConfigurationError, OpenAIServiceError
from app.api import audit_debug, chat, health, operations, resourceplus_debug, telemetry, voice
from app.config import get_settings
from app.observability import reset_voice_trace, start_voice_trace
from app.telemetry import (
    TRACE_HEADER,
    ResponseSendTelemetryMiddleware,
    emit_structured_event,
    record_http_metrics,
    reset_trace,
    start_trace,
)
from app.resourceplus import (
    ResourcePlusError,
    ResourcePlusHTTPError,
    ResourcePlusTimeoutError,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
# httpx's INFO message includes the complete query string. ResourcePlus queries
# contain employee identity, so only retain warning/error messages from it.
logging.getLogger("httpx").setLevel(logging.WARNING)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FRONTEND_DIST = (PROJECT_ROOT / "frontend" / "dist").resolve()
FRONTEND_INDEX = FRONTEND_DIST / "index.html"
PROTECTED_FRONTEND_ROOTS = frozenset(
    {
        "api",
        "app",
        "data",
        "docs",
        "frontend",
        "health",
        "logs",
        "openapi.json",
        "redoc",
        "tests",
    }
)

app = FastAPI(
    title="ResourcePlus AI Assistant",
    version="0.2.0",
    description="POC text assistant backend for ResourcePlus HRMS/Payroll.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_settings().cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", TRACE_HEADER],
    expose_headers=[TRACE_HEADER],
)

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(voice.router)
app.include_router(resourceplus_debug.router)
app.include_router(audit_debug.router)
app.include_router(telemetry.router)
app.include_router(operations.router)


@app.middleware("http")
async def observability_middleware(request: Request, call_next):
    started_at = time.perf_counter()
    trace_token = None
    try:
        trace_id, trace_token = start_trace(request.headers.get(TRACE_HEADER))
    except ValueError:
        trace_id, trace_token = start_trace()
        response = JSONResponse(
            status_code=400,
            content={"detail": "Invalid X-Trace-ID header."},
        )
        response.headers[TRACE_HEADER] = trace_id
        record_http_metrics(
            method=request.method,
            path=request.url.path,
            status_code=400,
            duration_ms=(time.perf_counter() - started_at) * 1000,
        )
        emit_structured_event(
            level="WARNING",
            component="http",
            event="trace_id_rejected",
            status=400,
            error_category="validation_error",
            error_owner="validation",
            error_stage="http_request",
            retryable=False,
        )
        reset_trace(trace_token)
        return response

    voice_trace = None
    voice_token = None
    if request.url.path == "/api/voice/chat":
        voice_trace, voice_token = start_voice_trace()
    status_code = 500
    try:
        content_length = request.headers.get("content-length")
        telemetry_too_large = (
            request.url.path == "/api/telemetry/frontend"
            and content_length is not None
            and (not content_length.isdigit() or int(content_length) > 2_048)
        )
        if request.url.path == "/api/telemetry/frontend" and not telemetry_too_large:
            # The Content-Length check rejects oversized requests before reading in
            # normal browsers. This verifies the actual body as well for chunked or
            # incorrectly declared clients; Starlette reuses the cached body downstream.
            telemetry_too_large = len(await request.body()) > 2_048
        if telemetry_too_large:
            response = JSONResponse(
                status_code=413,
                content={"detail": "Frontend telemetry payload is too large."},
            )
        else:
            response = await call_next(request)
        status_code = response.status_code
        response.headers[TRACE_HEADER] = trace_id
        return response
    finally:
        duration_ms = (time.perf_counter() - started_at) * 1000
        recorded_category = getattr(request.state, "error_category", None)
        recorded_owner = getattr(request.state, "error_owner", None)
        if recorded_category and recorded_owner:
            error_category = recorded_category
            error_owner = recorded_owner
        elif status_code < 400:
            error_category = None
            error_owner = None
        elif status_code in {400, 404, 413, 422, 429}:
            error_category = "validation_error"
            error_owner = "validation"
        elif status_code in {401, 403, 409, 410}:
            error_category = "access_denied"
            error_owner = "transaction"
        else:
            error_category = "unknown_safe_category"
            error_owner = "ai_backend"
        record_http_metrics(
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration_ms=duration_ms,
        )
        emit_structured_event(
            level="INFO" if status_code < 400 else "WARNING",
            component="http",
            event="request_completed",
            duration_ms=duration_ms,
            status=status_code,
            error_category=error_category,
            error_owner=error_owner,
            error_stage="http_request" if error_category else None,
            retryable=None if status_code < 400 else status_code >= 500,
        )
        if voice_trace is not None and voice_token is not None:
            voice_trace.finish_response_serialization()
            logging.getLogger("app.voice_latency").info(
                voice_trace.format_summary(status_code=status_code)
            )
            reset_voice_trace(voice_token)
        if trace_token is not None:
            reset_trace(trace_token)


app.add_middleware(ResponseSendTelemetryMiddleware)


@app.exception_handler(ResourcePlusError)
async def resourceplus_error_handler(
    request: Request,
    exc: ResourcePlusError,
) -> JSONResponse:
    request.state.error_owner = "resourceplus_api"
    request.state.error_category = (
        "resourceplus_timeout"
        if isinstance(exc, ResourcePlusTimeoutError)
        else "resourceplus_error"
    )
    if isinstance(exc, ResourcePlusTimeoutError):
        status_code = 504
    else:
        status_code = 502
    if isinstance(exc, ResourcePlusHTTPError):
        message = f"ResourcePlus returned an upstream error (HTTP {exc.status_code})."
    else:
        message = str(exc)
    return JSONResponse(
        status_code=status_code,
        content={
            "success": False,
            "message": message,
            "code": "resourceplus_unavailable",
        },
    )


@app.exception_handler(AIConfigurationError)
async def ai_configuration_error_handler(
    request: Request,
    exc: AIConfigurationError,
) -> JSONResponse:
    request.state.error_owner = "configuration"
    request.state.error_category = "configuration_error"
    return JSONResponse(
        status_code=503,
        content={
            "success": False,
            "message": str(exc),
            "code": "assistant_unavailable",
        },
    )


@app.exception_handler(OpenAIServiceError)
async def openai_error_handler(
    request: Request,
    exc: OpenAIServiceError,
) -> JSONResponse:
    request.state.error_owner = "openai"
    request.state.error_category = "openai_error"
    return JSONResponse(
        status_code=502,
        content={
            "success": False,
            "message": str(exc),
            "code": "assistant_unavailable",
        },
    )


def _frontend_file(frontend_path: str) -> Path | None:
    """Resolve an existing file strictly within the built frontend directory."""

    try:
        candidate = (FRONTEND_DIST / frontend_path).resolve()
        candidate.relative_to(FRONTEND_DIST)
    except (OSError, ValueError):
        return None
    return candidate if candidate.is_file() else None


def _require_frontend_index() -> FileResponse:
    if not FRONTEND_INDEX.is_file():
        raise HTTPException(status_code=404, detail="Frontend build is unavailable.")
    return FileResponse(FRONTEND_INDEX)


@app.get("/", include_in_schema=False)
async def frontend_index() -> FileResponse:
    return _require_frontend_index()


@app.get("/{frontend_path:path}", include_in_schema=False)
async def frontend_spa(frontend_path: str) -> FileResponse:
    """Serve built assets or the SPA shell without exposing project files."""

    parts = Path(frontend_path).parts
    first_part = parts[0].casefold() if parts else ""
    if (
        first_part in PROTECTED_FRONTEND_ROOTS
        or any(part.startswith(".") for part in parts)
    ):
        raise HTTPException(status_code=404, detail="Not Found")

    built_file = _frontend_file(frontend_path)
    if built_file is not None:
        return FileResponse(built_file)

    # Unknown file-like paths are genuine 404s. Extensionless browser routes are
    # handled by React after receiving the production index.
    if Path(frontend_path).suffix:
        raise HTTPException(status_code=404, detail="Not Found")
    return _require_frontend_index()
