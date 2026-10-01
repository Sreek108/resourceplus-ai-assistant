import json
import time
from collections.abc import AsyncIterator

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.audit import (
    persist_audit,
    reset_interaction_audit,
    safe_error_category,
    safe_session_reference,
    start_interaction_audit,
)
from app.models.schemas import ChatRequest, ChatResponse
from app.identity import RequestIdentityError, resolve_request_identity
from app.observability import measure_stage
from app.services.chat import process_chat
from app.services.fast_reads import classify_fast_read
from app.telemetry import metrics


router = APIRouter(prefix="/api", tags=["chat"])


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False, default=str)}\n\n"

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    try:
        resolve_request_identity(request.email, request.instance)
    except RequestIdentityError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_identity", "message": str(exc)},
        ) from exc
    audit, token = start_interaction_audit(
        input_mode="text",
        input_source="typed",
        user_text=request.message,
        session_id=request.session_id,
    )
    try:
        with measure_stage("agent"):
            response = await process_chat(request)
        audit.session_reference = audit.session_reference or safe_session_reference(
            response.session_id
        )
        audit.resolved_language = response.language
        audit.response_language = response.language
        audit.display_message = response.message
        audit.speech_message = response.speech_message
        audit.tools_used = list(dict.fromkeys([*audit.tools_used, *response.tools_used]))
        audit.success = response.success
        audit.result_status = "success" if response.success else "failed"
        audit.confirmation_required = response.requires_confirmation
        await persist_audit(audit)
        return response
    except Exception as exc:
        audit.error_category = safe_error_category(exc)
        audit.result_status = "error"
        await persist_audit(audit)
        raise
    finally:
        reset_interaction_audit(token)


@router.post("/chat/stream")
async def chat_stream(request: ChatRequest) -> StreamingResponse:
    """Stream progress and the v2 response while preserving POST /chat semantics."""

    try:
        resolve_request_identity(request.email, request.instance)
    except RequestIdentityError as exc:
        raise HTTPException(
            status_code=422,
            detail={"code": "invalid_identity", "message": str(exc)},
        ) from exc

    async def events() -> AsyncIterator[str]:
        started_at = time.perf_counter()
        audit, token = start_interaction_audit(
            input_mode="text",
            input_source="typed",
            user_text=request.message,
            session_id=request.session_id,
        )
        try:
            yield _sse("accepted", {"state": "accepted"})
            first_status_ms = (time.perf_counter() - started_at) * 1000
            metrics.observe(
                "resourceplus_assistant_text_first_status_ms",
                first_status_ms,
                mode="text",
            )
            yield _sse("status", {"state": "processing"})
            if classify_fast_read(request.message):
                yield _sse("status", {"state": "fetching_hr_data"})
        except BaseException:
            # A client may disconnect before processing starts. The main lifecycle
            # below has its own cleanup once these initial events have completed.
            reset_interaction_audit(token)
            raise
        try:
            response = await process_chat(request)
            audit.session_reference = audit.session_reference or safe_session_reference(
                response.session_id
            )
            audit.resolved_language = response.language
            audit.response_language = response.language
            audit.display_message = response.message
            audit.speech_message = response.speech_message
            audit.tools_used = list(dict.fromkeys([*audit.tools_used, *response.tools_used]))
            audit.success = response.success
            audit.result_status = "success" if response.success else "failed"
            audit.confirmation_required = response.requires_confirmation
            first_text_ms = (time.perf_counter() - started_at) * 1000
            metrics.observe(
                "resourceplus_assistant_text_first_text_ms",
                first_text_ms,
                mode="text",
            )
            yield _sse("text_delta", {"text": response.message})
            if response.blocks:
                yield _sse(
                    "blocks",
                    {
                        "response_schema_version": response.response_schema_version,
                        "blocks": [
                            block.model_dump(mode="json") for block in response.blocks
                        ],
                    },
                )
            if response.requires_confirmation:
                yield _sse(
                    "confirmation",
                    {
                        "confirmation_id": response.confirmation_id,
                        "requires_confirmation": True,
                    },
                )
            yield _sse("complete", response.model_dump(mode="json"))
            metrics.observe(
                "resourceplus_assistant_text_total_ms",
                (time.perf_counter() - started_at) * 1000,
                mode="text",
                status="success",
            )
            await persist_audit(audit)
        except Exception as exc:
            audit.error_category = safe_error_category(exc)
            audit.result_status = "error"
            await persist_audit(audit)
            metrics.observe(
                "resourceplus_assistant_text_total_ms",
                (time.perf_counter() - started_at) * 1000,
                mode="text",
                status="failure",
            )
            yield _sse(
                "error",
                {
                    "code": "assistant_unavailable",
                    "message": "The assistant could not complete the request.",
                    "category": safe_error_category(exc),
                },
            )
        finally:
            reset_interaction_audit(token)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
