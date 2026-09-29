from fastapi import APIRouter, HTTPException

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


router = APIRouter(prefix="/api", tags=["chat"])

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
