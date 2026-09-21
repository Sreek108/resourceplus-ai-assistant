import re

from app.ai.actions import execute_pending_action
from app.ai.agent import (
    AgentResult,
    OpenAIServiceError,
    classify_confirmation_intent,
    detect_language,
    render_user_message,
    run_agent,
)
from app.ai.sessions import (
    PendingActionExpired,
    PendingActionMismatch,
    SessionStore,
    session_store,
)
from app.audit import record_action_state, record_safe_error
from app.config import get_settings
from app.models.schemas import ChatRequest, ChatResponse
from app.speech.language import count_script_letters


UNAMBIGUOUS_ENGLISH_CONFIRMATIONS = {
    "yes",
    "yes please",
    "confirm",
    "confirmed",
    "continue",
    "proceed",
    "do it",
    "submit it",
}
UNAMBIGUOUS_ENGLISH_REJECTIONS = {
    "no",
    "cancel",
    "don't do it",
    "do not do it",
    "stop",
}
MIN_EXPLICIT_LANGUAGE_SWITCH_LETTERS = 4
CONFIRMATION_MESSAGES = {
    "cancelled": {
        "en": "The pending action was cancelled. Nothing was submitted.",
        "ar": "تم إلغاء الإجراء المعلّق. لم يتم إرسال أي طلب.",
    },
    "expired": {
        "en": "The pending confirmation expired. Please start the request again.",
        "ar": "انتهت صلاحية التأكيد المعلّق. يرجى بدء الطلب من جديد.",
    },
    "mismatch": {
        "en": "The confirmation reference does not match. Nothing was executed.",
        "ar": "مرجع التأكيد غير مطابق. لم يتم تنفيذ الإجراء.",
    },
    "missing": {
        "en": "There is no pending action to confirm or reject.",
        "ar": "لا يوجد إجراء معلّق لتأكيده أو إلغائه.",
    },
}


def _normalized_reply(message: str) -> str:
    normalized = re.sub(r"[,.!?،؟]+", "", message.strip().casefold())
    return re.sub(r"\s+", " ", normalized)


def _unambiguous_english_decision(message: str, language: str) -> str:
    if language != "en":
        return "OTHER"
    normalized = _normalized_reply(message)
    if normalized in UNAMBIGUOUS_ENGLISH_CONFIRMATIONS:
        return "CONFIRM"
    if normalized in UNAMBIGUOUS_ENGLISH_REJECTIONS:
        return "REJECT"
    return "OTHER"


def _confirmation_response_language(
    message: str,
    *,
    detected_language: str,
    flow_language: str | None,
) -> str:
    """Keep short replies in the PendingAction language without blocking real switches."""

    established = flow_language if flow_language in {"en", "ar"} else detected_language
    if detected_language == established:
        return established
    arabic_letters, latin_letters = count_script_letters(message)
    current_letters = arabic_letters if detected_language == "ar" else latin_letters
    if current_letters >= MIN_EXPLICIT_LANGUAGE_SWITCH_LETTERS:
        return detected_language
    return established


def _confirmation_message(kind: str, language: str) -> str:
    resolved = language if language in {"en", "ar"} else "en"
    return CONFIRMATION_MESSAGES[kind][resolved]


def _response_message(result: object) -> tuple[bool, str]:
    if not isinstance(result, dict):
        return False, "ResourcePlus returned an unexpected response."
    success = result.get("success") is True
    message = result.get("message")
    if not isinstance(message, str) or not message.strip():
        message = (
            "ResourcePlus confirmed the action."
            if success
            else "ResourcePlus did not confirm the action."
        )
    return success, message


def _safe_action_result(action_type: str, success: bool) -> str:
    if not success:
        return "failed"
    if action_type in {"book_day_type", "create_exceptional_entry"}:
        return "submitted_for_approval"
    if action_type == "cancel_day_type_request":
        return "cancelled"
    if action_type in {"approve_supervisor_request", "approve_all_requests"}:
        return "approved"
    if action_type == "update_notification_read_status":
        return "updated"
    return "succeeded"


async def _confirmation_decision(
    message: str,
    *,
    pending_summary: str,
    language: str,
) -> str:
    fast_decision = _unambiguous_english_decision(message, language)
    if fast_decision != "OTHER":
        return fast_decision
    try:
        return await classify_confirmation_intent(
            message,
            pending_summary=pending_summary,
        )
    except OpenAIServiceError:
        # A classification failure must never execute a pending write.
        return "OTHER"


async def _render_or_fallback(
    facts: str,
    *,
    language: str,
    purpose: str,
    fallback: str,
) -> str:
    try:
        return await render_user_message(
            facts,
            language=language,
            purpose=purpose,
        )
    except OpenAIServiceError:
        return fallback


async def process_chat(
    request: ChatRequest,
    *,
    detected_language: str | None = None,
    store: SessionStore = session_store,
) -> ChatResponse:
    """Run one chat turn for either the text or voice API."""

    lang = request.lang or get_settings().rp_default_lang
    session_id = store.ensure_session(request.session_id)
    language = detected_language or detect_language(request.message)
    pending, expired = store.get_pending_action(session_id)
    expired_action_type = (
        store.get_expired_pending_action_type(session_id) if expired else None
    )
    if pending is not None:
        record_action_state(
            action_type=pending.action_type,
            state="pending_confirmation",
            confirmation_required=True,
            confirmed=False,
        )
    elif expired:
        record_action_state(
            action_type=expired_action_type,
            state="expired",
            confirmation_required=False,
            confirmed=False,
        )
        record_safe_error("confirmation_expired")
    expired_language = store.get_expired_pending_language(session_id) if expired else None
    flow_language = _confirmation_response_language(
        request.message,
        detected_language=language,
        flow_language=pending.language if pending is not None else expired_language,
    )
    if pending is not None:
        decision = await _confirmation_decision(
            request.message,
            pending_summary=pending.summary,
            language=language,
        )
    else:
        # Confirmation interpretation is meaningful only while an immutable,
        # validated PendingAction exists. Preserve the exact English button/CLI
        # fast paths so an orphaned "Yes" or "No" receives the established safe
        # no-pending response without adding a model call.
        decision = _unambiguous_english_decision(request.message, language)

    if pending is not None and decision == "CONFIRM":
        try:
            action = store.consume_pending_action(
                session_id,
                request.confirmation_id,
            )
        except PendingActionExpired:
            record_action_state(
                action_type=pending.action_type,
                state="expired",
                confirmation_required=False,
                confirmed=False,
            )
            record_safe_error("confirmation_expired")
            return ChatResponse(
                success=False,
                message=_confirmation_message("expired", flow_language),
                language=flow_language,
                session_id=session_id,
            ).set_speech_message(_confirmation_message("expired", flow_language))
        except PendingActionMismatch:
            record_action_state(
                action_type=pending.action_type,
                state="pending_confirmation",
                confirmation_required=True,
                confirmed=False,
            )
            record_safe_error("access_denied")
            return ChatResponse(
                success=False,
                message=_confirmation_message("mismatch", flow_language),
                language=flow_language,
                session_id=session_id,
                requires_confirmation=True,
                confirmation_id=pending.confirmation_id,
            ).set_speech_message(_confirmation_message("mismatch", flow_language))
        record_action_state(
            action_type=action.action_type,
            state="prepared",
            confirmation_required=False,
            confirmed=True,
        )
        try:
            operation_result = await execute_pending_action(
                action.action_type,
                action.validated_arguments,
            )
        except Exception:
            record_action_state(
                action_type=action.action_type,
                state="failed",
                confirmation_required=False,
                confirmed=True,
                result="failed",
            )
            raise
        success, source_message = _response_message(operation_result)
        record_action_state(
            action_type=action.action_type,
            state="executed" if success else "failed",
            confirmation_required=False,
            confirmed=True,
            result=_safe_action_result(action.action_type, success),
        )
        message = await _render_or_fallback(
            source_message,
            language=flow_language,
            purpose="report the confirmed ResourcePlus operation result",
            fallback=source_message,
        )
        store.append_history(session_id, "user", request.message)
        store.append_history(session_id, "assistant", message)
        return ChatResponse(
            success=success,
            message=message,
            language=flow_language,
            tools_used=[action.action_type],
            session_id=session_id,
        ).set_speech_message(message)

    if pending is not None and decision == "REJECT":
        store.discard_pending_action(session_id)
        record_action_state(
            action_type=pending.action_type,
            state="rejected",
            confirmation_required=False,
            confirmed=False,
            result="cancelled",
        )
        # This acknowledgement is deliberately deterministic: the action has
        # already been discarded and a model must not change its language or
        # imply that anything was submitted.
        message = _confirmation_message("cancelled", flow_language)
        store.append_history(session_id, "user", request.message)
        store.append_history(session_id, "assistant", message)
        return ChatResponse(
            success=True,
            message=message,
            language=flow_language,
            session_id=session_id,
        ).set_speech_message(message)

    if pending is not None:
        message = await _render_or_fallback(
            pending.summary,
            language=flow_language,
            purpose="ask whether to confirm or reject the pending action",
            fallback=pending.summary,
        )
        return ChatResponse(
            success=True,
            message=message,
            language=flow_language,
            session_id=session_id,
            requires_confirmation=True,
            confirmation_id=pending.confirmation_id,
        ).set_speech_message(message)

    if expired or decision in {"CONFIRM", "REJECT"}:
        message = _confirmation_message(
            "expired" if expired else "missing",
            flow_language,
        )
        return ChatResponse(
            success=False,
            message=message,
            language=flow_language,
            session_id=session_id,
        ).set_speech_message(message)

    result: AgentResult = await run_agent(
        request.message,
        lang=lang,
        session_id=session_id,
        history=store.get_history(session_id),
        response_language=language,
    )
    store.append_history(session_id, "user", request.message)
    store.append_history(session_id, "assistant", result.message)
    if result.requires_confirmation:
        prepared, _ = store.get_pending_action(session_id)
        if prepared is not None:
            record_action_state(
                action_type=prepared.action_type,
                state="pending_confirmation",
                confirmation_required=True,
                confirmed=False,
            )
    return ChatResponse(
        success=not result.tool_failed,
        message=result.message,
        language=language,
        tools_used=result.tools_used,
        session_id=session_id,
        requires_confirmation=result.requires_confirmation,
        confirmation_id=result.confirmation_id,
    ).set_speech_message(result.speech_message)
