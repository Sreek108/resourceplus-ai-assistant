import re

from app.ai.actions import (
    ActionResolutionRequired,
    execute_pending_action,
    prepare_exceptional_entry_reason_follow_up,
)
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
from app.audit import record_action_state, record_safe_error, record_tool_usage
from app.config import get_settings
from app.models.schemas import ChatRequest, ChatResponse
from app.resourceplus import ResourcePlusError
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

DRAFT_CANCELLATIONS = {
    "cancel",
    "cancel it",
    "never mind",
    "nevermind",
    "leave it",
    "stop",
    "إلغاء",
    "الغاء",
    "خلاص",
    "اتركه",
}
DRAFT_GREETINGS = {
    "hello",
    "hi",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "مرحبا",
    "أهلا",
    "اهلا",
    "السلام عليكم",
}
DRAFT_TOPIC_STARTERS = {
    "show",
    "what",
    "how",
    "when",
    "where",
    "who",
    "can",
    "could",
    "would",
    "tell",
    "check",
    "أرني",
    "ارني",
    "ورني",
    "وش",
    "ماذا",
    "كيف",
    "كم",
    "هل",
}

ACTION_RESULT_MESSAGES = {
    "create_exceptional_entry": {
        "submitted_for_approval": {
            "en": {
                "display": "Your exceptional-entry request was submitted successfully for approval.",
                "speech": "Your exceptional-entry request was submitted for approval successfully.",
            },
            "ar": {
                "display": "تم إرسال طلب الإدخال الاستثنائي للموافقة بنجاح.",
                "speech": "تم إرسال طلب الإدخال الاستثنائي للموافقة بنجاح.",
            },
        },
        "failed": {
            "en": {
                "display": (
                    "ResourcePlus could not submit the exceptional-entry request. "
                    "The request was not recorded."
                ),
                "speech": (
                    "ResourcePlus couldn't submit your exceptional-entry request, "
                    "so it wasn't recorded."
                ),
            },
            "ar": {
                "display": (
                    "تعذّر على ResourcePlus إرسال طلب الإدخال الاستثنائي. "
                    "لم يتم تسجيل الطلب."
                ),
                "speech": (
                    "ما قدر ResourcePlus يرسل طلب الإدخال الاستثنائي، "
                    "وعشان كذا الطلب ما تسجّل."
                ),
            },
        },
    }
}


def _normalized_reply(message: str) -> str:
    normalized = re.sub(r"[,.!?،؟]+", "", message.strip().casefold())
    return re.sub(r"\s+", " ", normalized)


def _draft_follow_up_kind(message: str) -> str:
    """Conservatively separate a short reason phrase from clear topic changes."""

    normalized = _normalized_reply(message)
    if normalized in DRAFT_CANCELLATIONS:
        return "cancel"
    if normalized in DRAFT_GREETINGS:
        return "topic_change"
    words = re.findall(r"[^\W_]+", message, flags=re.UNICODE)
    if not words:
        return "topic_change"
    if words[0].casefold() in DRAFT_TOPIC_STARTERS:
        return "topic_change"
    if "?" in message or "؟" in message or len(words) > 6:
        return "topic_change"
    return "reason"


def _draft_cancelled_message(language: str) -> str:
    if language == "ar":
        return "تم إلغاء مسودة طلب الإدخال الاستثنائي. لم يتم إرسال أي طلب."
    return "The exceptional-entry draft was cancelled. Nothing was submitted."


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


def _deterministic_action_result(
    action_type: str,
    action_result: str,
    language: str,
) -> tuple[str, str] | None:
    """Render known write outcomes without trusting upstream response language."""

    action_messages = ACTION_RESULT_MESSAGES.get(action_type)
    if action_messages is None:
        return None
    result_messages = action_messages.get(action_result)
    if result_messages is None:
        return None
    localized = result_messages[language if language in {"en", "ar"} else "en"]
    return localized["display"], localized["speech"]


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
    ignore_prior_history_for_topic_change = False
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
        except ResourcePlusError:
            record_action_state(
                action_type=action.action_type,
                state="failed",
                confirmation_required=False,
                confirmed=True,
                result="failed",
            )
            record_safe_error("resourceplus_error")
            deterministic_result = _deterministic_action_result(
                action.action_type,
                "failed",
                flow_language,
            )
            if deterministic_result is None:
                raise
            message, speech_message = deterministic_result
            store.append_history(session_id, "user", request.message)
            store.append_history(session_id, "assistant", message)
            return ChatResponse(
                success=False,
                message=message,
                language=flow_language,
                tools_used=[action.action_type],
                session_id=session_id,
            ).set_speech_message(speech_message)
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
        action_result = _safe_action_result(action.action_type, success)
        record_action_state(
            action_type=action.action_type,
            state="executed" if success else "failed",
            confirmation_required=False,
            confirmed=True,
            result=action_result,
        )
        deterministic_result = _deterministic_action_result(
            action.action_type,
            action_result,
            flow_language,
        )
        if deterministic_result is None:
            message = await _render_or_fallback(
                source_message,
                language=flow_language,
                purpose="report the confirmed ResourcePlus operation result",
                fallback=source_message,
            )
            speech_message = message
        else:
            message, speech_message = deterministic_result
        store.append_history(session_id, "user", request.message)
        store.append_history(session_id, "assistant", message)
        return ChatResponse(
            success=success,
            message=message,
            language=flow_language,
            tools_used=[action.action_type],
            session_id=session_id,
        ).set_speech_message(speech_message)

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

    draft = store.get_exceptional_entry_draft(session_id)
    if draft is not None:
        draft_follow_up = _draft_follow_up_kind(request.message)
        if draft_follow_up == "cancel":
            store.clear_exceptional_entry_draft(session_id)
            message = _draft_cancelled_message(draft.language)
            store.append_history(session_id, "user", request.message)
            store.append_history(session_id, "assistant", message)
            return ChatResponse(
                success=True,
                message=message,
                language=draft.language,
                session_id=session_id,
            ).set_speech_message(message)
        if draft_follow_up == "reason":
            record_tool_usage("prepare_exceptional_entry")
            try:
                intent = await prepare_exceptional_entry_reason_follow_up(
                    attendance_date=draft.attendance_date,
                    entry_type=draft.entry_type,
                    suggested_entry_time=draft.suggested_entry_time,
                    reason_text=request.message,
                    lang=lang,
                    response_language=draft.language,
                )
            except ActionResolutionRequired as exc:
                if exc.category == "draft_stale":
                    store.clear_exceptional_entry_draft(session_id)
                message = str(exc)
                store.append_history(session_id, "user", request.message)
                store.append_history(session_id, "assistant", message)
                return ChatResponse(
                    success=True,
                    message=message,
                    language=draft.language,
                    tools_used=["prepare_exceptional_entry"],
                    session_id=session_id,
                    needs_reason=exc.category in {
                        "reason_unknown",
                        "reason_ambiguous",
                    },
                    reason_options=[
                        {"label": name, "value": name}
                        for name in (exc.reason_options or [])
                    ] or None,
                ).set_speech_message(message)
            except ResourcePlusError:
                record_safe_error("resourceplus_error")
                message = (
                    "تعذر التحقق من بيانات ResourcePlus الآن. حاول مرة أخرى."
                    if draft.language == "ar"
                    else (
                        "I couldn't verify the ResourcePlus data right now. "
                        "Please try again."
                    )
                )
                return ChatResponse(
                    success=False,
                    message=message,
                    language=draft.language,
                    tools_used=["prepare_exceptional_entry"],
                    session_id=session_id,
                ).set_speech_message(message)
            except (ValueError, TypeError, KeyError):
                record_safe_error("validation_error")
                message = (
                    "تعذر التحقق من بيانات التصحيح الآن. حاول مرة أخرى."
                    if draft.language == "ar"
                    else (
                        "I couldn't validate the correction data right now. "
                        "Please try again."
                    )
                )
                return ChatResponse(
                    success=False,
                    message=message,
                    language=draft.language,
                    tools_used=["prepare_exceptional_entry"],
                    session_id=session_id,
                ).set_speech_message(message)
            pending_action = store.create_pending_action(
                session_id,
                action_type=intent.action_type,
                validated_arguments=intent.validated_arguments,
                summary=intent.summary,
                language=intent.language,
            )
            record_action_state(
                action_type=intent.action_type,
                state="pending_confirmation",
                confirmation_required=True,
                confirmed=False,
            )
            message = intent.summary
            store.append_history(session_id, "user", request.message)
            store.append_history(session_id, "assistant", message)
            return ChatResponse(
                success=True,
                message=message,
                language=draft.language,
                tools_used=["prepare_exceptional_entry"],
                session_id=session_id,
                requires_confirmation=True,
                confirmation_id=pending_action.confirmation_id,
            ).set_speech_message(message)

        # An explicit topic change abandons this non-executable draft so it cannot
        # hijack later turns. The ordinary agent remains responsible for the new topic.
        store.clear_exceptional_entry_draft(session_id)
        ignore_prior_history_for_topic_change = True

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
        history=(
            []
            if ignore_prior_history_for_topic_change
            else store.get_history(session_id)
        ),
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
        needs_reason=result.needs_reason,
        reason_options=[
            {"label": name, "value": name}
            for name in (result.reason_options or [])
        ] or None,
    ).set_speech_message(result.speech_message)
