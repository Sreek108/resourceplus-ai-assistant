import re

from app.ai.actions import (
    ActionResolutionRequired,
    execute_pending_action,
    prepare_exceptional_entry_reason_follow_up,
)
from app.ai.attendance_intent import (
    is_ambiguous_transactional_utterance,
    is_existing_late_punch,
    is_explicit_missing_punch_correction,
    is_prospective_late_arrival,
    late_arrival_unavailable_message,
)
from app.ai.reason_matcher import deterministic_reason_match
from app.ai.conversation import continue_conversation, is_less_hours_request
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
    SessionIdentityMismatch,
    SessionStore,
    TrustedResultContext,
    session_store,
)
from app.audit import record_action_state, record_safe_error, record_tool_usage
from app.config import get_settings
from app.identity import (
    RequestIdentityError,
    bind_request_identity,
    reset_request_identity,
    resolve_request_identity,
)
from app.models.schemas import ChatRequest, ChatResponse
from app.services.fast_reads import is_noisy_balance_reference, try_fast_read
from app.services.response_blocks import (
    confirmation_block,
    exceptional_submission_blocks,
    reason_actions,
)
from app.resourceplus import ResourcePlusError
from app.speech.language import count_script_letters
from app.time_context import resourceplus_today


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
UNAMBIGUOUS_ARABIC_CONFIRMATIONS = {"نعم", "أجل", "اجل"}
UNAMBIGUOUS_ARABIC_REJECTIONS = {"لا"}
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

SUPPORTED_HR_TOPIC = re.compile(
    r"\b(?:attendance|profile|leave|vacation|holiday|notification|alert|approval|"
    r"request(?:s|\s+status)?|balance|payslip|salary|business\s+travel)\b",
    flags=re.IGNORECASE,
)
SUPPORTED_ARABIC_HR_TERMS = (
    "\u0627\u0644\u062d\u0636\u0648\u0631",  # attendance
    "\u0645\u0644\u0641\u064a",  # my profile
    "\u0627\u0644\u0645\u0644\u0641",  # profile
    "\u0625\u062c\u0627\u0632",  # leave (hamza spelling)
    "\u0627\u062c\u0627\u0632",  # leave (plain spelling)
    "\u0637\u0644\u0628\u0627\u062a",  # requests
    "\u0637\u0644\u0628\u0627\u062a\u064a",  # my requests
    "\u0625\u0634\u0639\u0627\u0631",  # notification (hamza spelling)
    "\u0627\u0634\u0639\u0627\u0631",  # notification (plain spelling)
    "\u062a\u0646\u0628\u064a\u0647",  # alert
    "\u0645\u0648\u0627\u0641\u0642",  # approval root
    "\u0631\u0635\u064a\u062f",  # balance
    "\u0631\u0627\u062a\u0628",  # salary
)

_TRUST_RETRACTION = re.compile(
    r"\b(?:was(?:n't|\s+not)\s+supported|not\s+supported|unsupported|incorrect|wrong|"
    r"not\s+accurate|mistake|fabricated|made\s+up|cannot\s+verify|can't\s+verify)\b",
    re.IGNORECASE,
)
_TRUST_REFERENCE = re.compile(
    r"\b(?:earlier|previous|prior|that\s+(?:figure|value|number|result|balance)|"
    r"the\s+(?:figure|value|number|result|balance))\b",
    re.IGNORECASE,
)
_PREVIOUS_MONTH_FOLLOW_UP = re.compile(
    r"^\s*(?:previous|last)\s+month[.!?\s]*$",
    re.IGNORECASE,
)


def _has_trusted_resourceplus_result(response: ChatResponse) -> bool:
    return response.success and any(
        tool.startswith(("get_", "create_", "cancel_", "book_", "approve_", "update_"))
        for tool in response.tools_used
    )


def _model_retracts_trusted_result(
    message: str,
    trusted: TrustedResultContext,
) -> bool:
    english_retraction = bool(_TRUST_RETRACTION.search(message))
    english_reference = bool(_TRUST_REFERENCE.search(message))
    arabic_retraction = any(
        phrase in message
        for phrase in ("غير مدعوم", "لم يكن مدعوم", "غير صحيح", "خاطئ", "لا يمكن التحقق")
    )
    arabic_reference = any(
        phrase in message
        for phrase in ("السابق", "السابقة", "القيمة", "الرقم", "النتيجة", "الرصيد")
    )
    trusted_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", trusted.message))
    response_numbers = set(re.findall(r"\b\d+(?:\.\d+)?\b", message))
    shared_number = bool(trusted_numbers & response_numbers)
    return (english_retraction and (english_reference or shared_number)) or (
        arabic_retraction and (arabic_reference or shared_number)
    )


def _grounding_guard_message(trusted: TrustedResultContext, language: str) -> str:
    is_balance = "get_exceptional_entry_balance" in trusted.tools_used
    if language == "ar":
        return (
            "تم جلب رصيدك السابق من سجل الموارد البشرية. يمكنني تحديثه إذا رغبت."
            if is_balance
            else "تم جلب النتيجة السابقة من سجل الموارد البشرية. يمكنني تحديثها إذا رغبت."
        )
    return (
        "Your previous balance came from a completed HR balance check. I can refresh it if you'd like."
        if is_balance
        else "That earlier result came from a completed HR data check. I can refresh it if you'd like."
    )

ACTION_RESULT_MESSAGES = {
    "create_exceptional_entry": {
        "submitted_for_approval": {
            "en": {
                "display": "Your exceptional-entry request was submitted for approval.",
                "speech": "Your exceptional-entry request was submitted for approval.",
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
    },
    "cancel_exceptional_entry": {
        "cancelled": {
            "en": {
                "display": "Your pending exceptional-entry request was cancelled.",
                "speech": "Your pending exceptional-entry request was cancelled.",
            },
            "ar": {
                "display": "تم إلغاء طلب الإدخال الاستثنائي المعلّق.",
                "speech": "تم إلغاء طلب الإدخال الاستثنائي المعلّق.",
            },
        },
        "failed": {
            "en": {
                "display": "ResourcePlus could not cancel that exceptional-entry request.",
                "speech": "ResourcePlus couldn't cancel that exceptional-entry request.",
            },
            "ar": {
                "display": "تعذّر على ResourcePlus إلغاء طلب الإدخال الاستثنائي.",
                "speech": "تعذّر على ResourcePlus إلغاء طلب الإدخال الاستثنائي.",
            },
        },
    },
    "create_exceptional_entry_from_summary": {
        "failed": {
            "en": {
                "display": "ResourcePlus could not submit the less-hours correction. It was not recorded.",
                "speech": "ResourcePlus couldn't submit the less-hours correction, so it wasn't recorded.",
            },
            "ar": {
                "display": "تعذّر على ResourcePlus إرسال تصحيح الساعات الناقصة، ولم يتم تسجيله.",
                "speech": "تعذّر على ResourcePlus إرسال تصحيح الساعات الناقصة، ولم يتم تسجيله.",
            },
        },
    },
}


def _normalized_reply(message: str) -> str:
    normalized = re.sub(r"[,.!?،؟]+", "", message.strip().casefold())
    return re.sub(r"\s+", " ", normalized)


def _is_clear_supported_hr_intent(message: str) -> bool:
    if is_explicit_missing_punch_correction(message):
        return True
    if SUPPORTED_HR_TOPIC.search(message):
        return True
    normalized = _normalized_reply(message)
    return any(term in normalized for term in SUPPORTED_ARABIC_HR_TERMS)


def _draft_follow_up_kind(
    message: str,
    reason_options: tuple[str, ...] = (),
) -> str:
    """Separate a reason answer from a new request or ordinary conversation."""

    normalized = _normalized_reply(message)
    if normalized in DRAFT_CANCELLATIONS:
        return "cancel"
    # These initial live labels are routing hints only. The reason workflow still
    # refetches ResourcePlus data and revalidates the selection before creating
    # an immutable PendingAction.
    if deterministic_reason_match(message, list(reason_options)).index is not None:
        return "reason"
    if _is_clear_supported_hr_intent(message):
        return "new_intent"
    if normalized in DRAFT_GREETINGS or any(
        normalized.startswith(f"{greeting} ")
        for greeting in DRAFT_GREETINGS
    ):
        return "casual"
    words = re.findall(r"[^\W_]+", message, flags=re.UNICODE)
    if not words:
        return "casual"
    if words[0].casefold() in DRAFT_TOPIC_STARTERS:
        return "casual"
    if "?" in message or "؟" in message or len(words) > 6:
        return "casual"
    return "reason"


def _draft_cancelled_message(language: str) -> str:
    if language == "ar":
        return "تم إلغاء مسودة طلب الإدخال الاستثنائي. لم يتم إرسال أي طلب."
    return "The exceptional-entry draft was cancelled. Nothing was submitted."


def _unambiguous_english_decision(message: str, language: str) -> str:
    normalized = _normalized_reply(message)
    if language == "en":
        if normalized in UNAMBIGUOUS_ENGLISH_CONFIRMATIONS:
            return "CONFIRM"
        if normalized in UNAMBIGUOUS_ENGLISH_REJECTIONS:
            return "REJECT"
    if language == "ar":
        if normalized in UNAMBIGUOUS_ARABIC_CONFIRMATIONS:
            return "CONFIRM"
        if normalized in UNAMBIGUOUS_ARABIC_REJECTIONS:
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


def _from_summary_operation_success(result: object) -> bool:
    """Apply the documented v2 success flag, with one narrow legacy fallback."""

    if not isinstance(result, dict):
        return False
    if "success" in result:
        return result.get("success") is True
    # Compatibility for early v2 responses that emitted only isAutoApproved.
    # Either boolean value represented an accepted operation: true meant immediate
    # approval and false meant manager approval. No other field implies success.
    return isinstance(result.get("isAutoApproved"), bool)


def _safe_action_result(
    action_type: str,
    success: bool,
    operation_result: object | None = None,
) -> str:
    if not success:
        return "failed"
    if action_type in {"book_day_type", "create_exceptional_entry"}:
        return "submitted_for_approval"
    if action_type == "create_exceptional_entry_from_summary":
        return (
            "auto_approved"
            if isinstance(operation_result, dict)
            and operation_result.get("isAutoApproved") is True
            else "submitted_for_approval"
        )
    if action_type in {"cancel_day_type_request", "cancel_exceptional_entry"}:
        return "cancelled"
    if action_type in {"approve_supervisor_request", "approve_all_requests"}:
        return "approved"
    if action_type == "update_notification_read_status":
        return "updated"
    return "succeeded"


def _from_summary_result_message(
    result: object,
    language: str,
    *,
    success: bool,
) -> str | None:
    if not isinstance(result, dict):
        return None
    api_message = result.get("message")
    warning = result.get("warning")
    requested = result.get("requestedMinutes")
    remaining = result.get("remaining")
    resets_on = result.get("resetsOn")
    auto_approved = result.get("isAutoApproved")
    if language == "ar":
        if not success:
            lead = "تعذّر إكمال تصحيح حضورك."
        elif auto_approved is True:
            lead = "تمت الموافقة تلقائيًا على تصحيح حضورك."
        elif auto_approved is False:
            lead = "تم إرسال تصحيح حضورك وهو بانتظار موافقة المدير."
        else:
            lead = "تم إكمال تصحيح حضورك."
        facts = []
        if success:
            if requested is not None:
                facts.append(f"الدقائق المطلوبة: {requested}.")
            if remaining is not None:
                facts.append(f"المتبقي: {remaining} دقيقة.")
            if resets_on not in (None, ""):
                facts.append(f"إعادة التعيين: {resets_on}.")
        parts = [lead, *facts]
        if isinstance(api_message, str) and api_message.strip():
            parts.append(api_message.strip())
        if isinstance(warning, str) and warning.strip():
            parts.append(f"تنبيه: {warning.strip()}")
        return " ".join(parts)
    if not success:
        lead = "Your attendance correction could not be completed."
    elif auto_approved is True:
        lead = "Your attendance correction was auto-approved."
    elif auto_approved is False:
        lead = "Your attendance correction was submitted and is waiting for manager approval."
    else:
        lead = "Your attendance correction was completed."
    facts = []
    if success:
        if requested is not None:
            facts.append(f"Requested: {requested} minutes.")
        if remaining is not None:
            facts.append(f"Remaining allowance: {remaining} minutes.")
        if resets_on not in (None, ""):
            facts.append(f"Resets on {resets_on}.")
    parts = [lead, *facts]
    if isinstance(api_message, str) and api_message.strip():
        parts.append(api_message.strip())
    if isinstance(warning, str) and warning.strip():
        parts.append(f"Warning: {warning.strip()}")
    return " ".join(parts)


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
    """Bind one validated identity to a text or voice chat turn."""

    language = detected_language or detect_language(request.message)
    try:
        identity = resolve_request_identity(request.email, request.instance)
    except RequestIdentityError:
        message = (
            "يجب إرسال البريد الإلكتروني والجهة معًا لهذا الطلب."
            if language == "ar"
            else "The demo user email and ResourcePlus instance must be sent together."
        )
        return ChatResponse(
            success=False,
            message=message,
            language=language,
            session_id=request.session_id or "identity-required",
        ).set_speech_message(message)

    identity_token = bind_request_identity(identity)
    try:
        response = await _process_chat(
            request,
            detected_language=detected_language,
            store=store,
        )
        if _has_trusted_resourceplus_result(response):
            store.save_trusted_result(
                response.session_id,
                message=response.message,
                tools_used=response.tools_used,
                language=response.language,
            )
        return response
    except SessionIdentityMismatch:
        record_safe_error("access_denied")
        message = (
            "لا يمكن استخدام هذه المحادثة مع المستخدم الحالي. ابدأ محادثة جديدة."
            if language == "ar"
            else (
                "This conversation cannot be used with the current signed-in user. "
                "Start a new conversation."
            )
        )
        return ChatResponse(
            success=False,
            message=message,
            language=language,
            session_id=request.session_id or "identity-mismatch",
        ).set_speech_message(message)
    finally:
        reset_request_identity(identity_token)


async def _process_chat(
    request: ChatRequest,
    *,
    detected_language: str | None = None,
    store: SessionStore = session_store,
) -> ChatResponse:
    """Run a chat turn inside an already-bound request identity."""

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
        if action.action_type == "create_exceptional_entry_from_summary":
            success = _from_summary_operation_success(operation_result)
        action_result = _safe_action_result(
            action.action_type,
            success,
            operation_result,
        )
        record_action_state(
            action_type=action.action_type,
            state="executed" if success else "failed",
            confirmation_required=False,
            confirmed=True,
            result=action_result,
        )
        response_blocks = []
        from_summary_message = (
            _from_summary_result_message(
                operation_result,
                flow_language,
                success=success,
            )
            if action.action_type == "create_exceptional_entry_from_summary"
            else None
        )
        deterministic_result = _deterministic_action_result(
            action.action_type,
            action_result,
            flow_language,
        )
        if from_summary_message is not None:
            message = from_summary_message
            speech_message = message
            response_blocks = exceptional_submission_blocks(
                operation_result,
                language=flow_language,
                success=success,
            )
        elif deterministic_result is None:
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
            blocks=response_blocks,
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
            blocks=[confirmation_block(message, flow_language)],
        ).set_speech_message(message)

    if is_ambiguous_transactional_utterance(request.message):
        message = (
            "ما فهمت طلبك بوضوح. ممكن تعيده وتوضح الإجراء الذي تريده؟"
            if language == "ar"
            else (
                "I didn't catch that clearly. Please repeat what you'd like me "
                "to help with."
            )
        )
        store.append_history(session_id, "user", request.message)
        store.append_history(session_id, "assistant", message)
        return ChatResponse(
            success=True,
            message=message,
            language=language,
            session_id=session_id,
        ).set_speech_message(message)

    existing_late_punch = is_existing_late_punch(request.message)
    if (
        existing_late_punch or is_prospective_late_arrival(request.message)
    ) and not is_less_hours_request(request.message):
        # A prospective arrival delay is not an existing attendance record to repair.
        # Clear only a non-executable reason draft so it cannot hijack this new topic.
        store.clear_exceptional_entry_draft(session_id)
        message = late_arrival_unavailable_message(
            language,
            already_punched=existing_late_punch,
        )
        store.append_history(session_id, "user", request.message)
        store.append_history(session_id, "assistant", message)
        return ChatResponse(
            success=True,
            message=message,
            language=language,
            session_id=session_id,
        ).set_speech_message(message)

    draft = store.get_exceptional_entry_draft(session_id)
    if draft is not None:
        draft_follow_up = _draft_follow_up_kind(
            request.message,
            draft.reason_options,
        )
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
                    blocks=[reason_actions(exc.reason_options or [], draft.language)]
                    if exc.reason_options
                    else [],
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
                blocks=[confirmation_block(message, draft.language)],
            ).set_speech_message(message)

        if draft_follow_up == "new_intent":
            # Abandon only the non-executable draft and isolate the new request from
            # its stale selection context. PendingAction handling occurred above and
            # is intentionally unaffected by this branch.
            store.clear_exceptional_entry_draft(session_id)
            ignore_prior_history_for_topic_change = True
        elif draft_follow_up == "casual":
            # Keep the recoverable draft, but do not feed its reason-question history
            # into this ordinary conversational turn.
            ignore_prior_history_for_topic_change = True

    trusted_context = store.get_trusted_result(session_id)
    contextual_fast_read: str | None = None
    if is_noisy_balance_reference(request.message):
        if (
            trusted_context is not None
            and "get_exceptional_entry_balance" in trusted_context.tools_used
        ):
            contextual_fast_read = "Show my buffer balance"
        else:
            message = "Do you mean your remaining buffer time?"
            store.append_history(session_id, "user", request.message)
            store.append_history(session_id, "assistant", message)
            return ChatResponse(
                success=True,
                message=message,
                language=language,
                session_id=session_id,
            ).set_speech_message(message)
    elif (
        trusted_context is not None
        and "get_attendance_summary" in trusted_context.tools_used
        and _PREVIOUS_MONTH_FOLLOW_UP.match(request.message)
    ):
        contextual_fast_read = "Show my attendance previous month"

    if contextual_fast_read is not None:
        fast_result = await try_fast_read(
            contextual_fast_read,
            lang=lang,
            response_language=language,
        )
        if fast_result is not None:
            store.append_history(session_id, "user", request.message)
            store.append_history(session_id, "assistant", fast_result.message)
            return ChatResponse(
                success=True,
                message=fast_result.message,
                language=language,
                tools_used=fast_result.tools_used,
                session_id=session_id,
                blocks=fast_result.blocks,
            ).set_speech_message(fast_result.speech_message)

    conversational = await continue_conversation(
        request.message,
        lang=lang,
        session_id=session_id,
        language=language,
        store=store,
    )
    if conversational is not None:
        store.append_history(session_id, "user", request.message)
        store.append_history(session_id, "assistant", conversational.message)
        return ChatResponse(
            success=conversational.success,
            message=conversational.message,
            language=language,
            tools_used=conversational.tools_used,
            session_id=session_id,
            requires_confirmation=conversational.requires_confirmation,
            confirmation_id=conversational.confirmation_id,
            needs_reason=conversational.needs_reason,
            reason_options=[
                {"label": name, "value": name}
                for name in (conversational.reason_options or [])
            ] or None,
            blocks=conversational.blocks,
        ).set_speech_message(conversational.speech_message)

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
    refreshed_same_source = bool(
        trusted_context is not None
        and not result.tool_failed
        and set(trusted_context.tools_used) & set(result.tools_used)
    )
    if (
        trusted_context is not None
        and not refreshed_same_source
        and _model_retracts_trusted_result(result.message, trusted_context)
    ):
        guarded_message = _grounding_guard_message(trusted_context, language)
        result = AgentResult(
            message=guarded_message,
            speech_message=guarded_message,
            tools_used=[],
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
    response_blocks = list(result.blocks or [])
    if result.requires_confirmation and result.confirmation_id and not response_blocks:
        response_blocks.append(confirmation_block(result.message, language))
    if result.needs_reason and result.reason_options and not response_blocks:
        response_blocks.append(reason_actions(result.reason_options, language))
    response = ChatResponse(
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
        blocks=response_blocks,
    ).set_speech_message(result.speech_message)
    if "get_missing_punch_suggestions" in result.tools_used:
        today = resourceplus_today()
        store.save_conversation_draft(
            session_id,
            intent="missing_punch_context",
            slots={
                "period_start": today.replace(day=1).isoformat(),
                "period_end": today.isoformat(),
            },
            language=language,
        )
    return response
