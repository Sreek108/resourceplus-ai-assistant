import asyncio
import json
import re
from datetime import date, datetime

from app.ai.actions import (
    ActionResolutionRequired,
    confirmation_summary,
    execute_pending_action,
    prepare_exceptional_entry_reason_follow_up,
    revalidate_pending_action,
)
from app.ai.tools import execute_tool, resolve_relative_date_range
from app.ai.attendance_intent import (
    is_ambiguous_transactional_utterance,
    is_existing_late_punch,
    is_explicit_missing_punch_correction,
    is_prospective_late_arrival,
    late_arrival_unavailable_message,
)
from app.ai.reason_matcher import deterministic_reason_match
from app.ai.conversation import (
    continue_conversation,
    is_leave_balance_request,
    is_less_hours_request,
    leave_day_type_choice_blocks,
    leave_day_type_options,
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
    ApprovalCandidate,
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
from app.services.fast_reads import (
    classify_fast_read,
    is_noisy_balance_reference,
    try_fast_read,
)
from app.services.response_blocks import (
    approvals_block,
    confirmation_block,
    exceptional_submission_blocks,
    leave_balance_block,
    leave_balance_value,
    reason_actions,
)
from app.services.approval_selection import (
    approval_candidates as resolve_approval_candidates,
    resolve_pending_approval,
)
from app.services.request_history import parse_request_history_query
from app.resourceplus import ResourcePlusError
from app.resourceplus.home import get_home_data
from app.resourceplus.reference_cache import cached_day_types
from app.speech.language import (
    content_matches_language,
    count_script_letters,
    resolve_session_language,
)
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
        "en": "Okay, I won't submit it.",
        "ar": "حسنًا، ما راح أرسله.",
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

VOICE_LANGUAGE_FALLBACK = {
    "en": (
        "I understood your request, but I couldn't produce a safe English response. "
        "Please try again."
    ),
    "ar": "فهمت طلبك، لكن ما قدرت أجهز ردًا عربيًا بشكل آمن. حاول مرة ثانية.",
}


def enforce_voice_response_language(
    response: ChatResponse,
    expected_language: str,
) -> tuple[ChatResponse, str]:
    """Make content, response metadata, and later TTS language agree."""

    expected = expected_language if expected_language in {"en", "ar"} else "en"
    display = response.display_message or response.message
    speech = response.speech_message or response.message
    metadata_matches = response.language == expected
    content_matches = (
        content_matches_language(response.message, expected)
        and content_matches_language(display, expected)
        and content_matches_language(speech, expected)
    )
    if metadata_matches and content_matches:
        return response, "passed"
    fallback = VOICE_LANGUAGE_FALLBACK[expected]
    response.message = fallback
    response.display_message = fallback
    response.language = expected
    response.set_speech_message(fallback)
    result = (
        "safe_fallback_content_mismatch"
        if not content_matches
        else "safe_fallback_metadata_mismatch"
    )
    return response, result

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
_NAMED_MONTH_FOLLOW_UP = re.compile(
    r"^\s*(?:what\s+about\s+|and\s+)?(?:january|february|march|april|may|"
    r"june|july|august|september|october|november|december|jan|feb|mar|apr|"
    r"jun|jul|aug|sep|sept|oct|nov|dec)(?:\s+20\d{2})?[.!?\s]*$",
    re.IGNORECASE,
)
_RECENT_REQUEST_READ = re.compile(
    r"^\s*(?:"
    r"(?:show|check)\s+(?:my\s+|the\s+)?(?:pending\s+)?request|"
    r"what\s+happened\s+(?:to|in)\s+(?:my\s+request|it)|"
    r"is\s+(?:my\s+(?:leave\s+)?request|my\s+leave|it)\s+approved|"
    r"what(?:'s|\s+is)\s+(?:the\s+)?(?:status|my\s+leave\s+status)|"
    r"what(?:'s|\s+is)\s+the\s+status\s+of\s+my\s+request"
    r")\s*[?.!]?\s*$",
    re.IGNORECASE,
)
_SHORT_APPROVAL_FOLLOW_UP = re.compile(
    r"^\s*(?:yes[,\s]+)?(?:please\s+)?(approve|reject)"
    r"(?:\s+(?:it|this|the\s+request|that\s+request))?\s*[.!]?\s*$",
    re.IGNORECASE,
)
_EXPLICIT_BULK_APPROVAL = re.compile(
    r"^\s*(approve|reject)\s+(?:everything|all(?:\s+pending)?(?:\s+requests?)?)\s*[.!]?\s*$",
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
    "approve_supervisor_request": {
        "approved": {
            "en": {
                "display": "Done — the request has been approved.",
                "speech": "Done, the request has been approved.",
            },
            "ar": {
                "display": "تمت الموافقة على الطلب.",
                "speech": "تمت الموافقة على الطلب.",
            },
        },
        "rejected": {
            "en": {
                "display": "Done — the request has been rejected.",
                "speech": "Done, the request has been rejected.",
            },
            "ar": {
                "display": "تم رفض الطلب.",
                "speech": "تم رفض الطلب.",
            },
        },
        "approval_pending_verification": {
            "en": {
                "display": (
                    "The approval was accepted, but the request is still showing as "
                    "pending. I won't submit it again."
                ),
                "speech": (
                    "The approval was accepted, but it's still showing as pending. "
                    "I won't submit it again."
                ),
            },
            "ar": {
                "display": (
                    "تم قبول إجراء الموافقة، لكن الطلب لا يزال ظاهر بانتظار الموافقة. "
                    "ما راح أعيد إرساله."
                ),
                "speech": (
                    "تم قبول الموافقة، لكن الطلب للحين ظاهر بانتظار الموافقة. "
                    "ما راح أعيد إرساله."
                ),
            },
        },
        "approval_verification_unavailable": {
            "en": {
                "display": (
                    "The approval was accepted, but I couldn't confirm the request's "
                    "current status. I won't submit it again."
                ),
                "speech": (
                    "The approval was accepted, but I couldn't confirm its current "
                    "status. I won't submit it again."
                ),
            },
            "ar": {
                "display": (
                    "تم قبول إجراء الموافقة، لكن ما قدرت أتأكد من حالة الطلب الحالية. "
                    "ما راح أعيد إرساله."
                ),
                "speech": (
                    "تم قبول الموافقة، لكن ما قدرت أتأكد من الحالة الحالية. "
                    "ما راح أعيد إرساله."
                ),
            },
        },
    },
    "create_exceptional_entry": {
        "submitted_for_approval": {
            "en": {
                "display": "I've sent your attendance correction for approval.",
                "speech": "I've sent your attendance correction for approval.",
            },
            "ar": {
                "display": "أرسلت تصحيح حضورك للموافقة.",
                "speech": "أرسلت تصحيح حضورك للموافقة.",
            },
        },
        "failed": {
            "en": {
                "display": (
                    "I couldn't send your attendance correction. Nothing was recorded."
                ),
                "speech": (
                    "I couldn't send your attendance correction, so nothing was recorded."
                ),
            },
            "ar": {
                "display": (
                    "ما قدرت أرسل تصحيح حضورك. ما تسجّل أي طلب."
                ),
                "speech": (
                    "ما قدرت أرسل تصحيح حضورك، وما تسجّل أي طلب."
                ),
            },
        },
    },
    "cancel_exceptional_entry": {
        "cancelled": {
            "en": {
                "display": "Your attendance correction request was cancelled.",
                "speech": "Your attendance correction request was cancelled.",
            },
            "ar": {
                "display": "تم إلغاء طلب تصحيح الحضور.",
                "speech": "تم إلغاء طلب تصحيح الحضور.",
            },
        },
        "failed": {
            "en": {
                "display": "I couldn't cancel that attendance correction request.",
                "speech": "I couldn't cancel that attendance correction request.",
            },
            "ar": {
                "display": "ما قدرت ألغي طلب تصحيح الحضور.",
                "speech": "ما قدرت ألغي طلب تصحيح الحضور.",
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
        return "حسنًا، ما راح أرسله."
    return "Okay, I won't submit it."


def _deterministic_greeting(message: str, language: str) -> str | None:
    normalized = _normalized_reply(message)
    english = {
        "hi", "hello", "hey", "good morning", "good afternoon", "good evening",
        "hi how are you", "hello how are you", "hey how are you",
    }
    arabic = {
        "مرحبا", "اهلا", "السلام عليكم", "صباح الخير", "مساء الخير",
        "مرحبا كيف حالك", "اهلا كيف حالك",
    }
    if normalized in english:
        return "Hi! What can I help you with?"
    if normalized in arabic:
        return "أهلًا! كيف أقدر أساعدك؟"
    return None


def _human_request_date(value: str, language: str) -> str:
    try:
        parsed = date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        return value
    return parsed.isoformat() if language == "ar" else f"{parsed.strftime('%b')} {parsed.day}"


def _canonical_request_date(value: object) -> str:
    text = str(value or "").strip()
    for candidate in (text[:10], text):
        try:
            return date.fromisoformat(candidate).isoformat()
        except ValueError:
            pass
    for pattern in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text[:10], pattern).date().isoformat()
        except ValueError:
            continue
    return text


def _recent_request_message(
    trusted: TrustedResultContext,
    rows: tuple[dict[str, object], ...],
    language: str,
) -> tuple[str, str | None]:
    target_date = trusted.recent_request_date or ""
    matching = [
        row for row in rows
        if _canonical_request_date(row.get("date")) == target_date
    ]
    approved = any(
        str(row.get("status", "")).strip().casefold() == "approved"
        for row in matching
    )
    rejected = any(
        str(row.get("status", "")).strip().casefold() == "rejected"
        for row in matching
    )
    category = trusted.recent_request_category or "attendance_correction"
    detail = trusted.recent_request_detail or (
        "attendance" if category == "attendance_correction" else "Leave"
    )
    shown_date = _human_request_date(target_date, language)
    is_leave = category in {"leave", "absence", "day_type"}
    if approved:
        if language == "ar":
            return (
                f"تمت الموافقة على طلب {detail} ليوم {shown_date}."
                if is_leave
                else f"تمت الموافقة على تصحيح {detail} ليوم {shown_date}.",
                "approved",
            )
        return (
            f"Your {detail} request for {shown_date} has been approved."
            if is_leave
            else f"Your {shown_date} {detail} correction has been approved.",
            "approved",
        )
    if rejected:
        if language == "ar":
            return (
                f"تم رفض طلب {detail} ليوم {shown_date}."
                if is_leave
                else f"تم رفض تصحيح {detail} ليوم {shown_date}.",
                "rejected",
            )
        return (
            f"Your {detail} request for {shown_date} was rejected."
            if is_leave
            else f"Your {shown_date} {detail} correction was rejected.",
            "rejected",
        )
    if trusted.recent_request_state == "submitted_for_approval":
        if language == "ar":
            return (
                (
                    f"تم إرسال طلب {detail} ليوم {shown_date} للموافقة، "
                    "ولا يزال بانتظار إجراء المدير."
                    if is_leave
                    else f"تم إرسال تصحيح {detail} ليوم {shown_date} للموافقة، "
                    "ولا يزال بانتظار إجراء المدير."
                ),
                None,
            )
        return (
            (
                f"Your {detail} request for {shown_date} has been sent for approval "
                "and is still awaiting manager action."
                if is_leave
                else f"Your {shown_date} {detail} correction has been sent for "
                "approval and is still awaiting manager action."
            ),
            None,
        )
    return "", None


def _apply_recent_request_context_to_blocks(
    blocks: list[object],
    trusted: TrustedResultContext,
) -> None:
    """Fill missing presentation fields without creating executable identity."""

    target_date = trusted.recent_request_date
    detail = trusted.recent_request_detail
    category = trusted.recent_request_category
    for block in blocks:
        if getattr(block, "type", None) != "table" or not getattr(block, "rows", None):
            continue
        rows = block.rows
        candidates = [
            row
            for row in rows
            if not target_date
            or _canonical_request_date(row.get("date")) == target_date
        ]
        if not candidates and len(rows) == 1:
            candidates = rows
        for row in candidates:
            if target_date and str(row.get("date", "")).strip() in {"", "—", "None"}:
                row["date"] = target_date
            if detail and str(row.get("detail", "")).strip() in {"", "—", "None"}:
                row["detail"] = detail
            raw_type = str(row.get("type", "")).strip()
            if category in {"leave", "absence", "day_type"} and detail:
                if raw_type.casefold().replace("_", "") in {
                    "",
                    "—",
                    "absence",
                    "leave",
                    "daytype",
                }:
                    row["type"] = detail
            elif category == "attendance_correction" and raw_type.casefold().replace(
                "_", ""
            ) in {"", "—", "exceptionalentry", "exceptionentry"}:
                row["type"] = "Attendance correction"


def _approval_follow_up_command(message: str) -> tuple[str, bool] | None:
    normalized = _normalized_reply(message)
    bulk = _EXPLICIT_BULK_APPROVAL.match(message)
    if bulk is not None:
        return bulk.group(1).casefold(), True
    single = _SHORT_APPROVAL_FOLLOW_UP.match(message)
    if single is not None:
        return single.group(1).casefold(), False
    tokens = normalized.split()
    if (
        tokens
        and tokens[0] in {"approve", "reject"}
        and len(tokens) <= 12
        and not any(token in {"all", "everything"} for token in tokens[1:])
    ):
        return tokens[0], False
    arabic_decision = (
        "reject" if any(token in {"ارفض", "رفض"} for token in tokens)
        else "approve" if "وافق" in tokens
        else None
    )
    if arabic_decision is None:
        return None
    explicit_bulk = any(cue in normalized for cue in ("الكل", "جميع", "كل الطلبات"))
    short_tokens = len(normalized.split()) <= 10
    return (arabic_decision, explicit_bulk) if explicit_bulk or short_tokens else None


def _approval_clarification_response(
    candidates: tuple[ApprovalCandidate, ...],
    *,
    language: str,
    session_id: str,
    no_match: bool = False,
) -> ChatResponse:
    if no_match:
        message = (
            "ما قدرت ألقى طلب الموافقة المعلّق الذي تقصده."
            if language == "ar"
            else "I couldn't find that pending request."
        )
        blocks = []
    else:
        message = (
            "لقيت أكثر من طلب مطابق. أي طلب تقصد؟"
            if language == "ar"
            else "I found more than one matching pending request. Which one do you mean?"
        )
        blocks = [approvals_block(candidates, language)]
    return ChatResponse(
        success=True,
        message=message,
        language=language,
        session_id=session_id,
        blocks=blocks,
    ).set_speech_message(message)


def _is_recent_request_read(message: str) -> bool:
    if _RECENT_REQUEST_READ.match(message):
        return True
    normalized = _normalized_reply(message)
    return any(
        cue in normalized
        for cue in (
            "اعرض طلبي",
            "اظهر طلبي",
            "ورني طلبي",
            "وش حالة طلبي",
            "ما حالة طلبي",
            "وش صار على طلبي",
            "وش صار عليه",
            "هل تمت الموافقة على طلبي",
            "هل اجازتي معتمدة",
            "اعرض الطلب المعلق",
            "ما هي الحالة",
            "وش الحالة",
        )
    )


_AMBIGUOUS_QUICK_LIST = re.compile(
    r"^\s*(?:please\s+)?(?:show|list|display|view)\s+(?:me\s+)?(?:my\s+)?"
    r"quick\s+list\s*[?.!]*\s*$",
    re.I,
)


def _is_ambiguous_quick_list(message: str) -> bool:
    return bool(_AMBIGUOUS_QUICK_LIST.fullmatch(message))


def _has_recent_request_history_context(
    trusted: TrustedResultContext | None,
    history: list[dict[str, str]],
) -> bool:
    if trusted is not None and (
        trusted.recent_request_date is not None
        or bool(trusted.request_correlations)
        or bool(
            {"get_my_request_status", "get_my_day_type_requests"}
            & set(trusted.tools_used)
        )
    ):
        return True
    return any(
        item.get("role") == "user"
        and parse_request_history_query(item.get("content", "")) is not None
        for item in history[-6:]
    )


def _tool_result_response(
    result: object,
    *,
    tool_name: str,
    language: str,
    session_id: str,
) -> ChatResponse:
    pending_action = getattr(result, "pending_action", None)
    if pending_action is not None:
        summary = pending_action.summary
        return ChatResponse(
            success=True,
            message=summary,
            language=language,
            tools_used=[tool_name],
            session_id=session_id,
            requires_confirmation=True,
            confirmation_id=pending_action.confirmation_id,
            blocks=[confirmation_block(summary, language)],
        ).set_speech_message(summary)
    message = getattr(result, "terminal_message", None)
    if not isinstance(message, str) or not message:
        message = (
            "I couldn't prepare that approval."
            if language == "en"
            else "ما قدرت أجهز إجراء الموافقة."
        )
    return ChatResponse(
        success=not bool(getattr(result, "failed", True)),
        message=message,
        language=language,
        tools_used=[tool_name] if getattr(result, "tool_used", False) else [],
        session_id=session_id,
    ).set_speech_message(message)


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
            "Your request was completed."
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
    if action_type == "approve_supervisor_request":
        if isinstance(operation_result, dict):
            verification = operation_result.get("_approval_verification")
            if verification == "verified":
                return (
                    "approved"
                    if int(operation_result.get("_approval_status", 1)) == 1
                    else "rejected"
                )
            if verification == "still_pending":
                return "approval_pending_verification"
        return "approval_verification_unavailable"
    if action_type == "approve_all_requests":
        if isinstance(operation_result, dict):
            verification = operation_result.get("_bulk_approval_verification")
            if verification == "verified":
                return "approved"
            if verification == "still_pending":
                return "approval_pending_verification"
        return "approval_verification_unavailable"
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
            lead = "ما قدرت أكمل تصحيح حضورك."
        elif auto_approved is True:
            lead = "تم تصحيح حضورك واعتماده تلقائياً."
        elif auto_approved is False:
            lead = "أرسلت تصحيح حضورك لموافقة مديرك."
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
        if not success and isinstance(api_message, str) and api_message.strip():
            parts.append(
                f"التفاصيل: {api_message.strip()}"
            )
        if isinstance(warning, str) and warning.strip():
            parts.append(f"تنبيه: {warning.strip()}")
        return " ".join(parts)
    if not success:
        lead = "I couldn't complete that attendance correction."
    elif auto_approved is True:
        lead = "Done — your attendance correction was approved automatically."
    elif auto_approved is False:
        lead = "I've sent your attendance correction to your manager for approval."
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
    if not success and isinstance(api_message, str) and api_message.strip():
        parts.append(f"Details: {api_message.strip()}")
    if isinstance(warning, str) and warning.strip():
        parts.append(f"Warning: {warning.strip()}")
    return " ".join(parts)


def _from_summary_speech_message(
    result: object,
    language: str,
    *,
    success: bool,
) -> str:
    payload = result if isinstance(result, dict) else {}
    if language == "ar":
        if not success:
            return "ما قدرت أكمل تصحيح حضورك. راجع التفاصيل المعروضة."
        if payload.get("isAutoApproved") is True:
            return "تم تصحيح حضورك واعتماده تلقائياً."
        if payload.get("isAutoApproved") is False:
            return "أرسلت تصحيح حضورك لموافقة مديرك."
        return "تم تصحيح حضورك."
    if not success:
        return "I couldn't complete your attendance correction. The details are on screen."
    if payload.get("isAutoApproved") is True:
        return "Done — your attendance correction was approved automatically."
    if payload.get("isAutoApproved") is False:
        return "I've sent your attendance correction to your manager for approval."
    return "Your attendance correction is complete."


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


def _book_day_type_result_message(
    arguments: dict[str, object],
    language: str,
    *,
    success: bool,
) -> tuple[str, str]:
    """Render a validated leave/travel result without another model request."""

    day_type = str(arguments.get("day_type_name") or "").strip()
    request_date = str(arguments.get("date_from") or "").strip()
    shown_date = _human_request_date(request_date, language)
    if not success:
        message = (
            f"ما قدرت أرسل طلب {day_type} ليوم {shown_date}."
            if language == "ar"
            else f"I couldn't submit your {day_type} request for {shown_date}."
        )
    elif language == "ar":
        message = f"تم إرسال طلب {day_type} ليوم {shown_date} إلى مديرك للموافقة."
    else:
        message = (
            f"Your {day_type} request for {shown_date} has been sent to your "
            "manager for approval."
        )
    return message, message


def _leave_balance_message(payload: object, language: str) -> str:
    value = leave_balance_value(payload)
    if value in (None, ""):
        return (
            "لم يُرجع ResourcePlus رصيد إجازة مستحقًا حاليًا."
            if language == "ar"
            else "ResourcePlus did not return a current eligible leave balance."
        )
    shown = str(value).strip()
    has_unit = bool(re.search(r"[A-Za-z\u0600-\u06ff]", shown))
    if language == "ar":
        return f"رصيد إجازتك المستحق حاليًا هو {shown}{'' if has_unit else ' يومًا'}."
    unit = " day" if shown == "1" else " days"
    return f"Your current eligible leave balance is {shown}{'' if has_unit else unit}."


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
            is_resolved_attendance = {
                "get_attendance_summary", "get_exceptional_entry_requests"
            } <= set(response.tools_used)
            correction_dates = (
                tuple(
                    match.group()
                    for block in response.blocks
                    if block.type == "actions"
                    for action in block.actions
                    if (match := re.search(r"\b20\d{2}-\d{2}-\d{2}\b", action.value))
                )
                if is_resolved_attendance
                else None
            )
            displayed_dates = tuple(
                str(row.get("date"))
                for block in response.blocks
                if block.type == "table" and block.title in {
                    "Attendance gaps", "فجوات الحضور",
                    "Less-hours attendance", "الحضور بساعات ناقصة",
                }
                for row in block.rows
                if re.fullmatch(r"20\d{2}-\d{2}-\d{2}", str(row.get("date", "")))
            )
            resolved_period = resolve_relative_date_range(
                request.message,
                today=resourceplus_today(),
            )
            existing_context = store.get_trusted_result(response.session_id)
            attendance_period = (
                (
                    resolved_period.from_date.isoformat(),
                    resolved_period.to_date.isoformat(),
                )
                if "get_attendance_summary" in response.tools_used
                and resolved_period is not None
                else (
                    existing_context.attendance_period
                    if "get_attendance_summary" in response.tools_used
                    and existing_context is not None
                    and existing_context.attendance_period is not None
                    else (min(displayed_dates), max(displayed_dates))
                    if displayed_dates
                    else (
                        (
                            resourceplus_today().replace(day=1).isoformat(),
                            resourceplus_today().isoformat(),
                        )
                        if "get_attendance_summary" in response.tools_used
                        else None
                    )
                )
            )
            attendance_period_label = None
            if attendance_period is not None:
                period_start = date.fromisoformat(attendance_period[0])
                period_end = date.fromisoformat(attendance_period[1])
                attendance_period_label = (
                    period_start.strftime("%B")
                    if period_start.year == period_end.year
                    and period_start.month == period_end.month
                    else f"{attendance_period[0]} to {attendance_period[1]}"
                )
            store.save_trusted_result(
                response.session_id,
                message=response.message,
                tools_used=response.tools_used,
                language=response.language,
                correction_dates=correction_dates,
                attendance_period=attendance_period,
                attendance_period_label=attendance_period_label,
                attendance_period_source=(
                    "explicit_user_period"
                    if resolved_period is not None
                    else existing_context.attendance_period_source
                    if existing_context is not None
                    and existing_context.attendance_period_source is not None
                    else "inferred_default"
                ),
                discussed_date=(
                    displayed_dates[0] if len(displayed_dates) == 1 else None
                ),
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
    if detected_language is None:
        text_language = resolve_session_language(
            request.message,
            last_confident_language=store.get_last_confident_language(session_id),
            fallback="ar" if lang == 2 else "en",
        )
        language = text_language.language
        store.set_last_confident_language(
            session_id,
            text_language.last_confident_language,
        )
    else:
        language = detected_language
    trusted_context = store.get_trusted_result(session_id)
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
        # The immutable PendingAction owner is the authoritative identity for the
        # rest of this turn. Rebind it after authorization so confirmation-time
        # reads, the write, and post-write session updates cannot fall through to
        # the local RP_INSTANCE default if ambient request context was lost.
        bind_request_identity(action.owner)
        record_action_state(
            action_type=action.action_type,
            state="prepared",
            confirmation_required=False,
            confirmed=True,
        )
        revalidation_tools: list[str] = []
        if (
            action.action_type == "create_exceptional_entry_from_summary"
            and isinstance(
                action.validated_arguments.get("attendance_snapshot"),
                dict,
            )
        ):
            revalidation_tools = ["get_attendance_summary", "get_exception_reasons"]
            for tool_name in revalidation_tools:
                record_tool_usage(tool_name)
        elif action.action_type == "approve_supervisor_request":
            revalidation_tools = ["get_pending_approvals"]
            record_tool_usage("get_pending_approvals")
        execution_tools = [*revalidation_tools, action.action_type]
        try:
            await revalidate_pending_action(
                action.action_type,
                action.validated_arguments,
                lang=lang,
                response_language=flow_language,
            )
            operation_result = await execute_pending_action(
                action.action_type,
                action.validated_arguments,
            )
        except ActionResolutionRequired as exc:
            record_action_state(
                action_type=action.action_type,
                state="failed",
                confirmation_required=False,
                confirmed=True,
                result="stale",
            )
            record_safe_error(exc.category)
            message = str(exc)
            store.append_history(session_id, "user", request.message)
            store.append_history(session_id, "assistant", message)
            return ChatResponse(
                success=False,
                message=message,
                language=flow_language,
                tools_used=execution_tools,
                session_id=session_id,
            ).set_speech_message(message)
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
                tools_used=execution_tools,
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
            speech_message = _from_summary_speech_message(
                operation_result, flow_language, success=success
            )
            response_blocks = exceptional_submission_blocks(
                operation_result,
                language=flow_language,
                success=success,
            )
        elif action.action_type == "approve_all_requests":
            count = int(action.validated_arguments.get("count", 0))
            if action_result == "approved":
                approving = action.validated_arguments.get("status") == 1
                if flow_language == "ar":
                    verb = "وافقت على" if approving else "رفضت"
                    message = f"تم — {verb} جميع الطلبات المعلقة وعددها {count}."
                else:
                    verb = "approved" if approving else "rejected"
                    message = f"Done — I {verb} all {count} pending requests."
            elif action_result == "approval_pending_verification":
                message = (
                    "تم قبول الإجراء، لكن لا تزال بعض الطلبات ظاهرة كمعلقة. "
                    "ما راح أعيد الإرسال."
                    if flow_language == "ar"
                    else "The bulk approval was accepted, but some requests are still showing as pending. I won't submit it again."
                )
            else:
                message = (
                    "تم قبول الإجراء، لكن ما قدرت أتحقق من حالة كل الطلبات الحالية. "
                    "ما راح أعيد الإرسال."
                    if flow_language == "ar"
                    else "The bulk approval was accepted, but I couldn't verify every request's current status. I won't submit it again."
                )
            speech_message = message
        elif action.action_type == "book_day_type":
            message, speech_message = _book_day_type_result_message(
                action.validated_arguments,
                flow_language,
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
            if (
                action.action_type == "approve_supervisor_request"
                and action.validated_arguments.get("status") == 2
                and success
                and action_result != "rejected"
            ):
                message = (
                    "تم قبول رفض الطلب، لكن ما قدرت أتأكد من حالته الحالية. ما راح أعيد إرساله."
                    if flow_language == "ar"
                    else "The rejection was accepted, but I couldn't confirm the request's current status. I won't submit it again."
                )
                speech_message = message
            if (
                action.action_type == "approve_supervisor_request"
                and action_result in {"approved", "rejected"}
            ):
                employee = action.validated_arguments.get("employee_name")
                detail = action.validated_arguments.get("detail")
                if isinstance(employee, str) and employee.strip() and isinstance(detail, str) and detail.strip():
                    approved = action_result == "approved"
                    message = (
                        f"تمت الموافقة على طلب {detail} الخاص بـ{employee}."
                        if flow_language == "ar" and approved
                        else f"تم رفض طلب {detail} الخاص بـ{employee}."
                        if flow_language == "ar"
                        else (
                            f"Done — {employee}'s {detail} request has been "
                            f"{'approved' if approved else 'rejected'}."
                        )
                    )
                    speech_message = message
                remaining_payload = (
                    operation_result.get("_remaining_pending_approvals")
                    if isinstance(operation_result, dict)
                    else None
                )
                remaining_candidates = resolve_approval_candidates(remaining_payload)
                response_blocks = [approvals_block(remaining_candidates, flow_language)]
                store.save_trusted_result(
                    session_id,
                    message=message,
                    tools_used=execution_tools,
                    language=flow_language,
                    approval_candidates=remaining_candidates,
                )
        if action.action_type == "create_exceptional_entry_from_summary" and success:
            request_date = action.validated_arguments.get("att_date")
            request_detail = action.validated_arguments.get("reason_name")
            if isinstance(request_date, str) and request_date:
                store.save_trusted_result(
                    session_id,
                    message=message,
                    tools_used=execution_tools,
                    language=flow_language,
                    discussed_date=request_date,
                    recent_request_date=request_date,
                    recent_request_detail=(
                        str(request_detail) if request_detail not in (None, "") else "attendance"
                    ),
                    recent_request_category="attendance_correction",
                    recent_request_state=action_result,
                )
        elif action.action_type == "book_day_type" and success:
            request_date = action.validated_arguments.get("date_from")
            request_detail = action.validated_arguments.get("day_type_name")
            if isinstance(request_date, str) and request_date:
                store.save_trusted_result(
                    session_id,
                    message=message,
                    tools_used=execution_tools,
                    language=flow_language,
                    discussed_date=request_date,
                    recent_request_category="leave",
                    recent_request_date=request_date,
                    recent_request_detail=(
                        str(request_detail)
                        if request_detail not in (None, "")
                        else "Leave"
                    ),
                    recent_request_state=action_result,
                )
        store.append_history(session_id, "user", request.message)
        store.append_history(session_id, "assistant", message)
        return ChatResponse(
            success=success,
            message=message,
            language=flow_language,
            tools_used=execution_tools,
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
        localized_summary = confirmation_summary(
            pending.action_type,
            pending.validated_arguments,
            flow_language,
        )
        message = localized_summary
        return ChatResponse(
            success=True,
            message=message,
            language=flow_language,
            session_id=session_id,
            requires_confirmation=True,
            confirmation_id=pending.confirmation_id,
            blocks=[confirmation_block(message, flow_language)],
        ).set_speech_message(message)

    structured_approval = request.approval_selection
    approval_command = (
        (structured_approval.decision, False)
        if structured_approval is not None
        else _approval_follow_up_command(request.message)
    )
    if approval_command is not None:
        decision_name, explicit_bulk = approval_command
        if explicit_bulk:
            result = await execute_tool(
                "prepare_approve_all_requests",
                {"decision": decision_name, "request_type": "all"},
                lang=lang,
                session_id=session_id,
                response_language=language,
                source_user_message=request.message,
                trusted_conversation_intent=True,
                store=store,
            )
            response = _tool_result_response(
                result,
                tool_name="prepare_approve_all_requests",
                language=language,
                session_id=session_id,
            )
        else:
            candidates = (
                trusted_context.approval_candidates
                if trusted_context is not None
                else ()
            )
            matches = resolve_pending_approval(
                candidates,
                request.message,
                ordinal=(structured_approval.ordinal if structured_approval else None),
            )
            if len(matches) == 1:
                selected = matches[0]
                result = await execute_tool(
                    "prepare_supervisor_request",
                    {
                        "employee_name": selected.employee_name,
                        "detail": selected.detail or None,
                        "decision": decision_name,
                        "request_id": selected.request_id,
                        "request_type": selected.request_type,
                        "category": selected.category,
                        "request_date": selected.request_date,
                        "_trusted_selector": True,
                    },
                    lang=lang,
                    session_id=session_id,
                    response_language=language,
                    source_user_message=request.message,
                    trusted_conversation_intent=True,
                    store=store,
                )
                response = _tool_result_response(
                    result,
                    tool_name="prepare_supervisor_request",
                    language=language,
                    session_id=session_id,
                )
            elif matches:
                response = _approval_clarification_response(
                    matches,
                    language=language,
                    session_id=session_id,
                )
                store.save_trusted_result(
                    session_id,
                    message=response.message,
                    tools_used=(
                        trusted_context.tools_used
                        if trusted_context is not None
                        else ("get_pending_approvals",)
                    ),
                    language=language,
                    approval_candidates=matches,
                )
            elif candidates:
                response = _approval_clarification_response(
                    (),
                    language=language,
                    session_id=session_id,
                    no_match=True,
                )
            else:
                message = (
                    "اعرض الطلبات المعلقة أولًا عشان أحدد الطلب الصحيح."
                    if language == "ar"
                    else "Show your pending approvals first so I can identify the right request."
                )
                response = ChatResponse(
                    success=True,
                    message=message,
                    language=language,
                    session_id=session_id,
                ).set_speech_message(message)
        store.append_history(session_id, "user", request.message)
        store.append_history(session_id, "assistant", response.message)
        return response

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
            message = _draft_cancelled_message(language)
            store.append_history(session_id, "user", request.message)
            store.append_history(session_id, "assistant", message)
            return ChatResponse(
                success=True,
                message=message,
                language=language,
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
                    response_language=language,
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
                    language=language,
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
                    blocks=[reason_actions(exc.reason_options or [], language)]
                    if exc.reason_options
                    else [],
                ).set_speech_message(message)
            except ResourcePlusError:
                record_safe_error("resourceplus_error")
                message = (
                    "تعذر التحقق من بيانات ResourcePlus الآن. حاول مرة أخرى."
                    if language == "ar"
                    else (
                        "I couldn't verify the ResourcePlus data right now. "
                        "Please try again."
                    )
                )
                return ChatResponse(
                    success=False,
                    message=message,
                    language=language,
                    tools_used=["prepare_exceptional_entry"],
                    session_id=session_id,
                ).set_speech_message(message)
            except (ValueError, TypeError, KeyError):
                record_safe_error("validation_error")
                message = (
                    "تعذر التحقق من بيانات التصحيح الآن. حاول مرة أخرى."
                    if language == "ar"
                    else (
                        "I couldn't validate the correction data right now. "
                        "Please try again."
                    )
                )
                return ChatResponse(
                    success=False,
                    message=message,
                    language=language,
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
                language=language,
                tools_used=["prepare_exceptional_entry"],
                session_id=session_id,
                requires_confirmation=True,
                confirmation_id=pending_action.confirmation_id,
                blocks=[confirmation_block(message, language)],
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

    request_history_message = request.message
    if _is_ambiguous_quick_list(request.message):
        if _has_recent_request_history_context(
            trusted_context,
            store.get_history(session_id),
        ):
            request_history_message = "show my request list"
        else:
            message = (
                "هل تقصد قائمة طلباتك؟"
                if language == "ar"
                else "Do you mean your request list?"
            )
            store.append_history(session_id, "user", request.message)
            store.append_history(session_id, "assistant", message)
            return ChatResponse(
                success=True,
                message=message,
                language=language,
                session_id=session_id,
            ).set_speech_message(message)

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
        and trusted_context.attendance_period is not None
        and _PREVIOUS_MONTH_FOLLOW_UP.match(request.message)
    ):
        contextual_fast_read = "Show my attendance previous month"
    elif (
        trusted_context is not None
        and trusted_context.attendance_period is not None
        and _NAMED_MONTH_FOLLOW_UP.match(request.message)
    ):
        contextual_fast_read = f"Show my attendance {request.message}"
    elif (
        trusted_context is not None
        and trusted_context.recent_request_date
        and _is_recent_request_read(request.message)
        and parse_request_history_query(request_history_message) is None
    ):
        contextual_fast_read = (
            f"Show my requests on {trusted_context.recent_request_date}"
        )

    if contextual_fast_read is not None:
        fast_result = await try_fast_read(
            contextual_fast_read,
            lang=lang,
            response_language=language,
            trusted_context=trusted_context,
        )
        if fast_result is not None:
            result_message = fast_result.message
            result_speech = fast_result.speech_message
            newer_request_state: str | None = None
            if (
                trusted_context is not None
                and trusted_context.recent_request_date
                and "requests" in classify_fast_read(contextual_fast_read)
            ):
                _apply_recent_request_context_to_blocks(
                    fast_result.blocks,
                    trusted_context,
                )
                contextual_message, newer_request_state = _recent_request_message(
                    trusted_context,
                    fast_result.request_rows,
                    language,
                )
                if contextual_message:
                    result_message = result_speech = contextual_message
                store.save_trusted_result(
                    session_id,
                    message=result_message,
                    tools_used=fast_result.tools_used,
                    language=language,
                    recent_request_state=newer_request_state,
                )
            store.append_history(session_id, "user", request.message)
            store.append_history(session_id, "assistant", result_message)
            return ChatResponse(
                success=True,
                message=result_message,
                language=language,
                tools_used=fast_result.tools_used,
                session_id=session_id,
                blocks=fast_result.blocks,
            ).set_speech_message(result_speech)

    # Employee request-history reads are explicit, side-effect-free intents.
    # Resolve them before draft/conversation routing so phrases such as Arabic
    # "attendance correction requests" cannot be mistaken for a write request.
    if parse_request_history_query(request_history_message) is not None:
        fast_result = await try_fast_read(
            request_history_message,
            lang=lang,
            response_language=language,
            trusted_context=trusted_context,
        )
        if fast_result is not None:
            # An explicit read starts a new conversational topic.  Only the
            # incomplete, non-executable draft is discarded; immutable
            # confirmation-backed PendingActions were handled above and are
            # deliberately untouched.
            store.clear_conversation_draft(session_id)
            store.save_trusted_result(
                session_id,
                message=fast_result.message,
                tools_used=fast_result.tools_used,
                language=language,
                recent_request_category=fast_result.recent_request_category,
                recent_request_date=fast_result.recent_request_date,
                recent_request_detail=fast_result.recent_request_detail,
                recent_request_state=fast_result.recent_request_state,
            )
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

    if is_leave_balance_request(request.message):
        record_tool_usage("get_home_data")
        record_tool_usage("get_day_types")
        home_result, day_type_result = await asyncio.gather(
            get_home_data(lang=lang),
            cached_day_types(lang, force_refresh=True),
            return_exceptions=True,
        )
        if isinstance(home_result, BaseException) and not isinstance(home_result, ResourcePlusError):
            raise home_result
        if isinstance(day_type_result, BaseException) and not isinstance(day_type_result, ResourcePlusError):
            raise day_type_result
        home_payload = {} if isinstance(home_result, ResourcePlusError) else home_result
        day_type_rows = (
            []
            if isinstance(day_type_result, ResourcePlusError)
            else leave_day_type_options(day_type_result)
        )
        message = _leave_balance_message(home_payload, language)
        blocks = []
        balance_ui = leave_balance_block(home_payload, language)
        if balance_ui is not None:
            blocks.append(balance_ui)
        blocks.extend(leave_day_type_choice_blocks(day_type_rows, language))
        store.clear_conversation_draft(session_id)
        if day_type_rows:
            store.save_conversation_draft(
                session_id,
                intent="book_day_type",
                slots={
                    "booking_group": "leave",
                    "state": "leave_options_shown",
                    "day_type_options": json.dumps(
                        [str(row["dayType"]) for row in day_type_rows],
                        ensure_ascii=False,
                    ),
                },
                language=language,
            )
        tools_used = ["get_home_data", "get_day_types"]
        store.save_trusted_result(
            session_id,
            message=message,
            tools_used=tools_used,
            language=language,
        )
        store.append_history(session_id, "user", request.message)
        store.append_history(session_id, "assistant", message)
        return ChatResponse(
            success=not isinstance(home_result, ResourcePlusError),
            message=message,
            language=language,
            tools_used=tools_used,
            session_id=session_id,
            blocks=blocks,
        ).set_speech_message(message)

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

    greeting = (
        _deterministic_greeting(request.message, language)
        if draft is None
        else None
    )
    if greeting is not None:
        store.append_history(session_id, "user", request.message)
        store.append_history(session_id, "assistant", greeting)
        return ChatResponse(
            success=True,
            message=greeting,
            language=language,
            session_id=session_id,
        ).set_speech_message(greeting)

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

    direct_fast_intents = classify_fast_read(request.message)
    if (
        getattr(get_settings(), "deterministic_read_fast_paths", False)
        and set(direct_fast_intents) & {"approvals", "requests"}
    ):
        fast_result = await try_fast_read(
            request.message,
            lang=lang,
            response_language=language,
            trusted_context=trusted_context,
        )
        if fast_result is not None:
            store.save_trusted_result(
                session_id,
                message=fast_result.message,
                tools_used=fast_result.tools_used,
                language=language,
                attendance_period=fast_result.period,
                recent_request_category=fast_result.recent_request_category,
                recent_request_date=fast_result.recent_request_date,
                recent_request_detail=fast_result.recent_request_detail,
                recent_request_state=fast_result.recent_request_state,
                approval_candidates=(
                    fast_result.approval_candidates
                    if "get_pending_approvals" in fast_result.tools_used
                    else None
                ),
            )
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
    result_tools = list(result.tools_used)
    response_blocks = list(result.blocks or [])
    if (
        not result.tool_failed
        and "get_home_data" in result_tools
        and is_leave_balance_request(request.message)
    ):
        try:
            day_type_rows = leave_day_type_options(
                await cached_day_types(lang, force_refresh=True)
            )
        except ResourcePlusError:
            day_type_rows = []
        if day_type_rows:
            if "get_day_types" not in result_tools:
                result_tools.append("get_day_types")
                record_tool_usage("get_day_types")
            response_blocks.extend(
                leave_day_type_choice_blocks(day_type_rows, language)
            )
            store.save_conversation_draft(
                session_id,
                intent="book_day_type",
                slots={
                    "booking_group": "leave",
                    "state": "leave_options_shown",
                    "day_type_options": json.dumps(
                        [str(row["dayType"]) for row in day_type_rows],
                        ensure_ascii=False,
                    ),
                },
                language=language,
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
    if result.requires_confirmation and result.confirmation_id and not response_blocks:
        response_blocks.append(confirmation_block(result.message, language))
    if result.needs_reason and result.reason_options and not response_blocks:
        response_blocks.append(reason_actions(result.reason_options, language))
    response = ChatResponse(
        success=not result.tool_failed,
        message=result.message,
        language=language,
        tools_used=result_tools,
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
