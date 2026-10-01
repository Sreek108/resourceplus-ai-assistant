import json
import logging
import re
from calendar import monthrange
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Any

from app.ai.actions import ActionResolutionRequired, prepare_write_action
from app.ai.attendance_intent import (
    explicit_missing_punch_direction,
    is_existing_late_punch,
    is_explicit_missing_punch_correction,
    is_prospective_late_arrival,
    late_arrival_unavailable_message,
)
from app.ai.sessions import PendingAction, SessionStore, session_store
from app.audit import record_action_state, record_safe_error, record_tool_usage
from app.resourceplus import ResourcePlusError
from app.resourceplus.approvals import get_pending_approvals
from app.resourceplus.attendance import (
    get_attendance_summary,
    get_exception_reasons,
    get_missing_punch_suggestions,
)
from app.resourceplus.employee import get_profile_data
from app.resourceplus.exceptional import get_exceptional_entry_balance
from app.resourceplus.home import get_home_data
from app.resourceplus.leave import get_day_types
from app.resourceplus.notifications import get_notifications
from app.resourceplus.requests import get_my_request_status
from app.resourceplus.reference_cache import cached_day_types, cached_exception_reasons
from app.resourceplus.missing_punch import (
    missing_punch_tool_data,
    normalize_missing_punch_suggestions,
)
from app.models.schemas import ResponseBlock
from app.services.response_blocks import attendance_blocks
from app.time_context import resourceplus_today


logger = logging.getLogger(__name__)


def _empty_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }


def _date_range_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "from_date": {
                "type": "string",
                "description": "Inclusive start date in YYYY-MM-DD format.",
            },
            "to_date": {
                "type": "string",
                "description": "Inclusive end date in YYYY-MM-DD format.",
            },
        },
        "required": ["from_date", "to_date"],
        "additionalProperties": False,
    }


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "type": "function",
        "name": "get_attendance_summary",
        "description": (
            "Get the authenticated employee's attendance summary for an exact "
            "inclusive date range."
        ),
        "parameters": _date_range_schema(),
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_home_data",
        "description": (
            "Get the authenticated employee's ResourcePlus home/dashboard data, "
            "including vacation balance, service days, assets, documents, and loans."
        ),
        "parameters": _empty_schema(),
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_profile_data",
        "description": "Get the authenticated employee's ResourcePlus profile.",
        "parameters": _empty_schema(),
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_missing_punch_suggestions",
        "description": (
            "Get ResourcePlus-reported missing IN/OUT punches grouped by date, with "
            "a separate correctable-suggestions collection. A missing punch remains "
            "visible when its suggested correction time is unavailable."
        ),
        "parameters": _date_range_schema(),
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_exception_reasons",
        "description": "Get valid ResourcePlus reasons for exceptional entries.",
        "parameters": _empty_schema(),
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_exceptional_entry_balance",
        "description": (
            "Get ResourcePlus's authoritative exceptional-entry allowance for one "
            "date. The backend supplies employee identity."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_date": {"type": "string", "description": "YYYY-MM-DD"},
            },
            "required": ["target_date"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_day_types",
        "description": "Get valid ResourcePlus leave and business-travel day types.",
        "parameters": _empty_schema(),
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_my_request_status",
        "description": (
            "Get and merge the authenticated employee's absence and exceptional-entry "
            "request statuses for an inclusive date range."
        ),
        "parameters": _date_range_schema(),
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_pending_approvals",
        "description": (
            "Get pending supervisor approvals for the signed-in manager identity."
        ),
        "parameters": _empty_schema(),
        "strict": True,
    },
    {
        "type": "function",
        "name": "get_notifications",
        "description": (
            "Get the authenticated employee's ResourcePlus notifications for unread, "
            "latest, HR-alert, and approval-notification questions."
        ),
        "parameters": _empty_schema(),
        "strict": True,
    },
    {
        "type": "function",
        "name": "prepare_exceptional_entry",
        "description": (
            "Prepare, but do not submit, a less-hours exceptional-entry correction. "
            "Use only for an explicit forgotten/missing punch or a request to correct "
            "an existing attendance record. Never use for an expected current/future "
            "late arrival or an already-completed late IN punch, even when the employee "
            "mentions punching in late. "
            "Call this directly for a correction request; do not prefetch suggestions "
            "or reasons. The backend reads suggestions once, resolves one exact punch, "
            "then reads live reasons only when an actionable suggestion exists. Call "
            "with a null reason when the employee has not supplied one; never ask for "
            "a reason before this tool verifies actionability. Provide IN or OUT only "
            "when the employee selected that direction after a clarification."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_date": {
                    "type": ["string", "null"],
                    "description": (
                        "Selected attendance date as YYYY-MM-DD, or null when the "
                        "employee has not selected one."
                    ),
                },
                "punch_direction": {
                    "type": ["string", "null"],
                    "enum": ["IN", "OUT", None],
                    "description": (
                        "IN or OUT only when the employee explicitly selected or stated "
                        "that missing direction; otherwise null. Preserve explicit "
                        "punch-in/punch-out wording exactly. The backend re-grounds this "
                        "from the user message. This never supplies the correction time."
                    ),
                },
                "reason_name": {
                    "type": ["string", "null"],
                    "description": (
                        "The employee's exact natural-language reason from this turn, "
                        "or null when none was supplied. Never infer a default and do "
                        "not supply a ResourcePlus reason ID."
                    ),
                },
                "remarks": {
                    "type": ["string", "null"],
                    "description": (
                        "The employee's own requested remarks, or null when none were "
                        "supplied; do not invent facts."
                    ),
                },
            },
            "required": [
                "target_date",
                "punch_direction",
                "reason_name",
                "remarks",
            ],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "prepare_book_day_type",
        "description": (
            "Prepare, but do not submit, leave or business travel. Use an exact day "
            "type name returned by get_day_types."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                "date_to": {"type": "string", "description": "YYYY-MM-DD"},
                "day_type_name": {"type": "string"},
            },
            "required": ["date_from", "date_to", "day_type_name"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "prepare_less_hours_correction",
        "description": (
            "Prepare, but do not submit, a ResourcePlus FromSummary less-hours "
            "correction. Do not request a punch time or direction. entry_type is "
            "allowed only when the employee explicitly asks for late IN only (1) "
            "or early OUT only (2); minutes is allowed only when the employee "
            "explicitly requests a partial number of minutes."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target_date": {"type": "string", "description": "YYYY-MM-DD"},
                "reason_name": {"type": ["string", "null"]},
                "remarks": {"type": ["string", "null"]},
                "entry_type": {"type": ["integer", "null"], "enum": [1, 2, None]},
                "minutes": {"type": ["integer", "null"], "minimum": 1},
            },
            "required": ["target_date", "reason_name", "remarks", "entry_type", "minutes"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "prepare_cancel_exceptional_entry",
        "description": (
            "Find and prepare cancellation of one ResourcePlus-reported pending, "
            "cancellable exceptional entry. Never accept or invent an exceptional ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                "date_to": {"type": "string", "description": "YYYY-MM-DD"},
                "target_date": {"type": ["string", "null"], "description": "YYYY-MM-DD"},
            },
            "required": ["date_from", "date_to", "target_date"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "prepare_cancel_day_type_request",
        "description": (
            "Find and prepare cancellation of one real pending absence request. Never "
            "accept or invent a mapping ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "day_type_name": {"type": "string"},
                "date_from": {"type": "string", "description": "YYYY-MM-DD"},
                "date_to": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "YYYY-MM-DD, or null when only one date was given.",
                },
            },
            "required": ["day_type_name", "date_from", "date_to"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "prepare_supervisor_request",
        "description": (
            "Find one real pending supervisor request and prepare approval or rejection. "
            "Never accept or invent request IDs or request types."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "employee_name": {"type": "string"},
                "detail": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                },
                "decision": {"type": "string", "enum": ["approve", "reject"]},
            },
            "required": ["employee_name", "detail", "decision"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "prepare_approve_all_requests",
        "description": (
            "Count current pending supervisor requests and prepare a confirmed bulk "
            "approve or reject action."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "decision": {"type": "string", "enum": ["approve", "reject"]},
                "request_type": {
                    "type": "string",
                    "enum": ["all", "Absence", "ExceptionEntry"],
                },
            },
            "required": ["decision", "request_type"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "type": "function",
        "name": "prepare_notification_read_status",
        "description": (
            "Prepare, but do not execute, marking one real ResourcePlus notification "
            "or all notifications read or unread. Never accept or invent notifcnID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "target": {
                    "type": "string",
                    "enum": ["latest", "one", "all"],
                },
                "notification_title": {
                    "anyOf": [{"type": "string"}, {"type": "null"}],
                    "description": "Exact title for target=one; otherwise null.",
                },
                "read_status": {
                    "type": "integer",
                    "enum": [0, 1],
                    "description": "0 for unread; 1 for read.",
                },
            },
            "required": ["target", "notification_title", "read_status"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]

READ_TOOL_NAMES = {
    "get_attendance_summary",
    "get_home_data",
    "get_profile_data",
    "get_missing_punch_suggestions",
    "get_exception_reasons",
    "get_exceptional_entry_balance",
    "get_day_types",
    "get_my_request_status",
    "get_pending_approvals",
    "get_notifications",
}
WRITE_INTENT_TOOL_NAMES = {
    "prepare_exceptional_entry",
    "prepare_less_hours_correction",
    "prepare_cancel_exceptional_entry",
    "prepare_book_day_type",
    "prepare_cancel_day_type_request",
    "prepare_supervisor_request",
    "prepare_approve_all_requests",
    "prepare_notification_read_status",
}
ALLOWED_TOOL_NAMES = READ_TOOL_NAMES | WRITE_INTENT_TOOL_NAMES


@dataclass(frozen=True)
class DateRange:
    label: str
    from_date: date
    to_date: date


@dataclass(frozen=True)
class ToolExecutionResult:
    output: str
    failed: bool = False
    tool_used: bool = True
    pending_action: PendingAction | None = None
    terminal_message: str | None = None
    needs_reason: bool = False
    reason_options: list[str] | None = None
    blocks: list[ResponseBlock] | None = None


def _write_intent_is_grounded(tool_name: str, message: str) -> bool:
    """Require explicit user language before any write preparation can begin."""

    normalized = " ".join(message.casefold().replace("’", "'").split())
    if not normalized:
        return False

    if tool_name == "prepare_exceptional_entry":
        if is_explicit_missing_punch_correction(message):
            return True
        return bool(
            re.search(
                r"\b(?:correct|fix|adjust|amend|regulari[sz]e)\b.{0,40}"
                r"\b(?:less\s+hours?|shortfall|attendance)\b",
                normalized,
            )
        )

    if tool_name == "prepare_less_hours_correction":
        return bool(
            re.search(
                r"\b(?:correct|fix|adjust|amend|regulari[sz]e)\b.{0,50}"
                r"\b(?:less[-\s]+hours?|short\s+hours?|shortfall|late\s+arrival|early\s+departure|entry)\b",
                normalized,
            )
        ) or bool(
            re.search(
                r"\b(?:use|apply)\b.{0,35}\b(?:buffer|excuse\s+time)\b",
                normalized,
            )
        ) or (
            any(cue in normalized for cue in ("صحح", "تصحيح", "عندي"))
            and any(cue in normalized for cue in ("ساعات ناقصة", "نقص ساعات", "تأخر", "خروج مبكر"))
        )

    if tool_name == "prepare_cancel_exceptional_entry":
        return (
            bool(re.search(r"\b(?:cancel|withdraw)\b", normalized))
            and bool(re.search(r"\b(?:exception|exceptional\s+entry)\b", normalized))
        ) or (
            any(cue in normalized for cue in ("إلغاء", "الغاء", "اسحب", "سحب"))
            and any(cue in normalized for cue in ("استثنائي", "استثناء"))
        )

    if tool_name == "prepare_book_day_type":
        return bool(
            re.search(r"\b(?:apply|request|book|take)\b", normalized)
            and re.search(
                r"\b(?:leave|vacation|day\s+off|absence|business\s+travel)\b",
                normalized,
            )
        ) or (
            any(cue in normalized for cue in ("أبغى", "ابغى", "اريد", "أريد", "قدم", "تقديم"))
            and any(cue in normalized for cue in ("إجاز", "اجاز", "سفر", "غياب"))
        )

    if tool_name == "prepare_cancel_day_type_request":
        return (
            bool(re.search(r"\b(?:cancel|withdraw)\b", normalized))
            and bool(re.search(r"\b(?:leave|vacation|absence|request|travel)\b", normalized))
        ) or (
            any(cue in normalized for cue in ("إلغاء", "الغاء", "اسحب", "سحب"))
            and any(cue in normalized for cue in ("إجاز", "اجاز", "طلب", "سفر", "غياب"))
        )

    if tool_name in {"prepare_supervisor_request", "prepare_approve_all_requests"}:
        return bool(re.search(r"\b(?:approve|reject)\b", normalized)) or any(
            cue in normalized for cue in ("وافق", "موافقة", "ارفض", "رفض")
        )

    if tool_name == "prepare_notification_read_status":
        return (
            bool(re.search(r"\b(?:mark|read|unread)\b", normalized))
            and bool(re.search(r"\b(?:notification|alert)s?\b", normalized))
        ) or (
            any(cue in normalized for cue in ("مقروء", "غير مقروء", "اقرأ"))
            and any(cue in normalized for cue in ("إشعار", "اشعار", "تنبيه"))
        )

    return False


def _ungrounded_write_message(tool_name: str, language: str) -> str:
    if tool_name in {"prepare_exceptional_entry", "prepare_less_hours_correction"}:
        if language == "ar":
            return "ما فهمت طلب تصحيح البصمة بوضوح. هل نسيت تسجيل الدخول أو الخروج؟"
        return (
            "I didn't catch that clearly. Did you mean you forgot to punch in "
            "or punch out?"
        )
    if language == "ar":
        return "ما فهمت بوضوح الإجراء الذي تريد تنفيذه. فضلاً أعد الطلب بشكل أوضح."
    return "I didn't catch a clear request to change anything. Please repeat your request."


def _grounded_less_hours_options(message: str) -> tuple[int | None, int | None]:
    """Derive optional v2 write scope from employee words, never model arguments."""

    normalized = " ".join(message.casefold().split())
    entry_type: int | None = None
    if re.search(r"\b(?:only\s+)?late\s+(?:in|arrival)\b", normalized) or any(
        cue in normalized for cue in ("تأخر الدخول", "التأخر في الدخول", "الدخول المتأخر")
    ):
        entry_type = 1
    elif re.search(r"\b(?:only\s+)?early\s+(?:out|departure)\b", normalized) or any(
        cue in normalized for cue in ("خروج مبكر", "الخروج المبكر", "انصراف مبكر")
    ):
        entry_type = 2
    minute_match = re.search(
        r"\b(?:only\s+)?(\d{1,3})\s*(?:minutes?|mins?)\b",
        normalized,
    ) or re.search(r"(?:فقط\s*)?(\d{1,3})\s*د(?:قيقة|قائق)", normalized)
    minutes = int(minute_match.group(1)) if minute_match else None
    if minutes is not None and minutes <= 0:
        minutes = None
    return entry_type, minutes


def resolve_relative_date_range(
    message: str,
    *,
    today: date | None = None,
) -> DateRange | None:
    """Resolve supported relative periods without relying on the model."""

    current = today or resourceplus_today()
    normalized = message.casefold()
    patterns: list[tuple[str, tuple[str, ...]]] = [
        ("last_week", (r"\blast week\b", "الأسبوع الماضي", "الاسبوع الماضي")),
        ("previous_month", (r"\bprevious month\b", r"\blast month\b")),
        ("this_week", (r"\bthis week\b", "هذا الأسبوع", "هذا الاسبوع", "الأسبوع الحالي", "الاسبوع الحالي")),
        ("this_month", (r"\bthis month\b", "هذا الشهر", "الشهر الحالي")),
        ("yesterday", (r"\byesterday\b", "أمس", "امس")),
        ("today", (r"\btoday\b", "اليوم")),
    ]
    selected: str | None = None
    for label, variants in patterns:
        if any(
            re.search(variant, normalized) if variant.startswith(r"\b") else variant in normalized
            for variant in variants
        ):
            selected = label
            break

    if selected == "today":
        return DateRange(selected, current, current)
    if selected == "yesterday":
        day = current - timedelta(days=1)
        return DateRange(selected, day, day)
    if selected == "this_week":
        start = current - timedelta(days=current.weekday())
        return DateRange(selected, start, current)
    if selected == "last_week":
        this_week_start = current - timedelta(days=current.weekday())
        start = this_week_start - timedelta(days=7)
        return DateRange(selected, start, this_week_start - timedelta(days=1))
    if selected == "previous_month":
        current_month_start = current.replace(day=1)
        end = current_month_start - timedelta(days=1)
        return DateRange(selected, end.replace(day=1), end)
    if selected == "this_month":
        return DateRange(selected, current.replace(day=1), current)
    return None


def date_context(message: str, *, today: date | None = None) -> tuple[str, DateRange | None]:
    current = today or resourceplus_today()
    resolved = resolve_relative_date_range(message, today=current)
    context = f"Backend date: {current.isoformat()}."
    if resolved:
        calendar_month_end = date(
            resolved.from_date.year,
            resolved.from_date.month,
            monthrange(resolved.from_date.year, resolved.from_date.month)[1],
        )
        context += (
            f" The backend resolved '{resolved.label}' to the inclusive range "
            f"{resolved.from_date.isoformat()} through {resolved.to_date.isoformat()}. "
            "Use these dates for attendance and other past-to-present queries."
        )
        if resolved.label == "this_month":
            context += (
                " For get_my_request_status only, use the complete calendar month "
                f"through {calendar_month_end.isoformat()}."
            )
    return context, resolved


def _tool_dates(
    tool_name: str,
    arguments: dict[str, Any],
    resolved_range: DateRange | None,
) -> tuple[object, object]:
    if resolved_range:
        end = resolved_range.to_date
        if tool_name == "get_my_request_status" and resolved_range.label == "this_month":
            end = date(
                resolved_range.from_date.year,
                resolved_range.from_date.month,
                monthrange(
                    resolved_range.from_date.year,
                    resolved_range.from_date.month,
                )[1],
            )
        return resolved_range.from_date.isoformat(), end.isoformat()
    return arguments.get("from_date"), arguments.get("to_date")


def _without_notification_query_strings(data: Any) -> Any:
    if not isinstance(data, list):
        return data
    return [
        {key: value for key, value in item.items() if key != "QueryString"}
        if isinstance(item, dict)
        else item
        for item in data
    ]


async def execute_tool(
    name: str,
    arguments: dict[str, Any],
    *,
    lang: int,
    session_id: str,
    response_language: str,
    resolved_range: DateRange | None = None,
    source_user_message: str | None = None,
    trusted_conversation_intent: bool = False,
    store: SessionStore = session_store,
) -> ToolExecutionResult:
    if name not in ALLOWED_TOOL_NAMES:
        return ToolExecutionResult(
            json.dumps({"success": False, "error": "Unsupported tool."}),
            failed=True,
        )

    existing_late_punch = (
        source_user_message is not None
        and is_existing_late_punch(source_user_message)
    )
    if (
        name in {"get_missing_punch_suggestions", "prepare_exceptional_entry"}
        and source_user_message is not None
        and (
            existing_late_punch
            or is_prospective_late_arrival(source_user_message)
        )
    ):
        message = late_arrival_unavailable_message(
            response_language,
            already_punched=existing_late_punch,
        )
        return ToolExecutionResult(
            json.dumps(
                {
                    "success": True,
                    "prepared": False,
                    "requires_confirmation": False,
                    "message": message,
                },
                ensure_ascii=False,
            ),
            terminal_message=message,
        )

    if name in WRITE_INTENT_TOOL_NAMES and source_user_message is not None:
        existing_draft = (
            store.get_exceptional_entry_draft(session_id)
            if name == "prepare_exceptional_entry"
            else None
        )
        if existing_draft is None and not trusted_conversation_intent and not _write_intent_is_grounded(
            name,
            source_user_message,
        ):
            message = _ungrounded_write_message(name, response_language)
            return ToolExecutionResult(
                json.dumps(
                    {
                        "success": True,
                        "prepared": False,
                        "requires_confirmation": False,
                        "message": message,
                    },
                    ensure_ascii=False,
                ),
                tool_used=False,
                terminal_message=message,
            )

    record_tool_usage(name)
    try:
        if name == "get_attendance_summary":
            start, end = _tool_dates(name, arguments, resolved_range)
            data = await get_attendance_summary(start, end, lang=lang)
        elif name == "get_home_data":
            data = await get_home_data(lang=lang)
        elif name == "get_profile_data":
            data = await get_profile_data(lang=lang)
        elif name == "get_missing_punch_suggestions":
            start, end = _tool_dates(name, arguments, resolved_range)
            data = missing_punch_tool_data(
                normalize_missing_punch_suggestions(
                    await get_missing_punch_suggestions(start, end, lang=lang)
                )
            )
        elif name == "get_exception_reasons":
            data = await cached_exception_reasons(lang)
        elif name == "get_exceptional_entry_balance":
            data = await get_exceptional_entry_balance(arguments.get("target_date"))
        elif name == "get_day_types":
            data = await cached_day_types(lang)
        elif name == "get_my_request_status":
            start, end = _tool_dates(name, arguments, resolved_range)
            data = await get_my_request_status(start, end, lang=lang)
        elif name == "get_pending_approvals":
            data = await get_pending_approvals(lang=lang)
        elif name == "get_notifications":
            data = _without_notification_query_strings(
                await get_notifications(lang=lang)
            )
        else:
            intent_arguments = dict(arguments)
            if name == "prepare_exceptional_entry" and source_user_message is not None:
                intent_arguments["_user_message"] = source_user_message
                if is_explicit_missing_punch_correction(source_user_message):
                    # The original user wording is authoritative. Override both a
                    # conflicting model direction and an invented direction for an
                    # otherwise ambiguous correction request.
                    intent_arguments["punch_direction"] = (
                        explicit_missing_punch_direction(source_user_message)
                    )
            if name == "prepare_less_hours_correction" and source_user_message is not None:
                intent_arguments["_user_message"] = source_user_message
                if not trusted_conversation_intent:
                    grounded_entry_type, grounded_minutes = _grounded_less_hours_options(
                        source_user_message
                    )
                    intent_arguments["entry_type"] = grounded_entry_type
                    intent_arguments["minutes"] = grounded_minutes
            if resolved_range and name == "prepare_exceptional_entry":
                intent_arguments["_range_from"] = resolved_range.from_date.isoformat()
                intent_arguments["_range_to"] = resolved_range.to_date.isoformat()
                if (
                    resolved_range.from_date == resolved_range.to_date
                    and not intent_arguments.get("target_date")
                ):
                    intent_arguments["target_date"] = resolved_range.from_date.isoformat()
            if (
                resolved_range
                and name == "prepare_less_hours_correction"
                and resolved_range.from_date == resolved_range.to_date
            ):
                intent_arguments["target_date"] = resolved_range.from_date.isoformat()
            if resolved_range and name in {
                "prepare_book_day_type",
                "prepare_cancel_day_type_request",
                "prepare_cancel_exceptional_entry",
            }:
                intent_arguments["date_from"] = resolved_range.from_date.isoformat()
                intent_arguments["date_to"] = resolved_range.to_date.isoformat()
            intent = await prepare_write_action(
                name,
                intent_arguments,
                lang=lang,
                response_language=response_language,
            )
            pending = store.create_pending_action(
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
            return ToolExecutionResult(
                json.dumps(
                    {
                        "success": True,
                        "requires_confirmation": True,
                        "confirmation_id": pending.confirmation_id,
                        "summary": pending.summary,
                    },
                    ensure_ascii=False,
                ),
                pending_action=pending,
            )
    except ActionResolutionRequired as exc:
        logger.info(
            "EXCEPTIONAL_ENTRY_RESOLUTION category=%s",
            exc.category,
        )
        if exc.category == "no_resourceplus_suggestion":
            record_safe_error(exc.category)
        if (
            name == "prepare_exceptional_entry"
            and exc.category == "reason_required"
            and exc.draft_context is not None
        ):
            store.create_exceptional_entry_draft(
                session_id,
                attendance_date=exc.draft_context["attendance_date"],
                entry_type=exc.draft_context["entry_type"],
                suggested_entry_time=exc.draft_context["suggested_entry_time"],
                language=exc.draft_context["language"],
                reason_options=exc.reason_options,
            )
        needs_reason = exc.category in {
            "reason_required",
            "reason_unknown",
            "reason_ambiguous",
        }
        reason_options = [
            {"label": name, "value": name}
            for name in (exc.reason_options or [])
        ]
        return ToolExecutionResult(
            json.dumps(
                {
                    "success": True,
                    "prepared": False,
                    "requires_confirmation": False,
                    "requires_clarification": exc.requires_clarification,
                    "needs_reason": needs_reason,
                    "reason_options": reason_options,
                    "message": str(exc),
                },
                ensure_ascii=False,
            ),
            terminal_message=str(exc),
            needs_reason=needs_reason,
            reason_options=list(exc.reason_options or []),
        )
    except ResourcePlusError as exc:
        record_safe_error("resourceplus_error")
        return ToolExecutionResult(
            json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False),
            failed=True,
        )
    except (ValueError, TypeError, KeyError) as exc:
        record_safe_error("validation_error")
        return ToolExecutionResult(
            json.dumps({"success": False, "error": str(exc)}, ensure_ascii=False),
            failed=True,
        )

    payload: dict[str, Any] = {"success": True, "data": data}
    if name == "get_attendance_summary":
        payload["attendance_semantics"] = {
            "NetHrs": "time actually worked",
            "LessHrs": "shortfall from required working hours",
            "speech_rule": (
                "State actual worked time and the shortfall as separate quantities; "
                "never describe NetHrs as the amount worked less than expected."
            ),
        }
    return ToolExecutionResult(
        json.dumps(payload, ensure_ascii=False, default=str),
        blocks=attendance_blocks(data) if name == "get_attendance_summary" else None,
    )
