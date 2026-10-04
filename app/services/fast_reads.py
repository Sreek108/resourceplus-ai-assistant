from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Awaitable, Callable

from app.ai.actions import classify_attendance_summary
from app.ai.sessions import ApprovalCandidate, TrustedResultContext
from app.ai.tools import resolve_relative_date_range
from app.audit import record_tool_usage
from app.models.schemas import ResponseBlock, TableBlock
from app.resourceplus.approvals import get_pending_approvals
from app.resourceplus.attendance import (
    get_attendance_summary,
    get_missing_punch_suggestions,
)
from app.resourceplus.employee import get_profile_data
from app.resourceplus.exceptional import get_exceptional_entry_balance
from app.resourceplus.leave import get_my_day_type_requests
from app.resourceplus.missing_punch import (
    apply_attendance_eligibility,
    missing_punch_tool_data,
    normalize_missing_punch_suggestions,
)
from app.resourceplus.notifications import get_notifications
from app.resourceplus.reference_cache import cached_day_types
from app.resourceplus.requests import (
    get_exceptional_entry_requests,
    get_my_request_status,
)
from app.services.response_blocks import (
    approvals_block,
    attendance_blocks,
    day_types_block,
    exceptional_balance_blocks,
    exceptional_entries_block,
    missing_punch_block,
    notifications_block,
    profile_block,
    request_history_block,
)
from app.services.approval_selection import approval_candidates as resolve_approval_candidates
from app.services.request_history import (
    filter_request_history,
    normalize_request_history,
    parse_request_history_query,
    request_history_message,
    request_history_range,
)
from app.time_context import resourceplus_today


@dataclass(frozen=True)
class FastReadResult:
    message: str
    speech_message: str
    tools_used: list[str]
    blocks: list[ResponseBlock] = field(default_factory=list)
    period: tuple[str, str] | None = None
    approval_candidates: tuple[ApprovalCandidate, ...] = ()
    request_rows: tuple[dict[str, object], ...] = ()
    recent_request_category: str | None = None
    recent_request_date: str | None = None
    recent_request_detail: str | None = None
    recent_request_state: str | None = None


_CORRECTION = re.compile(r"\b(?:fix|correct|regulari[sz]e|change|submit)\b", re.I)
_PROFILE = re.compile(r"\b(?:my\s+)?profile\b|\bmy\s+(?:employee\s+)?details\b", re.I)
_ATTENDANCE = re.compile(r"\battendance\b|\bpresence\b", re.I)
_MISSING = re.compile(r"\b(?:missing|missed)\s+(?:punch(?:es)?|check[- ]?(?:in|out)s?)\b", re.I)
_NOTIFICATIONS = re.compile(r"\b(?:notifications?|alerts?)\b", re.I)
_DAY_TYPES = re.compile(r"\b(?:leave|day)\s+types?\b|\bavailable\s+leaves?\b", re.I)
_REQUESTS = re.compile(
    r"\b(?:my\s+requests?|request\s+status|"
    r"(?:my\s+)?(?:leave|vacation|business\s+travel)\s+requests?)\b",
    re.I,
)
_EXCEPTIONAL_ENTRIES = re.compile(
    r"\b(?:show|list|display|view|what|which|do\s+i\s+have)\b.{0,80}"
    r"\b(?:exceptional|exception)\s+"
    r"(?:entries?|entry\s+requests?|requests?|energies)\b",
    re.I,
)
_APPROVALS = re.compile(r"\b(?:pending\s+)?approvals?\b", re.I)
_BALANCE = re.compile(
    r"\b(?:buffer(?:\s+(?:time|minutes?|balance))?|excuse\s+minutes?|"
    r"(?:attendance|exceptional[-\s]+entry)\s+allowance|"
    r"allowance\s+(?:balance|(?:is\s+)?left))\b",
    re.I,
)
_BALANCE_STT_VARIANT = re.compile(
    r"\b(?:how\s+much|how\s+many|what(?:'s|\s+is)|show|check)\b.{0,45}"
    r"\b(?:cover\s+time|allowance\s+(?:time|minutes?))\b"
    r"|\b(?:cover\s+time|allowance\s+(?:time|minutes?))\b.{0,35}"
    r"\b(?:do\s+i\s+have|remaining|left)\b",
    re.I,
)
_ARABIC_BALANCE_CUES = (
    "رصيد البفر",
    "وقت البفر",
    "دقائق البفر",
    "دقائق الاستئذان",
    "دقائق استئذان",
    "رصيد الاستئذان",
    "بدل الحضور",
    "رصيد الاستثناء",
    "دقائق السماح",
    "وقت السماح",
    "رصيد السماح",
    "الرصيد المتبقي",
    "دقائق التعويض",
    "رصيد التعويض",
)

_ARABIC_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u0640]")


def _arabic_normalized(message: str) -> str:
    normalized = _ARABIC_DIACRITICS.sub("", message.casefold())
    normalized = normalized.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}))
    return " ".join(re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE).split())


def _is_arabic_balance_read(message: str) -> bool:
    normalized = _arabic_normalized(message)
    if any(_arabic_normalized(cue) in normalized for cue in _ARABIC_BALANCE_CUES):
        return True
    allowance = any(cue in normalized for cue in ("السماح", "التعويض", "الاستئذان", "البفر"))
    amount = any(cue in normalized for cue in ("دقائق", "وقت", "رصيد", "متبقي", "بقي", "عندي"))
    return allowance and amount


def _is_arabic_exceptional_read(message: str) -> bool:
    normalized = _arabic_normalized(message)
    read_cue = any(
        cue in normalized
        for cue in ("اعرض", "اظهر", "ارني", "ورني", "ما هي", "ماهي", "وش", "ايش")
    )
    topic = any(
        cue in normalized
        for cue in (
            "طلبات الاستثناء",
            "ادخالات الاستثناء",
            "الاستثناءات",
            "استثناءاتي",
            "طلبات الحضور الاستثنائي",
            "طلبات الحضور الاستثنائية",
        )
    )
    return read_cue and topic


def is_noisy_balance_reference(message: str) -> bool:
    """Recognize only a small, high-signal family of degraded buffer STT text."""

    tokens = re.findall(r"[a-z]+", message.casefold())
    return (
        2 <= len(tokens) <= 4
        and "buffet" in tokens
        and "time" in tokens
        and set(tokens) <= {"buffet", "time"}
    )


def classify_fast_read(message: str) -> list[str]:
    """Return only high-confidence, side-effect-free read intents."""

    if _CORRECTION.search(message):
        return []
    if (
        _BALANCE.search(message)
        or _BALANCE_STT_VARIANT.search(message)
        or _is_arabic_balance_read(message)
    ):
        return ["balance"]
    if parse_request_history_query(message) is not None:
        return ["requests"]
    if _EXCEPTIONAL_ENTRIES.search(message) or _is_arabic_exceptional_read(message):
        return ["exceptional_entries"]
    intents: list[str] = []
    if _PROFILE.search(message):
        intents.append("profile")
    if _MISSING.search(message):
        intents.append("missing_punches")
    if _ATTENDANCE.search(message):
        intents.append("attendance")
    if _NOTIFICATIONS.search(message):
        intents.append("notifications")
    if _DAY_TYPES.search(message):
        intents.append("day_types")
    if _REQUESTS.search(message):
        intents.append("requests")
    if _APPROVALS.search(message):
        intents.append("approvals")
    normalized = _arabic_normalized(message)
    if any(
        cue in normalized
        for cue in (
            "الموافقات المعلقة",
            "طلبات بانتظار موافقتي",
            "طلبات تنتظر موافقتي",
        )
    ):
        intents.append("approvals")
    if any(
        cue in normalized
        for cue in (
            "اعرض طلباتي",
            "اظهر طلباتي",
            "ورني طلباتي",
            "اعرض طلبي",
            "حالة طلبي",
        )
    ):
        intents.append("requests")
    return list(dict.fromkeys(intents))


def _date_range(message: str) -> tuple[str, str]:
    today = resourceplus_today()
    explicit = re.search(r"\b(20\d{2}-\d{2}-\d{2})\b", message)
    if explicit is not None:
        try:
            selected = date.fromisoformat(explicit.group(1)).isoformat()
            return selected, selected
        except ValueError:
            pass
    resolved = resolve_relative_date_range(message, today=today)
    if resolved is None:
        return today.replace(day=1).isoformat(), today.isoformat()
    return resolved.from_date.isoformat(), resolved.to_date.isoformat()


def _balance_date(message: str) -> str:
    today = resourceplus_today()
    resolved = resolve_relative_date_range(message, today=today)
    return (resolved.to_date if resolved is not None else today).isoformat()


def _display_number(value: Any) -> str:
    """Remove display-only decimal padding without changing the numeric value."""

    text = str(value)
    try:
        number = Decimal(text)
    except (InvalidOperation, ValueError):
        return text
    if not number.is_finite():
        return text
    if number == number.to_integral_value():
        return format(number.quantize(Decimal("1")), "f")
    return format(number.normalize(), "f")


def _english_count(count: int, singular: str, plural: str, period: str = "") -> str:
    if count == 0:
        return f"You don't have any {plural}{period}."
    amount = str(count)
    noun = singular if count == 1 else plural
    return f"You have {amount} {noun}{period}."


def _balance_message(payload: Any, language: str) -> tuple[str, str]:
    if not isinstance(payload, dict):
        message = (
            "تعذر قراءة بدل إدخال الحضور الاستثنائي من ResourcePlus."
            if language == "ar"
            else "I couldn't get your attendance allowance details right now."
        )
        return message, message
    if payload.get("hasPolicy") is False:
        message = (
            "لا توجد لديك سياسة رصيد سماح لهذا التاريخ."
            if language == "ar"
            else "You do not have an allowance policy for that date."
        )
        return message, message
    remaining = payload.get("remaining")
    limit_type = payload.get("limitType")
    if remaining not in (None, ""):
        displayed = _display_number(remaining)
        current_period = False
        period_start = payload.get("periodStart")
        period_end = payload.get("periodEnd")
        if isinstance(period_start, str) and isinstance(period_end, str):
            try:
                start_date = date.fromisoformat(period_start)
                end_date = date.fromisoformat(period_end)
                current_period = (
                    start_date.weekday() == 0
                    and (end_date - start_date).days == 6
                    and start_date <= resourceplus_today() <= end_date
                )
            except ValueError:
                pass
        if language == "ar":
            if limit_type == 2:
                period = " هالأسبوع" if current_period else ""
                message = f"باقي لك {displayed} دقيقة من وقت السماح{period}."
            elif limit_type == 1:
                message = f"لديك {displayed} حالة متبقية."
            else:
                message = f"لديك {displayed} متبقيًا."
        elif limit_type == 1:
            noun = "attendance correction" if displayed == "1" else "attendance corrections"
            message = f"You have {displayed} {noun} remaining."
        elif limit_type == 2:
            noun = "minute" if displayed == "1" else "minutes"
            period = " this week" if current_period else ""
            message = f"You have {displayed} {noun} of buffer time left{period}."
        else:
            message = f"You have {displayed} remaining."
        return message, message
    message = (
        "هذه تفاصيل رصيد السماح الخاص بك."
        if language == "ar"
        else "Here are your attendance correction allowance details."
    )
    return message, message


def _rows(blocks: list[ResponseBlock]) -> int:
    return sum(len(block.rows) for block in blocks if isinstance(block, TableBlock))


def _payload_rows(payload: Any, *keys: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        folded = {str(key).casefold(): value for key, value in payload.items()}
        for key in keys:
            value = folded.get(key.casefold())
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def _first_value(row: dict[str, Any], *keys: str) -> str:
    folded = {str(key).casefold(): value for key, value in row.items()}
    for key in keys:
        value = folded.get(key.casefold())
        if value not in (None, ""):
            return str(value)
    return ""


def _missing_punch_message(table: TableBlock | None, language: str) -> str:
    rows = table.rows if table is not None else []
    total = len(rows)
    if total == 0:
        return (
            "لا توجد لديك بصمات مفقودة خلال هذه الفترة."
            if language == "ar" else
            "You don't have any missing punches for this period."
        )
    correctable = sum(row.get("correctable") is True for row in rows)
    fully_classified = all(isinstance(row.get("correctable"), bool) for row in rows)
    if language == "ar":
        record = "سجل بصمة مفقودة واحد" if total == 1 else f"{total} سجلات لبصمات مفقودة"
        if not fully_classified:
            return f"وجدت {record} خلال هذه الفترة."
        if correctable == 0:
            return f"وجدت {record} خلال هذه الفترة، لكن لا توجد أي بصمة متاحة للتصحيح حاليًا."
        if correctable == total:
            available = "بصمة مفقودة واحدة" if total == 1 else f"{total} بصمات مفقودة"
            return f"لديك {available} متاحة للتصحيح."
        return f"وجدت {record} خلال هذه الفترة. {correctable} منها متاحة للتصحيح حاليًا."
    record = "record" if total == 1 else "records"
    if not fully_classified:
        return f"I found {total} missing-punch {record} for this period."
    if correctable == 0:
        return (
            f"I found {total} missing-punch {record} for this period, "
            "but none are currently eligible for correction."
        )
    if correctable == total:
        punch = "punch" if total == 1 else "punches"
        return f"You have {total} missing {punch} available for correction."
    return (
        f"You have {total} missing-punch {record} for this period. "
        f"{correctable} {'is' if correctable == 1 else 'are'} currently available for correction."
    )


def _attendance_period_message(
    table: TableBlock | None,
    language: str,
    period: tuple[str, str],
) -> tuple[str, str]:
    start = date.fromisoformat(period[0])
    end = date.fromisoformat(period[1])
    today = resourceplus_today()
    rows = table.rows if table is not None else []
    row_count = len(rows)
    short_days = sum(
        str(row.get("shortfall", "")).strip() not in {"", "0", "00:00", "0:00", "—", "None"}
        for row in rows
    )
    if start == end:
        period_label = f"{start.strftime('%b')} {start.day}"
        lead = (
            f"هذا حضورك ليوم {period_label}."
            if language == "ar"
            else f"Here's your attendance for {period_label}."
        )
    elif start.day == 1 and start.year == end.year and start.month == end.month:
        month = start.strftime("%B")
        if language == "ar":
            lead = f"هذا سجل حضورك لشهر {month}{' حتى الآن' if end == today else ''}."
        else:
            lead = (
                f"Here's your attendance for {month} so far."
                if end == today
                else f"Here's your {month} attendance."
            )
    else:
        lead = (
            f"هذا سجل حضورك من {period[0]} إلى {period[1]}."
            if language == "ar"
            else f"Here's your attendance from {period[0]} through {period[1]}."
        )
    if not row_count:
        empty = (
            " ما عندك سجلات حضور خلال هذه الفترة."
            if language == "ar"
            else " You don't have any attendance records for that period."
        )
        return lead + empty, lead + empty
    day_noun = "day" if row_count == 1 else "days"
    summary = (
        f" يعرض الجدول {row_count} يوم، منها {short_days} بساعات ناقصة."
        if language == "ar" and short_days
        else f" The table shows {row_count} {day_noun}, including {short_days} with recorded short hours."
        if short_days
        else f" يعرض الجدول {row_count} يوم."
        if language == "ar"
        else f" The table shows {row_count} {day_noun}."
    )
    return lead + summary, lead


def _message_for(
    intent: str,
    blocks: list[ResponseBlock],
    language: str,
    *,
    period: tuple[str, str] | None = None,
) -> tuple[str, str]:
    row_count = _rows(blocks)
    item_count = len(getattr(blocks[0], "items", [])) if blocks else 0
    table = next((block for block in blocks if isinstance(block, TableBlock)), None)
    if intent == "attendance" and period is not None:
        return _attendance_period_message(table, language, period)
    if intent == "profile" and not blocks:
        message = (
            "لم أجد تفاصيل ملف وظيفي متاحة."
            if language == "ar"
            else "I couldn't find any available profile details."
        )
        return message, message
    if language == "ar":
        leads = {
            "profile": "إليك تفاصيل ملفك.",
            "attendance": (
                "إليك سجل حضورك خلال هذه الفترة."
                if row_count
                else "لا توجد لديك سجلات حضور خلال هذه الفترة."
            ),
            "missing_punches": _missing_punch_message(table, language),
            "notifications": (
                "لديك إشعار واحد."
                if item_count == 1
                else "لديك إشعاران."
                if item_count == 2
                else f"لديك {item_count} إشعارات."
                if item_count > 2
                else "لا توجد لديك إشعارات."
            ),
            "day_types": (
                f"لديك {row_count} من أنواع الأيام المتاحة."
                if row_count
                else "لا توجد أنواع أيام متاحة حاليًا."
            ),
            "requests": (
                "لديك طلب واحد خلال هذه الفترة."
                if row_count == 1
                else "لديك طلبان خلال هذه الفترة."
                if row_count == 2
                else f"لديك {row_count} طلبات خلال هذه الفترة."
                if row_count > 2
                else "لا توجد لديك طلبات خلال هذه الفترة."
            ),
            "approvals": (
                "لديك طلب واحد بانتظار موافقتك."
                if row_count == 1
                else "لديك طلبان بانتظار موافقتك."
                if row_count == 2
                else f"لديك {row_count} طلبات بانتظار موافقتك."
                if row_count > 2
                else "أمورك تمام — ما فيه طلبات تنتظر موافقتك."
            ),
        }
    else:
        leads = {
            "profile": "Here are your profile details.",
            "attendance": (
                "Here's your attendance."
                if row_count
                else "You don't have any attendance records for this period."
            ),
            "missing_punches": _missing_punch_message(table, language),
            "notifications": _english_count(item_count, "notification", "notifications"),
            "day_types": _english_count(
                row_count, "available day type", "available day types"
            ),
            "requests": _english_count(
                row_count, "request", "requests", " for this period"
            ),
            "approvals": (
                f"You have {row_count} request{'s' if row_count != 1 else ''} "
                "waiting for your approval."
                if row_count
                else "You're all caught up — there's nothing waiting for your approval."
            ),
        }
    if intent == "exceptional_entries":
        if language == "ar":
            lead = (
                "ما عندك طلبات تصحيح حضور خلال هالفترة."
                if row_count == 0
                else "عندك طلب تصحيح حضور واحد خلال هالفترة."
                if row_count == 1
                else "عندك طلبين تصحيح حضور خلال هالفترة."
                if row_count == 2
                else f"عندك {row_count} طلبات تصحيح حضور خلال هالفترة."
            )
        else:
            lead = _english_count(
                row_count,
                "attendance correction request",
                "attendance correction requests",
                " for this period",
            )
    else:
        lead = leads[intent]
    return lead, lead


async def try_fast_read(
    message: str,
    *,
    lang: int,
    response_language: str,
    trusted_context: TrustedResultContext | None = None,
) -> FastReadResult | None:
    intents = classify_fast_read(message)
    if not intents:
        return None
    request_query = parse_request_history_query(message) if "requests" in intents else None
    if request_query is not None:
        request_start, request_end, _ = request_history_range(
            message,
            today=resourceplus_today(),
        )
        start, end = request_start.isoformat(), request_end.isoformat()
    else:
        start, end = _date_range(message)
    attendance_task: asyncio.Task[Any] | None = None

    async def attendance_payload() -> Any:
        nonlocal attendance_task
        if attendance_task is None:
            attendance_task = asyncio.create_task(
                get_attendance_summary(start, end, lang=lang)
            )
        return await attendance_task

    async def guarded_missing_punches() -> dict[str, object]:
        suggestions, attendance = await asyncio.gather(
            get_missing_punch_suggestions(start, end, lang=lang),
            attendance_payload(),
        )
        inspection = classify_attendance_summary(
            attendance,
            date.fromisoformat(start),
            date.fromisoformat(end),
        )
        normalized = apply_attendance_eligibility(
            normalize_missing_punch_suggestions(suggestions),
            {day.attendance_date for day in inspection.eligible_days},
        )
        return missing_punch_tool_data(normalized)

    async def request_payload() -> Any:
        if request_query is not None and request_query.scope == "leave":
            return await get_my_day_type_requests(start, end, lang=lang)
        if request_query is not None and request_query.scope == "attendance_correction":
            return await get_exceptional_entry_requests(start, end, lang=lang)
        return await get_my_request_status(start, end, lang=lang)

    request_tool = (
        "get_my_day_type_requests"
        if request_query is not None and request_query.scope == "leave"
        else "get_exceptional_entries"
        if request_query is not None and request_query.scope == "attendance_correction"
        else "get_my_request_status"
    )

    loaders: dict[str, tuple[str, Callable[[], Awaitable[Any]]]] = {
        "balance": (
            "get_exceptional_entry_balance",
            lambda: get_exceptional_entry_balance(_balance_date(message)),
        ),
        "profile": ("get_profile_data", lambda: get_profile_data(lang=lang)),
        "attendance": (
            "get_attendance_summary",
            attendance_payload,
        ),
        "missing_punches": (
            "get_missing_punch_suggestions",
            guarded_missing_punches,
        ),
        "notifications": ("get_notifications", lambda: get_notifications(lang=lang)),
        "day_types": ("get_day_types", lambda: cached_day_types(lang)),
        "requests": (
            request_tool,
            request_payload,
        ),
        "exceptional_entries": (
            "get_exceptional_entries",
            lambda: get_exceptional_entry_requests(start, end, lang=lang),
        ),
        "approvals": ("get_pending_approvals", lambda: get_pending_approvals(lang=lang)),
    }
    selected = [loaders[intent] for intent in intents]
    for tool_name, _ in selected:
        record_tool_usage(tool_name)
    if "missing_punches" in intents and "attendance" not in intents:
        record_tool_usage("get_attendance_summary")
    payloads = await asyncio.gather(*(loader() for _, loader in selected))

    all_blocks: list[ResponseBlock] = []
    display_parts: list[str] = []
    speech_parts: list[str] = []
    approval_candidates: tuple[tuple[str, str, str, str], ...] = ()
    request_rows: tuple[dict[str, object], ...] = ()
    recent_request_category: str | None = None
    recent_request_date: str | None = None
    recent_request_detail: str | None = None
    recent_request_state: str | None = None
    for intent, payload in zip(intents, payloads, strict=True):
        if intent == "balance":
            blocks = exceptional_balance_blocks(payload, response_language)
            display, speech = _balance_message(payload, response_language)
            all_blocks.extend(blocks)
            display_parts.append(display)
            speech_parts.append(speech)
            continue
        if intent == "profile":
            block = profile_block(payload)
            blocks = [block] if block is not None else []
        elif intent == "attendance":
            blocks = attendance_blocks(payload)
        elif intent == "missing_punches":
            blocks = [missing_punch_block(payload)]
        elif intent == "notifications":
            safe_payload = [
                {key: value for key, value in row.items() if key != "QueryString"}
                for row in payload
                if isinstance(row, dict)
            ] if isinstance(payload, list) else payload
            blocks = [notifications_block(safe_payload)]
        elif intent == "day_types":
            blocks = [day_types_block(payload)]
        elif intent == "requests":
            assert request_query is not None
            all_requests = normalize_request_history(
                payload,
                source_hint=request_query.scope,
                correlations=(
                    trusted_context.request_correlations
                    if trusted_context is not None
                    else ()
                ),
            )
            all_requests = tuple(
                item for item in all_requests
                if not item.request_date or start <= item.request_date <= end
            )
            selected_requests = filter_request_history(
                all_requests,
                request_query,
                recent_category=(
                    trusted_context.recent_request_category
                    if trusted_context is not None
                    else None
                ),
                recent_date=(
                    trusted_context.recent_request_date
                    if trusted_context is not None
                    else None
                ),
            )
            blocks = [request_history_block(selected_requests, response_language)]
            request_rows = tuple(dict(row) for row in blocks[0].rows)
            if request_query.latest and len(selected_requests) == 1:
                selected_request = selected_requests[0]
                recent_request_category = selected_request.category
                recent_request_date = selected_request.request_date or None
                recent_request_detail = selected_request.detail
                recent_request_state = (
                    "approved"
                    if selected_request.resolved_status == "Approved"
                    else "rejected"
                    if selected_request.resolved_status == "Rejected"
                    else "submitted_for_approval"
                    if selected_request.resolved_status == "Pending"
                    else None
                )
            display, speech = request_history_message(
                selected_requests,
                request_query,
                language=response_language,
                unfiltered_requests=all_requests,
            )
            all_blocks.extend(blocks)
            display_parts.append(display)
            speech_parts.append(speech)
            continue
        elif intent == "exceptional_entries":
            blocks = [exceptional_entries_block(payload, language=response_language)]
        else:
            approval_candidates = resolve_approval_candidates(payload)
            blocks = [approvals_block(approval_candidates, response_language)]
        display, speech = _message_for(
            intent,
            blocks,
            response_language,
            period=(start, end) if intent == "attendance" else None,
        )
        all_blocks.extend(blocks)
        display_parts.append(display)
        speech_parts.append(speech)

    tools_used = [tool for tool, _ in selected]
    if "missing_punches" in intents and "get_attendance_summary" not in tools_used:
        tools_used.append("get_attendance_summary")
    return FastReadResult(
        message="\n\n".join(display_parts),
        speech_message=" ".join(speech_parts),
        tools_used=tools_used,
        blocks=all_blocks,
        period=(start, end),
        approval_candidates=approval_candidates,
        request_rows=request_rows,
        recent_request_category=recent_request_category,
        recent_request_date=recent_request_date,
        recent_request_detail=recent_request_detail,
        recent_request_state=recent_request_state,
    )
