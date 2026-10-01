from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import Any, Awaitable, Callable

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
from app.resourceplus.missing_punch import (
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
    markdown_table,
    missing_punch_block,
    notifications_block,
    profile_block,
    request_status_block,
)
from app.time_context import resourceplus_today


@dataclass(frozen=True)
class FastReadResult:
    message: str
    speech_message: str
    tools_used: list[str]
    blocks: list[ResponseBlock] = field(default_factory=list)


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
    return list(dict.fromkeys(intents))


def _date_range(message: str) -> tuple[str, str]:
    today = resourceplus_today()
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
    amount = "no" if count == 0 else str(count)
    noun = singular if count == 1 else plural
    return f"You have {amount} {noun}{period}."


def _balance_message(payload: Any, language: str) -> tuple[str, str]:
    if not isinstance(payload, dict):
        message = (
            "تعذر قراءة بدل إدخال الحضور الاستثنائي من ResourcePlus."
            if language == "ar"
            else "ResourcePlus did not return allowance details."
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
        if language == "ar":
            if limit_type == 2:
                message = f"لديك {displayed} دقيقة متبقية."
            elif limit_type == 1:
                message = f"لديك {displayed} حالة متبقية."
            else:
                message = f"لديك {displayed} متبقيًا."
        elif limit_type == 1:
            noun = "exceptional entry" if displayed == "1" else "exceptional entries"
            message = f"You have {displayed} {noun} remaining."
        elif limit_type == 2:
            noun = "minute" if displayed == "1" else "minutes"
            message = f"You have {displayed} {noun} remaining."
        else:
            message = f"You have {displayed} remaining."
        return message, message
    message = (
        "هذه تفاصيل رصيد السماح الخاص بك."
        if language == "ar"
        else "Here are your exceptional-entry allowance details."
    )
    return message, message


def _rows(blocks: list[ResponseBlock]) -> int:
    return sum(len(block.rows) for block in blocks if isinstance(block, TableBlock))


def _message_for(intent: str, blocks: list[ResponseBlock], language: str) -> tuple[str, str]:
    row_count = _rows(blocks)
    item_count = len(getattr(blocks[0], "items", [])) if blocks else 0
    table = next((block for block in blocks if isinstance(block, TableBlock)), None)
    if language == "ar":
        leads = {
            "profile": "هذه تفاصيل ملفك.",
            "attendance": f"لديك {row_count} سجل حضور خلال هذه الفترة.",
            "missing_punches": (
                f"لديك {row_count} بصمة مفقودة خلال هذه الفترة."
                if row_count
                else "لا توجد لديك بصمات مفقودة خلال هذه الفترة."
            ),
            "notifications": f"لديك {row_count or item_count} إشعارًا.",
            "day_types": f"لديك {row_count} نوعًا متاحًا.",
            "requests": (
                f"لديك {row_count} طلبًا خلال هذه الفترة."
                if row_count
                else "لا توجد لديك طلبات خلال هذه الفترة."
            ),
            "approvals": f"لديك {row_count} طلب موافقة معلق.",
        }
    else:
        leads = {
            "profile": "Here are your profile details.",
            "attendance": _english_count(
                row_count, "attendance record", "attendance records", " for this period"
            ),
            "missing_punches": (
                _english_count(
                    row_count, "missing punch", "missing punches", " for this period"
                )
            ),
            "notifications": _english_count(item_count, "notification", "notifications"),
            "day_types": _english_count(
                row_count, "available day type", "available day types"
            ),
            "requests": _english_count(
                row_count, "request", "requests", " for this period"
            ),
            "approvals": _english_count(
                row_count, "pending approval", "pending approvals"
            ),
        }
    if intent == "exceptional_entries":
        if language == "ar":
            lead = (
                "لا توجد طلبات استثناء خلال هذه الفترة."
                if row_count == 0
                else "لديك طلب استثناء واحد خلال هذه الفترة."
                if row_count == 1
                else f"لديك {row_count} من طلبات الاستثناء خلال هذه الفترة."
            )
        else:
            lead = _english_count(
                row_count,
                "exceptional-entry request",
                "exceptional-entry requests",
                " for this period",
            )
    else:
        lead = leads[intent]
    display = f"{lead}\n\n{markdown_table(table)}" if table and table.rows else lead
    return display, lead


async def try_fast_read(
    message: str,
    *,
    lang: int,
    response_language: str,
) -> FastReadResult | None:
    intents = classify_fast_read(message)
    if not intents:
        return None
    start, end = _date_range(message)

    loaders: dict[str, tuple[str, Callable[[], Awaitable[Any]]]] = {
        "balance": (
            "get_exceptional_entry_balance",
            lambda: get_exceptional_entry_balance(_balance_date(message)),
        ),
        "profile": ("get_profile_data", lambda: get_profile_data(lang=lang)),
        "attendance": (
            "get_attendance_summary",
            lambda: get_attendance_summary(start, end, lang=lang),
        ),
        "missing_punches": (
            "get_missing_punch_suggestions",
            lambda: get_missing_punch_suggestions(start, end, lang=lang),
        ),
        "notifications": ("get_notifications", lambda: get_notifications(lang=lang)),
        "day_types": ("get_day_types", lambda: cached_day_types(lang)),
        "requests": (
            "get_my_request_status",
            lambda: get_my_request_status(start, end, lang=lang),
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
    payloads = await asyncio.gather(*(loader() for _, loader in selected))

    all_blocks: list[ResponseBlock] = []
    display_parts: list[str] = []
    speech_parts: list[str] = []
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
            normalized = missing_punch_tool_data(normalize_missing_punch_suggestions(payload))
            blocks = [missing_punch_block(normalized)]
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
            blocks = [request_status_block(payload)]
        elif intent == "exceptional_entries":
            blocks = [exceptional_entries_block(payload, language=response_language)]
        else:
            blocks = [approvals_block(payload)]
        display, speech = _message_for(intent, blocks, response_language)
        all_blocks.extend(blocks)
        display_parts.append(display)
        speech_parts.append(speech)

    return FastReadResult(
        message="\n\n".join(display_parts),
        speech_message=" ".join(speech_parts),
        tools_used=[tool for tool, _ in selected],
        blocks=all_blocks,
    )
