from __future__ import annotations

import json
import asyncio
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from difflib import SequenceMatcher
from typing import Any
from uuid import uuid4

from app.ai.actions import (
    LessHoursDay,
    bookable_day_type_group,
    exceptional_entry_attributes,
    find_pending_exceptional_entries,
    inspect_less_hours_period,
    less_hours_duplicate_guard_message,
    prepare_cancel_exceptional_candidate,
)
from app.ai.attendance_intent import is_explicit_missing_punch_correction
from app.ai.reason_matcher import deterministic_reason_match
from app.ai.sessions import SessionStore
from app.ai.tools import (
    ToolExecutionResult,
    execute_tool,
    resolve_relative_date_range,
)
from app.models.schemas import ActionsBlock, BlockAction, ResponseBlock
from app.audit import record_action_state
from app.resourceplus import ResourcePlusError
from app.resourceplus.exceptional import get_exceptional_entry_balance
from app.resourceplus.reference_cache import cached_day_types, cached_exception_reasons
from app.services.fast_reads import classify_fast_read
from app.services.response_blocks import (
    cancellable_exception_actions,
    cancellable_exceptions_block,
    confirmation_block,
    day_types_block,
    exceptional_balance_block,
    less_hours_block,
    less_hours_correction_actions,
    reason_actions,
)
from app.time_context import resourceplus_today


@dataclass(frozen=True)
class ConversationOutcome:
    success: bool
    message: str
    speech_message: str
    tools_used: list[str] = field(default_factory=list)
    requires_confirmation: bool = False
    confirmation_id: str | None = None
    needs_reason: bool = False
    reason_options: list[str] | None = None
    blocks: list[ResponseBlock] = field(default_factory=list)


_MISSING_CORRECTION = re.compile(
    r"(?:\b(?:fix|correct|regulari[sz]e|amend)\b.{0,50}\b(?:missing|missed)?\s*punch(?:es)?\b)"
    r"|(?:\b(?:missing|missed)\s+punch(?:es)?\b.{0,50}\b(?:fix|correct|regulari[sz]e)\b)",
    re.I,
)
_EXCEPTIONAL_ENTRY_CORRECTION = re.compile(
    r"\b(?:book|create|submit|fix|correct|regulari[sz]e)\b.{0,60}"
    r"\b(?:exceptional|exception)[-\s]+entr(?:y|ies)\b",
    re.I,
)
_LEAVE_BOOKING = re.compile(
    r"\b(?:need|want|book|take|use|apply(?:\s+for)?|request)\b.{0,60}"
    r"\b(?:leave|vacation|day\s+off|business\s+travel)\b",
    re.I,
)
_DIRECT_TRAVEL_BOOKING = re.compile(
    r"\bbusiness\s+travel\b.{0,35}\b(?:on|for)\b", re.I
)
_LESS_HOURS = re.compile(
    r"\b(?:less[-\s]+hours?|missing\s+hours?|short\s+hours?|attendance\s+gaps?|"
    r"attendance\s+issues?|shortfall|late\s+(?:arrival|attendance)|early\s+departure)\b",
    re.I,
)
_ATTENDANCE_GAP_QUESTION = re.compile(
    r"\b(?:do\s+i\s+have\s+(?:anything|something)\s+to\s+(?:fix|correct)|"
    r"anything\s+(?:i\s+need\s+to|to)\s+(?:fix|correct)|"
    r"what\s+attendance\s+issues?\s+do\s+i\s+have|"
    r"any\s+gaps?|which\s+ones?\s+can\s+i\s+(?:fix|correct)|"
    r"what\s+can\s+i\s+(?:fix|correct))\b",
    re.I,
)
_CONTEXTUAL_CORRECTION = re.compile(
    r"^\s*(?:please\s+)?(?:fix|correct|do)\s*[,;:]?\s+"
    r"(?:my\s+attendance|that\s+day|this\s+one|it|(?:the\s+)?\d{1,2}(?:st|nd|rd|th)?|"
    r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|"
    r"aug(?:ust)?|sep(?:tember)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s+\d{1,2})"
    r"\s*[.!]?\s*$",
    re.I,
)
_GENERAL_ATTENDANCE_CORRECTION = re.compile(
    r"\b(?:need|want|would\s+like)\s+to\s+(?:fix|correct)\s+(?:my\s+)?attendance\b",
    re.I,
)
_LESS_HOURS_CORRECTION = re.compile(
    r"(?:\b(?:fix|correct|adjust|amend|regulari[sz]e)\b.{0,60}"
    r"\b(?:less[-\s]+hours?|short\s+hours?|shortfall|late\s+(?:arrival|attendance)|early\s+departure|entry)\b)"
    r"|(?:\b(?:use|apply)\b.{0,35}\b(?:buffer|excuse\s+time)\b)",
    re.I,
)
_CANCEL_EXCEPTION = re.compile(
    r"\b(?:cancel|withdraw)\b.{0,60}\b"
    r"(?:exceptions?|(?:exception|exceptional)[-\s]+entr(?:y|ies)(?:[-\s]+requests?)?)\b"
    r"|\b(?:exceptions?|(?:exception|exceptional)[-\s]+entr(?:y|ies)(?:[-\s]+requests?)?)\b"
    r".{0,60}\b(?:cancel|withdraw)\b",
    re.I,
)
_EXCEPTIONAL_ENTRY_READ = re.compile(
    r"\b(?:show|list|display|view|what|which|do\s+i\s+have)\b.{0,80}"
    r"\b(?:exceptional|exception)\s+"
    r"(?:entries?|entry\s+requests?|requests?|energies)\b",
    re.I,
)
_OPAQUE_CANDIDATE_SELECTION = re.compile(
    r"^\s*select\s+exceptional\s+entry\s+(ce-[a-f0-9]{8,64})\s*[.!]?\s*$",
    re.I,
)
_PARTIAL_HOURS_CORRECTION = re.compile(
    r"\b(?:fix|correct|adjust|amend|regulari[sz]e)\b.{0,40}"
    r"\b\d{1,3}\s*(?:minutes?|mins?)\b",
    re.I,
)
_CASUAL = re.compile(
    r"^(?:thanks?(?:\s+you)?|thank\s+you|ok(?:ay)?|great|hello|hi|hey)[.!\s]*$",
    re.I,
)
_MODEL_ROUTED_HR_TOPIC = re.compile(
    r"\b(?:supervisor|payslip|salary|leave\s+balance|business\s+travel)\b",
    re.I,
)
_INDEPENDENT_REQUEST_CUE = re.compile(
    r"\b(?:show|list|display|view|what|which|how|check|tell|approve|reject)\b",
    re.I,
)
_MONTHS = {
    name.casefold(): number
    for number, name in enumerate(
        (
            "",
            "January",
            "February",
            "March",
            "April",
            "May",
            "June",
            "July",
            "August",
            "September",
            "October",
            "November",
            "December",
        )
    )
    if name
}
_MONTHS.update({name[:3].casefold(): number for name, number in list(_MONTHS.items())})
_MONTHS.update(
    {
        "يناير": 1,
        "فبراير": 2,
        "مارس": 3,
        "أبريل": 4,
        "ابريل": 4,
        "مايو": 5,
        "يونيو": 6,
        "يوليو": 7,
        "أغسطس": 8,
        "اغسطس": 8,
        "سبتمبر": 9,
        "أكتوبر": 10,
        "اكتوبر": 10,
        "نوفمبر": 11,
        "ديسمبر": 12,
    }
)
_ENGLISH_WEEKDAYS = {
    "monday": 0,
    "tuesday": 1,
    "wednesday": 2,
    "thursday": 3,
    "friday": 4,
    "saturday": 5,
    "sunday": 6,
}
_ARABIC_WEEKDAYS = {
    "الاثنين": 0,
    "الإثنين": 0,
    "الثلاثاء": 1,
    "الأربعاء": 2,
    "الاربعاء": 2,
    "الخميس": 3,
    "الجمعة": 4,
    "السبت": 5,
    "الأحد": 6,
    "الاحد": 6,
}
_ARABIC_WEEKDAY_LABELS = (
    "الاثنين",
    "الثلاثاء",
    "الأربعاء",
    "الخميس",
    "الجمعة",
    "السبت",
    "الأحد",
)
_ARABIC_MONTH_LABELS = (
    "",
    "يناير",
    "فبراير",
    "مارس",
    "أبريل",
    "مايو",
    "يونيو",
    "يوليو",
    "أغسطس",
    "سبتمبر",
    "أكتوبر",
    "نوفمبر",
    "ديسمبر",
)
_ARABIC_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u0640]")


def _normalized(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", value.casefold(), flags=re.UNICODE).split())


def _arabic_normalized(value: str) -> str:
    normalized = _ARABIC_DIACRITICS.sub("", value.casefold())
    normalized = normalized.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}))
    return " ".join(re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE).split())


def _is_missing_request(message: str) -> bool:
    normalized = _normalized(message)
    return bool(
        _MISSING_CORRECTION.search(message)
        or _EXCEPTIONAL_ENTRY_CORRECTION.search(message)
        or is_explicit_missing_punch_correction(message)
    ) or (
        any(cue in normalized for cue in ("صحح", "تصحيح", "تعديل"))
        and "بصم" in normalized
    )


def _is_leave_request(message: str) -> bool:
    normalized = _normalized(message)
    arabic_booking = (
        any(
            cue in normalized
            for cue in (
                "أبغى",
                "ابغى",
                "أريد",
                "اريد",
                "أحتاج",
                "احتاج",
                "أبي",
                "ابي",
                "استخدم",
                "طلب",
            )
        )
        and any(cue in normalized for cue in ("إجاز", "اجاز", "سفر"))
        and not any(cue in normalized for cue in ("رصيد", "أعرف", "اعرف", "متبقي"))
    )
    return bool(
        _LEAVE_BOOKING.search(message)
        or _DIRECT_TRAVEL_BOOKING.search(message)
    ) or arabic_booking


def is_leave_balance_request(message: str) -> bool:
    """Identify a leave balance read without confusing it with buffer balance."""

    normalized = _arabic_normalized(message)
    english = bool(
        re.search(r"\b(?:leave|vacation)\s+balance\b", message, re.I)
        or re.search(
            r"\b(?:how\s+much|how\s+many|what(?:'s|\s+is)|show|check)\b"
            r".{0,45}\b(?:leave|vacation)\b",
            message,
            re.I,
        )
        or re.search(
            r"\b(?:leave|vacation)\b.{0,35}\b(?:remaining|left|available)\b",
            message,
            re.I,
        )
    )
    arabic_leave = any(cue in normalized for cue in ("اجازة", "اجازتي", "الاجازات"))
    arabic_amount = any(
        cue in normalized for cue in ("رصيد", "باقي", "متبقي", "كم", "المتاح")
    )
    return english or (arabic_leave and arabic_amount)


def is_less_hours_request(message: str) -> bool:
    normalized = _arabic_normalized(message)
    if is_attendance_gap_read_request(message):
        return True
    return bool(
        _LESS_HOURS.search(message) or _ATTENDANCE_GAP_QUESTION.search(message)
    ) or any(
        cue in normalized
        for cue in (
            "ساعات ناقصة",
            "نقص ساعات",
            "الساعات الناقصة",
            "تاخر في الدخول",
            "تاخر الدخول",
            "التاخير",
            "خروج مبكر",
            "الخروج المبكر",
            "فجوات الحضور",
            "فجوات حضور",
            "مشاكل الحضور",
            "شيء احتاج اصححه",
            "شي احتاج اصححه",
        )
    )


def is_attendance_gap_read_request(message: str) -> bool:
    """Recognize deterministic attendance-gap reads, including natural questions."""

    normalized = _arabic_normalized(message)
    if _ATTENDANCE_GAP_QUESTION.search(message):
        return True
    if (
        _LESS_HOURS_CORRECTION.search(message)
        or _PARTIAL_HOURS_CORRECTION.search(message)
        or _GENERAL_ATTENDANCE_CORRECTION.search(message)
        or _CONTEXTUAL_CORRECTION.search(message)
    ):
        return False
    if _LESS_HOURS.search(message):
        return bool(
            re.search(
                r"\b(?:show|list|display|view|what|which|do\s+i\s+have|how\s+many)\b",
                message,
                re.I,
            )
            or re.search(r"\b(?:last|previous|this)\s+(?:month|week)\b", message, re.I)
        )
    return any(
        cue in normalized
        for cue in (
            "اعرض الساعات الناقصة",
            "اظهر الساعات الناقصة",
            "ورني الساعات الناقصة",
            "هل عندي ساعات ناقصة",
            "هل عندي شيء اصلحه",
            "هل عندي شي اصلحه",
            "وش احتاج اصحح",
            "ما مشاكل الحضور",
        )
    )


def is_less_hours_correction_request(message: str) -> bool:
    if is_attendance_gap_read_request(message):
        return False
    normalized = _arabic_normalized(message)
    arabic_action = any(
        cue in normalized
        for cue in ("صحح", "تصحيح", "عدل", "تعديل", "عوض", "تعويض", "استخدم")
    )
    arabic_target = is_less_hours_request(message) or (
        "دقائق" in normalized and any(cue in normalized for cue in ("صحح", "عوض"))
    ) or (
        "استخدم" in normalized
        and any(cue in normalized for cue in ("السماح", "التعويض", "الاستئذان"))
    )
    return bool(
        _LESS_HOURS_CORRECTION.search(message)
        or _PARTIAL_HOURS_CORRECTION.search(message)
        or _CONTEXTUAL_CORRECTION.search(message)
        or _GENERAL_ATTENDANCE_CORRECTION.search(message)
    ) or (arabic_action and arabic_target)


def _is_cancel_exception_request(message: str) -> bool:
    normalized = _arabic_normalized(message)
    if (
        bool(re.search(r"\b(?:cancel|withdraw)\b", normalized))
        and "pending" in normalized
        and "exceptional" in normalized
        and bool(re.search(r"\b(?:entry|interest)\b", normalized))
    ):
        return True
    return bool(_CANCEL_EXCEPTION.search(message)) or (
        any(cue in normalized for cue in ("الغ", "الغاء", "اسحب", "سحب"))
        and any(cue in normalized for cue in ("استثنائي", "استثناء"))
    )


def _is_exceptional_entry_read_request(message: str) -> bool:
    if _EXCEPTIONAL_ENTRY_READ.search(message):
        return True
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


def _is_explicit_less_hours_read_request(message: str) -> bool:
    return is_attendance_gap_read_request(message)


def _is_clear_independent_hr_intent(message: str) -> bool:
    """Separate a fresh HR request from a value for the active draft."""

    normalized = _arabic_normalized(message)
    model_routed_request = bool(
        (
            _INDEPENDENT_REQUEST_CUE.search(message)
            and _MODEL_ROUTED_HR_TOPIC.search(message)
        )
        or (
            any(
                cue in normalized
                for cue in (
                    "اعرض",
                    "اظهر",
                    "ارني",
                    "ورني",
                    "ما هي",
                    "ماهي",
                    "كم",
                    "هل",
                    "وافق",
                    "ارفض",
                )
            )
            and any(
                topic in normalized
                for topic in (
                    "الحضور",
                    "ملفي",
                    "الملف",
                    "اشعار",
                    "تنبيه",
                    "موافق",
                    "المشرف",
                    "المدير",
                    "راتب",
                    "رصيد الاجاز",
                    "سفر العمل",
                )
            )
        )
    )
    return bool(
        _is_missing_request(message)
        or _is_leave_request(message)
        or is_less_hours_correction_request(message)
        or _is_cancel_exception_request(message)
        or _is_exceptional_entry_read_request(message)
        or _is_explicit_less_hours_read_request(message)
        or classify_fast_read(message)
        or model_routed_request
    )


def extract_direction(message: str) -> str | None:
    normalized = _normalized(message)
    has_in = bool(re.search(r"\b(?:in|punch\s+in|check\s+in)\b", normalized)) or "دخول" in normalized
    has_out = bool(re.search(r"\b(?:out|punch\s+out|check\s+out)\b", normalized)) or any(
        cue in normalized for cue in ("خروج", "انصراف")
    )
    if has_in == has_out:
        return None
    return "IN" if has_in else "OUT"


def extract_date(message: str, *, today: date | None = None) -> date | None:
    current = today or resourceplus_today()
    normalized = _normalized(message)
    arabic = _arabic_normalized(message)
    if re.search(r"\btomorrow\b", normalized) or any(cue in arabic for cue in ("غدا", "بكرة")):
        return current + timedelta(days=1)
    if re.search(r"\byesterday\b", normalized) or "أمس" in message or "امس" in message:
        return current - timedelta(days=1)
    if re.search(r"\btoday\b", normalized) or "اليوم" in normalized:
        return current
    iso = re.search(r"\b(20\d{2})-(\d{1,2})-(\d{1,2})\b", message)
    if iso:
        try:
            return date(int(iso.group(1)), int(iso.group(2)), int(iso.group(3)))
        except ValueError:
            return None
    numeric = re.search(r"\b(\d{1,2})[/-](\d{1,2})(?:[/-](20\d{2}))?\b", message)
    if numeric:
        try:
            return date(int(numeric.group(3) or current.year), int(numeric.group(2)), int(numeric.group(1)))
        except ValueError:
            return None
    named = re.search(
        r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(" + "|".join(_MONTHS) + r")(?:\s+(20\d{2}))?\b",
        normalized,
    ) or re.search(
        r"\b(" + "|".join(_MONTHS) + r")\s+(\d{1,2})(?:st|nd|rd|th)?(?:\s+(20\d{2}))?\b",
        normalized,
    )
    if named:
        first, second, year = named.groups()
        day_value, month_value = (
            (int(first), _MONTHS[second]) if first.isdigit() else (int(second), _MONTHS[first])
        )
        try:
            return date(int(year or current.year), month_value, day_value)
        except ValueError:
            return None
    ordinal = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)\b", normalized)
    if ordinal:
        try:
            return current.replace(day=int(ordinal.group(1)))
        except ValueError:
            return None
    weekday_match = re.search(
        r"\b(?:(next|this)\s+)?(" + "|".join(_ENGLISH_WEEKDAYS) + r")\b",
        normalized,
    )
    target_weekday: int | None = None
    explicitly_next = False
    if weekday_match is not None:
        explicitly_next = weekday_match.group(1) == "next"
        target_weekday = _ENGLISH_WEEKDAYS[weekday_match.group(2)]
    else:
        for label, weekday in _ARABIC_WEEKDAYS.items():
            if label in normalized:
                target_weekday = weekday
                explicitly_next = any(
                    cue in normalized
                    for cue in ("القادم", "القادمة", "الجاي", "الجاية", "المقبل", "المقبلة")
                )
                break
    if target_weekday is not None:
        days_ahead = (target_weekday - current.weekday()) % 7
        if days_ahead == 0 and explicitly_next:
            days_ahead = 7
        return current + timedelta(days=days_ahead)
    return None


def _less_hours_range(message: str) -> tuple[date, date] | None:
    current = resourceplus_today()
    selected_date = extract_date(message, today=current)
    if selected_date is not None:
        return selected_date, selected_date
    resolved = resolve_relative_date_range(message, today=current)
    if resolved is not None:
        return resolved.from_date, resolved.to_date
    normalized = _normalized(message)
    if any(cue in normalized for cue in ("هذا الأسبوع", "الاسبوع هذا", "الأسبوع الحالي")):
        start = current - timedelta(days=current.weekday())
        return start, current
    return None


def _candidate_dates_from_context(
    draft: object | None,
    trusted: object | None,
) -> tuple[date, ...]:
    values: list[str] = []
    slots = getattr(draft, "slots", {}) if draft is not None else {}
    if isinstance(slots, dict):
        values.extend(
            value
            for value in str(slots.get("candidate_dates", "")).split("\x1f")
            if value
        )
    values.extend(getattr(trusted, "correction_dates", ()) or ())
    parsed: list[date] = []
    for value in dict.fromkeys(values):
        try:
            parsed.append(date.fromisoformat(value))
        except (TypeError, ValueError):
            continue
    return tuple(parsed)


def _verified_no_actionable_gap_message(trusted: object, language: str) -> str | None:
    tools = set(getattr(trusted, "tools_used", ()) or ())
    if not {"get_attendance_summary", "get_exceptional_entry_requests"} <= tools:
        return None
    if getattr(trusted, "correction_dates", ()):
        return None
    period = getattr(trusted, "attendance_period", None)
    if not isinstance(period, tuple) or len(period) != 2:
        return None
    try:
        start, end = (date.fromisoformat(value) for value in period)
    except (TypeError, ValueError):
        return None
    if start.year == end.year and start.month == end.month:
        if language == "ar":
            month_names = (
                "", "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو",
                "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر",
            )
            period_label = month_names[start.month]
            if end == resourceplus_today() and end.day < 28:
                period_label += " حتى الآن"
            return (
                f"لا أرى أي فجوات في الحضور خلال {period_label}. "
                "إذا كنت تقصد شهراً آخر أو تاريخاً محدداً، فأخبرني به."
            )
        period_label = start.strftime("%B")
        if end == resourceplus_today() and end.day < 28:
            period_label += " so far"
        return (
            f"I don't see any attendance gaps for {period_label}. "
            "If you mean another month or a specific date, tell me which one."
        )
    if language == "ar":
        return (
            f"لا أرى أي فجوات في الحضور من {start.isoformat()} إلى {end.isoformat()}. "
            "إذا كنت تقصد فترة أخرى أو تاريخاً محدداً، فأخبرني به."
        )
    return (
        f"I don't see any attendance gaps from {start.isoformat()} to {end.isoformat()}. "
        "If you mean another period or a specific date, tell me which one."
    )


def _contextual_correction_date(
    message: str,
    *,
    draft: object | None,
    trusted: object | None,
) -> date | None:
    """Resolve conversational dates without letting context authorize a write."""

    normalized = _normalized(message)
    candidates = _candidate_dates_from_context(draft, trusted)
    has_named_month = any(month in normalized.split() for month in _MONTHS)
    has_absolute_date = bool(
        re.search(r"\b20\d{2}-\d{1,2}-\d{1,2}\b", message)
        or re.search(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]20\d{2})?\b", message)
        or has_named_month
    )
    if has_absolute_date:
        return extract_date(message)

    ordinal = re.search(r"\b(\d{1,2})(?:st|nd|rd|th)?\b", normalized)
    if ordinal is not None:
        day_number = int(ordinal.group(1))
        matches = [candidate for candidate in candidates if candidate.day == day_number]
        if len(matches) == 1:
            return matches[0]

    contextual_reference = bool(
        re.search(r"\b(?:that\s+day|this\s+one|it)\b", normalized)
        or any(cue in _arabic_normalized(message) for cue in ("هذا اليوم", "هاليوم", "هذا", "ذا"))
    )
    if contextual_reference:
        discussed = getattr(trusted, "discussed_date", None)
        if isinstance(discussed, str):
            try:
                return date.fromisoformat(discussed)
            except ValueError:
                pass
        if len(candidates) == 1:
            return candidates[0]
    return None


def _format_correction_choices(values: tuple[date, ...], language: str) -> str:
    if language == "ar":
        return "، ".join(value.isoformat() for value in values)
    labels = [
        (
            f"{value.strftime('%b')} {value.day}"
            if index == 0 or value.month != values[index - 1].month
            else str(value.day)
        )
        for index, value in enumerate(values)
    ]
    if len(labels) == 1:
        return labels[0]
    if len(labels) == 2:
        return " and ".join(labels)
    return ", ".join(labels[:-1]) + f", and {labels[-1]}"


def _explicit_entry_type(message: str) -> int | None:
    normalized = _arabic_normalized(message)
    if _is_missing_request(message):
        direction = extract_direction(message)
        if direction is not None:
            return 1 if direction == "IN" else 2
    if re.search(r"\b(?:only\s+)?(?:late\s+(?:in|arrival)|arrival\s+late)\b", normalized) or any(
        cue in normalized
        for cue in (
            "تاخر الدخول",
            "التاخر في الدخول",
            "الدخول المتاخر",
            "التاخير",
            "وصول متاخر",
        )
    ):
        return 1
    if re.search(r"\b(?:only\s+)?(?:early\s+(?:out|departure)|departure\s+early)\b", normalized) or any(
        cue in normalized for cue in ("خروج مبكر", "الخروج المبكر", "انصراف مبكر")
    ):
        return 2
    return None


def _explicit_minutes(message: str) -> int | None:
    normalized = _arabic_normalized(message)
    match = re.search(r"\b(?:only\s+)?(\d{1,3})\s*(?:minutes?|mins?)\b", normalized)
    if match is None:
        match = re.search(r"(?:فقط\s*)?(\d{1,3})\s*د(?:قيقة|قائق)", normalized)
    if match is None:
        return None
    value = int(match.group(1))
    return value if value > 0 else None


def _reason_names(payload: Any) -> list[str]:
    if not isinstance(payload, list):
        return []
    names: list[str] = []
    for row in payload:
        if not isinstance(row, dict):
            continue
        folded = {str(key).casefold(): value for key, value in row.items()}
        name = folded.get("reasonname")
        if isinstance(name, str) and name.strip():
            names.append(name.strip())
    return names


def _less_hours_message(day: LessHoursDay, language: str) -> str:
    displayed_date = (
        f"{day.attendance_date.day} "
        f"{day.attendance_date.strftime('%B')}"
    )
    if day.eligibility == "no_punches":
        return (
            f"لا توجد بصمات حضور في {day.attendance_date.isoformat()}، لذلك لا يمكن "
            "تصحيح هذا اليوم كإدخال استثنائي. يمكنك طلب إجازة أو مهمة عمل بدلاً من ذلك."
            if language == "ar"
            else (
                f"I don't see any attendance punches for {displayed_date}, so this "
                "can't be corrected as an attendance gap. You can apply for "
                "Leave or Business Travel for that day instead."
            )
        )
    if day.eligibility == "no_missing_hours":
        return (
            "حضورك في هذا اليوم مكتمل، وما يحتاج تصحيح."
            if language == "ar"
            else "Your attendance for that day is complete. Nothing needs to be fixed."
        )
    labels = (
        {
            "week_end": "عطلة أسبوعية",
            "holiday": "عطلة",
            "leave": "إجازة",
            "business_travel": "مهمة عمل",
        }
        if language == "ar"
        else {
            "week_end": "Week End",
            "holiday": "Holiday",
            "leave": "Leave",
            "business_travel": "Business Travel",
        }
    )
    label = labels.get(day.eligibility, day.day_type)
    if language == "ar":
        return (
            f"هذا اليوم مسجّل كـ{label}، لذلك ما فيه ساعات تحتاج تصحيح."
        )
    return f"That day is marked as {label}, so there are no hours to correct."


def _correction_reason_prompt(
    day: LessHoursDay,
    entry_type: int | None,
    language: str,
) -> str:
    if language == "ar":
        scope = (
            " لتصحيح جهة الدخول"
            if entry_type == 1
            else " لتصحيح جهة الخروج"
            if entry_type == 2
            else ""
        )
        return (
            f"تمام، عندك {day.less_hours} ساعات ناقصة يوم {day.attendance_date.isoformat()}{scope}. "
            "وش سبب التصحيح؟"
        )
    scope = (
        "IN-side "
        if entry_type == 1
        else "OUT-side "
        if entry_type == 2
        else ""
    )
    return (
        f"Sure. You were {day.less_hours} short on "
        f"{day.attendance_date.strftime('%b')} {day.attendance_date.day}. "
        + (f"What was the reason for the {scope}correction?" if scope else "What was the reason?")
    )


def _less_hours_read_copy(
    resolved_days: tuple[object, ...],
    language: str,
) -> tuple[str, str]:
    """Summarize verified rows; the table retains per-date detail."""

    available = [item.day for item in resolved_days if item.correction_available]
    approved = sum(item.state == "approved_exception" for item in resolved_days)
    existing = sum(item.state == "existing_request" for item in resolved_days)
    unavailable = sum(item.state == "correction_unavailable" for item in resolved_days)
    count = len(available)
    if unavailable:
        message = (
            "لقيت أيام فيها ساعات ناقصة، لكن ما قدرت أتحقق من طلبات التصحيح الحالية. "
            "ما راح أعرض تصحيح جديد حالياً."
            if language == "ar"
            else "I found short-hour days, but couldn't check your existing requests. I can't offer a new correction right now."
        )
        return message, message
    if len(resolved_days) == 1 and approved:
        message = (
            "تصحيح هذا اليوم معتمد، لكن الحضور لا يزال يظهر ساعات ناقصة."
            if language == "ar"
            else "That day is covered by an approved attendance correction, but your attendance still shows a shortfall."
        )
        return message, message
    if len(resolved_days) == 1 and existing:
        message = (
            "فيه طلب تصحيح موجود لهذا اليوم، فما راح أبدأ طلب ثاني."
            if language == "ar"
            else "There's already a correction request for that day, so I won't create another one."
        )
        return message, message
    if language == "ar":
        if count:
            dates = "، ".join(day.attendance_date.isoformat() for day in available)
            lead = (
                "عندك يوم حضور واحد يحتاج تصحيح خلال هالفترة"
                if count == 1
                else "عندك يومين حضور يحتاجون تصحيح خلال هالفترة"
                if count == 2
                else f"عندك {count} أيام حضور تحتاج تصحيح خلال هالفترة"
            )
            display = f"{lead}: {dates}." if count <= 5 else f"{lead}."
            speech = f"{lead}."
        elif approved or existing:
            display = speech = "ما عندك ساعات ناقصة متاحة للتصحيح حالياً."
        elif len(resolved_days) == 1 and resolved_days[0].state == "no_missing_hours":
            display = speech = "حضورك في هذا اليوم مكتمل، وما يحتاج تصحيح."
        else:
            display = speech = "ما عندك ساعات ناقصة تحتاج تصحيح خلال هالفترة."
        spoken_statuses: list[str] = []
        if approved:
            days = "يوم ثاني" if approved == 1 else "يومين ثانية" if approved == 2 else f"{approved} أيام ثانية"
            detail = f"وفيه {days} عليها تصحيحات معتمدة."
            display += f" {detail}"
            spoken_statuses.append(f"{days} عليها تصحيحات معتمدة")
        if existing:
            days = "يوم ثاني" if existing == 1 else "يومين ثانية" if existing == 2 else f"{existing} أيام ثانية"
            detail = f"وفيه {days} عليها طلبات موجودة، فما راح أكررها."
            display += f" {detail}"
            spoken_statuses.append(f"{days} عليها طلبات موجودة")
        if spoken_statuses:
            speech += " والباقي: " + "، و".join(spoken_statuses) + "."
        return display, speech

    period = (
        available[0].attendance_date.strftime("%B")
        if available and all(
            day.attendance_date.year == available[0].attendance_date.year
            and day.attendance_date.month == available[0].attendance_date.month
            for day in available
        )
        else "this period"
    )
    if count:
        noun = "gap" if count == 1 else "gaps"
        lead = f"You have {count} {period} attendance {noun} left to fix"
        if count <= 5:
            same_month = period != "this period"
            date_labels = [
                (
                    f"{day.attendance_date.day}"
                    if same_month and index > 0
                    else f"{day.attendance_date.strftime('%b')} {day.attendance_date.day}"
                )
                for index, day in enumerate(available)
            ]
            dates = (
                date_labels[0]
                if len(date_labels) == 1
                else " and ".join(date_labels)
                if len(date_labels) == 2
                else ", ".join(date_labels[:-1]) + f", and {date_labels[-1]}"
            )
            display = f"{lead}: {dates}."
        else:
            display = f"{lead}."
        speech = f"{lead}."
    elif approved or existing:
        display = speech = "You have no attendance gaps available to correct right now."
    elif len(resolved_days) == 1 and resolved_days[0].state == "no_missing_hours":
        display = speech = "Your attendance for that day is complete. Nothing needs to be fixed."
    else:
        display = speech = "You have no attendance gaps to fix during this period."
    spoken_statuses: list[str] = []
    if approved:
        detail = (
            f"{approved} other short-hour {'day is' if approved == 1 else 'days are'} "
            "covered by approved corrections."
        )
        display += f" {detail}"
        spoken_statuses.append(
            f"{approved} {'other day is' if approved == 1 else 'other days are'} already covered"
        )
    if existing:
        detail = (
            f"{existing} other short-hour {'day already has a request' if existing == 1 else 'days already have requests'}, "
            "so I won't create duplicates."
        )
        display += f" {detail}"
        spoken_statuses.append(
            f"{existing} {'other day has' if existing == 1 else 'other days have'} a request already"
        )
    if spoken_statuses:
        speech += " " + "; ".join(spoken_statuses) + "."
    return display, speech


def _day_type_rows(payload: Any) -> list[dict[str, Any]]:
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


def _booking_group(message: str) -> str | None:
    normalized = _arabic_normalized(message)
    if "business travel" in normalized or any(
        cue in normalized for cue in ("مهمة عمل", "سفر عمل")
    ):
        return "business_travel"
    if any(cue in normalized for cue in ("leave", "vacation", "day off", "اجازة")):
        return "leave"
    return None


def _bookable_day_types(rows: list[dict[str, Any]], requested_group: str | None) -> list[dict[str, Any]]:
    return [
        row for row in rows
        if (group := bookable_day_type_group(row.get("group"))) is not None
        and requested_group in (None, group)
        and isinstance(row.get("dayType"), str)
        and row["dayType"].strip()
    ]


def leave_day_type_options(payload: Any) -> list[dict[str, Any]]:
    """Return only live ResourcePlus rows applicable to leave booking."""

    return _bookable_day_types(_day_type_rows(payload), "leave")


def _stored_day_type_options(slots: dict[str, str]) -> list[dict[str, Any]]:
    raw = slots.get("day_type_options")
    if not raw:
        return []
    try:
        names = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    if not isinstance(names, list):
        return []
    rows: list[dict[str, Any]] = []
    fallback_group = (
        "Business Travel"
        if slots.get("booking_group") == "business_travel"
        else "Leave"
    )
    for item in names:
        if isinstance(item, str) and item.strip():
            rows.append({"dayType": item, "group": fallback_group})
        elif isinstance(item, dict):
            name = item.get("dayType")
            group = item.get("group")
            if isinstance(name, str) and name.strip() and isinstance(group, str):
                rows.append({"dayType": name, "group": group})
    return rows


def _match_day_type(message: str, rows: list[dict[str, Any]]) -> tuple[str, str | None]:
    """Select one live name conservatively; never guess a day type ID."""

    normalized = _arabic_normalized(message)
    names = [str(row["dayType"]) for row in rows]
    direct = [name for name in names if _arabic_normalized(name) == normalized]
    if len(direct) == 1:
        return "matched", direct[0]
    if len(direct) > 1:
        return "ambiguous", None

    contained = [
        name for name in names
        if f" {_arabic_normalized(name)} " in f" {normalized} "
        and not (
            _arabic_normalized(name) in {"leave", "اجازة"}
            and _arabic_normalized(name) != normalized
        )
    ]
    if contained:
        longest = max(len(_arabic_normalized(name)) for name in contained)
        strongest = [name for name in contained if len(_arabic_normalized(name)) == longest]
        return ("matched", strongest[0]) if len(strongest) == 1 else ("ambiguous", None)

    # A short bare answer may omit "Leave". A partial name is accepted only
    # when every supplied word points to exactly one live bookable option.
    words = normalized.split()
    if not 1 <= len(words) <= 3:
        return "unknown", None
    partial = [
        name for name in names
        if all(word in _arabic_normalized(name).split() for word in words)
    ]
    if len(partial) == 1:
        return "matched", partial[0]
    if partial:
        return "ambiguous", None

    # Voice recognition can substitute a semantically nearby word (for example,
    # "relief" for "leave"). Compare only with the live choices retained for
    # this identity-bound draft; no production DayType names are embedded here.
    scored = sorted(
        (
            SequenceMatcher(None, normalized, _arabic_normalized(name)).ratio(),
            name,
        )
        for name in names
    )
    if not scored:
        return "unknown", None
    best_score, best_name = scored[-1]
    runner_up = scored[-2][0] if len(scored) > 1 else 0.0
    margin = best_score - runner_up
    if best_score >= 0.78 and margin >= 0.08:
        return "matched", best_name
    if best_score >= 0.62 and margin >= 0.08:
        return "suggested", best_name
    return "unknown", None


def _is_generic_leave_intent(message: str, rows: list[dict[str, Any]]) -> bool:
    """Return true when a booking utterance names leave, but no live type.

    Generic leave words describe the workflow.  They are not DayType
    candidates, even when a prior leave draft already exists.
    """

    if not _is_leave_request(message) or _booking_group(message) != "leave":
        return False
    message_words = set(_arabic_normalized(message).split())
    generic_words = {
        "leave",
        "vacation",
        "day",
        "off",
        "إجازة",
        "اجازة",
        "اجازه",
    }
    for row in rows:
        name = row.get("dayType")
        if not isinstance(name, str):
            continue
        distinctive_words = set(_arabic_normalized(name).split()) - generic_words
        if distinctive_words and distinctive_words <= message_words:
            return False
    return True


def _day_type_choice_blocks(
    rows: list[dict[str, Any]], language: str, requested_group: str | None
) -> list[ResponseBlock]:
    if not rows:
        return []
    table = day_types_block(rows)
    table.title = (
        "خيارات الإجازة" if requested_group == "leave" else
        "خيارات مهمة العمل" if requested_group == "business_travel" else
        "خيارات الإجازة ومهمة العمل"
    ) if language == "ar" else (
        "Leave options" if requested_group == "leave" else
        "Business Travel options" if requested_group == "business_travel" else
        "Leave & Business Travel options"
    )
    actions = ActionsBlock(
        title="اختر النوع" if language == "ar" else "Choose a day type",
        actions=[BlockAction(label=str(row["dayType"]), value=str(row["dayType"])) for row in rows],
    )
    return [table, actions]


def leave_day_type_choice_blocks(
    rows: list[dict[str, Any]], language: str
) -> list[ResponseBlock]:
    return _day_type_choice_blocks(rows, language, "leave")


def _display_booking_date(target: date, language: str) -> str:
    if language == "ar":
        weekday = _ARABIC_WEEKDAY_LABELS[target.weekday()]
        month = _ARABIC_MONTH_LABELS[target.month]
        suffix = f" {target.year}" if target.year != resourceplus_today().year else ""
        return f"{weekday}، {target.day} {month}{suffix}"
    suffix = f", {target.year}" if target.year != resourceplus_today().year else ""
    return f"{target.strftime('%A')}, {target.strftime('%b')} {target.day}{suffix}"


def _question(intent: str, slots: dict[str, str], language: str) -> str:
    if intent == "correct_missing_punch":
        missing = [key for key in ("date", "direction") if key not in slots]
        if language == "ar":
            return "أكيد. لأي تاريخ، وهل البصمة دخول أم خروج؟" if len(missing) == 2 else (
                "ما تاريخ البصمة المفقودة؟" if missing == ["date"] else "هل البصمة المفقودة دخول أم خروج؟"
            )
        return "Sure. Which date, and was it your IN or OUT punch?" if len(missing) == 2 else (
            "Which date was the missing punch?" if missing == ["date"] else "Was it your IN or OUT punch?"
        )
    missing = [key for key in ("date_from", "day_type") if key not in slots]
    if language == "ar":
        return "أكيد. ما نوع الإجازة ولأي تاريخ؟" if len(missing) == 2 else (
            "لأي تاريخ تريد الإجازة؟" if missing == ["date_from"] else "ما نوع الإجازة؟"
        )
    return "Sure. Which day type and date would you like?" if len(missing) == 2 else (
        "What date would you like?" if missing == ["date_from"] else "Which leave or day type would you like?"
    )


def _outcome_from_tool(
    result: ToolExecutionResult,
    language: str,
    tool_name: str,
) -> ConversationOutcome:
    if result.pending_action is not None:
        summary = result.pending_action.summary
        speech_summary = summary
        if result.pending_action.action_type == "create_exceptional_entry_from_summary":
            values = result.pending_action.validated_arguments
            try:
                target = date.fromisoformat(str(values["att_date"]))
            except (KeyError, ValueError):
                target = None
            requested_minutes = values.get("minutes")
            if not isinstance(requested_minutes, int):
                raw_duration = str(values.get("less_hours", ""))
                match = re.fullmatch(r"(\d{1,3}):(\d{2})", raw_duration)
                requested_minutes = (
                    int(match.group(1)) * 60 + int(match.group(2))
                    if match is not None else None
                )
            reason = str(values.get("reason_name", "attendance correction"))
            if target is not None and requested_minutes is not None:
                speech_summary = (
                    f"أرسل تصحيح {requested_minutes} دقيقة ليوم {target.isoformat()} بسبب {reason}؟"
                    if language == "ar"
                    else f"Submit the {requested_minutes}-minute correction for {target.strftime('%b')} {target.day} as {reason}?"
                )
        return ConversationOutcome(
            success=True,
            message=summary,
            speech_message=speech_summary,
            tools_used=[tool_name],
            requires_confirmation=True,
            confirmation_id=result.pending_action.confirmation_id,
            blocks=[confirmation_block(summary, language)],
        )
    message = result.terminal_message
    if not message:
        try:
            payload = json.loads(result.output)
            message = payload.get("message") or payload.get("error")
        except (TypeError, json.JSONDecodeError):
            message = None
    message = message or ("I couldn't prepare that request." if language == "en" else "تعذر تجهيز الطلب.")
    options = result.reason_options or []
    return ConversationOutcome(
        success=not result.failed,
        message=message,
        speech_message=message,
        tools_used=[tool_name] if result.tool_used else [],
        needs_reason=result.needs_reason,
        reason_options=options or None,
        blocks=[reason_actions(options, language)] if options else [],
    )


def _candidate_field(row: dict[str, Any], *names: str) -> Any:
    folded = {
        "".join(character for character in str(key).casefold() if character.isalnum()): value
        for key, value in row.items()
    }
    for name in names:
        key = "".join(character for character in name.casefold() if character.isalnum())
        if key in folded:
            return folded[key]
    return None


def _cancellation_candidate(row: dict[str, Any]) -> dict[str, str]:
    exceptional_id = _candidate_field(row, "exceptionalID", "exceptionID", "id")
    if exceptional_id in (None, ""):
        raise ValueError("ResourcePlus returned a cancellable entry without an ID.")
    return {
        "key": f"ce-{uuid4().hex[:12]}",
        "exceptional_id": str(exceptional_id),
        **exceptional_entry_attributes(row),
    }


def _stored_cancellation_candidates(slots: dict[str, str]) -> list[dict[str, str]]:
    raw = slots.get("cancellation_candidates")
    if not raw:
        return []
    try:
        candidates = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return []
    if not isinstance(candidates, list):
        return []
    required = {"key", "exceptional_id", "date", "type", "reason", "status"}
    return [
        candidate
        for candidate in candidates
        if isinstance(candidate, dict)
        and required.issubset(candidate)
        and all(isinstance(candidate[key], str) for key in required)
    ]


def _stemmed_words(value: str) -> tuple[str, ...]:
    words = tuple(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))
    return tuple(word[:-1] if len(word) > 3 and word.endswith("s") else word for word in words)


def _contains_words(message: tuple[str, ...], phrase: tuple[str, ...]) -> bool:
    if not phrase or len(phrase) > len(message):
        return False
    return any(
        message[index : index + len(phrase)] == phrase
        for index in range(len(message) - len(phrase) + 1)
    )


def _match_cancellation_candidates(
    message: str,
    candidates: list[dict[str, str]],
) -> list[dict[str, str]]:
    normalized = _normalized(message)
    arabic = _arabic_normalized(message)
    raw_message = message.casefold()
    keyed = [candidate for candidate in candidates if candidate["key"].casefold() in raw_message]
    if keyed:
        return keyed

    ordinal_words = {
        "first": 1,
        "second": 2,
        "third": 3,
        "fourth": 4,
        "fifth": 5,
        "الاول": 1,
        "الثاني": 2,
        "الثالث": 3,
        "الرابع": 4,
        "الخامس": 5,
    }
    ordinal: int | None = None
    number_match = re.search(r"\b(?:number|no)\s*(\d+)\b|#\s*(\d+)\b", normalized)
    if number_match:
        ordinal = int(number_match.group(1) or number_match.group(2))
    elif re.fullmatch(r"\s*(\d+)(?:st|nd|rd|th)?\s*", normalized):
        ordinal = int(re.sub(r"\D", "", normalized))
    else:
        for word, value in ordinal_words.items():
            comparison = arabic if any("\u0600" <= char <= "\u06ff" for char in word) else normalized
            if re.search(rf"\b{word}\b", comparison):
                ordinal = value
                break
    if ordinal is not None:
        return [candidates[ordinal - 1]] if 1 <= ordinal <= len(candidates) else []

    matches = list(candidates)
    used_criterion = False
    selected_date = extract_date(message)
    if selected_date is not None:
        used_criterion = True
        matches = [candidate for candidate in matches if candidate["date"] == selected_date.isoformat()]

    requested_type = None
    if re.search(r"\blate(?:\s+arrival)?\b", normalized):
        requested_type = "late arrival"
    elif re.search(r"\bearly(?:\s+departure)?\b", normalized):
        requested_type = "early departure"
    elif any(cue in arabic for cue in ("الخروج المبكر", "خروج مبكر")):
        requested_type = "early departure"
    elif any(cue in arabic for cue in ("الوصول المتاخر", "التاخير", "تاخر الدخول")):
        requested_type = "late arrival"
    if requested_type is not None:
        used_criterion = True
        matches = [candidate for candidate in matches if candidate["type"].casefold() == requested_type]

    message_words = _stemmed_words(message)
    reason_matches = [
        candidate
        for candidate in candidates
        if _contains_words(message_words, _stemmed_words(candidate["reason"]))
    ]
    if reason_matches:
        used_criterion = True
        reason_keys = {candidate["key"] for candidate in reason_matches}
        matches = [candidate for candidate in matches if candidate["key"] in reason_keys]
    return matches if used_criterion else []


def _candidate_as_resourceplus_row(candidate: dict[str, str]) -> dict[str, str]:
    return {
        "exceptionalID": candidate["exceptional_id"],
        "entryTime": candidate["date"],
        "entryTypeName": candidate["type"],
        "reason": candidate["reason"],
        "status": candidate["status"],
    }


def _cancellation_selection_blocks(
    candidates: list[dict[str, str]],
    language: str = "en",
) -> list[ResponseBlock]:
    block = cancellable_exceptions_block(
        [_candidate_as_resourceplus_row(candidate) for candidate in candidates],
        language,
    )
    actions = cancellable_exception_actions(candidates, language)
    return [block, *([actions] if actions is not None else [])]


def _prepare_cancellation_outcome(
    candidate: dict[str, str],
    *,
    session_id: str,
    language: str,
    store: SessionStore,
    discovered: bool,
) -> ConversationOutcome:
    intent = prepare_cancel_exceptional_candidate(
        _candidate_as_resourceplus_row(candidate),
        response_language=language,
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
    return ConversationOutcome(
        success=True,
        message=intent.summary,
        speech_message=intent.summary,
        tools_used=[
            *(["get_exceptional_entries"] if discovered else []),
            "prepare_cancel_exceptional_entry",
        ],
        requires_confirmation=True,
        confirmation_id=pending.confirmation_id,
        blocks=[confirmation_block(intent.summary, language)],
    )


async def continue_conversation(
    message: str,
    *,
    lang: int,
    session_id: str,
    language: str,
    store: SessionStore,
) -> ConversationOutcome | None:
    draft = store.get_conversation_draft(session_id)
    if draft is not None and _CASUAL.match(message.strip()):
        reply = "على الرحب والسعة." if language == "ar" else "You're welcome."
        return ConversationOutcome(success=True, message=reply, speech_message=reply)

    opaque_selection = _OPAQUE_CANDIDATE_SELECTION.match(message)
    if opaque_selection is not None:
        active_candidates = (
            _stored_cancellation_candidates(draft.slots)
            if draft is not None and draft.intent == "cancel_exceptional_entry"
            else []
        )
        active_keys = {candidate["key"].casefold() for candidate in active_candidates}
        if opaque_selection.group(1).casefold() not in active_keys:
            stale_message = (
                "هذا الاختيار لم يعد نشطاً. افتح طلبات الإدخال الاستثنائي المعلقة مرة أخرى."
                if language == "ar"
                else (
                    "That selection is no longer active. Please open your pending "
                    "exceptional entries again."
                )
            )
            return ConversationOutcome(True, stale_message, stale_message)

    missing_request = _is_missing_request(message)
    leave_request = _is_leave_request(message)
    less_hours_request = is_less_hours_request(message)
    less_hours_correction = is_less_hours_correction_request(message)
    cancel_exception_request = _is_cancel_exception_request(message)
    trusted = store.get_trusted_result(session_id)
    contextual_date = (
        _contextual_correction_date(
            message,
            draft=draft,
            trusted=trusted,
        )
        if less_hours_correction
        or (draft is not None and draft.intent == "less_hours_correction")
        else None
    )
    viewing_booking_options = bool(
        draft is not None
        and draft.intent == "book_day_type"
        and _arabic_normalized(message) in {
            "show available day types",
            "view leave business travel options",
            "show leave business travel options",
            "عرض خيارات الاجازة ومهمة العمل",
            "اعرض خيارات الاجازة ومهمة العمل",
        }
    )
    continues_booking_draft = bool(
        draft is not None
        and draft.intent == "book_day_type"
        and leave_request
    )
    if (
        draft is not None
        and not viewing_booking_options
        and not continues_booking_draft
        and _is_clear_independent_hr_intent(message)
    ):
        store.clear_conversation_draft(session_id)
        draft = None

    supported_drafts = {
        "correct_missing_punch",
        "book_day_type",
        "less_hours_correction",
        "cancel_exceptional_entry",
    }
    intent = draft.intent if draft is not None and draft.intent in supported_drafts else (
        "less_hours_correction"
        if missing_request or less_hours_correction or contextual_date is not None
        else "cancel_exceptional_entry"
        if cancel_exception_request
        else "book_day_type"
        if leave_request
        else None
    )
    if intent is None:
        if not less_hours_request:
            return None

        current = resourceplus_today()
        requested_range = _less_hours_range(message)
        if requested_range is None and trusted is not None:
            remembered_period = getattr(trusted, "attendance_period", None)
            if isinstance(remembered_period, tuple) and len(remembered_period) == 2:
                try:
                    requested_range = tuple(
                        date.fromisoformat(value) for value in remembered_period
                    )
                except (TypeError, ValueError):
                    requested_range = None
        start, end = requested_range or (current.replace(day=1), current)

        inspection = await inspect_less_hours_period(start, end, lang=lang)
        resolved_days = inspection.resolved_days
        candidates = tuple(
            resolved.day
            for resolved in resolved_days
            if resolved.correction_available
        )
        displayed_days = tuple(
            resolved
            for resolved in resolved_days
            if resolved.correction_available
            or resolved.state in {
                "approved_exception", "existing_request", "correction_unavailable"
            }
        )
        if not displayed_days and len(resolved_days) == 1:
            displayed_days = resolved_days
        block = less_hours_block(displayed_days, language)
        blocks: list[ResponseBlock] = [block]
        tools_used = ["get_attendance_summary", "get_exceptional_entry_requests"]
        if len(candidates) == 1:
            tools_used.append("get_exceptional_entry_balance")
            try:
                balance = await get_exceptional_entry_balance(
                    candidates[0].attendance_date
                )
            except ResourcePlusError:
                balance = None
            balance_ui = exceptional_balance_block(balance, language)
            if balance_ui is not None:
                blocks.append(balance_ui)
        actions = less_hours_correction_actions(displayed_days, language)
        if actions is not None:
            blocks.append(actions)
        message_text, speech_text = _less_hours_read_copy(resolved_days, language)
        return ConversationOutcome(
            True,
            message_text,
            speech_text,
            tools_used=tools_used,
            blocks=blocks,
        )
    slots = dict(draft.slots) if draft is not None else {}
    # Drafts retain semantic slots only; the current turn decides presentation.
    flow_language = language

    selected_date = contextual_date or extract_date(message)
    if intent == "less_hours_correction":
        entry_type = _explicit_entry_type(message)
        minutes = _explicit_minutes(message)
        if entry_type is not None:
            slots["entry_type"] = str(entry_type)
        if minutes is not None:
            slots["minutes"] = str(minutes)

        stored_options = [
            value for value in slots.get("reason_options", "").split("\x1f") if value
        ]
        if draft is not None and "date" in slots and stored_options:
            matched_reason = deterministic_reason_match(message, stored_options)
            if matched_reason.index is not None:
                result = await execute_tool(
                    "prepare_less_hours_correction",
                    {
                        "target_date": slots["date"],
                        "reason_name": stored_options[matched_reason.index],
                        "remarks": message,
                        "entry_type": int(slots["entry_type"]) if "entry_type" in slots else None,
                        "minutes": int(slots["minutes"]) if "minutes" in slots else None,
                    },
                    lang=lang,
                    session_id=session_id,
                    response_language=flow_language,
                    source_user_message=message,
                    trusted_conversation_intent=True,
                    store=store,
                )
                store.clear_conversation_draft(session_id)
                return _outcome_from_tool(
                    result,
                    flow_language,
                    "prepare_less_hours_correction",
                )
            prompt = (
                "اختر سبباً من الخيارات المتاحة."
                if flow_language == "ar"
                else "I couldn't match that reason. Please choose one of the available reasons."
            )
            return ConversationOutcome(
                success=True,
                message=prompt,
                speech_message=prompt,
                needs_reason=True,
                reason_options=stored_options,
                blocks=[reason_actions(stored_options, flow_language)],
            )

        requested_range = (
            (contextual_date, contextual_date)
            if contextual_date is not None
            else _less_hours_range(message)
        )
        if requested_range is None and "range_from" in slots:
            requested_range = (
                date.fromisoformat(slots["range_from"]),
                date.fromisoformat(slots["range_to"]),
            )
        if requested_range is None:
            contextual_candidates = _candidate_dates_from_context(draft, trusted)
            if contextual_candidates:
                slots["candidate_dates"] = "\x1f".join(
                    value.isoformat() for value in contextual_candidates
                )
                slots["range_from"] = min(contextual_candidates).isoformat()
                slots["range_to"] = max(contextual_candidates).isoformat()
                store.save_conversation_draft(
                    session_id,
                    intent=intent,
                    slots=slots,
                    validated_slots=("candidate_dates",),
                    language=flow_language,
                )
                choices = _format_correction_choices(
                    contextual_candidates,
                    flow_language,
                )
                question = (
                    f"أكيد — التواريخ التي لا تزال تحتاج تصحيح هي {choices}. أي واحد؟"
                    if flow_language == "ar"
                    else f"Sure — {choices} still need attention. Which one?"
                )
                return ConversationOutcome(True, question, question)
            no_gap_message = (
                _verified_no_actionable_gap_message(trusted, flow_language)
                if draft is None
                else None
            )
            if no_gap_message is not None:
                return ConversationOutcome(True, no_gap_message, no_gap_message)
            store.save_conversation_draft(
                session_id,
                intent=intent,
                slots=slots,
                language=flow_language,
            )
            question = (
                "ما تاريخ الساعات الناقصة؟"
                if flow_language == "ar"
                else "Which date has the missing hours?"
            )
            return ConversationOutcome(True, question, question)
        start, end = requested_range
        slots["range_from"] = start.isoformat()
        slots["range_to"] = end.isoformat()

        inspection = await inspect_less_hours_period(start, end, lang=lang)
        if len(inspection.eligible_days) > 1:
            slots["candidate_dates"] = "\x1f".join(
                day.attendance_date.isoformat() for day in inspection.eligible_days
            )
            store.save_conversation_draft(
                session_id,
                intent=intent,
                slots=slots,
                validated_slots=("candidate_dates",),
                language=flow_language,
            )
            available_days = tuple(
                day for day in inspection.resolved_days if day.correction_available
            )
            block = less_hours_block(available_days, flow_language)
            actions = less_hours_correction_actions(available_days, flow_language)
            question = (
                "وجدت أكثر من تاريخ مرشح للتصحيح. أي تاريخ تريد تصحيحه؟"
                if flow_language == "ar"
                else "I found more than one correction candidate. Which date would you like to correct?"
            )
            return ConversationOutcome(
                True,
                question,
                question,
                tools_used=[
                    "get_attendance_summary",
                    "get_exceptional_entry_requests",
                ],
                blocks=[block, *([actions] if actions is not None else [])],
            )
        if not inspection.eligible_days:
            if not inspection.days:
                message_text = (
                    "ما لقيت سجل حضور لهذا التاريخ."
                    if flow_language == "ar"
                    else "I couldn't find an attendance entry for that date."
                )
                return ConversationOutcome(
                    False,
                    message_text,
                    message_text,
                    tools_used=[
                        "get_attendance_summary",
                        "get_exceptional_entry_requests",
                    ],
                )
            selected = inspection.days[0]
            message_text = less_hours_duplicate_guard_message(
                inspection,
                selected.attendance_date,
                flow_language,
            ) or _less_hours_message(selected, flow_language)
            if selected.eligibility == "no_punches":
                store.save_conversation_draft(
                    session_id,
                    intent="book_day_type",
                    slots={
                        "date_from": selected.attendance_date.isoformat(),
                        "date_to": selected.attendance_date.isoformat(),
                        "state": "awaiting_day_type",
                    },
                    language=flow_language,
                )
            return ConversationOutcome(
                True,
                message_text,
                message_text,
                tools_used=[
                    "get_attendance_summary",
                    "get_exceptional_entry_requests",
                ],
                blocks=[less_hours_block(inspection.resolved_days, flow_language)],
            )

        selected = inspection.eligible_days[0]
        async def _balance() -> Any:
            try:
                return await get_exceptional_entry_balance(selected.attendance_date)
            except ResourcePlusError:
                return None

        balance, reasons_payload = await asyncio.gather(
            _balance(),
            cached_exception_reasons(lang),
        )
        reason_options = _reason_names(reasons_payload)
        slots["date"] = selected.attendance_date.isoformat()
        slots["less_hours"] = selected.less_hours
        slots["reason_options"] = "\x1f".join(reason_options)
        store.save_conversation_draft(
            session_id,
            intent=intent,
            slots=slots,
            validated_slots=("date",),
            language=flow_language,
        )
        blocks: list[ResponseBlock] = [
            less_hours_block(
                [day for day in inspection.resolved_days if day.day == selected],
                flow_language,
            )
        ]
        balance_ui = exceptional_balance_block(balance, flow_language)
        if balance_ui is not None:
            blocks.append(balance_ui)
        if reason_options:
            blocks.append(reason_actions(reason_options, flow_language))
        requested_entry_type = (
            int(slots["entry_type"]) if "entry_type" in slots else None
        )
        question = _correction_reason_prompt(
            selected,
            requested_entry_type,
            flow_language,
        )
        return ConversationOutcome(
            True,
            question,
            question,
            tools_used=[
                "get_attendance_summary",
                "get_exceptional_entry_requests",
                "get_exceptional_entry_balance",
                "get_exception_reasons",
            ],
            needs_reason=True,
            reason_options=reason_options or None,
            blocks=blocks,
        )

    if intent == "cancel_exceptional_entry":
        stored_candidates = _stored_cancellation_candidates(slots)
        if stored_candidates:
            matches = _match_cancellation_candidates(message, stored_candidates)
            if len(matches) == 1:
                store.clear_conversation_draft(session_id)
                return _prepare_cancellation_outcome(
                    matches[0],
                    session_id=session_id,
                    language=flow_language,
                    store=store,
                    discovered=False,
                )
            displayed = matches if len(matches) > 1 else stored_candidates
            blocks = _cancellation_selection_blocks(displayed, flow_language)
            question = (
                "حدّد الطلب باستخدام التاريخ أو النوع أو السبب أو الرقم."
                if flow_language == "ar"
                else (
                    "That still matches more than one request. Please use the date, type, reason, or number."
                    if len(matches) > 1
                    else "I couldn't match that selection. Please use the date, type, reason, or number."
                )
            )
            return ConversationOutcome(
                True,
                question,
                question,
                blocks=blocks,
            )

        current = resourceplus_today()
        requested_range = _less_hours_range(message)
        if requested_range is None and "range_from" in slots:
            requested_range = (
                date.fromisoformat(slots["range_from"]),
                date.fromisoformat(slots["range_to"]),
            )
        if requested_range is None:
            requested_range = current.replace(day=1), current
        start, end = requested_range
        rows = await find_pending_exceptional_entries(start, end, lang=lang)
        if selected_date is not None:
            selected_iso = selected_date.isoformat()
            rows = [
                row
                for row in rows
                if exceptional_entry_attributes(row)["date"] == selected_iso
            ]
        if not rows:
            store.clear_conversation_draft(session_id)
            message_text = (
                "ما عندك طلبات تصحيح حضور تقدر تلغيها خلال هالفترة."
                if flow_language == "ar"
                else "You don't have any attendance correction requests to cancel this period."
            )
            return ConversationOutcome(True, message_text, message_text, tools_used=["get_exceptional_entries"])
        candidates = [_cancellation_candidate(row) for row in rows]
        if len(candidates) > 1:
            slots.update({"range_from": start.isoformat(), "range_to": end.isoformat()})
            slots["cancellation_candidates"] = json.dumps(candidates, ensure_ascii=False)
            store.save_conversation_draft(
                session_id,
                intent=intent,
                slots=slots,
                validated_slots=("range_from", "range_to"),
                language=flow_language,
            )
            blocks = _cancellation_selection_blocks(candidates, flow_language)
            block = blocks[0]
            question = (
                "وجدت أكثر من طلب قابل للإلغاء. أي طلب تقصد؟"
                if flow_language == "ar"
                else "I found more than one cancellable request. Which one do you mean?"
            )
            return ConversationOutcome(
                True,
                question,
                question,
                tools_used=["get_exceptional_entries"],
                blocks=blocks,
            )
        store.clear_conversation_draft(session_id)
        return _prepare_cancellation_outcome(
            candidates[0],
            session_id=session_id,
            language=flow_language,
            store=store,
            discovered=True,
        )

    if intent == "correct_missing_punch":
        direction = extract_direction(message)
        if selected_date is not None:
            slots["date"] = selected_date.isoformat()
        if direction is not None:
            slots["direction"] = direction
        if "date" not in slots or "direction" not in slots:
            store.save_conversation_draft(
                session_id,
                intent=intent,
                slots=slots,
                language=flow_language,
            )
            question = _question(intent, slots, flow_language)
            return ConversationOutcome(success=True, message=question, speech_message=question)
        result = await execute_tool(
            "prepare_less_hours_correction",
            {
                "target_date": slots["date"],
                "entry_type": 1 if slots["direction"] == "IN" else 2,
                "minutes": None,
                "reason_name": None,
                "remarks": None,
            },
            lang=lang,
            session_id=session_id,
            response_language=flow_language,
            source_user_message=message,
            trusted_conversation_intent=True,
            store=store,
        )
        store.clear_conversation_draft(session_id)
        return _outcome_from_tool(
            result,
            flow_language,
            "prepare_less_hours_correction",
        )

    if selected_date is not None:
        slots["date_from"] = selected_date.isoformat()
        slots["date_to"] = selected_date.isoformat()
    if draft is None:
        requested_group = _booking_group(message)
        if requested_group is not None:
            slots["booking_group"] = requested_group
    requested_group = slots.get("booking_group")
    if requested_group not in {None, "leave", "business_travel"}:
        requested_group = None
    stored_rows = _stored_day_type_options(slots) if draft is not None else []
    live_rows = (
        _bookable_day_types(stored_rows, requested_group)
        if stored_rows
        else _bookable_day_types(
            _day_type_rows(
                await cached_day_types(lang, force_refresh=draft is None)
            ),
            requested_group,
        )
    )
    generic_leave_intent = _is_generic_leave_intent(message, live_rows)
    match_status, matched_day_type = (
        ("unknown", None)
        if viewing_booking_options or generic_leave_intent
        else _match_day_type(message, live_rows)
    )
    if match_status == "matched" and matched_day_type is not None:
        slots["day_type"] = matched_day_type
    if "date_from" not in slots or "day_type" not in slots:
        if "date_from" in slots:
            slots["state"] = "awaiting_day_type"
        else:
            slots["state"] = "awaiting_date"
        if live_rows:
            slots["day_type_options"] = json.dumps(
                [
                    {"dayType": row["dayType"], "group": row.get("group", "")}
                    for row in live_rows
                ],
                ensure_ascii=False,
            )
        store.save_conversation_draft(
            session_id,
            intent=intent,
            slots=slots,
            validated_slots=("day_type",) if "day_type" in slots else (),
            language=flow_language,
        )
        if "date_from" not in slots:
            question = _question(intent, slots, flow_language)
            return ConversationOutcome(success=True, message=question, speech_message=question)
        target = date.fromisoformat(slots["date_from"])
        displayed_date = _display_booking_date(target, flow_language)
        if not live_rows:
            question = (
                "لم أجد أنواع إجازة أو مهمة عمل متاحة حاليًا في ResourcePlus."
                if flow_language == "ar" else
                "I couldn't find available Leave or Business Travel types in ResourcePlus right now."
            )
        elif match_status == "ambiguous":
            question = (
                f"وجدت أكثر من نوع مطابق ليوم {displayed_date}. أي نوع تقصد؟"
                if flow_language == "ar" else
                f"More than one day type matches for {displayed_date}. Which one do you mean?"
            )
        elif match_status == "suggested" and matched_day_type is not None:
            question = (
                f"هل تقصد {matched_day_type} ليوم {displayed_date}؟"
                if flow_language == "ar" else
                f"Did you mean {matched_day_type} for {displayed_date}?"
            )
        elif draft is not None and not viewing_booking_options and not generic_leave_intent:
            question = (
                f"لم أجد هذا النوع ضمن خيارات الإجازة أو مهمة العمل ليوم {displayed_date}. اختر نوعًا من الخيارات المتاحة."
                if flow_language == "ar" else
                f"I couldn't match that day type for {displayed_date}. Please choose one of the available options."
            )
        else:
            if flow_language == "ar":
                label = (
                    "الإجازة" if requested_group == "leave" else
                    "مهمة العمل" if requested_group == "business_travel" else
                    "الإجازة أو مهمة العمل"
                )
                question = f"أكيد — {displayed_date}. ما نوع {label} الذي تريد استخدامه؟"
            else:
                label = (
                    "leave" if requested_group == "leave" else
                    "business travel" if requested_group == "business_travel" else
                    "leave or business travel"
                )
                question = f"Sure — {displayed_date}. Which {label} type would you like to use?"
        return ConversationOutcome(
            success=True,
            message=question,
            speech_message=question,
            tools_used=[] if stored_rows else ["get_day_types"],
            blocks=_day_type_choice_blocks(live_rows, flow_language, requested_group),
        )
    result = await execute_tool(
        "prepare_book_day_type",
        {
            "date_from": slots["date_from"],
            "date_to": slots.get("date_to", slots["date_from"]),
            "day_type_name": slots["day_type"],
            "_booking_group": requested_group or "leave_or_travel",
        },
        lang=lang,
        session_id=session_id,
        response_language=flow_language,
        source_user_message=message,
        trusted_conversation_intent=True,
        store=store,
    )
    store.clear_conversation_draft(session_id)
    return _outcome_from_tool(result, flow_language, "prepare_book_day_type")
