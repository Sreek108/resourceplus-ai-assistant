from __future__ import annotations

import json
import asyncio
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any
from uuid import uuid4

from app.ai.actions import (
    LessHoursDay,
    exceptional_entry_attributes,
    find_pending_exceptional_entries,
    inspect_less_hours_period,
    prepare_cancel_exceptional_candidate,
)
from app.ai.reason_matcher import deterministic_reason_match
from app.ai.sessions import SessionStore
from app.ai.tools import (
    ToolExecutionResult,
    execute_tool,
    resolve_relative_date_range,
)
from app.models.schemas import ResponseBlock
from app.audit import record_action_state
from app.resourceplus import ResourcePlusError
from app.resourceplus.exceptional import get_exceptional_entry_balance
from app.resourceplus.reference_cache import cached_day_types, cached_exception_reasons
from app.services.fast_reads import classify_fast_read
from app.services.response_blocks import (
    cancellable_exception_actions,
    cancellable_exceptions_block,
    confirmation_block,
    exceptional_balance_block,
    less_hours_block,
    less_hours_correction_actions,
    markdown_table,
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
_LEAVE_BOOKING = re.compile(
    r"\b(?:need|want|book|take|apply\s+for|request)\b.{0,60}"
    r"\b(?:leave|vacation|day\s+off|business\s+travel)\b",
    re.I,
)
_LESS_HOURS = re.compile(
    r"\b(?:less[-\s]+hours?|short\s+hours?|shortfall|late\s+(?:arrival|attendance)|early\s+departure)\b",
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
_ARABIC_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u0640]")


def _normalized(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", value.casefold(), flags=re.UNICODE).split())


def _arabic_normalized(value: str) -> str:
    normalized = _ARABIC_DIACRITICS.sub("", value.casefold())
    normalized = normalized.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}))
    return " ".join(re.sub(r"[^\w\s]", " ", normalized, flags=re.UNICODE).split())


def _is_missing_request(message: str) -> bool:
    normalized = _normalized(message)
    return bool(_MISSING_CORRECTION.search(message)) or (
        any(cue in normalized for cue in ("صحح", "تصحيح", "تعديل"))
        and "بصم" in normalized
    )


def _is_leave_request(message: str) -> bool:
    normalized = _normalized(message)
    arabic_booking = (
        any(cue in normalized for cue in ("أبغى", "ابغى", "أريد", "اريد", "طلب"))
        and any(cue in normalized for cue in ("إجاز", "اجاز", "سفر"))
        and not any(cue in normalized for cue in ("رصيد", "أعرف", "اعرف", "متبقي"))
    )
    return bool(_LEAVE_BOOKING.search(message)) or arabic_booking


def is_less_hours_request(message: str) -> bool:
    normalized = _arabic_normalized(message)
    return bool(_LESS_HOURS.search(message)) or any(
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
        )
    )


def is_less_hours_correction_request(message: str) -> bool:
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
    if not is_less_hours_request(message) or is_less_hours_correction_request(message):
        return False
    normalized = _arabic_normalized(message)
    return bool(
        re.search(r"\b(?:show|list|display|view|what|which)\b", normalized)
        or any(cue in normalized for cue in ("اعرض", "اظهر", "ارني", "ورني", "ما هي", "ماهي"))
    )


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
    if re.search(r"\btomorrow\b", normalized) or any(cue in normalized for cue in ("غدا", "غداً", "بكرة")):
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


def _explicit_entry_type(message: str) -> int | None:
    normalized = _arabic_normalized(message)
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
    if day.eligibility == "no_punches":
        return (
            "لا توجد بصمات حضور لهذا اليوم، لذلك لا يمكن تصحيحه كساعات ناقصة. يمكنني مساعدتك في طلب إجازة أو مهمة عمل بدلاً من ذلك."
            if language == "ar"
            else "You don't have attendance punches for that day, so it can't be corrected as a less-hours entry. I can help you apply for Leave or Business Travel instead."
        )
    if day.eligibility == "no_missing_hours":
        return (
            "لا توجد لديك ساعات ناقصة لتصحيحها في هذا اليوم."
            if language == "ar"
            else "You have no missing hours to correct for that day."
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
            f"يصنّف ResourcePlus هذا اليوم على أنه {label}، لذلك لا ينطبق عليه "
            "تصحيح الإدخال الاستثنائي."
        )
    return (
        f"ResourcePlus classifies that day as {label}, so an exceptional-entry correction does not apply."
    )


def _day_type_rows(payload: Any) -> list[dict[str, Any]]:
    return [row for row in payload if isinstance(row, dict)] if isinstance(payload, list) else []


async def _match_day_type(message: str, lang: int) -> str | None:
    rows = _day_type_rows(await cached_day_types(lang))
    normalized = _normalized(message)
    exact = [
        str(row["dayType"])
        for row in rows
        if isinstance(row.get("dayType"), str)
        and _normalized(str(row["dayType"])) in normalized
    ]
    if len(exact) == 1:
        return exact[0]
    message_words = set(normalized.split()) - {
        "i", "need", "want", "book", "take", "apply", "for", "request", "on", "from", "to",
        "today", "tomorrow", "leave", "vacation", "day", "off",
    }
    scored: list[tuple[int, str]] = []
    for row in rows:
        name = row.get("dayType")
        if isinstance(name, str):
            score = len(message_words & set(_normalized(name).split()))
            if score:
                scored.append((score, name))
    if not scored:
        return None
    scored.sort(reverse=True)
    return scored[0][1] if len(scored) == 1 or scored[0][0] > scored[1][0] else None


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
        return ConversationOutcome(
            success=True,
            message=summary,
            speech_message=summary,
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

    # Named natural-language dates remain on the semantic model path on the first
    # turn. The model still calls the v2 preparation tool, while the deterministic
    # path handles the common today/yesterday/this-week and numeric/ISO forms.
    # This also preserves the established API dependency-injection contract.
    if (
        draft is None
        and is_less_hours_correction_request(message)
        and re.search(r"\b(?:" + "|".join(_MONTHS) + r")\s+\d{1,2}\b", _normalized(message))
    ):
        return None

    missing_request = _is_missing_request(message)
    leave_request = _is_leave_request(message)
    less_hours_request = is_less_hours_request(message)
    less_hours_correction = is_less_hours_correction_request(message)
    cancel_exception_request = _is_cancel_exception_request(message)
    if draft is not None and _is_clear_independent_hr_intent(message):
        store.clear_conversation_draft(session_id)
        draft = None

    supported_drafts = {
        "correct_missing_punch",
        "book_day_type",
        "less_hours_correction",
        "cancel_exceptional_entry",
    }
    intent = draft.intent if draft is not None and draft.intent in supported_drafts else (
        "correct_missing_punch"
        if missing_request
        else "less_hours_correction"
        if less_hours_correction
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
        start, end = requested_range or (current.replace(day=1), current)

        inspection = await inspect_less_hours_period(start, end, lang=lang)
        candidates = inspection.eligible_days
        block = less_hours_block(candidates, language)
        blocks: list[ResponseBlock] = [block]
        tools_used = ["get_attendance_summary"]
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
        actions = less_hours_correction_actions(candidates, language)
        if actions is not None:
            blocks.append(actions)
        count = len(candidates)
        if language == "ar":
            message_text = (
                "لا توجد حالات لساعات ناقصة خلال هذه الفترة."
                if count == 0
                else "لديك حالة واحدة لساعات ناقصة خلال هذه الفترة."
                if count == 1
                else f"لديك {count} حالات لساعات ناقصة خلال هذه الفترة."
            )
        else:
            amount = "no" if count == 0 else str(count)
            entry = "entry" if count == 1 else "entries"
            message_text = (
                f"You have {amount} less-hours {entry} eligible for correction "
                "during this period."
            )
        return ConversationOutcome(
            True,
            message_text,
            message_text,
            tools_used=tools_used,
            blocks=blocks,
        )
    slots = dict(draft.slots) if draft is not None else {}
    flow_language = draft.language if draft is not None else language

    selected_date = extract_date(message)
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

        requested_range = _less_hours_range(message)
        if requested_range is None and "range_from" in slots:
            requested_range = (
                date.fromisoformat(slots["range_from"]),
                date.fromisoformat(slots["range_to"]),
            )
        if requested_range is None:
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
            block = less_hours_block(inspection.eligible_days, flow_language)
            actions = less_hours_correction_actions(inspection.eligible_days, flow_language)
            question = (
                "وجدت أكثر من تاريخ مرشح للتصحيح. أي تاريخ تريد تصحيحه؟"
                if flow_language == "ar"
                else "I found more than one correction candidate. Which date would you like to correct?"
            )
            return ConversationOutcome(
                True,
                question + "\n\n" + markdown_table(block),
                question,
                tools_used=["get_attendance_summary"],
                blocks=[block, *([actions] if actions is not None else [])],
            )
        if not inspection.eligible_days:
            if not inspection.days:
                message_text = (
                    "لم يُرجع ResourcePlus سجل حضور لهذا التاريخ."
                    if flow_language == "ar"
                    else "ResourcePlus returned no attendance row for that date."
                )
                return ConversationOutcome(False, message_text, message_text, tools_used=["get_attendance_summary"])
            selected = inspection.days[0]
            message_text = _less_hours_message(selected, flow_language)
            if selected.eligibility == "no_punches":
                store.save_conversation_draft(
                    session_id,
                    intent="book_day_type",
                    slots={
                        "date_from": selected.attendance_date.isoformat(),
                        "date_to": selected.attendance_date.isoformat(),
                    },
                    language=flow_language,
                )
            return ConversationOutcome(
                True,
                message_text,
                message_text,
                tools_used=["get_attendance_summary"],
                blocks=[less_hours_block(inspection.days, flow_language)],
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
        blocks: list[ResponseBlock] = [less_hours_block([selected], flow_language)]
        balance_ui = exceptional_balance_block(balance, flow_language)
        if balance_ui is not None:
            blocks.append(balance_ui)
        if reason_options:
            blocks.append(reason_actions(reason_options, flow_language))
        question = "ما سبب التصحيح؟" if flow_language == "ar" else "What was the reason?"
        return ConversationOutcome(
            True,
            question,
            question,
            tools_used=[
                "get_attendance_summary",
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
                question + "\n\n" + markdown_table(blocks[0]),
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
                "لا يوجد لديك طلب إدخال استثنائي معلّق وقابل للإلغاء خلال هذه الفترة."
                if flow_language == "ar"
                else "You have no cancellable pending exceptional entry for this period."
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
                question + "\n\n" + markdown_table(block),
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
            "prepare_exceptional_entry",
            {
                "target_date": slots["date"],
                "punch_direction": slots["direction"],
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
        return _outcome_from_tool(result, flow_language, "prepare_exceptional_entry")

    if selected_date is not None:
        slots["date_from"] = selected_date.isoformat()
        slots["date_to"] = selected_date.isoformat()
    matched_day_type = await _match_day_type(message, lang)
    if matched_day_type is not None:
        slots["day_type"] = matched_day_type
    if "date_from" not in slots or "day_type" not in slots:
        store.save_conversation_draft(
            session_id,
            intent=intent,
            slots=slots,
            validated_slots=("day_type",) if "day_type" in slots else (),
            language=flow_language,
        )
        question = _question(intent, slots, flow_language)
        return ConversationOutcome(success=True, message=question, speech_message=question)
    result = await execute_tool(
        "prepare_book_day_type",
        {
            "date_from": slots["date_from"],
            "date_to": slots.get("date_to", slots["date_from"]),
            "day_type_name": slots["day_type"],
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
