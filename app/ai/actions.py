import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from app.ai.reason_matcher import deterministic_reason_match, match_live_reason
from app.resourceplus import ResourcePlusError
from app.resourceplus.approvals import (
    approve_all_requests,
    approve_supervisor_request,
    get_pending_approvals,
)
from app.resourceplus.attendance import (
    create_exceptional_entry,
    get_attendance_summary,
    get_exception_reasons,
    get_missing_punch_suggestions,
)
from app.resourceplus.exceptional import (
    cancel_exceptional_entry,
    create_exceptional_entry_from_summary,
)
from app.resourceplus.leave import (
    book_day_type,
    cancel_day_type_request,
    get_day_types,
    get_my_day_type_requests,
)
from app.resourceplus.notifications import (
    get_notifications,
    update_notification_read_status,
)
from app.resourceplus.missing_punch import (
    ENTRY_TYPE_NUMBER,
    MissingPunchRow,
    normalize_missing_punch_suggestions,
    parse_missing_punch_date,
)
from app.resourceplus.requests import get_exceptional_entry_requests
from app.time_context import resourceplus_today


@dataclass(frozen=True)
class ActionIntent:
    action_type: str
    validated_arguments: dict[str, object]
    summary: str
    language: str


@dataclass(frozen=True)
class LessHoursDay:
    attendance_date: date
    day_type: str
    check_in: str | None
    check_out: str | None
    worked_hours: str
    less_hours: str
    eligibility: str
    raw: dict[str, Any]


@dataclass(frozen=True)
class LessHoursInspection:
    days: tuple[LessHoursDay, ...]
    eligible_days: tuple[LessHoursDay, ...]
    duplicate_guards: tuple["ExceptionalEntryGuard", ...] = ()
    exceptional_entries_checked: bool = True

    @property
    def resolved_days(self) -> tuple["ResolvedLessHoursDay", ...]:
        """Resolve display status and actionability from one per-date state."""

        eligible_dates = {day.attendance_date for day in self.eligible_days}
        guards_by_date: dict[date, ExceptionalEntryGuard] = {}
        for guard in self.duplicate_guards:
            current = guards_by_date.get(guard.attendance_date)
            if current is None or guard.classification == "approved":
                guards_by_date[guard.attendance_date] = guard

        resolved: list[ResolvedLessHoursDay] = []
        for day in self.days:
            state = day.eligibility
            if day.eligibility == "eligible":
                guard = guards_by_date.get(day.attendance_date)
                if guard is not None and guard.classification == "approved":
                    state = "approved_exception"
                elif guard is not None:
                    state = "existing_request"
                elif (
                    self.exceptional_entries_checked
                    and day.attendance_date in eligible_dates
                ):
                    state = "correction_available"
                else:
                    state = "correction_unavailable"
            resolved.append(
                ResolvedLessHoursDay(
                    day=day,
                    state=state,
                    correction_available=state == "correction_available",
                )
            )
        return tuple(resolved)


@dataclass(frozen=True)
class ExceptionalEntryGuard:
    attendance_date: date
    raw_status: str
    classification: str


@dataclass(frozen=True)
class ResolvedLessHoursDay:
    day: LessHoursDay
    state: str
    correction_available: bool


class ActionResolutionRequired(Exception):
    """A safe employee clarification or business limitation, not a system failure."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        requires_clarification: bool = False,
        draft_context: dict[str, str] | None = None,
        reason_options: list[str] | None = None,
    ) -> None:
        super().__init__(message)
        self.category = category
        self.requires_clarification = requires_clarification
        self.draft_context = draft_context
        self.reason_options = reason_options


def _require_list(value: Any, source: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise ValueError(f"ResourcePlus returned an unexpected {source} response.")
    return value


def _parse_resourceplus_date(value: object) -> date:
    if not isinstance(value, str):
        raise ValueError("ResourcePlus returned an invalid date.")
    normalized = value.strip()
    try:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    for date_format in (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d/%m/%Y %H:%M",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(normalized, date_format).date()
        except ValueError:
            continue
    raise ValueError("ResourcePlus returned an invalid date.")


def _parse_suggested_datetime(value: object) -> datetime:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}",
        value,
    ):
        raise ValueError("ResourcePlus returned an invalid suggested entry time.")
    try:
        parsed = datetime.strptime(value, "%d/%m/%Y %H:%M")
    except ValueError as exc:
        raise ValueError(
            "ResourcePlus returned an invalid suggested entry time."
        ) from exc
    if parsed.strftime("%d/%m/%Y %H:%M") != value:
        raise ValueError("ResourcePlus returned an invalid suggested entry time.")
    return parsed


def _same_text(left: object, right: object) -> bool:
    return isinstance(left, str) and isinstance(right, str) and (
        left.strip().casefold() == right.strip().casefold()
    )


def _normalized_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def bookable_day_type_group(value: object) -> str | None:
    """Classify only ResourcePlus groups supported by leave/travel booking."""

    normalized = _normalized_key(value)
    if normalized in {"leave", "إجازة", "اجازة", "الإجازة", "الاجازة"}:
        return "leave"
    if normalized in {
        "businesstravel", "travel", "مهمةعمل", "مهمةرسمية", "سفرعمل",
    }:
        return "business_travel"
    return None


def _normalized_words(value: object) -> tuple[str, ...]:
    if not isinstance(value, str):
        return ()
    return tuple(re.findall(r"[^\W_]+", value.casefold(), flags=re.UNICODE))


def _reason_was_supplied_in_user_message(reason: str, user_message: object) -> bool:
    """Require the model's reason text to be grounded in the current user turn."""

    if user_message is None:
        return False
    reason_words = _normalized_words(reason)
    message_words = _normalized_words(user_message)
    if not reason_words or len(reason_words) > len(message_words):
        return False
    width = len(reason_words)
    return any(
        message_words[index : index + width] == reason_words
        for index in range(len(message_words) - width + 1)
    )


def _field(item: dict[str, Any], *names: str) -> object:
    normalized = {_normalized_key(key): value for key, value in item.items()}
    for name in names:
        key = _normalized_key(name)
        if key in normalized:
            return normalized[key]
    return None


def _attendance_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if not isinstance(payload, dict):
        raise ValueError("ResourcePlus returned an unexpected attendance response.")
    normalized = {_normalized_key(key): value for key, value in payload.items()}
    for name in ("Days", "Attendance", "AttendanceDetails", "Details"):
        value = normalized.get(_normalized_key(name))
        if isinstance(value, list):
            return [row for row in value if isinstance(row, dict)]
    return []


def _duration_minutes(value: object) -> int:
    if not isinstance(value, str):
        return 0
    match = re.fullmatch(r"\s*(\d{1,3}):(\d{2})\s*", value)
    if match is None or int(match.group(2)) >= 60:
        return 0
    return int(match.group(1)) * 60 + int(match.group(2))


def _punch_value(row: dict[str, Any], *names: str) -> str | None:
    value = _field(row, *names)
    if not isinstance(value, str):
        return None
    normalized = value.strip()
    if not normalized or normalized.casefold() in {"-", "--", "n/a", "null", "00:00"}:
        return None
    return normalized


def _less_hours_eligibility(
    *,
    day_type: str,
    less_minutes: int,
    has_punch: bool,
) -> str:
    normalized = _normalized_key(day_type)
    if any(term in normalized for term in ("weekend", "weekoff", "weekendday")):
        return "week_end"
    if "holiday" in normalized:
        return "holiday"
    if "businesstravel" in normalized:
        return "business_travel"
    if "leave" in normalized:
        return "leave"
    if "absent" in normalized or not has_punch:
        return "no_punches"
    if less_minutes <= 0:
        return "no_missing_hours"
    return "eligible"


async def inspect_less_hours_period(
    date_from: date | str,
    date_to: date | str,
    *,
    lang: int,
) -> LessHoursInspection:
    start = _parse_resourceplus_date(str(date_from)) if not isinstance(date_from, date) else date_from
    end = _parse_resourceplus_date(str(date_to)) if not isinstance(date_to, date) else date_to
    if start > end:
        raise ValueError("date_from must be on or before date_to.")
    payload = await get_attendance_summary(start.isoformat(), end.isoformat(), lang=lang)
    inspection = classify_attendance_summary(payload, start, end)
    try:
        exceptional_payload = await get_exceptional_entry_requests(
            start,
            end,
            lang=lang,
        )
    except ResourcePlusError:
        # Keep the attendance rows visible, but fail closed for write
        # eligibility when the duplicate check is unavailable.
        return LessHoursInspection(
            days=inspection.days,
            eligible_days=(),
            exceptional_entries_checked=False,
        )

    guards = exceptional_entry_guards(exceptional_payload, start, end)
    guarded_dates = {guard.attendance_date for guard in guards}
    return LessHoursInspection(
        days=inspection.days,
        eligible_days=tuple(
            day
            for day in inspection.eligible_days
            if day.attendance_date not in guarded_dates
        ),
        duplicate_guards=guards,
    )


def classify_attendance_summary(
    payload: Any,
    date_from: date,
    date_to: date,
) -> LessHoursInspection:
    """Classify trusted AttendanceSummary rows without performing another read."""

    rows = _attendance_rows(payload)
    days: list[LessHoursDay] = []
    for row in rows:
        raw_date = _field(row, "AttDate", "attendanceDate", "date")
        try:
            attendance_date = _parse_resourceplus_date(raw_date)
        except ValueError:
            continue
        if not date_from <= attendance_date <= date_to:
            continue
        day_type_value = _field(row, "DayType", "dayType", "Status", "attendanceStatus")
        day_type = str(day_type_value).strip() if day_type_value not in (None, "") else "—"
        check_in = _punch_value(row, "IN", "InTime", "PunchIn", "FirstIn", "checkIn")
        check_out = _punch_value(row, "OUT", "OutTime", "PunchOut", "LastOut", "checkOut")
        worked = _field(row, "NetHrs", "WorkedHours", "Worked")
        less = _field(row, "LessHrs", "Shortfall")
        worked_hours = str(worked).strip() if worked not in (None, "") else "00:00"
        less_hours = str(less).strip() if less not in (None, "") else "00:00"
        eligibility = _less_hours_eligibility(
            day_type=day_type,
            less_minutes=_duration_minutes(less_hours),
            has_punch=check_in is not None or check_out is not None,
        )
        days.append(
            LessHoursDay(
                attendance_date=attendance_date,
                day_type=day_type,
                check_in=check_in,
                check_out=check_out,
                worked_hours=worked_hours,
                less_hours=less_hours,
                eligibility=eligibility,
                raw=dict(row),
            )
        )
    return LessHoursInspection(
        days=tuple(days),
        eligible_days=tuple(day for day in days if day.eligibility == "eligible"),
    )


def _attendance_snapshot(day: LessHoursDay) -> dict[str, object]:
    return {
        "day_type": day.day_type,
        "check_in": day.check_in,
        "check_out": day.check_out,
        "worked_hours": day.worked_hours,
        "less_hours": day.less_hours,
        "eligibility": day.eligibility,
    }


def _exceptional_request_rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        for key in ("entries", "requests", "data", "exceptionalEntries"):
            value = _field(payload, key)
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def exceptional_entry_guards(
    payload: Any,
    date_from: date,
    date_to: date,
) -> tuple[ExceptionalEntryGuard, ...]:
    """Identify only statuses that safely prevent offering a duplicate.

    ResourcePlus has not documented whether ``Not Approved`` means pending or
    rejected. It is retained verbatim and classified only as ambiguous.
    """

    guards: list[ExceptionalEntryGuard] = []
    for row in _exceptional_request_rows(payload):
        raw_date = _field(
            row,
            "entryTime",
            "attDate",
            "attendanceDate",
            "date",
            "requestDate",
        )
        try:
            attendance_date = _parse_resourceplus_date(raw_date)
        except ValueError:
            continue
        if not date_from <= attendance_date <= date_to:
            continue
        status_value = _field(row, "status", "requestStatus")
        if not isinstance(status_value, str) or not status_value.strip():
            continue
        raw_status = status_value.strip()
        normalized = _normalized_key(raw_status)
        if normalized == "approved":
            classification = "approved"
        elif normalized == "notapproved":
            classification = "ambiguous_not_approved"
        else:
            continue
        guards.append(
            ExceptionalEntryGuard(
                attendance_date=attendance_date,
                raw_status=raw_status,
                classification=classification,
            )
        )
    return tuple(guards)


def less_hours_duplicate_guard_message(
    inspection: LessHoursInspection,
    attendance_date: date,
    language: str,
) -> str | None:
    if not inspection.exceptional_entries_checked:
        if language == "ar":
            return (
                "ما قدرت أتحقق من طلبات التصحيح الحالية، لذلك ما راح أبدأ تصحيح "
                "جديد قد يكون مكرر."
            )
        return (
            "I couldn't check your existing correction requests, so I won't start "
            "another one that might be a duplicate."
        )
    matching_guards = [
        item for item in inspection.duplicate_guards
        if item.attendance_date == attendance_date
    ]
    guard = next(
        (item for item in matching_guards if item.classification == "approved"),
        matching_guards[0] if matching_guards else None,
    )
    if guard is None:
        return None
    displayed_date = attendance_date.isoformat()
    if guard.classification == "approved":
        if language == "ar":
            return (
                f"يوم {displayed_date} مغطى بالفعل بتصحيح حضور معتمد، "
                "لذلك ما فيه شيء آخر يحتاج إرسال."
            )
        return (
            f"{attendance_date.strftime('%b')} {attendance_date.day} is already "
            "covered by an approved attendance correction, so there's nothing else "
            "to submit."
        )
    if language == "ar":
        return (
            f"فيه طلب تصحيح موجود ليوم {displayed_date}، فما راح أبدأ طلب ثاني. "
            f"حالته الظاهرة «{guard.raw_status}» وما نقدر نعتبرها موافقة أو رفض."
        )
    return (
        f"There's already a correction request for {displayed_date}, so I won't "
        f'create another one. Its status is "{guard.raw_status}"; that does not '
        "tell us whether it is pending or rejected."
    )


def _is_cancellable_exception(row: dict[str, Any]) -> bool:
    cancellable = _field(row, "isCancellable", "canCancel")
    if isinstance(cancellable, bool):
        return cancellable
    status = _field(row, "status", "requestStatus")
    if not isinstance(status, str):
        return False
    return _normalized_key(status) in {
        "notapproved",
        "pending",
        "pendingapproval",
        "pendingforapproval",
        "submitted",
    }


async def find_pending_exceptional_entries(
    date_from: date | str,
    date_to: date | str,
    *,
    lang: int,
) -> list[dict[str, Any]]:
    rows = _exceptional_request_rows(
        await get_exceptional_entry_requests(date_from, date_to, lang=lang)
    )
    return [row for row in rows if _is_cancellable_exception(row)]


def _format_time(value: datetime) -> str:
    hour = value.hour % 12 or 12
    suffix = "AM" if value.hour < 12 else "PM"
    return f"{hour}:{value.minute:02d} {suffix}"


def _format_date(value: date) -> str:
    return value.strftime("%d %B %Y")


def exceptional_entry_display(row: dict[str, Any]) -> str:
    """Return a safe, employee-facing description without exposing its ID."""

    attributes = exceptional_entry_attributes(row)
    raw_date = attributes["date"]
    try:
        displayed_date = _format_date(_parse_resourceplus_date(raw_date))
    except ValueError:
        displayed_date = str(raw_date).strip() if raw_date not in (None, "") else "the selected date"
    reason = attributes["reason"]
    status = attributes["status"]
    details = [displayed_date]
    entry_type = attributes["type"]
    if entry_type not in (None, ""):
        details.append(str(entry_type).strip())
    if reason not in (None, ""):
        details.append(str(reason).strip())
    if status not in (None, ""):
        details.append(str(status).strip())
    return " — ".join(details)


def exceptional_entry_attributes(row: dict[str, Any]) -> dict[str, str]:
    """Return only safe, user-facing candidate attributes from an RP row."""

    raw_date = _field(
        row,
        "entryTime",
        "attDate",
        "attendanceDate",
        "date",
        "requestDate",
    )
    try:
        displayed_date = _parse_resourceplus_date(raw_date).isoformat()
    except ValueError:
        displayed_date = str(raw_date).strip() if raw_date not in (None, "") else ""
    raw_type = _field(row, "entryTypeName", "type", "entryType")
    normalized_type = str(raw_type).strip() if raw_type not in (None, "") else ""
    entry_type = (
        "Late Arrival"
        if normalized_type == "1"
        else "Early Departure"
        if normalized_type == "2"
        else normalized_type
    )
    return {
        "date": displayed_date,
        "type": entry_type,
        "reason": str(_field(row, "reason", "reasonName", "description") or "").strip(),
        "status": str(_field(row, "status", "requestStatus") or "").strip(),
    }


def prepare_cancel_exceptional_candidate(
    row: dict[str, Any],
    *,
    response_language: str,
) -> ActionIntent:
    """Prepare cancellation from a candidate already fetched for this session."""

    if not _is_cancellable_exception(row):
        raise ValueError("The selected exceptional entry is no longer cancellable.")
    exceptional_id = _field(row, "exceptionalID", "exceptionID", "id")
    if exceptional_id in (None, ""):
        raise ValueError("ResourcePlus returned a cancellable entry without an ID.")
    display = exceptional_entry_display(row)
    if response_language == "ar":
        attributes = exceptional_entry_attributes(row)
        entry_type = {
            "Late Arrival": "وصول متأخر",
            "Early Departure": "خروج مبكر",
        }.get(attributes["type"], attributes["type"])
        status = {
            "Not Approved": "غير موافق عليه",
            "Approved": "موافق عليه",
            "Rejected": "مرفوض",
            "Pending": "معلّق",
        }.get(attributes["status"], attributes["status"])
        display = " — ".join(
            value
            for value in (
                attributes["date"],
                entry_type,
                attributes["reason"],
                status,
            )
            if value
        )
    internal = {
        "exceptional_id": str(exceptional_id),
        "display": display,
    }
    return ActionIntent(
        "cancel_exceptional_entry",
        internal,
        _confirmation_summary("cancel_exceptional_entry", internal, response_language),
        response_language,
    )


async def _live_exception_reasons(lang: int) -> list[tuple[str, str]]:
    reason_rows = _require_list(
        await get_exception_reasons(lang=lang),
        "exception reasons",
    )
    live_reasons: list[tuple[str, str]] = []
    for reason in reason_rows:
        reason_id = _field(reason, "reasonID")
        reason_name_value = _field(reason, "reasonName")
        if (
            not isinstance(reason_id, str)
            or not reason_id.strip()
            or not isinstance(reason_name_value, str)
            or not reason_name_value.strip()
        ):
            continue
        live_reasons.append((reason_id, reason_name_value.strip()))
    if not live_reasons:
        raise ValueError("ResourcePlus returned no valid exceptional-entry reasons.")
    return live_reasons


def _reason_required_message(
    selected: MissingPunchRow,
    reason_names: list[str],
    language: str,
) -> str:
    choices = "\n".join(f"- {name}" for name in reason_names)
    suggested = _parse_suggested_datetime(selected.suggested_entry_time)
    if language == "ar":
        direction = "الدخول" if selected.entry_type == "IN" else "الخروج"
        return (
            f"يقترح ResourcePlus تصحيح بصمة {direction} بتاريخ "
            f"{selected.att_date.strftime('%d/%m/%Y')} الساعة "
            f"{suggested.strftime('%H:%M')}.\n\nما سبب التصحيح؟\n\n"
            f"الأسباب المتاحة:\n{choices}"
        )
    return (
        f"ResourcePlus suggests correcting the missing {selected.entry_type} "
        f"punch on {_format_date(selected.att_date)} at "
        f"{_format_time(suggested)}.\n\nWhat was the reason?\n\n"
        f"Available reasons:\n{choices}"
    )


def _exceptional_entry_intent(
    selected: MissingPunchRow,
    *,
    reason_id: str,
    reason_name: str,
    remarks: str,
    response_language: str,
) -> ActionIntent:
    internal = {
        "entry_time": selected.suggested_entry_time,
        "entry_type": ENTRY_TYPE_NUMBER[selected.entry_type],
        "reason_id": reason_id,
        "reason_name": reason_name,
        "remarks": remarks,
        "attendance_date": selected.att_date.isoformat(),
        "shift": selected.shift,
        "is_night_shift": selected.is_night_shift,
    }
    return ActionIntent(
        "create_exceptional_entry",
        internal,
        _confirmation_summary("create_exceptional_entry", internal, response_language),
        response_language,
    )


def _parse_notification_datetime(value: object) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    normalized = value.strip()
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        pass
    for date_format in (
        "%d/%m/%Y %I:%M:%S %p",
        "%d-%m-%Y %I:%M:%S %p",
        "%d-%m-%Y %H:%M",
        "%d/%m/%Y %H:%M",
        "%d-%m-%Y",
        "%d/%m/%Y",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(normalized, date_format)
        except ValueError:
            continue
    return None


def _latest_notification(items: list[dict[str, Any]]) -> dict[str, Any]:
    dated = [
        (parsed, -index, item)
        for index, item in enumerate(items)
        if (parsed := _parse_notification_datetime(item.get("NotifcnDate")))
        is not None
    ]
    if dated:
        return max(dated, key=lambda candidate: (candidate[0], candidate[1]))[2]
    return items[0]


def _confirmation_summary(
    action_type: str,
    values: dict[str, object],
    language: str,
) -> str:
    # This is a safe factual description stored with the immutable action.
    # The conversational layer renders it dynamically in the user's language.
    if action_type == "create_exceptional_entry":
        direction = "IN" if values["entry_type"] == 1 else "OUT"
        entry_time = _parse_suggested_datetime(values["entry_time"])
        if language == "ar":
            arabic_direction = "دخول" if direction == "IN" else "خروج"
            return (
                "تقدر ترسل طلب تصحيح البصمة بالتفاصيل التالية:\n\n"
                f"التاريخ: {entry_time.strftime('%d/%m/%Y')}\n"
                f"البصمة: {arabic_direction}\n"
                f"الوقت المقترح: {entry_time.strftime('%H:%M')}\n"
                f"السبب: {values['reason_name']}\n\n"
                "تبغى ترسل الطلب؟"
            )
        return (
            f"I found a suggested {direction} punch correction for "
            f"{_format_date(entry_time.date())} to {_format_time(entry_time)}. "
            f"Reason: {values['reason_name']}. Submit this attendance correction?"
        )
    if action_type == "create_exceptional_entry_from_summary":
        side = values.get("entry_type")
        side_text = (
            "IN-side"
            if side == 1
            else "OUT-side"
            if side == 2
            else None
        )
        requested_minutes = values.get("minutes")
        if not isinstance(requested_minutes, int):
            requested_minutes = _duration_minutes(values.get("less_hours"))
        target = _parse_resourceplus_date(values["att_date"])
        duration = (
            f"{requested_minutes}-minute " if requested_minutes is not None else ""
        )
        if language == "ar":
            arabic_side = (
                "التأخر في الدخول فقط"
                if side == 1
                else "الخروج المبكر فقط"
                if side == 2
                else None
            )
            arabic_duration = (
                f"لمدة {requested_minutes} دقيقة "
                if requested_minutes is not None else ""
            )
            arabic_scope = f" ({arabic_side})" if arabic_side is not None else ""
            confirmation = (
                f"للتأكيد: أرسل تصحيح حضور {arabic_duration}ليوم "
                f"{target.isoformat()} بسبب {values['reason_name']}{arabic_scope}؟"
            )
            arabic_details = [
                f"التاريخ: {values['att_date']}",
                f"المدة الناقصة: {values['less_hours']}",
                f"السبب: {values['reason_name']}",
            ]
            if arabic_side is not None:
                arabic_details.append(f"النطاق: {arabic_side}")
            if "minutes" in values:
                arabic_details.append(f"الدقائق المطلوبة: {values['minutes']}")
            return confirmation + "\n\n" + "\n".join(arabic_details)
        scope = f" {side_text}" if side_text is not None else ""
        confirmation = (
            f"Just to confirm: submit a {duration}attendance correction for "
            f"{target.strftime('%b')} {target.day} for {values['reason_name']}{scope}?"
        )
        details = [
            f"Date: {values['att_date']}",
            f"Missing duration: {values['less_hours']}",
            f"Reason: {values['reason_name']}",
        ]
        if side_text is not None:
            details.append(f"Scope: {'late IN only' if side == 1 else 'early OUT only'}")
        if "minutes" in values:
            details.append(f"Requested minutes: {values['minutes']}")
        return confirmation + "\n\n" + "\n".join(details)
    if action_type == "cancel_exceptional_entry":
        if language == "ar":
            return f"تبغى تلغي طلب تصحيح الحضور: {values['display']}؟"
        return (
            f"Cancel this attendance correction request: {values['display']}?"
        )
    if action_type == "book_day_type":
        start = date.fromisoformat(str(values["date_from"]))
        end = date.fromisoformat(str(values["date_to"]))
        if language == "ar":
            period = (
                f"يوم {start.isoformat()}" if start == end
                else f"من {start.isoformat()} إلى {end.isoformat()}"
            )
            return f"أنت على وشك طلب {values['day_type_name']} {period}. هل تريد إرساله؟"
        period = (
            f"on {start.day} {start.strftime('%B')} {start.year}"
            if start == end
            else (
                f"from {start.day} {start.strftime('%B')} {start.year} "
                f"to {end.day} {end.strftime('%B')} {end.year}"
            )
        )
        return f"You're about to apply for {values['day_type_name']} {period}. Submit it?"
    if action_type == "cancel_day_type_request":
        return f"Cancel this request: {values['display']}. Confirmation is required."
    if action_type == "approve_supervisor_request":
        verb = "approve" if values["status"] == 1 else "reject"
        category = str(values.get("category") or "Request")
        detail = str(values.get("detail") or "request")
        request_date = values.get("request_date")
        date_label = ""
        if isinstance(request_date, str) and request_date.strip():
            try:
                parsed_date = _parse_resourceplus_date(request_date)
                date_label = (
                    parsed_date.isoformat()
                    if language == "ar"
                    else f"{parsed_date.strftime('%b')} {parsed_date.day}"
                )
            except ValueError:
                date_label = request_date.strip()
        if language == "ar":
            arabic_verb = "الموافقة على" if verb == "approve" else "رفض"
            arabic_date = f" ليوم {date_label}" if date_label else ""
            return (
                f"تأكيد {arabic_verb} طلب {detail} الخاص بـ"
                f"{values['employee_name']}{arabic_date}؟"
            )
        date_text = f" for {date_label}" if date_label else ""
        request_text = (
            f"{detail} attendance correction"
            if category == "Attendance correction"
            else f"{detail} request"
        )
        return (
            f"{verb.capitalize()} {values['employee_name']}'s {request_text}"
            f"{date_text}?"
        )
    if action_type == "update_notification_read_status":
        state = "read" if values["read_status"] == 1 else "unread"
        if values["notifcn_id"] == 0:
            return (
                f"Mark all {values['count']} notifications as {state}. "
                "Confirmation is required."
            )
        return (
            f"Mark the notification '{values['title']}' as {state}. "
            "Confirmation is required."
        )
    verb = "approve" if values["status"] == 1 else "reject"
    scope = values["scope"]
    count = values["count"]
    return (
        f"You currently have {count} pending {scope} requests. This will {verb} all "
        f"{count}. Confirmation is required."
    )


def confirmation_summary(
    action_type: str,
    values: dict[str, object],
    language: str,
) -> str:
    """Re-render an immutable pending action in the current response language."""

    return _confirmation_summary(action_type, values, language)


async def _prepare_exceptional_entry(
    arguments: dict[str, Any],
    *,
    lang: int,
    response_language: str,
) -> ActionIntent:
    target_value = arguments.get("target_date")
    target_date = (
        parse_missing_punch_date(target_value)
        if target_value is not None and str(target_value).strip()
        else None
    )
    requested_direction = arguments.get("punch_direction")
    if requested_direction is not None and (
        not isinstance(requested_direction, str)
        or requested_direction not in ENTRY_TYPE_NUMBER
    ):
        raise ValueError("punch_direction must be IN, OUT, or null.")
    if target_date is not None:
        range_start = range_end = target_date
    else:
        raw_start = arguments.get("_range_from")
        raw_end = arguments.get("_range_to")
        if raw_start is not None and raw_end is not None:
            range_start = parse_missing_punch_date(raw_start)
            range_end = parse_missing_punch_date(raw_end)
        else:
            range_end = resourceplus_today()
            range_start = range_end.replace(day=1)
    if range_start > range_end:
        raise ValueError("The exceptional-entry date range is invalid.")

    normalized = normalize_missing_punch_suggestions(
        await get_missing_punch_suggestions(
            range_start.isoformat(),
            range_end.isoformat(),
            lang=lang,
        )
    )
    parsed_suggestions = [
        suggestion
        for suggestion in normalized.correctable_suggestions
        if range_start <= suggestion.att_date <= range_end
    ]
    matching_suggestions = (
        [
            suggestion
            for suggestion in parsed_suggestions
            if suggestion.att_date == target_date
        ]
        if target_date is not None
        else parsed_suggestions
    )
    if requested_direction is not None:
        matching_suggestions = [
            suggestion
            for suggestion in matching_suggestions
            if suggestion.entry_type == requested_direction
        ]
    if not matching_suggestions:
        scope = (
            f"for {_format_date(target_date)}"
            if target_date is not None
            else "in that period"
        )
        if requested_direction is not None:
            message = (
                f"I couldn't find a missing {requested_direction} punch {scope} "
                "in ResourcePlus."
            )
        else:
            message = (
                "ResourcePlus does not currently provide a valid suggested punch "
                f"correction {scope}."
            )
        raise ActionResolutionRequired(
            message,
            category="no_resourceplus_suggestion",
        )

    candidate_dates = sorted(
        {suggestion.att_date for suggestion in matching_suggestions}
    )
    if target_date is None and len(candidate_dates) > 1:
        choices = "\n".join(f"- {_format_date(day)}" for day in candidate_dates)
        raise ActionResolutionRequired(
            f"ResourcePlus has missing-punch suggestions for more than one date "
            f"in that period:\n\n{choices}\n\nWhich date would you like to regularize?",
            category="multiple_resourceplus_suggestion_dates",
            requires_clarification=True,
        )
    if len(matching_suggestions) > 1:
        candidate_date = matching_suggestions[0].att_date
        directions = {suggestion.entry_type for suggestion in matching_suggestions}
        if directions == {"IN", "OUT"}:
            raise ActionResolutionRequired(
                f"{_format_date(candidate_date)} has both a missing IN and a missing "
                "OUT punch. Which one do you want to correct?",
                category="multiple_resourceplus_suggestions",
                requires_clarification=True,
            )
        choices = "\n".join(
            f"- {suggestion.entry_type} at {suggestion.suggested_entry_time}"
            for suggestion in matching_suggestions
        )
        raise ActionResolutionRequired(
            f"ResourcePlus provides more than one punch correction for "
            f"{_format_date(candidate_date)}:\n\n"
            f"{choices}\n\nWhich one should I use?",
            category="multiple_resourceplus_suggestions",
            requires_clarification=True,
        )

    selected = matching_suggestions[0]
    live_reasons = await _live_exception_reasons(lang)

    reason_name = arguments.get("reason_name")
    remarks = arguments.get("remarks")
    reason_names = [name for _, name in live_reasons]
    if (
        not isinstance(reason_name, str)
        or not reason_name.strip()
        or not _reason_was_supplied_in_user_message(
            reason_name,
            arguments.get("_user_message"),
        )
    ):
        raise ActionResolutionRequired(
            _reason_required_message(selected, reason_names, response_language),
            category="reason_required",
            requires_clarification=True,
            draft_context={
                "attendance_date": selected.att_date.isoformat(),
                "entry_type": selected.entry_type,
                "suggested_entry_time": str(selected.suggested_entry_time),
                "language": response_language,
            },
            reason_options=reason_names,
        )
    if not isinstance(remarks, str) or not remarks.strip():
        remarks = reason_name

    selected_reason_index = await match_live_reason(reason_name, reason_names)
    if selected_reason_index is None:
        choices = ", ".join(reason_names[:6])
        raise ActionResolutionRequired(
            "I couldn't confidently match that reason to one ResourcePlus reason."
            f" Available reasons include: {choices}. Which reason should I use?",
            category="reason_ambiguous",
            requires_clarification=True,
            reason_options=reason_names,
        )
    reason_id, selected_reason_name = live_reasons[selected_reason_index]

    return _exceptional_entry_intent(
        selected,
        reason_id=reason_id,
        reason_name=selected_reason_name,
        remarks=remarks.strip(),
        response_language=response_language,
    )


async def prepare_exceptional_entry_reason_follow_up(
    *,
    attendance_date: str,
    entry_type: str,
    suggested_entry_time: str,
    reason_text: str,
    lang: int,
    response_language: str,
) -> ActionIntent:
    """Resolve a draft reason once, using only fresh authoritative live data."""

    target_date = parse_missing_punch_date(attendance_date)
    if entry_type not in ENTRY_TYPE_NUMBER:
        raise ValueError("Exceptional-entry draft direction is invalid.")
    normalized = normalize_missing_punch_suggestions(
        await get_missing_punch_suggestions(
            target_date.isoformat(),
            target_date.isoformat(),
            lang=lang,
        )
    )
    exact_matches = [
        suggestion
        for suggestion in normalized.correctable_suggestions
        if suggestion.att_date == target_date
        and suggestion.entry_type == entry_type
        and suggestion.suggested_entry_time == suggested_entry_time
    ]
    if not exact_matches:
        message = (
            "تغير اقتراح ResourcePlus لهذه البصمة. ابدأ طلب التصحيح من جديد."
            if response_language == "ar"
            else (
                "The ResourcePlus punch suggestion changed. Please start the "
                "correction request again."
            )
        )
        raise ActionResolutionRequired(
            message,
            category="draft_stale",
        )

    selected = exact_matches[0]
    live_reasons = await _live_exception_reasons(lang)
    reason_names = [name for _, name in live_reasons]
    match = deterministic_reason_match(reason_text, reason_names)
    if match.index is None:
        choices = "\n".join(f"- {name}" for name in reason_names)
        if response_language == "ar":
            lead = (
                "السبب يطابق أكثر من خيار. اختر سببًا واحدًا من القائمة:"
                if match.status == "ambiguous"
                else "لم أجد سببًا مطابقًا. اختر سببًا من القائمة:"
            )
        else:
            lead = (
                "That reason matches more than one option. Please choose one:"
                if match.status == "ambiguous"
                else "I couldn't match that reason. Please choose one of these:"
            )
        raise ActionResolutionRequired(
            f"{lead}\n\n{choices}",
            category=(
                "reason_ambiguous"
                if match.status == "ambiguous"
                else "reason_unknown"
            ),
            requires_clarification=True,
            reason_options=reason_names,
        )

    reason_id, selected_reason_name = live_reasons[match.index]
    return _exceptional_entry_intent(
        selected,
        reason_id=reason_id,
        reason_name=selected_reason_name,
        remarks=reason_text.strip(),
        response_language=response_language,
    )


def _less_hours_not_eligible_message(day: LessHoursDay, language: str) -> str:
    if day.eligibility == "no_punches":
        if language == "ar":
            return (
                f"لا توجد بصمات حضور في {day.attendance_date.isoformat()}، لذلك لا يمكن "
                "تصحيح هذا اليوم كإدخال استثنائي. يمكنك طلب إجازة أو مهمة عمل بدلاً من ذلك."
            )
        return (
            f"I don't see any attendance punches for {_format_date(day.attendance_date)}, "
            "so this can't be corrected as an exceptional entry. You can apply for "
            "Leave or Business Travel for that day instead."
        )
    labels = {
        "week_end": "a week end",
        "holiday": "a holiday",
        "leave": "leave",
        "business_travel": "business travel",
    }
    if day.eligibility in labels:
        return (
            f"That day is recorded as {labels[day.eligibility]}, so "
            "there are no missing hours to correct with an exceptional entry."
        )
    return "You have no missing hours to correct for that day."


async def _prepare_less_hours_correction(
    arguments: dict[str, Any],
    *,
    lang: int,
    response_language: str,
) -> ActionIntent:
    try:
        target_date = date.fromisoformat(str(arguments.get("target_date")))
    except ValueError as exc:
        raise ValueError("The correction date must use YYYY-MM-DD format.") from exc

    entry_type = arguments.get("entry_type")
    if entry_type is not None:
        if isinstance(entry_type, bool):
            raise ValueError("entry_type must be 1, 2, or omitted.")
        try:
            entry_type = int(entry_type)
        except (TypeError, ValueError) as exc:
            raise ValueError("entry_type must be 1, 2, or omitted.") from exc
        if entry_type not in {1, 2}:
            raise ValueError("entry_type must be 1, 2, or omitted.")

    minutes = arguments.get("minutes")
    if minutes is not None:
        if isinstance(minutes, bool):
            raise ValueError("minutes must be a positive whole number or omitted.")
        try:
            minutes = int(minutes)
        except (TypeError, ValueError) as exc:
            raise ValueError("minutes must be a positive whole number or omitted.") from exc
        if minutes <= 0:
            raise ValueError("minutes must be a positive whole number or omitted.")

    inspection = await inspect_less_hours_period(target_date, target_date, lang=lang)
    matching_days = [day for day in inspection.days if day.attendance_date == target_date]
    if not matching_days:
        raise ActionResolutionRequired(
            "I couldn't find an attendance entry for that date.",
            category="attendance_not_found",
        )
    selected = matching_days[0]
    duplicate_message = less_hours_duplicate_guard_message(
        inspection,
        target_date,
        response_language,
    )
    if duplicate_message is not None:
        raise ActionResolutionRequired(
            duplicate_message,
            category="existing_exceptional_entry",
        )
    if selected.eligibility != "eligible":
        raise ActionResolutionRequired(
            _less_hours_not_eligible_message(selected, response_language),
            category=selected.eligibility,
        )

    live_reasons = await _live_exception_reasons(lang)
    reason_names = [name for _, name in live_reasons]
    reason_name = arguments.get("reason_name")
    if (
        not isinstance(reason_name, str)
        or not reason_name.strip()
        or (
            arguments.get("_user_message") is not None
            and not _reason_was_supplied_in_user_message(
                reason_name,
                arguments.get("_user_message"),
            )
        )
    ):
        choices = "\n".join(f"- {name}" for name in reason_names)
        question = (
            "ما سبب التصحيح؟\n\n" if response_language == "ar" else "What was the reason?\n\n"
        )
        raise ActionResolutionRequired(
            question + choices,
            category="reason_required",
            requires_clarification=True,
            draft_context={
                "date": target_date.isoformat(),
                "less_hours": selected.less_hours,
                **({"entry_type": str(entry_type)} if entry_type is not None else {}),
                **({"minutes": str(minutes)} if minutes is not None else {}),
            },
            reason_options=reason_names,
        )

    match = deterministic_reason_match(reason_name, reason_names)
    selected_reason_index = match.index
    if selected_reason_index is None:
        selected_reason_index = await match_live_reason(reason_name, reason_names)
    if selected_reason_index is None:
        raise ActionResolutionRequired(
            "I couldn't confidently match that reason to one live ResourcePlus reason.",
            category="reason_ambiguous",
            requires_clarification=True,
            reason_options=reason_names,
        )
    reason_id, selected_reason_name = live_reasons[selected_reason_index]
    remarks = arguments.get("remarks")
    if not isinstance(remarks, str) or not remarks.strip():
        remarks = reason_name
    internal: dict[str, object] = {
        "att_date": target_date.isoformat(),
        "reason_id": reason_id,
        "reason_name": selected_reason_name,
        "remarks": remarks.strip(),
        "less_hours": selected.less_hours,
        "attendance_snapshot": _attendance_snapshot(selected),
    }
    if entry_type is not None:
        internal["entry_type"] = entry_type
    if minutes is not None:
        internal["minutes"] = minutes
    return ActionIntent(
        "create_exceptional_entry_from_summary",
        internal,
        _confirmation_summary(
            "create_exceptional_entry_from_summary",
            internal,
            response_language,
        ),
        response_language,
    )


async def prepare_less_hours_reason_follow_up(
    *,
    target_date: str,
    reason_text: str,
    lang: int,
    response_language: str,
    entry_type: int | None = None,
    minutes: int | None = None,
) -> ActionIntent:
    """Revalidate attendance and the live reason before preparing the write."""

    arguments: dict[str, Any] = {
        "target_date": target_date,
        "reason_name": reason_text,
        "remarks": reason_text,
    }
    if entry_type is not None:
        arguments["entry_type"] = entry_type
    if minutes is not None:
        arguments["minutes"] = minutes
    return await _prepare_less_hours_correction(
        arguments,
        lang=lang,
        response_language=response_language,
    )


async def _prepare_cancel_exceptional_entry(
    arguments: dict[str, Any],
    *,
    lang: int,
    response_language: str,
) -> ActionIntent:
    try:
        start = date.fromisoformat(str(arguments.get("date_from")))
        end = date.fromisoformat(str(arguments.get("date_to") or start.isoformat()))
    except ValueError as exc:
        raise ValueError("Cancellation dates must use YYYY-MM-DD format.") from exc
    if start > end:
        raise ValueError("date_from must be on or before date_to.")
    target_date_raw = arguments.get("target_date")
    target_date = date.fromisoformat(str(target_date_raw)) if target_date_raw else None
    rows = await find_pending_exceptional_entries(start, end, lang=lang)
    if target_date is not None:
        matching: list[dict[str, Any]] = []
        for row in rows:
            try:
                row_date = _parse_resourceplus_date(
                    _field(
                        row,
                        "entryTime",
                        "attDate",
                        "attendanceDate",
                        "date",
                        "requestDate",
                    )
                )
            except ValueError:
                continue
            if row_date == target_date:
                matching.append(row)
        rows = matching
    if not rows:
        raise ActionResolutionRequired(
            "You have no cancellable pending exceptional entry for that period.",
            category="no_cancellable_exception",
        )
    if len(rows) > 1:
        choices = "\n".join(f"- {exceptional_entry_display(row)}" for row in rows)
        raise ActionResolutionRequired(
            "I found more than one cancellable exceptional entry:\n\n"
            + choices
            + "\n\nWhich one should I cancel?",
            category="multiple_cancellable_exceptions",
            requires_clarification=True,
        )
    return prepare_cancel_exceptional_candidate(
        rows[0],
        response_language=response_language,
    )


async def _prepare_book_day_type(
    arguments: dict[str, Any],
    *,
    lang: int,
    response_language: str,
) -> ActionIntent:
    try:
        start = date.fromisoformat(arguments.get("date_from"))
        end = date.fromisoformat(arguments.get("date_to"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Leave dates must use YYYY-MM-DD format.") from exc
    if start > end:
        raise ValueError("date_from must be on or before date_to.")
    requested_name = arguments.get("day_type_name")
    if not isinstance(requested_name, str) or not requested_name.strip():
        raise ValueError("A ResourcePlus day type is required.")

    day_types = _require_list(await get_day_types(lang=lang), "day types")
    matches = [
        item for item in day_types if _same_text(item.get("dayType"), requested_name)
    ]
    if len(matches) != 1:
        raise ValueError("The selected day type does not match a ResourcePlus day type.")
    requested_group = arguments.get("_booking_group")
    if requested_group is not None:
        group = bookable_day_type_group(matches[0].get("group"))
        if group is None or requested_group not in {group, "leave_or_travel"}:
            raise ValueError("The selected day type is not available for this leave or travel request.")
    day_type_id = matches[0].get("dayID")
    if isinstance(day_type_id, bool) or not isinstance(day_type_id, int):
        raise ValueError("ResourcePlus returned an invalid day type ID.")

    internal = {
        "date_from": start.isoformat(),
        "date_to": end.isoformat(),
        "day_type_id": day_type_id,
        "day_type_name": str(matches[0]["dayType"]),
    }
    return ActionIntent(
        "book_day_type",
        internal,
        _confirmation_summary("book_day_type", internal, response_language),
        response_language,
    )


async def _prepare_cancel_request(
    arguments: dict[str, Any],
    *,
    lang: int,
    response_language: str,
) -> ActionIntent:
    try:
        target_start = date.fromisoformat(arguments.get("date_from"))
        raw_end = arguments.get("date_to")
        target_end = date.fromisoformat(raw_end) if raw_end else target_start
    except (TypeError, ValueError) as exc:
        raise ValueError("Request dates must use YYYY-MM-DD format.") from exc
    if target_start > target_end:
        raise ValueError("date_from must be on or before date_to.")
    requested_type = arguments.get("day_type_name")
    if not isinstance(requested_type, str) or not requested_type.strip():
        raise ValueError("The request day type is required.")

    requests = _require_list(
        await get_my_day_type_requests(target_start, target_end, lang=lang),
        "absence requests",
    )
    matches: list[dict[str, Any]] = []
    for item in requests:
        try:
            item_start = _parse_resourceplus_date(item.get("dateFrom"))
            raw_item_end = item.get("dateTo")
            item_end = (
                _parse_resourceplus_date(raw_item_end)
                if isinstance(raw_item_end, str) and raw_item_end.strip()
                else item_start
            )
        except ValueError:
            continue
        single_requested_date = raw_end is None or target_start == target_end
        date_matches = (
            item_start <= target_start <= item_end
            if single_requested_date
            else item_start == target_start and item_end == target_end
        )
        mapping_id = item.get("mappingID")
        if (
            date_matches
            and _same_text(item.get("dayType"), requested_type)
            and _same_text(item.get("status"), "Pending")
            and mapping_id is not None
            and bool(str(mapping_id).strip())
        ):
            matches.append(item)
    if not matches:
        raise ValueError("No matching pending absence request was found.")
    if len(matches) > 1:
        raise ValueError(
            "More than one matching pending absence request was found. "
            "Please clarify the request date or date range."
        )

    selected = matches[0]
    display = (
        f"{selected.get('dayType')} from {selected.get('dateFrom')} "
        f"through {selected.get('dateTo')} ({selected.get('status')})"
    )
    internal = {"mapping_id": str(selected["mappingID"]), "display": display}
    return ActionIntent(
        "cancel_day_type_request",
        internal,
        _confirmation_summary("cancel_day_type_request", internal, response_language),
        response_language,
    )


async def _prepare_supervisor_request(
    arguments: dict[str, Any],
    *,
    lang: int,
    response_language: str,
) -> ActionIntent:
    employee_name = arguments.get("employee_name")
    detail = arguments.get("detail")
    decision = arguments.get("decision")
    trusted_request_id = arguments.get("request_id")
    trusted_request_type = arguments.get("request_type")
    use_trusted_selector = arguments.get("_trusted_selector") is True
    if not isinstance(employee_name, str) or not employee_name.strip():
        raise ValueError("The employee name is required.")
    if detail is not None and not isinstance(detail, str):
        raise ValueError("detail must be text or null.")
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject.")

    approvals = _require_list(
        await get_pending_approvals(lang=lang),
        "pending approvals",
    )
    matches = []
    for item in approvals:
        if use_trusted_selector:
            if not (
                isinstance(trusted_request_id, str)
                and isinstance(trusted_request_type, str)
                and str(item.get("requestId")) == trusted_request_id
                and str(item.get("requestType")) == trusted_request_type
            ):
                continue
        else:
            if not _same_text(item.get("employeeName"), employee_name):
                continue
            if detail and not _same_text(item.get("detail"), detail):
                continue
        live_status = str(item.get("status") or "").strip().casefold()
        if live_status in {"approved", "rejected", "cancelled", "canceled"}:
            continue
        if item.get("requestId") and item.get("requestType"):
            matches.append(item)
    if not matches and use_trusted_selector:
        message = (
            "ما قدرت ألقى طلب الموافقة المعلّق الذي تقصده."
            if response_language == "ar"
            else "I couldn't find that pending request."
        )
        raise ActionResolutionRequired(
            message,
            category="pending_request_unavailable",
        )
    if len(matches) != 1:
        raise ValueError("No single matching pending supervisor request was found.")

    selected = matches[0]
    live_detail = str(
        _field(selected, "detail", "description", "dayType", "reasonName")
        or detail
        or "request"
    )
    normalized_type = re.sub(
        r"[^a-z]", "", str(selected.get("requestType", "")).casefold()
    )
    category = (
        "Attendance correction"
        if normalized_type in {"exceptionentry", "exceptionalentry"}
        else "Business Travel"
        if normalized_type in {"absence", "leave", "daytype"}
        and "business travel" in live_detail.casefold()
        else "Leave"
        if normalized_type in {"absence", "leave", "daytype"}
        else str(selected.get("requestType") or "Request")
    )
    internal = {
        "request_id": str(selected["requestId"]),
        "request_type": str(selected["requestType"]),
        "status": 1 if decision == "approve" else 2,
        "employee_name": str(selected.get("employeeName", employee_name)),
        "detail": live_detail,
        "category": category,
        "approval_snapshot_verified": True,
    }
    request_date = _field(
        selected,
        "date",
        "dateFrom",
        "AttDate",
        "attendanceDate",
        "requestDate",
    )
    if request_date not in (None, ""):
        internal["request_date"] = str(request_date)
    return ActionIntent(
        "approve_supervisor_request",
        internal,
        _confirmation_summary(
            "approve_supervisor_request", internal, response_language
        ),
        response_language,
    )


async def _prepare_approve_all(
    arguments: dict[str, Any],
    *,
    lang: int,
    response_language: str,
) -> ActionIntent:
    decision = arguments.get("decision")
    scope = arguments.get("request_type")
    if decision not in {"approve", "reject"}:
        raise ValueError("decision must be approve or reject.")
    if scope not in {"all", "Absence", "ExceptionEntry"}:
        raise ValueError("request_type must be all, Absence, or ExceptionEntry.")

    approvals = _require_list(
        await get_pending_approvals(lang=lang),
        "pending approvals",
    )
    selected = (
        approvals
        if scope == "all"
        else [item for item in approvals if item.get("requestType") == scope]
    )
    if not selected:
        raise ValueError(f"There are no pending {scope} requests.")

    internal: dict[str, object] = {
        "status": 1 if decision == "approve" else 2,
        "request_type": None if scope == "all" else scope,
        "scope": "all types" if scope == "all" else scope,
        "count": len(selected),
    }
    return ActionIntent(
        "approve_all_requests",
        internal,
        _confirmation_summary("approve_all_requests", internal, response_language),
        response_language,
    )


async def _prepare_notification_read_status(
    arguments: dict[str, Any],
    *,
    lang: int,
    response_language: str,
) -> ActionIntent:
    target = arguments.get("target")
    title = arguments.get("notification_title")
    read_status = arguments.get("read_status")
    if target not in {"latest", "one", "all"}:
        raise ValueError("target must be latest, one, or all.")
    if isinstance(read_status, bool) or read_status not in {0, 1}:
        raise ValueError("read_status must be 0 (unread) or 1 (read).")

    notifications = _require_list(
        await get_notifications(lang=lang),
        "notifications",
    )
    if not notifications:
        raise ValueError("There are no notifications to update.")

    if target == "all":
        internal: dict[str, object] = {
            "notifcn_id": 0,
            "read_status": read_status,
            "count": len(notifications),
            "title": "all notifications",
        }
    else:
        if target == "latest":
            selected = _latest_notification(notifications)
        else:
            if not isinstance(title, str) or not title.strip():
                raise ValueError("A notification title is required.")
            matches = [
                item
                for item in notifications
                if _same_text(item.get("NotifcnTitle"), title)
            ]
            if not matches:
                raise ValueError("No matching notification was found.")
            if len(matches) > 1:
                raise ValueError(
                    "More than one notification has that title. Please clarify."
                )
            selected = matches[0]

        raw_id = selected.get("NotifcnID")
        try:
            notifcn_id = int(raw_id)
        except (TypeError, ValueError) as exc:
            raise ValueError("ResourcePlus returned an invalid notification ID.") from exc
        if isinstance(raw_id, bool) or notifcn_id <= 0:
            raise ValueError("ResourcePlus returned an invalid notification ID.")
        internal = {
            "notifcn_id": notifcn_id,
            "read_status": read_status,
            "count": 1,
            "title": str(selected.get("NotifcnTitle") or "notification"),
        }

    return ActionIntent(
        "update_notification_read_status",
        internal,
        _confirmation_summary(
            "update_notification_read_status",
            internal,
            response_language,
        ),
        response_language,
    )


async def prepare_write_action(
    tool_name: str,
    arguments: dict[str, Any],
    *,
    lang: int,
    response_language: str,
) -> ActionIntent:
    if tool_name == "prepare_exceptional_entry":
        return await _prepare_exceptional_entry(
            arguments, lang=lang, response_language=response_language
        )
    if tool_name == "prepare_less_hours_correction":
        return await _prepare_less_hours_correction(
            arguments, lang=lang, response_language=response_language
        )
    if tool_name == "prepare_cancel_exceptional_entry":
        return await _prepare_cancel_exceptional_entry(
            arguments, lang=lang, response_language=response_language
        )
    if tool_name == "prepare_book_day_type":
        return await _prepare_book_day_type(
            arguments, lang=lang, response_language=response_language
        )
    if tool_name == "prepare_cancel_day_type_request":
        return await _prepare_cancel_request(
            arguments, lang=lang, response_language=response_language
        )
    if tool_name == "prepare_supervisor_request":
        return await _prepare_supervisor_request(
            arguments, lang=lang, response_language=response_language
        )
    if tool_name == "prepare_approve_all_requests":
        return await _prepare_approve_all(
            arguments, lang=lang, response_language=response_language
        )
    if tool_name == "prepare_notification_read_status":
        return await _prepare_notification_read_status(
            arguments, lang=lang, response_language=response_language
        )
    raise ValueError("Unsupported write action.")


async def revalidate_pending_action(
    action_type: str,
    arguments: dict[str, object],
    *,
    lang: int,
    response_language: str,
) -> None:
    """Revalidate new v2 correction state immediately before its one-shot write."""

    if (
        action_type == "approve_supervisor_request"
        and arguments.get("approval_snapshot_verified") is True
    ):
        pending = _require_list(
            await get_pending_approvals(lang=lang),
            "pending approvals",
        )
        request_id = str(arguments.get("request_id") or "")
        request_type = str(arguments.get("request_type") or "")
        still_pending = any(
            str(item.get("requestId")) == request_id
            and str(item.get("requestType")) == request_type
            and str(item.get("status") or "").strip().casefold()
            not in {"approved", "rejected", "cancelled", "canceled"}
            for item in pending
        )
        if not still_pending:
            message = (
                "هذا الطلب لم يعد موجودًا ضمن الموافقات المعلقة، لذلك ما تم إرسال أي إجراء."
                if response_language == "ar"
                else (
                    "That request is no longer in your pending approvals, so nothing "
                    "was submitted."
                )
            )
            raise ActionResolutionRequired(
                message,
                category="pending_request_changed",
            )
        return
    if action_type != "create_exceptional_entry_from_summary":
        return
    stored_snapshot = arguments.get("attendance_snapshot")
    # In-memory actions created before this field existed cannot survive a process
    # restart. The compatibility branch is retained for directly constructed tests.
    if not isinstance(stored_snapshot, dict):
        return
    try:
        target_date = date.fromisoformat(str(arguments["att_date"]))
    except (KeyError, ValueError) as exc:
        raise ValueError("The stored correction date is invalid.") from exc

    inspection = await inspect_less_hours_period(target_date, target_date, lang=lang)
    matching = [day for day in inspection.days if day.attendance_date == target_date]
    selected = matching[0] if matching else None
    duplicate_message = less_hours_duplicate_guard_message(
        inspection,
        target_date,
        response_language,
    )
    if duplicate_message is not None:
        raise ActionResolutionRequired(
            duplicate_message,
            category="existing_exceptional_entry",
        )
    current_snapshot = _attendance_snapshot(selected) if selected is not None else None
    if (
        selected is None
        or selected.eligibility != "eligible"
        or current_snapshot != stored_snapshot
    ):
        message = (
            "تغيّرت بيانات الحضور منذ إعداد التصحيح. لم يتم إرسال الطلب؛ راجع "
            "بيانات الحضور وابدأ التصحيح من جديد."
            if response_language == "ar"
            else (
                "Your attendance changed after this correction was prepared. Nothing "
                "was submitted; review the attendance entry and start again."
            )
        )
        raise ActionResolutionRequired(message, category="attendance_changed")

    stored_reason_id = arguments.get("reason_id")
    stored_reason_name = arguments.get("reason_name")
    live_reasons = await _live_exception_reasons(lang)
    reason_is_live = any(
        reason_id == stored_reason_id and reason_name == stored_reason_name
        for reason_id, reason_name in live_reasons
    )
    if not reason_is_live:
        message = (
            "تغيّرت أسباب التصحيح المتاحة. لم يتم إرسال الطلب؛ ابدأ التصحيح من جديد."
            if response_language == "ar"
            else (
                "The available correction reasons changed. Nothing was submitted; "
                "start the correction again."
            )
        )
        raise ActionResolutionRequired(message, category="reason_changed")


async def execute_pending_action(action_type: str, arguments: dict[str, object]) -> Any:
    """Execute only backend-stored validated arguments after confirmation."""

    if action_type == "create_exceptional_entry":
        return await create_exceptional_entry(
            entry_time=str(arguments["entry_time"]),
            entry_type=int(arguments["entry_type"]),
            reason_id=str(arguments["reason_id"]),
            remarks=str(arguments["remarks"]),
        )
    if action_type == "create_exceptional_entry_from_summary":
        return await create_exceptional_entry_from_summary(
            att_date=str(arguments["att_date"]),
            reason_id=str(arguments["reason_id"]),
            remarks=str(arguments["remarks"]),
            entry_type=(
                int(arguments["entry_type"])
                if "entry_type" in arguments
                else None
            ),
            minutes=int(arguments["minutes"]) if "minutes" in arguments else None,
        )
    if action_type == "cancel_exceptional_entry":
        return await cancel_exceptional_entry(str(arguments["exceptional_id"]))
    if action_type == "book_day_type":
        return await book_day_type(
            date_from=str(arguments["date_from"]),
            date_to=str(arguments["date_to"]),
            day_type_id=int(arguments["day_type_id"]),
        )
    if action_type == "cancel_day_type_request":
        return await cancel_day_type_request(str(arguments["mapping_id"]))
    if action_type == "approve_supervisor_request":
        result = await approve_supervisor_request(
            request_id=str(arguments["request_id"]),
            request_type=str(arguments["request_type"]),
            status=int(arguments["status"]),
        )
        if not isinstance(result, dict) or result.get("success") is not True:
            return result

        verified_result = dict(result)
        verified_result["_approval_status"] = int(arguments["status"])
        try:
            pending = _require_list(
                await get_pending_approvals(),
                "pending approvals",
            )
        except (ResourcePlusError, ValueError):
            verified_result["_approval_verification"] = "unavailable"
            return verified_result
        request_id = str(arguments["request_id"])
        request_type = str(arguments["request_type"])
        still_pending = any(
            str(item.get("requestId")) == request_id
            and str(item.get("requestType")) == request_type
            for item in pending
            if item.get("requestId") not in (None, "")
        )
        verified_result["_approval_verification"] = (
            "still_pending" if still_pending else "verified"
        )
        verified_result["_remaining_pending_approvals"] = pending
        return verified_result
    if action_type == "approve_all_requests":
        request_type = arguments.get("request_type")
        result = await approve_all_requests(
            status=int(arguments["status"]),
            request_type=str(request_type) if request_type is not None else None,
        )
        if not isinstance(result, dict) or result.get("success") is not True:
            return result
        verified_result = dict(result)
        try:
            pending = _require_list(
                await get_pending_approvals(),
                "pending approvals",
            )
        except (ResourcePlusError, ValueError):
            verified_result["_bulk_approval_verification"] = "unavailable"
            return verified_result
        remaining = (
            pending
            if request_type is None
            else [
                item for item in pending
                if item.get("requestType") == str(request_type)
            ]
        )
        verified_result["_bulk_approval_verification"] = (
            "verified" if not remaining else "still_pending"
        )
        verified_result["_verified_count"] = int(arguments.get("count", 0))
        return verified_result
    if action_type == "update_notification_read_status":
        return await update_notification_read_status(
            notifcn_id=int(arguments["notifcn_id"]),
            read_status=int(arguments["read_status"]),
        )
    raise ValueError("Unsupported pending action.")
