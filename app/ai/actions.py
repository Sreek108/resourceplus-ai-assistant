import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from app.ai.reason_matcher import deterministic_reason_match, match_live_reason
from app.resourceplus.approvals import (
    approve_all_requests,
    approve_supervisor_request,
    get_pending_approvals,
)
from app.resourceplus.attendance import (
    create_exceptional_entry,
    get_exception_reasons,
    get_missing_punch_suggestions,
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


@dataclass(frozen=True)
class ActionIntent:
    action_type: str
    validated_arguments: dict[str, object]
    summary: str
    language: str


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
    for date_format in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y"):
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


def _format_time(value: datetime) -> str:
    hour = value.hour % 12 or 12
    suffix = "AM" if value.hour < 12 else "PM"
    return f"{hour}:{value.minute:02d} {suffix}"


def _format_date(value: date) -> str:
    return value.strftime("%d %B %Y")


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
                "إرسال طلب إدخال استثنائي للبيانات التالية:\n\n"
                f"التاريخ: {entry_time.strftime('%d/%m/%Y')}\n"
                f"البصمة: {arabic_direction}\n"
                f"الوقت المقترح: {entry_time.strftime('%H:%M')}\n"
                f"السبب: {values['reason_name']}\n\n"
                "هل ترغب في إرسال الطلب؟ التأكيد مطلوب."
            )
        return (
            f"ResourcePlus suggests correcting the {direction} punch on "
            f"{_format_date(entry_time.date())} to {_format_time(entry_time)}. "
            f"Reason: {values['reason_name']}. Submit this exceptional-entry request? "
            "Confirmation is required."
        )
    if action_type == "book_day_type":
        return (
            f"Submit {values['day_type_name']} from {values['date_from']} through "
            f"{values['date_to']}. Confirmation is required."
        )
    if action_type == "cancel_day_type_request":
        return f"Cancel this request: {values['display']}. Confirmation is required."
    if action_type == "approve_supervisor_request":
        verb = "approve" if values["status"] == 1 else "reject"
        return (
            f"{verb.capitalize()} {values['employee_name']}'s request: "
            f"{values['detail']}. Confirmation is required."
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
            range_end = date.today()
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
        direction = f" {requested_direction}" if requested_direction else ""
        message = (
            f"ResourcePlus does not currently provide a valid{direction} suggested "
            f"punch correction {scope}."
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
        if not _same_text(item.get("employeeName"), employee_name):
            continue
        if detail and not _same_text(item.get("detail"), detail):
            continue
        if item.get("requestId") and item.get("requestType"):
            matches.append(item)
    if len(matches) != 1:
        raise ValueError("No single matching pending supervisor request was found.")

    selected = matches[0]
    internal = {
        "request_id": str(selected["requestId"]),
        "request_type": str(selected["requestType"]),
        "status": 1 if decision == "approve" else 2,
        "employee_name": str(selected.get("employeeName", employee_name)),
        "detail": str(selected.get("detail", "request")),
    }
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


async def execute_pending_action(action_type: str, arguments: dict[str, object]) -> Any:
    """Execute only backend-stored validated arguments after confirmation."""

    if action_type == "create_exceptional_entry":
        return await create_exceptional_entry(
            entry_time=str(arguments["entry_time"]),
            entry_type=int(arguments["entry_type"]),
            reason_id=str(arguments["reason_id"]),
            remarks=str(arguments["remarks"]),
        )
    if action_type == "book_day_type":
        return await book_day_type(
            date_from=str(arguments["date_from"]),
            date_to=str(arguments["date_to"]),
            day_type_id=int(arguments["day_type_id"]),
        )
    if action_type == "cancel_day_type_request":
        return await cancel_day_type_request(str(arguments["mapping_id"]))
    if action_type == "approve_supervisor_request":
        return await approve_supervisor_request(
            request_id=str(arguments["request_id"]),
            request_type=str(arguments["request_type"]),
            status=int(arguments["status"]),
        )
    if action_type == "approve_all_requests":
        request_type = arguments.get("request_type")
        return await approve_all_requests(
            status=int(arguments["status"]),
            request_type=str(request_type) if request_type is not None else None,
        )
    if action_type == "update_notification_read_status":
        return await update_notification_read_status(
            notifcn_id=int(arguments["notifcn_id"]),
            read_status=int(arguments["read_status"]),
        )
    raise ValueError("Unsupported pending action.")
