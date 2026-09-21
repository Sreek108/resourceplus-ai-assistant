import re
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

from app.ai.reason_matcher import match_live_reason
from app.resourceplus.approvals import (
    approve_all_requests,
    approve_supervisor_request,
    get_pending_approvals,
)
from app.resourceplus.attendance import (
    _parse_iso_datetime,
    create_exceptional_entry,
    get_attendance_summary,
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
    ) -> None:
        super().__init__(message)
        self.category = category
        self.requires_clarification = requires_clarification


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
    if not isinstance(value, str):
        raise ValueError("ResourcePlus returned an invalid suggested entry time.")
    for date_format in ("%d/%m/%Y %H:%M", "%Y-%m-%dT%H:%M:%S"):
        try:
            return datetime.strptime(value, date_format)
        except ValueError:
            continue
    raise ValueError("ResourcePlus returned an invalid suggested entry time.")


def _same_text(left: object, right: object) -> bool:
    return isinstance(left, str) and isinstance(right, str) and (
        left.strip().casefold() == right.strip().casefold()
    )


def _normalized_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _field(item: dict[str, Any], *names: str) -> object:
    normalized = {_normalized_key(key): value for key, value in item.items()}
    for name in names:
        key = _normalized_key(name)
        if key in normalized:
            return normalized[key]
    return None


def _attendance_days(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return _require_list(payload, "attendance summary")
    if not isinstance(payload, dict):
        raise ValueError("ResourcePlus returned an unexpected attendance summary response.")
    days = _field(payload, "Days")
    return _require_list(days, "attendance days")


def _less_hours_seconds(value: object) -> int:
    if not isinstance(value, str):
        return 0
    match = re.fullmatch(r"\s*(\d+):(\d{1,2})(?::(\d{1,2}))?\s*", value)
    if not match:
        return 0
    hours, minutes, seconds = (int(part or 0) for part in match.groups())
    if minutes >= 60 or seconds >= 60:
        return 0
    return hours * 3600 + minutes * 60 + seconds


def _less_hours_days(payload: Any) -> dict[date, int]:
    applicable: dict[date, int] = {}
    for item in _attendance_days(payload):
        try:
            attendance_date = _parse_resourceplus_date(
                _field(item, "Date", "AttDate", "AttendanceDate", "WorkDate")
            )
        except ValueError:
            continue
        less_seconds = _less_hours_seconds(
            _field(item, "LessHrs", "LessHours", "MissingHours")
        )
        if less_seconds > 0:
            applicable[attendance_date] = max(
                applicable.get(attendance_date, 0),
                less_seconds,
            )
    return applicable


def _format_duration(total_seconds: int) -> str:
    hours, remainder = divmod(total_seconds, 3600)
    minutes = remainder // 60
    if hours and minutes:
        return f"{hours}h {minutes}m"
    if hours:
        return f"{hours}h"
    return f"{minutes}m"


def _format_time(value: datetime) -> str:
    hour = value.hour % 12 or 12
    suffix = "AM" if value.hour < 12 else "PM"
    return f"{hour}:{value.minute:02d} {suffix}"


def _suggestion_entry_type(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if value in {1, 2}:
        return int(value)
    normalized = str(value).strip().upper()
    if normalized in {"1", "IN"}:
        return 1
    if normalized in {"2", "OUT"}:
        return 2
    return None


def _format_date(value: date) -> str:
    return value.strftime("%d %B %Y")


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
    del language
    if action_type == "create_exceptional_entry":
        direction = "IN" if values["entry_type"] == 1 else "OUT"
        entry_time = datetime.fromisoformat(str(values["entry_time"]))
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
        _parse_resourceplus_date(target_value)
        if target_value is not None and str(target_value).strip()
        else None
    )
    reason_name = arguments.get("reason_name")
    remarks = arguments.get("remarks")
    if not isinstance(reason_name, str) or not reason_name.strip():
        raise ActionResolutionRequired(
            "What reason should I use for the exceptional-entry request?",
            category="reason_required",
            requires_clarification=True,
        )
    if not isinstance(remarks, str) or not remarks.strip():
        remarks = reason_name

    if target_date is not None:
        range_start = range_end = target_date
    else:
        raw_start = arguments.get("_range_from")
        raw_end = arguments.get("_range_to")
        if raw_start is not None and raw_end is not None:
            range_start = _parse_resourceplus_date(raw_start)
            range_end = _parse_resourceplus_date(raw_end)
        else:
            range_end = date.today()
            range_start = range_end.replace(day=1)
    if range_start > range_end:
        raise ValueError("The exceptional-entry date range is invalid.")

    attendance = await get_attendance_summary(
        range_start.isoformat(),
        range_end.isoformat(),
        lang=lang,
    )
    applicable = _less_hours_days(attendance)
    if target_date is not None:
        if target_date not in applicable:
            raise ActionResolutionRequired(
                f"I couldn't find an applicable less-hours record for "
                f"{_format_date(target_date)}, so I can't safely prepare an "
                "exceptional-entry request for that day.",
                category="attendance_not_applicable",
            )
        selected_date = target_date
    else:
        candidates = sorted(applicable.items())
        if not candidates:
            raise ActionResolutionRequired(
                "I couldn't find an applicable less-hours day in that period, so I "
                "can't safely prepare an exceptional-entry request.",
                category="no_less_hours_days",
            )
        if len(candidates) > 1:
            choices = "\n".join(
                f"- {_format_date(day)} — {_format_duration(seconds)}"
                for day, seconds in candidates
            )
            raise ActionResolutionRequired(
                f"You have {len(candidates)} less-hours days in that period:\n\n"
                f"{choices}\n\nWhich date would you like to regularize?",
                category="multiple_less_hours_days",
                requires_clarification=True,
            )
        selected_date = candidates[0][0]

    suggestions = _require_list(
        await get_missing_punch_suggestions(
            selected_date.isoformat(),
            selected_date.isoformat(),
            lang=lang,
        ),
        "missing-punch suggestions",
    )
    matching_suggestions: list[tuple[datetime, int]] = []
    for suggestion in suggestions:
        try:
            suggested_time = _parse_suggested_datetime(
                _field(suggestion, "suggestedEntryTime")
            )
        except ValueError:
            continue
        entry_type = _suggestion_entry_type(_field(suggestion, "entryType"))
        if suggested_time.date() == selected_date and entry_type is not None:
            matching_suggestions.append((suggested_time, entry_type))
    if not matching_suggestions:
        raise ActionResolutionRequired(
            f"I found the less-hours record for {_format_date(selected_date)}, but "
            "ResourcePlus doesn't currently provide a suggested punch correction "
            "for that day, so I can't safely submit an exceptional-entry request for it.",
            category="no_resourceplus_suggestion",
        )
    if len(matching_suggestions) > 1:
        choices = "\n".join(
            f"- {'IN' if entry_type == 1 else 'OUT'} at {_format_time(suggested_time)}"
            for suggested_time, entry_type in matching_suggestions
        )
        raise ActionResolutionRequired(
            f"ResourcePlus provides more than one punch correction for "
            f"{_format_date(selected_date)}:\n\n{choices}\n\nWhich one should I use?",
            category="multiple_resourceplus_suggestions",
            requires_clarification=True,
        )

    reasons = _require_list(
        await get_exception_reasons(lang=lang),
        "exception reasons",
    )
    reason_names = [str(_field(reason, "reasonName") or "").strip() for reason in reasons]
    selected_reason_index = await match_live_reason(reason_name, reason_names)
    if selected_reason_index is None:
        available = [name for name in reason_names if name]
        choices = ", ".join(available[:6])
        suffix = f" Available reasons include: {choices}." if choices else ""
        raise ActionResolutionRequired(
            "I couldn't confidently match that reason to one ResourcePlus reason."
            f"{suffix} Which reason should I use?",
            category="reason_ambiguous",
            requires_clarification=True,
        )
    selected_reason = reasons[selected_reason_index]
    reason_id = _field(selected_reason, "reasonID")
    selected_reason_name = reason_names[selected_reason_index]
    if (
        reason_id is None
        or isinstance(reason_id, bool)
        or not str(reason_id).strip()
        or not selected_reason_name
    ):
        raise ValueError("ResourcePlus returned an invalid exceptional-entry reason.")

    selected_time, selected_entry_type = matching_suggestions[0]
    internal = {
        "entry_time": selected_time.isoformat(timespec="seconds"),
        "entry_type": selected_entry_type,
        "reason_id": str(reason_id),
        "reason_name": selected_reason_name,
        "remarks": remarks.strip(),
    }
    return ActionIntent(
        "create_exceptional_entry",
        internal,
        _confirmation_summary("create_exceptional_entry", internal, response_language),
        response_language,
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
