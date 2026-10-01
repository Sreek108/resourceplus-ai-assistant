from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime
from typing import Any

from app.models.schemas import (
    ActionsBlock,
    BlockAction,
    BlockColumn,
    BlockItem,
    ConfirmationBlock,
    KeyValueBlock,
    ListBlock,
    NoticeBlock,
    ResponseBlock,
    StatCardsBlock,
    TableBlock,
)


Scalar = str | int | float | bool | None
MAX_MARKDOWN_ROWS = 8


def _scalar(value: object) -> Scalar:
    return value if isinstance(value, (str, int, float, bool)) or value is None else str(value)


def _first(record: dict[str, Any], *keys: str, default: Scalar = "—") -> Scalar:
    folded = {str(key).casefold(): value for key, value in record.items()}
    for key in keys:
        value = folded.get(key.casefold())
        if value not in (None, ""):
            return _scalar(value)
    return default


def _records(payload: Any, *container_names: str) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if not isinstance(payload, dict):
        return []
    folded = {str(key).casefold(): value for key, value in payload.items()}
    for name in container_names:
        value = folded.get(name.casefold())
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def _exceptional_entry_date(row: dict[str, Any]) -> Scalar:
    value = _first(
        row,
        "entryTime",
        "attDate",
        "attendanceDate",
        "date",
        "requestDate",
    )
    if isinstance(value, (date, datetime)):
        return value.date().isoformat() if isinstance(value, datetime) else value.isoformat()
    if not isinstance(value, str):
        return value
    normalized = value.strip()
    try:
        return datetime.fromisoformat(normalized.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    for date_format in (
        "%d/%m/%Y %H:%M",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(normalized, date_format).date().isoformat()
        except ValueError:
            continue
    return normalized


def _exceptional_entry_type(row: dict[str, Any], language: str = "en") -> Scalar:
    value = _first(row, "entryTypeName", "type", "entryType")
    normalized = str(value).strip() if value not in (None, "") else ""
    if normalized == "1":
        return "وصول متأخر" if language == "ar" else "Late Arrival"
    if normalized == "2":
        return "خروج مبكر" if language == "ar" else "Early Departure"
    if language == "ar" and normalized.casefold() == "late arrival":
        return "وصول متأخر"
    if language == "ar" and normalized.casefold() == "early departure":
        return "خروج مبكر"
    return value


def _exceptional_entry_status(row: dict[str, Any], language: str = "en") -> Scalar:
    value = _first(row, "status", "requestStatus")
    if language != "ar" or not isinstance(value, str):
        return value
    localized = {
        "not approved": "غير موافق عليه",
        "approved": "موافق عليه",
        "rejected": "مرفوض",
        "pending": "معلّق",
        "pending approval": "بانتظار الموافقة",
    }
    return localized.get(value.strip().casefold(), value)


def _exceptional_entry_rows(
    payload: Any,
    language: str = "en",
) -> list[dict[str, Scalar]]:
    return [
        {
            "date": _exceptional_entry_date(row),
            "type": _exceptional_entry_type(row, language),
            "reason": _first(row, "reason", "reasonName", "description"),
            "status": _exceptional_entry_status(row, language),
        }
        for row in _records(payload, "ExceptionalEntries", "Entries", "Requests", "Data")
    ]


def markdown_table(block: TableBlock, *, row_limit: int = MAX_MARKDOWN_ROWS) -> str:
    columns = block.columns
    header = "| " + " | ".join(column.label for column in columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for row in block.rows[:row_limit]:
        values = [str(row.get(column.key, "—")).replace("|", "\\|") for column in columns]
        rows.append("| " + " | ".join(values) + " |")
    suffix = ""
    if len(block.rows) > row_limit:
        suffix = f"\n\nShowing {row_limit} of {len(block.rows)} rows."
    return "\n".join([header, separator, *rows]) + suffix


def profile_block(payload: Any) -> KeyValueBlock | None:
    if not isinstance(payload, dict):
        return None
    source = payload
    for candidate in ("Profile", "Employee", "Data"):
        nested = payload.get(candidate)
        if isinstance(nested, dict):
            source = nested
            break
    labels = {
        "employeename": "Name",
        "employeeemail": "Email",
        "email": "Email",
        "employeecode": "Employee code",
        "department": "Department",
        "designation": "Designation",
        "position": "Position",
        "company": "Company",
        "location": "Location",
        "joiningdate": "Joining date",
    }
    items = [
        BlockItem(label=labels.get(str(key).casefold(), str(key)), value=_scalar(value))
        for key, value in source.items()
        if isinstance(value, (str, int, float, bool)) and value not in (None, "")
        and not str(key).casefold().endswith("id")
        and str(key).casefold() not in {"instance", "instancename", "tenant"}
    ]
    return KeyValueBlock(title="Profile", items=items) if items else None


def attendance_blocks(payload: Any) -> list[ResponseBlock]:
    blocks: list[ResponseBlock] = []
    if isinstance(payload, dict):
        counts = _records(payload, "Attendance Counts", "AttendanceCounts", "Summary")
        if counts:
            items: list[BlockItem] = []
            for row in counts:
                label = _first(row, "Status", "Name", "Type", "Label")
                value = _first(row, "Count", "Value", "Days")
                if label != "—" and value != "—":
                    items.append(BlockItem(label=str(label), value=value))
            if items:
                blocks.append(StatCardsBlock(title="Attendance summary", items=items))

    days = _records(payload, "Days", "Attendance", "AttendanceDetails", "Details")
    if not days and isinstance(payload, list):
        days = _records(payload)
    rows = [
        {
            "date": _first(row, "AttDate", "Date", "AttendanceDate"),
            "status": _first(row, "DayType", "Status", "AttendanceStatus", "DayStatus"),
            "in": _first(row, "CheckIN", "In", "InTime", "PunchIn", "FirstIn"),
            "out": _first(row, "CheckOut", "Out", "OutTime", "PunchOut", "LastOut"),
            "worked": _first(row, "NetHrs", "WorkedHours", "Worked"),
            "shortfall": _first(row, "LessHrs", "Shortfall"),
        }
        for row in days
    ]
    if rows:
        blocks.append(
            TableBlock(
                title="Attendance",
                columns=[
                    BlockColumn(key="date", label="Date"),
                    BlockColumn(key="status", label="Status"),
                    BlockColumn(key="in", label="IN"),
                    BlockColumn(key="out", label="OUT"),
                    BlockColumn(key="worked", label="Worked"),
                    BlockColumn(key="shortfall", label="Shortfall"),
                ],
                rows=rows,
            )
        )
    return blocks


def missing_punch_block(payload: Any) -> TableBlock:
    groups = _records(payload, "missing_punches_by_date")
    rows: list[dict[str, Scalar]] = []
    for group in groups:
        attendance_date = _first(group, "attendance_date", "date")
        for punch in _records(group, "missing_punches"):
            rows.append(
                {
                    "date": attendance_date,
                    "direction": _first(punch, "entry_type"),
                    "suggested": _first(punch, "suggested_entry_time"),
                    "correctable": bool(punch.get("is_correctable")),
                }
            )
    return TableBlock(
        title="Missing punches",
        columns=[
            BlockColumn(key="date", label="Date"),
            BlockColumn(key="direction", label="Punch"),
            BlockColumn(key="suggested", label="Suggested time"),
            BlockColumn(key="correctable", label="Correctable"),
        ],
        rows=rows,
    )


def day_types_block(payload: Any) -> TableBlock:
    rows = [
        {
            "type": _first(row, "dayType", "name"),
            "group": _first(row, "group", "category"),
        }
        for row in _records(payload)
    ]
    return TableBlock(
        title="Available day types",
        columns=[
            BlockColumn(key="type", label="Day type"),
            BlockColumn(key="group", label="Group"),
        ],
        rows=rows,
    )


def notifications_block(payload: Any) -> ListBlock:
    items = []
    for row in _records(payload):
        title = _first(row, "NotifcnTitle", "Title", "Subject", default="Notification")
        detail = _first(row, "NotifcnText", "Message", "Description", "Body", default="")
        when = _first(row, "NotifcnDate", "Date", default="")
        text = " — ".join(str(value) for value in (title, detail, when) if value not in ("", "—"))
        if text:
            items.append(text)
    return ListBlock(title="Notifications", items=items)


def request_status_block(payload: Any) -> TableBlock:
    records = _records(payload, "merged_requests")
    rows = []
    for wrapped in records:
        record = wrapped.get("record") if isinstance(wrapped.get("record"), dict) else wrapped
        rows.append(
            {
                "type": _first(wrapped, "request_kind", "requestType", "type"),
                "date": _first(record, "date", "dateFrom", "AttDate", "requestDate"),
                "detail": _first(record, "dayType", "reasonName", "detail", "description"),
                "status": _first(wrapped, "raw_status", "status", default=_first(record, "status")),
            }
        )
    return TableBlock(
        title="My requests",
        columns=[
            BlockColumn(key="type", label="Type"),
            BlockColumn(key="date", label="Date"),
            BlockColumn(key="detail", label="Detail"),
            BlockColumn(key="status", label="Status"),
        ],
        rows=rows,
    )


def approvals_block(payload: Any) -> TableBlock:
    rows = [
        {
            "employee": _first(row, "employeeName", "EmployeeName", "name"),
            "type": _first(row, "requestType", "type"),
            "detail": _first(row, "detail", "description", "dayType", "reasonName"),
            "status": _first(row, "status", default="Pending"),
        }
        for row in _records(payload, "PendingApprovals", "Requests", "Data")
    ]
    return TableBlock(
        title="Pending approvals",
        columns=[
            BlockColumn(key="employee", label="Employee"),
            BlockColumn(key="type", label="Type"),
            BlockColumn(key="detail", label="Detail"),
            BlockColumn(key="status", label="Status"),
        ],
        rows=rows,
    )


def reason_actions(options: Iterable[str], language: str = "en") -> ActionsBlock:
    return ActionsBlock(
        title="اختر السبب" if language == "ar" else "Choose a reason",
        actions=[BlockAction(label=value, value=value) for value in options],
    )


def confirmation_block(summary: str, language: str = "en") -> ConfirmationBlock:
    if language == "ar":
        return ConfirmationBlock(
            title="التأكيد مطلوب",
            summary=summary,
            actions=[
                BlockAction(label="تأكيد", value="confirm", style="primary"),
                BlockAction(label="إلغاء", value="cancel", style="secondary"),
            ],
        )
    return ConfirmationBlock(title="Confirmation required", summary=summary)


def less_hours_block(days: Iterable[object], language: str = "en") -> TableBlock:
    eligibility_labels = (
        {
            "eligible": "قابل للتصحيح",
            "no_punches": "لا توجد بصمات حضور",
            "week_end": "عطلة أسبوعية",
            "holiday": "عطلة",
            "leave": "إجازة",
            "business_travel": "مهمة عمل",
            "no_missing_hours": "لا توجد ساعات ناقصة",
        }
        if language == "ar"
        else {
            "eligible": "Correction candidate",
            "no_punches": "No attendance punches",
            "week_end": "Week End",
            "holiday": "Holiday",
            "leave": "Leave",
            "business_travel": "Business Travel",
            "no_missing_hours": "No missing hours",
        }
    )
    rows = []
    for day in days:
        rows.append(
            {
                "date": getattr(day, "attendance_date").isoformat(),
                "day_type": getattr(day, "day_type"),
                "in": getattr(day, "check_in") or "—",
                "out": getattr(day, "check_out") or "—",
                "worked": getattr(day, "worked_hours"),
                "less": getattr(day, "less_hours"),
                "action": eligibility_labels.get(
                    getattr(day, "eligibility"),
                    str(getattr(day, "eligibility")),
                ),
            }
        )
    return TableBlock(
        title="الحضور بساعات ناقصة" if language == "ar" else "Less-hours attendance",
        columns=(
            [
                BlockColumn(key="date", label="التاريخ"),
                BlockColumn(key="day_type", label="نوع اليوم"),
                BlockColumn(key="in", label="الدخول"),
                BlockColumn(key="out", label="الخروج"),
                BlockColumn(key="worked", label="ساعات العمل"),
                BlockColumn(key="less", label="الساعات الناقصة"),
                BlockColumn(key="action", label="الحالة"),
            ]
            if language == "ar"
            else [
                BlockColumn(key="date", label="Date"),
                BlockColumn(key="day_type", label="Day Type"),
                BlockColumn(key="in", label="Check In"),
                BlockColumn(key="out", label="Check Out"),
                BlockColumn(key="worked", label="Worked Hours"),
                BlockColumn(key="less", label="Less Hours"),
                BlockColumn(key="action", label="Action"),
            ]
        ),
        rows=rows,
    )


def less_hours_correction_actions(
    days: Iterable[object],
    language: str = "en",
) -> ActionsBlock | None:
    actions = [
        BlockAction(
            label=(
                f"صحح {getattr(day, 'attendance_date').isoformat()}"
                if language == "ar"
                else f"Correct {getattr(day, 'attendance_date').isoformat()}"
            ),
            value=(
                (
                    "صحح الساعات الناقصة يوم "
                    if language == "ar"
                    else "Correct my less hours on "
                )
                + f"{getattr(day, 'attendance_date').isoformat()}"
            ),
            style="primary",
        )
        for day in days
    ]
    title = "الإجراءات المتاحة" if language == "ar" else "Available actions"
    return ActionsBlock(title=title, actions=actions) if actions else None


def exceptional_balance_block(payload: Any, language: str = "en") -> KeyValueBlock | None:
    if not isinstance(payload, dict) or payload.get("hasPolicy") is not True:
        return None
    labels = (
        (
            ("السياسة", "policyName"),
            ("الحد", "limitValue"),
            ("المستخدم", "used"),
            ("المتبقي", "remaining"),
            ("دورة إعادة التعيين", "resetPeriod"),
            ("بداية الفترة", "periodStart"),
            ("نهاية الفترة", "periodEnd"),
            ("إعادة التعيين في", "resetsOn"),
        )
        if language == "ar"
        else (
            ("Policy", "policyName"),
            ("Limit", "limitValue"),
            ("Used", "used"),
            ("Remaining", "remaining"),
            ("Reset period", "resetPeriod"),
            ("Period start", "periodStart"),
            ("Period end", "periodEnd"),
            ("Resets on", "resetsOn"),
        )
    )
    items = [
        BlockItem(label=label, value=_scalar(payload[key]))
        for label, key in labels
        if key in payload and payload[key] not in (None, "")
    ]
    limit_type = payload.get("limitType")
    if limit_type == 1:
        items.insert(1, BlockItem(label="الوحدة" if language == "ar" else "Unit", value="عدد الحالات" if language == "ar" else "Exception count"))
    elif limit_type == 2:
        items.insert(1, BlockItem(label="الوحدة" if language == "ar" else "Unit", value="دقائق" if language == "ar" else "Minutes"))
    title = "رصيد السماح" if language == "ar" else "Exceptional-entry allowance"
    return KeyValueBlock(title=title, items=items) if items else None


def exceptional_balance_blocks(payload: Any, language: str = "en") -> list[ResponseBlock]:
    block = exceptional_balance_block(payload, language)
    if block is not None:
        return [block]
    if isinstance(payload, dict) and payload.get("hasPolicy") is False:
        return [
            NoticeBlock(
                title=(
                    "بدل إدخال الحضور الاستثنائي"
                    if language == "ar"
                    else "Exceptional-entry allowance"
                ),
                message=(
                    "لا توجد لديك سياسة رصيد سماح لهذا التاريخ."
                    if language == "ar"
                    else "You do not have an allowance policy for that date."
                ),
                level="info",
            )
        ]
    return []


def exceptional_entries_block(
    payload: Any,
    *,
    title: str | None = None,
    language: str = "en",
) -> TableBlock:
    return TableBlock(
        title=title or ("طلبات الاستثناء" if language == "ar" else "Exceptional entries"),
        columns=(
            [
                BlockColumn(key="date", label="التاريخ"),
                BlockColumn(key="type", label="النوع"),
                BlockColumn(key="reason", label="السبب"),
                BlockColumn(key="status", label="الحالة"),
            ]
            if language == "ar"
            else [
                BlockColumn(key="date", label="Date"),
                BlockColumn(key="type", label="Type"),
                BlockColumn(key="reason", label="Reason"),
                BlockColumn(key="status", label="Status"),
            ]
        ),
        rows=_exceptional_entry_rows(payload, language),
    )


def cancellable_exceptions_block(
    rows: Iterable[dict[str, Any]],
    language: str = "en",
) -> TableBlock:
    return exceptional_entries_block(
        list(rows),
        title="طلبات استثناء قابلة للإلغاء" if language == "ar" else "Cancellable exceptional entries",
        language=language,
    )


def cancellable_exception_actions(
    candidates: Iterable[dict[str, str]],
    language: str = "en",
) -> ActionsBlock | None:
    actions: list[BlockAction] = []
    for candidate in candidates:
        key = candidate.get("key", "").strip()
        if not key:
            continue
        raw_date = candidate.get("date", "")
        try:
            parsed = date.fromisoformat(raw_date)
            if language == "ar":
                months = ("", "يناير", "فبراير", "مارس", "أبريل", "مايو", "يونيو", "يوليو", "أغسطس", "سبتمبر", "أكتوبر", "نوفمبر", "ديسمبر")
                date_label = f"{parsed.day} {months[parsed.month]}"
            else:
                date_label = f"{parsed.day} {parsed.strftime('%b')}"
        except ValueError:
            date_label = raw_date or "Entry"
        label_parts = [
            date_label,
            candidate.get("reason", "").strip(),
            (
                "وصول متأخر"
                if language == "ar" and candidate.get("type", "").casefold() == "late arrival"
                else "خروج مبكر"
                if language == "ar" and candidate.get("type", "").casefold() == "early departure"
                else candidate.get("type", "").strip()
            ),
        ]
        label = " · ".join(part for part in label_parts if part and part != "—")
        actions.append(
            BlockAction(
                label=label,
                value=f"Select exceptional entry {key}",
            )
        )
    title = "اختر الطلب" if language == "ar" else "Select an entry"
    return ActionsBlock(title=title, actions=actions) if actions else None


def exceptional_submission_blocks(
    payload: Any,
    *,
    language: str,
    success: bool | None = None,
) -> list[ResponseBlock]:
    if not isinstance(payload, dict):
        return []
    blocks: list[ResponseBlock] = []
    resolved_success = (
        (
            payload.get("success") is True
            if "success" in payload
            else isinstance(payload.get("isAutoApproved"), bool)
        )
        if success is None
        else success
    )
    auto_approved = payload.get("isAutoApproved")
    if not resolved_success:
        title = "لم يكتمل التصحيح" if language == "ar" else "Correction not completed"
    elif language == "ar" and auto_approved is True:
        title = "تمت الموافقة تلقائياً"
    elif language == "ar":
        title = "تم إرسال التصحيح"
    elif auto_approved is True:
        title = "Correction auto-approved"
    else:
        title = "Correction submitted"
    message = payload.get("message")
    if not isinstance(message, str) or not message.strip():
        if not resolved_success:
            message = (
                "تعذّر إكمال تصحيح حضورك."
                if language == "ar"
                else "Your attendance correction could not be completed."
            )
        elif auto_approved is True:
            message = (
                "تمت الموافقة تلقائيًا على تصحيح حضورك."
                if language == "ar"
                else "Your attendance correction was auto-approved."
            )
        elif auto_approved is False:
            message = (
                "تم إرسال تصحيح حضورك وهو بانتظار موافقة المدير."
                if language == "ar"
                else "Your attendance correction was submitted for manager approval."
            )
        else:
            message = (
                "تم إكمال تصحيح حضورك."
                if language == "ar"
                else "Your attendance correction was completed."
            )
    blocks.append(
        NoticeBlock(
            title=title,
            message=message.strip(),
            level="success" if resolved_success else "error",
        )
    )
    warning = payload.get("warning")
    if isinstance(warning, str) and warning.strip():
        blocks.append(
            NoticeBlock(
                title="Warning" if language != "ar" else "تنبيه",
                message=warning.strip(),
                level="warning",
            )
        )
    entries = _records(payload, "entries") if resolved_success else []
    if entries:
        blocks.append(
            TableBlock(
                title="Created entries",
                columns=[
                    BlockColumn(key="entry", label="Entry"),
                    BlockColumn(key="type", label="Type"),
                    BlockColumn(key="minutes", label="Minutes"),
                    BlockColumn(key="status", label="Status"),
                ],
                rows=[
                    {
                        "entry": _first(row, "entry", "name", "description", default=str(index)),
                        "type": _first(row, "entryTypeName", "type", "entryType"),
                        "minutes": _first(row, "minutes", "requestedMinutes"),
                        "status": _first(row, "status"),
                    }
                    for index, row in enumerate(entries, start=1)
                ],
            )
        )
    if (
        not resolved_success
        and isinstance(message, str)
        and "apply leave" in message.casefold()
    ):
        blocks.append(
            ActionsBlock(
                title="Next step" if language != "ar" else "الخطوة التالية",
                actions=[
                    BlockAction(
                        label=(
                            "View Leave & Business Travel options"
                            if language != "ar"
                            else "عرض خيارات الإجازة ومهمة العمل"
                        ),
                        value=(
                            "Show available day types"
                            if language != "ar"
                            else "اعرض خيارات الإجازة ومهمة العمل"
                        ),
                        style="secondary",
                    )
                ],
            )
        )
    return blocks
