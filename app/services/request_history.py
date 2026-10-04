from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta
from typing import Any, Iterable, Literal

from app.ai.sessions import RequestCorrelation


REQUEST_HISTORY_PAST_DAYS = 120
REQUEST_HISTORY_FUTURE_DAYS = 365


RequestScope = Literal["all", "leave", "attendance_correction"]
RequestStatus = Literal["all", "pending", "approved", "rejected"]


@dataclass(frozen=True)
class RequestHistoryQuery:
    scope: RequestScope = "all"
    status: RequestStatus = "all"
    latest: bool = False
    all_approved: bool = False


@dataclass(frozen=True)
class EmployeeRequest:
    category: str
    friendly_category: str
    request_date: str
    detail: str
    raw_status: str
    resolved_status: str
    source: str
    source_reference: str = ""
    raw_status_history: tuple[str, ...] = ()
    source_history: tuple[str, ...] = ()


_ENGLISH_REQUEST_READ = re.compile(
    r"(?:"
    r"\b(?:show|list|display|view|check)\b.{0,55}\brequests?\b|"
    r"\b(?:my\s+requests?|request\s+status|my\s+(?:leave|vacation|business\s+travel)\s+requests?)\b|"
    r"\b(?:what|which)\b.{0,55}\brequests?\b|"
    r"\b(?:any|pending|approved|rejected)\s+requests?\b|"
    r"\brequests?\s+(?:are|were|is|was)\s+(?:pending|approved|rejected)\b|"
    r"\bwhat\s+happened\s+to\s+my\s+(?:latest\s+request|request|leave|attendance\s+correction)\b|"
    r"\bwhat\s+about\s+my\s+(?:leave|attendance\s+correction)\b|"
    r"\bis\s+anything\s+still\s+pending\b|"
    r"\bare\s+all\s+my\s+requests\s+approved\b|"
    r"\bwhat\s+requests\s+do\s+i\s+have\b"
    r")",
    re.I,
)


def _arabic_normalized(value: str) -> str:
    value = re.sub(r"[\u064b-\u065f\u0670\u0640]", "", value.casefold())
    value = value.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}))
    return " ".join(re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE).split())


def parse_request_history_query(message: str) -> RequestHistoryQuery | None:
    normalized = " ".join(message.casefold().split())
    arabic = _arabic_normalized(message)
    # A request noun plus a status word can occur inside cancellation or other
    # write commands (for example, "cancel my pending requests").  Those must
    # remain in the existing confirmation-protected workflow, not this read
    # fast path.
    if re.search(
        r"\b(?:cancel|delete|withdraw|apply|book|submit|correct|fix|approve|reject)\b",
        normalized,
    ) or any(
        arabic.startswith(prefix)
        for prefix in (
            "الغ ",
            "الغي ",
            "اسحب ",
            "قدم ",
            "احجز ",
            "صحح ",
            "وافق ",
            "ارفض ",
        )
    ):
        return None
    arabic_request = "طلب" in arabic
    arabic_read = arabic_request and any(
        cue in arabic
        for cue in (
            "اعرض", "اظهر", "ارني", "ورني", "وش", "ايش", "ما هي",
            "ماذا حدث", "حاله", "حالة", "هل عندي", "هل كل", "معلقه", "معلقة", "معتمده",
            "معتمدة", "مرفوضه", "مرفوضة",
        )
    )
    if not _ENGLISH_REQUEST_READ.search(message) and not arabic_read:
        return None

    scope: RequestScope = "all"
    if re.search(r"\b(?:attendance\s+correction|exceptional\s+entr(?:y|ies)|exception\s+requests?)\b", normalized) or any(
        cue in arabic for cue in ("تصحيح الحضور", "ادخال استثنائي", "الاستثناء")
    ):
        scope = "attendance_correction"
    elif re.search(r"\b(?:leave|vacation|day\s*type|business\s+travel)\b", normalized) or any(
        cue in arabic for cue in ("اجازه", "اجازة", "نوع اليوم", "مهمه عمل", "مهمة عمل")
    ):
        scope = "leave"

    status: RequestStatus = "all"
    if re.search(r"\b(?:pending|awaiting|still\s+waiting)\b", normalized) or any(
        cue in arabic for cue in ("معلق", "بانتظار", "قيد الموافقه", "قيد الموافقة")
    ):
        status = "pending"
    elif re.search(r"\bapproved\b", normalized) or any(
        cue in arabic for cue in ("معتمد", "تمت الموافقه", "تمت الموافقة")
    ):
        status = "approved"
    elif re.search(r"\brejected\b", normalized) or "مرفوض" in arabic:
        status = "rejected"

    all_approved = bool(
        re.search(r"\bare\s+all\s+my\s+requests\s+approved\b", normalized)
        or ("هل كل" in arabic and "معتمد" in arabic)
    )
    if all_approved:
        status = "all"

    singular_recent = bool(
        status == "all"
        and re.fullmatch(
            r"\s*(?:show|check)\s+my\s+request\s*[?.!]?\s*",
            message,
            re.I,
        )
    )
    latest = bool(
        re.search(r"\b(?:latest|most\s+recent)\b", normalized)
        or re.search(r"\bwhat\s+happened\s+to\s+my\s+(?:request|leave|attendance\s+correction)\b", normalized)
        or re.search(r"\bwhat\s+about\s+my\s+(?:leave|attendance\s+correction)\b", normalized)
        or singular_recent
        or "اخر طلب" in arabic
        or "ماذا حدث" in arabic
        or "وش صار" in arabic
        or "ايش صار" in arabic
    )
    return RequestHistoryQuery(
        scope=scope,
        status=status,
        latest=latest,
        all_approved=all_approved,
    )


def request_history_range(message: str, *, today: date) -> tuple[date, date, bool]:
    iso_values = re.findall(r"\b(20\d{2}-\d{2}-\d{2})\b", message)
    iso_dates: list[date] = []
    for value in iso_values[:2]:
        try:
            iso_dates.append(date.fromisoformat(value))
        except ValueError:
            continue
    if iso_dates:
        return min(iso_dates), max(iso_dates), True

    month_numbers = {
        "jan": 1, "january": 1, "feb": 2, "february": 2,
        "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
        "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8,
        "august": 8, "sep": 9, "sept": 9, "september": 9,
        "oct": 10, "october": 10, "nov": 11, "november": 11,
        "dec": 12, "december": 12,
    }
    month_pattern = "|".join(sorted(month_numbers, key=len, reverse=True))
    named_days = re.findall(
        rf"\b({month_pattern})\s+(\d{{1,2}})(?:,?\s+(20\d{{2}}))?\b",
        message,
        re.I,
    )
    parsed_named: list[date] = []
    for month_name, day_text, year_text in named_days[:2]:
        try:
            parsed_named.append(date(
                int(year_text or today.year),
                month_numbers[month_name.casefold()],
                int(day_text),
            ))
        except ValueError:
            continue
    if parsed_named:
        return min(parsed_named), max(parsed_named), True
    # Local import avoids the tools -> response blocks -> request history cycle.
    from app.ai.tools import resolve_relative_date_range

    resolved = resolve_relative_date_range(message, today=today)
    if resolved is not None:
        return resolved.from_date, resolved.to_date, True
    return (
        today - timedelta(days=REQUEST_HISTORY_PAST_DAYS),
        today + timedelta(days=REQUEST_HISTORY_FUTURE_DAYS),
        False,
    )


def _folded(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key).casefold(): value for key, value in row.items()}


def _field(row: dict[str, Any], *keys: str) -> str:
    folded = _folded(row)
    for key in keys:
        value = folded.get(key.casefold())
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _canonical_date(value: object) -> str:
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


def _normalized_status(value: str) -> str:
    return re.sub(r"[^a-z]", "", value.casefold())


def _resolved_status(raw_status: str) -> str:
    normalized = _normalized_status(raw_status)
    if normalized == "approved":
        return "Approved"
    if normalized in {"rejected", "declined"}:
        return "Rejected"
    if normalized in {
        "pending",
        "notapproved",
        "awaitingapproval",
        "submitted",
    }:
        return "Pending"
    if raw_status:
        return f"Existing request — {raw_status}"
    return "Existing request"


def _normalized_business_text(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", normalized, flags=re.UNICODE).split())


def _source_reference(record: dict[str, Any], *, attendance: bool) -> str:
    if attendance:
        return _field(
            record,
            "exceptionalID",
            "exceptionalEntryID",
            "exceptionID",
            "entryID",
            "requestID",
            "id",
        )
    return _field(
        record,
        "mappingID",
        "dayTypeMappingID",
        "requestID",
        "id",
    )


def _request_dedupe_key(item: EmployeeRequest) -> tuple[str, ...]:
    business_identity = (
        item.category,
        item.request_date,
        _normalized_business_text(item.detail),
        _normalized_business_text(item.resolved_status),
        item.source.casefold(),
    )
    if item.source_reference:
        return (*business_identity, "source_reference", item.source_reference.casefold())
    return (*business_identity, "business_identity")


def _dedupe_requests(
    requests: Iterable[EmployeeRequest],
) -> list[EmployeeRequest]:
    deduplicated: list[EmployeeRequest] = []
    positions: dict[tuple[str, ...], int] = {}
    for item in requests:
        key = _request_dedupe_key(item)
        existing_position = positions.get(key)
        if existing_position is not None:
            existing = deduplicated[existing_position]
            deduplicated[existing_position] = replace(
                existing,
                raw_status_history=tuple(dict.fromkeys((
                    *existing.raw_status_history,
                    item.raw_status,
                    *item.raw_status_history,
                ))),
                source_history=tuple(dict.fromkeys((
                    *existing.source_history,
                    item.source,
                    *item.source_history,
                ))),
            )
            continue
        positions[key] = len(deduplicated)
        deduplicated.append(replace(
            item,
            raw_status_history=item.raw_status_history or (item.raw_status,),
            source_history=item.source_history or (item.source,),
        ))
    return deduplicated


def normalize_request_history(
    payload: Any,
    *,
    source_hint: str = "all",
    correlations: Iterable[RequestCorrelation] = (),
) -> tuple[EmployeeRequest, ...]:
    wrapped_rows: list[tuple[str, str, dict[str, Any]]] = []
    if isinstance(payload, dict) and isinstance(payload.get("merged_requests"), list):
        for wrapped in payload["merged_requests"]:
            if not isinstance(wrapped, dict):
                continue
            record = wrapped.get("record")
            if not isinstance(record, dict):
                record = wrapped
            wrapped_rows.append((str(wrapped.get("request_kind") or ""), str(wrapped.get("raw_status") or ""), record))
    elif isinstance(payload, dict):
        for key, kind in (
            ("absence_requests", "absence"),
            ("exceptional_entry_requests", "exceptional_entry"),
        ):
            rows = payload.get(key)
            if isinstance(rows, list):
                wrapped_rows.extend(
                    (kind, _field(row, "status"), row)
                    for row in rows if isinstance(row, dict)
                )
    elif isinstance(payload, list):
        kind = "absence" if source_hint == "leave" else "exceptional_entry"
        wrapped_rows.extend(
            (kind, _field(row, "status"), row)
            for row in payload if isinstance(row, dict)
        )

    normalized_requests: list[EmployeeRequest] = []
    for kind, wrapper_status, record in wrapped_rows:
        compact_kind = re.sub(r"[^a-z]", "", kind.casefold())
        attendance = compact_kind in {"exceptionalentry", "exceptionentry"}
        category = "attendance_correction" if attendance else "leave"
        friendly = "Attendance correction" if attendance else "Leave"
        request_date = _canonical_date(_field(
            record, "date", "dateFrom", "fromDate", "AttDate", "attDate",
            "attendanceDate", "requestDate", "entryTime",
        ))
        detail = _field(
            record, "dayType", "DayType", "reasonName", "ReasonName", "detail",
            "reason", "description", "remarks",
        ) or friendly
        raw_status = wrapper_status or _field(record, "status", "requestStatus")
        normalized_requests.append(EmployeeRequest(
            category=category,
            friendly_category=friendly,
            request_date=request_date,
            detail=detail,
            raw_status=raw_status,
            resolved_status=_resolved_status(raw_status),
            source="ExceptionalEntries" if attendance else "DayTypeMapping/MyRequests",
            source_reference=_source_reference(record, attendance=attendance),
            raw_status_history=(raw_status,),
            source_history=(
                "ExceptionalEntries" if attendance else "DayTypeMapping/MyRequests",
            ),
        ))
    normalized_requests = _dedupe_requests(normalized_requests)
    for correlation in correlations:
        if correlation.state not in {"submitted_for_approval", "approved", "rejected"}:
            continue
        if any(
            item.category == correlation.category
            and item.request_date == correlation.request_date
            for item in normalized_requests
        ):
            continue
        friendly = (
            "Attendance correction"
            if correlation.category == "attendance_correction"
            else "Leave"
        )
        normalized_requests.append(EmployeeRequest(
            category=correlation.category,
            friendly_category=friendly,
            request_date=correlation.request_date,
            detail=correlation.detail or friendly,
            raw_status="",
            resolved_status=(
                "Pending"
                if correlation.state == "submitted_for_approval"
                else "Approved"
                if correlation.state == "approved"
                else "Rejected"
            ),
            source="session_correlation",
            raw_status_history=("",),
            source_history=("session_correlation",),
        ))
    return tuple(sorted(
        normalized_requests,
        key=lambda item: (item.request_date, item.friendly_category, item.detail),
        reverse=True,
    ))


def filter_request_history(
    requests: Iterable[EmployeeRequest],
    query: RequestHistoryQuery,
    *,
    recent_category: str | None = None,
    recent_date: str | None = None,
) -> tuple[EmployeeRequest, ...]:
    selected = tuple(
        item for item in requests
        if query.scope == "all" or item.category == query.scope
    )
    if query.status == "pending":
        selected = tuple(item for item in selected if item.resolved_status == "Pending")
    elif query.status == "approved":
        selected = tuple(item for item in selected if item.resolved_status == "Approved")
    elif query.status == "rejected":
        selected = tuple(item for item in selected if item.resolved_status == "Rejected")
    if query.latest and selected:
        contextual = tuple(
            item for item in selected
            if (not recent_category or item.category == recent_category)
            and (not recent_date or item.request_date == recent_date)
        )
        selected = contextual[:1] if contextual else selected[:1]
    return selected


def _human_date(value: str, language: str) -> str:
    try:
        parsed = date.fromisoformat(value[:10])
    except (TypeError, ValueError):
        return value
    return parsed.isoformat() if language == "ar" else f"{parsed.strftime('%b')} {parsed.day}"


def _request_description(item: EmployeeRequest, language: str) -> str:
    shown_date = _human_date(item.request_date, language)
    if language == "ar":
        if item.category == "attendance_correction":
            return f"تصحيح الحضور بسبب {item.detail} ليوم {shown_date}"
        return f"طلب {item.detail} ليوم {shown_date}"
    if item.category == "attendance_correction":
        return f"your {shown_date} {item.detail} attendance correction"
    return f"your {shown_date} {item.detail} request"


def _sentence_case(value: str) -> str:
    return value[:1].upper() + value[1:] if value else value


def request_history_message(
    requests: tuple[EmployeeRequest, ...],
    query: RequestHistoryQuery,
    *,
    language: str,
    unfiltered_requests: tuple[EmployeeRequest, ...],
) -> tuple[str, str]:
    count = len(requests)
    if query.all_approved:
        pending = sum(
            item.resolved_status == "Pending"
            for item in unfiltered_requests
        )
        rejected = sum(item.resolved_status == "Rejected" for item in unfiltered_requests)
        if not unfiltered_requests:
            message = "لا توجد لديك طلبات حديثة." if language == "ar" else "You don't have any recent requests."
        elif pending:
            message = (
                f"لا — لديك {pending} من الطلبات ما زالت بانتظار الموافقة."
                if language == "ar"
                else f"No — you have {pending} request{'s' if pending != 1 else ''} still waiting for approval."
            )
        elif rejected:
            message = (
                f"لا — لديك {rejected} من الطلبات مرفوضة."
                if language == "ar"
                else f"No — {rejected} of your recent requests {'were' if rejected != 1 else 'was'} rejected."
            )
        else:
            message = (
                "نعم — كل طلباتك الحديثة معتمدة."
                if language == "ar"
                else "Yes — all of your recent requests are approved."
            )
        return message, message

    if query.latest and count == 1:
        item = requests[0]
        description = _request_description(item, language)
        if language == "ar":
            if item.resolved_status == "Approved":
                message = f"تمت الموافقة على {description}."
            elif item.resolved_status == "Rejected":
                message = f"تم رفض {description}."
            elif item.resolved_status == "Pending":
                message = f"{description} ما زال بانتظار موافقة المدير."
            else:
                message = f"{description}: {item.resolved_status}."
        elif item.resolved_status == "Approved":
            message = f"{_sentence_case(description)} has been approved."
        elif item.resolved_status == "Rejected":
            message = f"{_sentence_case(description)} was rejected."
        elif item.resolved_status == "Pending":
            message = f"{_sentence_case(description)} is still awaiting manager approval."
        else:
            message = f"{_sentence_case(description)} is recorded as {item.resolved_status}."
        return message, message

    if query.status == "pending":
        if count == 0:
            message = "لا توجد طلبات بانتظار الموافقة." if language == "ar" else "You have no requests currently waiting for approval."
        elif count == 1:
            description = _request_description(requests[0], language)
            message = (
                f"لديك طلب واحد ما زال بانتظار الموافقة: {description}."
                if language == "ar"
                else f"You have one request still waiting for approval: {description}."
            )
        else:
            message = (
                f"لديك {count} طلبات ما زالت بانتظار الموافقة."
                if language == "ar"
                else f"You have {count} requests still waiting for approval."
            )
        return message, message

    if query.status == "approved":
        message = (
            (f"وجدت {count} من الطلبات المعتمدة في سجل طلباتك الحديث." if count else "لا توجد طلبات معتمدة في سجل طلباتك الحديث.")
            if language == "ar"
            else (f"I found {count} approved request{'s' if count != 1 else ''} in your recent request history." if count else "I found no approved requests in your recent request history.")
        )
        return message, message

    if query.status == "rejected":
        message = (
            (f"وجدت {count} من الطلبات المرفوضة في سجل طلباتك الحديث." if count else "لا توجد طلبات مرفوضة في سجل طلباتك الحديث.")
            if language == "ar"
            else (f"I found {count} rejected request{'s' if count != 1 else ''} in your recent request history." if count else "I found no rejected requests in your recent request history.")
        )
        return message, message

    if count == 0:
        message = "لا توجد لديك طلبات حديثة." if language == "ar" else "You don't have any recent requests."
    else:
        message = (
            f"هذه طلباتك الحديثة وعددها {count}."
            if language == "ar"
            else f"Here are your {count} recent request{'s' if count != 1 else ''}."
        )
    return message, message
