from __future__ import annotations

import re
from datetime import date, datetime
from typing import Any, Iterable

from app.ai.sessions import ApprovalCandidate


_MONTHS = {
    "jan": 1, "january": 1, "feb": 2, "february": 2,
    "mar": 3, "march": 3, "apr": 4, "april": 4, "may": 5,
    "jun": 6, "june": 6, "jul": 7, "july": 7, "aug": 8,
    "august": 8, "sep": 9, "sept": 9, "september": 9,
    "oct": 10, "october": 10, "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}


def _folded(row: dict[str, Any]) -> dict[str, Any]:
    return {str(key).casefold(): value for key, value in row.items()}


def _field(row: dict[str, Any], *keys: str) -> str:
    folded = _folded(row)
    for key in keys:
        value = folded.get(key.casefold())
        if value not in (None, ""):
            return str(value).strip()
    return ""


def _rows(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [row for row in payload if isinstance(row, dict)]
    if isinstance(payload, dict):
        folded = _folded(payload)
        for key in ("PendingApprovals", "Requests", "Data"):
            value = folded.get(key.casefold())
            if isinstance(value, list):
                return [row for row in value if isinstance(row, dict)]
    return []


def friendly_approval_category(request_type: str, detail: str) -> str:
    normalized = re.sub(r"[^a-z]", "", request_type.casefold())
    if normalized in {"exceptionentry", "exceptionalentry"}:
        return "Attendance correction"
    if normalized in {"absence", "leave", "daytype"}:
        if "business travel" in detail.casefold():
            return "Business Travel"
        return "Leave"
    return " ".join(request_type.replace("_", " ").split()) or "Request"


def approval_candidates(payload: Any) -> tuple[ApprovalCandidate, ...]:
    candidates: list[ApprovalCandidate] = []
    for row in _rows(payload):
        request_id = _field(row, "requestId")
        request_type = _field(row, "requestType", "type")
        employee = _field(row, "employeeName", "EmployeeName", "name")
        if not request_id or not request_type or not employee:
            continue
        detail = _field(row, "detail", "description", "dayType", "reasonName")
        request_date = _field(
            row,
            "date",
            "dateFrom",
            "fromDate",
            "AttDate",
            "attendanceDate",
            "requestDate",
        )
        candidates.append(
            ApprovalCandidate(
                ordinal=len(candidates) + 1,
                request_id=request_id,
                request_type=request_type,
                employee_name=employee,
                category=friendly_approval_category(request_type, detail),
                detail=detail or "Request",
                request_date=request_date,
                status=_field(row, "status") or "Pending",
            )
        )
    return tuple(candidates)


def _normalized(value: str) -> str:
    value = value.casefold().replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    return " ".join(re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE).split())


def _candidate_date(value: str) -> date | None:
    text = value.strip()
    for candidate in (text[:10], text):
        try:
            return date.fromisoformat(candidate)
        except ValueError:
            pass
    for pattern in ("%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y"):
        try:
            return datetime.strptime(text[:10], pattern).date()
        except ValueError:
            continue
    return None


def _date_matches(message: str, value: str) -> bool:
    parsed = _candidate_date(value)
    if parsed is None:
        return False
    normalized = _normalized(message)
    if parsed.isoformat() in message:
        return True
    for name, month in _MONTHS.items():
        if month == parsed.month and re.search(
            rf"\b{re.escape(name)}\s+{parsed.day}\b", normalized
        ):
            return True
    return False


def _has_explicit_date_reference(message: str) -> bool:
    normalized = _normalized(message)
    if re.search(r"\b20\d{2}-\d{2}-\d{2}\b", normalized):
        return True
    month_names = "|".join(sorted(_MONTHS, key=len, reverse=True))
    return re.search(rf"\b(?:{month_names})\s+\d{{1,2}}\b", normalized) is not None


def _ordinal_reference(message: str) -> int | str | None:
    normalized = _normalized(message)
    if re.search(r"\b(?:first(?:\s+one)?|الاول|الاولى)\b", normalized):
        return 1
    if re.search(r"\b(?:second(?:\s+one)?|الثاني|الثانية)\b", normalized):
        return 2
    if re.search(r"\b(?:last(?:\s+one)?|الاخير|الاخيرة)\b", normalized):
        return "last"
    numbered = re.search(r"\b(?:request|طلب|الطلب)\s*(\d{1,3})\b", normalized)
    return int(numbered.group(1)) if numbered else None


def resolve_pending_approval(
    candidates: Iterable[ApprovalCandidate],
    message: str,
    *,
    ordinal: int | None = None,
) -> tuple[ApprovalCandidate, ...]:
    """Resolve a displayed request deterministically; never guess among matches."""

    available = tuple(candidates)
    if ordinal is not None:
        return tuple(candidate for candidate in available if candidate.ordinal == ordinal)

    ordinal_reference = _ordinal_reference(message)
    if ordinal_reference == "last":
        return (available[-1],) if available else ()
    if isinstance(ordinal_reference, int):
        return (
            (available[ordinal_reference - 1],)
            if 1 <= ordinal_reference <= len(available)
            else ()
        )

    normalized = _normalized(message)
    message_words = set(normalized.split())
    selected = available
    used_filter = False

    name_matches = tuple(
        candidate
        for candidate in selected
        if (
            _normalized(candidate.employee_name) in normalized
            or any(
                part in message_words
                for part in _normalized(candidate.employee_name).split()
                if len(part) >= 3
            )
        )
    )
    if name_matches:
        selected = name_matches
        used_filter = True

    detail_matches = tuple(
        candidate
        for candidate in selected
        if _normalized(candidate.detail) in normalized
    )
    if detail_matches:
        selected = detail_matches
        used_filter = True

    category_requested: str | None = None
    if any(cue in normalized for cue in ("attendance correction", "correction", "تصحيح", "حضور")):
        category_requested = "Attendance correction"
    elif re.search(r"\b(?:leave|vacation)\b", normalized) or any(
        cue in normalized for cue in ("اجازة", "اجازه")
    ):
        category_requested = "Leave"
    if category_requested is not None:
        category_matches = tuple(
            candidate for candidate in selected
            if candidate.category == category_requested
        )
        if not category_matches:
            return ()
        selected = category_matches
        used_filter = True

    date_matches = tuple(
        candidate
        for candidate in selected
        if _date_matches(message, candidate.request_date)
    )
    if date_matches:
        selected = date_matches
        used_filter = True
    elif _has_explicit_date_reference(message):
        return ()

    return selected if used_filter else available
