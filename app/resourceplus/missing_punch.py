from __future__ import annotations

import re
from dataclasses import dataclass, replace
from datetime import date, datetime
from typing import Any, Literal


PunchDirection = Literal["IN", "OUT"]
ENTRY_TYPE_NUMBER: dict[PunchDirection, int] = {"IN": 1, "OUT": 2}


@dataclass(frozen=True)
class MissingPunchRow:
    """One factual ResourcePlus missing-punch row in canonical backend form."""

    att_date: date
    entry_type: PunchDirection
    suggested_entry_time: str | None
    shift: str | None
    is_night_shift: int | None
    is_missing_punch: bool
    is_correctable: bool


@dataclass(frozen=True)
class MissingPunchNormalization:
    rows: tuple[MissingPunchRow, ...]
    invalid_row_count: int

    @property
    def missing_punches(self) -> tuple[MissingPunchRow, ...]:
        return tuple(row for row in self.rows if row.is_missing_punch)

    @property
    def correctable_suggestions(self) -> tuple[MissingPunchRow, ...]:
        return tuple(row for row in self.rows if row.is_correctable)


def apply_attendance_eligibility(
    normalized: MissingPunchNormalization,
    eligible_dates: set[date],
) -> MissingPunchNormalization:
    """Remove legacy correction availability unless v2 attendance agrees."""

    return MissingPunchNormalization(
        rows=tuple(
            replace(
                row,
                is_correctable=(
                    row.is_correctable and row.att_date in eligible_dates
                ),
            )
            for row in normalized.rows
        ),
        invalid_row_count=normalized.invalid_row_count,
    )


def parse_missing_punch_date(value: object) -> date:
    """Resolve API and user-facing representations to one calendar date."""

    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if not isinstance(value, str) or not value.strip():
        raise ValueError("A valid missing-punch date is required.")
    normalized = value.strip()
    for date_format in (
        "%Y-%m-%d",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%B %d, %Y",
        "%b %d, %Y",
        # ResourcePlus currently also emits this .NET-style midnight value.
        "%m/%d/%Y %I:%M:%S %p",
    ):
        try:
            return datetime.strptime(normalized, date_format).date()
        except ValueError:
            continue
    raise ValueError("A valid missing-punch date is required.")


def _is_valid_suggested_entry_time(value: object, att_date: date) -> bool:
    if not isinstance(value, str) or not re.fullmatch(
        r"\d{2}/\d{2}/\d{4} \d{2}:\d{2}",
        value,
    ):
        return False
    try:
        parsed = datetime.strptime(value, "%d/%m/%Y %H:%M")
    except ValueError:
        return False
    return parsed.strftime("%d/%m/%Y %H:%M") == value and parsed.date() == att_date


def normalize_missing_punch_suggestions(payload: Any) -> MissingPunchNormalization:
    """Normalize factual missing punches and independently mark correction safety.

    `suggestedEntryTime` is deliberately retained byte-for-byte as a Python string.
    Date parsing is only for matching; it never rewrites the authoritative POST value.
    """

    if not isinstance(payload, list):
        raise ValueError(
            "ResourcePlus returned an unexpected missing-punch suggestions response."
        )

    rows: list[MissingPunchRow] = []
    invalid = 0
    for row in payload:
        if not isinstance(row, dict):
            invalid += 1
            continue
        raw_date = row.get("attDate")
        raw_time = row.get("suggestedEntryTime")
        raw_direction = row.get("entryType")
        try:
            att_date = parse_missing_punch_date(raw_date)
        except (TypeError, ValueError):
            invalid += 1
            continue
        if not isinstance(raw_direction, str) or raw_direction not in ENTRY_TYPE_NUMBER:
            invalid += 1
            continue

        shift_value = row.get("shift")
        night_value = row.get("isNightShift")
        rows.append(
            MissingPunchRow(
                att_date=att_date,
                entry_type=raw_direction,
                suggested_entry_time=raw_time if isinstance(raw_time, str) else None,
                shift=shift_value if isinstance(shift_value, str) else None,
                is_night_shift=(
                    night_value
                    if isinstance(night_value, int)
                    and not isinstance(night_value, bool)
                    and night_value in {0, 1}
                    else None
                ),
                is_missing_punch=True,
                is_correctable=_is_valid_suggested_entry_time(raw_time, att_date),
            )
        )

    return MissingPunchNormalization(tuple(rows), invalid)


def missing_punch_tool_data(
    normalized: MissingPunchNormalization,
) -> dict[str, object]:
    """Return grouped, actionable-only facts for informational assistant answers."""

    def grouped_rows(
        rows: tuple[MissingPunchRow, ...],
        *,
        item_key: str,
    ) -> list[dict[str, object]]:
        grouped: dict[date, list[MissingPunchRow]] = {}
        for row in rows:
            grouped.setdefault(row.att_date, []).append(row)
        return [
            {
                "att_date": att_date.isoformat(),
                item_key: [
                    {
                        "entry_type": item.entry_type,
                        "suggested_entry_time": item.suggested_entry_time,
                        "shift": item.shift,
                        "is_night_shift": item.is_night_shift,
                        "is_missing_punch": item.is_missing_punch,
                        "is_correctable": item.is_correctable,
                    }
                    for item in items
                ],
            }
            for att_date, items in sorted(grouped.items())
        ]

    missing_by_date = grouped_rows(
        normalized.missing_punches,
        item_key="missing_punches",
    )
    correctable_by_date = grouped_rows(
        normalized.correctable_suggestions,
        item_key="suggestions",
    )
    result: dict[str, object] = {
        "missing_punches_by_date": missing_by_date,
        "missing_punch_count": len(normalized.missing_punches),
        "missing_punch_date_count": len(missing_by_date),
        "correctable_suggestions_by_date": correctable_by_date,
        "correctable_suggestion_count": len(normalized.correctable_suggestions),
        "non_correctable_missing_punch_count": (
            len(normalized.missing_punches)
            - len(normalized.correctable_suggestions)
        ),
        "invalid_row_count": normalized.invalid_row_count,
    }
    if missing_by_date and not correctable_by_date:
        result["message"] = (
            "ResourcePlus reports missing punches on these dates but has not provided "
            "valid suggested correction times, so they cannot currently be submitted "
            "automatically."
        )
    elif not missing_by_date:
        result["message"] = (
            "ResourcePlus does not currently report any valid missing IN or OUT "
            "punches for that date range."
        )
    return result
