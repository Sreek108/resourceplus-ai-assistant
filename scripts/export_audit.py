"""Export the existing ResourcePlus assistant SQLite interaction audit."""

from __future__ import annotations

import argparse
import asyncio
import csv
import io
import json
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.audit import AuditStore  # noqa: E402
from app.config import get_settings  # noqa: E402


CSV_FIELDS = (
    "timestamp",
    "trace_id",
    "interaction_id",
    "app_version",
    "git_commit",
    "environment",
    "deployment_id",
    "session_ref",
    "mode",
    "input_source",
    "user_text",
    "detected_locale",
    "resolved_language",
    "response_language",
    "display_message",
    "speech_message",
    "tts_text",
    "tts_requested",
    "tts_generated",
    "tts_locale",
    "tts_voice",
    "autoplay_result",
    "tools",
    "rp_endpoints",
    "rp_statuses",
    "resourceplus_ms",
    "openai_ms",
    "agent_ms",
    "stt_finalize_ms",
    "speech_normalization_ms",
    "tts_ms",
    "response_send_ms",
    "audit_persist_ms",
    "total_after_release_ms",
    "total_ms",
    "model_requests",
    "confirmation_required",
    "confirmed",
    "action_type",
    "action_state",
    "action_result",
    "result_status",
    "error_category",
    "error_owner",
    "error_stage",
)


def _date_boundary(value: str | None, *, upper: bool) -> str | None:
    if value is None:
        return None
    try:
        if len(value) == 10:
            day = date.fromisoformat(value)
            if upper:
                day += timedelta(days=1)
            return datetime.combine(day, time.min, tzinfo=timezone.utc).isoformat()
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Invalid ISO date or timestamp: {value}"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc).isoformat()


def _latency(record: dict[str, Any], name: str) -> float | None:
    value = record.get("latencies_ms", {}).get(name)
    return value if isinstance(value, (int, float)) else None


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    calls = record.get("resourceplus_calls") or []
    openai_stages = (
        "openai_main",
        "follow_up_classifier",
        "confirmation_classifier",
        "response_renderer",
    )
    openai_values = [_latency(record, name) for name in openai_stages]
    openai_ms = round(sum(value for value in openai_values if value is not None), 3)
    resourceplus_ms = _latency(record, "resourceplus")
    if resourceplus_ms is None:
        resourceplus_ms = round(
            sum(
                float(call.get("duration_ms", 0))
                for call in calls
                if isinstance(call, dict)
                and isinstance(call.get("duration_ms"), (int, float))
            ),
            3,
        )
    return {
        "timestamp": record.get("timestamp"),
        "trace_id": record.get("trace_id"),
        "interaction_id": record.get("interaction_id"),
        "app_version": record.get("app_version"),
        "git_commit": record.get("git_commit"),
        "environment": record.get("environment"),
        "deployment_id": record.get("deployment_id"),
        "session_ref": record.get("session_reference"),
        "mode": record.get("input_mode"),
        "input_source": record.get("input_source"),
        "user_text": record.get("user_text"),
        "detected_locale": record.get("raw_detected_locale"),
        "resolved_language": record.get("resolved_language"),
        "response_language": record.get("response_language"),
        "display_message": record.get("display_message"),
        "speech_message": record.get("speech_message"),
        "tts_text": record.get("tts_text"),
        "tts_requested": record.get("tts_requested"),
        "tts_generated": record.get("tts_generated"),
        "tts_locale": record.get("tts_locale"),
        "tts_voice": record.get("tts_voice"),
        "autoplay_result": record.get("autoplay_result"),
        "tools": " | ".join(str(item) for item in record.get("tools") or []),
        "rp_endpoints": " | ".join(
            str(call.get("endpoint")) for call in calls if isinstance(call, dict)
        ),
        "rp_statuses": " | ".join(
            str(call.get("status")) for call in calls if isinstance(call, dict)
        ),
        "resourceplus_ms": resourceplus_ms,
        "openai_ms": openai_ms,
        "agent_ms": _latency(record, "agent"),
        "stt_finalize_ms": _latency(record, "post_release_stt_finalize"),
        "speech_normalization_ms": _latency(record, "speech_normalization"),
        "tts_ms": _latency(record, "tts"),
        "response_send_ms": _latency(record, "response_send"),
        "audit_persist_ms": _latency(record, "audit_persist"),
        "total_after_release_ms": _latency(record, "total_after_release"),
        "total_ms": _latency(record, "total"),
        "model_requests": record.get("model_requests"),
        "confirmation_required": record.get("confirmation_required"),
        "confirmed": record.get("confirmed"),
        "action_type": record.get("action_type"),
        "action_state": record.get("action_state"),
        "action_result": record.get("action_result"),
        "result_status": record.get("result_status"),
        "error_category": record.get("error_category"),
        "error_owner": record.get("error_owner"),
        "error_stage": record.get("error_stage"),
    }


def render_csv(records: Sequence[dict[str, Any]]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDS)
    writer.writeheader()
    writer.writerows(flatten_record(record) for record in records)
    return output.getvalue()


def render_json(records: Sequence[dict[str, Any]]) -> str:
    cleaned = []
    for record in records:
        item = dict(record)
        item.pop("latencies", None)
        cleaned.append(item)
    return json.dumps(cleaned, ensure_ascii=False, indent=2) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from", dest="from_value", help="UTC ISO date/timestamp")
    parser.add_argument("--to", dest="to_value", help="UTC ISO date/timestamp (date is inclusive)")
    parser.add_argument("--language", choices=("en", "ar"))
    parser.add_argument("--mode", choices=("text", "voice"))
    parser.add_argument("--status")
    parser.add_argument("--format", choices=("csv", "json"), default="json")
    parser.add_argument("--output", type=Path)
    return parser


async def _load_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    settings = get_settings()
    database = Path(settings.ai_audit_db_path)
    if not database.is_file():
        raise FileNotFoundError(f"Audit database does not exist: {database}")
    store = AuditStore(str(database), retention_days=settings.ai_audit_retention_days)
    return await store.query(
        from_timestamp=_date_boundary(args.from_value, upper=False),
        to_timestamp=_date_boundary(args.to_value, upper=True),
        language=args.language,
        mode=args.mode,
        status=args.status,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        records = asyncio.run(_load_records(args))
    except (FileNotFoundError, argparse.ArgumentTypeError) as exc:
        parser.error(str(exc))
    rendered = render_csv(records) if args.format == "csv" else render_json(records)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="")
        print(f"Exported {len(records)} interaction(s) to {args.output}")
    else:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(encoding="utf-8")
        sys.stdout.write(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
