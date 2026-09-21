"""Print a safe timeline for one ResourcePlus assistant trace."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.audit import AuditStore  # noqa: E402
from app.config import get_settings  # noqa: E402
from app.telemetry import validate_trace_id  # noqa: E402


SAFE_STAGE_ORDER = (
    "audio_receive",
    "audio_stream_duration",
    "stt",
    "post_release_stt_finalize",
    "language_resolution",
    "agent",
    "openai_main",
    "follow_up_classifier",
    "confirmation_classifier",
    "response_renderer",
    "resourceplus",
    "speech_normalization",
    "tts",
    "response_send",
    "audit_persist",
    "total_after_release",
    "total",
)


def build_timeline(
    trace_id: str,
    records: list[dict[str, Any]],
    frontend_events: list[dict[str, Any]],
) -> dict[str, Any]:
    timeline: list[dict[str, Any]] = []
    for item in frontend_events:
        timeline.append(
            {
                "timestamp": item.get("timestamp"),
                "source": "frontend",
                "event": item.get("event"),
                "duration_ms": item.get("duration_ms"),
                "http_status": item.get("http_status"),
                "error_category": item.get("error_category"),
            }
        )
    for record in records:
        timestamp = record.get("timestamp")
        timeline.append(
            {
                "timestamp": timestamp,
                "source": "assistant",
                "event": "interaction_started",
                "mode": record.get("input_mode"),
                "input_source": record.get("input_source"),
                "raw_detected_locale": record.get("raw_detected_locale"),
                "resolved_language": record.get("resolved_language"),
                "response_language": record.get("response_language"),
                "model_requests": record.get("model_requests"),
            }
        )
        timings = record.get("latencies_ms") or {}
        for stage in SAFE_STAGE_ORDER:
            duration = timings.get(stage)
            if isinstance(duration, (int, float)):
                timeline.append(
                    {
                        "timestamp": timestamp,
                        "source": "assistant",
                        "event": "stage",
                        "stage": stage,
                        "duration_ms": duration,
                    }
                )
        for call in record.get("resourceplus_calls") or []:
            timeline.append(
                {
                    "timestamp": timestamp,
                    "source": "resourceplus",
                    "event": "request",
                    "method": call.get("method"),
                    "endpoint": call.get("endpoint"),
                    "status": call.get("status"),
                    "duration_ms": call.get("duration_ms"),
                }
            )
        timeline.append(
            {
                "timestamp": timestamp,
                "source": "assistant",
                "event": "interaction_completed",
                "result_status": record.get("result_status"),
                "action_type": record.get("action_type"),
                "action_state": record.get("action_state"),
                "action_result": record.get("action_result"),
                "confirmation_required": record.get("confirmation_required"),
                "error_category": record.get("error_category"),
                "error_owner": record.get("error_owner"),
                "error_stage": record.get("error_stage"),
            }
        )
    timeline.sort(key=lambda item: str(item.get("timestamp") or ""))
    return {"trace_id": trace_id, "timeline": timeline}


def render_text(diagnostic: dict[str, Any]) -> str:
    lines = [f"Trace {diagnostic['trace_id']}"]
    for item in diagnostic["timeline"]:
        fields = [
            str(item.get("timestamp") or "unknown-time"),
            str(item.get("source") or "unknown"),
            str(item.get("event") or "unknown"),
        ]
        for key in (
            "mode",
            "resolved_language",
            "stage",
            "duration_ms",
            "method",
            "endpoint",
            "status",
            "result_status",
            "action_type",
            "action_state",
            "error_category",
            "error_owner",
            "error_stage",
        ):
            if item.get(key) is not None:
                fields.append(f"{key}={item[key]}")
        lines.append(" | ".join(fields))
    return "\n".join(lines) + "\n"


async def _load(trace_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    settings = get_settings()
    database = Path(settings.ai_audit_db_path)
    if not database.is_file():
        raise FileNotFoundError(f"Audit database does not exist: {database}")
    store = AuditStore(str(database), retention_days=settings.ai_audit_retention_days)
    return await store.by_trace_id(trace_id), await store.frontend_events(trace_id)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace-id", required=True)
    parser.add_argument("--json", action="store_true", dest="as_json")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        trace_id = validate_trace_id(args.trace_id)
        records, frontend_events = asyncio.run(_load(trace_id))
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    diagnostic = build_timeline(trace_id, records, frontend_events)
    rendered = (
        json.dumps(diagnostic, ensure_ascii=False, indent=2) + "\n"
        if args.as_json
        else render_text(diagnostic)
    )
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if diagnostic["timeline"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
