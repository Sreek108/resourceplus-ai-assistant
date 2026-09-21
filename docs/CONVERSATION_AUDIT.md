# Conversation interaction audit

## Purpose

The assistant has one opt-in SQLite audit path for local/UAT diagnostics. Production
correlation, structured logs, metrics, and safe browser events extend this same path;
see [Production observability](PRODUCTION_OBSERVABILITY.md). It records
one safe interaction trace for each text or voice turn so engineers can compare what
the employee asked, what the UI displayed, what voice output was prepared, which
backend systems ran, and where time was spent. It is not an employee activity ledger
or a replacement for ResourcePlus transaction records.

The existing controls remain authoritative:

```dotenv
AI_AUDIT_ENABLED=true
AI_AUDIT_STORE_CONTENT=true
AI_AUDIT_RETENTION_DAYS=7
AI_AUDIT_DB_PATH=data/assistant_audit.db
```

Auditing is disabled by default. `AI_AUDIT_STORE_CONTENT=false` retains safe metadata
and timings but stores `NULL` for user, display, speech, and TTS text.

## Captured interaction fields

Every new row also captures `trace_id`, `app_version`, `git_commit`, `environment`, and
`deployment_id`. Failures can record allowlisted `error_owner` and `error_stage`
alongside `error_category`. Legacy rows retain `NULL` for fields that did not exist
when they were written.

Every new trace has a UUID `interaction_id`, UTC timestamp, SHA-256-derived
16-character `session_reference`, `input_mode`, `input_source`, overall success and
`result_status`. Text turns use `mode=text`, `source=typed`, and the exact submitted
message. Voice turns use `mode=voice`, `source=stt`, and only the final STT transcript.

Language fields are intentionally separate:

- `raw_detected_locale`: Azure's raw STT locale, for example `ar-SA`.
- `resolved_language`: transcript/script-resolved input language (`en` or `ar`).
- `response_language`: language actually selected for the assistant response/TTS.

This separation makes a transcript/locale disagreement visible without allowing the
raw Azure locale to override language resolution.

## Text and voice flows

Text flow:

```text
typed message -> language/agent/tools -> display response -> audit persistence
```

Voice flow:

```text
PCM/WAV (memory only) -> Azure STT -> resolved language -> agent/tools
-> display/speech text -> speech normalization -> Azure TTS -> audit persistence
```

Audio bytes are never placed in the audit object or database. Streaming disconnects,
client cancellation, and no-speech outcomes can still create metadata-only diagnostic
records.

## Display, speech, and TTS

- `display_message` is the exact chat text returned for rendering.
- `speech_message` is the voice-oriented response returned by the primary agent. It
  can be `NULL` when no separate speech representation was produced.
- `tts_text` is the exact normalized string supplied to Azure Speech synthesis.

`tts_requested` and `tts_generated` are nullable. They remain `NULL` for text-only
turns, become `true/false` when synthesis was attempted, and `true/true` when audio was
generated. Successful synthesis also records safe `tts_locale` and `tts_voice`
metadata. Arabic synthesis is restricted to locale `ar-SA` with either
`ar-SA-ZariyahNeural` or `ar-SA-HamedNeural`. Browser autoplay success is not currently
reported safely to the backend, so `autoplay_result` remains `NULL`.

## Tools and ResourcePlus API trace

`tools_json` contains logical assistant tool names. `resourceplus_json` contains an
ordered array with only:

```json
{
  "method": "GET",
  "endpoint": "api/AI/AttendanceSummary",
  "status": 200,
  "duration_ms": 335.0
}
```

The endpoint is a relative path with any query string removed. The trace never records
query parameters, employee email parameters, request bodies, full URLs, authorization
headers, tokens, or credentials. Status can be an HTTP integer or a safe transport
label such as `timeout` or `connection_error`.

## Latency and model requests

`latencies_json` stores milliseconds for current records. Available keys
include:

- `audio_stream_duration`
- `post_release_stt_finalize`
- `language_resolution`
- `agent`
- `openai_main`
- `confirmation_classifier`
- `resourceplus`
- `speech_normalization`
- `tts`
- `response_send`
- `audit_persist`
- `total_after_release`
- `total`

Only stages measurable in a given transport are present. For example, a text turn has
no STT/TTS fields, and the HTTP voice endpoint cannot observe browser autoplay.
`model_requests` counts all measured model requests in the interaction. Terminal
`VOICE_LATENCY` log values remain seconds-based; this preserves the established
production log semantics while persisted/exported data uses milliseconds.

Legacy schema-version 1 rows are marked `latency_unit=seconds`; reads and exports
normalize those values to milliseconds without rewriting or deleting the row.
Schema-version 2 introduced millisecond interaction traces, schema-version 3 added
nullable TTS locale/voice metadata, and schema-version 4 adds correlation/deployment
metadata plus error owner/stage. All migrations are additive.

Allowlisted browser events use the normalized `frontend_telemetry` table in this same
database. It stores only trace, event, duration, safe error category, and HTTP status—
never arbitrary logs, stack traces, identity, or conversation content.

## Confirmation and action trace

The audit stores only safe transaction metadata:

- logical `action_type`
- `action_state`: `none`, `prepared`, `pending_confirmation`, `rejected`, `expired`,
  `executed`, or `failed`
- `confirmation_required` and nullable `confirmed`
- allowlisted `action_result`, such as `submitted_for_approval`, `cancelled`,
  `approved`, `updated`, `succeeded`, or `failed`

It never stores confirmation IDs, reason/mapping/request IDs, action arguments, or a
copy of `PendingAction`. PendingAction creation, TTL, immutable validated arguments,
confirmation matching, and execution remain authoritative and unchanged.

## Safe errors

Only allowlisted categories are stored, such as `no_speech`, `azure_canceled`,
`resourceplus_error`, `openai_error`, `tts_error`, `validation_error`,
`websocket_disconnected`, `confirmation_expired`, `access_denied`, and
`unknown_safe_category`. Provider bodies, stack traces, exception messages, and
secret-bearing values are excluded.

## Retention and migration

Retention cleanup remains lazy: the configured database removes records older than
`AI_AUDIT_RETENTION_DAYS` when the store initializes and during explicit cleanup. The
schema migration uses additive columns detected with `PRAGMA table_info`; existing UAT
rows are preserved. Keep the database local, access-controlled, and excluded from git.

## Export

Export all retained rows as structured JSON:

```powershell
uv run python scripts/export_audit.py --format json --output audit_exports/uat.json
```

Export one UTC date to flattened CSV:

```powershell
uv run python scripts/export_audit.py --from 2026-09-21 --to 2026-09-21 --format csv --output audit_exports/uat_2026-09-21.csv
```

Optional filters are `--from`, `--to`, `--language en|ar`, `--mode text|voice`, and
`--status`. A date-only `--to` is inclusive through that UTC calendar day. Without
`--output`, data is written to stdout. JSON retains structured ResourcePlus call arrays
and `latencies_ms`; CSV flattens tool/API lists and key timing stages.

## Troubleshooting examples

For "the screen showed one answer but the voice said another," filter by time/session
and compare `display_message`, `speech_message`, `tts_text`, `response_language`, and
`tts_generated`.

For a slow attendance turn, compare `post_release_stt_finalize`, `openai_main`,
`resourceplus`, and `tts`; then inspect the ordered ResourcePlus paths/statuses without
exposing employee query values.

For a confirmation report, inspect `action_type`, `action_state`, `confirmed`, and the
safe result. The audit deliberately cannot reconstruct or execute the pending action.
