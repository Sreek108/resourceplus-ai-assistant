# Production observability

## Architecture

The assistant has one correlated observability path built around the existing
`InteractionAudit` and SQLite `AuditStore`:

```text
browser event / HTTP / WebSocket
             |
             +-- safe trace_id (request-local context)
             +-- structured JSON events
             +-- bounded in-process Prometheus metrics
             +-- existing interaction audit
             +-- OpenAI / Azure Speech / ResourcePlus spans
```

Audit, metric, log, and frontend-event storage failures are fail-open. They never
authorize an action, alter a ResourcePlus payload, or fail a chat request. No external
exporter runs in the response path. The span shape is intentionally compatible with a
future OpenTelemetry adapter without requiring a vendor or SDK today.

## Trace lifecycle

An HTTP client may send `X-Trace-ID`. The accepted format is 8-64 safe ASCII letters,
digits, dots, underscores, colons, and hyphens. Surrounding whitespace is removed;
invalid values are rejected with HTTP 400 and are never reflected. When absent, the
backend generates a UUID. Every HTTP response includes `X-Trace-ID`.

The request-local trace is copied into new interaction-audit rows, component/provider
structured events, ResourcePlus correlation headers, and frontend telemetry records.
Browser WebSockets cannot set arbitrary handshake headers consistently, so streaming
voice also accepts the same validated `trace_id` in its existing `start` object. If a
handshake header and start value are both present, they must match.

Never derive a trace from an employee email, session token, confirmation ID, or other
business identifier.

## Frontend integration

For one interaction, the frontend should generate one UUID, send it as `X-Trace-ID` on
HTTP requests, include it as `trace_id` in the streaming-voice start message, and use
that same value for related browser telemetry. It should retain the returned response
header only for approved diagnostics.

`POST /api/telemetry/frontend` accepts only `trace_id`, an allowlisted `event`, optional
`duration_ms` (0-600000), optional allowlisted `error_category`, and optional
`http_status` (100-599). Events are limited to:

- `request_started`, `response_received`, `http_error`, and `network_error`;
- `websocket_opened` and `websocket_closed`;
- `microphone_started` and `microphone_released`;
- `audio_play_started`, `audio_play_completed`, and `audio_play_failed`;
- `render_error`.

Extra fields and arbitrary logs/stack content are rejected. Payloads are capped at 2
KiB, event acceptance is rate-limited with bounded memory, and storage runs as a safe
background task. This endpoint is independent of chat execution.

## ResourcePlus correlation

Every ResourcePlus request inside a trace carries `X-Correlation-ID: <trace_id>`. This
is informational only; request parameters, bodies, validation, identities, and business
semantics are unchanged. Telemetry stores only method, relative endpoint path, status,
and duration. It never stores full URLs, query values, or bodies.

An HTTP 200 that contains a business outcome such as an already-existing booking is
provider success, even when the business operation itself cannot proceed. It is not
misclassified as an infrastructure error.

## Structured logs and spans

The centralized JSON helper supports `timestamp`, `level`, `service`, `environment`,
`deployment_id`, `app_version`, `trace_id`, `interaction_id`, hashed
`session_reference`, `component`, `event`, `duration_ms`, `status`, `error_category`,
`error_owner`, `error_stage`, and `retryable`. A safe ResourcePlus endpoint path may
also be present.

Fields are allowlisted scalars. Secret-like or email-shaped values are redacted.
Credentials, authorization, request/query bodies, PendingAction arguments,
ResourcePlus IDs, audio, provider bodies, and stack traces are excluded. Existing
readable `VOICE_LATENCY` and `RP_API` developer lines remain compatible.

Structured milliseconds cover HTTP, WebSocket/audio streaming, uploaded audio, Azure
STT, post-release finalization, language resolution, agent execution, every OpenAI
request, confirmation classification, response rendering, each ResourcePlus call,
speech normalization, Azure TTS, response sending, and audit persistence. Persisted
audit timing remains milliseconds; the legacy `VOICE_LATENCY` line remains seconds.

## Prometheus metrics

`GET /metrics` exposes Prometheus text for request success/failure and text/voice mode,
HTTP latency, voice latency after release, OpenAI latency/errors, ResourcePlus
latency/errors/timeouts, Azure STT/TTS latency/errors, WebSocket failures,
PendingAction state events, and safe frontend events.

Histograms have fixed millisecond buckets from 5 ms through 60 seconds. A monitoring
backend can calculate p50/p90/p95/p99 with `histogram_quantile`. Labels are fixed and
low-cardinality: trace/session IDs, employees, utterances, endpoint arguments, and raw
errors can never become labels. The in-process registry is per worker, so production
scraping must aggregate all worker targets.

## Health, readiness, and operations

- `GET /health` returns only `{"status":"ok"}`.
- `GET /ready` checks required local OpenAI/ResourcePlus configuration and reports
  whether voice configuration is present. It makes no live provider call and exposes no
  configured value.
- `GET /api/ops/diagnostics` returns 404 by default. It requires
  `OPS_DIAGNOSTICS_ENABLED=true` and an exact `X-Ops-Key` matching
  `OPS_DIAGNOSTICS_TOKEN`. It returns safe booleans and release metadata only.

Do not expose operations diagnostics on a public ingress without separate network and
identity controls.

## Error taxonomy

Owners are restricted to `frontend`, `network`, `ai_backend`, `openai`, `azure_stt`,
`azure_tts`, `resourceplus_api`, `configuration`, `validation`, `transaction`, and
`unknown`. Stages are independently allowlisted across HTTP, WebSocket, audio/STT,
language resolution, model/agent work, ResourcePlus, speech/TTS, response send, audit,
configuration, validation, and transaction.

Provider failures map to their provider, malformed input maps to validation, and
expired/mismatched confirmation state maps to transaction. Unknown data cannot create
a new category or metric label.

## Diagnostic CLI

Use an approved response trace:

```powershell
uv run python scripts/diagnose_trace.py --trace-id 17ddf407-35d5-43df-9cb9-b8b458df4a7c
uv run python scripts/diagnose_trace.py --trace-id 17ddf407-35d5-43df-9cb9-b8b458df4a7c --json
```

The safe timeline combines browser events and the existing audit. It includes
mode/language, stage durations, model count, ResourcePlus path/status/duration, action
state, result, and error owner/stage. It excludes conversation content, sessions,
confirmation/internal IDs, provider payloads, identity, and audio.

## Privacy, PDPL, and retention

No raw/generated audio is stored. Logs, metrics, browser events, and the diagnostic CLI
contain no employee email, credentials, authentication data, PendingAction arguments,
internal IDs, request bodies, query values, provider bodies, or stack traces.

Conversation content remains governed solely by `AI_AUDIT_STORE_CONTENT`; when false,
user/display/speech/TTS text is `NULL`. Access, purpose limitation, retention, deletion,
and incident handling must follow the organization's approved PDPL policy.
`AI_AUDIT_RETENTION_DAYS` applies to interactions and frontend telemetry in the same
SQLite database. Migrations are additive and preserve legacy rows.

SQLite is suitable for local and controlled single-instance UAT. It is not sufficient
as the sole centralized store for multi-instance production. `APP_VERSION`,
`GIT_COMMIT`, `DEPLOYMENT_ID`, and `APP_ENVIRONMENT` identify a release.
`OBSERVABILITY_EXPORTER` and `OBSERVABILITY_ENDPOINT` reserve a vendor-neutral
deployment contract; this implementation enables no external network exporter.
Production should collect stdout JSON and `/metrics`, or add a bounded asynchronous
exporter with backpressure and fail-open behavior.

## Incident workflow

1. Obtain `X-Trace-ID` and the approximate timestamp.
2. Run `scripts/diagnose_trace.py` against the approved local/UAT database.
3. Check browser network, WebSocket, render, and playback events.
4. Locate the slow/error stage and normalized owner.
5. For ResourcePlus, compare safe path/status with its correlated server log.
6. Separate HTTP/provider failures from valid HTTP 200 business outcomes.
7. Confirm deployment metadata before comparing releases.
8. Escalate only the safe trace/timeline—never credentials, audio, identity, full URLs,
   or provider bodies.

For example, successful STT/OpenAI followed by a ResourcePlus timeout and no TTS points
to the HRMS provider/network boundary. ResourcePlus HTTP 200 followed by a failed action
state instead indicates a business or transaction result, not provider downtime.
