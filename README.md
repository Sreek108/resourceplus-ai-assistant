# ResourcePlus AI Assistant

FastAPI backend and lightweight React client-demo UI for the ResourcePlus HRMS
assistant. The POC supports English and Arabic text, English and Saudi Arabic
voice, employee self-service reads, notifications, and confirmation-gated HR
actions.

The browser talks only to the assistant backend. It never calls ResourcePlus
directly and it never receives ResourcePlus credentials.

## Architecture

```text
React/Vite -> POST /api/chat -----------\
              POST /api/voice/chat -----+-> FastAPI assistant -> ResourcePlus
                    PCM/WAV upload       |        |
                    spoken WAV response  |        +-> pending action
                                         |            + explicit confirmation
                                         +-> Azure Speech (voice turns only)
```

The model never receives a tool that directly performs a write. It can prepare an
intent using descriptive fields; Python resolves authoritative ResourcePlus IDs,
stores the validated operation, and executes it only after confirmation in the same
short-lived session.

## Backend setup

Python 3.11 or newer and `uv` are required.

```powershell
py -3.11 -m venv .venv
uv pip install --python .\.venv\Scripts\python.exe -r requirements.txt
Copy-Item .env.example .env
```

Configure `OPENAI_API_KEY`, `OPENAI_MODEL`, and the Azure Speech settings for voice.
For the demo/UAT integration, the frontend sends the current user's `email` and
ResourcePlus `instance` as one identity pair on every text or voice request. The
backend binds that pair to the request and uses it for employee and supervisor
ResourcePlus calls. `RP_DEFAULT_EMAIL` and `RP_INSTANCE` are retained only as a
paired local-development/test fallback when neither request field is present.
UAT and production reject requests that omit the pair and never use those defaults.
Production must replace the frontend-supplied pair with validated ResourcePlus
authentication/session identity.

The default CORS setting permits only the local Vite origins
`http://127.0.0.1:5173` and `http://localhost:5173`. Override
`CORS_ALLOWED_ORIGINS` with a comma-separated allowlist when the frontend is served
from a different trusted origin. Do not use `*` with production credentials.

Start the backend on the demo port:

```powershell
uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
```

- API documentation: <http://127.0.0.1:8001/docs>
- Health check: <http://127.0.0.1:8001/health>
- Readiness check: <http://127.0.0.1:8001/ready>
- Prometheus metrics: <http://127.0.0.1:8001/metrics>

Run backend tests with:

```powershell
uv run python -m pytest -q
```

## Frontend setup

Node.js 20.19+ or 22.12+ is required by Vite 8.

```powershell
cd frontend
npm install
npm run dev
```

Open <http://127.0.0.1:5173>. The frontend defaults to the backend at
`http://127.0.0.1:8001`. To use another backend, create `frontend/.env.local` with:

```dotenv
VITE_API_BASE_URL=http://127.0.0.1:8001
```

Developer-only response metadata is available at
<http://127.0.0.1:5173/?debug=true>. It includes language/session diagnostics but
never secrets or internal ResourcePlus IDs. The normal UI hides all internal
metadata.

Frontend checks:

```powershell
cd frontend
npm test
npm run build
```

Production builds use same-origin `/api/...` URLs. Streaming voice derives `ws://`
or `wss://` from the browser page URL, so an HTTPS reverse proxy automatically uses
WSS. `VITE_API_BASE_URL` is a development-only override; production does not embed a
backend hostname.

## Temporary Lead UAT

The FastAPI application serves only the files generated under `frontend/dist`. API,
health, documentation, and WebSocket routes keep priority over the React SPA fallback.
Project source, `.env`, `data/`, audit databases, and logs are never static roots.

1. In `.env`, enable the approved short-lived audit configuration. Keep the public
   debug endpoint disabled:

   ```dotenv
   AI_AUDIT_ENABLED=true
   AI_AUDIT_STORE_CONTENT=true
   AI_AUDIT_RETENTION_DAYS=7
   AI_AUDIT_DEBUG_ENDPOINT_ENABLED=false
   UAT_ALLOWED_ORIGINS=https://your-ngrok-host.ngrok-free.app
   ```

2. Build the existing React application:

   ```powershell
   cd frontend
   npm run build
   cd ..
   ```

3. Start FastAPI without local TLS:

   ```powershell
   uv run python -m uvicorn app.main:app --host 127.0.0.1 --port 8001
   ```

4. In a second terminal, start the TLS tunnel:

   ```powershell
   ngrok http 8001
   ```

   If ngrok assigns a random hostname, copy its HTTPS forwarding origin into
   `UAT_ALLOWED_ORIGINS` and restart FastAPI before sharing it. A reserved ngrok
   hostname can be configured in advance to avoid that restart. Never use `*`.

5. Send the lead only the HTTPS forwarding URL shown by ngrok, such as
   `https://your-ngrok-host.ngrok-free.app/`. Do not share the local URLs, audit DB,
   `.env`, or the ngrok inspection interface.

6. Test English and natural Saudi Arabic text and voice through that HTTPS page.
   The browser uses the same origin for HTTP and `wss://.../api/voice/stream`.

7. Stop ngrok immediately after the UAT window, then clear `UAT_ALLOWED_ORIGINS` and
   disable content auditing unless another approved session is scheduled.

8. Review `data/assistant_audit.db` locally using an approved SQLite tool. Do not
   enable or expose `/api/debug/audit/recent` on the public tunnel.

## Demo UI behavior

- Every suggestion and sidebar shortcut sends its visible text through the ordinary
  free-form `/api/chat` endpoint; there is no browser-side intent routing.
- Each message independently detects Arabic characters and uses `dir="rtl"` and
  `lang="ar"`, while English messages remain LTR. The composer adapts as the user
  types, so mixed conversations do not flip the whole application.
- The backend-issued `session_id` is reused for every subsequent text or voice turn
  and retained in `sessionStorage`. **New conversation** clears only local browser
  state; it does not add a server-side deletion API.
- Confirmation buttons submit `Yes` or `No` through `/api/chat` with the current
  `session_id` and backend-issued `confirmation_id`. Spoken confirmations continue
  through `/api/voice/chat`. The UI never reconstructs transaction arguments.
- Voice recording is push-to-talk: hold the microphone while speaking and release to
  send. Pointer events cover mouse, touch, and stylus input. Very short accidental
  recordings are discarded locally. During the hold, the browser downsamples mono
  microphone data to raw 16 kHz, signed 16-bit PCM and streams binary chunks to
  `/api/voice/stream`. A complete WAV is retained in memory and uploaded to the
  compatible `/api/voice/chat` endpoint if streaming cannot start or finalize. A
  pending confirmation is never automatically retried through the fallback path.
- Voice answers use a concise spoken rendering of the authoritative on-screen answer.
  Returned base64 audio is decoded using `audio_mime_type` to a Blob URL, played once
  when browser policy permits, and can be replayed from the assistant message.

Microphone access requires a secure context. Loopback origins such as `127.0.0.1`
and `localhost` are treated as secure by current desktop browsers. Autoplay policy
may occasionally block the spoken response; the replay button remains available.
The current recorder uses `ScriptProcessorNode` for broad POC compatibility; an
AudioWorklet is the recommended production replacement.

## Chat API

Initial message:

```json
{
  "message": "I need annual leave from 2026-09-22 to 2026-09-24",
  "session_id": "session-123",
  "email": "employee@company.com",
  "instance": "Universal",
  "confirmation_id": null
}
```

The response generates a session and may request confirmation. Confirm using the
same chat endpoint:

```json
{
  "message": "Yes",
  "session_id": "backend-issued-session-id",
  "email": "employee@company.com",
  "instance": "Universal",
  "confirmation_id": "backend-issued-confirmation-id"
}
```

The confirmation ID is optional only when a session has exactly one current pending
action, but clients should always return it. A `Yes` without a pending action never
executes anything. `email` and `instance` must be supplied together. Conversation
state and pending confirmations are owned by the combined
`email + instance + session_id`; reusing a session or confirmation under a different
identity is rejected without a ResourcePlus write.

The complete HTTP fallback and WebSocket frontend contract is documented in
[Voice Chat API](docs/VOICE_CHAT_API.md).

Voice fallback requests use `multipart/form-data` at `POST /api/voice/chat` with:

- `audio`: PCM WAV file (required)
- `session_id`: current session when present
- `email`: current demo/UAT user's email
- `instance`: current demo/UAT ResourcePlus instance
- `confirmation_id`: current pending confirmation when present

Streaming voice uses `WS /api/voice/stream`. In summary:

1. Client sends a `start` object with `sample_rate: 16000`, the `email` + `instance`
   identity pair, and optional session, confirmation, trace, and debug fields.
2. Server replies `{"type":"ready"}`.
3. Client sends raw mono PCM as binary frames while the microphone is held.
4. Client sends `{"type":"end"}` on release or `{"type":"cancel"}` on cancellation.
5. Server returns one `final` JSON message using the same public voice response fields.

## Conversation audit controls

SQLite conversation auditing is disabled by default and is independent of terminal
latency logging. For approved local/UAT use only:

```dotenv
AI_AUDIT_ENABLED=true
AI_AUDIT_STORE_CONTENT=true
AI_AUDIT_RETENTION_DAYS=7
AI_AUDIT_DB_PATH=data/assistant_audit.db
AI_AUDIT_DEBUG_ENDPOINT_ENABLED=true
```

When content storage is disabled, transcript, display, speech, and TTS text are stored
as `NULL`; operational metadata remains available. Raw audio, generated audio,
credentials, confirmation IDs, and internal ResourcePlus IDs are never stored. The
hidden `GET /api/debug/audit/recent?limit=20` endpoint returns recent records only when
both auditing and its debug endpoint are explicitly enabled. Production retention and
access must follow ResourcePlus privacy and security policy.

See [Latency and observability](docs/LATENCY_AND_OBSERVABILITY.md) for metrics,
rollback notes, and the optimization history.

See [Chat API v2](docs/CHAT_API_V2.md) for text streaming, structured response
blocks, grounding, date defaults, identity requirements, and confirmation behavior.

## Production observability configuration

Set safe release metadata in each deployment:

```dotenv
APP_VERSION=0.2.0
GIT_COMMIT=your-build-commit
DEPLOYMENT_ID=assistant-uat-1
APP_ENVIRONMENT=uat
OBSERVABILITY_EXPORTER=none
OBSERVABILITY_ENDPOINT=
```

HTTP clients may supply `X-Trace-ID`; the backend validates it, generates one when
absent, returns it in the response header, and forwards it to ResourcePlus as
`X-Correlation-ID`. Streaming voice may supply the same value as `trace_id` in the
WebSocket start message. The existing audit receives trace and deployment metadata.

`GET /api/ops/diagnostics` is off by default. On an approved private operations path,
set both `OPS_DIAGNOSTICS_ENABLED=true` and a strong `OPS_DIAGNOSTICS_TOKEN`, then send
the value only in `X-Ops-Key`. Do not expose this route on a public UAT tunnel.

See [Production observability](docs/PRODUCTION_OBSERVABILITY.md) for the frontend
telemetry contract, structured-log schema, metrics, readiness, diagnostic CLI,
privacy/PDPL rules, retention, and multi-instance deployment guidance.

## ResourcePlus integrations

Read operations include employee profile, home/leave balances, attendance,
exceptional-entry allowance, request status, manager approvals, and notifications.
Confirmed write operations include FromSummary less-hours correction, legacy
exact-time exceptional entries, exceptional-entry cancellation, day-type
booking/cancellation, supervisor approvals, and notification read status. There are
no public write-debug endpoints, and automated tests mock all ResourcePlus writes.

The two exceptional-entry contracts are documented in
[Exceptional-entry flows](docs/EXCEPTIONAL_ENTRY_FLOW.md). Normal less-hours uses
`AttendanceSummary` -> live reasons -> immutable confirmation -> `FromSummary`.
Explicit missing-punch/exact-time cases retain `MissingPunchSuggestions` ->
`ExceptionalEntries/Request`. ResourcePlus, not the AI, owns auto approval.

## Current POC limitations

- Configured employee/manager identities replace production authentication.
- Sessions are in memory, disappear on process restart, and are not shared between
  workers.
- Natural-language tool selection depends on the configured AI model.
- Voice requires configured Azure Speech credentials and live network access.
- Browser microphone behavior varies; current Chrome and Edge are the primary demo
  targets.
- Exceptional-entry cancellation awaits an authoritative ResourcePlus request
  schema.
