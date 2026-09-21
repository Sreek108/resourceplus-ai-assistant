# Latency and observability

For trace propagation, structured JSON logs, Prometheus metrics, readiness, browser
telemetry, privacy rules, and the diagnostic CLI, see
[Production observability](PRODUCTION_OBSERVABILITY.md).

## Baseline

Physical warm-path measurements before streaming STT were approximately:

- Attendance: 6.579 seconds
- Notifications: 6.901 seconds
- Follow-up: 7.196 seconds
- Simple conversation: 4.760 seconds

Terminal `VOICE_LATENCY` records are operational metadata only. They do not enable
conversation-content storage.

## Optimization history

### Phase 1 — remove voice-summary LLM call

The primary agent now returns `display_message` and `speech_message` together. Azure
TTS consumes the normalized speech representation, with the display representation as
a no-model fallback. This removed one sequential model request while retaining exact
display/voice traceability.

Rollback: restore a separate renderer only if the configured primary model cannot
reliably produce the structured pair. Doing so restores its measured latency cost.

### Phase 2 — remove factual follow-up classifier

Conversation history goes directly to the main tool-capable agent. The freshness rule
requires a new ResourcePlus read whenever an answer depends on current or date-specific
HR facts; history identifies context but never replaces authoritative data.

Rollback: a classifier can be restored without changing PendingAction, but adds a
sequential model request to follow-ups.

### Phase 3 — stream STT during push-to-talk

The browser sends 16 kHz, signed 16-bit mono PCM binary frames over
`WS /api/voice/stream`. Azure continuous recognition starts before microphone release.
Only final recognition enters the existing language resolver and chat service. The
complete WAV remains memory-only for the existing POST fallback.

New comparison metrics:

- `stream_session_total`: WebSocket session including speaking time.
- `audio_stream_duration`: first audio chunk through release; not backend latency.
- `post_release_stt_finalize`: release through Azure final transcript.
- `total_after_release`: release through final WebSocket response.
- `response_send`: final response transmission initiation.

Rollback: the frontend can use `POST /api/voice/chat` exclusively; that endpoint and
its WAV recognition path remain unchanged.

### Phase 4 — gate confirmation classification on PendingAction

Confirmation classification now runs only when the current session contains a valid,
unexpired backend-owned `PendingAction`. Previously, the no-pending fast return was
limited to English, so ordinary Arabic reads incurred an unnecessary classifier model
request. Read-only English and Arabic turns now report
`confirmation_classifier=0.000s`. Existing deterministic English yes/no handling may
still return the safe no-pending response, but cannot execute anything without the
stored action. Natural English, Arabic, and code-switched replies continue through the
classifier when a pending action exists, and only the stored validated arguments may
be consumed.

Rollback: revert the service-level gate only if confirmation state is redesigned.
Never restore classification for sessions without a pending action.

## Safe conversation audit

When explicitly enabled, one SQLite row records safe per-interaction diagnostics.
Content storage is controlled independently by `AI_AUDIT_STORE_CONTENT`. The audit
preserves the exact display output, agent speech output, and normalized TTS input at
execution time. It stores safe ResourcePlus endpoint names/statuses and hashes session
references. It does not store raw/generated audio, API credentials, authorization
headers, confirmation IDs, mapping/reason/request IDs, or stack traces.

Retention cleanup runs lazily when the audit store initializes. Production retention,
filesystem permissions, encryption, and access control must align with ResourcePlus
privacy/security policy.

The complete field contract, millisecond persistence rules, additive schema migration,
transaction states, privacy exclusions, and CSV/JSON export commands are documented in
[Conversation interaction audit](CONVERSATION_AUDIT.md).

## Preserved safeguards

- ResourcePlus identity and tool validation are unchanged.
- Current HR data still requires authoritative reads.
- PendingAction remains immutable and confirmation-controlled.
- The confirmation classifier remains in place.
- Streaming and audit validation never require ResourcePlus writes.

## Known limitations

- Streaming depends on browser WebSocket support and Azure Speech connectivity.
- The POC recorder uses `ScriptProcessorNode`; AudioWorklet is recommended later.
- Partial Azure transcripts are not shown or sent to the model.
- SQLite is appropriate for single-instance local/UAT use, not multi-worker production.
- Physical post-release measurements must be collected in the target browser and
  network; automated audio tests validate protocol and correctness, not microphone UX.
