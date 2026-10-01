# Chat API v2

Version 2 is additive. Existing clients can continue using `POST /api/chat` and
reading `message`, `display_message`, `session_id`, confirmation fields,
`reason_options`, and `language`. Responses now also include
`response_schema_version: 2` and an optional `blocks` array; clients may ignore both.

## Identity and language

`email` and `instance` form one request identity and must be supplied together. UAT
requires both explicitly. A session is owned by that exact pair; the model never
selects an employee identity. English, Arabic, and natural code-switching use the same
endpoints and safety pipeline.

## JSON and streaming endpoints

`POST /api/chat` retains the existing JSON request and response contract.

`POST /api/chat/stream` accepts the same JSON body and returns
`text/event-stream`. Events are ordered as follows:

- `accepted`: request validation completed.
- `status`: a real state such as `processing` or, for a deterministic read,
  `fetching_hr_data`.
- `text_delta`: display text. Clients should concatenate chunks for forward
  compatibility.
- `blocks`: the complete trusted structured-block collection, when present.
- `confirmation`: emitted only when an immutable action requires confirmation.
- `complete`: the full backward-compatible response.
- `error`: a safe terminal error with no identity, secret, or upstream details.

## Structured blocks

Blocks are built by the backend from ResourcePlus results or immutable confirmation
state. The model does not create trusted table rows.

- `table`: `title`, typed `columns`, and complete `rows`.
- `key_value`: profile/detail `items` with `label` and `value`.
- `stat_cards`: summary `items`.
- `list`: trusted notification or result strings.
- `actions`: safe options such as live exceptional-entry reasons.
- `confirmation`: immutable pending-action summary and confirm/cancel controls.
- `notice`: informational, success, warning, or error content.

Large tables remain complete in `blocks` where practical. `display_message` contains a
short answer and caps Markdown rows for older clients.

## Conversation and confirmation safety

Incomplete missing-punch and day-type requests use short-lived, non-executable
backend slots. Casual conversation does not overwrite the draft; a clear new HR
request replaces only that draft. A validated `PendingAction` is separate and
immutable. It executes only after explicit confirmation by the same identity and
session, within its TTL, and cannot be replayed after success or failure.

Exceptional-entry reason options come from ResourcePlus. The backend re-fetches and
revalidates the punch suggestion and reason before creating the pending action.
Reference-data caching never replaces this write-time check.

### ResourcePlus v2 less-hours and cancellation flows

Normal less-hours corrections are backend-owned slot flows with `date`, `reason`,
optional explicit `entry_type`, and optional explicit partial `minutes`. The backend
reads AttendanceSummary before asking for a reason. A positive `LessHrs` value is a
correction candidate only when at least one punch exists. Candidate does not mean the
write is guaranteed to be accepted. Absent/no-punch days offer Leave or
Business Travel; Week End, Holiday, Leave, Business Travel, and zero-less-hours days
do not create an exceptional entry.

Allowance data comes from the date-scoped `ExceptionalEntries/Balance` endpoint and
is informational. Range reads skip Balance for zero candidates, use the sole
candidate date for one candidate, and omit a generic allowance for multiple
candidates until a date is selected. A
missing/unavailable policy does not invent one and need not block the authoritative
booking endpoint. `limitType=1` is displayed as exception count and `limitType=2` as
minutes; other values receive no inferred unit. Confirmation re-fetches attendance
candidate status and live reasons,
then stores an immutable FromSummary action. Normal actions omit `entryType` and
`minutes`; they are included only for an employee's explicit one-side or partial-time
request.

ResourcePlus—not the AI—decides auto approval. The documented `success` boolean is
authoritative: false is failure; true allows `isAutoApproved=true` to mean
auto-approved and false to mean manager approval. The completed response surfaces the
API `message`, any `warning`, allowance values, and split created entries. A missing
`success` with a boolean `isAutoApproved` is supported only as an early-v2
compatibility response; no other field implies success. Natural
exception cancellation resolves a real pending/cancellable row from
`GET ExceptionalEntries`, never exposes its ID, and requires the same PendingAction
confirmation and replay protections before `POST ExceptionalEntries/Cancel`.

ResourcePlus is authoritative for business rejection. A returned instruction to
apply leave is shown as a next step, not converted into an automatic leave request or
an AI-side universal four-hour rule.

For cancellation, an explicit `isCancellable`/`canCancel` boolean has priority.
Without one, documented status `Not Approved` is cancellable; `Approved` and
`Rejected` are not. Older pending/submitted labels remain compatibility fallbacks.

Relevant structured output includes the less-hours attendance table, allowance
key/value card, live-reason actions, detailed confirmation, submission notices and
created-entry table, and cancellation candidate table. Explicit missing-punch and
exact-time requests retain the legacy MissingPunchSuggestions/Request workflow.

## Grounding and dates

Live employee facts come only from ResourcePlus results. General conversational text
may be phrased naturally. Policy, salary, approval decisions, buffer rules, and other
unintegrated facts are reported as unavailable rather than inferred.

`today`, `yesterday`, and `this month` are backend-resolved. A missing-punch listing
without a period defaults to the current month through today; it never silently
selects a historical year.
