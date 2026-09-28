# Exceptional-entry flow

The assistant follows the ResourcePlus exceptional-entry contract in this order:

1. `GET api/AI/MissingPunchSuggestions` with `usrEmail`, `fromDate`, `toDate`,
   `instanceName`, and `lang`.
2. Run every row through the shared missing-punch normalizer used by both the
   informational read tool and transaction preparation. A valid date plus exact
   `IN` or `OUT` direction is a factual missing punch and remains visible in read
   answers. It becomes correctable only when it also has a valid, non-empty,
   authoritative `suggestedEntryTime` for that date.
3. Select exactly one normalized suggestion. A selected date is sent as both
   `fromDate` and `toDate`. A range with suggestions on multiple dates requires
   employee clarification. Both `IN` and `OUT` on one date require direction
   clarification before reasons are fetched.
4. `GET api/AI/ExceptionalEntries/Reasons` with `instanceName` and `lang`.
5. Verify that a reason was actually supplied in the employee's current message,
   then match that natural-language reason only against the returned live
   `reasonName` values. A reason emitted by the model but absent from the employee's
   message is rejected; the first live reason is never treated as a default.
6. Create an immutable `PendingAction` and ask for confirmation.
7. Only after confirmation, `POST api/AI/ExceptionalEntries/Request` with
   `instanceName` in the query string and `usrEmail`, `entryTime`, `entryType`,
   `reasonID`, and `remarks` in the body.

The two live ResourcePlus endpoints use different timestamp formats. The
`MissingPunchSuggestions` response supplies `suggestedEntryTime` as
`DD/MM/YYYY HH:mm`, while the live `ExceptionalEntries/Request` server requires
`entryTime` as `YYYY/MM/DD HH:mm`. The backend strictly parses the authoritative
suggestion as a calendar date and time, then formats that same value only while
serializing the POST body. For example, `22/09/2026 08:00` is submitted as
`2026/09/22 08:00`; the date and time themselves do not change. Invalid or
unsupported values fail before any POST is attempted.

ResourcePlus is authoritative for the correction. The backend preserves the exact
`suggestedEntryTime` string (`DD/MM/YYYY HH:MM`) and live `reasonID`; it never asks
the model to invent either value. The documented string direction is mapped in
backend code as follows:

- `IN` -> `1`
- `OUT` -> `2`

`attDate`, `shift`, and `isNightShift` are retained safely with the pending action.
No shift-time calculation is performed; `suggestedEntryTime` remains authoritative
for both normal and night shifts.

If there is no matching valid suggestion, the backend stops immediately. It does not
fetch reasons or AttendanceSummary, create a pending action, calculate a replacement
time, or submit a request. A valid date+direction row with an empty suggestion time
is still displayed as a factual missing punch, but is explicitly non-correctable.
Shift metadata alone—including `00:00 : 00:00`—never creates a missing punch without
an explicit `IN` or `OUT`. This business outcome uses the safe
`no_resourceplus_suggestion` observability category.

If one exact correction is available but the employee has not supplied a reason, the
backend returns `needs_reason=true` with the safe live reason names. It does not select
a `reasonID`, create a `PendingAction`, request confirmation, or perform a write. A
short-lived, non-executable `ExceptionalEntryDraft` retains only the selected date,
IN/OUT direction, exact suggested time, and transaction language. It has the same
short TTL policy as confirmation state but has no execution or confirmation operation.

A later reason-selection turn is routed before the generic agent loop. It reads
`MissingPunchSuggestions` once, verifies that the draft's exact date, direction, and
suggested time are still live, reads `ExceptionalEntries/Reasons` once, and performs a
deterministic constrained match over the live reason names. Singular/plural and casing
differences are normalized. A unique match creates the immutable `PendingAction` and
clears the draft. Unknown or ambiguous text returns one clarification and retains the
non-executable draft; it never repeats either ResourcePlus read in the same turn.

Explicit cancellation clears the draft without submitting anything. A clear unrelated
turn, such as a greeting or a new question, clears the draft and proceeds through the
normal conversation path so unfinished reason selection cannot hijack later requests.
The terminal-clarification contract also stops the generic agent immediately when a
prepare tool asks for clarification, preventing repeated prepare-tool rounds.

`AttendanceSummary.LessHrs` is not a missing-punch authorization. A date can contain
valid IN and OUT punches and still have less hours. Every correction attempt therefore
calls `MissingPunchSuggestions` before reasons. If that response contains no canonical
actionable suggestion, the backend does not fetch reasons, create a draft or pending
action, infer IN/OUT, invent a correction time, or submit anything.

When a live actionable suggestion needs a reason, `/api/chat` and the voice responses
include the backward-compatible optional fields below in addition to `message`:

```json
{
  "display_message": "What was the reason?",
  "needs_reason": true,
  "reason_options": [
    {"label": "Embassy Purposes", "value": "Embassy Purposes"}
  ],
  "requires_confirmation": false
}
```

Only live `reasonName` values are exposed. ResourcePlus `reasonID` values remain
server-owned. The frontend renders these values as wrapping, keyboard-accessible chips
and sends the selected name through the ordinary session-aware chat path. Typed reasons
remain supported, and every selection is revalidated against fresh suggestions and
fresh live reasons before a `PendingAction` can be created.

After a confirmed exceptional-entry attempt, the backend maps the safe outcome to a
semantic result (`submitted_for_approval` or `failed`) and renders it deterministically
in the language stored with the `PendingAction`. A short confirmation such as `Yes` or
`نعم` therefore cannot make a ResourcePlus response in another language override the
established transaction language. Known outcomes do not require another model call.

Rejection, expiry, mismatched confirmation, and replay retain the existing
PendingAction safeguards and never submit the ResourcePlus request. Automated tests
use mocked HTTP transports and do not call live ResourcePlus write endpoints.
