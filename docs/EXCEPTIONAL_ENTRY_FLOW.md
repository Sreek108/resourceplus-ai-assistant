# Exceptional-entry workflows

ResourcePlus has two distinct correction workflows. The backend chooses between
them from the employee's request; the model never supplies identity or internal IDs.

## Less-hours correction (preferred v2 flow)

Normal less-hours, late-arrival, and early-departure corrections use this sequence:

1. `GET api/AI/AttendanceSummary` for the backend-resolved date or date range.
2. Classify each authoritative day row:
   - `LessHrs > 00:00` with at least one punch is a correction candidate.
   - `Absent` or no punches is not an exception; offer Leave or Business Travel.
   - Week End, Holiday, Leave, and Business Travel are not correctable.
   - `LessHrs = 00:00` has nothing to correct.
3. If multiple days are candidates, show the grounded dates and require a selection.
4. `GET api/AI/ExceptionalEntries/Balance` is date-scoped and informational. A
   less-hours range read skips it for zero candidates, uses the sole candidate date
   for one candidate, and omits a generic allowance for multiple candidates until a
   date is selected. Clear standalone balance questions use this read directly;
   only returned policy, limit, used, remaining, reset-period, and reset-date values
   may be displayed. `limitType=1` means an exception count and `limitType=2` means
   buffer minutes. Other values are not interpreted. `hasPolicy=false` does not
   create an inferred policy.
5. Read live `GET api/AI/ExceptionalEntries/Reasons`, expose names only, and match
   the employee's selection. The ID remains backend-owned.
6. Re-fetch AttendanceSummary and live reasons while preparing an immutable
   `PendingAction`, then ask for explicit confirmation.
7. After same-email, same-instance, same-session confirmation, execute exactly the
   stored `POST api/AI/ExceptionalEntries/FromSummary` action once.

The POST body always contains backend identity `usrEmail`, verified `attDate`, the
live `reasonID`, and employee remarks. It normally omits `entryType` and `minutes`:

- `entryType=1` only when the employee explicitly requests late IN only.
- `entryType=2` only when the employee explicitly requests early OUT only.
- `minutes` only when the employee explicitly requests a partial duration.
- Selecting partial minutes does not imply an entry type.

The assistant does not calculate a corrected punch time or split the missing duration.
ResourcePlus performs that logic. ResourcePlus also exclusively decides whether the
result is auto-approved or submitted for manager approval. The assistant never
approves, rejects, consumes allowance, or predicts that outcome. `success` is the
authoritative operation result: `true` permits interpretation of `isAutoApproved`,
while `false` is a failure even if other fields appear success-like. The returned
`message`, `warning`, `isAutoApproved`, `requestedMinutes`, `remaining`, `resetsOn`,
and created `entries` are surfaced without hiding partial/split failures.

A correction-candidate row is not a promise that FromSummary will accept the request.
ResourcePlus remains authoritative for business-policy rejection. In particular, an
observed response advising leave when missing time exceeded four hours is surfaced as
returned and is not hardcoded as a universal AI-side threshold.

For compatibility with early v2 responses only, a response with no `success` member
but a boolean `isAutoApproved` is treated as an accepted operation; either boolean
value is accepted because `false` means manager approval, not failure. No other field
can substitute for an absent `success` value.

## Explicit missing-punch / exact-time legacy flow

`GET api/AI/MissingPunchSuggestions` and
`POST api/AI/ExceptionalEntries/Request` remain supported for explicit missing IN/OUT
or exact-time legacy cases. The backend selects one authoritative suggested time,
fetches live reasons, creates a PendingAction, and posts only after confirmation.
This flow is not used as the normal solution for an Absent/no-punch day or a regular
LessHrs correction.

## Cancellation

Natural cancellation first reads `GET api/AI/ExceptionalEntries`. Only rows that
explicitly report `isCancellable=true`/`canCancel=true`, or have documented status
`Not Approved`, are normal candidates. An explicit false flag overrides any status.
`Approved` and `Rejected` are not cancellable. Pending, Pending Approval, Pending For
Approval, and Submitted remain conservative compatibility aliases when no explicit
flag is present. No match produces no write; multiple matches produce a safe table
and clarification. One match creates a cancellation PendingAction whose raw
`exceptionalID` is never displayed. Confirmation executes
`POST api/AI/ExceptionalEntries/Cancel` exactly once with backend identity and that
stored ID.

All flows retain confirmation TTL, consume-before-execute replay prevention,
failure non-replayability, identity/session isolation, write grounding, and private
audit behavior. Automated tests mock all write endpoints; validation performs no live
ResourcePlus writes.
