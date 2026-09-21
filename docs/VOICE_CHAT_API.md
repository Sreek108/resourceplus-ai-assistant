# Voice Chat API

HTTP voice clients may send optional `X-Trace-ID` and receive the effective value in
the response header. Streaming clients may add the same value as `trace_id` in the
WebSocket `start` object. This is correlation metadata only; speech, session, and
confirmation behavior are unchanged. See
[Production observability](PRODUCTION_OBSERVABILITY.md).

## Endpoint

`POST /api/voice/chat`

The endpoint accepts `multipart/form-data` and uses the same chat orchestration,
OpenAI agent, ResourcePlus tools, sessions, and confirmation store as
`POST /api/chat`.

## Multipart request

- `audio` (required file): WAV audio. The initial demo accepts `audio/wav`,
  `audio/x-wav`, or WAV bytes uploaded as `application/octet-stream`.
- `session_id` (optional text): reuse this value across text and voice turns.
- `confirmation_id` (optional text): send the pending confirmation ID when the
  client has one.

The maximum upload size is 10 MB. The client does not send transcript text or a
language selection.

## Language detection

Azure Speech performs at-start language identification with these configurable
candidates:

- `en-US`, normalized internally to `en`
- `ar-SA`, normalized internally to `ar`

The transcript is passed to the same OpenAI agent as text chat. Azure Speech does
not classify HR intents. ResourcePlus's numeric `lang` configuration remains
separate from conversational language detection.

## Successful response

```json
{
  "success": true,
  "transcript": "Show my notifications",
  "detected_language": "en",
  "detected_locale": "en-US",
  "message": "You have two unread notifications.",
  "language": "en",
  "tools_used": ["get_notifications"],
  "session_id": "session-value",
  "requires_confirmation": false,
  "confirmation_id": null,
  "audio_base64": "UklGR...",
  "audio_mime_type": "audio/wav"
}
```

`audio_base64` contains WAV audio synthesized only from the final user-facing
`message`. Tool arguments, ResourcePlus IDs, confirmation IDs, URLs, and debug
details are never sent to text-to-speech.

## Sessions and confirmation

Use the returned `session_id` on later calls. The same ID may move between
`/api/chat` and `/api/voice/chat`.

Write requests create the same immutable `PendingAction` used by text chat. The
voice response returns `requires_confirmation=true` and a `confirmation_id`. A
later natural affirmative or negative utterance is interpreted conversationally;
it is not limited to a hardcoded Arabic phrase list. A confirmed write executes
only the already-stored arguments.

## Errors

- `400`: empty audio, unsupported audio type, or no recognizable speech.
- `413`: upload exceeds 10 MB.
- `502`: Azure recognition or synthesis temporarily failed.
- `503`: Azure Speech SDK, key, region, or required voice is not configured.

Errors do not expose Azure keys, SDK cancellation details, stack traces, or
ResourcePlus internals.

## Configuration

```dotenv
AZURE_SPEECH_KEY=
AZURE_SPEECH_REGION=
AZURE_SPEECH_EN_LOCALE=en-US
AZURE_SPEECH_AR_LOCALE=ar-SA
AZURE_SPEECH_EN_VOICE=en-US-AvaNeural
AZURE_SPEECH_AR_VOICE=ar-SA-ZariyahNeural
```

The Arabic voice may instead be set to `ar-SA-HamedNeural`.

## PowerShell test

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/voice/chat" `
  -F "audio=@C:\path\to\utterance.wav;type=audio/wav" `
  -F "session_id=demo-session"
```

Do not send a confirmation during a live demo unless the displayed pending action
has been reviewed and the ResourcePlus write is intended.
