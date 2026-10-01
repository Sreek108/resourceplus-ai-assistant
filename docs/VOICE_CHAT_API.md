# Voice Chat API

ResourcePlus AI Assistant exposes two voice paths:

- `POST /api/voice/chat` is the multipart HTTP fallback.
- `/api/voice/stream` is the push-to-talk WebSocket path.

Both paths use the same sessions, confirmation store, language resolver, AI pipeline,
ResourcePlus tools, and speech synthesis behavior as text chat. HTTP voice clients may
send `X-Trace-ID`; streaming clients may put the same correlation value in the
WebSocket `start` object. See
[Production observability](PRODUCTION_OBSERVABILITY.md) for the broader tracing
contract.

## Language support

Both voice paths support the configured speech-identification candidates:

- English: `en-US`, normalized internally to `en`
- Arabic: `ar-SA`, normalized internally to `ar`

The frontend does not select a separate endpoint for Arabic. Speech-to-text language
detection, conversational language resolution, the AI response language, and text to
speech are backend responsibilities. ResourcePlus's numeric `lang` configuration is
separate from conversational language detection.

## HTTP fallback

### Endpoint and request

```text
POST /api/voice/chat
Content-Type: multipart/form-data
```

Multipart fields:

- `audio` is required. It must be a WAV upload sent as `audio/wav`, `audio/x-wav`,
  or `application/octet-stream`.
- `session_id` is optional. Reuse it across text and voice turns.
- `email` is the current demo/UAT user's email.
- `instance` is the current demo/UAT ResourcePlus instance.
- `confirmation_id` is optional. Send the backend-issued pending confirmation ID
  when one exists.

`email` and `instance` are one identity pair: send both or neither. When neither is
sent, the backend may use the complete `RP_DEFAULT_EMAIL` + `RP_INSTANCE` pair as a
local-development/test fallback. UAT and production require the request pair and
never use those defaults. The backend never combines one request field with one
default field.

The upload limit is 10 MiB. The client does not send transcript text or choose a
language. The response is JSON.

### Successful HTTP response

```json
{
  "success": true,
  "message": "You have two unread notifications.",
  "display_message": "You have two unread notifications.",
  "language": "en",
  "tools_used": ["get_notifications"],
  "session_id": "session-value",
  "requires_confirmation": false,
  "confirmation_id": null,
  "needs_reason": false,
  "reason_options": null,
  "transcript": "Show my notifications",
  "detected_language": "en",
  "detected_locale": "en-US",
  "audio_base64": "UklGR...",
  "audio_mime_type": "audio/wav"
}
```

`audio_base64` is the Base64-encoded synthesized response. It is not an audio URL,
and the endpoint does not return a direct MP3 or file response. The frontend decodes
the Base64 data and uses `audio_mime_type` when constructing playable audio instead
of hardcoding a MIME type. The current TTS output is `audio/wav`.

The spoken audio is synthesized from the assistant's dedicated, user-facing speech
rendering. Tool arguments, ResourcePlus IDs, confirmation IDs, URLs, and debug details
are not sent to text to speech.

### HTTP errors

- `400`: empty audio, unsupported audio type, or no recognizable speech
- `413`: upload exceeds 10 MiB
- `502`: Azure recognition or synthesis temporarily failed
- `503`: Azure Speech SDK, key, region, or required voice is not configured

Errors do not expose Azure keys, SDK cancellation details, stack traces, or
ResourcePlus internals.

## WebSocket streaming

### Endpoint

```text
/api/voice/stream
```

Use `ws://` when the application is served over HTTP and `wss://` when it is served
over HTTPS. The frontend derives the WebSocket scheme and host from the configured
API base URL or the browser origin.

Production deployment example:

```text
wss://app.resourceplus.app/ai-assistant/api/voice/stream
```

### Step 1: START

The first WebSocket frame must be a text frame containing a JSON object:

```json
{
  "type": "start",
  "sample_rate": 16000,
  "session_id": "session-value",
  "email": "employee@company.com",
  "instance": "Universal",
  "confirmation_id": null,
  "trace_id": "voice-demo-001",
  "debug": false,
  "progressive_events": true
}
```

Fields:

- `type` is required and must be exactly `"start"`.
- `sample_rate` is required and must be exactly `16000`.
- `session_id` is optional. When supplied, it is a string from 1 to 128
  characters.
- `email` and `instance` identify the current demo/UAT ResourcePlus user and tenant.
  They must be supplied together. Neither value is chosen by the AI model.
- `confirmation_id` is optional. When supplied, it is a string from 1 to 128
  characters.
- `trace_id` is optional. It is trimmed, must be 8 to 64 ASCII characters, must
  start with an ASCII letter or digit, and may otherwise contain only ASCII letters,
  digits, `.`, `_`, `:`, or `-`. If an `X-Trace-ID` handshake header is also present,
  the two validated values must match.
- `debug` is optional. Only the literal JSON value `true` adds diagnostics to the
  final response.
- `progressive_events` is optional. When `true`, the server adds progress events.
  When omitted, the established READY/binary PCM/END/FINAL contract is unchanged.

There are no START fields for encoding, channels, bit depth, or endianness.

### Step 2: READY

After the streaming recognizer starts, the server sends:

```json
{
  "type": "ready"
}
```

The frontend waits for this message before treating voice streaming as ready and
sending queued microphone chunks.

### Step 3: audio frames

Client-to-server streaming audio is:

- raw binary WebSocket frames
- signed PCM16 / 16-bit PCM
- 16,000 Hz
- mono
- no WAV header
- not Base64
- not wrapped in JSON

Each PCM chunk must be nonempty and have an even byte length. The maximum cumulative
audio per connection is 10 MiB.

The backend does not expose an endianness field and does not transform the sample
bytes. It passes each PCM chunk to Azure's configured 16-bit PCM stream.

```javascript
websocket.send(pcmArrayBuffer);
```

### Step 4: END or CANCEL

When recording ends normally, send a text JSON frame:

```json
{
  "type": "end"
}
```

To abandon the stream, send:

```json
{
  "type": "cancel"
}
```

`cancel` terminates recognition and closes the connection normally with WebSocket
code `1000`. It does not produce a final JSON result.

### Step 5: FINAL

After END, successful recognition, AI processing, and speech synthesis produce one
text JSON frame:

```json
{
  "type": "final",
  "success": true,
  "message": "Assistant display response",
  "display_message": "Assistant display response",
  "language": "en",
  "tools_used": [],
  "session_id": "session-value",
  "requires_confirmation": false,
  "confirmation_id": null,
  "needs_reason": false,
  "reason_options": null,
  "transcript": "Recognized speech",
  "detected_language": "en",
  "detected_locale": "en-US",
  "audio_base64": "UklGR...",
  "audio_mime_type": "audio/wav"
}
```

When reason selection is required, `reason_options` contains objects with `label`
and `value` string fields. Otherwise it is `null`.

The two audio directions deliberately use different wire representations:

- Client to server: raw binary PCM16 WebSocket frames.
- Server to client: Base64-encoded WAV in `audio_base64` inside the final JSON.

The current synthesized output is RIFF/WAV, 24 kHz, 16-bit, mono PCM. The frontend
still uses the returned `audio_mime_type` rather than hardcoding the MIME type.

With `progressive_events: true`, the server may additionally emit `listening`,
`transcript_final`, `processing`, and `assistant_text`. The text event is sent before
TTS, so visual output is not blocked by synthesis. A TTS-only failure emits an `error`
with `scope: "tts"`, followed by `final` with the valid text and an empty
`audio_base64`. It does not retry or repeat any ResourcePlus operation. Progressive
partial STT and binary TTS streaming are intentionally deferred.

If START contains `"debug": true`, the final object additionally contains:

```json
{
  "interaction_id": "interaction-uuid",
  "trace_id": "voice-demo-001",
  "timings": {
    "stage_name": 0.123456,
    "total": 0.456789
  }
}
```

`timings` maps the measured stages that ran to durations in seconds. Its stage keys
vary with the executed path.

The server closes a completed stream normally with WebSocket code `1000` after
sending FINAL.

### WebSocket errors

WebSocket JSON errors have this structure:

```json
{
  "type": "error",
  "code": "error-code",
  "message": "Safe client-facing message"
}
```

Currently implemented error codes:

- `no_speech`
- `invalid_stream_state`
- `voice_unavailable`
- `speech_recognition_failed`
- `speech_synthesis_failed`
- `voice_processing_failed`

Invalid trace-header validation or a disallowed browser origin can close the socket
with WebSocket code `1008` without sending a JSON error. Legacy clients that omit
`progressive_events` receive no additional progress event types.

### Complete streaming example

```text
CONNECT wss://app.resourceplus.app/ai-assistant/api/voice/stream

-> text JSON
   {
     "type": "start",
     "sample_rate": 16000,
     "session_id": "session-value",
     "email": "employee@company.com",
     "instance": "Universal",
     "confirmation_id": null,
     "trace_id": "voice-demo-001",
     "debug": false
   }

<- text JSON
   {"type": "ready"}

-> binary PCM16 chunk
-> binary PCM16 chunk
-> ...

-> text JSON
   {"type": "end"}

<- text JSON
   {
     "type": "final",
     "success": true,
     "message": "You have two unread notifications.",
     "display_message": "You have two unread notifications.",
     "language": "en",
     "tools_used": ["get_notifications"],
     "session_id": "session-value",
     "requires_confirmation": false,
     "confirmation_id": null,
     "needs_reason": false,
     "reason_options": null,
     "transcript": "Show my notifications",
     "detected_language": "en",
     "detected_locale": "en-US",
     "audio_base64": "UklGR...",
     "audio_mime_type": "audio/wav"
   }

Frontend renders transcript/message.
Frontend decodes audio_base64.
Frontend plays the decoded bytes using audio_mime_type.
Server closes with WebSocket code 1000.
```

## Sessions and confirmation

Use the returned `session_id` on later calls. The same ID may move among
`/api/chat`, `/api/voice/chat`, and `/api/voice/stream`.

Conversation ownership is the combined `email + instance + session_id`. If the same
session ID is presented with a different email or instance, the backend rejects that
turn and does not reuse the existing history. Pending actions carry the identity that
created them and can be confirmed only by the same identity in the same session; an
identity mismatch performs no ResourcePlus write.

Write requests create the same immutable `PendingAction` used by text chat. Voice
responses return `requires_confirmation=true` and a `confirmation_id`. A later
natural affirmative or negative utterance is interpreted conversationally; it is
not limited to a hardcoded Arabic phrase list. A confirmed write executes only the
already-stored arguments.

This includes ResourcePlus v2 less-hours FromSummary and exceptional-entry
cancellation. Attendance eligibility, live reasons, optional side/minutes rules,
and cancellation candidate resolution are identical to text chat. ResourcePlus is
the sole auto-approval authority. With progressive events enabled, `assistant_text`
can show the returned message/warning before TTS completes; a TTS-only failure never
retries the HR write.

The frontend-supplied `email` + `instance` pair is demo/UAT identity transport only.
It is not production authentication. Production must replace it with identity derived
from a validated ResourcePlus authentication/session mechanism.

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

## HTTP fallback PowerShell example

```powershell
curl.exe -X POST "http://127.0.0.1:8000/api/voice/chat" `
  -F "audio=@C:\path\to\utterance.wav;type=audio/wav" `
  -F "session_id=demo-session" `
  -F "email=employee@company.com" `
  -F "instance=Universal"
```

Do not send a confirmation during a live demo unless the displayed pending action
has been reviewed and the ResourcePlus write is intended.
