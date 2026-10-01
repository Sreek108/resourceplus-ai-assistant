export function resolveApiBaseUrl({
  configuredBase = "",
  developmentDefault = "",
  isDev = false,
} = {}) {
  if (!isDev) return "";
  return configuredBase.trim().replace(/\/$/, "") || developmentDefault;
}

const API_BASE_URL = resolveApiBaseUrl({
  configuredBase: import.meta.env.VITE_API_BASE_URL,
  developmentDefault: import.meta.env.DEV ? "http://127.0.0.1:8001" : "",
  isDev: import.meta.env.DEV,
});

export function buildVoiceStreamUrl({
  apiBaseUrl = API_BASE_URL,
  pageLocation = globalThis.location,
} = {}) {
  const origin = pageLocation?.origin;
  if (!apiBaseUrl && !origin) {
    throw new Error("A browser origin is required for streaming voice.");
  }
  const socketUrl = new URL("/api/voice/stream", apiBaseUrl || origin);
  socketUrl.protocol = socketUrl.protocol === "https:" ? "wss:" : "ws:";
  return socketUrl.toString();
}

export class ClientRequestError extends Error {
  constructor(message, code) {
    super(message);
    this.name = "ClientRequestError";
    this.code = code;
  }
}

function demoIdentityFields(email, instance) {
  const normalizedEmail = typeof email === "string" ? email.trim() : "";
  const normalizedInstance = typeof instance === "string" ? instance.trim() : "";
  if (Boolean(normalizedEmail) !== Boolean(normalizedInstance)) {
    throw new ClientRequestError(
      "The current demo user identity is incomplete.",
      "invalid_identity",
    );
  }
  return normalizedEmail
    ? { email: normalizedEmail, instance: normalizedInstance }
    : {};
}

function recoverCompletedVoiceTtsFailure(payload) {
  const detail = payload?.detail;
  const completed = detail?.response;
  const hasConfirmation = Boolean(completed?.requires_confirmation);
  if (
    detail?.code !== "speech_synthesis_failed"
    || detail?.error_category !== "speech_synthesis_failed"
    || detail?.result_status !== "completed_with_tts_error"
    || detail?.tts_generated !== false
    || completed?.success !== true
    || typeof completed?.message !== "string"
    || !completed.message.trim()
    || typeof completed?.language !== "string"
    || detail?.response_language !== completed.language
    || detail?.assistant_text !== completed.message
    || typeof completed?.session_id !== "string"
    || !completed.session_id.trim()
    || typeof detail?.transcript !== "string"
    || !detail.transcript.trim()
    || typeof detail?.detected_language !== "string"
    || !detail.detected_language.trim()
    || (hasConfirmation && (
      typeof completed?.confirmation_id !== "string"
      || !completed.confirmation_id.trim()
    ))
  ) {
    return null;
  }
  return {
    ...completed,
    transcript: detail.transcript,
    detected_language: detail.detected_language || completed.language,
    detected_locale: detail.detected_locale,
    audio_base64: "",
    audio_mime_type: "audio/wav",
    tts_unavailable: true,
    tts_error_category: detail.error_category,
  };
}

async function parseResponse(response, requestType) {
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    // A non-JSON upstream failure is still reported using a safe status message.
  }
  if (!response.ok) {
    if (requestType === "voice") {
      const completedVoiceResult = recoverCompletedVoiceTtsFailure(payload);
      if (completedVoiceResult) return completedVoiceResult;
    }
    const serverCode = payload?.code || payload?.detail?.code;
    const failure = friendlyStatusMessage(response.status, requestType, serverCode);
    throw new ClientRequestError(failure.message, failure.code);
  }
  return payload;
}

function friendlyStatusMessage(status, requestType, serverCode) {
  if (serverCode === "no_speech" || (status === 400 && requestType === "voice")) {
    return {
      code: "no_speech",
      message: "I couldn’t hear enough speech. Hold the mic and try again.",
    };
  }
  if (status === 413) {
    return {
      code: "audio_too_large",
      message: "That recording is too long. Please try a shorter message.",
    };
  }
  if (status === 409 || status === 410) {
    return {
      code: "confirmation_expired",
      message: "That confirmation has expired. Please ask me to prepare it again.",
    };
  }
  if (status === 422) {
    return {
      code: "invalid_request",
      message: "I couldn’t process that message. Please try again.",
    };
  }
  if (serverCode === "speech_recognition_failed") {
    return {
      code: serverCode,
      message: "I couldn’t recognize that voice message. Please try again or type instead.",
    };
  }
  if (serverCode === "speech_synthesis_failed") {
    return {
      code: serverCode,
      message: "I understood you, but couldn’t create a spoken reply. Please try again or type instead.",
    };
  }
  if (serverCode === "voice_unavailable") {
    return {
      code: serverCode,
      message: "Voice is temporarily unavailable. You can keep chatting by typing.",
    };
  }
  if (serverCode === "resourceplus_unavailable" || status === 504) {
    return {
      code: "resourceplus_unavailable",
      message: "ResourcePlus is temporarily unavailable. Please try again shortly.",
    };
  }
  if (serverCode === "assistant_unavailable") {
    return {
      code: "assistant_unavailable",
      message: "The assistant is temporarily unavailable. Please try again shortly.",
    };
  }
  if (requestType === "voice" && status === 502) {
    return {
      code: "speech_processing_failed",
      message: "I couldn’t process that voice message. You can try again or type instead.",
    };
  }
  if (requestType === "voice" && status === 503) {
    return {
      code: "voice_backend_unavailable",
      message: "Voice is temporarily unavailable. You can keep chatting by typing.",
    };
  }
  if (status === 502) {
    return {
      code: "request_failed",
      message: "The assistant couldn’t complete that request. Please try again.",
    };
  }
  if (status === 503) {
    return {
      code: "assistant_unavailable",
      message: "The assistant is temporarily unavailable. Please try again shortly.",
    };
  }
  return { code: "request_failed", message: "Something went wrong. Please try again." };
}

export async function sendChat({ message, sessionId, confirmationId, email, instance }) {
  try {
    const identity = demoIdentityFields(email, instance);
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message,
        ...(sessionId ? { session_id: sessionId } : {}),
        ...identity,
        ...(confirmationId ? { confirmation_id: confirmationId } : {}),
      }),
    });
    return await parseResponse(response, "text");
  } catch (error) {
    if (error instanceof TypeError) {
      throw new ClientRequestError(
        "I couldn’t connect to the assistant. Check the connection and try again.",
        "network_failure",
      );
    }
    throw error;
  }
}

export async function sendVoice({ audio, sessionId, confirmationId, email, instance }) {
  const form = new FormData();
  form.append("audio", audio, "resourceplus-voice.wav");
  if (sessionId) form.append("session_id", sessionId);
  const identity = demoIdentityFields(email, instance);
  if (identity.email) {
    form.append("email", identity.email);
    form.append("instance", identity.instance);
  }
  if (confirmationId) form.append("confirmation_id", confirmationId);
  try {
    const response = await fetch(`${API_BASE_URL}/api/voice/chat`, {
      method: "POST",
      body: form,
    });
    return await parseResponse(response, "voice");
  } catch (error) {
    if (error instanceof TypeError) {
      throw new ClientRequestError(
        "I couldn’t connect to voice right now. You can keep chatting by typing.",
        "network_failure",
      );
    }
    throw error;
  }
}

export function openVoiceStream({
  sessionId,
  confirmationId,
  email,
  instance,
  debug = false,
  onEvent,
}) {
  let identity;
  try {
    identity = demoIdentityFields(email, instance);
  } catch (error) {
    return Promise.reject(error);
  }
  if (!globalThis.WebSocket) {
    return Promise.reject(
      new ClientRequestError("Streaming voice is unavailable.", "stream_unavailable"),
    );
  }
  return new Promise((resolve, reject) => {
    const socket = new WebSocket(buildVoiceStreamUrl());
    socket.binaryType = "arraybuffer";
    let ready = false;
    let settled = false;
    let finalSettled = false;
    let cancelled = false;
    let ended = false;
    let resolveFinal;
    let rejectFinal;
    const finalResponse = new Promise((resolveResult, rejectResult) => {
      resolveFinal = resolveResult;
      rejectFinal = rejectResult;
    });

    const connectTimer = window.setTimeout(() => {
      fail("Streaming voice is unavailable.", "stream_connect_timeout");
      socket.close();
    }, 10_000);

    function rejectFinalOnce(error) {
      if (finalSettled) return;
      finalSettled = true;
      rejectFinal(error);
    }

    function fail(message, code = "stream_unavailable") {
      const error = new ClientRequestError(message, code);
      if (!ready) {
        if (!settled) {
          settled = true;
          window.clearTimeout(connectTimer);
          reject(error);
        }
        return;
      }
      rejectFinalOnce(error);
    }

    socket.onopen = () => {
      socket.send(JSON.stringify({
        type: "start",
        sample_rate: 16_000,
        ...(sessionId ? { session_id: sessionId } : {}),
        ...identity,
        ...(confirmationId ? { confirmation_id: confirmationId } : {}),
        ...(debug ? { debug: true } : {}),
        ...(onEvent ? { progressive_events: true } : {}),
      }));
    };
    socket.onmessage = (event) => {
      if (typeof event.data !== "string") return;
      let payload;
      try {
        payload = JSON.parse(event.data);
      } catch {
        fail("I couldnâ€™t process that voice message.", "stream_invalid_response");
        return;
      }
      if (payload.type === "ready" && !ready) {
        ready = true;
        settled = true;
        window.clearTimeout(connectTimer);
        resolve({
          sendChunk(chunk) {
            if (socket.readyState !== WebSocket.OPEN) {
              throw new ClientRequestError(
                "The streaming voice connection was interrupted.",
                "stream_disconnected",
              );
            }
            socket.send(chunk);
          },
          finish() {
            if (!ended) {
              ended = true;
              if (socket.readyState === WebSocket.OPEN) {
                socket.send(JSON.stringify({ type: "end" }));
              } else {
                fail(
                  "The streaming voice connection was interrupted.",
                  "stream_disconnected",
                );
              }
            }
            return finalResponse;
          },
          cancel() {
            if (cancelled) return;
            cancelled = true;
            if (socket.readyState === WebSocket.OPEN) {
              socket.send(JSON.stringify({ type: "cancel" }));
            }
            finalSettled = true;
            socket.close();
          },
        });
        return;
      }
      if (payload.type === "final") {
        if (!finalSettled) {
          finalSettled = true;
          resolveFinal(payload);
        }
        socket.close();
        return;
      }
      if (["listening", "transcript_final", "processing", "assistant_text"].includes(payload.type)) {
        onEvent?.(payload);
        return;
      }
      if (payload.type === "error") {
        if (payload.scope === "tts") {
          onEvent?.(payload);
          return;
        }
        fail(
          payload.message || "I couldnâ€™t process that voice message.",
          payload.code || "stream_failed",
        );
        socket.close();
      }
    };
    socket.onerror = () => {
      fail("Streaming voice is unavailable.", "stream_unavailable");
      socket.close();
    };
    socket.onclose = () => {
      window.clearTimeout(connectTimer);
      if (!settled) {
        fail("Streaming voice is unavailable.", "stream_unavailable");
      } else if (!finalSettled && !cancelled) {
        fail(
          "The streaming voice connection was interrupted.",
          "stream_disconnected",
        );
      }
    };
  });
}

export function audioBase64ToUrl(base64, mimeType = "audio/wav") {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index += 1) {
    bytes[index] = binary.charCodeAt(index);
  }
  return URL.createObjectURL(new Blob([bytes], { type: mimeType }));
}
