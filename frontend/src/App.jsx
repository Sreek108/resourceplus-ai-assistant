import { useEffect, useMemo, useRef, useState } from "react";
import ChatMessage from "./components/ChatMessage";
import Composer from "./components/Composer";
import Header from "./components/Header";
import Sidebar from "./components/Sidebar";
import Welcome from "./components/Welcome";
import { createAudioPlaybackManager } from "./lib/audioPlayback";
import { audioBase64ToUrl, openVoiceStream, sendChat, sendVoice } from "./lib/api";
import { messageLanguage } from "./lib/language";
import { isValidWavBlob, startWavRecording } from "./lib/wavRecorder";

const SESSION_KEY = "resourceplus.demo.session";
const CONFIRMATION_KEY = "resourceplus.demo.confirmation";
const MESSAGES_KEY = "resourceplus.demo.messages";
const MIN_RECORDING_MS = 600;
const MIN_WAV_BYTES = 1_000;
const NO_SPEECH_CODES = new Set(["no_speech", "no_audio", "no_recognized_speech"]);

function uniqueId() {
  return globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`;
}

function loadMessages() {
  try {
    const stored = JSON.parse(sessionStorage.getItem(MESSAGES_KEY) || "[]");
    return Array.isArray(stored)
      ? stored.map(({ audioUrl, ...message }) => message)
      : [];
  } catch {
    return [];
  }
}

export default function App() {
  const [messages, setMessages] = useState(loadMessages);
  const [sessionId, setSessionId] = useState(
    () => sessionStorage.getItem(SESSION_KEY) || "",
  );
  const [confirmationId, setConfirmationId] = useState(
    () => sessionStorage.getItem(CONFIRMATION_KEY) || "",
  );
  const [input, setInput] = useState("");
  const [status, setStatus] = useState("idle");
  const [recordingSeconds, setRecordingSeconds] = useState(0);
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const recorderRef = useRef(null);
  const voiceStreamRef = useRef(null);
  const voiceStreamPromiseRef = useRef(null);
  const voiceStreamErrorRef = useRef(null);
  const voiceAttemptRef = useRef(0);
  const pendingPcmRef = useRef([]);
  const holdActiveRef = useRef(false);
  const recordingStartedAtRef = useRef(0);
  const voiceSubmissionRef = useRef(false);
  const playbackManagerRef = useRef(null);
  const playbackTokenRef = useRef(0);
  const reasonSelectionRef = useRef(false);
  const chatScrollRef = useRef(null);
  const audioUrlsRef = useRef(new Set());
  const debug = useMemo(
    () => new URLSearchParams(window.location.search).get("debug") === "true",
    [],
  );
  const busy = ["sending", "listening", "processing", "speaking"].includes(status);
  if (!playbackManagerRef.current) {
    playbackManagerRef.current = createAudioPlaybackManager();
  }

  useEffect(() => {
    const serializable = messages.map(({ audioUrl, ...message }) => message);
    sessionStorage.setItem(MESSAGES_KEY, JSON.stringify(serializable));
    const scrollArea = chatScrollRef.current;
    if (!scrollArea) return;
    const top = messages.length ? scrollArea.scrollHeight : 0;
    if (typeof scrollArea.scrollTo === "function") {
      scrollArea.scrollTo({
        top,
        behavior: messages.length ? "smooth" : "auto",
      });
    } else {
      scrollArea.scrollTop = top;
    }
  }, [messages]);

  useEffect(() => {
    if (sessionId) sessionStorage.setItem(SESSION_KEY, sessionId);
    else sessionStorage.removeItem(SESSION_KEY);
  }, [sessionId]);

  useEffect(() => {
    if (confirmationId) sessionStorage.setItem(CONFIRMATION_KEY, confirmationId);
    else sessionStorage.removeItem(CONFIRMATION_KEY);
  }, [confirmationId]);

  useEffect(() => {
    if (status !== "listening") return undefined;
    const timer = window.setInterval(
      () => setRecordingSeconds((seconds) => seconds + 1),
      1000,
    );
    return () => window.clearInterval(timer);
  }, [status]);

  useEffect(() => {
    const stopForPageLifecycle = () => cleanupVoiceResources({ resetStatus: false });
    const stopWhenHidden = () => {
      if (document.visibilityState === "hidden") stopForPageLifecycle();
    };
    window.addEventListener("pagehide", stopForPageLifecycle);
    document.addEventListener("visibilitychange", stopWhenHidden);
    return () => {
      window.removeEventListener("pagehide", stopForPageLifecycle);
      document.removeEventListener("visibilitychange", stopWhenHidden);
      cleanupVoiceResources({ resetStatus: false });
      playbackTokenRef.current += 1;
      void playbackManagerRef.current?.dispose();
      for (const url of audioUrlsRef.current) URL.revokeObjectURL(url);
    };
  }, []);

  function cleanupVoiceResources({ resetStatus = true } = {}) {
    voiceAttemptRef.current += 1;
    holdActiveRef.current = false;
    voiceSubmissionRef.current = false;
    const recorder = recorderRef.current;
    const stream = voiceStreamRef.current;
    const streamPromise = voiceStreamPromiseRef.current;
    recorderRef.current = null;
    voiceStreamRef.current = null;
    voiceStreamPromiseRef.current = null;
    voiceStreamErrorRef.current = null;
    pendingPcmRef.current = [];
    void recorder?.cancel?.();
    stream?.cancel?.();
    if (streamPromise) {
      void streamPromise
        .then((resolvedStream) => {
          if (resolvedStream && resolvedStream !== stream) resolvedStream.cancel?.();
        })
        .catch(() => {});
    }
    if (resetStatus) {
      setStatus("idle");
      setRecordingSeconds(0);
    }
  }

  function applySession(response) {
    if (response.session_id) setSessionId(response.session_id);
    if (response.requires_confirmation && response.confirmation_id) {
      setConfirmationId(response.confirmation_id);
    } else {
      setConfirmationId("");
    }
  }

  function assistantMessage(response, extras = {}) {
    return {
      id: uniqueId(),
      role: "assistant",
      text:
        response.display_message
        || response.message
        || "I couldn’t complete that request. Please try again.",
      language:
        response.language
        || messageLanguage(response.display_message || response.message),
      requiresConfirmation: Boolean(response.requires_confirmation),
      needsReason: Boolean(response.needs_reason),
      reasonOptionsActive: Boolean(response.needs_reason),
      reasonOptions: Array.isArray(response.reason_options)
        ? response.reason_options
            .filter(
              (option) =>
                option
                && typeof option.label === "string"
                && typeof option.value === "string",
            )
            .map(({ label, value }) => ({ label, value }))
        : [],
      ...extras,
      debug: {
        detected_language: extras.detectedLanguage || response.language,
        detected_locale: extras.detectedLocale,
        tools_used: response.tools_used || [],
        session_id: response.session_id,
        requires_confirmation: Boolean(response.requires_confirmation),
        ...(response.interaction_id ? { interaction_id: response.interaction_id } : {}),
        ...(response.timings ? { timings: JSON.stringify(response.timings) } : {}),
      },
    };
  }

  function appendTurn(userMessage, assistant) {
    setMessages((current) => [
      ...current.map((message) => ({
        ...message,
        requiresConfirmation: false,
        reasonOptionsActive: false,
      })),
      userMessage,
      assistant,
    ]);
  }

  async function submitText(textOverride, confirmationOverride) {
    const text = (textOverride ?? input).trim();
    if (!text || busy) return false;
    setInput("");
    setError("");
    setNotice("");
    setStatus("sending");
    const userMessage = {
      id: uniqueId(),
      role: "user",
      text,
      language: messageLanguage(text),
    };
    setMessages((current) => [
      ...current.map((message) => ({
        ...message,
        reasonOptionsActive: false,
      })),
      userMessage,
    ]);
    try {
      const response = await sendChat({
        message: text,
        sessionId,
        confirmationId: confirmationOverride ?? confirmationId,
      });
      applySession(response);
      setMessages((current) => [
        ...current.map((message) => ({ ...message, requiresConfirmation: false })),
        assistantMessage(response),
      ]);
      return true;
    } catch (requestError) {
      setError(requestError.message || "Could not send your message. Please try again.");
      return false;
    } finally {
      setStatus("idle");
    }
  }

  async function submitReason(messageId, value) {
    if (busy || reasonSelectionRef.current) return;
    reasonSelectionRef.current = true;
    setMessages((current) =>
      current.map((message) =>
        message.id === messageId
          ? {
              ...message,
              selectedReason: value,
              reasonSelectionPending: true,
              reasonOptionsActive: false,
            }
          : message,
      ),
    );
    const sent = await submitText(value);
    if (!sent) {
      setMessages((current) =>
        current.map((message) =>
          message.id === messageId
            ? {
                ...message,
                selectedReason: undefined,
                reasonSelectionPending: false,
                reasonOptionsActive: true,
              }
            : message,
        ),
      );
    }
    reasonSelectionRef.current = false;
  }

  async function beginVoiceHold() {
    if (busy || holdActiveRef.current || voiceSubmissionRef.current) return;
    // This runs inside the microphone gesture, before asynchronous voice work.
    // Reusing the same context also lets a later gesture resume it after iOS
    // backgrounding without creating a context per response.
    void playbackManagerRef.current.unlock();
    holdActiveRef.current = true;
    recordingStartedAtRef.current = performance.now();
    setError("");
    setNotice("");
    setRecordingSeconds(0);
    setStatus("listening");
    const attempt = voiceAttemptRef.current + 1;
    voiceAttemptRef.current = attempt;
    try {
      pendingPcmRef.current = [];
      voiceStreamRef.current = null;
      voiceStreamErrorRef.current = null;
      voiceStreamPromiseRef.current = openVoiceStream({
        sessionId,
        confirmationId,
        debug,
      })
        .then((streamController) => {
          if (
            voiceAttemptRef.current !== attempt
            || (!holdActiveRef.current && !voiceSubmissionRef.current)
          ) {
            streamController.cancel?.();
            return null;
          }
          voiceStreamRef.current = streamController;
          for (const chunk of pendingPcmRef.current) streamController.sendChunk(chunk);
          pendingPcmRef.current = [];
          return streamController;
        })
        .catch((streamError) => {
          if (voiceAttemptRef.current === attempt) {
            voiceStreamErrorRef.current = streamError;
          }
          pendingPcmRef.current = [];
          return null;
        });
      const recorder = await startWavRecording({
        onPcmChunk(chunk) {
          if (voiceAttemptRef.current !== attempt) return;
          if (voiceStreamRef.current) {
            try {
              voiceStreamRef.current.sendChunk(chunk);
            } catch {
              // The complete in-memory WAV remains available for safe fallback.
            }
          } else if (voiceStreamPromiseRef.current) {
            pendingPcmRef.current.push(chunk);
          }
        },
      });
      if (!holdActiveRef.current || voiceAttemptRef.current !== attempt) {
        await recorder.cancel?.();
        return;
      }
      recorderRef.current = recorder;
    } catch (recordingError) {
      cleanupVoiceResources({ resetStatus: false });
      setStatus("error");
      setError(recordingError.message || "Microphone access is unavailable.");
    }
  }

  async function finishVoiceHold() {
    if ((!holdActiveRef.current && !recorderRef.current) || voiceSubmissionRef.current) {
      return;
    }
    holdActiveRef.current = false;
    const recorder = recorderRef.current;
    recorderRef.current = null;
    if (!recorder) {
      cleanupVoiceResources();
      setNotice("Hold the mic while you speak.");
      return;
    }

    voiceSubmissionRef.current = true;
    setStatus("processing");
    setError("");
    setNotice("");
    let playbackStarted = false;
    let failed = false;
    try {
      const durationMs = performance.now() - recordingStartedAtRef.current;
      const audio = await recorder.stop();
      const validFallbackAudio = (
        durationMs >= MIN_RECORDING_MS
        && audio.size >= MIN_WAV_BYTES
        && await isValidWavBlob(audio, { minDataBytes: MIN_WAV_BYTES - 44 })
      );
      if (!validFallbackAudio) {
        voiceStreamRef.current?.cancel?.();
        setNotice("Hold the mic while you speak.");
        return;
      }
      const streamController = await voiceStreamPromiseRef.current;
      let response;
      if (streamController) {
        try {
          response = await streamController.finish();
        } catch (streamError) {
          streamController.cancel?.();
          if (NO_SPEECH_CODES.has(streamError.code)) {
            setNotice("I didn't catch that. Hold the mic and try again.");
            return;
          }
          if (confirmationId) throw streamError;
          response = await sendVoice({ audio, sessionId, confirmationId });
        }
      } else {
        const streamError = voiceStreamErrorRef.current;
        if (confirmationId) {
          throw streamError || new Error("The streaming voice connection was interrupted.");
        }
        response = await sendVoice({ audio, sessionId, confirmationId });
      }
      applySession(response);

      const transcript = response.transcript?.trim();
      if (!transcript) throw new Error("No speech was detected. Please try again.");
      const userMessage = {
        id: uniqueId(),
        role: "user",
        text: transcript,
        language: response.detected_language || messageLanguage(transcript),
        voice: true,
        detectedLanguage: response.detected_language,
      };

      let audioUrl;
      if (response.audio_base64) {
        audioUrl = audioBase64ToUrl(
          response.audio_base64,
          response.audio_mime_type || "audio/wav",
        );
        audioUrlsRef.current.add(audioUrl);
      }
      const assistant = assistantMessage(response, {
        voice: true,
        audioUrl,
        detectedLanguage: response.detected_language,
        detectedLocale: response.detected_locale,
      });
      appendTurn(userMessage, assistant);
      if (audioUrl) {
        playbackStarted = await playAudio(audioUrl, { automatic: true });
      }
    } catch (voiceError) {
      if (NO_SPEECH_CODES.has(voiceError.code)) {
        setNotice("I didn't catch that. Hold the mic and try again.");
      } else {
        failed = true;
        setError(voiceError.message || "The voice message could not be processed.");
      }
    } finally {
      cleanupVoiceResources({ resetStatus: false });
      if (failed) setStatus("error");
      else if (!playbackStarted) setStatus("idle");
      setRecordingSeconds(0);
    }
  }

  async function cancelVoiceHold() {
    if (!holdActiveRef.current && !recorderRef.current && !voiceStreamPromiseRef.current) return;
    cleanupVoiceResources();
  }

  async function playAudio(url, { automatic = false } = {}) {
    const playbackToken = playbackTokenRef.current + 1;
    playbackTokenRef.current = playbackToken;
    const playback = await playbackManagerRef.current.play(url, {
      userGesture: !automatic,
    });
    if (playbackTokenRef.current !== playbackToken) {
      playback.stop();
      return false;
    }
    if (!playback.started) {
      setStatus("idle");
      if (!automatic) setNotice("Audio playback couldn’t start. Please try again.");
      return false;
    }
    setStatus("speaking");
    void playback.ended.then(() => {
      if (playbackTokenRef.current === playbackToken) setStatus("idle");
    });
    return true;
  }

  function replayAudio(url) {
    setNotice("");
    void playAudio(url);
  }

  function newConversation() {
    cleanupVoiceResources({ resetStatus: false });
    playbackTokenRef.current += 1;
    playbackManagerRef.current.stop();
    for (const url of audioUrlsRef.current) URL.revokeObjectURL(url);
    audioUrlsRef.current.clear();
    reasonSelectionRef.current = false;
    setMessages([]);
    setSessionId("");
    setConfirmationId("");
    setInput("");
    setError("");
    setNotice("");
    setStatus("idle");
    sessionStorage.removeItem(SESSION_KEY);
    sessionStorage.removeItem(CONFIRMATION_KEY);
    sessionStorage.removeItem(MESSAGES_KEY);
  }

  return (
    <div className="app-shell">
      <Header
        onNewConversation={newConversation}
        disabled={["sending", "processing"].includes(status)}
      />
      <div className="workspace">
        <Sidebar onShortcut={(text) => submitText(text)} disabled={busy} />
        <main className="chat-panel">
          <div
            ref={chatScrollRef}
            className="chat-scroll"
            role="log"
            aria-label="Conversation history"
            aria-live="polite"
          >
            {messages.length === 0 ? (
              <Welcome onSuggestion={(text) => submitText(text)} disabled={busy} />
            ) : (
              <div className="message-list">
                {messages.map((message) => (
                  <ChatMessage
                    key={message.id}
                    message={message}
                    onReplay={replayAudio}
                    onConfirm={() => submitText("Yes", confirmationId)}
                    onCancel={() => submitText("No", confirmationId)}
                    onReasonSelect={(value) => submitReason(message.id, value)}
                    busy={busy}
                    debug={debug}
                  />
                ))}
                {(status === "sending" || status === "processing") && (
                  <div className="assistant-thinking" role="status">
                    <div className="assistant-avatar">R+</div>
                    <span /><span /><span />
                  </div>
                )}
              </div>
            )}
          </div>
          {error && (
            <div className="error-banner" role="alert">
              <span aria-hidden="true">!</span>
              {error}
              <button type="button" onClick={() => setError("")} aria-label="Dismiss error">
                ×
              </button>
            </div>
          )}
          {notice && (
            <div className="notice-banner" role="status">
              {notice}
              <button type="button" onClick={() => setNotice("")} aria-label="Dismiss notice">
                ×
              </button>
            </div>
          )}
          <Composer
            value={input}
            onChange={setInput}
            onSend={() => submitText()}
            onVoiceStart={() => void beginVoiceHold()}
            onVoiceEnd={() => void finishVoiceHold()}
            onVoiceCancel={() => void cancelVoiceHold()}
            status={status}
            recordingSeconds={recordingSeconds}
          />
        </main>
      </div>
    </div>
  );
}
