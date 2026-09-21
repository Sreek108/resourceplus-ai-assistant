import { messageDirection, messageLanguage } from "../lib/language";
import { MicIcon, SendIcon } from "./Icons";

export default function Composer({
  value,
  onChange,
  onSend,
  onVoiceStart,
  onVoiceEnd,
  onVoiceCancel,
  status,
  recordingSeconds,
}) {
  const busy = ["sending", "processing", "speaking"].includes(status);
  const listening = status === "listening";
  const direction = messageDirection(value);
  const language = messageLanguage(value);
  const timer = `${String(Math.floor(recordingSeconds / 60)).padStart(2, "0")}:${String(
    recordingSeconds % 60,
  ).padStart(2, "0")}`;

  function handleKeyDown(event) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      if (!busy && !listening && value.trim()) onSend();
    }
  }

  function handlePointerDown(event) {
    if (event.pointerType === "mouse" && event.button !== 0) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture?.(event.pointerId);
    onVoiceStart();
  }

  function handlePointerUp(event) {
    event.preventDefault();
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    onVoiceEnd();
  }

  function handlePointerLeave(event) {
    // Pointer capture normally delivers pointerup. This recovers a release that
    // occurred outside the button on browsers without reliable capture support.
    if (listening && event.buttons === 0) onVoiceEnd();
  }

  function handlePointerCancel(event) {
    event.preventDefault();
    if (event.currentTarget.hasPointerCapture?.(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    onVoiceCancel();
  }

  function handleKeyDownOnMic(event) {
    if (["Enter", " "].includes(event.key) && !event.repeat) {
      event.preventDefault();
      onVoiceStart();
    }
  }

  function handleKeyUpOnMic(event) {
    if (["Enter", " "].includes(event.key)) {
      event.preventDefault();
      onVoiceEnd();
    }
  }

  return (
    <div className="composer-wrap">
      {listening && (
        <div className="recording-banner" role="status" aria-live="polite">
          <span className="recording-pulse" />
          Listening… {timer}
          <span>Release to send</span>
        </div>
      )}
      {status === "processing" && (
        <div className="processing-banner" role="status" aria-live="polite">
          <span className="processing-spinner" /> Understanding…
        </div>
      )}
      {status === "speaking" && (
        <div className="processing-banner speaking-banner" role="status" aria-live="polite">
          <span className="speaking-pulse" /> Speaking…
        </div>
      )}
      <div className={`composer ${listening ? "is-recording" : ""}`}>
        <textarea
          value={value}
          onChange={(event) => onChange(event.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask ResourcePlus…"
          rows={1}
          dir={direction}
          lang={language}
          disabled={busy || listening}
          aria-label="Message ResourcePlus"
        />
        <button
          type="button"
          className={`mic-button ${listening ? "recording" : ""}`}
          onPointerDown={handlePointerDown}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerCancel}
          onPointerLeave={handlePointerLeave}
          onKeyDown={handleKeyDownOnMic}
          onKeyUp={handleKeyUpOnMic}
          onContextMenu={(event) => event.preventDefault()}
          disabled={busy}
          aria-label={listening ? "Release to send voice message" : "Hold to talk"}
        >
          <MicIcon recording={listening} />
        </button>
        <button
          type="button"
          className="send-button"
          onClick={onSend}
          disabled={busy || listening || !value.trim()}
          aria-label="Send message"
        >
          <SendIcon />
        </button>
      </div>
      <p className="composer-hint">Enter to send · Shift + Enter for a new line</p>
    </div>
  );
}
