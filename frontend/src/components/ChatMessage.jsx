import ReactMarkdown from "react-markdown";
import { messageDirection, messageLanguage } from "../lib/language";
import ConfirmationCard from "./ConfirmationCard";
import { SpeakerIcon } from "./Icons";
import ReasonOptions from "./ReasonOptions";

export default function ChatMessage({
  message,
  onReplay,
  onConfirm,
  onCancel,
  onReasonSelect,
  busy,
  debug,
}) {
  const language = message.language || messageLanguage(message.text);
  const direction = messageDirection(message.text);
  const assistant = message.role === "assistant";

  return (
    <article className={`message-row ${message.role}`}>
      {assistant && <div className="assistant-avatar" aria-hidden="true">R+</div>}
      <div className="message-stack">
        <div className="message-meta">
          <span>{assistant ? "ResourcePlus Assistant" : "You"}</span>
          {message.voice && <span className="voice-tag">Voice</span>}
          {message.detectedLanguage && (
            <span className="language-tag">
              {message.detectedLanguage === "ar" ? "Arabic" : "English"}
            </span>
          )}
        </div>
        <div className="message-bubble" dir={direction} lang={language}>
          {assistant ? (
            <div className="message-text markdown-content">
              <ReactMarkdown>{message.text}</ReactMarkdown>
            </div>
          ) : (
            <span className="message-text">{message.text}</span>
          )}
          {assistant && message.audioUrl && (
            <button
              type="button"
              className="replay-button"
              onClick={() => onReplay(message.audioUrl)}
              aria-label="Replay assistant voice response"
            >
              <SpeakerIcon />
            </button>
          )}
        </div>
        {assistant && message.requiresConfirmation && (
          <ConfirmationCard
            onConfirm={onConfirm}
            onCancel={onCancel}
            disabled={busy}
          />
        )}
        {assistant && message.needsReason && (
          <ReasonOptions
            options={message.reasonOptionsActive === false ? [] : message.reasonOptions}
            onSelect={onReasonSelect}
            disabled={busy || message.reasonSelectionPending}
            direction={direction}
            language={language}
            selected={message.selectedReason}
          />
        )}
        {assistant && debug && message.debug && (
          <details className="debug-panel">
            <summary>Developer details</summary>
            <dl>
              {Object.entries(message.debug).map(([key, value]) => (
                <div key={key}>
                  <dt>{key}</dt>
                  <dd>{Array.isArray(value) ? value.join(", ") : String(value ?? "—")}</dd>
                </div>
              ))}
            </dl>
          </details>
        )}
      </div>
    </article>
  );
}
