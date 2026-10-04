import ReactMarkdown from "react-markdown";
import { messageDirection, messageLanguage } from "../lib/language";
import ConfirmationCard from "./ConfirmationCard";
import { SpeakerIcon } from "./Icons";
import ReasonOptions from "./ReasonOptions";
import ResponseBlocks from "./ResponseBlocks";

export default function ChatMessage({
  message,
  onReplay,
  onConfirm,
  onCancel,
  onReasonSelect,
  onActionSelect,
  busy,
  debug,
}) {
  const language = message.language || messageLanguage(message.text);
  const direction = messageDirection(message.text);
  const assistant = message.role === "assistant";
  const structuredBlocks = Array.isArray(message.blocks) ? message.blocks : [];
  const confirmationBlock = structuredBlocks.find((block) => block?.type === "confirmation");
  const hasStructuredContent = assistant && structuredBlocks.length > 0;

  return (
    <article className={`message-row ${message.role}${hasStructuredContent ? " has-structured-content" : ""}`}>
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
          {assistant && (
            <button
              type="button"
              className="replay-button"
              onClick={() => onReplay(message)}
              aria-label={
                message.audioUrl
                  ? "Replay assistant voice response"
                  : "Read assistant message aloud"
              }
            >
              <SpeakerIcon />
            </button>
          )}
        </div>
        {assistant && (
          <ResponseBlocks
            blocks={message.needsReason ? message.blocks?.filter((block) => block.type !== "actions") : message.blocks}
            onAction={onActionSelect}
            actionsDisabled={busy || message.actionsActive === false}
            direction={direction}
            language={language}
          />
        )}
        {assistant && message.requiresConfirmation && (
          <ConfirmationCard
            onConfirm={onConfirm}
            onCancel={onCancel}
            disabled={busy}
            block={confirmationBlock}
            direction={direction}
            language={language}
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
