import { PlusIcon } from "./Icons";

export default function Header({ onNewConversation, disabled }) {
  return (
    <header className="topbar">
      <div className="brand-lockup">
        <div className="brand-mark" aria-hidden="true">
          <span>R</span>
          <i />
        </div>
        <div>
          <div className="brand-name">ResourcePlus</div>
          <div className="brand-subtitle">AI HR Assistant</div>
        </div>
      </div>
      <div className="topbar-actions">
        <div className="online-badge" aria-label="Assistant is online">
          <span className="online-dot" />
          Online
        </div>
        <span className="powered-label">Powered by AI</span>
        <button
          type="button"
          className="new-chat-button"
          onClick={onNewConversation}
          disabled={disabled}
          aria-label="Start a new conversation"
        >
          <PlusIcon />
          <span>New conversation</span>
        </button>
      </div>
    </header>
  );
}
