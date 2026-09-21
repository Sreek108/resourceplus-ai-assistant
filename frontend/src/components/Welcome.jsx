const suggestions = [
  "Show my attendance this week",
  "What is my vacation balance?",
  "Show my notifications",
  "كم رصيد إجازتي؟",
  "ورّني حضوري هذا الأسبوع",
  "ورّني تنبيهاتي",
];

export default function Welcome({ onSuggestion, disabled }) {
  return (
    <section className="welcome" aria-labelledby="welcome-title">
      <div className="welcome-orbit" aria-hidden="true">
        <div className="welcome-core">R+</div>
      </div>
      <p className="eyebrow">ResourcePlus AI Assistant</p>
      <h1 id="welcome-title">How can I help you today?</h1>
      <p className="welcome-copy">
        Ask about your profile, attendance, leave, requests, or notifications—in
        English or Arabic.
      </p>
      <div className="suggestion-grid" aria-label="Example questions">
        {/* Demo suggestions only: every chip uses the ordinary free-form chat API. */}
        {suggestions.map((suggestion) => (
          <button
            type="button"
            key={suggestion}
            onClick={() => onSuggestion(suggestion)}
            disabled={disabled}
            dir={/[\u0600-\u06ff]/.test(suggestion) ? "rtl" : "ltr"}
          >
            {suggestion}
          </button>
        ))}
      </div>
    </section>
  );
}
