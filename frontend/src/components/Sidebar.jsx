const capabilities = [
  ["profile", "My Profile", "Show my employee profile"],
  ["attendance", "Attendance", "Show my attendance this week"],
  ["leave", "Leave & Travel", "What leave and travel options are available?"],
  ["requests", "Requests", "Show all my requests this month"],
  ["notifications", "Notifications", "Show my notifications"],
  ["manager", "Manager Approvals", "Show my pending team approvals"],
];

const glyphs = {
  profile: "PR",
  attendance: "AT",
  leave: "LT",
  requests: "RQ",
  notifications: "NT",
  manager: "MA",
};

export default function Sidebar({ onShortcut, disabled }) {
  return (
    <aside className="sidebar" aria-label="Demo capabilities">
      <div className="sidebar-heading">
        <span>Explore</span>
        <small>Demo shortcuts</small>
      </div>
      <nav className="capability-list">
        {capabilities.map(([key, label, prompt]) => (
          <button
            key={key}
            type="button"
            onClick={() => onShortcut(prompt)}
            disabled={disabled}
            aria-label={`Ask about ${label}`}
          >
            <span className="shortcut-glyph" aria-hidden="true">
              {glyphs[key]}
            </span>
            <span>{label}</span>
          </button>
        ))}
      </nav>
      <div className="sidebar-note">
        <span className="sidebar-note-icon">✦</span>
        <div>
          <strong>Ask naturally</strong>
          <p>These shortcuts are suggestions. You can ask any HR question.</p>
        </div>
      </div>
    </aside>
  );
}
