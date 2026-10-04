import { TEST_USERS } from "../demoUsers";

export default function TestUserPanel({ selectedUser, onChange, onNewSession }) {
  return (
    <section className="test-user-panel" aria-label="Active test identity">
      <div className="test-user-picker">
        <label htmlFor="test-user-select">Test User</label>
        <select
          id="test-user-select"
          value={selectedUser.id}
          onChange={(event) => onChange(event.target.value)}
        >
          {TEST_USERS.map((user) => (
            <option key={user.id} value={user.id}>{user.label}</option>
          ))}
        </select>
      </div>
      <dl className="test-user-details">
        <div><dt>Testing as</dt><dd>{selectedUser.name}</dd></div>
        <div><dt>Role</dt><dd>{selectedUser.role}</dd></div>
        <div><dt>Email</dt><dd>{selectedUser.email}</dd></div>
        <div><dt>Instance</dt><dd>{selectedUser.instance}</dd></div>
      </dl>
      <button type="button" className="test-user-new-session" onClick={onNewSession}>
        Start New Session
      </button>
    </section>
  );
}
