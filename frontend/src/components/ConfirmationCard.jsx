export default function ConfirmationCard({ onConfirm, onCancel, disabled }) {
  return (
    <div className="confirmation-card" role="group" aria-label="Action confirmation">
      <div className="confirmation-symbol" aria-hidden="true">!</div>
      <div className="confirmation-content">
        <strong>Action requires confirmation</strong>
        <span>Review the assistant summary above before continuing.</span>
        <div className="confirmation-actions">
          <button type="button" onClick={onConfirm} disabled={disabled}>
            Confirm
          </button>
          <button type="button" onClick={onCancel} disabled={disabled}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
}
