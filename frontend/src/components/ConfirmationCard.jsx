export default function ConfirmationCard({
  onConfirm,
  onCancel,
  disabled,
  block,
  direction,
  language,
}) {
  const confirmLabel = block?.actions?.find((action) => action.value === "confirm")?.label;
  const cancelLabel = block?.actions?.find((action) => action.value === "cancel")?.label;
  const isArabic = language === "ar";

  return (
    <div
      className="confirmation-card"
      role="group"
      aria-label="Action confirmation"
      dir={direction}
      lang={language}
    >
      <div className="confirmation-symbol" aria-hidden="true">!</div>
      <div className="confirmation-content">
        <strong>{block?.title || (isArabic ? "التأكيد مطلوب" : "Action requires confirmation")}</strong>
        {block?.summary ? (
          <p className="confirmation-summary">{block.summary}</p>
        ) : (
          <span>
            {isArabic
              ? "راجع ملخص المساعد أعلاه قبل المتابعة."
              : "Review the assistant summary above before continuing."}
          </span>
        )}
        <div className="confirmation-actions">
          <button type="button" onClick={onConfirm} disabled={disabled}>
            {confirmLabel || (isArabic ? "تأكيد" : "Confirm")}
          </button>
          <button type="button" onClick={onCancel} disabled={disabled}>
            {cancelLabel || (isArabic ? "إلغاء" : "Cancel")}
          </button>
        </div>
      </div>
    </div>
  );
}
