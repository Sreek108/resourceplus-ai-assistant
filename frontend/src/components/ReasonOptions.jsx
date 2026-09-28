export default function ReasonOptions({
  options,
  onSelect,
  disabled,
  direction,
  language,
  selected,
}) {
  if (selected) {
    return (
      <div
        className="reason-selected"
        role="status"
        dir={direction}
        lang={language}
      >
        <span aria-hidden="true">✓</span>
        <span>
          {language === "ar" ? "السبب المختار" : "Selected reason"}: {selected}
        </span>
      </div>
    );
  }

  if (!options?.length) return null;

  return (
    <div
      className="reason-options"
      role="group"
      aria-label="Select an exceptional-entry reason"
      dir={direction}
      lang={language}
    >
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          className="reason-option"
          onClick={() => onSelect(option.value)}
          disabled={disabled}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
