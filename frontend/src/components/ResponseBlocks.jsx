import { useId, useState } from "react";

const INITIAL_TABLE_ROWS = 10;

function TableCell({ column, value, language }) {
  if (column.key === "correctable" && typeof value === "boolean") {
    const label = language === "ar"
      ? (value ? "متاح" : "غير متاح")
      : (value ? "Available" : "Not available");
    return (
      <span className={`availability-badge ${value ? "available" : "unavailable"}`}>
        {label}
      </span>
    );
  }
  if (column.key === "status" && typeof value === "string") {
    const normalized = value.toLocaleLowerCase();
    const kind = normalized === "approved"
      ? "approved"
      : normalized === "rejected"
        ? "rejected"
        : normalized.includes("awaiting") || normalized === "pending"
          ? "pending"
          : normalized.includes("existing request")
            ? "existing"
            : "neutral";
    return <span className={`request-status-badge status-${kind}`}>{value}</span>;
  }
  return String(value ?? "—");
}

function TableResponseBlock({ block, language, onAction, actionsDisabled }) {
  const [expanded, setExpanded] = useState(false);
  const tableId = useId();
  const rows = Array.isArray(block.rows) ? block.rows : [];
  const columns = Array.isArray(block.columns) ? block.columns : [];
  const isLong = rows.length > INITIAL_TABLE_ROWS;
  const visibleRows = expanded ? rows : rows.slice(0, INITIAL_TABLE_ROWS);

  return (
    <section className="response-block table-block">
      <h4>{block.title}</h4>
      <div
        className="response-table-wrap"
        role="region"
        aria-label={language === "ar"
          ? `جدول ${block.title || "النتائج"}`
          : `${block.title || "Results"} table`}
        tabIndex={0}
      >
        <table id={tableId} data-total-rows={rows.length}>
          <thead>
            <tr>
              {columns.map((column) => (
                <th key={column.key} scope="col">{column.label}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {visibleRows.map((row, rowIndex) => (
              <tr key={rowIndex}>
                {columns.map((column) => {
                  const rowActionList = block.row_actions?.[rowIndex] || [];
                  return (
                    <td key={column.key}>
                      {column.key === "action" && rowActionList.length && onAction ? (
                        <div className="table-row-actions" role="group" aria-label={language === "ar" ? "إجراءات الطلب" : "Request actions"}>
                          {rowActionList.map((action) => (
                            <button
                              key={`${action.payload?.decision || action.value}-${action.payload?.ordinal || rowIndex}`}
                              type="button"
                              className={`row-action action-${action.style || "secondary"}`}
                              onClick={() => onAction(action)}
                              disabled={actionsDisabled}
                            >
                              {action.label}
                            </button>
                          ))}
                        </div>
                      ) : (
                        <TableCell
                          column={column}
                          value={row[column.key]}
                          language={language}
                        />
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {isLong && (
        <button
          type="button"
          className="table-expansion"
          aria-expanded={expanded}
          aria-controls={tableId}
          onClick={() => setExpanded((current) => !current)}
        >
          {language === "ar"
            ? (expanded ? "عرض أقل" : `عرض الكل (${rows.length})`)
            : (expanded ? "Show less" : `View all (${rows.length})`)}
        </button>
      )}
    </section>
  );
}

function KeyValueBlock({ block }) {
  return (
    <section className="response-block key-value-block">
      <h4>{block.title}</h4>
      <dl className="details-grid">
        {block.items?.map((item) => (
          <div key={item.label}>
            <dt>{item.label}</dt>
            <dd>{String(item.value ?? "—")}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

function StatCardsBlock({ block }) {
  return (
    <section className="response-block stat-cards-block">
      <h4>{block.title}</h4>
      <dl className="stat-card-grid">
        {block.items?.map((item) => (
          <div className="stat-card" key={item.label}>
            <dt>{item.label}</dt>
            <dd>{String(item.value ?? "—")}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

export default function ResponseBlocks({
  blocks = [],
  onAction,
  actionsDisabled = false,
  direction,
  language,
}) {
  const visible = blocks.filter((block) => block?.type !== "confirmation");
  if (!visible.length) return null;
  return (
    <div className="response-blocks" dir={direction} lang={language}>
      {visible.map((block, index) => {
        const key = `${block.type}-${block.title || index}-${index}`;
        if (block.type === "table") {
          return (
            <TableResponseBlock
              block={block}
              language={language}
              onAction={onAction}
              actionsDisabled={actionsDisabled}
              key={key}
            />
          );
        }
        if (block.type === "key_value") {
          return <KeyValueBlock block={block} key={key} />;
        }
        if (block.type === "stat_cards") {
          return <StatCardsBlock block={block} key={key} />;
        }
        if (block.type === "list") {
          return (
            <section className="response-block list-block" key={key}>
              <h4>{block.title}</h4>
              <ul>{block.items?.map((item, itemIndex) => <li key={itemIndex}>{item}</li>)}</ul>
            </section>
          );
        }
        if (block.type === "notice") {
          return (
            <section
              className={`response-block notice-block ${block.level || "info"}`}
              role={block.level === "error" ? "alert" : "status"}
              key={key}
            >
              <h4>{block.title}</h4>
              <p>{block.message}</p>
            </section>
          );
        }
        if (block.type === "actions" && onAction) {
          return (
            <section className="response-block actions-block" key={key}>
              <h4>{block.title}</h4>
              <div className="response-actions" role="group" aria-label={block.title}>
                {block.actions?.map((action) => (
                  <button
                    key={action.value}
                    type="button"
                    className={`reason-option action-${action.style || "secondary"}`}
                    onClick={() => onAction(action)}
                    disabled={actionsDisabled}
                  >
                    {action.label}
                  </button>
                ))}
              </div>
            </section>
          );
        }
        return null;
      })}
    </div>
  );
}
