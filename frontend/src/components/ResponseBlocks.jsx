export default function ResponseBlocks({ blocks = [], onAction, actionsDisabled = false }) {
  const visible = blocks.filter((block) => block?.type !== "confirmation");
  if (!visible.length) return null;
  return (
    <div className="response-blocks">
      {visible.map((block, index) => {
        const key = `${block.type}-${block.title || index}-${index}`;
        if (block.type === "table") {
          return (
            <section className="response-block" key={key}>
              <h4>{block.title}</h4>
              <div className="response-table-wrap">
                <table>
                  <thead><tr>{block.columns?.map((column) => <th key={column.key}>{column.label}</th>)}</tr></thead>
                  <tbody>{block.rows?.map((row, rowIndex) => (
                    <tr key={rowIndex}>{block.columns?.map((column) => <td key={column.key}>{String(row[column.key] ?? "—")}</td>)}</tr>
                  ))}</tbody>
                </table>
              </div>
            </section>
          );
        }
        if (["key_value", "stat_cards"].includes(block.type)) {
          return (
            <section className={`response-block ${block.type}`} key={key}>
              <h4>{block.title}</h4>
              <dl>{block.items?.map((item) => <div key={item.label}><dt>{item.label}</dt><dd>{String(item.value ?? "—")}</dd></div>)}</dl>
            </section>
          );
        }
        if (block.type === "list") {
          return <section className="response-block" key={key}><h4>{block.title}</h4><ul>{block.items?.map((item, itemIndex) => <li key={itemIndex}>{item}</li>)}</ul></section>;
        }
        if (block.type === "notice") {
          return <section className={`response-block notice ${block.level || "info"}`} key={key}><h4>{block.title}</h4><p>{block.message}</p></section>;
        }
        if (block.type === "actions" && onAction) {
          return (
            <section className="response-block" key={key}>
              <h4>{block.title}</h4>
              <div className="response-actions" role="group" aria-label={block.title}>
                {block.actions?.map((action) => (
                  <button
                    key={action.value}
                    type="button"
                    className="reason-option"
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
