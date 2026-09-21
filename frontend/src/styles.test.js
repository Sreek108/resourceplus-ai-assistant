import { describe, expect, it } from "vitest";
import { readFileSync } from "node:fs";
import { join } from "node:path";

const styles = readFileSync(join(process.cwd(), "src", "styles.css"), "utf8");

function declarations(selector) {
  const escaped = selector.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const match = styles.match(new RegExp(`${escaped}\\s*\\{([^}]*)\\}`, "s"));
  expect(match, `Missing CSS rule for ${selector}`).toBeTruthy();
  return match[1];
}

describe("viewport-bound chat layout", () => {
  it("uses a dynamic viewport shell and prevents document scrolling", () => {
    expect(declarations("body")).toMatch(/overflow:\s*hidden/);
    expect(declarations(".app-shell")).toMatch(/height:\s*100dvh/);
    expect(declarations(".app-shell")).toMatch(/overflow:\s*hidden/);
  });

  it("allows nested workspace and chat flex children to shrink", () => {
    expect(declarations(".workspace")).toMatch(/min-height:\s*0/);
    expect(declarations(".chat-panel")).toMatch(/min-height:\s*0/);
    expect(declarations(".chat-panel")).toMatch(/flex-direction:\s*column/);
  });

  it("makes only the message history the normal conversation scroll region", () => {
    expect(declarations(".chat-scroll")).toMatch(/flex:\s*1 1 auto/);
    expect(declarations(".chat-scroll")).toMatch(/min-height:\s*0/);
    expect(declarations(".chat-scroll")).toMatch(/overflow-y:\s*auto/);
  });

  it("keeps the composer non-shrinking and mobile safe-area aware", () => {
    expect(declarations(".composer-wrap")).toMatch(/flex:\s*0 0 auto/);
    expect(declarations(".composer-wrap")).toMatch(/env\(safe-area-inset-bottom\)/);
  });
});
