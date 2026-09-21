import { afterEach, vi } from "vitest";
import { cleanup } from "@testing-library/react";

afterEach(() => {
  cleanup();
});

if (!Element.prototype.scrollIntoView) {
  Element.prototype.scrollIntoView = vi.fn();
}

Object.defineProperty(globalThis, "Audio", {
  configurable: true,
  value: vi.fn(() => {
    const audio = {
      pause: vi.fn(),
      play: vi.fn(() => {
        queueMicrotask(() => audio.onended?.());
        return Promise.resolve();
      }),
      onended: null,
      onerror: null,
    };
    return audio;
  }),
});

if (!URL.createObjectURL) URL.createObjectURL = vi.fn(() => "blob:test-audio");
if (!URL.revokeObjectURL) URL.revokeObjectURL = vi.fn();
