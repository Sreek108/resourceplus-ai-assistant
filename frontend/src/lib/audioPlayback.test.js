import { describe, expect, it, vi } from "vitest";
import { createAudioPlaybackManager } from "./audioPlayback";

function webAudioHarness() {
  const sources = [];
  const context = {
    state: "suspended",
    destination: {},
    resume: vi.fn(async () => {
      context.state = "running";
    }),
    close: vi.fn(async () => {
      context.state = "closed";
    }),
    decodeAudioData: vi.fn(async () => ({ duration: 1 })),
    createBufferSource: vi.fn(() => {
      const source = {
        buffer: null,
        connect: vi.fn(),
        start: vi.fn(),
        stop: vi.fn(),
        onended: null,
      };
      sources.push(source);
      return source;
    }),
  };
  const ContextConstructor = vi.fn(function FakeAudioContext() {
    return context;
  });
  const fetchAudio = vi.fn(async () => ({
    ok: true,
    arrayBuffer: async () => new ArrayBuffer(16),
  }));
  return { context, ContextConstructor, fetchAudio, sources };
}

describe("persistent assistant audio playback", () => {
  it("unlocks one persistent AudioContext and reuses it", async () => {
    const harness = webAudioHarness();
    const manager = createAudioPlaybackManager({
      getContextConstructor: () => harness.ContextConstructor,
      fetchAudio: harness.fetchAudio,
    });

    expect(await manager.unlock()).toBe(true);
    expect(await manager.unlock()).toBe(true);

    expect(harness.ContextConstructor).toHaveBeenCalledTimes(1);
    expect(harness.context.resume).toHaveBeenCalledTimes(1);
  });

  it("decodes and starts assistant audio through the unlocked context", async () => {
    const harness = webAudioHarness();
    const manager = createAudioPlaybackManager({
      getContextConstructor: () => harness.ContextConstructor,
      fetchAudio: harness.fetchAudio,
    });
    await manager.unlock();

    const playback = await manager.play("blob:assistant-audio");

    expect(playback.started).toBe(true);
    expect(harness.fetchAudio).toHaveBeenCalledTimes(1);
    expect(harness.context.decodeAudioData).toHaveBeenCalledTimes(1);
    expect(harness.sources[0].connect).toHaveBeenCalledWith(harness.context.destination);
    expect(harness.sources[0].start).toHaveBeenCalledTimes(1);

    harness.sources[0].onended();
    await expect(playback.ended).resolves.toBeUndefined();
  });

  it("stops active playback without closing the reusable context", async () => {
    const harness = webAudioHarness();
    const manager = createAudioPlaybackManager({
      getContextConstructor: () => harness.ContextConstructor,
      fetchAudio: harness.fetchAudio,
    });
    await manager.unlock();
    const playback = await manager.play("blob:assistant-audio");

    manager.stop();

    expect(harness.sources[0].stop).toHaveBeenCalledTimes(1);
    await expect(playback.ended).resolves.toBeUndefined();
    expect(harness.context.close).not.toHaveBeenCalled();
  });
});
