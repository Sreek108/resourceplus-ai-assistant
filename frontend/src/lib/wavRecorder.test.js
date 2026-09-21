import { describe, expect, it } from "vitest";
import { encodePcm16, encodeWav, isValidWavBlob } from "./wavRecorder";

describe("WAV encoding", () => {
  it("encodes mono PCM samples as a WAV upload", () => {
    const wav = encodeWav(new Float32Array([-1, 0, 1]), 16000);
    expect(wav.type).toBe("audio/wav");
    expect(wav.size).toBe(50);
  });

  it("validates a complete PCM WAV and rejects malformed fallback audio", async () => {
    const valid = encodeWav(new Float32Array(800).fill(0.25), 16_000);
    const malformed = new Blob([new Uint8Array(2_000)], { type: "audio/wav" });

    await expect(isValidWavBlob(valid, { minDataBytes: 1_000 })).resolves.toBe(true);
    await expect(isValidWavBlob(malformed, { minDataBytes: 1_000 })).resolves.toBe(false);
  });
});

describe("streaming PCM encoding", () => {
  it("downsamples browser audio to raw 16 kHz signed mono PCM", () => {
    const pcm = encodePcm16(
      new Float32Array([-1, -0.5, 0, 0.5, 1, 1]),
      48_000,
      16_000,
    );
    const samples = new Int16Array(pcm);

    expect(samples).toHaveLength(2);
    expect(samples[0]).toBeLessThan(0);
    expect(samples[1]).toBeGreaterThan(0);
  });
});
