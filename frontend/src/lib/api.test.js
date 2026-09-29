import { afterEach, describe, expect, it, vi } from "vitest";
import {
  audioBase64ToUrl,
  buildVoiceStreamUrl,
  openVoiceStream,
  resolveApiBaseUrl,
  sendChat,
  sendVoice,
} from "./api";

describe("same-origin production configuration", () => {
  it("uses relative HTTP URLs in production even when a development override exists", () => {
    expect(resolveApiBaseUrl({
      configuredBase: "http://127.0.0.1:9000",
      isDev: false,
    })).toBe("");
  });

  it("keeps local development pointed at backend port 8001", () => {
    expect(resolveApiBaseUrl({
      developmentDefault: "http://127.0.0.1:8001",
      isDev: true,
    })).toBe("http://127.0.0.1:8001");
    expect(resolveApiBaseUrl({
      configuredBase: "http://127.0.0.1:9000/",
      developmentDefault: "http://127.0.0.1:8001",
      isDev: true,
    })).toBe("http://127.0.0.1:9000");
  });

  it("uses wss on an HTTPS page and the same host", () => {
    expect(buildVoiceStreamUrl({
      apiBaseUrl: "",
      pageLocation: { origin: "https://lead-uat.example" },
    })).toBe("wss://lead-uat.example/api/voice/stream");
  });

  it("uses ws on a local HTTP page", () => {
    expect(buildVoiceStreamUrl({
      apiBaseUrl: "",
      pageLocation: { origin: "http://127.0.0.1:8001" },
    })).toBe("ws://127.0.0.1:8001/api/voice/stream");
  });
});

describe("voice response audio", () => {
  afterEach(() => vi.restoreAllMocks());

  it("decodes base64 into a Blob with the returned MIME type", () => {
    let decodedBlob;
    vi.spyOn(URL, "createObjectURL").mockImplementation((blob) => {
      decodedBlob = blob;
      return "blob:decoded-voice";
    });

    const url = audioBase64ToUrl("AQIDBA==", "audio/wav");

    expect(url).toBe("blob:decoded-voice");
    expect(decodedBlob).toBeInstanceOf(Blob);
    expect(decodedBlob.type).toBe("audio/wav");
    expect(decodedBlob.size).toBe(4);
  });
});

describe("streaming voice protocol", () => {
  afterEach(() => {
    vi.restoreAllMocks();
    vi.unstubAllGlobals();
  });

  it("sends one start control, binary PCM, and one end control", async () => {
    class FakeWebSocket {
      static OPEN = 1;

      constructor(url) {
        this.url = url;
        this.readyState = FakeWebSocket.OPEN;
        this.sent = [];
        FakeWebSocket.instance = this;
        queueMicrotask(() => this.onopen?.());
      }

      send(data) {
        this.sent.push(data);
        if (typeof data !== "string") return;
        const control = JSON.parse(data);
        if (control.type === "start") {
          queueMicrotask(() => this.onmessage?.({ data: JSON.stringify({ type: "ready" }) }));
        }
        if (control.type === "end") {
          queueMicrotask(() => this.onmessage?.({
            data: JSON.stringify({
              type: "final",
              transcript: "Show my notifications",
              detected_language: "en",
            }),
          }));
        }
      }

      close() {
        this.readyState = 3;
      }
    }
    vi.stubGlobal("WebSocket", FakeWebSocket);

    const stream = await openVoiceStream({
      sessionId: "session-1",
      confirmationId: "confirmation-1",
      email: "employee@example.com",
      instance: "Universal",
      debug: true,
    });
    const pcm = new ArrayBuffer(640);
    stream.sendChunk(pcm);
    const resultPromise = stream.finish();
    const result = await resultPromise;

    const sent = FakeWebSocket.instance.sent;
    expect(JSON.parse(sent[0])).toEqual({
      type: "start",
      sample_rate: 16000,
      session_id: "session-1",
      email: "employee@example.com",
      instance: "Universal",
      confirmation_id: "confirmation-1",
      debug: true,
    });
    expect(sent[1]).toBe(pcm);
    expect(JSON.parse(sent[2])).toEqual({ type: "end" });
    expect(result.transcript).toBe("Show my notifications");
  });

  it("rejects an unexpected close after ready instead of leaving finalization pending", async () => {
    class FakeWebSocket {
      static OPEN = 1;

      constructor() {
        this.readyState = FakeWebSocket.OPEN;
        FakeWebSocket.instance = this;
        queueMicrotask(() => this.onopen?.());
      }

      send(data) {
        if (typeof data === "string" && JSON.parse(data).type === "start") {
          queueMicrotask(() => this.onmessage?.({ data: JSON.stringify({ type: "ready" }) }));
        }
      }

      close() {
        this.readyState = 3;
      }
    }
    vi.stubGlobal("WebSocket", FakeWebSocket);
    const stream = await openVoiceStream({});
    const result = stream.finish();

    FakeWebSocket.instance.onclose();

    await expect(result).rejects.toMatchObject({ code: "stream_disconnected" });
  });
});

describe("demo identity transport", () => {
  afterEach(() => vi.restoreAllMocks());

  it("sends the complete identity pair for text and voice HTTP", async () => {
    const requests = [];
    vi.spyOn(globalThis, "fetch").mockImplementation(async (url, options) => {
      requests.push({ url, options });
      return new Response(JSON.stringify({ success: true }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      });
    });

    await sendChat({
      message: "Show my profile",
      sessionId: "session-1",
      email: "employee@example.com",
      instance: "Universal",
    });
    await sendVoice({
      audio: new Blob(["wav"]),
      sessionId: "session-1",
      email: "employee@example.com",
      instance: "Universal",
    });

    expect(JSON.parse(requests[0].options.body)).toMatchObject({
      email: "employee@example.com",
      instance: "Universal",
    });
    expect(requests[1].options.body.get("email")).toBe("employee@example.com");
    expect(requests[1].options.body.get("instance")).toBe("Universal");
  });

  it.each([
    [{ email: "employee@example.com" }],
    [{ instance: "Universal" }],
  ])("rejects a partial identity before sending", async (identity) => {
    const fetchSpy = vi.spyOn(globalThis, "fetch");

    await expect(sendChat({ message: "Hello", ...identity })).rejects.toMatchObject({
      code: "invalid_identity",
    });
    expect(fetchSpy).not.toHaveBeenCalled();
  });
});

describe("safe API errors", () => {
  afterEach(() => vi.restoreAllMocks());

  it("does not expose internal voice-service details", async () => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: "internal provider stack trace" }), {
        status: 503,
        headers: { "Content-Type": "application/json" },
      }),
    );

    let caught;
    try {
      await sendVoice({ audio: new Blob(["wav"]), sessionId: "", confirmationId: "" });
    } catch (error) {
      caught = error;
    }
    expect(caught.code).toBe("voice_backend_unavailable");
    expect(caught.message).toContain("Voice is temporarily unavailable");
    expect(caught.message).not.toContain("internal provider stack trace");
  });

  it.each([
    ["no_speech", 400, "I couldn’t hear enough speech"],
    ["speech_recognition_failed", 502, "couldn’t recognize"],
    ["speech_synthesis_failed", 502, "couldn’t create a spoken reply"],
    ["resourceplus_unavailable", 502, "ResourcePlus is temporarily unavailable"],
    ["assistant_unavailable", 502, "assistant is temporarily unavailable"],
  ])("classifies %s independently", async (code, status, expectedMessage) => {
    vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response(JSON.stringify({ detail: { code, message: "internal" }, code }), {
        status,
        headers: { "Content-Type": "application/json" },
      }),
    );

    let caught;
    try {
      await sendVoice({ audio: new Blob(["wav"]), sessionId: "", confirmationId: "" });
    } catch (error) {
      caught = error;
    }
    expect(caught.code).toBe(code);
    expect(caught.message).toContain(expectedMessage);
    expect(caught.message).not.toContain("internal");
  });
});
