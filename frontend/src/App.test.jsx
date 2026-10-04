import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import App from "./App";
import ChatMessage from "./components/ChatMessage";
import {
  audioBase64ToUrl,
  openVoiceStream,
  sendChat,
  sendVoice,
  synthesizeVoice,
} from "./lib/api";
import { isValidWavBlob, startWavRecording } from "./lib/wavRecorder";

vi.mock("./lib/api", () => ({
  sendChat: vi.fn(),
  sendVoice: vi.fn(),
  openVoiceStream: vi.fn(),
  audioBase64ToUrl: vi.fn(() => "blob:assistant-audio"),
  synthesizeVoice: vi.fn(),
}));

vi.mock("./lib/wavRecorder", () => ({
  startWavRecording: vi.fn(),
  isValidWavBlob: vi.fn(),
}));

function response(overrides = {}) {
  return {
    success: true,
    message: "Here is your information.",
    language: "en",
    tools_used: [],
    session_id: "session-new",
    requires_confirmation: false,
    confirmation_id: null,
    ...overrides,
  };
}

async function sendTypedMessage(text = "Hello") {
  fireEvent.change(screen.getByLabelText("Message ResourcePlus"), {
    target: { value: text },
  });
  fireEvent.click(screen.getByLabelText("Send message"));
  await waitFor(() => expect(sendChat).toHaveBeenCalled());
}

async function holdAndRelease(durationMs = 800) {
  const now = vi.spyOn(performance, "now").mockReturnValue(1_000);
  fireEvent.pointerDown(screen.getByLabelText("Hold to talk"), {
    pointerId: 1,
    pointerType: "mouse",
    button: 0,
    buttons: 1,
  });
  await waitFor(() => expect(startWavRecording).toHaveBeenCalled());
  await act(async () => {});
  now.mockReturnValue(1_000 + durationMs);
  fireEvent.pointerUp(screen.getByLabelText("Release to send voice message"), {
    pointerId: 1,
    pointerType: "mouse",
    button: 0,
    buttons: 0,
  });
  now.mockRestore();
}

describe("ResourcePlus demo UI", () => {
  beforeEach(() => {
    delete globalThis.AudioContext;
    delete globalThis.webkitAudioContext;
    sessionStorage.clear();
    sessionStorage.setItem("resourceplus.demo.test-user", "talal");
    window.history.replaceState({}, "", "/");
    vi.clearAllMocks();
    isValidWavBlob.mockResolvedValue(true);
    sendChat.mockResolvedValue(response());
    synthesizeVoice.mockResolvedValue({
      audio_base64: "UklGRg==",
      audio_mime_type: "audio/wav",
      language: "en",
      tts_locale: "en-US",
      tts_voice: "en-US-AvaNeural",
    });
    openVoiceStream.mockRejectedValue(new Error("stream unavailable"));
    sendVoice.mockResolvedValue(
      response({
        transcript: "Show my attendance",
        detected_language: "en",
        detected_locale: "en-US",
        audio_base64: "UklGRg==",
        audio_mime_type: "audio/wav",
      }),
    );
    startWavRecording.mockResolvedValue({
      stop: vi.fn().mockResolvedValue(
        new Blob([new Uint8Array(2_000)], { type: "audio/wav" }),
      ),
      cancel: vi.fn().mockResolvedValue(undefined),
    });
  });

  it("renders English messages LTR and Arabic messages RTL", () => {
    const { rerender } = render(
      <ChatMessage message={{ role: "user", text: "Show my profile" }} />,
    );
    expect(screen.getByText("Show my profile").parentElement.dir).toBe("ltr");
    expect(screen.getByText("Show my profile").parentElement.lang).toBe("en");

    rerender(<ChatMessage message={{ role: "user", text: "ورّني تنبيهاتي" }} />);
    expect(screen.getByText("ورّني تنبيهاتي").parentElement.dir).toBe("rtl");
    expect(screen.getByText("ورّني تنبيهاتي").parentElement.lang).toBe("ar");
  });

  it("keeps the conversation in a dedicated scroll region with the composer mounted", () => {
    sessionStorage.setItem(
      "resourceplus.demo.messages",
      JSON.stringify(
        Array.from({ length: 40 }, (_, index) => ({
          id: `message-${index}`,
          role: index % 2 ? "assistant" : "user",
          text: `Conversation message ${index}`,
          language: "en",
        })),
      ),
    );

    render(<App />);

    const history = screen.getByRole("log", { name: "Conversation history" });
    const composer = screen.getByLabelText("Message ResourcePlus");
    expect(history.classList.contains("chat-scroll")).toBe(true);
    expect(screen.getByText("Conversation message 39")).toBeTruthy();
    expect(composer.closest(".composer-wrap")).toBeTruthy();
    expect(composer.closest(".chat-panel")).toBe(history.parentElement);
  });

  it("scrolls the chat region when user and assistant messages are added", async () => {
    render(<App />);
    const history = screen.getByRole("log", { name: "Conversation history" });
    Object.defineProperty(history, "scrollHeight", { configurable: true, value: 1_600 });
    history.scrollTo = vi.fn();

    await sendTypedMessage("Show my profile");

    await waitFor(() => expect(history.scrollTo).toHaveBeenCalled());
    expect(history.scrollTo).toHaveBeenLastCalledWith({
      top: 1_600,
      behavior: "smooth",
    });
  });

  it("keeps the composer mounted beside a long assistant Markdown response", async () => {
    const longMarkdown = Array.from(
      { length: 60 },
      (_, index) => `- **Attendance day ${index + 1}:** Present`,
    ).join("\n");
    sendChat.mockResolvedValueOnce(response({ message: `### Attendance\n\n${longMarkdown}` }));
    render(<App />);
    const composer = screen.getByLabelText("Message ResourcePlus");

    await sendTypedMessage("Show my complete attendance history");

    expect(await screen.findByRole("heading", { name: "Attendance" })).toBeTruthy();
    expect(screen.getByLabelText("Message ResourcePlus")).toBe(composer);
    expect(
      screen.getByRole("log").contains(screen.getByText("Attendance day 60:")),
    ).toBe(true);
  });

  it("renders assistant headings, emphasis, and lists as Markdown", () => {
    render(
      <ChatMessage
        message={{
          role: "assistant",
          text: "### Contact Information\n\n- **Employee Number:** UN003\n- *Name:* Saneesh",
        }}
      />,
    );

    expect(screen.getByRole("heading", { name: "Contact Information" })).toBeTruthy();
    expect(screen.getByText("Employee Number:").tagName).toBe("STRONG");
    expect(screen.getByText("Name:").tagName).toBe("EM");
    expect(screen.getByRole("list").children).toHaveLength(2);
    expect(document.body.textContent).not.toContain("###");
    expect(document.body.textContent).not.toContain("**");
  });

  it("sends an explicit correction request from a less-hours action", () => {
    const onActionSelect = vi.fn();
    render(
      <ChatMessage
        message={{
          role: "assistant",
          text: "You have one less-hours entry eligible for correction.",
          blocks: [
            {
              type: "table",
              title: "Attendance gaps",
              columns: [
                { key: "date", label: "Date" },
                { key: "action", label: "Action" },
              ],
              rows: [{ date: "2026-09-10", action: "Correction candidate" }],
            },
            {
              type: "actions",
              title: "Available actions",
              actions: [{
                label: "Correct 2026-09-10",
                value: "Correct my less hours on 2026-09-10",
                style: "primary",
              }],
            },
          ],
        }}
        onActionSelect={onActionSelect}
        busy={false}
      />,
    );

    fireEvent.click(screen.getByRole("button", { name: "Correct 2026-09-10" }));
    expect(onActionSelect).toHaveBeenCalledWith({
      label: "Correct 2026-09-10",
      value: "Correct my less hours on 2026-09-10",
      style: "primary",
    });
    expect(screen.getByText("Correction candidate")).toBeTruthy();
    expect(document.body.textContent).not.toContain("Eligible");
  });

  it("shows an action label while sending its opaque candidate value", async () => {
    const label = "22 Sep · Family Circumstances · Early Departure";
    const opaqueValue = "Select exceptional entry ce-6b829709ddeb";
    sendChat
      .mockResolvedValueOnce(response({
        message: "Choose a cancellable request.",
        blocks: [{
          type: "actions",
          title: "Select an entry",
          actions: [{ label, value: opaqueValue, style: "secondary" }],
        }],
      }))
      .mockResolvedValueOnce(response({
        message: "Confirm cancellation.",
        requires_confirmation: true,
        confirmation_id: "confirmation-one",
      }))
      .mockResolvedValueOnce(response({ message: "Cancelled." }));

    render(<App />);
    await sendTypedMessage("Cancel my pending exceptional entries");
    const candidateButton = await screen.findByRole("button", { name: label });
    fireEvent.click(candidateButton);

    await waitFor(() => expect(sendChat).toHaveBeenCalledTimes(2));
    expect(sendChat.mock.calls[1][0].message).toBe(opaqueValue);
    expect(screen.getAllByText(label).length).toBeGreaterThan(1);
    expect(document.body.textContent).not.toContain("ce-6b829709ddeb");
    expect(candidateButton.disabled).toBe(true);

    fireEvent.click(await screen.findByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(sendChat).toHaveBeenCalledTimes(3));
    expect(candidateButton.disabled).toBe(true);
  });

  it("sends a structured per-request selector and disables stale row actions", async () => {
    const selection = { kind: "pending_approval", decision: "approve", ordinal: 1 };
    sendChat
      .mockResolvedValueOnce(response({
        message: "Here are the pending requests.",
        blocks: [{
          type: "table",
          title: "Pending approvals",
          columns: [
            { key: "employee", label: "Employee" },
            { key: "request", label: "Request" },
            { key: "date", label: "Date" },
            { key: "detail", label: "Detail" },
            { key: "status", label: "Status" },
            { key: "action", label: "Action" },
          ],
          rows: [{
            employee: "Talal Sabbagh",
            request: "Leave",
            date: "Oct 9",
            detail: "Compensatory Leave",
            status: "Pending",
            action: "",
          }],
          row_actions: [[{
            label: "Approve",
            value: "Approve request 1",
            style: "primary",
            payload: selection,
          }, {
            label: "Reject",
            value: "Reject request 1",
            style: "danger",
            payload: { ...selection, decision: "reject" },
          }]],
        }],
      }))
      .mockResolvedValueOnce(response({
        message: "Approve Talal Sabbagh's Compensatory Leave request for Oct 9?",
        requires_confirmation: true,
        confirmation_id: "confirmation-one",
      }));

    render(<App />);
    await sendTypedMessage("Show pending approvals");
    const approveButton = await screen.findByRole("button", { name: "Approve" });
    fireEvent.click(approveButton);

    await waitFor(() => expect(sendChat).toHaveBeenCalledTimes(2));
    expect(sendChat.mock.calls[1][0]).toMatchObject({
      message: "Approve request 1",
      approvalSelection: selection,
      sessionId: "session-new",
    });
    expect(approveButton.disabled).toBe(true);
    expect(document.body.textContent).not.toContain("requestId");
  });

  it("keeps Arabic assistant Markdown RTL", () => {
    render(
      <ChatMessage
        message={{
          role: "assistant",
          text: "### معلومات الموظف\n\n- **الاسم:** سنيش\n- **الوظيفة:** مهندس",
        }}
      />,
    );

    const heading = screen.getByRole("heading", { name: "معلومات الموظف" });
    const bubble = heading.closest(".message-bubble");
    expect(bubble.dir).toBe("rtl");
    expect(bubble.lang).toBe("ar");
    expect(screen.getByRole("list").children).toHaveLength(2);
  });

  it("keeps Markdown-looking user text literal", () => {
    const { container } = render(
      <ChatMessage message={{ role: "user", text: "### My note\n- **keep this literal**" }} />,
    );

    expect(screen.queryByRole("heading")).toBeNull();
    expect(screen.queryByRole("list")).toBeNull();
    expect(container.querySelector(".message-text").textContent).toBe(
      "### My note\n- **keep this literal**",
    );
  });

  it("sends the current session ID with an ordinary chat message", async () => {
    sessionStorage.setItem("resourceplus.demo.session", "session-existing");
    render(<App />);
    await sendTypedMessage();

    expect(sendChat).toHaveBeenCalledWith({
      message: "Hello",
      sessionId: "session-existing",
      email: "talal.sabbagh@example.com",
      instance: "portalv21",
      confirmationId: "",
    });
  });

  it("does not automatically play a text-originated response", async () => {
    render(<App />);

    await sendTypedMessage("Show my notifications");

    expect(await screen.findByText("Here is your information.")).toBeTruthy();
    expect(globalThis.Audio).not.toHaveBeenCalled();
  });

  it("reads an English text response aloud and caches it for replay", async () => {
    sendChat.mockResolvedValueOnce(response({
      message: "Your profile is ready.",
      language: "en",
    }));
    render(<App />);
    await sendTypedMessage("Show my profile");

    fireEvent.click(await screen.findByLabelText("Read assistant message aloud"));

    await waitFor(() => expect(synthesizeVoice).toHaveBeenCalledWith({
      text: "Your profile is ready.",
      language: "en",
    }));
    expect(audioBase64ToUrl).toHaveBeenCalledWith("UklGRg==", "audio/wav");
    await waitFor(() => expect(globalThis.Audio).toHaveBeenCalledTimes(1));

    fireEvent.click(await screen.findByLabelText("Replay assistant voice response"));
    await waitFor(() => expect(globalThis.Audio).toHaveBeenCalledTimes(2));
    expect(synthesizeVoice).toHaveBeenCalledTimes(1);
  });

  it("uses Arabic synthesis metadata for an Arabic text response", async () => {
    const arabicMessage = "طلبك لا يزال قيد الانتظار.";
    sendChat.mockResolvedValueOnce(response({
      message: arabicMessage,
      language: "ar",
    }));
    synthesizeVoice.mockResolvedValueOnce({
      audio_base64: "UklGRg==",
      audio_mime_type: "audio/wav",
      language: "ar",
      tts_locale: "ar-SA",
      tts_voice: "ar-SA-ZariyahNeural",
    });
    render(<App />);
    await sendTypedMessage("اعرض طلباتي");

    fireEvent.click(await screen.findByLabelText("Read assistant message aloud"));

    await waitFor(() => expect(synthesizeVoice).toHaveBeenCalledWith({
      text: arabicMessage,
      language: "ar",
    }));
    expect(audioBase64ToUrl).toHaveBeenCalledWith("UklGRg==", "audio/wav");
    await waitFor(() => expect(globalThis.Audio).toHaveBeenCalledTimes(1));
  });

  it("keeps text usable and offers retry when message synthesis fails", async () => {
    synthesizeVoice.mockRejectedValueOnce(new Error("Voice audio is unavailable right now."));
    render(<App />);
    await sendTypedMessage("Show my profile");

    fireEvent.click(await screen.findByLabelText("Read assistant message aloud"));

    expect(await screen.findByText("Voice audio is unavailable right now.")).toBeTruthy();
    expect(screen.getByText("Here is your information.")).toBeTruthy();
    expect(screen.getByLabelText("Read assistant message aloud")).toBeTruthy();
    expect(globalThis.Audio).not.toHaveBeenCalled();
  });

  it("does not crash or play when message synthesis returns empty audio", async () => {
    synthesizeVoice.mockResolvedValueOnce({
      audio_base64: "",
      audio_mime_type: "audio/wav",
      language: "en",
      tts_locale: "en-US",
      tts_voice: "en-US-AvaNeural",
    });
    render(<App />);
    await sendTypedMessage("Show my profile");

    fireEvent.click(await screen.findByLabelText("Read assistant message aloud"));

    expect(await screen.findByText("Voice audio is unavailable for this message.")).toBeTruthy();
    expect(screen.getByText("Here is your information.")).toBeTruthy();
    expect(audioBase64ToUrl).not.toHaveBeenCalled();
    expect(globalThis.Audio).not.toHaveBeenCalled();
  });

  it("does not play a stale synthesis result after starting a new conversation", async () => {
    let resolveSynthesis;
    synthesizeVoice.mockReturnValueOnce(new Promise((resolve) => {
      resolveSynthesis = resolve;
    }));
    render(<App />);
    await sendTypedMessage("Show my profile");
    fireEvent.click(await screen.findByLabelText("Read assistant message aloud"));
    await waitFor(() => expect(synthesizeVoice).toHaveBeenCalledTimes(1));

    fireEvent.click(screen.getByLabelText("Start a new conversation"));
    await act(async () => {
      resolveSynthesis({
        audio_base64: "UklGRg==",
        audio_mime_type: "audio/wav",
        language: "en",
        tts_locale: "en-US",
        tts_voice: "en-US-AvaNeural",
      });
    });

    expect(audioBase64ToUrl).not.toHaveBeenCalled();
    expect(globalThis.Audio).not.toHaveBeenCalled();
  });

  it("stops one message before reading a different message", async () => {
    sendChat
      .mockResolvedValueOnce(response({ message: "First response." }))
      .mockResolvedValueOnce(response({ message: "Second response." }));
    let firstAudio;
    globalThis.Audio.mockImplementationOnce(function ActiveAudio() {
      firstAudio = {
        pause: vi.fn(),
        play: vi.fn(() => Promise.resolve()),
        onended: null,
        onerror: null,
      };
      return firstAudio;
    });
    render(<App />);
    await sendTypedMessage("First question");
    await sendTypedMessage("Second question");
    const readButtons = await screen.findAllByLabelText("Read assistant message aloud");

    fireEvent.click(readButtons[0]);
    await waitFor(() => expect(globalThis.Audio).toHaveBeenCalledTimes(1));
    fireEvent.click(readButtons[1]);

    await waitFor(() => expect(firstAudio.pause).toHaveBeenCalled());
    await waitFor(() => expect(globalThis.Audio).toHaveBeenCalledTimes(2));
    expect(synthesizeVoice).toHaveBeenCalledTimes(2);
  });

  it("unlocks a persistent AudioContext from the microphone gesture", async () => {
    const context = {
      state: "suspended",
      resume: vi.fn(async () => {
        context.state = "running";
      }),
      close: vi.fn(async () => {
        context.state = "closed";
      }),
    };
    const AudioContext = vi.fn(function FakeAudioContext() {
      return context;
    });
    Object.defineProperty(globalThis, "AudioContext", {
      configurable: true,
      value: AudioContext,
    });
    render(<App />);

    fireEvent.pointerDown(screen.getByLabelText("Hold to talk"), {
      pointerId: 11,
      pointerType: "mouse",
      button: 0,
      buttons: 1,
    });

    await waitFor(() => expect(context.resume).toHaveBeenCalledTimes(1));
    expect(AudioContext).toHaveBeenCalledTimes(1);
    fireEvent.pointerCancel(screen.getByLabelText("Release to send voice message"), {
      pointerId: 11,
      pointerType: "mouse",
    });
  });

  it("starts listening on pointerdown and sends on pointerup", async () => {
    sendVoice.mockResolvedValue(
      response({
        transcript: "أرني تنبيهاتي",
        detected_language: "ar",
        detected_locale: "ar-SA",
        language: "ar",
        message: "هذه هي تنبيهاتك.",
        audio_base64: "UklGRg==",
        audio_mime_type: "audio/wav",
      }),
    );
    render(<App />);

    const now = vi.spyOn(performance, "now").mockReturnValue(1_000);
    fireEvent.pointerDown(screen.getByLabelText("Hold to talk"), {
      pointerId: 7,
      pointerType: "mouse",
      button: 0,
      buttons: 1,
    });
    expect(await screen.findByText(/Listening…/)).toBeTruthy();
    expect(startWavRecording).toHaveBeenCalledTimes(1);
    expect(sendVoice).not.toHaveBeenCalled();
    await act(async () => {});

    now.mockReturnValue(1_800);
    fireEvent.pointerUp(screen.getByLabelText("Release to send voice message"), {
      pointerId: 7,
      pointerType: "mouse",
      buttons: 0,
    });
    now.mockRestore();

    await waitFor(() => expect(screen.getByText("أرني تنبيهاتي")).toBeTruthy());
    expect(sendVoice).toHaveBeenCalledWith(
      expect.objectContaining({
        sessionId: expect.any(String),
        confirmationId: "",
        email: "talal.sabbagh@example.com",
        instance: "portalv21",
      }),
    );
    expect(audioBase64ToUrl).toHaveBeenCalledWith("UklGRg==", "audio/wav");
    expect(globalThis.Audio).toHaveBeenCalledWith("blob:assistant-audio");
    expect(screen.getByLabelText("Replay assistant voice response")).toBeTruthy();
    await waitFor(() => expect(screen.getByLabelText("Hold to talk")).toBeTruthy());
    expect(screen.queryByText(/Listening…/)).toBeNull();
  });

  it("auto-scrolls the chat region after a voice transcript is inserted", async () => {
    render(<App />);
    const history = screen.getByRole("log", { name: "Conversation history" });
    Object.defineProperty(history, "scrollHeight", { configurable: true, value: 2_200 });
    history.scrollTo = vi.fn();

    await holdAndRelease();

    expect(await screen.findByText("Show my attendance")).toBeTruthy();
    await waitFor(() => expect(history.scrollTo).toHaveBeenCalled());
    expect(history.scrollTo).toHaveBeenLastCalledWith({
      top: 2_200,
      behavior: "smooth",
    });
  });

  it("manually replays retained voice audio from the speaker button", async () => {
    render(<App />);
    await holdAndRelease();

    const replay = await screen.findByLabelText("Replay assistant voice response");
    await waitFor(() => expect(globalThis.Audio).toHaveBeenCalledTimes(1));
    fireEvent.click(replay);

    await waitFor(() => expect(globalThis.Audio).toHaveBeenCalledTimes(2));
  });

  it("enters speaking only after playback starts, clears on end, and auto-plays once", async () => {
    let activeAudio;
    globalThis.Audio.mockImplementationOnce(function ActiveAudio() {
      activeAudio = {
        pause: vi.fn(),
        play: vi.fn(() => Promise.resolve()),
        onended: null,
        onerror: null,
      };
      return activeAudio;
    });
    render(<App />);

    await holdAndRelease();

    expect(await screen.findByText(/Speaking/)).toBeTruthy();
    expect(activeAudio.play).toHaveBeenCalledTimes(1);
    expect(globalThis.Audio).toHaveBeenCalledTimes(1);
    act(() => activeAudio.onended());
    await waitFor(() => expect(screen.queryByText(/Speaking/)).toBeNull());
    expect(screen.getByLabelText("Hold to talk")).toBeTruthy();
  });

  it("streams PCM while held and uses the WebSocket final response once", async () => {
    const streamController = {
      sendChunk: vi.fn(),
      finish: vi.fn().mockResolvedValue(
        response({
          transcript: "Show my notifications",
          detected_language: "en",
          detected_locale: "en-US",
          audio_base64: "UklGRg==",
          audio_mime_type: "audio/wav",
        }),
      ),
      cancel: vi.fn(),
    };
    openVoiceStream.mockResolvedValueOnce(streamController);
    startWavRecording.mockImplementationOnce(async ({ onPcmChunk }) => {
      onPcmChunk(new ArrayBuffer(640));
      return {
        stop: vi.fn().mockResolvedValue(
          new Blob([new Uint8Array(2_000)], { type: "audio/wav" }),
        ),
        cancel: vi.fn(),
      };
    });
    render(<App />);

    await holdAndRelease();

    await waitFor(() => expect(streamController.finish).toHaveBeenCalledTimes(1));
    expect(streamController.sendChunk).toHaveBeenCalledTimes(1);
    expect(sendVoice).not.toHaveBeenCalled();
    expect(await screen.findByText("Show my notifications")).toBeTruthy();
  });

  it("falls back to the retained WAV after a streaming failure", async () => {
    const streamController = {
      sendChunk: vi.fn(),
      finish: vi.fn().mockRejectedValue(new Error("stream failed")),
      cancel: vi.fn(),
    };
    openVoiceStream.mockResolvedValueOnce(streamController);
    render(<App />);

    await holdAndRelease();

    await waitFor(() => expect(sendVoice).toHaveBeenCalledTimes(1));
  });

  it("renders a completed HTTP voice result when only TTS failed", async () => {
    const transcript = "Only correct 10 minutes of my late arrival on 10 September";
    sendVoice.mockResolvedValueOnce(
      response({
        message: "Confirm the 10-minute late-arrival correction.",
        transcript,
        detected_language: "en",
        detected_locale: "en-US",
        session_id: "tts-confirmation-session",
        requires_confirmation: true,
        confirmation_id: "tts-confirmation-id",
        audio_base64: "",
        tts_unavailable: true,
        blocks: [
          {
            type: "table",
            title: "Correction details",
            columns: [{ key: "date", label: "Date" }],
            rows: [{ date: "2026-09-10" }],
          },
          {
            type: "key_value",
            title: "Request scope",
            items: [{ label: "Minutes", value: 10 }],
          },
          {
            type: "stat_cards",
            title: "Allowance",
            items: [{ label: "Remaining", value: 90 }],
          },
          {
            type: "actions",
            title: "Available actions",
            actions: [{ label: "Review correction", value: "review" }],
          },
          {
            type: "confirmation",
            title: "Confirmation required",
            summary: "Confirm the correction.",
          },
          {
            type: "notice",
            title: "ResourcePlus",
            message: "No request has been submitted yet.",
            level: "info",
          },
        ],
      }),
    );
    render(<App />);

    await holdAndRelease();

    expect(await screen.findByText(transcript)).toBeTruthy();
    expect(screen.getByText("Confirm the 10-minute late-arrival correction.")).toBeTruthy();
    expect(screen.getByText("Correction details")).toBeTruthy();
    expect(screen.getByText("Request scope")).toBeTruthy();
    expect(screen.getByText("Allowance")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Review correction" })).toBeTruthy();
    expect(screen.getByRole("heading", { name: "ResourcePlus" })).toBeTruthy();
    expect(screen.getByText("Confirmation required")).toBeTruthy();
    expect(screen.getByText("Confirm the correction.")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Confirm" })).toBeTruthy();
    expect(await screen.findByText(
      "Voice audio is unavailable right now. The response is shown as text.",
    )).toBeTruthy();
    await waitFor(() => {
      expect(sessionStorage.getItem("resourceplus.demo.session")).toBe(
        "tts-confirmation-session",
      );
      expect(sessionStorage.getItem("resourceplus.demo.confirmation")).toBe(
        "tts-confirmation-id",
      );
    });
    expect(sendVoice).toHaveBeenCalledTimes(1);
    expect(audioBase64ToUrl).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole("button", { name: "Cancel" }));
    await waitFor(() => expect(sendChat).toHaveBeenCalledTimes(1));
    expect(sendChat).toHaveBeenLastCalledWith({
      message: "No",
      sessionId: "tts-confirmation-session",
      email: "talal.sabbagh@example.com",
      instance: "portalv21",
      confirmationId: "tts-confirmation-id",
    });
    expect(sendVoice).toHaveBeenCalledTimes(1);
  });

  it("keeps an Arabic completed TTS-failure response visible and RTL", async () => {
    const transcript = "اعرض طلبات الاستثناء لهذا الشهر";
    const message = "هذه طلبات الاستثناء الخاصة بك.";
    sendVoice.mockResolvedValueOnce(
      response({
        message,
        transcript,
        language: "ar",
        detected_language: "ar",
        detected_locale: "ar-SA",
        session_id: "arabic-tts-session",
        audio_base64: "",
        tts_unavailable: true,
        blocks: [{
          type: "table",
          title: "طلبات الاستثناء",
          columns: [{ key: "status", label: "الحالة" }],
          rows: [{ status: "معلّق" }],
        }],
      }),
    );
    render(<App />);

    await holdAndRelease();

    const assistantText = await screen.findByText(message);
    expect(screen.getByText(transcript)).toBeTruthy();
    expect(assistantText.closest(".message-bubble").dir).toBe("rtl");
    expect(assistantText.closest(".message-bubble").lang).toBe("ar");
    expect(screen.getByText("طلبات الاستثناء")).toBeTruthy();
    expect(await screen.findByText(
      "الصوت غير متاح حاليًا. تم عرض الرد كنص.",
    )).toBeTruthy();
    expect(sendVoice).toHaveBeenCalledTimes(1);
    expect(audioBase64ToUrl).not.toHaveBeenCalled();
  });

  it("does not retry a finalized HTTP voice result after TTS failure", async () => {
    sendVoice.mockResolvedValueOnce(
      response({
        message: "Your request has been submitted.",
        transcript: "Yes",
        session_id: "completed-action-session",
        audio_base64: "",
        tts_unavailable: true,
      }),
    );
    render(<App />);

    await holdAndRelease();

    expect(await screen.findByText("Your request has been submitted.")).toBeTruthy();
    expect(sendVoice).toHaveBeenCalledTimes(1);
    expect(sendChat).not.toHaveBeenCalled();
    expect(audioBase64ToUrl).not.toHaveBeenCalled();
  });

  it("does not POST an invalid retained WAV after a streaming failure", async () => {
    isValidWavBlob.mockResolvedValueOnce(false);
    render(<App />);

    await holdAndRelease();

    expect(await screen.findByText("Hold the mic while you speak.")).toBeTruthy();
    expect(sendVoice).not.toHaveBeenCalled();
  });

  it("does not retry a stream no-speech result through the WAV fallback", async () => {
    const noSpeech = new Error("safe no speech");
    noSpeech.code = "no_speech";
    const streamController = {
      sendChunk: vi.fn(),
      finish: vi.fn().mockRejectedValue(noSpeech),
      cancel: vi.fn(),
    };
    openVoiceStream.mockResolvedValueOnce(streamController);
    render(<App />);

    await holdAndRelease();

    expect(
      await screen.findByText("I didn't catch that. Hold the mic and try again."),
    ).toBeTruthy();
    expect(sendVoice).not.toHaveBeenCalled();
  });

  it("does not retry a confirmation through WAV after streaming failure", async () => {
    sessionStorage.setItem("resourceplus.demo.confirmation", "pending-confirmation");
    const streamController = {
      sendChunk: vi.fn(),
      finish: vi.fn().mockRejectedValue(new Error("stream failed safely")),
      cancel: vi.fn(),
    };
    openVoiceStream.mockResolvedValueOnce(streamController);
    render(<App />);

    await holdAndRelease();

    expect(await screen.findByRole("alert")).toBeTruthy();
    expect(sendVoice).not.toHaveBeenCalled();
  });

  it("does not retry a pending confirmation when the stream cannot open", async () => {
    sessionStorage.setItem("resourceplus.demo.confirmation", "pending-confirmation");
    openVoiceStream.mockRejectedValueOnce(new Error("stream unavailable"));
    render(<App />);

    await holdAndRelease();

    expect(await screen.findByRole("alert")).toBeTruthy();
    expect(sendVoice).not.toHaveBeenCalled();
  });

  it("shows the resolved English badge when Azure's raw locale is Arabic", async () => {
    sendVoice.mockResolvedValueOnce(
      response({
        transcript: "How was my attendance this week?",
        detected_language: "en",
        detected_locale: "ar-SA",
        language: "en",
        message: "Here is your attendance for this week.",
        audio_base64: "UklGRg==",
        audio_mime_type: "audio/wav",
      }),
    );
    render(<App />);

    await holdAndRelease();

    expect(await screen.findByText("How was my attendance this week?")).toBeTruthy();
    expect(screen.getAllByText("English")).toHaveLength(2);
    expect(screen.queryByText("Arabic")).toBeNull();
    expect(document.body.textContent).not.toContain("ar-SA");
    expect(screen.getByText("How was my attendance this week?").parentElement.dir).toBe("ltr");
  });

  it("cancels safely on pointercancel without sending", async () => {
    const streamController = {
      sendChunk: vi.fn(),
      finish: vi.fn(),
      cancel: vi.fn(),
    };
    const recorder = {
      stop: vi.fn(),
      cancel: vi.fn().mockResolvedValue(undefined),
    };
    openVoiceStream.mockResolvedValueOnce(streamController);
    startWavRecording.mockResolvedValue(recorder);
    render(<App />);

    fireEvent.pointerDown(screen.getByLabelText("Hold to talk"), {
      pointerId: 2,
      pointerType: "touch",
      buttons: 1,
    });
    await waitFor(() => expect(startWavRecording).toHaveBeenCalledTimes(1));
    await act(async () => {});
    fireEvent.pointerCancel(screen.getByLabelText("Release to send voice message"), {
      pointerId: 2,
      pointerType: "touch",
    });

    await waitFor(() => expect(recorder.cancel).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(streamController.cancel).toHaveBeenCalledTimes(1));
    expect(recorder.stop).not.toHaveBeenCalled();
    expect(sendVoice).not.toHaveBeenCalled();
    await waitFor(() => expect(screen.getByLabelText("Hold to talk")).toBeTruthy());
  });

  it("prevents duplicate voice submission after repeated pointerup", async () => {
    render(<App />);
    await holdAndRelease();
    fireEvent.pointerUp(screen.getByLabelText("Hold to talk"), {
      pointerId: 1,
      pointerType: "mouse",
    });

    await waitFor(() => expect(sendVoice).toHaveBeenCalledTimes(1));
  });

  it("opens only one streaming socket for repeated pointerdown during one hold", async () => {
    const streamController = {
      sendChunk: vi.fn(),
      finish: vi.fn(),
      cancel: vi.fn(),
    };
    openVoiceStream.mockResolvedValueOnce(streamController);
    render(<App />);
    const mic = screen.getByLabelText("Hold to talk");

    fireEvent.pointerDown(mic, { pointerId: 20, pointerType: "touch", buttons: 1 });
    fireEvent.pointerDown(mic, { pointerId: 21, pointerType: "touch", buttons: 1 });

    await waitFor(() => expect(startWavRecording).toHaveBeenCalledTimes(1));
    expect(openVoiceStream).toHaveBeenCalledTimes(1);
    fireEvent.pointerCancel(screen.getByLabelText("Release to send voice message"), {
      pointerId: 20,
      pointerType: "touch",
    });
  });

  it("cleans up voice resources on New Conversation", async () => {
    const streamController = {
      sendChunk: vi.fn(),
      finish: vi.fn(),
      cancel: vi.fn(),
    };
    const recorder = { stop: vi.fn(), cancel: vi.fn().mockResolvedValue(undefined) };
    openVoiceStream.mockResolvedValueOnce(streamController);
    startWavRecording.mockResolvedValueOnce(recorder);
    render(<App />);

    fireEvent.pointerDown(screen.getByLabelText("Hold to talk"), {
      pointerId: 22,
      pointerType: "touch",
      buttons: 1,
    });
    await waitFor(() => expect(startWavRecording).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByLabelText("Start a new conversation"));

    await waitFor(() => expect(recorder.cancel).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(streamController.cancel).toHaveBeenCalledTimes(1));
    expect(sendVoice).not.toHaveBeenCalled();
  });

  it("cleans up voice resources when the component unmounts", async () => {
    const streamController = {
      sendChunk: vi.fn(),
      finish: vi.fn(),
      cancel: vi.fn(),
    };
    const recorder = { stop: vi.fn(), cancel: vi.fn().mockResolvedValue(undefined) };
    openVoiceStream.mockResolvedValueOnce(streamController);
    startWavRecording.mockResolvedValueOnce(recorder);
    const view = render(<App />);

    fireEvent.pointerDown(screen.getByLabelText("Hold to talk"), {
      pointerId: 23,
      pointerType: "touch",
      buttons: 1,
    });
    await waitFor(() => expect(startWavRecording).toHaveBeenCalledTimes(1));
    view.unmount();

    await waitFor(() => expect(recorder.cancel).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(streamController.cancel).toHaveBeenCalledTimes(1));
  });

  it("discards an accidental short recording and resets the timer", async () => {
    render(<App />);
    await holdAndRelease(150);

    await waitFor(() => expect(screen.getByText("Hold the mic while you speak.")).toBeTruthy());
    expect(sendVoice).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Hold to talk")).toBeTruthy();

    const now = vi.spyOn(performance, "now").mockReturnValue(2_000);
    fireEvent.pointerDown(screen.getByLabelText("Hold to talk"), {
      pointerId: 3,
      pointerType: "touch",
      buttons: 1,
    });
    expect(await screen.findByText(/Listening… 00:00/)).toBeTruthy();
    now.mockRestore();
    fireEvent.pointerCancel(screen.getByLabelText("Release to send voice message"), {
      pointerId: 3,
    });
  });

  it("offers manual playback when autoplay is blocked", async () => {
    globalThis.Audio.mockImplementationOnce(function BlockedAudio() {
      return {
        pause: vi.fn(),
        play: vi.fn(() => Promise.reject(new Error("NotAllowedError"))),
        onended: null,
        onerror: null,
      };
    });
    render(<App />);
    await holdAndRelease();

    const replay = await screen.findByLabelText("Replay assistant voice response");
    expect(await screen.findByText(
      "Audio didn’t autoplay. Use the speaker button to play it.",
    )).toBeTruthy();
    expect(screen.queryByRole("alert")).toBeNull();
    expect(document.body.textContent).not.toContain("Voice is temporarily unavailable");
    fireEvent.click(replay);
    await waitFor(() => expect(globalThis.Audio).toHaveBeenCalledTimes(2));
  });

  it("shows a classified voice backend failure while keeping typing available", async () => {
    sendVoice.mockRejectedValue(
      new Error("I couldn’t process that voice message. You can try again or type instead."),
    );
    render(<App />);
    await holdAndRelease();

    expect((await screen.findByRole("alert")).textContent).toContain(
      "I couldn’t process that voice message",
    );
    expect(screen.getByLabelText("Message ResourcePlus").disabled).toBe(false);
  });

  it("continues one session from voice to text", async () => {
    sendVoice.mockResolvedValueOnce(
      response({
        session_id: "cross-modal-session",
        transcript: "Show my attendance",
        detected_language: "en",
        detected_locale: "en-US",
        audio_base64: "",
      }),
    );
    render(<App />);
    await holdAndRelease();
    await waitFor(() => expect(sendVoice).toHaveBeenCalledTimes(1));
    await sendTypedMessage("What about last week?");

    expect(sendChat).toHaveBeenLastCalledWith(
      expect.objectContaining({
        message: "What about last week?",
        sessionId: "cross-modal-session",
      }),
    );
  });

  it("continues one session from text to voice", async () => {
    sendChat.mockResolvedValueOnce(response({ session_id: "text-first-session" }));
    render(<App />);
    await sendTypedMessage("Show my notifications");
    await waitFor(() => expect(screen.getByLabelText("Hold to talk")).toBeTruthy());
    await holdAndRelease();

    await waitFor(() => expect(sendVoice).toHaveBeenCalledTimes(1));
    expect(sendVoice).toHaveBeenCalledWith(
      expect.objectContaining({ sessionId: "text-first-session" }),
    );
  });

  it("shows confirmation controls without exposing the confirmation ID", async () => {
    sendChat.mockResolvedValueOnce(
      response({
        message: "Business Travel on 22 September 2026. Shall I submit it?",
        requires_confirmation: true,
        confirmation_id: "private-confirmation-123",
      }),
    );
    render(<App />);
    await sendTypedMessage("Book business travel");

    expect(await screen.findByText("Action requires confirmation")).toBeTruthy();
    expect(screen.getByRole("button", { name: "Confirm" })).toBeTruthy();
    expect(document.body.textContent).not.toContain("private-confirmation-123");
  });

  it("preserves the confirmation ID when Confirm is clicked", async () => {
    sendChat
      .mockResolvedValueOnce(
        response({
          message: "Please confirm this request.",
          requires_confirmation: true,
          confirmation_id: "confirmation-456",
        }),
      )
      .mockResolvedValueOnce(response({ message: "Request submitted." }));
    render(<App />);
    await sendTypedMessage("Submit a request");
    fireEvent.click(await screen.findByRole("button", { name: "Confirm" }));

    await waitFor(() => expect(sendChat).toHaveBeenCalledTimes(2));
    expect(sendChat).toHaveBeenLastCalledWith({
      message: "Yes",
      sessionId: "session-new",
      email: "talal.sabbagh@example.com",
      instance: "portalv21",
      confirmationId: "confirmation-456",
    });
  });

  it("renders live reason chips and submits one exact value only once", async () => {
    sendChat
      .mockResolvedValueOnce(
        response({
          message: "What was the reason?",
          needs_reason: true,
          reason_options: [
            { label: "Embassy Purposes", value: "Embassy Purposes" },
            { label: "Family Circumstances", value: "Family Circumstances" },
          ],
        }),
      )
      .mockResolvedValueOnce(
        response({
          message: "Submit this exceptional-entry request?",
          requires_confirmation: true,
          confirmation_id: "confirmation-reason",
        }),
      );
    render(<App />);
    await sendTypedMessage("Correct my September 1 IN punch");

    const group = await screen.findByRole("group", {
      name: "Select an exceptional-entry reason",
    });
    const embassy = screen.getByRole("button", { name: "Embassy Purposes" });
    expect(group.dir).toBe("ltr");
    embassy.focus();
    expect(document.activeElement).toBe(embassy);

    fireEvent.click(embassy);
    fireEvent.click(embassy);

    await waitFor(() => expect(sendChat).toHaveBeenCalledTimes(2));
    expect(sendChat).toHaveBeenLastCalledWith({
      message: "Embassy Purposes",
      sessionId: "session-new",
      email: "talal.sabbagh@example.com",
      instance: "portalv21",
      confirmationId: "",
    });
    expect(await screen.findByText(/Selected reason: Embassy Purposes/)).toBeTruthy();
    expect(screen.queryByRole("button", { name: "Embassy Purposes" })).toBeNull();
    expect(await screen.findByText("Action requires confirmation")).toBeTruthy();
    expect(document.body.textContent).not.toContain("reasonID");
  });

  it("renders Arabic reason options in an RTL accessible group", () => {
    render(
      <ChatMessage
        message={{
          role: "assistant",
          text: "ما سبب التصحيح؟",
          language: "ar",
          needsReason: true,
          reasonOptionsActive: true,
          reasonOptions: [
            { label: "ظروف عائلية", value: "ظروف عائلية" },
            { label: "أخرى", value: "أخرى" },
          ],
        }}
        onReasonSelect={vi.fn()}
        busy={false}
      />,
    );

    const group = screen.getByRole("group", {
      name: "Select an exceptional-entry reason",
    });
    expect(group.dir).toBe("rtl");
    expect(group.lang).toBe("ar");
    expect(screen.getByRole("button", { name: "ظروف عائلية" })).toBeTruthy();
  });

  it("clears the retained session and messages for a new conversation", async () => {
    sessionStorage.setItem("resourceplus.demo.session", "session-existing");
    sessionStorage.setItem(
      "resourceplus.demo.messages",
      JSON.stringify([{ id: "old", role: "user", text: "Old message" }]),
    );
    render(<App />);
    expect(screen.getByText("Old message")).toBeTruthy();

    fireEvent.click(screen.getByLabelText("Start a new conversation"));
    await waitFor(() => expect(screen.getByText("How can I help you today?")).toBeTruthy());
    expect(sessionStorage.getItem("resourceplus.demo.session")).toEqual(expect.any(String));
    expect(sessionStorage.getItem("resourceplus.demo.session")).not.toBe("session-existing");
    expect(screen.queryByText("Old message")).toBeNull();
  });

  it("stops active playback and releases audio URLs for a new conversation", async () => {
    const revokeObjectURL = vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});
    let activeAudio;
    globalThis.Audio.mockImplementationOnce(function ActiveAudio() {
      activeAudio = {
        pause: vi.fn(),
        play: vi.fn(() => Promise.resolve()),
        onended: null,
        onerror: null,
      };
      return activeAudio;
    });
    render(<App />);
    await holdAndRelease();
    expect(await screen.findByText(/Speaking/)).toBeTruthy();

    fireEvent.click(screen.getByLabelText("Start a new conversation"));

    expect(activeAudio.pause).toHaveBeenCalledTimes(1);
    expect(revokeObjectURL).toHaveBeenCalledWith("blob:assistant-audio");
    expect(await screen.findByText("How can I help you today?")).toBeTruthy();
    expect(screen.queryByText(/Speaking/)).toBeNull();
  });

  it("resets the conversation scroll position for a new conversation", async () => {
    sessionStorage.setItem(
      "resourceplus.demo.messages",
      JSON.stringify([{ id: "old", role: "user", text: "Old message" }]),
    );
    render(<App />);
    const history = screen.getByRole("log", { name: "Conversation history" });
    history.scrollTo = vi.fn();

    fireEvent.click(screen.getByLabelText("Start a new conversation"));

    await waitFor(() => expect(history.scrollTo).toHaveBeenCalledWith({
      top: 0,
      behavior: "auto",
    }));
    expect(screen.getByLabelText("Message ResourcePlus")).toBeTruthy();
  });

  it("retains the composer in a mobile-sized viewport", () => {
    Object.defineProperty(window, "innerWidth", { configurable: true, value: 390 });
    Object.defineProperty(window, "innerHeight", { configurable: true, value: 700 });
    render(<App />);

    expect(screen.getByRole("banner")).toBeTruthy();
    expect(screen.getByRole("log")).toBeTruthy();
    expect(screen.getByLabelText("Message ResourcePlus")).toBeTruthy();
  });

  it("uses shortcut text through the same ordinary chat path", async () => {
    render(<App />);
    fireEvent.click(screen.getByLabelText("Ask about Notifications"));

    await waitFor(() => expect(sendChat).toHaveBeenCalledTimes(1));
    expect(sendChat).toHaveBeenCalledWith({
      message: "Show my notifications",
      sessionId: expect.any(String),
      email: "talal.sabbagh@example.com",
      instance: "portalv21",
      confirmationId: "",
    });
  });

  it("offers both explicit demo identities and sends Talal identity", async () => {
    render(<App />);

    const selector = screen.getByLabelText("Test User");
    expect(selector.value).toBe("talal");
    expect(screen.getByRole("option", { name: "Talal Sabbagh - Employee" })).toBeTruthy();
    expect(screen.getByRole("option", { name: "Hana Haddad - HOD" })).toBeTruthy();
    expect(screen.getByText("talal.sabbagh@example.com")).toBeTruthy();

    await sendTypedMessage("Show my profile");
    expect(sendChat).toHaveBeenLastCalledWith(expect.objectContaining({
      email: "talal.sabbagh@example.com",
      instance: "portalv21",
    }));
  });

  it("switches users with fresh isolated sessions and clears old messages", async () => {
    render(<App />);
    const selector = screen.getByLabelText("Test User");
    const talalSession = sessionStorage.getItem("resourceplus.demo.session");
    await sendTypedMessage("Talal message");
    expect(await screen.findByText("Talal message")).toBeTruthy();

    fireEvent.change(selector, { target: { value: "hana" } });
    await waitFor(() => expect(selector.value).toBe("hana"));
    const hanaSession = sessionStorage.getItem("resourceplus.demo.session");
    expect(hanaSession).not.toBe(talalSession);
    expect(screen.queryByText("Talal message")).toBeNull();
    expect(screen.getByText("hana.haddad@example.com")).toBeTruthy();

    await sendTypedMessage("Hana message");
    expect(sendChat).toHaveBeenLastCalledWith(expect.objectContaining({
      sessionId: hanaSession,
      email: "hana.haddad@example.com",
      instance: "portalv21",
    }));

    fireEvent.change(selector, { target: { value: "talal" } });
    await waitFor(() => expect(selector.value).toBe("talal"));
    const newTalalSession = sessionStorage.getItem("resourceplus.demo.session");
    expect(newTalalSession).not.toBe(hanaSession);
    expect(newTalalSession).not.toBe(talalSession);
    expect(screen.queryByText("Hana message")).toBeNull();
  });

  it("starts a fresh session without changing the selected user", async () => {
    render(<App />);
    const selector = screen.getByLabelText("Test User");
    fireEvent.change(selector, { target: { value: "hana" } });
    await waitFor(() => expect(selector.value).toBe("hana"));
    await sendTypedMessage("Keep Hana selected");
    await waitFor(() => expect(screen.getByText("Here is your information.")).toBeTruthy());
    const previousSession = sessionStorage.getItem("resourceplus.demo.session");

    fireEvent.click(screen.getByRole("button", { name: "Start New Session" }));

    await waitFor(() => expect(screen.getByText("How can I help you today?")).toBeTruthy());
    expect(selector.value).toBe("hana");
    expect(screen.getByText("hana.haddad@example.com")).toBeTruthy();
    expect(sessionStorage.getItem("resourceplus.demo.session")).not.toBe(previousSession);
    expect(screen.queryByText("Keep Hana selected")).toBeNull();
  });

  it("uses the selected identity in the voice WebSocket START source", async () => {
    render(<App />);
    fireEvent.change(screen.getByLabelText("Test User"), { target: { value: "hana" } });
    const selectedSession = sessionStorage.getItem("resourceplus.demo.session");

    fireEvent.pointerDown(screen.getByLabelText("Hold to talk"), {
      pointerId: 31,
      pointerType: "mouse",
      button: 0,
      buttons: 1,
    });

    await waitFor(() => expect(openVoiceStream).toHaveBeenCalledWith(
      expect.objectContaining({
        sessionId: selectedSession,
        email: "hana.haddad@example.com",
        instance: "portalv21",
      }),
    ));
    fireEvent.pointerCancel(screen.getByLabelText("Release to send voice message"), {
      pointerId: 31,
      pointerType: "mouse",
    });
  });

  it("cancels active voice resources when the test user changes", async () => {
    const streamController = {
      sendChunk: vi.fn(),
      finish: vi.fn(),
      cancel: vi.fn(),
    };
    const recorder = { stop: vi.fn(), cancel: vi.fn().mockResolvedValue(undefined) };
    openVoiceStream.mockResolvedValueOnce(streamController);
    startWavRecording.mockResolvedValueOnce(recorder);
    render(<App />);

    fireEvent.pointerDown(screen.getByLabelText("Hold to talk"), {
      pointerId: 32,
      pointerType: "touch",
      buttons: 1,
    });
    await waitFor(() => expect(startWavRecording).toHaveBeenCalledTimes(1));

    fireEvent.change(screen.getByLabelText("Test User"), { target: { value: "hana" } });

    await waitFor(() => expect(recorder.cancel).toHaveBeenCalledTimes(1));
    await waitFor(() => expect(streamController.cancel).toHaveBeenCalledTimes(1));
    expect(sendVoice).not.toHaveBeenCalled();
    expect(screen.getByLabelText("Test User").value).toBe("hana");
  });
});
