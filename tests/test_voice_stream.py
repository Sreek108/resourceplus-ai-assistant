from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from app.api import voice as voice_module
from app.main import app
from app.models.schemas import ChatResponse
from app.speech import (
    SpeechAudio,
    SpeechInputError,
    SpeechRecognitionError,
    SpeechTranscript,
)
from app.speech import streaming


client = TestClient(app)


def test_websocket_stream_is_reachable_from_allowed_browser_origin() -> None:
    with client.websocket_connect(
        "/api/voice/stream",
        headers={"origin": "http://127.0.0.1:5173"},
    ) as websocket:
        websocket.send_json({"type": "invalid"})
        response = websocket.receive_json()

    assert response["type"] == "error"
    assert response["code"] == "invalid_stream_state"


def test_websocket_stream_rejects_unlisted_browser_origin() -> None:
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(
            "/api/voice/stream",
            headers={"origin": "https://untrusted.example"},
        ):
            pass

    assert exc_info.value.code == 1008


class FakeStreamingRecognizer:
    def __init__(self, transcript=None, *, finish_error=None):
        self.transcript = transcript or SpeechTranscript(
            "Show my notifications",
            "en-US",
            "en",
        )
        self.finish_error = finish_error
        self.started = False
        self.chunks = []
        self.cancelled = False

    async def start(self):
        self.started = True

    def write(self, chunk):
        self.chunks.append(chunk)

    async def finish(self):
        if self.finish_error:
            raise self.finish_error
        return self.transcript

    async def cancel(self):
        self.cancelled = True


@pytest.mark.parametrize(
    ("transcript", "locale", "language"),
    [
        ("Show my notifications", "ar-SA", "en"),
        ("أرني تنبيهاتي", "en-US", "ar"),
        ("أبغى أشوف attendance حقي", "en-US", "ar"),
    ],
)
def test_websocket_stream_start_chunks_end_and_final_result(
    monkeypatch,
    transcript: str,
    locale: str,
    language: str,
) -> None:
    recognizer = FakeStreamingRecognizer(
        SpeechTranscript(transcript, locale, language)
    )
    calls = []

    async def process(request, *, detected_language):
        calls.append((request, detected_language))
        return ChatResponse(
            success=True,
            message="### Current result\n\n- Detail",
            language=detected_language,
            session_id=request.session_id or "stream-session",
        ).set_speech_message("Your current result is ready.")

    async def synthesize(text, *, language: str):
        assert text == "Your current result is ready."
        assert language == expected_language
        return SpeechAudio(b"stream-audio")

    async def persist(*args, **kwargs):
        return None

    expected_language = language

    monkeypatch.setattr(voice_module, "StreamingSpeechRecognizer", lambda: recognizer)
    monkeypatch.setattr(voice_module, "process_chat", process)
    monkeypatch.setattr(voice_module, "synthesize_speech", synthesize)
    monkeypatch.setattr(voice_module, "persist_audit", persist)

    with client.websocket_connect("/api/voice/stream") as websocket:
        websocket.send_json(
            {
                "type": "start",
                "sample_rate": 16_000,
                "session_id": "existing-session",
            }
        )
        assert websocket.receive_json() == {"type": "ready"}
        websocket.send_bytes(b"\x01\x00" * 320)
        websocket.send_bytes(b"\x02\x00" * 320)
        websocket.send_json({"type": "end"})
        result = websocket.receive_json()

    assert result["type"] == "final"
    assert result["transcript"] == transcript
    assert result["detected_locale"] == locale
    assert result["detected_language"] == language
    assert result["session_id"] == "existing-session"
    assert recognizer.started is True
    assert len(recognizer.chunks) == 2
    assert len(calls) == 1
    assert calls[0][0].session_id == "existing-session"
    assert calls[0][1] == language


def test_stream_disconnect_cleans_up_recognizer(monkeypatch) -> None:
    recognizer = FakeStreamingRecognizer()
    monkeypatch.setattr(voice_module, "StreamingSpeechRecognizer", lambda: recognizer)

    with client.websocket_connect("/api/voice/stream") as websocket:
        websocket.send_json({"type": "start", "sample_rate": 16_000})
        assert websocket.receive_json() == {"type": "ready"}
        websocket.send_bytes(b"\x00\x00" * 320)

    assert recognizer.cancelled is True


def test_short_stream_without_recognized_speech_is_recoverable_and_skips_agent(
    monkeypatch,
    caplog,
) -> None:
    recognizer = FakeStreamingRecognizer(
        finish_error=SpeechInputError(
            "No speech could be recognized.",
            safe_category="no_recognized_speech",
        )
    )
    process_calls = []

    async def process(*args, **kwargs):
        process_calls.append((args, kwargs))
        raise AssertionError("No-speech must not call OpenAI or ResourcePlus through the agent")

    async def persist(*args, **kwargs):
        return None

    monkeypatch.setattr(voice_module, "StreamingSpeechRecognizer", lambda: recognizer)
    monkeypatch.setattr(voice_module, "process_chat", process)
    monkeypatch.setattr(voice_module, "persist_audit", persist)
    caplog.set_level("INFO", logger="app.voice_latency")

    with client.websocket_connect("/api/voice/stream") as websocket:
        websocket.send_json({"type": "start", "sample_rate": 16_000})
        assert websocket.receive_json() == {"type": "ready"}
        websocket.send_bytes(b"\x00\x00" * 40)
        websocket.send_json({"type": "end"})
        result = websocket.receive_json()

    assert result == {
        "type": "error",
        "code": "no_speech",
        "message": "I didn't catch that. Hold the mic and try again.",
    }
    assert process_calls == []
    summaries = [
        record.getMessage()
        for record in caplog.records
        if record.name == "app.voice_latency"
    ]
    assert any("status=204" in item for item in summaries)
    assert any("stream_error_category=no_recognized_speech" in item for item in summaries)


def test_stream_with_no_audio_is_not_logged_as_server_failure(monkeypatch, caplog) -> None:
    recognizer = FakeStreamingRecognizer()

    async def persist(*args, **kwargs):
        return None

    monkeypatch.setattr(voice_module, "StreamingSpeechRecognizer", lambda: recognizer)
    monkeypatch.setattr(voice_module, "persist_audit", persist)
    caplog.set_level("INFO", logger="app.voice_latency")

    with client.websocket_connect("/api/voice/stream") as websocket:
        websocket.send_json({"type": "start", "sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_json({"type": "end"})
        result = websocket.receive_json()

    assert result["code"] == "no_speech"
    assert any(
        "status=204" in record.getMessage()
        and "stream_error_category=no_audio" in record.getMessage()
        for record in caplog.records
        if record.name == "app.voice_latency"
    )


def test_stream_failure_does_not_corrupt_next_request(monkeypatch) -> None:
    recognizers = [
        FakeStreamingRecognizer(
            finish_error=SpeechRecognitionError("provider detail")
        ),
        FakeStreamingRecognizer(),
    ]

    async def process(request, *, detected_language):
        return ChatResponse(
            success=True,
            message="Current notifications.",
            language=detected_language,
            session_id="next-session",
        ).set_speech_message("Current notifications.")

    async def synthesize(text, *, language):
        return SpeechAudio(b"audio")

    async def persist(*args, **kwargs):
        return None

    monkeypatch.setattr(
        voice_module,
        "StreamingSpeechRecognizer",
        lambda: recognizers.pop(0),
    )
    monkeypatch.setattr(voice_module, "process_chat", process)
    monkeypatch.setattr(voice_module, "synthesize_speech", synthesize)
    monkeypatch.setattr(voice_module, "persist_audit", persist)

    with client.websocket_connect("/api/voice/stream") as websocket:
        websocket.send_json({"type": "start", "sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(b"\x00\x00" * 320)
        websocket.send_json({"type": "end"})
        error = websocket.receive_json()
    assert error["type"] == "error"
    assert error["code"] == "speech_recognition_failed"
    assert "provider detail" not in error["message"]

    with client.websocket_connect("/api/voice/stream") as websocket:
        websocket.send_json({"type": "start", "sample_rate": 16_000})
        websocket.receive_json()
        websocket.send_bytes(b"\x00\x00" * 320)
        websocket.send_json({"type": "end"})
        result = websocket.receive_json()
    assert result["type"] == "final"


class FakeSignal:
    def __init__(self):
        self.callback = None

    def connect(self, callback):
        self.callback = callback

    def emit(self, event):
        self.callback(event)


class FakeFuture:
    def get(self):
        return None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("text", "raw_locale", "expected_language"),
    [
        ("How was my attendance?", "ar-SA", "en"),
        ("أبغى أعرف حضوري", "en-US", "ar"),
        ("أبغى أشوف attendance حقي", "en-US", "ar"),
    ],
)
async def test_azure_streaming_final_transcript_uses_existing_language_resolver(
    monkeypatch,
    text: str,
    raw_locale: str,
    expected_language: str,
) -> None:
    class PushStream:
        def __init__(self, stream_format):
            self.stream_format = stream_format
            self.chunks = []

        def write(self, chunk):
            self.chunks.append(chunk)

        def close(self):
            pass

    class Recognizer:
        def __init__(self, **kwargs):
            self.recognized = FakeSignal()
            self.canceled = FakeSignal()
            self.session_stopped = FakeSignal()

        def start_continuous_recognition_async(self):
            return FakeFuture()

        def stop_continuous_recognition_async(self):
            return FakeFuture()

    fake_sdk = SimpleNamespace(
        ResultReason=SimpleNamespace(RecognizedSpeech="recognized"),
        SpeechConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        SpeechRecognizer=Recognizer,
        AutoDetectSourceLanguageResult=lambda result: SimpleNamespace(
            language=result.locale
        ),
        languageconfig=SimpleNamespace(
            AutoDetectSourceLanguageConfig=lambda languages: SimpleNamespace(
                languages=languages
            )
        ),
        audio=SimpleNamespace(
            AudioStreamFormat=lambda **kwargs: SimpleNamespace(**kwargs),
            PushAudioInputStream=PushStream,
            AudioConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        ),
    )
    monkeypatch.setattr(streaming, "speechsdk", fake_sdk)
    monkeypatch.setattr(
        streaming,
        "get_settings",
        lambda: SimpleNamespace(
            azure_speech_key="test",
            azure_speech_region="test",
            azure_speech_en_locale="en-US",
            azure_speech_ar_locale="ar-SA",
        ),
    )

    session = streaming.StreamingSpeechRecognizer()
    await session.start()
    session.write(b"\x00\x00" * 320)
    result = SimpleNamespace(reason="recognized", text=text, locale=raw_locale)
    session._recognizer.recognized.emit(SimpleNamespace(result=result))
    session._recognizer.session_stopped.emit(SimpleNamespace())
    recognized = await session.finish()

    assert recognized.transcript == text
    assert recognized.detected_locale == raw_locale
    assert recognized.detected_language == expected_language
