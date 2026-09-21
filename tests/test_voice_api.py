import base64

import pytest
from fastapi.testclient import TestClient

from app.api import chat as chat_module
from app.api import voice as voice_module
from app.main import app
from app.models.schemas import ChatResponse
from app.services import chat as chat_service
from app.speech import SpeechAudio, SpeechInputError, SpeechTranscript


client = TestClient(app)


def test_text_and_voice_routes_share_process_chat() -> None:
    assert chat_module.process_chat is chat_service.process_chat
    assert voice_module.process_chat is chat_service.process_chat


@pytest.mark.parametrize(
    ("locale", "language", "transcript", "message"),
    [
        ("en-US", "en", "Show my profile", "Your profile is ready."),
        ("ar-SA", "ar", "أرني ملفي", "هذا ملخص ملفك."),
        (
            "ar-SA",
            "en",
            "How was my attendance this week?",
            "Here is your attendance for this week.",
        ),
        (
            "en-US",
            "ar",
            "أبغى أشوف attendance حقي هذا الأسبوع",
            "هذا ملخص حضورك لهذا الأسبوع.",
        ),
    ],
)
def test_voice_chat_runs_transcript_through_shared_agent_and_tts(
    monkeypatch,
    locale: str,
    language: str,
    transcript: str,
    message: str,
) -> None:
    calls = {}

    async def transcribe(payload, *, content_type):
        calls["audio"] = (payload, content_type)
        return SpeechTranscript(transcript, locale, language)

    async def process(request, *, detected_language):
        calls["chat"] = (request, detected_language)
        return ChatResponse(
            success=True,
            message=message,
            language=language,
            tools_used=["get_profile_data"],
            session_id=request.session_id or "voice-session",
        ).set_speech_message(message)

    async def synthesize(text, *, language):
        calls["tts"] = (text, language)
        return SpeechAudio(b"RIFFresponse")

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    monkeypatch.setattr(voice_module, "process_chat", process)
    monkeypatch.setattr(voice_module, "synthesize_speech", synthesize)
    response = client.post(
        "/api/voice/chat",
        files={"audio": ("utterance.wav", b"RIFFinput", "audio/wav")},
        data={"session_id": "shared-session"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["transcript"] == transcript
    assert body["detected_locale"] == locale
    assert body["detected_language"] == language
    assert body["session_id"] == "shared-session"
    assert body["audio_base64"] == base64.b64encode(b"RIFFresponse").decode()
    assert calls["chat"][0].message == transcript
    assert calls["chat"][1] == language
    assert calls["tts"] == (message, language)


def test_voice_chat_preserves_pending_confirmation_fields(monkeypatch) -> None:
    async def transcribe(*args, **kwargs):
        return SpeechTranscript("Proceed naturally", "en-US", "en")

    async def process(request, *, detected_language):
        assert request.session_id == "existing-session"
        assert request.confirmation_id == "existing-confirmation"
        return ChatResponse(
            success=True,
            message="Would you like me to continue?",
            language=detected_language,
            session_id=request.session_id,
            requires_confirmation=True,
            confirmation_id=request.confirmation_id,
        ).set_speech_message("Would you like me to continue?")

    async def synthesize(text, *, language):
        return SpeechAudio(b"audio")

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    monkeypatch.setattr(voice_module, "process_chat", process)
    monkeypatch.setattr(voice_module, "synthesize_speech", synthesize)
    response = client.post(
        "/api/voice/chat",
        files={"audio": ("utterance.wav", b"RIFFinput", "audio/wav")},
        data={
            "session_id": "existing-session",
            "confirmation_id": "existing-confirmation",
        },
    )

    assert response.json()["requires_confirmation"] is True
    assert response.json()["confirmation_id"] == "existing-confirmation"


@pytest.mark.parametrize(
    ("language", "locale", "message", "expected_speech"),
    [
        (
            "en",
            "en-US",
            "### Contact Information\n\n- **Employee Number:** UN003\n- [Portal](https://example.com)",
            "Contact Information. Employee Number: UN003. Portal.",
        ),
        (
            "ar",
            "ar-SA",
            "### معلومات الموظف\n\n- **الاسم:** سنيش\n- **الرقم:** UN003",
            "معلومات الموظف. الاسم: سنيش. الرقم: UN003.",
        ),
    ],
)
def test_voice_chat_preserves_display_markdown_but_synthesizes_clean_text(
    monkeypatch,
    language: str,
    locale: str,
    message: str,
    expected_speech: str,
) -> None:
    synthesized = {}

    async def transcribe(*args, **kwargs):
        return SpeechTranscript("Show my profile", locale, language)

    async def process(request, *, detected_language):
        return ChatResponse(
            success=True,
            message=message,
            language=detected_language,
            session_id="markdown-session",
        ).set_speech_message(expected_speech)

    async def synthesize(text, *, language):
        synthesized["call"] = (text, language)
        return SpeechAudio(b"clean-audio")

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    monkeypatch.setattr(voice_module, "process_chat", process)
    monkeypatch.setattr(voice_module, "synthesize_speech", synthesize)

    response = client.post(
        "/api/voice/chat",
        files={"audio": ("utterance.wav", b"RIFFinput", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["message"] == message
    assert synthesized["call"] == (expected_speech, language)
    assert "###" not in synthesized["call"][0]
    assert "**" not in synthesized["call"][0]
    assert "https://" not in synthesized["call"][0]


def test_voice_never_falls_back_from_missing_speech_message_to_display(
    monkeypatch,
) -> None:
    async def transcribe(*args, **kwargs):
        return SpeechTranscript("Show my profile", "en-US", "en")

    async def process(request, *, detected_language):
        return ChatResponse(
            success=True,
            message="This display-only response must not be synthesized.",
            language=detected_language,
            session_id="missing-speech-session",
        )

    async def synthesize(*args, **kwargs):
        raise AssertionError("Display text must never be used as the TTS source")

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    monkeypatch.setattr(voice_module, "process_chat", process)
    monkeypatch.setattr(voice_module, "synthesize_speech", synthesize)

    response = client.post(
        "/api/voice/chat",
        files={"audio": ("utterance.wav", b"RIFFinput", "audio/wav")},
    )

    assert response.status_code == 502
    assert response.json()["detail"]["code"] == "speech_synthesis_failed"


def test_voice_chat_speaks_concise_summary_but_keeps_detailed_screen_text(
    monkeypatch,
) -> None:
    display_message = (
        "### Attendance\n\n- Sunday: Present\n- Monday: Absent\n"
        "- Tuesday: Present\n- Wednesday: Present"
    )
    calls = {}

    async def transcribe(*args, **kwargs):
        return SpeechTranscript("How did my week look?", "en-US", "en")

    async def process(request, *, detected_language):
        return ChatResponse(
            success=True,
            message=display_message,
            language=detected_language,
            tools_used=["get_attendance_summary"],
            session_id="voice-summary-session",
        ).set_speech_message(
            "You were absent on Monday. The daily details are on screen."
        )

    async def synthesize(text, *, language):
        calls["tts"] = (text, language)
        return SpeechAudio(b"summary-audio")

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    monkeypatch.setattr(voice_module, "process_chat", process)
    monkeypatch.setattr(voice_module, "synthesize_speech", synthesize)

    response = client.post(
        "/api/voice/chat",
        files={"audio": ("utterance.wav", b"RIFFinput", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["message"] == display_message
    assert calls["tts"] == (
        "You were absent on Monday. The daily details are on screen.",
        "en",
    )
    assert "get_attendance_summary" not in calls["tts"][0]


@pytest.mark.parametrize(
    ("language", "locale", "transcript", "speech_message"),
    [
        ("en", "en-US", "Show my notifications", "You have two notifications."),
        (
            "ar",
            "ar-SA",
            "أرني تنبيهاتي",
            "لديك تنبيهان جديدان.",
        ),
    ],
)
def test_voice_uses_main_agent_speech_message_in_resolved_language(
    monkeypatch,
    language: str,
    locale: str,
    transcript: str,
    speech_message: str,
) -> None:
    calls = {}

    async def transcribe(*args, **kwargs):
        return SpeechTranscript(transcript, locale, language)

    async def process(request, *, detected_language):
        assert detected_language == language
        return ChatResponse(
            success=True,
            message="### Detailed result\n\n- Item one\n- Item two",
            language=language,
            session_id="speech-language-session",
        ).set_speech_message(speech_message)

    async def synthesize(text, *, language):
        calls["tts"] = (text, language)
        return SpeechAudio(b"speech-audio")

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    monkeypatch.setattr(voice_module, "process_chat", process)
    monkeypatch.setattr(voice_module, "synthesize_speech", synthesize)

    response = client.post(
        "/api/voice/chat",
        files={"audio": ("utterance.wav", b"RIFFinput", "audio/wav")},
    )

    assert response.status_code == 200
    assert response.json()["message"].startswith("### Detailed result")
    assert "speech_message" not in response.json()
    assert calls["tts"] == (speech_message, language)


def test_voice_path_has_no_separate_voice_summary_model_dependency() -> None:
    assert not hasattr(voice_module, "render_voice_message")


def test_voice_chat_rejects_empty_audio() -> None:
    response = client.post(
        "/api/voice/chat",
        files={"audio": ("empty.wav", b"", "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == {
        "code": "no_speech",
        "message": "The uploaded audio is empty.",
    }


def test_fallback_stt_no_match_is_recoverable_and_skips_agent(
    monkeypatch,
    caplog,
) -> None:
    process_calls = []
    audits = []

    async def transcribe(*args, **kwargs):
        raise SpeechInputError(
            "No speech could be recognized from the audio.",
            safe_category="fallback_stt_no_match",
        )

    async def process(*args, **kwargs):
        process_calls.append((args, kwargs))
        raise AssertionError("Fallback no-match must not call the agent or ResourcePlus")

    async def persist(audit):
        audits.append(audit)

    monkeypatch.setattr(voice_module, "transcribe_audio", transcribe)
    monkeypatch.setattr(voice_module, "process_chat", process)
    monkeypatch.setattr(voice_module, "persist_audit", persist)
    caplog.set_level("INFO", logger="app.voice_latency")

    response = client.post(
        "/api/voice/chat",
        files={"audio": ("utterance.wav", b"RIFFinput", "audio/wav")},
    )

    assert response.status_code == 400
    assert response.json()["detail"]["code"] == "no_speech"
    assert process_calls == []
    assert audits[0].error_category == "fallback_stt_no_match"
    assert any(
        "stream_error_category=fallback_stt_no_match" in record.getMessage()
        for record in caplog.records
        if record.name == "app.voice_latency"
    )
