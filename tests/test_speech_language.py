import pytest

from app.speech.language import (
    analyze_script,
    content_matches_language,
    detect_text_language,
    resolve_session_language,
    resolve_spoken_language,
)


@pytest.mark.parametrize(
    ("transcript", "azure_locale", "expected_language", "expected_locale"),
    [
        ("How was my attendance this week?", "ar-SA", "en", "en-US"),
        ("Show my notifications", "ar-SA", "en", "en-US"),
        ("أبغى أعرف حضوري هذا الأسبوع", "en-US", "ar", "ar-SA"),
        ("أبغى أشوف attendance حقي هذا الأسبوع", "en-US", "ar", "ar-SA"),
        ("Show 3 notifications for 2026", "ar-SA", "en", "en-US"),
        ("عندي ٣ تنبيهات في ٢٠٢٦", "en-US", "ar", "ar-SA"),
    ],
)
def test_transcript_script_overrides_conflicting_azure_locale(
    transcript: str,
    azure_locale: str,
    expected_language: str,
    expected_locale: str,
) -> None:
    resolution = resolve_spoken_language(transcript, azure_locale)

    assert resolution.language == expected_language
    assert resolution.locale == expected_locale
    assert resolution.source.startswith("transcript_")


@pytest.mark.parametrize(
    ("transcript", "azure_locale", "expected_language"),
    [
        ("", "ar-SA", "ar"),
        ("... 123 🎙️", "ar-SA", "ar"),
        ("... 123 🎙️", "en-US", "en"),
        ("... 123 🎙️", None, "en"),
    ],
)
def test_ambiguous_transcript_safely_falls_back_to_azure_locale(
    transcript: str,
    azure_locale: str | None,
    expected_language: str,
) -> None:
    resolution = resolve_spoken_language(transcript, azure_locale)

    assert resolution.language == expected_language
    assert resolution.source == "azure_locale_fallback"


def test_typed_text_reuses_script_based_detection() -> None:
    assert detect_text_language("Show 2 notifications") == "en"
    assert detect_text_language("عندي ٢ تنبيهات") == "ar"
    assert detect_text_language("أبغى أشوف attendance حقي") == "ar"


def test_short_voice_turn_keeps_last_confident_language_and_ignores_raw_locale() -> None:
    resolution = resolve_session_language(
        "Yes",
        last_confident_language="en",
        fallback="ar",
    )

    assert resolution.language == "en"
    assert resolution.source == "last_confident_language"
    assert resolution.short_utterance is True
    assert resolution.last_confident_language == "en"


@pytest.mark.parametrize(
    ("transcript", "established", "expected"),
    [
        ("Continue in Arabic", "en", "ar"),
        ("خلينا نكمل بالعربي", "en", "ar"),
        ("Continue in English", "ar", "en"),
        ("خلينا نكمل بالإنجليزي", "ar", "en"),
    ],
)
def test_explicit_language_switch_overrides_session_language(
    transcript: str,
    established: str,
    expected: str,
) -> None:
    resolution = resolve_session_language(
        transcript,
        last_confident_language=established,
    )

    assert resolution.language == expected
    assert resolution.source == "explicit_language_switch"
    assert resolution.last_confident_language == expected


def test_materially_mixed_transcript_is_ambiguous_evidence() -> None:
    analysis = analyze_script("Hay resource بلس.")

    assert analysis.classification == "mixed"
    assert analysis.latin_letters > analysis.arabic_letters > 0


def test_response_language_validation_catches_clear_opposite_script() -> None:
    assert content_matches_language("Your attendance is ready.", "en") is True
    assert content_matches_language("حضورك جاهز.", "ar") is True
    assert content_matches_language("حضورك جاهز.", "en") is False
    assert content_matches_language("Your attendance is ready.", "ar") is False
    assert content_matches_language("OK", "ar") is False
    assert content_matches_language("تم", "en") is False

