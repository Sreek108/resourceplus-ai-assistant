import pytest

from app.speech.language import detect_text_language, resolve_spoken_language


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

