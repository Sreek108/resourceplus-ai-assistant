from dataclasses import dataclass
import unicodedata


SUPPORTED_LANGUAGES = {"en", "ar"}


@dataclass(frozen=True)
class SpokenLanguageResolution:
    language: str
    locale: str
    source: str
    arabic_letters: int
    latin_letters: int


def count_script_letters(text: str) -> tuple[int, int]:
    """Count Arabic and Latin Unicode letters without vocabulary assumptions."""

    arabic = 0
    latin = 0
    for character in text:
        if not unicodedata.category(character).startswith("L"):
            continue
        name = unicodedata.name(character, "")
        if "ARABIC" in name:
            arabic += 1
        elif "LATIN" in name:
            latin += 1
    return arabic, latin


def _locale_language(locale: str | None) -> str:
    normalized = (locale or "").strip().casefold()
    return "ar" if normalized == "ar" or normalized.startswith("ar-") else "en"


def detect_text_language(text: str, *, fallback: str = "en") -> str:
    """Resolve English/Arabic from script evidence for typed text."""

    arabic, latin = count_script_letters(text)
    if arabic > latin:
        return "ar"
    if latin > arabic:
        return "en"
    if arabic and latin:
        return fallback if fallback in SUPPORTED_LANGUAGES else "en"
    return fallback if fallback in SUPPORTED_LANGUAGES else "en"


def resolve_spoken_language(
    transcript: str,
    detected_locale: str | None,
    *,
    english_locale: str = "en-US",
    arabic_locale: str = "ar-SA",
) -> SpokenLanguageResolution:
    """Resolve speech language using transcript script before raw Azure locale.

    Digits, punctuation, whitespace, and emoji do not influence the result. A
    transcript containing letters from only one supported script always wins over
    the raw locale. Mixed text uses the dominant script; ties and text without
    supported letters safely fall back to Azure's locale.
    """

    arabic, latin = count_script_letters(transcript)
    if arabic > latin:
        language = "ar"
        source = "transcript_arabic_script"
    elif latin > arabic:
        language = "en"
        source = "transcript_latin_script"
    else:
        language = _locale_language(detected_locale)
        source = "azure_locale_fallback"
    locale = arabic_locale if language == "ar" else english_locale
    return SpokenLanguageResolution(
        language=language,
        locale=locale,
        source=source,
        arabic_letters=arabic,
        latin_letters=latin,
    )
