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


@dataclass(frozen=True)
class SessionLanguageResolution:
    """Session-aware language decision made before conversational routing."""

    language: str
    source: str
    short_utterance: bool
    last_confident_language: str | None


@dataclass(frozen=True)
class ScriptAnalysis:
    classification: str
    arabic_letters: int
    latin_letters: int


_ENGLISH_SWITCHES = (
    "continue in english",
    "switch to english",
    "speak english",
    "in english please",
    "كمل بالإنجليزي",
    "نكمل بالإنجليزي",
    "تكلم إنجليزي",
    "بالإنجليزي لو سمحت",
    "كمل بالانجليزي",
    "تكلم انجليزي",
)
_ARABIC_SWITCHES = (
    "continue in arabic",
    "switch to arabic",
    "speak arabic",
    "in arabic please",
    "كمل بالعربي",
    "نكمل بالعربي",
    "تكلم عربي",
    "بالعربي لو سمحت",
)
_AMBIGUOUS_SHORT_TURNS = {
    "yes", "yes please", "no", "no thanks", "confirm", "cancel",
    "do it", "this one", "that one", "that day", "it",
    "نعم", "لا", "أكد", "الغاء", "إلغاء", "هذا", "هذه", "ذاك",
}


def explicit_language_request(text: str) -> str | None:
    normalized = " ".join(text.casefold().split())
    if any(phrase in normalized for phrase in _ENGLISH_SWITCHES):
        return "en"
    if any(phrase in normalized for phrase in _ARABIC_SWITCHES):
        return "ar"
    return None


def analyze_script(text: str) -> ScriptAnalysis:
    """Classify script evidence without treating it as spoken-language truth."""

    arabic, latin = count_script_letters(text)
    if not arabic and not latin:
        classification = "script_neutral"
    elif arabic and latin:
        minority = min(arabic, latin)
        total = arabic + latin
        classification = (
            "mixed"
            if minority >= 3 or minority / total >= 0.2
            else "primarily_arabic" if arabic > latin else "primarily_latin"
        )
    elif arabic:
        classification = "primarily_arabic"
    else:
        classification = "primarily_latin"
    return ScriptAnalysis(classification, arabic, latin)


def content_matches_language(text: str, language: str) -> bool:
    """Reject only clear response-language contradictions, not names/HR terms."""

    evidence = analyze_script(text)
    if evidence.classification == "script_neutral":
        return True
    expected = (
        evidence.arabic_letters if language == "ar" else evidence.latin_letters
    )
    opposite = (
        evidence.latin_letters if language == "ar" else evidence.arabic_letters
    )
    if expected == 0:
        return opposite == 0
    return not (opposite >= 6 and opposite > expected * 1.5)


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


def resolve_session_language(
    transcript: str,
    *,
    last_confident_language: str | None = None,
    fallback: str = "en",
) -> SessionLanguageResolution:
    """Stabilize short voice turns without treating a raw STT locale as intent.

    An explicit language request wins. Otherwise, a short or ambiguous turn keeps
    the session's last confidently established language. Longer single-script
    speech can establish or change the language. The configured fallback is used
    only when neither transcript evidence nor session context is available.
    """

    normalized = " ".join(transcript.casefold().split())
    explicit = explicit_language_request(transcript)

    arabic, latin = count_script_letters(transcript)
    words = tuple(part for part in transcript.split() if any(char.isalpha() for char in part))
    short = (
        len(words) <= 1
        or arabic + latin <= 6
        or normalized in _AMBIGUOUS_SHORT_TURNS
        or not words
    )
    established = (
        last_confident_language
        if last_confident_language in SUPPORTED_LANGUAGES
        else None
    )
    configured = fallback if fallback in SUPPORTED_LANGUAGES else "en"

    if explicit is not None:
        return SessionLanguageResolution(
            explicit,
            "explicit_language_switch",
            short,
            explicit,
        )
    if short and established is not None:
        return SessionLanguageResolution(
            established,
            "last_confident_language",
            True,
            established,
        )
    if arabic > latin:
        return SessionLanguageResolution(
            "ar",
            "transcript_arabic_script",
            short,
            "ar",
        )
    if latin > arabic:
        return SessionLanguageResolution(
            "en",
            "transcript_latin_script",
            short,
            "en",
        )
    if established is not None:
        return SessionLanguageResolution(
            established,
            "last_confident_language",
            short,
            established,
        )
    return SessionLanguageResolution(
        configured,
        "configured_fallback",
        short,
        None,
    )
