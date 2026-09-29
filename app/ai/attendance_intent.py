from __future__ import annotations

import re


_ENGLISH_CORRECTION_PATTERNS = (
    re.compile(
        r"\b(?:forgot|forget|forgotten|missed|did(?:n't| not)|could(?:n't| not)|"
        r"failed to)\s+(?:to\s+)?(?:punch|clock|check)\b"
    ),
    re.compile(
        r"\b(?:missing|correct|fix|regulari[sz]e|adjust|amend)\b.{0,32}"
        r"\b(?:punch|attendance|entry|clock|check(?:-?in|-?out)?)\b"
    ),
    re.compile(
        r"\b(?:punch|attendance|entry|clock|check(?:-?in|-?out)?)\b.{0,32}"
        r"\b(?:missing|wrong|incorrect|not recorded|failed)\b"
    ),
    re.compile(r"\b(?:in|out)\s+punch\b.{0,16}\bmissing\b"),
    re.compile(r"\b(?:forgot|missed)\s+(?:my\s+)?(?:morning|evening)\s+punch\b"),
)
_ENGLISH_LATE_CONTEXT = re.compile(r"\b(?:late|delayed?)\b")
_ENGLISH_PROSPECTIVE = re.compile(
    r"\b(?:i|we)(?:'ll|\s+will|\s+may|\s+might|\s+expect(?:\s+to)?|"
    r"\s+(?:am|are)\s+going\s+to)\b"
    r"|\bi\s+think\s+i(?:'ll|\s+will|\s+may|\s+might)\b"
    r"|\bstuck\s+in\s+traffic\b"
)
_ENGLISH_ARRIVAL_WITH_TIME = re.compile(
    r"\b(?:arrive|reach|get\s+to\s+work|punch\s+in|clock\s+in)\b.{0,28}"
    r"[0-9\u0660-\u0669]{1,2}:[0-9\u0660-\u0669]{2}\b"
)
_ENGLISH_COMPLETED_LATE_PUNCH = (
    re.compile(
        r"\b(?:i|we)\s+(?:(?:have|'ve)\s+)?(?:already\s+)?"
        r"(?:punched|clocked|checked)\s+in\b.{0,32}\b(?:late|delayed?)\b"
    ),
    re.compile(
        r"\b(?:i|we)\s+(?:arrived|reached|got\s+to\s+work)\s+late\b.{0,40}"
        r"\b(?:already\s+)?(?:punched|clocked|checked)\s+in\b"
    ),
)
_ENGLISH_IN_DIRECTION = (
    re.compile(r"\bpunch(?:ed|ing)?\s+in\b"),
    re.compile(r"\bin\s+punch\b"),
)
_ENGLISH_OUT_DIRECTION = (
    re.compile(r"\bpunch(?:ed|ing)?\s+out\b"),
    re.compile(r"\bout\s+punch\b"),
)
_ARABIC_DIACRITICS = re.compile(r"[\u064b-\u065f\u0670\u0640]")


def _normalized(message: str) -> str:
    return " ".join(message.casefold().replace("’", "'").split())


def _canonical_arabic(value: str) -> str:
    return value.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}))


def _has_explicit_correction_intent(normalized: str) -> bool:
    if any(pattern.search(normalized) for pattern in _ENGLISH_CORRECTION_PATTERNS):
        return True

    arabic = _ARABIC_DIACRITICS.sub("", normalized)
    canonical_arabic = _canonical_arabic(arabic)
    has_directional_registration = (
        any(cue in canonical_arabic for cue in ("اسجل", "سجل"))
        and any(cue in canonical_arabic for cue in ("دخول", "خروج", "انصراف"))
    )
    has_punch_term = (
        "بصم" in arabic
        or "الحضور" in arabic
        or has_directional_registration
    )
    forgot_or_missing = any(
        cue in arabic
        for cue in ("نسيت", "ما بصمت", "لم ابصم", "ناقص", "مفقود")
    )
    correction_request = any(
        cue in arabic
        for cue in ("صحح", "تصحيح", "عدل", "تعديل")
    )
    return has_punch_term and (forgot_or_missing or correction_request)


def is_ambiguous_transactional_utterance(message: str) -> bool:
    """Recognize an incomplete forgotten-action utterance without guessing its intent."""

    normalized = _normalized(message)
    if not normalized or _has_explicit_correction_intent(normalized):
        return False

    if re.search(r"\b(?:forgot|forget)\s+to\b", normalized):
        has_time_context = bool(re.search(r"\b(?:today|yesterday|tonight)\b", normalized))
        supported_object = bool(
            re.search(
                r"\b(?:punch|clock|check|leave|vacation|request|notification|"
                r"approve|reject|cancel)\b",
                normalized,
            )
        )
        return has_time_context and not supported_object

    arabic = _canonical_arabic(_ARABIC_DIACRITICS.sub("", normalized))
    has_forgotten_cue = "نسيت" in arabic
    has_time_context = any(cue in arabic for cue in ("اليوم", "امس", "البارح"))
    supported_object = any(
        cue in arabic
        for cue in (
            "دخول",
            "خروج",
            "انصراف",
            "بصم",
            "اجاز",
            "طلب",
            "اشعار",
            "موافق",
            "رفض",
            "الغاء",
        )
    )
    return has_forgotten_cue and has_time_context and not supported_object


def is_explicit_missing_punch_correction(message: str) -> bool:
    """Return whether the user explicitly describes an existing punch problem."""

    normalized = _normalized(message)
    return bool(normalized) and _has_explicit_correction_intent(normalized)


def explicit_missing_punch_direction(message: str) -> str | None:
    """Ground one explicit IN/OUT direction without guessing implicit time-of-day."""

    normalized = _normalized(message)
    if not normalized or not _has_explicit_correction_intent(normalized):
        return None

    arabic = _ARABIC_DIACRITICS.sub("", normalized)
    has_in = any(pattern.search(normalized) for pattern in _ENGLISH_IN_DIRECTION) or (
        "دخول" in arabic
    )
    has_out = any(pattern.search(normalized) for pattern in _ENGLISH_OUT_DIRECTION) or any(
        cue in arabic for cue in ("خروج", "انصراف")
    )
    if has_in == has_out:
        return None
    return "IN" if has_in else "OUT"


def is_prospective_late_arrival(message: str) -> bool:
    """Distinguish expected arrival lateness from an existing punch correction."""

    normalized = _normalized(message)
    if not normalized or _has_explicit_correction_intent(normalized):
        return False

    has_english_late_context = bool(_ENGLISH_LATE_CONTEXT.search(normalized))
    if _ENGLISH_PROSPECTIVE.search(normalized) and (
        has_english_late_context or _ENGLISH_ARRIVAL_WITH_TIME.search(normalized)
    ):
        return True

    arabic = _ARABIC_DIACRITICS.sub("", normalized)
    has_arabic_late_context = any(
        cue in arabic
        for cue in ("اتاخر", "أتأخر", "بتاخر", "بأتأخر", "متاخر", "متأخر", "تأخير")
    )
    has_arabic_prospective_context = any(
        cue in arabic
        for cue in (
            "راح",
            "سوف",
            "يمكن",
            "ممكن",
            "احتمال",
            "غالب",
            "بتاخر",
            "بأتأخر",
            "حاتاخر",
            "حأتأخر",
            "بوصل",
            "ساصل",
            "سأصل",
        )
    )
    return has_arabic_late_context and has_arabic_prospective_context


def is_existing_late_punch(message: str) -> bool:
    """Recognize a completed late IN punch without treating it as missing."""

    normalized = _normalized(message)
    if not normalized or _has_explicit_correction_intent(normalized):
        return False
    if any(pattern.search(normalized) for pattern in _ENGLISH_COMPLETED_LATE_PUNCH):
        return True

    arabic = _ARABIC_DIACRITICS.sub("", normalized)
    has_completed_punch = "بصمت" in arabic or "سجلت دخول" in arabic
    has_late_context = any(
        cue in arabic
        for cue in ("متاخر", "متأخر", "تاخرت", "تأخرت")
    )
    return has_completed_punch and has_late_context


def late_arrival_unavailable_message(
    language: str,
    *,
    already_punched: bool = False,
) -> str:
    if already_punched:
        if language == "ar":
            return (
                "فاهم عليك—أنت بصمت دخول متأخر. هذه ليست حالة بصمة مفقودة. مسار "
                "التأخير أو الرصيد المسموح غير مرتبط حاليًا، لذلك ما أقدر أتحقق إذا "
                "كان يلزم تعديل أو موافقة."
            )
        return (
            "I understand—you already punched in late. This isn't a missing-punch "
            "correction. The late-arrival or buffer workflow isn't connected yet, so "
            "I can't check whether any adjustment or approval is required."
        )
    if language == "ar":
        return (
            "فاهم عليك—تتوقع تتأخر عن الدوام. خدمة طلب التأخير أو الرصيد المسموح "
            "غير مرتبطة حاليًا، لذلك ما أقدر أرسل الطلب الآن."
        )
    return (
        "I understand—you expect to be late. The late-arrival or buffer request "
        "service isn't connected yet, so I can't submit that request at the moment."
    )
