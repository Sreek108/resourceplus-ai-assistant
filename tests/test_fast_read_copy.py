import pytest

from app.models.schemas import BlockColumn, ListBlock, TableBlock
from app.services.chat import _response_message
from app.services.fast_reads import _balance_message, _message_for


def _table(row_count: int) -> TableBlock:
    return TableBlock(
        title="Trusted data",
        columns=[BlockColumn(key="value", label="Value")],
        rows=[{"value": index} for index in range(row_count)],
    )


@pytest.mark.parametrize(
    ("intent", "blocks", "expected"),
    [
        ("attendance", [_table(2)], "Here's your attendance."),
        (
            "attendance",
            [_table(0)],
            "You don't have any attendance records for this period.",
        ),
        (
            "missing_punches",
            [_table(2)],
            "I found 2 missing-punch records for this period.",
        ),
        (
            "missing_punches",
            [_table(0)],
            "You don't have any missing punches for this period.",
        ),
        (
            "notifications",
            [ListBlock(title="Notifications", items=["One", "Two", "Three"])],
            "You have 3 notifications.",
        ),
        ("requests", [_table(2)], "You have 2 requests for this period."),
        (
            "approvals",
            [_table(5)],
            "You have 5 requests waiting for your approval.",
        ),
        (
            "approvals",
            [_table(0)],
            "You're all caught up — there's nothing waiting for your approval.",
        ),
    ],
)
def test_english_fast_read_copy_is_personal_and_grounded(
    intent: str,
    blocks: list[TableBlock | ListBlock],
    expected: str,
) -> None:
    display, speech = _message_for(intent, blocks, "en")

    assert display.startswith(expected)
    assert speech == expected
    assert "ResourcePlus shows" not in display


@pytest.mark.parametrize(
    ("intent", "blocks", "expected"),
    [
        ("attendance", [_table(2)], "إليك سجل حضورك خلال هذه الفترة."),
        (
            "attendance",
            [_table(0)],
            "لا توجد لديك سجلات حضور خلال هذه الفترة.",
        ),
        (
            "missing_punches",
            [_table(0)],
            "لا توجد لديك بصمات مفقودة خلال هذه الفترة.",
        ),
        (
            "notifications",
            [ListBlock(title="الإشعارات", items=["واحد", "اثنان", "ثلاثة"])],
            "لديك 3 إشعارات.",
        ),
        ("requests", [_table(2)], "لديك طلبان خلال هذه الفترة."),
        (
            "approvals",
            [_table(5)],
            "لديك 5 طلبات بانتظار موافقتك.",
        ),
        (
            "approvals",
            [_table(0)],
            "أمورك تمام — ما فيه طلبات تنتظر موافقتك.",
        ),
    ],
)
def test_arabic_fast_read_copy_is_personal_and_grounded(
    intent: str,
    blocks: list[TableBlock | ListBlock],
    expected: str,
) -> None:
    display, speech = _message_for(intent, blocks, "ar")

    assert display.startswith(expected)
    assert speech == expected
    assert "ResourcePlus" not in display


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ([False], "I found 1 missing-punch record for this period, but none are currently eligible for correction."),
        ([False, False], "I found 2 missing-punch records for this period, but none are currently eligible for correction."),
        ([True], "You have 1 missing punch available for correction."),
        ([True, True, True], "You have 3 missing punches available for correction."),
        ([True, False, False], "You have 3 missing-punch records for this period. 1 is currently available for correction."),
        ([True, True, False], "You have 3 missing-punch records for this period. 2 are currently available for correction."),
    ],
)
def test_missing_punch_copy_uses_trusted_correction_flags(flags, expected) -> None:
    table = TableBlock(
        title="Missing punches",
        columns=[BlockColumn(key="correctable", label="Correctable")],
        rows=[{"correctable": flag} for flag in flags],
    )
    display, speech = _message_for("missing_punches", [table], "en")
    assert speech == expected
    assert display.startswith(expected)


@pytest.mark.parametrize(
    ("flags", "expected_fragment"),
    [
        ([False, False], "لا توجد أي بصمة متاحة للتصحيح"),
        ([True], "بصمة مفقودة واحدة متاحة للتصحيح"),
        ([True, False], "1 منها متاحة للتصحيح"),
    ],
)
def test_arabic_missing_punch_copy_uses_trusted_correction_flags(
    flags, expected_fragment
) -> None:
    table = TableBlock(
        title="البصمات المفقودة",
        columns=[BlockColumn(key="correctable", label="قابل للتصحيح")],
        rows=[{"correctable": flag} for flag in flags],
    )
    display, speech = _message_for("missing_punches", [table], "ar")
    assert expected_fragment in speech
    assert display.startswith(speech)


@pytest.mark.parametrize(
    ("language", "expected"),
    [
        ("en", "You have 120 minutes of buffer time left."),
        ("ar", "باقي لك 120 دقيقة من وقت السماح."),
    ],
)
def test_balance_copy_preserves_authoritative_value_without_decimal_padding(
    language: str,
    expected: str,
) -> None:
    payload = {"hasPolicy": True, "limitType": 2, "remaining": 120.0}

    display, speech = _balance_message(payload, language)

    assert display == expected
    assert speech == expected


def test_success_fallback_is_personal_while_failure_keeps_source_attribution() -> None:
    assert _response_message({"success": True}) == (
        True,
        "Your request was completed.",
    )
    assert _response_message({"success": False}) == (
        False,
        "ResourcePlus did not confirm the action.",
    )
