import pytest

from app.speech.text import markdown_to_speech_text


def test_markdown_profile_becomes_natural_speech_without_formatting() -> None:
    markdown = """### Contact Information

- **Employee Number:** UN003
- **Name:** Saneesh
- **Email:** saneesh.netsoftpro@gmail.com
- Started: `09/12/2013`
"""

    speech = markdown_to_speech_text(markdown)

    assert speech == (
        "Contact Information. Employee Number: UN003. Name: Saneesh. "
        "Email: saneesh.netsoftpro@gmail.com. Started: 09/12/2013."
    )
    assert "###" not in speech
    assert "**" not in speech
    assert "`" not in speech
    assert "- " not in speech


def test_markdown_link_uses_label_without_speaking_destination() -> None:
    speech = markdown_to_speech_text(
        "Open the [ResourcePlus Portal](https://example.com/internal/path)."
    )

    assert speech == "Open the ResourcePlus Portal."
    assert "https://" not in speech
    assert "example.com" not in speech


def test_arabic_markdown_is_cleaned_without_translation_or_value_changes() -> None:
    markdown = """### معلومات الموظف
- **الاسم:** سنيش
- **الرقم:** UN003
- **الرصيد:** 12.5
"""

    speech = markdown_to_speech_text(markdown)

    assert speech == "معلومات الموظف. الاسم: سنيش. الرقم: UN003. الرصيد: 12.5."
    assert "###" not in speech
    assert "**" not in speech
    assert "- " not in speech


@pytest.mark.parametrize(
    "plain_text",
    [
        "Your profile is ready.",
        "هذا ملخص ملفك.",
        "Balance: 8.5 days.",
    ],
)
def test_plain_response_content_remains_unchanged(plain_text: str) -> None:
    assert markdown_to_speech_text(plain_text) == plain_text
