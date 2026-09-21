import re

from markdown_it import MarkdownIt
from markdown_it.token import Token


_MARKDOWN = MarkdownIt("commonmark", {"html": False})
_WHITESPACE = re.compile(r"\s+")
_SENTENCE_ENDINGS = frozenset(".!?؟؛:")


def _inline_text(children: list[Token] | None) -> str:
    """Extract only the human-readable content from parsed inline tokens."""
    parts: list[str] = []
    for token in children or []:
        if token.type in {"text", "code_inline"}:
            parts.append(token.content)
        elif token.type == "image":
            # markdown-it stores image alt text in content; never include src URLs.
            parts.append(token.content)
        elif token.type in {"softbreak", "hardbreak"}:
            parts.append(" ")
        # Formatting and link boundary tokens intentionally contribute no text.
        # Their readable child text is represented by adjacent text tokens.
    return _WHITESPACE.sub(" ", "".join(parts)).strip()


def _as_spoken_phrase(text: str) -> str:
    if not text or text[-1] in _SENTENCE_ENDINGS:
        return text
    return f"{text}."


def markdown_to_speech_text(text: str) -> str:
    """Convert assistant display Markdown to plain, natural speech text.

    Markdown is parsed structurally, so emphasis, heading, list, code, and link
    syntax is removed without changing the readable English, Arabic, identifiers,
    dates, email addresses, or numeric values contained in those nodes.
    """
    spoken_blocks: list[str] = []
    for token in _MARKDOWN.parse(text):
        if token.type == "inline":
            content = _inline_text(token.children)
        elif token.type in {"fence", "code_block"}:
            content = _WHITESPACE.sub(" ", token.content).strip()
        else:
            continue
        if content:
            spoken_blocks.append(_as_spoken_phrase(content))
    return " ".join(spoken_blocks).strip()
