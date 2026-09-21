from types import SimpleNamespace

import pytest

from app.ai import reason_matcher


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "natural_reason",
    ["family circumstance", "family circumstances", "family issue"],
)
async def test_natural_reason_matches_live_resourceplus_reason(
    monkeypatch,
    natural_reason: str,
) -> None:
    monkeypatch.setattr(
        reason_matcher,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key="", openai_model=""),
    )

    selected = await reason_matcher.match_live_reason(
        natural_reason,
        ["Family Circumstances", "Other", "Work From Home"],
    )

    assert selected == 0


@pytest.mark.asyncio
async def test_ambiguous_live_reasons_do_not_guess_without_clear_semantics(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        reason_matcher,
        "get_settings",
        lambda: SimpleNamespace(openai_api_key="", openai_model=""),
    )

    selected = await reason_matcher.match_live_reason(
        "family issue",
        ["Family Circumstances", "Family Emergency"],
    )

    assert selected is None
