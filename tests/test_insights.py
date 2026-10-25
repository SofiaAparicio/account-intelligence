import os
import re

import pytest

from src.data import load_accounts
from src.insights import NotConfigured, build_prompt, generate_brief, get_brief, list_briefed, validate_brief
from src.scoring import score_accounts


def test_committed_briefs_are_valid_and_cover_the_top_of_the_list():
    briefed = list_briefed()
    assert len(briefed) >= 20
    top20 = score_accounts(load_accounts()).head(20)["account_id"].tolist()
    assert set(top20) <= set(briefed), "every account in the default top-20 view must have a brief"
    for aid in briefed:
        b = get_brief(aid)
        validate_brief(b["brief"])
        assert b["prompt"] and b["context_card"] and b["generator"]


CALL_SIGNAL = re.compile(r"\b(call|QBR|review|session|said|stated|cited|mentioned|asked|requested|proposed|confirmed|identified|initiated|positioned|value[sd]?)\b", re.I)


def test_briefs_cite_numbers_or_call_signals_not_vibes():
    """Prompt rule: every risk/opportunity names the number or call signal it rests on."""
    for aid in list_briefed():
        brief = get_brief(aid)["brief"]
        for bullet in brief["risks"] + brief["opportunities"]:
            assert re.search(r"\d", bullet) or CALL_SIGNAL.search(bullet), \
                f"{aid}: ungrounded bullet: {bullet}"


def test_prompt_contains_card_and_rules():
    s = score_accounts(load_accounts())
    acc = s.iloc[0].to_dict()
    prompt, card = build_prompt(acc)
    assert card in prompt
    assert "Use ONLY facts from the context card" in prompt
    assert acc["account_name"] in card


def test_live_generation_is_optional():
    os.environ.pop("OLLAMA_MODEL", None)
    with pytest.raises(NotConfigured):
        generate_brief({})
