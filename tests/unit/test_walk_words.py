"""Regression table for data/walk_words.yaml: benign phrases must pass the blocklist and
time-of-day checks exactly as hub.safety compiles them (FR-M6b), true positives stay caught."""

from __future__ import annotations

import pytest

from hub.safety import _blocked, check_fallback_text, mentions_time_of_day

SAFE = [
    "Notice the rush of the wind.",
    "Watch the squirrels jump between the branches.",
    "Take your time and listen for one bird.",
    "Notice a stream of people near the gate.",
    "Watch the birds feed near the trees.",
    "Walk along the lane and listen for a bird.",
    "Look into the garden as you pass.",
    "Smell the night jasmine as you pass.",
    "Listen for a bell that baje.",
    "Notice the sham of the old signboard.",
]

# Safe imperatives that also pass the full fallback filter (you/your, allowed verbs and nouns).
SAFE_FULL = [
    "Look into the garden as you pass.",
]

BLOCKED_PHRASES = [
    "Give the puppy a biscuit.",
    "Pet the dog.",
    "Stroke it.",
    "Follow the cows.",
    "Ask the watchman.",
    "Walk into the society garden.",
    "Climb the wall.",
    "Jog to the corner.",
    "Take a photo.",
    "Bring your phone.",
    "Cross the road.",
    "Feed the birds.",
    "Give the birds some grain.",
    "Run to the gate.",
    "Race to the corner.",
    "Jump over the puddle.",
    "Go into the compound.",
    "Enter the temple garden.",
    "Record the koel.",
    "Use your phone.",
    "Check your phone.",
    "Stop on the road.",
    "Walk on the road.",
    "Jump up and down.",
    "You jump.",
    "Jump!",
    "Rush to the gate.",
    "Stream it live.",
    "Check the time.",
    "Walk in the middle of the road.",
    "Stand in the lane.",
    "Go and feed them.",
    # Hinglish
    "Khana dena.",
    "Kutte ko khilana.",
    "Chhoona mat.",
    "Peeche jaana.",
    "Daudna.",
    "Photo lena.",
]

BLOCKED_TIME = [
    "Walk tonight.",
    "Notice the light this evening.",
    "Listen at night.",
    "Listen for birds in the morning.",
    "Look at the afternoon sky.",
    "Watch the sunset.",
    "Be at the gate by 6 baje.",
    "Listen for a bird shaam ko.",
    "Come back sham ko.",
    "Walk this night.",
    "Look at the night sky.",
]


def _rejected(text: str) -> bool:
    return bool(_blocked(text)) or mentions_time_of_day(text)


@pytest.mark.parametrize("text", SAFE)
def test_safe_phrase_not_rejected(text: str) -> None:
    assert _blocked(text) == ()
    assert not mentions_time_of_day(text)


@pytest.mark.parametrize("text", SAFE_FULL)
def test_safe_phrase_passes_full_filter(text: str) -> None:
    result = check_fallback_text(text)
    assert result.passed, result.reasons


@pytest.mark.xfail(strict=True, reason="'take' is not in verb_allowlist (not a blocklist matter)")
def test_take_your_time_full_filter() -> None:
    assert check_fallback_text("Take your time and listen for one bird.").passed


@pytest.mark.parametrize("text", BLOCKED_PHRASES)
def test_blocked_phrase_still_caught(text: str) -> None:
    assert _blocked(text), text


@pytest.mark.parametrize("text", BLOCKED_TIME)
def test_time_phrase_still_caught(text: str) -> None:
    assert _rejected(text), text
