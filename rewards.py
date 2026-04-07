from __future__ import annotations

from typing import Mapping, Optional, Sequence

try:
    from .graders import reply_slot_coverage
except ImportError:  # pragma: no cover
    from graders import reply_slot_coverage


def inspection_reward(section: str, seen_sections: Sequence[str], required_sections: Sequence[str]) -> float:
    if section in seen_sections:
        return -0.02
    return 0.05 if section in required_sections else 0.01


def categorical_delta_reward(old_value: Optional[str], new_value: Optional[str], gold_value: Optional[str], positive: float, negative: float) -> float:
    old_correct = 1.0 if (old_value or "").strip().lower() == (gold_value or "").strip().lower() and old_value else 0.0
    new_correct = 1.0 if (new_value or "").strip().lower() == (gold_value or "").strip().lower() and new_value else 0.0
    if old_value == new_value and new_value is not None:
        return -0.01
    if new_correct > old_correct:
        return positive
    if new_correct < old_correct:
        return negative
    return -0.02 if new_value else 0.0


def tag_reward(tag: str, existing_tags: Sequence[str], gold_tags: Sequence[str]) -> float:
    if tag in existing_tags:
        return -0.02
    return 0.03 if tag in gold_tags else -0.01


def item_resolution_reward(item_id: str, old_value: Optional[str], new_value: Optional[str], gold_item_resolutions: Mapping[str, str]) -> float:
    gold_value = gold_item_resolutions.get(item_id)
    return categorical_delta_reward(old_value, new_value, gold_value, positive=0.08, negative=-0.06)


def ticket_resolution_reward(old_value: Optional[str], new_value: Optional[str], gold_value: Optional[str]) -> float:
    return categorical_delta_reward(old_value, new_value, gold_value, positive=0.07, negative=-0.05)


def reply_reward(old_message: str, new_message: str, reply_requirements: Mapping[str, Sequence[str]]) -> float:
    old_score = reply_slot_coverage(old_message, reply_requirements)
    new_score = reply_slot_coverage(new_message, reply_requirements)
    delta = new_score - old_score
    if delta > 0:
        return round(0.10 * delta, 6)
    if new_message.strip() == old_message.strip():
        return -0.01
    return -0.01


def submit_reward(final_score: float, evidence_coverage: float) -> float:
    premature_penalty = 0.10 if evidence_coverage < 1.0 else 0.0
    return max(0.0, round(final_score - premature_penalty, 6))
