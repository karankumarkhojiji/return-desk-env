from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple


def _norm(value: Optional[str]) -> str:
    return (value or "").strip().lower()


def exact_match(predicted: Optional[str], gold: Optional[str]) -> float:
    if gold is None:
        return 1.0
    return 1.0 if _norm(predicted) == _norm(gold) else 0.0


def tag_f1(predicted_tags: Sequence[str], gold_tags: Sequence[str]) -> float:
    pred = {_norm(tag) for tag in predicted_tags if _norm(tag)}
    gold = {_norm(tag) for tag in gold_tags if _norm(tag)}
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    precision = tp / len(pred)
    recall = tp / len(gold)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def average_exact_matches(
    predicted: Mapping[str, str],
    gold: Mapping[str, str],
) -> float:
    if not gold:
        return 1.0
    total = 0.0
    for item_id, gold_value in gold.items():
        total += exact_match(predicted.get(item_id), gold_value)
    return total / max(len(gold), 1)


def inspection_coverage(seen_sections: Iterable[str], required_sections: Sequence[str]) -> float:
    required = {_norm(section) for section in required_sections}
    if not required:
        return 1.0
    seen = {_norm(section) for section in seen_sections}
    return len(required & seen) / len(required)


def reply_slot_coverage(message: str, requirements: Mapping[str, Sequence[str]]) -> float:
    if not requirements:
        return 1.0
    text = _norm(message)
    if not text:
        return 0.0
    satisfied = 0
    for phrases in requirements.values():
        if any(_norm(phrase) in text for phrase in phrases):
            satisfied += 1
    return satisfied / max(len(requirements), 1)


def policy_compliance(
    predicted_item_resolutions: Mapping[str, str],
    predicted_ticket_resolution: Optional[str],
    allowed_item_resolutions: Mapping[str, Sequence[str]],
    allowed_ticket_resolutions: Sequence[str],
) -> float:
    item_scores = []
    for item_id, allowed in allowed_item_resolutions.items():
        pred = _norm(predicted_item_resolutions.get(item_id))
        allowed_set = {_norm(value) for value in allowed}
        item_scores.append(1.0 if pred in allowed_set and pred else 0.0)

    ticket_allowed = {_norm(value) for value in allowed_ticket_resolutions}
    ticket_score = 1.0 if _norm(predicted_ticket_resolution) in ticket_allowed and predicted_ticket_resolution else 0.0

    if not item_scores:
        return ticket_score
    return 0.7 * (sum(item_scores) / len(item_scores)) + 0.3 * ticket_score


def grade_submission(
    task: Dict[str, Any],
    seen_sections: Sequence[str],
    current_priority: Optional[str],
    current_tags: Sequence[str],
    item_resolutions: Mapping[str, str],
    ticket_resolution: Optional[str],
    drafted_reply: str,
) -> Tuple[float, Dict[str, float]]:
    gold = task["gold"]

    breakdown = {
        "item_resolution_accuracy": average_exact_matches(item_resolutions, gold["item_resolutions"]),
        "ticket_resolution_accuracy": exact_match(ticket_resolution, gold["ticket_resolution"]),
        "priority_accuracy": exact_match(current_priority, gold["priority"]),
        "tag_quality": tag_f1(current_tags, gold["tags"]),
        "evidence_coverage": inspection_coverage(seen_sections, gold["required_sections"]),
        "policy_compliance": policy_compliance(
            predicted_item_resolutions=item_resolutions,
            predicted_ticket_resolution=ticket_resolution,
            allowed_item_resolutions=gold["allowed_item_resolutions"],
            allowed_ticket_resolutions=gold["allowed_ticket_resolutions"],
        ),
        "reply_quality": reply_slot_coverage(drafted_reply, gold["reply_requirements"]),
    }

    weights = {
        "item_resolution_accuracy": 0.30,
        "ticket_resolution_accuracy": 0.15,
        "priority_accuracy": 0.10,
        "tag_quality": 0.10,
        "evidence_coverage": 0.10,
        "policy_compliance": 0.10,
        "reply_quality": 0.15,
    }
    score = sum(breakdown[key] * weights[key] for key in weights)
    score = max(0.0, min(1.0, round(score, 6)))
    rounded_breakdown = {key: round(value, 6) for key, value in breakdown.items()}
    return score, rounded_breakdown


def feedback_from_breakdown(breakdown: Mapping[str, float]) -> str:
    misses = []
    if breakdown.get("item_resolution_accuracy", 1.0) < 1.0:
        misses.append("item-level resolutions need correction")
    if breakdown.get("ticket_resolution_accuracy", 1.0) < 1.0:
        misses.append("overall ticket resolution is incorrect or missing")
    if breakdown.get("priority_accuracy", 1.0) < 1.0:
        misses.append("priority is incorrect or missing")
    if breakdown.get("tag_quality", 1.0) < 1.0:
        misses.append("tags are incomplete or noisy")
    if breakdown.get("evidence_coverage", 1.0) < 1.0:
        misses.append("required evidence was not fully inspected")
    if breakdown.get("policy_compliance", 1.0) < 1.0:
        misses.append("one or more decisions are not policy-compliant")
    if breakdown.get("reply_quality", 1.0) < 1.0:
        misses.append("reply is missing required content")
    if not misses:
        return "Submission is fully correct and policy-compliant."
    return "; ".join(misses).capitalize() + "."
