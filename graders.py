from __future__ import annotations

import math
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

# ---------------------------------------------------------------------------
# Score range constants — strictly inside (0, 1) per validator requirements
# ---------------------------------------------------------------------------
SCORE_MIN = 0.01
SCORE_MAX = 0.99

# ---------------------------------------------------------------------------
# Resolution cost model (Repo 3 inspiration — multi-objective trade-off)
# Lower cost = more cost-efficient for the business.
# When multiple resolutions are allowed, the grader rewards the cheaper one.
# ---------------------------------------------------------------------------
RESOLUTION_COSTS: Dict[str, float] = {
    "refund":         1.00,   # Full financial cost
    "exchange":       0.75,   # Ship new item + return logistics
    "partial_refund": 0.60,   # Partial cost
    "store_credit":   0.45,   # Deferred, likely < full redemption
    "escalate":       0.30,   # Specialist time cost
    "request_info":   0.10,   # Near free
    "deny":           0.00,   # No direct cost
}

# ---------------------------------------------------------------------------
# Grading weights — must sum to exactly 1.00
# ---------------------------------------------------------------------------
W_ITEM_RESOLUTION   = 0.22   # Did each item get the right resolution?
W_TICKET_RESOLUTION = 0.13   # Is the overall ticket decision correct?
W_PRIORITY          = 0.08   # Was urgency level correctly assessed?
W_TAG_QUALITY       = 0.10   # F1 between predicted and gold tags
W_EVIDENCE_COVERAGE = 0.10   # Were required sections inspected?
W_POLICY_COMPLIANCE = 0.12   # Are decisions within allowed resolution set?
W_REPLY_QUALITY     = 0.10   # Does reply cover all required information slots?
W_EFFICIENCY        = 0.08   # Fewer steps used = higher score
W_COST_EFFICIENCY   = 0.07   # Multi-objective: choose the most cost-efficient valid resolution
# Total: 0.22+0.13+0.08+0.10+0.10+0.12+0.10+0.08+0.07 = 1.00

# Fraud penalty: processing a refund on a confirmed fraud-risk case
FRAUD_REFUND_PENALTY = 0.40


def _clamp(score: float) -> float:
    """Clamp score to the open interval (0.01, 0.99) required by hackathon validators."""
    if score is None or (isinstance(score, float) and math.isnan(score)):
        return SCORE_MIN
    return round(min(SCORE_MAX, max(SCORE_MIN, float(score))), 6)


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
    """Original keyword-matching reply score. Used for per-step reward shaping in rewards.py."""
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


def reply_semantic_similarity(message: str, requirements: Mapping[str, Sequence[str]]) -> float:
    """
    Semantic word-overlap (Jaccard) reply quality score — from Repo 8 pattern.

    Improvement over pure keyword matching:
    - Exact phrase match always gives full credit for a slot.
    - Partial word overlap gives graduated partial credit.
    - An agent writing 'the amount you paid' gets ~0.5 credit for the slot
      that requires 'amount paid', rather than 0.

    This makes the grader fairer for agents that paraphrase correctly.
    """
    if not requirements:
        return 1.0
    text = _norm(message)
    if not text:
        return 0.0
    text_words = set(text.split())
    satisfied = 0.0

    for phrases in requirements.values():
        # First try exact phrase match — full credit
        if any(_norm(phrase) in text for phrase in phrases):
            satisfied += 1.0
            continue
        # Fall back to word-overlap (Jaccard) — partial credit
        best_overlap = 0.0
        for phrase in phrases:
            phrase_words = set(_norm(phrase).split())
            if not phrase_words:
                continue
            intersection = len(text_words & phrase_words)
            union = len(text_words | phrase_words)
            overlap = intersection / union if union > 0 else 0.0
            best_overlap = max(best_overlap, overlap)
        # Scale: Jaccard >= 0.25 → full credit, 0 → 0 credit
        satisfied += min(1.0, best_overlap * 4.0)

    return round(satisfied / max(len(requirements), 1), 4)


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
    ticket_score = (
        1.0 if _norm(predicted_ticket_resolution) in ticket_allowed and predicted_ticket_resolution else 0.0
    )

    if not item_scores:
        return ticket_score
    return 0.7 * (sum(item_scores) / len(item_scores)) + 0.3 * ticket_score


def efficiency_score(step_count: int, max_steps: int) -> float:
    """Reward agents that solve tasks in fewer steps. Full score at <=60% steps used."""
    if max_steps <= 0:
        return 0.0
    ratio = step_count / max_steps
    if ratio <= 0.60:
        return 1.0
    if ratio >= 1.0:
        return 0.0
    return round(1.0 - (ratio - 0.60) / 0.40, 4)


def cost_efficiency_score(
    ticket_resolution: Optional[str],
    allowed_ticket_resolutions: Sequence[str],
) -> float:
    """
    Multi-objective cost-efficiency score — from Repo 3 (SafeSpace-RL) inspiration.

    Models the real business trade-off: refunds are good for customers but costly
    for the company. When multiple valid resolutions exist, the grader rewards the
    more cost-efficient choice.

    Examples:
    - allowed=["refund", "exchange"], chose "exchange"  → higher score (cheaper)
    - allowed=["refund", "store_credit"], chose "refund" → lower score (overly generous)
    - allowed=["refund"] only, chose "refund"            → full score (no cheaper option)
    - allowed=["deny"], chose "deny"                     → full score (correct & cheapest)
    - chose something not in allowed                     → 0.0

    This means:
    - Agents that blindly refund every ticket score lower than agents that
      read inventory or policy and choose the cheaper-but-still-valid option.
    - Agents that deny when they should refund also score 0.0.
    """
    if not ticket_resolution or not allowed_ticket_resolutions:
        return 0.5  # neutral when no resolution set yet

    pred = _norm(ticket_resolution)
    allowed_norms = [_norm(r) for r in allowed_ticket_resolutions]

    if pred not in allowed_norms:
        return 0.0  # invalid resolution — policy_compliance will also penalize

    # Single allowed option → always full score (no trade-off to make)
    if len(allowed_norms) == 1:
        return 1.0

    # Multiple allowed options are present — reward cost efficiency
    pred_cost = RESOLUTION_COSTS.get(pred, 0.5)
    costs = [RESOLUTION_COSTS.get(r, 0.5) for r in allowed_norms]
    min_cost = min(costs)
    max_cost = max(costs)

    if max_cost == min_cost:
        return 1.0  # all options same cost — no meaningful choice

    # Full credit for choosing the cheapest; scaled down for more expensive choices
    return round(1.0 - (pred_cost - min_cost) / (max_cost - min_cost), 4)


def grade_submission(
    task: Dict[str, Any],
    seen_sections: Sequence[str],
    current_priority: Optional[str],
    current_tags: Sequence[str],
    item_resolutions: Mapping[str, str],
    ticket_resolution: Optional[str],
    drafted_reply: str,
    step_count: int = 0,
    fraud_flagged: bool = False,
) -> Tuple[float, Dict[str, float]]:
    gold = task["gold"]
    max_steps = task.get("max_steps", 10)

    # Core component scores
    item_res_acc    = average_exact_matches(item_resolutions, gold["item_resolutions"])
    ticket_res_acc  = exact_match(ticket_resolution, gold["ticket_resolution"])
    priority_acc    = exact_match(current_priority, gold["priority"])
    tag_qual        = tag_f1(current_tags, gold["tags"])
    evidence_cov    = inspection_coverage(seen_sections, gold["required_sections"])
    pol_comp        = policy_compliance(
        predicted_item_resolutions=item_resolutions,
        predicted_ticket_resolution=ticket_resolution,
        allowed_item_resolutions=gold["allowed_item_resolutions"],
        allowed_ticket_resolutions=gold["allowed_ticket_resolutions"],
    )
    # Semantic reply quality — word-overlap based (Repo 8 pattern)
    reply_qual      = reply_semantic_similarity(drafted_reply, gold["reply_requirements"])
    step_eff        = efficiency_score(step_count, max_steps)
    # Multi-objective cost efficiency (Repo 3 pattern)
    cost_eff        = cost_efficiency_score(ticket_resolution, gold["allowed_ticket_resolutions"])

    breakdown = {
        "item_resolution_accuracy":    round(item_res_acc, 6),
        "ticket_resolution_accuracy":  round(ticket_res_acc, 6),
        "priority_accuracy":           round(priority_acc, 6),
        "tag_quality":                 round(tag_qual, 6),
        "evidence_coverage":           round(evidence_cov, 6),
        "policy_compliance":           round(pol_comp, 6),
        "reply_quality":               round(reply_qual, 6),
        "efficiency":                  round(step_eff, 6),
        "cost_efficiency":             round(cost_eff, 6),
    }

    raw_score = (
        breakdown["item_resolution_accuracy"]   * W_ITEM_RESOLUTION
        + breakdown["ticket_resolution_accuracy"] * W_TICKET_RESOLUTION
        + breakdown["priority_accuracy"]          * W_PRIORITY
        + breakdown["tag_quality"]                * W_TAG_QUALITY
        + breakdown["evidence_coverage"]          * W_EVIDENCE_COVERAGE
        + breakdown["policy_compliance"]          * W_POLICY_COMPLIANCE
        + breakdown["reply_quality"]              * W_REPLY_QUALITY
        + breakdown["efficiency"]                 * W_EFFICIENCY
        + breakdown["cost_efficiency"]            * W_COST_EFFICIENCY
    )

    # FRAUD PENALTY — processing a refund on a known fraud-risk ticket is a severe error
    is_fraud_task = gold.get("fraud_risk", False)
    if is_fraud_task:
        agent_refunded = _norm(ticket_resolution) == "refund" or any(
            _norm(v) == "refund" for v in item_resolutions.values()
        )
        if agent_refunded:
            raw_score -= FRAUD_REFUND_PENALTY
            breakdown["fraud_penalty"] = -FRAUD_REFUND_PENALTY
        elif not fraud_flagged:
            # Didn't refund but also didn't flag it — partial penalty
            raw_score -= FRAUD_REFUND_PENALTY * 0.20
            breakdown["fraud_no_flag_penalty"] = round(-FRAUD_REFUND_PENALTY * 0.20, 6)

    final = _clamp(raw_score)
    return final, breakdown


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
    if breakdown.get("reply_quality", 1.0) < 0.8:
        misses.append("reply is missing required content")
    if breakdown.get("cost_efficiency", 1.0) < 0.5:
        misses.append("a more cost-efficient resolution was available (check policy for alternatives)")
    if breakdown.get("fraud_penalty") is not None:
        misses.append("SEVERE: refund was processed on a fraud-risk order (-0.40 penalty)")
    if breakdown.get("fraud_no_flag_penalty") is not None:
        misses.append("fraud risk was not flagged before decision")
    if not misses:
        return "Submission is fully correct and policy-compliant."
    return "; ".join(misses).capitalize() + "."
