from __future__ import annotations

import pytest

try:
    from return_desk_env.graders import (
        cost_efficiency_score,
        grade_submission,
        reply_semantic_similarity,
        reply_slot_coverage,
        tag_f1,
    )
    from return_desk_env.tasks.catalog import get_task, list_task_ids, task_ids_for_difficulty
except ImportError:  # pragma: no cover
    from graders import (
        cost_efficiency_score,
        grade_submission,
        reply_semantic_similarity,
        reply_slot_coverage,
        tag_f1,
    )
    from tasks.catalog import get_task, list_task_ids, task_ids_for_difficulty


# ---------------------------------------------------------------------------
# tag_f1
# ---------------------------------------------------------------------------

def test_tag_f1_is_perfect_for_exact_match() -> None:
    assert tag_f1(["damaged", "refund_request"], ["damaged", "refund_request"]) == 1.0


def test_tag_f1_is_zero_for_complete_mismatch() -> None:
    assert tag_f1(["vip_exception"], ["damaged", "refund_request"]) == 0.0


def test_tag_f1_partial_overlap() -> None:
    score = tag_f1(["damaged", "wrong_item"], ["damaged", "refund_request"])
    assert 0.0 < score < 1.0


# ---------------------------------------------------------------------------
# reply_slot_coverage (legacy — still used in rewards.py)
# ---------------------------------------------------------------------------

def test_reply_slot_coverage_detects_missing_slots() -> None:
    requirements = {
        "decision": ["refund"],
        "timeline": ["3-5 business days"],
    }
    assert reply_slot_coverage("We will issue a refund.", requirements) == 0.5


def test_reply_slot_coverage_full_when_all_slots_present() -> None:
    requirements = {
        "decision": ["refund"],
        "timeline": ["3-5 business days"],
    }
    assert reply_slot_coverage("We will issue a refund within 3-5 business days.", requirements) == 1.0


# ---------------------------------------------------------------------------
# reply_semantic_similarity  (Repo 8 pattern — partial word-overlap credit)
# ---------------------------------------------------------------------------

def test_semantic_similarity_exact_phrase_scores_full() -> None:
    """Exact phrase match → full credit for that slot."""
    reqs = {"amount": ["amount paid"]}
    score = reply_semantic_similarity("We will refund the amount paid immediately.", reqs)
    assert score == 1.0


def test_semantic_similarity_partial_word_overlap_gives_partial_credit() -> None:
    """Paraphrase ('paid amount') overlaps with 'amount paid' → partial credit, not 0."""
    reqs = {"amount": ["amount paid"]}
    # 'paid amount' has the same words but not as exact substring phrase
    score = reply_semantic_similarity("We will refund the paid amount.", reqs)
    # Should be positive (word overlap) — not 0.0
    assert score > 0.0


def test_semantic_similarity_completely_irrelevant_reply_scores_low() -> None:
    reqs = {"decision": ["refund"], "timeline": ["3-5 business days"]}
    score = reply_semantic_similarity("Hello, goodbye.", reqs)
    assert score < 0.3


def test_semantic_similarity_no_requirements_scores_full() -> None:
    assert reply_semantic_similarity("anything", {}) == 1.0


# ---------------------------------------------------------------------------
# cost_efficiency_score  (Repo 3 multi-objective pattern)
# ---------------------------------------------------------------------------

def test_cost_efficiency_single_allowed_resolution_always_full() -> None:
    """Only one valid option → always 1.0 regardless of cost."""
    assert cost_efficiency_score("refund", ["refund"]) == 1.0
    assert cost_efficiency_score("deny", ["deny"]) == 1.0


def test_cost_efficiency_cheaper_option_scores_higher() -> None:
    """store_credit is cheaper than refund; choosing it should score higher."""
    score_credit = cost_efficiency_score("store_credit", ["refund", "store_credit"])
    score_refund  = cost_efficiency_score("refund",       ["refund", "store_credit"])
    assert score_credit > score_refund


def test_cost_efficiency_deny_is_most_efficient() -> None:
    """deny has cost=0.0, so it should score 1.0 when it's allowed."""
    score = cost_efficiency_score("deny", ["refund", "deny"])
    assert score == 1.0


def test_cost_efficiency_invalid_resolution_scores_zero() -> None:
    """Resolution not in allowed list → 0.0."""
    assert cost_efficiency_score("refund", ["deny", "exchange"]) == 0.0


def test_cost_efficiency_no_resolution_returns_neutral() -> None:
    """No resolution set yet → neutral 0.5."""
    assert cost_efficiency_score(None, ["refund"]) == 0.5


# ---------------------------------------------------------------------------
# grade_submission — unit interval and breakdown completeness
# ---------------------------------------------------------------------------

def test_grade_submission_returns_unit_interval_score() -> None:
    task = get_task("medium_exchange")
    score, breakdown = grade_submission(
        task=task,
        seen_sections=["order", "policy", "inventory"],
        current_priority="medium",
        current_tags=["exchange_request", "inventory_issue", "coupon_order"],
        item_resolutions={"item-1": "store_credit"},
        ticket_resolution="store_credit",
        drafted_reply=(
            "We are sorry the exact blue size L is unavailable. "
            "We can issue store credit for the amount paid, $51.00."
        ),
    )
    assert 0.0 <= score <= 1.0
    assert breakdown["item_resolution_accuracy"] == 1.0


def test_grade_submission_has_all_9_components() -> None:
    """All 9 grading components must be present in the breakdown."""
    task = get_task("easy_refund")
    _, breakdown = grade_submission(
        task=task,
        seen_sections=["order", "policy"],
        current_priority="high",
        current_tags=["damaged", "refund_request"],
        item_resolutions={"item-1": "refund"},
        ticket_resolution="refund",
        drafted_reply="We will process a refund within 3-5 business days.",
        step_count=5,
    )
    expected_keys = {
        "item_resolution_accuracy",
        "ticket_resolution_accuracy",
        "priority_accuracy",
        "tag_quality",
        "evidence_coverage",
        "policy_compliance",
        "reply_quality",
        "efficiency",
        "cost_efficiency",
    }
    assert expected_keys.issubset(breakdown.keys())


def test_fraud_penalty_applied_when_refunding_fraud_order() -> None:
    """Refunding a fraud_risk order must trigger a -0.40 penalty in breakdown."""
    task = get_task("fraud_risk")
    _, breakdown = grade_submission(
        task=task,
        seen_sections=["order", "customer", "policy"],
        current_priority="high",
        current_tags=["refund_request"],
        item_resolutions={"item-1": "refund"},
        ticket_resolution="refund",
        drafted_reply="We will process a full refund.",
        fraud_flagged=False,
    )
    assert "fraud_penalty" in breakdown
    assert breakdown["fraud_penalty"] == -0.40


def test_fraud_no_flag_penalty_when_escalated_but_not_flagged() -> None:
    """Escalating without flagging fraud → smaller penalty (not the full -0.40)."""
    task = get_task("fraud_risk")
    _, breakdown = grade_submission(
        task=task,
        seen_sections=["order", "customer", "policy"],
        current_priority="high",
        current_tags=["escalation_required"],
        item_resolutions={"item-1": "escalate"},
        ticket_resolution="escalate",
        drafted_reply="We have escalated this ticket for investigation.",
        fraud_flagged=False,
    )
    assert "fraud_no_flag_penalty" in breakdown
    assert "fraud_penalty" not in breakdown


def test_no_fraud_penalty_when_correctly_escalated_and_flagged() -> None:
    """Correct fraud resolution: escalate + fraud_flagged=True → no penalty at all."""
    task = get_task("fraud_risk")
    _, breakdown = grade_submission(
        task=task,
        seen_sections=["order", "customer", "policy"],
        current_priority="high",
        current_tags=["fraud_flag", "escalation_required"],
        item_resolutions={"item-1": "escalate"},
        ticket_resolution="escalate",
        drafted_reply="We have flagged and escalated this ticket for fraud investigation.",
        fraud_flagged=True,
    )
    assert "fraud_penalty" not in breakdown
    assert "fraud_no_flag_penalty" not in breakdown


# ---------------------------------------------------------------------------
# catalog — 4 difficulty tiers, 7 tasks
# ---------------------------------------------------------------------------

def test_all_7_tasks_are_registered() -> None:
    all_ids = list_task_ids()
    assert len(all_ids) == 7
    expected = {
        "easy_refund", "medium_exchange", "hard_partial_resolution",
        "expired_return", "fraud_risk", "wrong_item_sent", "extreme_chargeback",
    }
    assert set(all_ids) == expected


def test_four_difficulty_tiers_exist() -> None:
    assert len(task_ids_for_difficulty("easy")) >= 1
    assert len(task_ids_for_difficulty("medium")) >= 1
    assert len(task_ids_for_difficulty("hard")) >= 1
    assert len(task_ids_for_difficulty("extreme")) >= 1


def test_extreme_chargeback_task_generates_correctly() -> None:
    """Extreme task must generate with 5 items and required gold fields."""
    task = get_task("extreme_chargeback", seed=42)
    assert task["task_id"] == "extreme_chargeback"
    assert task["difficulty"] == "extreme"
    items = task["customer_ticket"]["mentioned_items"]
    assert len(items) == 5
    gold = task["gold"]
    assert gold["ticket_resolution"] == "partial_refund"
    assert gold["fraud_risk"] is True
    assert "item-5" in gold["item_resolutions"]
    assert gold["item_resolutions"]["item-5"] == "escalate"


def test_extreme_task_grade_submission_in_unit_interval() -> None:
    """grade_submission must return a clamped score for the extreme task."""
    task = get_task("extreme_chargeback", seed=7)
    score, breakdown = grade_submission(
        task=task,
        seen_sections=["order", "customer", "policy", "inventory"],
        current_priority="urgent",
        current_tags=["damaged", "fraud_flag", "partial_resolution", "escalation_required", "policy_violation"],
        item_resolutions={
            "item-1": "partial_refund",
            "item-2": "partial_refund",
            "item-3": "deny",
            "item-4": "deny",
            "item-5": "escalate",
        },
        ticket_resolution="partial_refund",
        drafted_reply=(
            "Two items were confirmed as damaged in transit and are covered — "
            "partial refunds will be issued within 48 hours. "
            "The change-of-mind items cannot be refunded: bulk order policy. "
            "The non-delivery claim has been escalated for carrier investigation."
        ),
        fraud_flagged=True,
        step_count=14,
    )
    assert 0.01 <= score <= 0.99
    assert breakdown["item_resolution_accuracy"] == 1.0
    assert breakdown["ticket_resolution_accuracy"] == 1.0
