from __future__ import annotations

try:
    from return_desk_env.graders import grade_submission, reply_slot_coverage, tag_f1
    from return_desk_env.tasks.catalog import get_task
except ImportError:  # pragma: no cover
    from graders import grade_submission, reply_slot_coverage, tag_f1
    from tasks.catalog import get_task



def test_tag_f1_is_perfect_for_exact_match() -> None:
    assert tag_f1(["damaged", "refund_request"], ["damaged", "refund_request"]) == 1.0



def test_reply_slot_coverage_detects_missing_slots() -> None:
    requirements = {
        "decision": ["refund"],
        "timeline": ["3-5 business days"],
    }
    assert reply_slot_coverage("We will issue a refund.", requirements) == 0.5



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
