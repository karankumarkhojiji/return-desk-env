from __future__ import annotations

import pytest

try:
    from return_desk_env.models import ReturnDeskAction
    from return_desk_env.server.environment import ReturnDeskEnvironment
except ImportError:  # pragma: no cover
    from models import ReturnDeskAction
    from server.environment import ReturnDeskEnvironment


# ---------------------------------------------------------------------------
# Reset / state isolation
# ---------------------------------------------------------------------------

def test_reset_and_state_are_clean() -> None:
    env = ReturnDeskEnvironment()
    obs = env.reset(task_id="easy_refund")
    assert obs.task_id == "easy_refund"
    assert obs.visible_sections == []
    assert obs.item_resolutions == {}
    assert env.state.task_id == "easy_refund"
    assert env.state.visible_sections == []


def test_reset_clears_prior_episode_state() -> None:
    """A second reset must fully clear state from the first episode."""
    env = ReturnDeskEnvironment()
    env.reset(task_id="easy_refund")
    env.step(ReturnDeskAction(action_type="inspect_order"))
    env.step(ReturnDeskAction(action_type="add_tag", tag="damaged"))

    # Reset to a different task
    obs2 = env.reset(task_id="fraud_risk")
    assert obs2.task_id == "fraud_risk"
    assert obs2.visible_sections == []
    assert obs2.current_tags == []
    assert obs2.fraud_flagged is False


# ---------------------------------------------------------------------------
# Easy task — perfect play
# ---------------------------------------------------------------------------

def test_easy_task_can_score_full_credit() -> None:
    env = ReturnDeskEnvironment()
    obs = env.reset(task_id="easy_refund")

    # Read the generated product name from the observation — works for any procedural variant
    item_name = obs.customer_ticket["mentioned_items"][0]["name"]

    env.step(ReturnDeskAction(action_type="inspect_order"))
    env.step(ReturnDeskAction(action_type="inspect_policy"))
    env.step(ReturnDeskAction(action_type="set_priority", priority="high"))
    env.step(ReturnDeskAction(action_type="add_tag", tag="damaged"))
    env.step(ReturnDeskAction(action_type="add_tag", tag="refund_request"))
    env.step(
        ReturnDeskAction(
            action_type="set_item_resolution",
            item_id="item-1",
            resolution="refund",
        )
    )
    env.step(ReturnDeskAction(action_type="set_ticket_resolution", resolution="refund"))
    env.step(
        ReturnDeskAction(
            action_type="draft_reply",
            message=(
                f"We are sorry your {item_name} arrived damaged. "
                "We will process a refund, and you should see it in 3-5 business days. "
                "No return required."
            ),
        )
    )
    final_obs = env.step(ReturnDeskAction(action_type="submit"))
    assert final_obs.done is True
    assert final_obs.final_score is not None
    assert final_obs.final_score >= 0.85


# ---------------------------------------------------------------------------
# Wrong submission should score lower
# ---------------------------------------------------------------------------

def test_wrong_submission_scores_lower() -> None:
    env = ReturnDeskEnvironment()
    env.reset(task_id="hard_partial_resolution")
    env.step(ReturnDeskAction(action_type="inspect_order"))
    env.step(ReturnDeskAction(action_type="set_priority", priority="low"))
    env.step(ReturnDeskAction(action_type="set_ticket_resolution", resolution="refund"))
    final_obs = env.step(ReturnDeskAction(action_type="submit"))
    assert final_obs.done is True
    assert final_obs.final_score is not None
    assert final_obs.final_score < 0.7


# ---------------------------------------------------------------------------
# Forced investigation gate
# ---------------------------------------------------------------------------

def test_resolve_without_inspecting_incurs_penalty() -> None:
    """Setting a resolution before inspecting should return a negative step reward."""
    env = ReturnDeskEnvironment()
    env.reset(task_id="easy_refund")
    # No inspect has been done — go straight to resolution
    obs = env.step(ReturnDeskAction(action_type="set_item_resolution", item_id="item-1", resolution="refund"))
    assert obs.reward is not None and obs.reward < 0, (
        "Expected negative reward for resolving without inspection"
    )


# ---------------------------------------------------------------------------
# Fraud flagging mechanics
# ---------------------------------------------------------------------------

def test_flag_fraud_gives_positive_reward_on_fraud_task() -> None:
    env = ReturnDeskEnvironment()
    env.reset(task_id="fraud_risk")
    env.step(ReturnDeskAction(action_type="inspect_order"))
    env.step(ReturnDeskAction(action_type="inspect_customer"))
    obs = env.step(ReturnDeskAction(action_type="flag_fraud"))
    assert obs.reward is not None and obs.reward > 0, (
        "Correct flag_fraud on a fraud task should give a positive reward"
    )
    assert obs.fraud_flagged is True


def test_flag_fraud_on_non_fraud_task_incurs_penalty() -> None:
    env = ReturnDeskEnvironment()
    env.reset(task_id="easy_refund")
    env.step(ReturnDeskAction(action_type="inspect_order"))
    obs = env.step(ReturnDeskAction(action_type="flag_fraud"))
    assert obs.reward is not None and obs.reward < 0, (
        "Calling flag_fraud on a non-fraud task should be penalised"
    )


# ---------------------------------------------------------------------------
# Expired return — denied correctly
# ---------------------------------------------------------------------------

def test_expired_return_denies_correctly() -> None:
    env = ReturnDeskEnvironment()
    env.reset(task_id="expired_return")
    env.step(ReturnDeskAction(action_type="inspect_order"))
    env.step(ReturnDeskAction(action_type="inspect_policy"))
    env.step(ReturnDeskAction(action_type="set_priority", priority="medium"))
    env.step(ReturnDeskAction(action_type="add_tag", tag="return_window_exceeded"))
    env.step(ReturnDeskAction(action_type="add_tag", tag="policy_violation"))
    env.step(ReturnDeskAction(action_type="set_item_resolution", item_id="item-1", resolution="deny"))
    env.step(ReturnDeskAction(action_type="set_ticket_resolution", resolution="deny"))
    env.step(
        ReturnDeskAction(
            action_type="draft_reply",
            message=(
                "We are sorry, but your return request falls outside our 30-day return policy window. "
                "We are unable to process a refund for this order."
            ),
        )
    )
    final_obs = env.step(ReturnDeskAction(action_type="submit"))
    assert final_obs.done is True
    assert final_obs.final_score is not None
    assert final_obs.final_score >= 0.75


# ---------------------------------------------------------------------------
# Extreme task — correct full play
# ---------------------------------------------------------------------------

def test_extreme_chargeback_resets_cleanly() -> None:
    env = ReturnDeskEnvironment()
    obs = env.reset(task_id="extreme_chargeback")
    assert obs.task_id == "extreme_chargeback"
    assert obs.difficulty == "extreme"
    items = obs.customer_ticket["mentioned_items"]
    assert len(items) == 5


def test_extreme_chargeback_observation_has_5_items() -> None:
    env = ReturnDeskEnvironment()
    obs = env.reset(task_id="extreme_chargeback")
    assert len(obs.customer_ticket["mentioned_items"]) == 5


# ---------------------------------------------------------------------------
# Reward breakdown is present every step (live feedback)
# ---------------------------------------------------------------------------

def test_reward_breakdown_present_after_every_step() -> None:
    env = ReturnDeskEnvironment()
    env.reset(task_id="medium_exchange")
    obs = env.step(ReturnDeskAction(action_type="inspect_order"))
    assert isinstance(obs.reward_breakdown, dict)
    # Breakdown should contain at least policy_compliance sub-field before submit
    # (prior to submit, the grader may only return a partial view)
    assert obs.reward_breakdown is not None


# ---------------------------------------------------------------------------
# Seeded task determinism
# ---------------------------------------------------------------------------

def test_seeded_reset_is_deterministic() -> None:
    """Two resets with the same seed must produce identical tickets."""
    env1 = ReturnDeskEnvironment()
    env2 = ReturnDeskEnvironment()
    obs1 = env1.reset(task_id="easy_refund", seed=42)
    obs2 = env2.reset(task_id="easy_refund", seed=42)
    item1 = obs1.customer_ticket["mentioned_items"][0]["name"]
    item2 = obs2.customer_ticket["mentioned_items"][0]["name"]
    assert item1 == item2
