from __future__ import annotations

try:
    from return_desk_env.models import ReturnDeskAction
    from return_desk_env.server.environment import ReturnDeskEnvironment
except ImportError:  # pragma: no cover
    from models import ReturnDeskAction
    from server.environment import ReturnDeskEnvironment



def test_reset_and_state_are_clean() -> None:
    env = ReturnDeskEnvironment()
    obs = env.reset(task_id="easy_refund")
    assert obs.task_id == "easy_refund"
    assert obs.visible_sections == []
    assert obs.item_resolutions == {}
    assert env.state.task_id == "easy_refund"
    assert env.state.visible_sections == []



def test_easy_task_can_score_full_credit() -> None:
    env = ReturnDeskEnvironment()
    env.reset(task_id="easy_refund")
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
                "We are sorry your BrewMaster Coffee Grinder arrived damaged. "
                "We will process a refund, and you should see it in 3-5 business days. "
                "No return required."
            ),
        )
    )
    final_obs = env.step(ReturnDeskAction(action_type="submit"))
    assert final_obs.done is True
    assert final_obs.final_score is not None
    assert final_obs.final_score >= 0.99



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
