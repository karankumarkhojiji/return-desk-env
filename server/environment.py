from __future__ import annotations

import random
import uuid
from typing import Any, Dict, Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import EnvironmentMetadata

try:
    from ..graders import feedback_from_breakdown, grade_submission
    from ..models import ACTION_HELP, ReturnDeskAction, ReturnDeskObservation, ReturnDeskState
    from ..rewards import (
        categorical_delta_reward,
        inspection_reward,
        item_resolution_reward,
        reply_reward,
        submit_reward,
        tag_reward,
        ticket_resolution_reward,
    )
    from ..tasks.catalog import get_task, list_task_ids, task_ids_for_difficulty
except ImportError:  # pragma: no cover - enables docker/local root execution
    from graders import feedback_from_breakdown, grade_submission
    from models import ACTION_HELP, ReturnDeskAction, ReturnDeskObservation, ReturnDeskState
    from rewards import (
        categorical_delta_reward,
        inspection_reward,
        item_resolution_reward,
        reply_reward,
        submit_reward,
        tag_reward,
        ticket_resolution_reward,
    )
    from tasks.catalog import get_task, list_task_ids, task_ids_for_difficulty


class ReturnDeskEnvironment(Environment[ReturnDeskAction, ReturnDeskObservation, ReturnDeskState]):
    """Real-world customer operations environment for e-commerce returns."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self) -> None:
        super().__init__()
        self._clear_runtime_state()

    def _clear_runtime_state(self) -> None:
        self._task: Optional[Dict[str, Any]] = None
        self._visible_sections: Dict[str, Dict[str, Any]] = {}
        self._current_priority: Optional[str] = None
        self._current_tags: list[str] = []
        self._item_resolutions: Dict[str, str] = {}
        self._ticket_resolution: Optional[str] = None
        self._drafted_reply: str = ""
        self._history: list[str] = []
        self._latest_note: str = ""
        self._final_score: Optional[float] = None
        self._grader_breakdown: Dict[str, float] = {}
        self._submitted: bool = False
        self._done: bool = False
        self._state = ReturnDeskState()

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="ReturnDeskEnv",
            description=(
                "A deterministic e-commerce returns and refunds operations environment with "
                "real-world customer support workflows, typed actions, shaped rewards, and "
                "three graded tasks."
            ),
            version="0.1.0",
            author="OpenAI + user customization",
        )

    def _select_task_id(self, seed: Optional[int], task_id: Optional[str], difficulty: Optional[str]) -> str:
        if task_id:
            return task_id
        if difficulty:
            candidates = task_ids_for_difficulty(difficulty)
            if not candidates:
                raise ValueError(f"No tasks available for difficulty='{difficulty}'")
        else:
            candidates = list_task_ids()
        candidates = sorted(candidates)
        if seed is None:
            return candidates[0]
        rng = random.Random(seed)
        return candidates[rng.randrange(len(candidates))]

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        difficulty: Optional[str] = None,
        **kwargs: Any,
    ) -> ReturnDeskObservation:
        self._clear_runtime_state()
        selected_task_id = self._select_task_id(seed=seed, task_id=task_id, difficulty=difficulty)
        self._task = get_task(selected_task_id)
        self._state = ReturnDeskState(
            episode_id=episode_id or f"{selected_task_id}-{uuid.uuid4()}",
            step_count=0,
            task_id=selected_task_id,
            difficulty=self._task["difficulty"],
            visible_sections=[],
            submitted=False,
            current_priority=None,
            current_tags=[],
            item_resolutions={},
            ticket_resolution=None,
            steps_remaining=self._task["max_steps"],
        )
        self._latest_note = (
            f"Loaded task '{selected_task_id}'. Inspect the relevant evidence, make decisions, "
            "draft a reply, then submit."
        )
        return self._build_observation(reward=None)

    def step(
        self,
        action: ReturnDeskAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> ReturnDeskObservation:
        del timeout_s, kwargs
        if self._task is None:
            raise RuntimeError("Environment must be reset before stepping")
        if self._done:
            self._latest_note = "Episode already complete. Reset to start a new ticket."
            return self._build_observation(reward=0.0)

        reward = 0.0
        note = ""
        gold = self._task["gold"]

        if action.action_type.startswith("inspect_"):
            section = action.action_type.replace("inspect_", "", 1)
            reward = inspection_reward(section, self._state.visible_sections, gold["required_sections"])
            if section in self._visible_sections:
                note = f"{section} was already inspected."
            else:
                section_payload = self._task["sections"].get(section)
                if section_payload is None:
                    reward = -0.05
                    note = f"Unknown section '{section}'."
                else:
                    self._visible_sections[section] = section_payload
                    note = f"Revealed {section} details."

        elif action.action_type == "set_priority":
            old_priority = self._current_priority
            self._current_priority = action.priority
            reward = categorical_delta_reward(
                old_value=old_priority,
                new_value=self._current_priority,
                gold_value=gold["priority"],
                positive=0.05,
                negative=-0.04,
            )
            note = f"Priority set to {self._current_priority}."

        elif action.action_type == "add_tag":
            tag = (action.tag or "").strip().lower()
            if tag not in self._task["available_tags"]:
                reward = -0.03
                note = f"Unknown tag '{tag}'."
            else:
                reward = tag_reward(tag, self._current_tags, gold["tags"])
                if tag not in self._current_tags:
                    self._current_tags.append(tag)
                note = f"Tag '{tag}' recorded."

        elif action.action_type == "set_item_resolution":
            valid_item_ids = {
                item["item_id"]
                for item in self._task["customer_ticket"]["mentioned_items"]
            }
            if action.item_id not in valid_item_ids:
                reward = -0.05
                note = f"Unknown item_id '{action.item_id}'."
            elif action.resolution == "partial_refund":
                reward = -0.05
                note = "partial_refund is a ticket-level resolution, not an item-level resolution."
            else:
                old_resolution = self._item_resolutions.get(action.item_id or "")
                self._item_resolutions[action.item_id or ""] = action.resolution or ""
                reward = item_resolution_reward(
                    item_id=action.item_id or "",
                    old_value=old_resolution,
                    new_value=action.resolution,
                    gold_item_resolutions=gold["item_resolutions"],
                )
                note = f"Resolution for {action.item_id} set to {action.resolution}."

        elif action.action_type == "set_ticket_resolution":
            old_resolution = self._ticket_resolution
            self._ticket_resolution = action.resolution
            reward = ticket_resolution_reward(
                old_value=old_resolution,
                new_value=self._ticket_resolution,
                gold_value=gold["ticket_resolution"],
            )
            note = f"Ticket resolution set to {self._ticket_resolution}."

        elif action.action_type == "draft_reply":
            old_reply = self._drafted_reply
            self._drafted_reply = action.message or ""
            reward = reply_reward(old_reply, self._drafted_reply, gold["reply_requirements"])
            note = "Customer reply drafted."

        elif action.action_type == "submit":
            score, breakdown = grade_submission(
                task=self._task,
                seen_sections=self._state.visible_sections,
                current_priority=self._current_priority,
                current_tags=self._current_tags,
                item_resolutions=self._item_resolutions,
                ticket_resolution=self._ticket_resolution,
                drafted_reply=self._drafted_reply,
            )
            self._final_score = score
            self._grader_breakdown = breakdown
            reward = submit_reward(score, breakdown["evidence_coverage"])
            self._submitted = True
            self._done = True
            note = f"Submitted. Final score={score:.2f}. {feedback_from_breakdown(breakdown)}"

        else:
            reward = -0.05
            note = f"Unhandled action type '{action.action_type}'."

        self._state.step_count += 1
        self._latest_note = note
        self._history.append(
            f"Step {self._state.step_count}: {action.action_type} -> {note} (reward {reward:+.2f})"
        )
        self._sync_state()

        if not self._done and self._state.step_count >= self._task["max_steps"]:
            score, breakdown = grade_submission(
                task=self._task,
                seen_sections=self._state.visible_sections,
                current_priority=self._current_priority,
                current_tags=self._current_tags,
                item_resolutions=self._item_resolutions,
                ticket_resolution=self._ticket_resolution,
                drafted_reply=self._drafted_reply,
            )
            self._final_score = score
            self._grader_breakdown = breakdown
            self._done = True
            self._latest_note = (
                f"Step limit reached. Partial final score={score:.2f}. "
                f"{feedback_from_breakdown(breakdown)}"
            )
            self._history[-1] = (
                f"Step {self._state.step_count}: {action.action_type} -> {self._latest_note} "
                f"(reward {reward:+.2f})"
            )
            reward = max(reward, round(score, 6))
            self._sync_state()

        return self._build_observation(reward=round(reward, 6))

    def _sync_state(self) -> None:
        self._state.visible_sections = sorted(self._visible_sections.keys())
        self._state.submitted = self._submitted
        self._state.current_priority = self._current_priority
        self._state.current_tags = list(self._current_tags)
        self._state.item_resolutions = dict(self._item_resolutions)
        self._state.ticket_resolution = self._ticket_resolution
        self._state.steps_remaining = self._steps_remaining()

    def _steps_remaining(self) -> int:
        if self._task is None:
            return 0
        return max(int(self._task["max_steps"]) - int(self._state.step_count), 0)

    def _build_observation(self, reward: Optional[float]) -> ReturnDeskObservation:
        assert self._task is not None
        return ReturnDeskObservation(
            task_id=self._task["task_id"],
            difficulty=self._task["difficulty"],
            objective=self._task["objective"],
            customer_ticket=self._task["customer_ticket"],
            available_actions=ACTION_HELP,
            available_tags=self._task["available_tags"],
            visible_sections=sorted(self._visible_sections.keys()),
            order_summary=self._visible_sections.get("order"),
            customer_summary=self._visible_sections.get("customer"),
            policy_summary=self._visible_sections.get("policy"),
            inventory_summary=self._visible_sections.get("inventory"),
            current_priority=self._current_priority,
            current_tags=list(self._current_tags),
            item_resolutions=dict(self._item_resolutions),
            ticket_resolution=self._ticket_resolution,
            drafted_reply=self._drafted_reply,
            history=self._history[-8:],
            steps_remaining=self._steps_remaining(),
            latest_note=self._latest_note,
            final_score=self._final_score,
            grader_breakdown=self._grader_breakdown if self._done else {},
            reward=reward,
            done=self._done,
        )

    @property
    def state(self) -> ReturnDeskState:
        return self._state
