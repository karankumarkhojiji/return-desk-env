from __future__ import annotations

import random
import uuid
from collections import deque
from typing import Any, Dict, List, Optional

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


# ---------------------------------------------------------------------------
# Curriculum state — tracks rolling performance to auto-advance difficulty
# ---------------------------------------------------------------------------

class CurriculumState:
    """
    Tracks agent performance across episodes and selects task difficulty
    based on a rolling mean score window.

    Difficulty ladder:
    - rolling mean < 0.55 → easy
    - rolling mean 0.55–0.70 → medium
    - rolling mean 0.70–0.82 → hard
    - rolling mean >= 0.82 → extreme
    """

    DIFFICULTY_THRESHOLDS = [
        (0.82, "extreme"),
        (0.70, "hard"),
        (0.55, "medium"),
        (0.00, "easy"),
    ]

    def __init__(self, window: int = 10) -> None:
        self._scores: deque[float] = deque(maxlen=window)
        self._episode_count: int = 0

    def record(self, score: float) -> None:
        self._scores.append(score)
        self._episode_count += 1

    @property
    def rolling_mean(self) -> float:
        if not self._scores:
            return 0.0
        return round(sum(self._scores) / len(self._scores), 4)

    @property
    def episode_count(self) -> int:
        return self._episode_count

    def select_difficulty(self) -> str:
        mean = self.rolling_mean
        for threshold, difficulty in self.DIFFICULTY_THRESHOLDS:
            if mean >= threshold:
                return difficulty
        return "easy"

    def summary(self) -> Dict[str, Any]:
        return {
            "episode_count": self._episode_count,
            "rolling_mean": self.rolling_mean,
            "current_difficulty": self.select_difficulty(),
            "window": self._scores.maxlen,
        }


# ---------------------------------------------------------------------------
# Global episode replay store — keyed by episode_id
# Bounded at 100 entries to prevent memory growth
# ---------------------------------------------------------------------------
_episode_replay_store: Dict[str, Dict[str, Any]] = {}
_MAX_REPLAY_ENTRIES = 100


def _store_replay(episode_id: str, replay: Dict[str, Any]) -> None:
    if len(_episode_replay_store) >= _MAX_REPLAY_ENTRIES:
        # Evict the oldest entry
        oldest = next(iter(_episode_replay_store))
        del _episode_replay_store[oldest]
    _episode_replay_store[episode_id] = replay


def get_episode_replay(episode_id: str) -> Optional[Dict[str, Any]]:
    """Public accessor for the /api/replay endpoint in app.py."""
    return _episode_replay_store.get(episode_id)


def list_episode_ids() -> List[str]:
    """Return all stored episode IDs, most recent last."""
    return list(_episode_replay_store.keys())


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ReturnDeskEnvironment(Environment[ReturnDeskAction, ReturnDeskObservation, ReturnDeskState]):
    """Real-world customer operations environment for e-commerce returns and fraud detection."""

    SUPPORTS_CONCURRENT_SESSIONS = True

    # Sentiment dynamics constants
    _SENTIMENT_DECAY_START_STEP = 5    # Steps after which idle penalty kicks in
    _SENTIMENT_DECAY_RATE = 0.04       # Per-step decay after threshold
    _SENTIMENT_CORRECT_BOOST = 0.06    # Boost for correct item resolution
    _SENTIMENT_WRONG_HIT = 0.04        # Drop for wrong resolution
    _SENTIMENT_FRAUD_HIT = 0.08        # Drop when fraud is flagged (customer upset)

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
        self._replay_trajectory: list[Dict[str, Any]] = []   # Full step records
        self._latest_note: str = ""
        self._final_score: Optional[float] = None
        self._grader_breakdown: Dict[str, float] = {}
        self._live_breakdown: Dict[str, float] = {}
        self._submitted: bool = False
        self._done: bool = False
        self._fraud_flagged: bool = False
        # Multi-turn dialogue
        self._customer_messages: list[Dict[str, str]] = []
        self._customer_follow_up_index: int = 0
        # Sentiment dynamics
        self._dynamic_sentiment: float = 0.0
        self._state = ReturnDeskState()

    def get_metadata(self) -> EnvironmentMetadata:
        return EnvironmentMetadata(
            name="ReturnDeskEnv",
            description=(
                "E-commerce returns and fraud-detection operations environment with 7 procedurally-generated "
                "tasks across 4 difficulty tiers (easy / medium / hard / extreme), 9-component weighted grader, "
                "semantic reply scoring, multi-objective cost-efficiency rewards, fraud detection with -0.40 "
                "penalty, forced evidence-based reasoning, and per-step shaped rewards."
            ),
            version="0.3.0",
            author="ReturnDeskEnv Team",
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
        mode: Optional[str] = None,                  # "curriculum" for auto-difficulty
        curriculum_state: Optional[CurriculumState] = None,  # external tracker
        **kwargs: Any,
    ) -> ReturnDeskObservation:
        self._clear_runtime_state()

        # Curriculum mode: override difficulty from tracker
        if mode == "curriculum" and curriculum_state is not None and task_id is None:
            difficulty = curriculum_state.select_difficulty()

        selected_task_id = self._select_task_id(seed=seed, task_id=task_id, difficulty=difficulty)
        self._task = get_task(selected_task_id, seed=seed)

        eid = episode_id or f"{selected_task_id}-{uuid.uuid4()}"
        self._state = ReturnDeskState(
            episode_id=eid,
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
            fraud_flagged=False,
        )

        # Sentiment dynamics: initialise from task baseline
        self._dynamic_sentiment = float(self._task.get("customer_sentiment", 0.0))

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

        # --- inspect actions ---
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

        # --- ask_customer action (multi-turn dialogue) ---
        elif action.action_type == "ask_customer":
            follow_ups = self._task.get("customer_follow_ups", [])
            if not follow_ups:
                reward = -0.02
                note = "No follow-up messages available for this task."
            elif self._customer_follow_up_index >= len(follow_ups):
                reward = -0.02
                note = "No more follow-up messages available from the customer."
            else:
                response = follow_ups[self._customer_follow_up_index]
                self._customer_follow_up_index += 1
                self._customer_messages.append({"role": "customer", "text": response})
                # Small positive reward for asking when needed, zero if unnecessary
                already_have_info = len(self._visible_sections) >= len(gold["required_sections"])
                reward = 0.02 if not already_have_info else -0.01
                note = f"Customer replied: \"{response}\""

        # --- flag_fraud action ---
        elif action.action_type == "flag_fraud":
            is_fraud_task = gold.get("fraud_risk", False)
            if self._fraud_flagged:
                reward = -0.02
                note = "Fraud flag already raised on this ticket."
            elif is_fraud_task:
                self._fraud_flagged = True
                reward = 0.10  # bonus for correctly identifying fraud risk
                # Sentiment dynamics: fraud flag makes customer feel accused → slight drop
                self._dynamic_sentiment = max(-1.0, self._dynamic_sentiment - self._SENTIMENT_FRAUD_HIT)
                note = "Fraud risk flagged. This ticket should be escalated, not refunded."
            else:
                self._fraud_flagged = True
                reward = -0.08  # penalty for false positive
                self._dynamic_sentiment = max(-1.0, self._dynamic_sentiment - self._SENTIMENT_FRAUD_HIT)
                note = "Fraud flag raised, but this ticket does not have fraud signals. False positive."

        # --- set_priority ---
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

        # --- add_tag ---
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

        # --- set_item_resolution ---
        elif action.action_type == "set_item_resolution":
            # Guard: agent must inspect at least one section before making decisions
            if not self._visible_sections:
                reward = -0.10
                note = "You must inspect at least one evidence section before setting resolutions."
            else:
                valid_item_ids = {
                    item["item_id"]
                    for item in self._task["customer_ticket"]["mentioned_items"]
                }
                if action.item_id not in valid_item_ids:
                    reward = -0.05
                    note = f"Unknown item_id '{action.item_id}'."
                elif action.resolution == "partial_refund":
                    # FIX: only block partial_refund at item level if this item's policy doesn't allow it
                    allowed_for_item = gold["allowed_item_resolutions"].get(action.item_id or "", [])
                    if "partial_refund" not in [v.lower() for v in allowed_for_item]:
                        reward = -0.05
                        note = f"partial_refund is not a valid item-level resolution for {action.item_id}."
                    else:
                        # partial_refund IS allowed at item level (e.g. extreme_chargeback)
                        old_resolution = self._item_resolutions.get(action.item_id or "")
                        self._item_resolutions[action.item_id or ""] = action.resolution or ""
                        reward = item_resolution_reward(
                            item_id=action.item_id or "",
                            old_value=old_resolution,
                            new_value=action.resolution,
                            gold_item_resolutions=gold["item_resolutions"],
                        )
                        if reward > 0:
                            self._dynamic_sentiment = min(1.0, self._dynamic_sentiment + self._SENTIMENT_CORRECT_BOOST)
                        else:
                            self._dynamic_sentiment = max(-1.0, self._dynamic_sentiment - self._SENTIMENT_WRONG_HIT)
                        note = f"Resolution for {action.item_id} set to {action.resolution}."
                else:
                    # Fraud guard: if fraud task and agent tries to refund without flagging
                    if gold.get("fraud_risk") and action.resolution == "refund" and not self._fraud_flagged:
                        reward = -0.15
                        note = (
                            f"WARNING: Issuing a refund on a fraud-risk order without flagging it first "
                            f"is a policy violation. Resolution for {action.item_id} set to {action.resolution}."
                        )
                        self._item_resolutions[action.item_id or ""] = action.resolution or ""
                    else:
                        old_resolution = self._item_resolutions.get(action.item_id or "")
                        self._item_resolutions[action.item_id or ""] = action.resolution or ""
                        reward = item_resolution_reward(
                            item_id=action.item_id or "",
                            old_value=old_resolution,
                            new_value=action.resolution,
                            gold_item_resolutions=gold["item_resolutions"],
                        )
                        if reward > 0:
                            self._dynamic_sentiment = min(1.0, self._dynamic_sentiment + self._SENTIMENT_CORRECT_BOOST)
                        else:
                            self._dynamic_sentiment = max(-1.0, self._dynamic_sentiment - self._SENTIMENT_WRONG_HIT)
                        note = f"Resolution for {action.item_id} set to {action.resolution}."

        # --- set_ticket_resolution ---
        elif action.action_type == "set_ticket_resolution":
            if not self._visible_sections:
                reward = -0.10
                note = "You must inspect at least one evidence section before setting the ticket resolution."
            else:
                # Trigger dialogue when agent asks for more customer info
                if action.resolution == "request_info":
                    follow_ups = self._task.get("customer_follow_ups", [])
                    if follow_ups and self._customer_follow_up_index < len(follow_ups):
                        response = follow_ups[self._customer_follow_up_index]
                        self._customer_follow_up_index += 1
                        self._customer_messages.append({"role": "customer", "text": response})
                    self._ticket_resolution = action.resolution
                    reward = 0.02
                    note = f"Ticket pending customer info. Customer responded: \"{self._customer_messages[-1]['text'] if self._customer_messages else 'No follow-up available.'}\""
                # Fraud guard at ticket level
                elif gold.get("fraud_risk") and action.resolution == "refund" and not self._fraud_flagged:
                    reward = -0.20
                    note = (
                        "SEVERE POLICY VIOLATION: Refunding a fraud-risk ticket without escalating. "
                        f"Ticket resolution set to {action.resolution}."
                    )
                    self._ticket_resolution = action.resolution
                else:
                    old_resolution = self._ticket_resolution
                    self._ticket_resolution = action.resolution
                    reward = ticket_resolution_reward(
                        old_value=old_resolution,
                        new_value=self._ticket_resolution,
                        gold_value=gold["ticket_resolution"],
                    )
                    note = f"Ticket resolution set to {self._ticket_resolution}."

        # --- draft_reply ---
        elif action.action_type == "draft_reply":
            old_reply = self._drafted_reply
            self._drafted_reply = action.message or ""
            reward = reply_reward(old_reply, self._drafted_reply, gold["reply_requirements"])
            note = "Customer reply drafted."

        # --- submit ---
        elif action.action_type == "submit":
            score, breakdown = grade_submission(
                task=self._task,
                seen_sections=self._state.visible_sections,
                current_priority=self._current_priority,
                current_tags=self._current_tags,
                item_resolutions=self._item_resolutions,
                ticket_resolution=self._ticket_resolution,
                drafted_reply=self._drafted_reply,
                step_count=self._state.step_count + 1,
                fraud_flagged=self._fraud_flagged,
            )
            self._final_score = score
            self._grader_breakdown = breakdown
            reward = submit_reward(score, breakdown.get("evidence_coverage", 0.0))
            self._submitted = True
            self._done = True
            note = f"Submitted. Final score={score:.3f}. {feedback_from_breakdown(breakdown)}"

        else:
            reward = -0.05
            note = f"Unhandled action type '{action.action_type}'."

        # Sentiment dynamics: decay per step past the threshold
        if not self._done:
            step_after = self._state.step_count + 1
            if step_after > self._SENTIMENT_DECAY_START_STEP:
                self._dynamic_sentiment = max(
                    -1.0, self._dynamic_sentiment - self._SENTIMENT_DECAY_RATE
                )

        self._state.step_count += 1
        self._latest_note = note
        history_entry = f"Step {self._state.step_count}: {action.action_type} -> {note} (reward {reward:+.2f})"
        self._history.append(history_entry)

        # Store full step record for episode replay
        self._replay_trajectory.append({
            "step": self._state.step_count,
            "action": action.model_dump(exclude_none=True),
            "reward": round(reward, 6),
            "note": note,
            "sentiment": round(self._dynamic_sentiment, 3),
        })

        self._sync_state()

        # Compute live reward breakdown for agent visibility at every step
        if not self._done:
            _, self._live_breakdown = grade_submission(
                task=self._task,
                seen_sections=self._state.visible_sections,
                current_priority=self._current_priority,
                current_tags=self._current_tags,
                item_resolutions=self._item_resolutions,
                ticket_resolution=self._ticket_resolution,
                drafted_reply=self._drafted_reply,
                step_count=self._state.step_count,
                fraud_flagged=self._fraud_flagged,
            )

        # Auto-submit at step limit
        if not self._done and self._state.step_count >= self._task["max_steps"]:
            score, breakdown = grade_submission(
                task=self._task,
                seen_sections=self._state.visible_sections,
                current_priority=self._current_priority,
                current_tags=self._current_tags,
                item_resolutions=self._item_resolutions,
                ticket_resolution=self._ticket_resolution,
                drafted_reply=self._drafted_reply,
                step_count=self._state.step_count,
                fraud_flagged=self._fraud_flagged,
            )
            self._final_score = score
            self._grader_breakdown = breakdown
            self._done = True
            self._latest_note = (
                f"Step limit reached. Partial final score={score:.3f}. "
                f"{feedback_from_breakdown(breakdown)}"
            )
            self._history[-1] = (
                f"Step {self._state.step_count}: {action.action_type} -> {self._latest_note} "
                f"(reward {reward:+.2f})"
            )
            reward = max(reward, round(score, 6))
            self._sync_state()

        # Save completed episode replay
        if self._done and self._state.episode_id:
            _store_replay(
                self._state.episode_id,
                {
                    "episode_id": self._state.episode_id,
                    "task_id": self._task["task_id"],
                    "difficulty": self._task["difficulty"],
                    "final_score": self._final_score,
                    "steps_taken": self._state.step_count,
                    "grader_breakdown": self._grader_breakdown,
                    "trajectory": list(self._replay_trajectory),
                    "customer_messages": list(self._customer_messages),
                    "final_sentiment": round(self._dynamic_sentiment, 3),
                },
            )

        return self._build_observation(reward=round(reward, 6))

    def _sync_state(self) -> None:
        self._state.visible_sections = sorted(self._visible_sections.keys())
        self._state.submitted = self._submitted
        self._state.current_priority = self._current_priority
        self._state.current_tags = list(self._current_tags)
        self._state.item_resolutions = dict(self._item_resolutions)
        self._state.ticket_resolution = self._ticket_resolution
        self._state.steps_remaining = self._steps_remaining()
        self._state.fraud_flagged = self._fraud_flagged

    def _steps_remaining(self) -> int:
        if self._task is None:
            return 0
        return max(int(self._task["max_steps"]) - int(self._state.step_count), 0)

    def _build_observation(self, reward: Optional[float]) -> ReturnDeskObservation:
        assert self._task is not None
        is_done = self._done
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
            customer_messages=list(self._customer_messages),
            history=self._history[-8:],
            steps_remaining=self._steps_remaining(),
            latest_note=self._latest_note,
            reward_breakdown=self._grader_breakdown if is_done else self._live_breakdown,
            customer_sentiment=round(self._dynamic_sentiment, 3),
            fraud_flagged=self._fraud_flagged,
            final_score=self._final_score,
            grader_breakdown=self._grader_breakdown if is_done else {},
            reward=reward,
            done=is_done,
        )

    @property
    def state(self) -> ReturnDeskState:
        return self._state
