from __future__ import annotations

from typing import Any, Dict

from openenv.core.client_types import StepResult
from openenv.core.env_client import EnvClient

try:
    from .models import ReturnDeskAction, ReturnDeskObservation, ReturnDeskState
except ImportError:  # pragma: no cover - enables direct root-level execution
    from models import ReturnDeskAction, ReturnDeskObservation, ReturnDeskState


class ReturnDeskEnv(EnvClient[ReturnDeskAction, ReturnDeskObservation, ReturnDeskState]):
    """Client wrapper for the ReturnDesk OpenEnv environment."""

    def _step_payload(self, action: ReturnDeskAction) -> Dict[str, Any]:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[ReturnDeskObservation]:
        observation = ReturnDeskObservation(**payload.get("observation", {}))
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=bool(payload.get("done", False)),
        )

    def _parse_state(self, payload: Dict[str, Any]) -> ReturnDeskState:
        return ReturnDeskState(**payload)
