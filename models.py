from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import Field, model_validator

from openenv.core.env_server.types import Action, Observation, State


PRIORITIES = ["low", "medium", "high", "urgent"]
RESOLUTIONS = [
    "refund",
    "exchange",
    "store_credit",
    "deny",
    "escalate",
    "request_info",
    "partial_refund",
]
ACTION_TYPES = [
    "inspect_order",
    "inspect_customer",
    "inspect_policy",
    "inspect_inventory",
    "flag_fraud",
    "ask_customer",
    "set_priority",
    "add_tag",
    "set_item_resolution",
    "set_ticket_resolution",
    "draft_reply",
    "submit",
]


class ReturnDeskAction(Action):
    """Typed action model for the ReturnDesk environment."""

    action_type: Literal[
        "inspect_order",
        "inspect_customer",
        "inspect_policy",
        "inspect_inventory",
        "flag_fraud",
        "ask_customer",
        "set_priority",
        "add_tag",
        "set_item_resolution",
        "set_ticket_resolution",
        "draft_reply",
        "submit",
    ]
    item_id: Optional[str] = Field(default=None, description="Target item id for item-level actions")
    question: Optional[str] = Field(default=None, description="Question text for ask_customer action")
    priority: Optional[Literal["low", "medium", "high", "urgent"]] = Field(
        default=None,
        description="Ticket priority for set_priority",
    )
    tag: Optional[str] = Field(default=None, description="Canonical tag for add_tag")
    resolution: Optional[Literal[
        "refund",
        "exchange",
        "store_credit",
        "deny",
        "escalate",
        "request_info",
        "partial_refund",
    ]] = Field(
        default=None,
        description="Resolution for set_item_resolution or set_ticket_resolution",
    )
    message: Optional[str] = Field(default=None, description="Drafted customer reply")

    @model_validator(mode="after")
    def validate_required_fields(self) -> "ReturnDeskAction":
        if self.action_type == "set_priority" and self.priority is None:
            raise ValueError("set_priority requires the 'priority' field")
        if self.action_type == "add_tag" and not self.tag:
            raise ValueError("add_tag requires the 'tag' field")
        if self.action_type == "set_item_resolution":
            if not self.item_id:
                raise ValueError("set_item_resolution requires the 'item_id' field")
            if self.resolution is None:
                raise ValueError("set_item_resolution requires the 'resolution' field")
        if self.action_type == "set_ticket_resolution" and self.resolution is None:
            raise ValueError("set_ticket_resolution requires the 'resolution' field")
        if self.action_type == "draft_reply" and not self.message:
            raise ValueError("draft_reply requires the 'message' field")
        return self


class ReturnDeskObservation(Observation):
    """Observation returned to agents after reset/step."""

    task_id: str
    difficulty: Literal["easy", "medium", "hard", "extreme"]
    objective: str
    customer_ticket: Dict[str, Any]
    available_actions: List[str]
    allowed_priorities: List[str] = Field(default_factory=lambda: list(PRIORITIES))
    allowed_resolutions: List[str] = Field(default_factory=lambda: list(RESOLUTIONS))
    available_tags: List[str] = Field(default_factory=list)
    visible_sections: List[str] = Field(default_factory=list)
    order_summary: Optional[Dict[str, Any]] = None
    customer_summary: Optional[Dict[str, Any]] = None
    policy_summary: Optional[Dict[str, Any]] = None
    inventory_summary: Optional[Dict[str, Any]] = None
    current_priority: Optional[str] = None
    current_tags: List[str] = Field(default_factory=list)
    item_resolutions: Dict[str, str] = Field(default_factory=dict)
    ticket_resolution: Optional[str] = None
    drafted_reply: str = ""
    customer_messages: List[Dict[str, str]] = Field(
        default_factory=list,
        description="Multi-turn dialogue: customer follow-up messages in response to ask_customer or request_info actions",
    )
    history: List[str] = Field(default_factory=list)
    steps_remaining: int = 0
    latest_note: str = ""
    # Live reward breakdown — shown at every step so the agent can course-correct
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
    # Customer sentiment: 0.0 = neutral, negative = upset, positive = satisfied
    customer_sentiment: Optional[float] = None
    # Fraud flag raised by agent using flag_fraud action
    fraud_flagged: bool = False
    final_score: Optional[float] = None
    grader_breakdown: Dict[str, float] = Field(default_factory=dict)


class ReturnDeskState(State):
    """Non-leaky environment state returned by state()."""

    task_id: Optional[str] = None
    difficulty: Optional[str] = None
    visible_sections: List[str] = Field(default_factory=list)
    submitted: bool = False
    current_priority: Optional[str] = None
    current_tags: List[str] = Field(default_factory=list)
    item_resolutions: Dict[str, str] = Field(default_factory=dict)
    ticket_resolution: Optional[str] = None
    steps_remaining: int = 0
    fraud_flagged: bool = False


ACTION_HELP = [
    "{\"action_type\": \"inspect_order\"}",
    "{\"action_type\": \"inspect_customer\"}",
    "{\"action_type\": \"inspect_policy\"}",
    "{\"action_type\": \"inspect_inventory\"}",
    "{\"action_type\": \"flag_fraud\"}",
    "{\"action_type\": \"ask_customer\", \"question\": \"Can you provide more details about the issue?\"}",
    "{\"action_type\": \"set_priority\", \"priority\": \"high\"}",
    "{\"action_type\": \"add_tag\", \"tag\": \"damaged\"}",
    "{\"action_type\": \"set_item_resolution\", \"item_id\": \"item-1\", \"resolution\": \"refund\"}",
    "{\"action_type\": \"set_ticket_resolution\", \"resolution\": \"refund\"}",
    "{\"action_type\": \"draft_reply\", \"message\": \"We are sorry...\"}",
    "{\"action_type\": \"submit\"}",
]
