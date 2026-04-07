from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, List

CANONICAL_TAGS = [
    "damaged",
    "refund_request",
    "exchange_request",
    "inventory_issue",
    "coupon_order",
    "vip_exception",
    "partial_resolution",
    "non_returnable",
]


TASKS: Dict[str, Dict[str, Any]] = {
    "easy_refund": {
        "task_id": "easy_refund",
        "difficulty": "easy",
        "max_steps": 10,
        "objective": (
            "Resolve a standard damaged-item return ticket. The correct flow is to gather "
            "the relevant evidence, set the right priority, choose the correct item and "
            "ticket resolution, and send a policy-compliant reply before submitting."
        ),
        "available_tags": CANONICAL_TAGS,
        "customer_ticket": {
            "ticket_id": "RTN-1001",
            "channel": "email",
            "customer_id": "CUST-1001",
            "message": (
                "Hi, the BrewMaster Coffee Grinder I got today arrived with a cracked glass jar. "
                "The unit is unusable, and I would like a refund please."
            ),
            "mentioned_items": [
                {
                    "item_id": "item-1",
                    "name": "BrewMaster Coffee Grinder",
                    "issue": "Cracked jar on arrival",
                    "requested_outcome": "refund",
                }
            ],
        },
        "sections": {
            "order": {
                "order_id": "ORD-1001",
                "ordered_at": "2026-03-22",
                "delivered_at": "2026-03-29",
                "payment_method": "credit_card",
                "paid_amount_usd": 89.99,
                "items": [
                    {
                        "item_id": "item-1",
                        "name": "BrewMaster Coffee Grinder",
                        "quantity": 1,
                        "condition_report": "Customer photo shows cracked glass jar.",
                        "personalized": False,
                    }
                ],
            },
            "customer": {
                "customer_tier": "standard",
                "lifetime_orders": 2,
                "lifetime_value_usd": 148.90,
                "prior_policy_exceptions": 0,
            },
            "policy": {
                "return_window_days": 30,
                "defect_policy": "Damaged or defective items delivered within 30 days are eligible for refund or exchange.",
                "preference_policy": "When the customer clearly requests a refund and the policy allows it, refund is preferred over exchange.",
                "logistics_policy": "Return shipment is not required when the product is unsafe or unusable on arrival.",
            },
            "inventory": {
                "item-1": {
                    "available_units": 24,
                    "note": "Replacement stock exists, but customer explicitly requested refund.",
                }
            },
        },
        "gold": {
            "required_sections": ["order", "policy"],
            "priority": "high",
            "tags": ["damaged", "refund_request"],
            "item_resolutions": {"item-1": "refund"},
            "ticket_resolution": "refund",
            "allowed_item_resolutions": {"item-1": ["refund", "exchange"]},
            "allowed_ticket_resolutions": ["refund", "exchange"],
            "reply_requirements": {
                "apology": ["sorry", "apologize"],
                "decision": ["refund"],
                "timeline": ["3-5 business days", "3 to 5 business days", "5 business days"],
                "next_step": ["no return required", "no need to return", "no return shipment"],
            },
        },
    },
    "medium_exchange": {
        "task_id": "medium_exchange",
        "difficulty": "medium",
        "max_steps": 12,
        "objective": (
            "Handle a size-exchange request where exact inventory is unavailable and the order "
            "used a coupon. The agent should inspect inventory and policy details, then choose "
            "the best policy-compliant resolution and explain it clearly."
        ),
        "available_tags": CANONICAL_TAGS,
        "customer_ticket": {
            "ticket_id": "RTN-2042",
            "channel": "chat",
            "customer_id": "CUST-2042",
            "message": (
                "I ordered the FlexFit Hoodie in size M, but I need size L. If the blue L is not "
                "available, store credit is okay. I also used the SPRING15 coupon on this order."
            ),
            "mentioned_items": [
                {
                    "item_id": "item-1",
                    "name": "FlexFit Hoodie - Blue / Size M",
                    "issue": "Needs exchange to size L",
                    "requested_outcome": "exchange_or_store_credit",
                }
            ],
        },
        "sections": {
            "order": {
                "order_id": "ORD-2042",
                "ordered_at": "2026-03-18",
                "delivered_at": "2026-03-23",
                "payment_method": "credit_card",
                "coupon_code": "SPRING15",
                "paid_amount_usd": 51.00,
                "items": [
                    {
                        "item_id": "item-1",
                        "name": "FlexFit Hoodie - Blue / Size M",
                        "quantity": 1,
                        "condition_report": "Unworn, original tags attached.",
                        "personalized": False,
                    }
                ],
            },
            "customer": {
                "customer_tier": "standard",
                "lifetime_orders": 6,
                "lifetime_value_usd": 321.15,
                "prior_policy_exceptions": 0,
            },
            "policy": {
                "return_window_days": 30,
                "exchange_policy": "Unworn apparel may be exchanged within 30 days.",
                "inventory_policy": "If the exact replacement is unavailable, offer refund or store credit for the amount actually paid.",
                "shipping_policy": "First exchange shipping may be waived.",
            },
            "inventory": {
                "requested_variant": "Blue / Size L",
                "requested_variant_available": False,
                "alternate_variant": "Black / Size L",
                "alternate_variant_available": True,
                "store_credit_enabled": True,
            },
        },
        "gold": {
            "required_sections": ["order", "policy", "inventory"],
            "priority": "medium",
            "tags": ["exchange_request", "inventory_issue", "coupon_order"],
            "item_resolutions": {"item-1": "store_credit"},
            "ticket_resolution": "store_credit",
            "allowed_item_resolutions": {"item-1": ["store_credit", "refund", "request_info"]},
            "allowed_ticket_resolutions": ["store_credit", "refund", "request_info"],
            "reply_requirements": {
                "decision": ["store credit"],
                "reason": ["unavailable", "out of stock"],
                "coupon": ["amount paid", "$51", "51.00"],
                "support": ["happy to help", "sorry"],
            },
        },
    },
    "hard_partial_resolution": {
        "task_id": "hard_partial_resolution",
        "difficulty": "hard",
        "max_steps": 14,
        "objective": (
            "Resolve a multi-item return ticket with mixed eligibility. One item qualifies via a "
            "VIP defect exception, one item is a standard damaged-arrival refund, and one "
            "personalized item must be denied. The agent should produce item-level and ticket-level "
            "decisions, then draft an itemized reply."
        ),
        "available_tags": CANONICAL_TAGS,
        "customer_ticket": {
            "ticket_id": "RTN-9003",
            "channel": "email",
            "customer_id": "CUST-9003",
            "message": (
                "I need help with order ORD-9003. The AirFry Pro stopped heating. The glass storage "
                "set arrived with two broken lids. The monogram apron was not a good fit, so I would "
                "like to return all three if possible."
            ),
            "mentioned_items": [
                {
                    "item_id": "item-1",
                    "name": "AirFry Pro",
                    "issue": "Stopped heating",
                    "requested_outcome": "return_or_refund",
                },
                {
                    "item_id": "item-2",
                    "name": "Glass Storage Set",
                    "issue": "Broken lids on arrival",
                    "requested_outcome": "return_or_refund",
                },
                {
                    "item_id": "item-3",
                    "name": "Monogram Apron",
                    "issue": "Changed mind / fit issue",
                    "requested_outcome": "return_or_refund",
                },
            ],
        },
        "sections": {
            "order": {
                "order_id": "ORD-9003",
                "payment_method": "credit_card",
                "paid_amount_usd": 214.50,
                "items": [
                    {
                        "item_id": "item-1",
                        "name": "AirFry Pro",
                        "delivered_at": "2026-02-16",
                        "price_usd": 129.00,
                        "personalized": False,
                    },
                    {
                        "item_id": "item-2",
                        "name": "Glass Storage Set",
                        "delivered_at": "2026-03-23",
                        "price_usd": 45.50,
                        "personalized": False,
                    },
                    {
                        "item_id": "item-3",
                        "name": "Monogram Apron",
                        "delivered_at": "2026-03-19",
                        "price_usd": 40.00,
                        "personalized": True,
                    },
                ],
            },
            "customer": {
                "customer_tier": "gold",
                "lifetime_orders": 19,
                "lifetime_value_usd": 1678.20,
                "prior_policy_exceptions": 1,
                "prior_support_note": (
                    "Customer reported the AirFry Pro heating issue within 5 days of delivery via chat, "
                    "but the ticket was never actioned while the customer was traveling."
                ),
            },
            "policy": {
                "return_window_days": 30,
                "defect_exception_policy": "Defect claims reported promptly may receive a VIP/manual exception beyond 30 days.",
                "damaged_arrival_policy": "Damaged arrivals within 30 days are eligible for refund.",
                "personalization_policy": "Personalized items are non-returnable unless defective.",
            },
            "inventory": {
                "item-1": {
                    "replacement_available": False,
                    "note": "No replacement inventory for AirFry Pro this week.",
                },
                "item-2": {
                    "replacement_available": True,
                    "note": "Replacement stock exists, but refund is valid because damage on arrival.",
                },
                "item-3": {
                    "replacement_available": True,
                    "note": "Inventory does not override personalization policy.",
                },
            },
        },
        "gold": {
            "required_sections": ["order", "customer", "policy"],
            "priority": "high",
            "tags": ["damaged", "vip_exception", "partial_resolution", "non_returnable"],
            "item_resolutions": {
                "item-1": "refund",
                "item-2": "refund",
                "item-3": "deny",
            },
            "ticket_resolution": "partial_refund",
            "allowed_item_resolutions": {
                "item-1": ["refund", "escalate"],
                "item-2": ["refund"],
                "item-3": ["deny"],
            },
            "allowed_ticket_resolutions": ["partial_refund", "escalate"],
            "reply_requirements": {
                "airfry": ["airfry pro", "airfry"],
                "glass": ["glass storage set", "glass set"],
                "apron": ["monogram apron", "personalized"],
                "partial": ["partial", "some items", "two items"],
            },
        },
    },
}


def get_task(task_id: str) -> Dict[str, Any]:
    if task_id not in TASKS:
        raise KeyError(f"Unknown task_id: {task_id}")
    return deepcopy(TASKS[task_id])


def list_task_ids() -> List[str]:
    return list(TASKS.keys())


def task_ids_for_difficulty(difficulty: str) -> List[str]:
    return [task_id for task_id, task in TASKS.items() if task["difficulty"] == difficulty]
