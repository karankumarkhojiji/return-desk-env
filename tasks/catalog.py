from __future__ import annotations

import random
from typing import Any, Dict, List, Optional

CANONICAL_TAGS = [
    "damaged",
    "refund_request",
    "exchange_request",
    "inventory_issue",
    "coupon_order",
    "vip_exception",
    "partial_resolution",
    "non_returnable",
    "fraud_flag",
    "policy_violation",
    "return_window_exceeded",
    "wrong_item",
    "duplicate_charge",
    "escalation_required",
]

# ---------------------------------------------------------------------------
# Task 1: easy_refund
# ---------------------------------------------------------------------------

def generate_easy_refund(rng: random.Random) -> Dict[str, Any]:
    products = [
        ("BrewMaster Coffee Grinder", "with a cracked glass jar", "Cracked jar on arrival"),
        ("Ultra HD Monitor", "with a shattered screen", "Shattered screen on delivery"),
        ("SoundWave Headphones", "with a broken left ear cup", "Broken ear cup"),
        ("Quantum Keyboard", "with three missing keys", "Missing keys on arrival"),
        ("AeroBlend Mixer", "with a snapped blade", "Snapped blade on delivery"),
    ]
    product_name, issue_desc, brief_issue = rng.choice(products)
    price = round(rng.uniform(29.99, 149.99), 2)
    ticket_id = f"RTN-100{rng.randint(1, 9)}"
    customer_id = f"CUST-100{rng.randint(1, 9)}"
    order_id = f"ORD-100{rng.randint(1, 9)}"

    return {
        "task_id": "easy_refund",
        "difficulty": "easy",
        "max_steps": 10,
        "customer_sentiment": 0.2,
        "customer_follow_ups": [
            "I have photos of the damaged packaging if you need them — happy to email them over.",
            "The product arrived in this condition, I haven't used it at all. The outer box was also dented.",
            "I paid by credit card. The order confirmation is in my email, order number is on the box.",
        ],
        "objective": (
            "Resolve a standard damaged-item return ticket. Inspect the evidence, "
            "set the right priority, choose the correct item and ticket resolution, "
            "and send a policy-compliant reply before submitting."
        ),
        "available_tags": CANONICAL_TAGS,
        "customer_ticket": {
            "ticket_id": ticket_id,
            "channel": "email",
            "customer_id": customer_id,
            "message": (
                f"Hi, the {product_name} I received today arrived {issue_desc}. "
                "The unit is unusable, and I would like a refund please."
            ),
            "mentioned_items": [
                {
                    "item_id": "item-1",
                    "name": product_name,
                    "issue": brief_issue,
                    "requested_outcome": "refund",
                }
            ],
        },
        "sections": {
            "order": {
                "order_id": order_id,
                "ordered_at": "2026-03-22",
                "delivered_at": "2026-03-29",
                "payment_method": "credit_card",
                "paid_amount_usd": price,
                "items": [
                    {
                        "item_id": "item-1",
                        "name": product_name,
                        "quantity": 1,
                        "condition_report": f"Customer photo shows {brief_issue}.",
                        "personalized": False,
                    }
                ],
            },
            "customer": {
                "customer_tier": "standard",
                "lifetime_orders": rng.randint(1, 5),
                "lifetime_value_usd": round(price + rng.uniform(20, 100), 2),
                "prior_policy_exceptions": 0,
            },
            "policy": {
                "return_window_days": 30,
                "defect_policy": "Damaged or defective items delivered within 30 days are eligible for refund or exchange.",
                "preference_policy": "When the customer clearly requests a refund and the policy allows it, refund is preferred.",
                "logistics_policy": "Return shipment is not required when the product is unsafe or unusable on arrival.",
            },
            "inventory": {
                "item-1": {
                    "available_units": rng.randint(10, 50),
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
            "fraud_risk": False,
        },
    }


# ---------------------------------------------------------------------------
# Task 2: medium_exchange
# ---------------------------------------------------------------------------

def generate_medium_exchange(rng: random.Random) -> Dict[str, Any]:
    products = [
        ("FlexFit Hoodie", "size M", "size L", "blue", "black"),
        ("ActiveWear Jacket", "size S", "size M", "red", "grey"),
        ("Runner Shoes", "size 9", "size 10", "white", "black"),
        ("ProSport Shorts", "size XS", "size S", "green", "navy"),
    ]
    product_name, old_size, new_size, target_color, alt_color = rng.choice(products)
    coupons = ["SPRING15", "WELCOME10", "SAVE20", "SUMMER25"]
    coupon = rng.choice(coupons)
    price = round(rng.uniform(40.00, 80.00), 2)
    ticket_id = f"RTN-20{rng.randint(10, 99)}"
    customer_id = f"CUST-20{rng.randint(10, 99)}"
    order_id = f"ORD-20{rng.randint(10, 99)}"

    return {
        "task_id": "medium_exchange",
        "difficulty": "medium",
        "max_steps": 12,
        "customer_sentiment": 0.4,
        "customer_follow_ups": [
            f"I used the {coupon} coupon, so I paid the discounted price, not the full retail price.",
            f"I'm happy with store credit as long as it reflects what I actually paid after the coupon.",
            "The item hasn't been opened. It's still in the original packaging, ready for return.",
        ],
        "objective": (
            "Handle a size-exchange request where exact inventory is unavailable and the order "
            "used a coupon. The agent must inspect inventory, apply the coupon-amount-paid policy, "
            "and offer store credit for the amount actually paid."
        ),
        "available_tags": CANONICAL_TAGS,
        "customer_ticket": {
            "ticket_id": ticket_id,
            "channel": "chat",
            "customer_id": customer_id,
            "message": (
                f"I ordered the {product_name} in {old_size}, but I need {new_size}. "
                f"If the {target_color} {new_size} is not available, store credit is fine. "
                f"I used the {coupon} coupon on this order."
            ),
            "mentioned_items": [
                {
                    "item_id": "item-1",
                    "name": f"{product_name} - {target_color.capitalize()} / {old_size.capitalize()}",
                    "issue": f"Needs exchange to {new_size}",
                    "requested_outcome": "exchange_or_store_credit",
                }
            ],
        },
        "sections": {
            "order": {
                "order_id": order_id,
                "ordered_at": "2026-03-18",
                "delivered_at": "2026-03-23",
                "payment_method": "credit_card",
                "coupon_code": coupon,
                "paid_amount_usd": price,
                "items": [
                    {
                        "item_id": "item-1",
                        "name": f"{product_name} - {target_color.capitalize()} / {old_size.capitalize()}",
                        "quantity": 1,
                        "condition_report": "Unworn, original tags attached.",
                        "personalized": False,
                    }
                ],
            },
            "customer": {
                "customer_tier": "standard",
                "lifetime_orders": rng.randint(3, 10),
                "lifetime_value_usd": round(price * rng.randint(3, 6), 2),
                "prior_policy_exceptions": 0,
            },
            "policy": {
                "return_window_days": 30,
                "exchange_policy": "Unworn apparel may be exchanged within 30 days.",
                "inventory_policy": "If the exact replacement is unavailable, offer store credit for the amount actually paid.",
                "coupon_policy": "Store credit is issued for the discounted price paid, not the original retail price.",
                "shipping_policy": "First exchange shipping may be waived.",
            },
            "inventory": {
                "requested_variant": f"{target_color.capitalize()} / {new_size.capitalize()}",
                "requested_variant_available": False,
                "alternate_variant": f"{alt_color.capitalize()} / {new_size.capitalize()}",
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
                "coupon": ["amount paid", f"${price:.2f}", f"{price:.2f}"],
                "support": ["happy to help", "sorry"],
            },
            "fraud_risk": False,
        },
    }


# ---------------------------------------------------------------------------
# Task 3: hard_partial_resolution
# ---------------------------------------------------------------------------

def generate_hard_partial(rng: random.Random) -> Dict[str, Any]:
    vip_items = ["AirFry Pro", "Espresso Machine", "Robot Vacuum", "Smart Projector"]
    vip_item = rng.choice(vip_items)
    dmg_items = ["Glass Storage Set", "Ceramic Plates", "Wine Glasses", "Crystal Vase"]
    dmg_item = rng.choice(dmg_items)
    pers_items = ["Monogram Apron", "Engraved Watch", "Custom Phone Case", "Personalized Pillow"]
    pers_item = rng.choice(pers_items)

    ticket_id = f"RTN-900{rng.randint(1, 9)}"
    customer_id = f"CUST-900{rng.randint(1, 9)}"
    order_id = f"ORD-900{rng.randint(1, 9)}"

    # Date conflict: note says "5 days" but actual dates show 15 days — agent must reason
    item1_price = round(rng.uniform(100.0, 250.0), 2)
    item2_price = round(rng.uniform(20.0, 60.0), 2)
    item3_price = round(rng.uniform(30.0, 80.0), 2)
    total_paid = round(item1_price + item2_price + item3_price, 2)

    return {
        "task_id": "hard_partial_resolution",
        "difficulty": "hard",
        "max_steps": 15,
        "customer_sentiment": 0.1,
        "customer_follow_ups": [
            "I contacted support within 5 days of delivery. The ticket reference from that call is in my email.",
            "The monogrammed item was a gift — I understand it's non-returnable, but the other two items are clearly defective.",
            "I can provide the unboxing video that shows all three items in the same delivery from the same order.",
        ],
        "objective": (
            "Resolve a multi-item return ticket with mixed eligibility. One item qualifies via a "
            "VIP defect exception (despite being outside the 30-day window — the customer reported "
            "it promptly but the ticket was lost). One item is a standard damaged-arrival refund. "
            "One personalized item must be denied. The agent must reconcile conflicting dates in "
            "the order record vs. the support note, then produce item-level decisions and "
            "an itemized reply."
        ),
        "available_tags": CANONICAL_TAGS,
        "customer_ticket": {
            "ticket_id": ticket_id,
            "channel": "email",
            "customer_id": customer_id,
            "message": (
                f"I need help with order {order_id}. The {vip_item} stopped heating after a month. "
                f"The {dmg_item} arrived broken. The {pers_item} was not what I expected — "
                "I'd like to return all three if possible."
            ),
            "mentioned_items": [
                {"item_id": "item-1", "name": vip_item, "issue": "Stopped working", "requested_outcome": "return_or_refund"},
                {"item_id": "item-2", "name": dmg_item, "issue": "Broken on arrival", "requested_outcome": "return_or_refund"},
                {"item_id": "item-3", "name": pers_item, "issue": "Changed mind", "requested_outcome": "return_or_refund"},
            ],
        },
        "sections": {
            "order": {
                "order_id": order_id,
                "payment_method": "credit_card",
                "paid_amount_usd": total_paid,
                "items": [
                    {"item_id": "item-1", "name": vip_item, "delivered_at": "2026-02-01", "price_usd": item1_price, "personalized": False},
                    {"item_id": "item-2", "name": dmg_item, "delivered_at": "2026-03-23", "price_usd": item2_price, "personalized": False},
                    {"item_id": "item-3", "name": pers_item, "delivered_at": "2026-03-19", "price_usd": item3_price, "personalized": True},
                ],
            },
            "customer": {
                "customer_tier": "gold",
                "lifetime_orders": rng.randint(15, 30),
                "lifetime_value_usd": round(total_paid * rng.randint(4, 10), 2),
                "prior_policy_exceptions": 1,
                "prior_support_note": (
                    f"Customer reported the {vip_item} issue on 2026-02-06 (5 days after delivery) "
                    "via chat, but the ticket was never actioned while the customer was traveling. "
                    "The system logged this on 2026-02-17 — 16 days after delivery. "
                    "Agent must trust the original report date, not the system log date."
                ),
            },
            "policy": {
                "return_window_days": 30,
                "defect_exception_policy": "Defect claims reported within 7 days of delivery qualify for a VIP exception even if actioned later.",
                "damaged_arrival_policy": "Damaged arrivals within 30 days are eligible for refund.",
                "personalization_policy": "Personalized items are non-returnable unless defective.",
            },
            "inventory": {
                "item-1": {"replacement_available": False, "note": f"No replacement for {vip_item} this week."},
                "item-2": {"replacement_available": True, "note": "Replacement exists, but refund is valid for damaged arrival."},
                "item-3": {"replacement_available": True, "note": "Inventory does not override personalization policy."},
            },
        },
        "gold": {
            "required_sections": ["order", "customer", "policy"],
            "priority": "high",
            "tags": ["damaged", "vip_exception", "partial_resolution", "non_returnable"],
            "item_resolutions": {"item-1": "refund", "item-2": "refund", "item-3": "deny"},
            "ticket_resolution": "partial_refund",
            "allowed_item_resolutions": {"item-1": ["refund", "escalate"], "item-2": ["refund"], "item-3": ["deny"]},
            "allowed_ticket_resolutions": ["partial_refund", "escalate"],
            "reply_requirements": {
                "vip_item": [vip_item.lower(), vip_item.split()[0].lower()],
                "dmg_item": [dmg_item.lower(), dmg_item.split()[-1].lower()],
                "pers_item": [pers_item.lower(), "personalized"],
                "partial": ["partial", "some items", "two items"],
            },
            "fraud_risk": False,
        },
    }


# ---------------------------------------------------------------------------
# Task 4: expired_return  (NEW — deny scenario)
# ---------------------------------------------------------------------------

def generate_expired_return(rng: random.Random) -> Dict[str, Any]:
    products = [
        ("Canvas Backpack", "changed my mind", "Lifestyle change"),
        ("Wireless Charger", "no longer needed", "No longer needed"),
        ("Bamboo Cutting Board", "not my style", "Style preference"),
        ("Smart Plug Set", "bought the wrong model", "Wrong model"),
    ]
    product_name, reason, brief_reason = rng.choice(products)

    days_late = rng.randint(35, 75)
    price = round(rng.uniform(19.99, 89.99), 2)
    ticket_id = f"RTN-400{rng.randint(1, 9)}"
    customer_id = f"CUST-400{rng.randint(1, 9)}"
    order_id = f"ORD-400{rng.randint(1, 9)}"
    prior_returns = rng.randint(2, 5)

    return {
        "task_id": "expired_return",
        "difficulty": "medium",
        "max_steps": 10,
        "customer_sentiment": 0.3,
        "customer_follow_ups": [
            "I was told during the sale period that returns were accepted within 60 days. Can you check that?",
            "I understand the policy but the product has a manufacturing defect, surely that changes things?",
            "If a refund is not possible, can I get store credit at least? I'm a long-term customer.",
        ],
        "objective": (
            "A customer is requesting a return outside the 30-day policy window. "
            "The agent must inspect the order to confirm the delivery date and calculate "
            "days since delivery, check the policy, and deny the request with a polite "
            "explanation. Issuing a refund here violates policy."
        ),
        "available_tags": CANONICAL_TAGS,
        "customer_ticket": {
            "ticket_id": ticket_id,
            "channel": "email",
            "customer_id": customer_id,
            "message": (
                f"Hi, I purchased the {product_name} {days_late} days ago and I'd like to return it — "
                f"I {reason}. Can you process a refund for me?"
            ),
            "mentioned_items": [
                {
                    "item_id": "item-1",
                    "name": product_name,
                    "issue": brief_reason,
                    "requested_outcome": "refund",
                }
            ],
        },
        "sections": {
            "order": {
                "order_id": order_id,
                "ordered_at": "2026-01-15",
                "delivered_at": "2026-01-20",
                "current_date": "2026-03-28",
                "days_since_delivery": days_late,
                "payment_method": "credit_card",
                "paid_amount_usd": price,
                "items": [
                    {
                        "item_id": "item-1",
                        "name": product_name,
                        "quantity": 1,
                        "condition_report": "No damage reported. Customer changed mind.",
                        "personalized": False,
                    }
                ],
            },
            "customer": {
                "customer_tier": "standard",
                "lifetime_orders": rng.randint(3, 10),
                "lifetime_value_usd": round(price * rng.randint(2, 5), 2),
                "prior_returns_this_quarter": prior_returns,
                "prior_policy_exceptions": 0,
            },
            "policy": {
                "return_window_days": 30,
                "window_policy": "Returns must be initiated within 30 days of delivery. Requests outside this window are not eligible.",
                "exception_policy": "Exceptions require manager approval and are reserved for defective items only.",
                "preference_policy": "Change-of-mind returns are not accepted outside the return window under any circumstances.",
            },
            "inventory": {
                "item-1": {
                    "available_units": rng.randint(5, 30),
                    "note": "Inventory status does not affect return eligibility.",
                }
            },
        },
        "gold": {
            "required_sections": ["order", "policy"],
            "priority": "low",
            "tags": ["return_window_exceeded", "policy_violation"],
            "item_resolutions": {"item-1": "deny"},
            "ticket_resolution": "deny",
            "allowed_item_resolutions": {"item-1": ["deny", "request_info"]},
            "allowed_ticket_resolutions": ["deny", "request_info"],
            "reply_requirements": {
                "policy": ["30-day", "30 day", "return window", "policy"],
                "denial": ["unable", "cannot", "not eligible", "outside"],
                "empathy": ["understand", "sorry", "apologize"],
            },
            "fraud_risk": False,
        },
    }


# ---------------------------------------------------------------------------
# Task 5: fraud_risk  (NEW — escalate, do NOT refund)
# ---------------------------------------------------------------------------

def generate_fraud_risk(rng: random.Random) -> Dict[str, Any]:
    products = [
        ("Gaming Laptop", 1299.99),
        ("4K OLED TV", 1899.00),
        ("Professional Camera Kit", 1150.00),
        ("High-End Smartwatch", 799.99),
    ]
    product_name, price = rng.choice(products)
    price = round(price + rng.uniform(-50, 50), 2)

    fraud_patterns = [
        ("new account created 3 days ago", "account_age_flag"),
        ("5 refund requests in the last 30 days", "refund_velocity_flag"),
        ("delivery address does not match billing address", "address_mismatch_flag"),
    ]
    # Pick 2 fraud signals to make it undeniable
    chosen = rng.sample(fraud_patterns, k=2)
    fraud_signals = [p[0] for p in chosen]
    fraud_signal_keys = [p[1] for p in chosen]

    ticket_id = f"RTN-110{rng.randint(1, 9)}"
    customer_id = f"CUST-110{rng.randint(1, 9)}"
    order_id = f"ORD-110{rng.randint(1, 9)}"

    return {
        "task_id": "fraud_risk",
        "difficulty": "hard",
        "max_steps": 12,
        "customer_sentiment": 0.3,
        "customer_follow_ups": [
            "I definitely did not receive the package. My neighbour was home all day and didn't see any delivery either.",
            "I have been a customer for years and never had a problem. This is very frustrating.",
            "Can you file a carrier investigation? I'm happy to cooperate with any process needed.",
        ],
        "objective": (
            "A customer is claiming a high-value item never arrived, but the order record shows "
            "confirmed delivery with a signature. Multiple fraud signals are present. "
            "The agent must inspect the evidence, flag the fraud risk, and ESCALATE — "
            "issuing a refund on a fraud-risk order is a severe policy violation."
        ),
        "available_tags": CANONICAL_TAGS,
        "customer_ticket": {
            "ticket_id": ticket_id,
            "channel": "email",
            "customer_id": customer_id,
            "message": (
                f"Hello, I ordered the {product_name} (order {order_id}) and it never arrived. "
                "The tracking says delivered but I never received it. "
                "I need an immediate full refund please."
            ),
            "mentioned_items": [
                {
                    "item_id": "item-1",
                    "name": product_name,
                    "issue": "Claims non-delivery",
                    "requested_outcome": "refund",
                }
            ],
        },
        "sections": {
            "order": {
                "order_id": order_id,
                "payment_method": "credit_card",
                "paid_amount_usd": price,
                "delivery_status": "DELIVERED",
                "delivery_confirmation": "Signature on file",
                "delivered_at": "2026-04-01",
                "items": [
                    {
                        "item_id": "item-1",
                        "name": product_name,
                        "quantity": 1,
                        "price_usd": price,
                        "personalized": False,
                    }
                ],
            },
            "customer": {
                "customer_tier": "standard",
                "lifetime_orders": 1,
                "lifetime_value_usd": price,
                "prior_refund_requests_30d": rng.randint(3, 6),
                "prior_policy_exceptions": 0,
                "fraud_signals": fraud_signals,
                "fraud_score": round(rng.uniform(0.78, 0.95), 2),
                "account_age_days": rng.randint(2, 8),
            },
            "policy": {
                "return_window_days": 30,
                "fraud_policy": (
                    "Orders flagged with a fraud score above 0.75 must be escalated to the "
                    "fraud review team. Do NOT process refunds on flagged accounts."
                ),
                "escalation_policy": "High-value items with delivery confirmation and fraud signals must always be escalated.",
            },
            "inventory": {
                "item-1": {
                    "available_units": 3,
                    "note": "Inventory does not affect fraud escalation.",
                }
            },
        },
        "gold": {
            "required_sections": ["order", "customer", "policy"],
            "priority": "urgent",
            "tags": ["fraud_flag", "escalation_required"],
            "item_resolutions": {"item-1": "escalate"},
            "ticket_resolution": "escalate",
            "allowed_item_resolutions": {"item-1": ["escalate"]},
            "allowed_ticket_resolutions": ["escalate"],
            "reply_requirements": {
                "acknowledge": ["received", "understand", "noted"],
                "escalate": ["escalated", "team", "specialist", "review"],
                "timeline": ["24 hours", "48 hours", "1-2 business days", "soon"],
            },
            "fraud_risk": True,
        },
    }


# ---------------------------------------------------------------------------
# Task 6: wrong_item_sent  (NEW — wrong product delivered)
# ---------------------------------------------------------------------------

def generate_wrong_item_sent(rng: random.Random) -> Dict[str, Any]:
    ordered = [
        ("Blue Running Shoes Size 10", "Red Casual Sneakers Size 9"),
        ("Black Leather Wallet", "Brown Canvas Wallet"),
        ("Stainless Steel Water Bottle 1L", "Plastic Sports Bottle 500ml"),
        ("Pro Wireless Mouse", "Basic Wired Mouse"),
    ]
    ordered_item, received_item = rng.choice(ordered)
    price = round(rng.uniform(29.99, 129.99), 2)

    ticket_id = f"RTN-500{rng.randint(1, 9)}"
    customer_id = f"CUST-500{rng.randint(1, 9)}"
    order_id = f"ORD-500{rng.randint(1, 9)}"

    replacement_available = rng.choice([True, False])

    return {
        "task_id": "wrong_item_sent",
        "difficulty": "medium",
        "max_steps": 12,
        "customer_sentiment": 0.1,
        "customer_follow_ups": [
            "I have the delivery box here — the label on the outside shows my order number but the item inside is wrong.",
            "I've attached photos of the item I received. It's definitely not what I ordered.",
            "I need the correct item as soon as possible. A refund is also fine if you can't ship it quickly.",
        ],
        "objective": (
            "A customer received a completely different product from what they ordered. "
            "The agent must confirm the mismatch via the order record, then offer the correct item "
            "if in stock (exchange) or a full refund if stock is unavailable. "
            "No return shipment is required for wrong-item-sent cases."
        ),
        "available_tags": CANONICAL_TAGS,
        "customer_ticket": {
            "ticket_id": ticket_id,
            "channel": "chat",
            "customer_id": customer_id,
            "message": (
                f"I ordered the {ordered_item} but received the {received_item} instead. "
                "This is clearly a fulfilment error. I'd like the correct item or a full refund."
            ),
            "mentioned_items": [
                {
                    "item_id": "item-1",
                    "name": ordered_item,
                    "issue": f"Received '{received_item}' instead",
                    "requested_outcome": "exchange_or_refund",
                }
            ],
        },
        "sections": {
            "order": {
                "order_id": order_id,
                "ordered_at": "2026-03-15",
                "delivered_at": "2026-03-21",
                "payment_method": "credit_card",
                "paid_amount_usd": price,
                "items": [
                    {
                        "item_id": "item-1",
                        "name": ordered_item,
                        "quantity": 1,
                        "condition_report": f"Fulfilment record shows '{received_item}' shipped in error.",
                        "personalized": False,
                    }
                ],
            },
            "customer": {
                "customer_tier": rng.choice(["standard", "silver"]),
                "lifetime_orders": rng.randint(2, 8),
                "lifetime_value_usd": round(price * rng.randint(2, 6), 2),
                "prior_policy_exceptions": 0,
            },
            "policy": {
                "return_window_days": 30,
                "wrong_item_policy": "Wrong items sent due to fulfilment error qualify for immediate exchange or full refund.",
                "return_policy": "No return shipment required for fulfilment errors — customer keeps the wrong item.",
                "shipping_policy": "Free replacement shipping for all fulfilment errors.",
            },
            "inventory": {
                "item-1": {
                    "correct_item_available": replacement_available,
                    "note": (
                        f"Correct item ({ordered_item}) {'is in stock' if replacement_available else 'is OUT OF STOCK'}."
                    ),
                }
            },
        },
        "gold": {
            "required_sections": ["order", "policy", "inventory"],
            "priority": "high",
            "tags": ["wrong_item", "refund_request"] if not replacement_available else ["wrong_item", "exchange_request"],
            "item_resolutions": {"item-1": "refund" if not replacement_available else "exchange"},
            "ticket_resolution": "refund" if not replacement_available else "exchange",
            "allowed_item_resolutions": {"item-1": ["refund", "exchange"]},
            "allowed_ticket_resolutions": ["refund", "exchange"],
            "reply_requirements": {
                "acknowledge": ["sent the wrong", "incorrect item", "fulfilment error", "ordered"],
                "decision": ["refund", "exchange", "correct item"],
                "logistics": ["no return", "keep", "do not return"],
            },
            "fraud_risk": False,
        },
    }



# ---------------------------------------------------------------------------
# Task 7: extreme_chargeback  (NEW — 4th difficulty tier inspired by Repo 9)
# ---------------------------------------------------------------------------

def generate_extreme_chargeback(rng: random.Random) -> Dict[str, Any]:
    """
    Extreme difficulty: A corporate client filed a formal bank chargeback on a
    5-item bulk order. The agent must reconcile conflicting records across 4 data
    sources (payment gateway, carrier, insurance policy, and customer account),
    then produce item-level resolutions that satisfy insurance coverage limits,
    correctly reject the non-covered items, and draft a formal chargeback response
    letter within a strict step budget.

    What makes this extreme:
    - 5 items, each with different resolution paths
    - Insurance only covers 3 of 5 items (agent must check policy to know which)
    - Carrier confirmation exists for 4 items but not 1 (claimed non-delivery)
    - The non-delivery claim is from the same customer who has a fraud_score=0.72
    - Cross-currency order: exchange rate must be applied for USD refund amounts
    - Agent must NOT refund the non-covered items (policy violation)
    - Agent must flag fraud suspicion for the non-delivery item
    - Correct ticket resolution: partial_refund (on 2 covered+damaged items only)
    """
    currencies = [("EUR", round(rng.uniform(0.88, 0.95), 4)),
                  ("GBP", round(rng.uniform(0.75, 0.82), 4)),
                  ("CAD", round(rng.uniform(1.31, 1.38), 4))]
    currency, rate = rng.choice(currencies)

    items_pool = [
        ("Industrial Printer Cartridge Pack", "business_supply"),
        ("Conference Room Webcam Kit",        "electronics"),
        ("Ergonomic Office Chair",             "furniture"),
        ("Standing Desk Converter",            "furniture"),
        ("Noise-Cancelling Headsets x5",       "electronics"),
    ]
    rng.shuffle(items_pool)
    selected = items_pool[:5]

    prices_local = [round(rng.uniform(80, 350), 2) for _ in range(5)]
    prices_usd   = [round(p / rate, 2) for p in prices_local]
    total_local  = round(sum(prices_local), 2)
    total_usd    = round(sum(prices_usd), 2)

    ticket_id   = f"RTN-EXT-{rng.randint(100, 999)}"
    customer_id = f"CORP-{rng.randint(1000, 9999)}"
    order_id    = f"BULK-{rng.randint(10000, 99999)}"

    # Items 0,1: damaged-in-transit (covered by insurance) → partial_refund
    # Items 2,3: delivered fine, customer changed mind (NOT covered) → deny
    # Item  4:   claimed non-delivery, but carrier has proof of delivery → escalate (fraud flag)
    insurance_items = [selected[0][0], selected[1][0]]
    denied_items    = [selected[2][0], selected[3][0]]
    fraud_item      = selected[4][0]

    return {
        "task_id": "extreme_chargeback",
        "difficulty": "extreme",
        "max_steps": 18,
        "customer_sentiment": -0.6,
        "customer_follow_ups": [
            "We have photographic evidence of the damaged items and packaging. Our legal team is standing by.",
            "The carrier tracking shows 'delivered' but our receiving dock has no record of accepting this shipment.",
            "We need this resolved within 48 hours or we will instruct our bank to complete the chargeback without your response.",
        ],
        "objective": (
            "A corporate client filed a formal bank chargeback on a 5-item bulk order. "
            "Inspect all 4 data sources: payment record, carrier report, insurance policy, "
            "and customer account. Determine which items are covered, which must be denied, "
            "and flag the suspected fraudulent non-delivery claim. Apply the exchange rate "
            "to compute correct USD refund amounts. Only items 1-2 qualify for partial refund; "
            "items 3-4 must be denied; item 5 must be escalated. Draft a formal response."
        ),
        "available_tags": CANONICAL_TAGS,
        "customer_ticket": {
            "ticket_id": ticket_id,
            "channel": "email",
            "customer_id": customer_id,
            "account_type": "corporate",
            "message": (
                f"We are filing a formal chargeback dispute for order {order_id} "
                f"({currency} {total_local:.2f} / USD {total_usd:.2f}). "
                f"Two items arrived damaged ({selected[0][0]}, {selected[1][0]}), "
                f"two items are no longer required by our team ({selected[2][0]}, {selected[3][0]}), "
                f"and one item was never received ({selected[4][0]}). "
                "We expect full resolution within 48 hours or we will proceed with the bank dispute."
            ),
            "mentioned_items": [
                {"item_id": "item-1", "name": selected[0][0], "issue": "Damaged in transit",     "requested_outcome": "refund"},
                {"item_id": "item-2", "name": selected[1][0], "issue": "Damaged in transit",     "requested_outcome": "refund"},
                {"item_id": "item-3", "name": selected[2][0], "issue": "No longer required",     "requested_outcome": "refund"},
                {"item_id": "item-4", "name": selected[3][0], "issue": "No longer required",     "requested_outcome": "refund"},
                {"item_id": "item-5", "name": selected[4][0], "issue": "Claims non-delivery",  "requested_outcome": "refund"},
            ],
        },
        "sections": {
            "order": {
                "order_id": order_id,
                "currency": currency,
                "exchange_rate_to_usd": rate,
                "paid_amount_local": total_local,
                "paid_amount_usd": total_usd,
                "payment_method": "corporate_card",
                "chargeback_filed": True,
                "chargeback_deadline_hours": 48,
                "items": [
                    {"item_id": "item-1", "name": selected[0][0], "price_local": prices_local[0], "price_usd": prices_usd[0], "delivered": True,  "carrier_photo_evidence": "Damaged packaging"},
                    {"item_id": "item-2", "name": selected[1][0], "price_local": prices_local[1], "price_usd": prices_usd[1], "delivered": True,  "carrier_photo_evidence": "Dented transit box"},
                    {"item_id": "item-3", "name": selected[2][0], "price_local": prices_local[2], "price_usd": prices_usd[2], "delivered": True,  "carrier_photo_evidence": "Clean delivery, no damage"},
                    {"item_id": "item-4", "name": selected[3][0], "price_local": prices_local[3], "price_usd": prices_usd[3], "delivered": True,  "carrier_photo_evidence": "Signed for by reception"},
                    {"item_id": "item-5", "name": selected[4][0], "price_local": prices_local[4], "price_usd": prices_usd[4], "delivered": True,  "carrier_photo_evidence": "GPS confirmed, signature on file"},
                ],
            },
            "customer": {
                "customer_tier": "corporate",
                "account_age_days": rng.randint(15, 40),
                "lifetime_orders": rng.randint(1, 3),
                "prior_chargebacks_filed": rng.randint(2, 4),
                "fraud_signals": ["high chargeback velocity", "new account", "corporate card mismatch"],
                "fraud_score": round(rng.uniform(0.70, 0.85), 2),
            },
            "policy": {
                "transit_damage_policy": "Items damaged in transit are covered for partial refund if photographic evidence exists.",
                "change_of_mind_policy": "Corporate bulk orders are non-refundable for change-of-mind after dispatch.",
                "insurance_coverage": (
                    f"Transit insurance covers up to 2 items per order: {selected[0][0]} and {selected[1][0]}. "
                    "Additional items in the same shipment are not covered unless separately insured."
                ),
                "non_delivery_policy": "Non-delivery claims require carrier investigation. If carrier confirms delivery, the claim must be escalated to fraud review.",
                "chargeback_response_policy": "All chargeback disputes must be responded to within 48 hours with itemized justification per item.",
                "exchange_rate_policy": f"Refunds are issued in USD at the exchange rate at time of order ({currency}/USD = {rate}).",
            },
            "inventory": {
                "item-1": {"replacement_available": False, "note": "Out of stock — refund only."},
                "item-2": {"replacement_available": False, "note": "Out of stock — refund only."},
                "item-3": {"note": "Not eligible for return — change of mind on bulk order."},
                "item-4": {"note": "Not eligible for return — change of mind on bulk order."},
                "item-5": {"note": "Carrier confirmed delivery. Escalate to fraud review."},
            },
        },
        "gold": {
            "required_sections": ["order", "customer", "policy", "inventory"],
            "priority": "urgent",
            "tags": ["damaged", "fraud_flag", "partial_resolution", "escalation_required", "policy_violation"],
            "item_resolutions": {
                "item-1": "partial_refund",
                "item-2": "partial_refund",
                "item-3": "deny",
                "item-4": "deny",
                "item-5": "escalate",
            },
            "ticket_resolution": "partial_refund",
            "allowed_item_resolutions": {
                "item-1": ["partial_refund", "refund"],
                "item-2": ["partial_refund", "refund"],
                "item-3": ["deny"],
                "item-4": ["deny"],
                "item-5": ["escalate"],
            },
            "allowed_ticket_resolutions": ["partial_refund"],
            "reply_requirements": {
                "items_covered": [selected[0][0].split()[0].lower(), selected[1][0].split()[0].lower(), "damaged", "covered"],
                "items_denied": ["change of mind", "non-refundable", "bulk order", "cannot"],
                "fraud_item": ["escalated", "investigation", "carrier", "review"],
                "timeline": ["48 hours", "48-hour", "within 48"],
            },
            "fraud_risk": True,
        },
    }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_TASK_DEFINITIONS = {
    "easy_refund":             {"difficulty": "easy",    "generator": generate_easy_refund},
    "medium_exchange":         {"difficulty": "medium",  "generator": generate_medium_exchange},
    "hard_partial_resolution": {"difficulty": "hard",    "generator": generate_hard_partial},
    "expired_return":          {"difficulty": "medium",  "generator": generate_expired_return},
    "fraud_risk":              {"difficulty": "hard",    "generator": generate_fraud_risk},
    "wrong_item_sent":         {"difficulty": "medium",  "generator": generate_wrong_item_sent},
    "extreme_chargeback":      {"difficulty": "extreme", "generator": generate_extreme_chargeback},
}


def get_task(task_id: str, seed: Optional[int] = None) -> Dict[str, Any]:
    if task_id not in _TASK_DEFINITIONS:
        raise KeyError(f"Unknown task_id: {task_id!r}. Available: {list(_TASK_DEFINITIONS)}")
    rng = random.Random(seed) if seed is not None else random.Random()
    return _TASK_DEFINITIONS[task_id]["generator"](rng)


def list_task_ids() -> List[str]:
    return list(_TASK_DEFINITIONS.keys())


def task_ids_for_difficulty(difficulty: str) -> List[str]:
    return [tid for tid, meta in _TASK_DEFINITIONS.items() if meta["difficulty"] == difficulty]
