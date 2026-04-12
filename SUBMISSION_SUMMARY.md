# ReturnDeskEnv — Submission Summary

> **Meta PyTorch OpenEnv Hackathon** | Submission date: April 12, 2026

---

## What This Environment Is

**ReturnDeskEnv** is a procedurally generated, multi-task reinforcement learning environment simulating a real e-commerce returns and fraud operations desk. An agent must process customer support tickets by inspecting evidence, making policy-compliant decisions, engaging in multi-turn dialogue with the customer, and drafting replies — under the pressure of a step budget, multi-objective shaped rewards, and severe penalties for policy violations.

The domain is deliberately chosen to mirror high-stakes, real-world enterprise operations where:
- Wrong decisions have **financial consequences** (unnecessary refunds)
- Some decisions are **irreversible** (refunding a fraud-risk order = -0.40 score penalty)
- The correct answer changes depending on **hidden information** the agent must actively uncover
- Multiple valid resolutions exist, but some are **more cost-efficient** than others
- The **customer can provide additional information** when asked — changing the optimal decision

---

## Task Catalog — 4 Difficulty Tiers, 7 Tasks

| Task ID | Difficulty | Correct Action | What Makes It Hard |
|---|---|---|---|
| `easy_refund` | Easy | refund | Damaged item within window — baseline |
| `medium_exchange` | Medium | store_credit | No stock + coupon → must not refund full price |
| `expired_return` | Medium | **deny** | Outside 30-day window — refunding is a policy violation |
| `wrong_item_sent` | Medium | exchange OR refund | Correct answer depends on live inventory state |
| `hard_partial_resolution` | Hard | partial_refund | 3 items, VIP exception, contradictory dates across two sources |
| `fraud_risk` | Hard | **escalate** | Refunding triggers -0.40 penalty; fraud must be flagged first |
| `extreme_chargeback` | **Extreme** ⭐ | partial_refund (5 items) | Cross-currency, insurance coverage, 4 data sources, fraud claim |

Every `reset()` call generates a **unique episode** using `random.Random(seed)`. Product names, prices, customer IDs, order IDs, currencies, and exchange rates are all randomised — preventing memorisation.

---

## Key Design Decisions

### 1. Fraud Detection — The Hardest Task

The `fraud_risk` task is the primary differentiator. Most frontier models are biased toward helpful actions. When a customer claims non-delivery on a high-value item, a GPT-class model's natural instinct is to refund. But the correct action here is:
1. Inspect the order (delivery confirmed with carrier signature)
2. Inspect the customer record (fraud signals: new account, 5 prior refund attempts, address mismatch)
3. Call `flag_fraud` before touching the resolution
4. Escalate — **never refund**

Processing a refund on this task incurs a **-0.40 score penalty**, pushing an overconfident wrong agent below `0.01`.

### 2. Extreme Difficulty Tier — 4th Tier

The `extreme_chargeback` task requires an agent to:
- Inspect **4 data sources** (order, customer, policy, inventory)
- Apply **insurance coverage limits** — only 2 of 5 items are covered
- Recognize a **fraudulent non-delivery claim** (carrier GPS + signature on file)
- Handle **cross-currency pricing** (EUR/GBP/CAD with dynamic exchange rates)
- Produce **5 different item-level resolutions**: `partial_refund × 2`, `deny × 2`, `escalate × 1`
- Draft a formal chargeback response citing policy per item

No existing competitor submission has a task requiring this level of multi-source reconciliation.

### 3. Multi-Turn Customer Dialogue (New)

Every task now includes a pool of 3 realistic customer follow-up messages. The new `ask_customer` action triggers a customer response mid-episode, giving the agent new information to work with. Setting ticket resolution to `request_info` also triggers a follow-up.

```json
{"action_type": "ask_customer", "question": "Can you provide proof of the damage?"}
```

Customer response is appended to `customer_messages` in the next observation:
```json
{"role": "customer", "text": "I have photos of the damaged packaging if you need them."}
```

This turns each episode into a real dialogue — the agent can gather clarifying information before making irreversible decisions.

### 4. Curriculum Difficulty Selection (New)

The environment supports automatic difficulty progression through `CurriculumState`. A rolling window of scores determines which difficulty tier to sample next:

| Rolling Mean | Selected Difficulty |
|---|---|
| < 0.55 | easy |
| 0.55–0.70 | medium |
| 0.70–0.82 | hard |
| ≥ 0.82 | extreme |

The API now includes `GET /api/curriculum` to show the current training state in real-time.

### 5. Customer Sentiment Dynamics (New)

`customer_sentiment` is no longer a static float — it changes throughout the episode:
- Starts at a task-specific baseline (e.g. -0.60 for extreme_chargeback, +0.40 for medium_exchange)
- Decays **-0.04/step** after step 5 (slow agent = frustrated customer)
- Rises **+0.06** on each correct item resolution
- Drops **-0.04** on wrong item resolution
- Drops **-0.08** when fraud is flagged (customer feels accused)

This creates a continuous feedback signal every step, rewarding agents that resolve tickets quickly and correctly.

### 6. Episode Replay Store (New)

Every completed episode is automatically saved to an in-memory replay store (bounded at 100 episodes). The replay includes every step: action taken, reward received, sentiment value, and agent note.

```
GET /api/replay/{episode_id}   → full trajectory JSON
GET /api/replay                → list of all stored episode IDs
```

This allows human reviewers and agents alike to inspect exactly what went wrong in any episode — without any additional code.

### 7. Multi-Episode Training Harness (New)

`trainer.py` is a standalone training harness that runs N episodes of the deterministic policy, records performance, and generates learning curve plots:

```bash
python trainer.py --episodes 70 --mode round_robin
python trainer.py --episodes 100 --mode curriculum
```

Output: `outputs/baseline_scores.json` (full results) + `outputs/training_curve.png` (matplotlib, dark theme, colour-coded by difficulty).

### 8. Multi-Objective Reward — Cost vs. Satisfaction

A 9-component grader weights the final score. One component — **cost efficiency (7%)** — models the real business trade-off between customer satisfaction and company cost:

```
RESOLUTION_COSTS: refund=1.00, exchange=0.75, partial_refund=0.60,
                  store_credit=0.45, escalate=0.30, deny=0.00
```

When multiple valid resolutions exist, agents that choose the more cost-efficient option score higher.

### 9. Semantic Reply Grading — Word Overlap

The reply quality component uses **Jaccard word-overlap** similarity with partial credit:

- Exact phrase match → **full credit** for that slot
- Word overlap ≥ 0.25 → **full credit** (agent paraphrased correctly)
- Word overlap 0.10–0.25 → **partial credit** (relevant but imprecise)
- Completely irrelevant → **0 credit**

### 10. Belief State Tracking in Inference Agent

The deterministic baseline and LLM agent both maintain a **per-episode belief state** dict:
```python
belief_state = {
    "suspected_resolution": "escalate",   # Updated from observation data
    "fraud_suspected": True,               # Set when customer fraud features detected
    "sections_still_needed": ["policy"],  # Tracks what's missing before deciding
    "confidence": 0.67,                    # Fraction of required sections inspected
}
```

This is updated before every LLM call and injected into the prompt — the agent sees its own running hypothesis about the ticket, enabling smarter multi-step reasoning.

### 11. Nine-Component Weighted Grader

| Component | Weight | Description |
|---|---|---|
| Item resolution accuracy | 22% | Did each item get the right resolution? |
| Ticket resolution accuracy | 13% | Is the overall ticket decision correct? |
| Policy compliance | 12% | Are decisions within the allowed resolution set? |
| Reply quality (semantic) | 10% | Word-overlap similarity to gold requirements |
| Tag quality | 10% | F1 between predicted and gold tags |
| Evidence coverage | 10% | Were required sections inspected before deciding? |
| Priority accuracy | 8% | Was urgency level correctly assessed? |
| Efficiency | 8% | Fewer steps used = higher score |
| **Cost efficiency** | **7%** | **Multi-objective: prefer cheaper valid resolutions** |

**Fraud penalty:** Refunding a fraud-risk order subtracts **0.40** from the raw score before clamping.  
All final scores are clamped to the strict interval `(0.01, 0.99)`.

### 12. Per-Step Shaped Rewards

Dense signal at every step — no sparse reward:

| Action | Reward |
|---|---|
| `inspect_*` required section | +0.05 |
| Re-inspect same section | -0.02 |
| `flag_fraud` on correct fraud task | +0.10 |
| `flag_fraud` false positive | -0.08 |
| `ask_customer` (info still missing) | +0.02 |
| `ask_customer` (info already available) | -0.01 |
| Correct `set_item_resolution` | +0.08 |
| Wrong `set_item_resolution` | -0.06 |
| `set_resolution` before any inspect | -0.10 |
| Correct `add_tag` | +0.03 |
| Noise tag | -0.01 |

### 13. Forced Evidence-Based Reasoning

A gate prevents agents from submitting before inspecting anything. Calling `set_item_resolution`, `set_ticket_resolution`, or `submit` without having called at least one `inspect_*` action returns a **-0.10 penalty** and a warning note.

### 14. Live Reward Breakdown in Every Observation

The `reward_breakdown` field is populated at every step — the agent sees its component-by-component running score before submitting. This enables agents that reason about their own debugging: if `evidence_coverage = 0.0`, the agent knows it missed a required section.

### 15. Dynamic Web UI with Auto-Play

The `/web` interface shows all 7 tasks (including the extreme tier). The **"Auto-play Perfect Solution"** button calls `/api/hint` at each step — a server-side endpoint that runs the deterministic policy on the current live observation and returns the optimal next action.

---

## Technical Specification

| Attribute | Value |
|---|---|
| Framework | FastAPI + OpenEnv SDK |
| Language | Python 3.12 |
| Action types | **12** (inspect × 4, flag_fraud, ask_customer, set_priority, add_tag, set_item_resolution, set_ticket_resolution, draft_reply, submit) |
| Observation fields | **27** typed fields (Pydantic v2) |
| Grader components | 9 weighted components + fraud penalty |
| Task count | 7 procedurally generated tasks |
| Difficulty tiers | 4 (easy / medium / hard / **extreme**) |
| Canonical tags | 14 domain-specific tags |
| Max steps | 10–18 per task |
| Score range | Strictly (0.01, 0.99) |
| Seed reproducibility | Full — `get_task(task_id, seed=N)` is deterministic |
| Test coverage | **36 automated tests** across environment, grader, inference, and catalog |
| Curriculum training | `CurriculumState` class with 10-episode rolling window |
| Episode replay | In-memory replay store, 100-episode buffer, `/api/replay/{id}` |
| Sentiment dynamics | Per-step decay + action-linked boost/penalty |
| Multi-turn dialogue | 3 follow-up messages per task, triggered by `ask_customer` or `request_info` |

---

## Verified Baseline Results — 70 Episodes

`python trainer.py --episodes 70 --mode round_robin` (deterministic policy, no LLM):

| Task | Difficulty | Score | Steps |
|---|---|---|---|
| `easy_refund` | Easy | 0.870 | 9 |
| `medium_exchange` | Medium | 0.909 | 11 |
| `expired_return` | Medium | 0.940 | 9 |
| `wrong_item_sent` | Medium | 0.863–0.933 | 10 |
| `hard_partial_resolution` | Hard | 0.863 | 14 |
| `fraud_risk` | Hard | 0.937 | 11 |
| `extreme_chargeback` | Extreme | 0.920 | 18 |
| **Mean (70 episodes)** | — | **0.9053** | — |

The deterministic policy proves the environment is **fully solvable** — all 7 tasks complete within budget. The curriculum tracker correctly advances to "extreme" difficulty after ~20 episodes, confirming the difficulty ladder is calibrated correctly.

Full episode-by-episode data with grader breakdowns: `outputs/baseline_scores.json`

---

## What Agents Learn

An agent that achieves high scores across all 7 tasks has learned:

1. **Investigate before deciding** — evidence must be gathered before resolution actions
2. **Approve vs. deny** — not every customer claim is valid or within policy
3. **Fraud detection** — recognise suspicious patterns and escalate rather than refund
4. **Contextual resolution** — the correct answer for `wrong_item_sent` changes based on live inventory state
5. **Cost efficiency** — when multiple valid resolutions exist, prefer the cheaper option for the business
6. **Multi-source reconciliation** — `extreme_chargeback` requires reasoning across 4 conflicting data sources simultaneously
7. **Dialogue management** — knowing when to ask the customer a follow-up question vs. inferring from available evidence
8. **Efficiency** — unnecessary inspection steps reduce the final score
9. **Sentiment management** — slow or wrong actions reduce customer satisfaction, affecting the episode outcome

These are transferable, real-world skills directly applicable to a deployed enterprise operations agent.
