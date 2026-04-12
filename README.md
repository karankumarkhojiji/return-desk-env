---
title: ReturnDeskEnv
emoji: 📦
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - customer-operations
  - returns
  - fraud-detection
---

# ReturnDeskEnv

**ReturnDeskEnv** is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible reinforcement learning environment that simulates a real e-commerce customer support operations desk handling returns, refunds, fraud detection, and corporate chargeback disputes.

An agent reads a customer ticket, gathers evidence by inspecting order details, customer history, company policy, and inventory, then makes a series of structured decisions before submitting for grading. Every `reset()` produces a **procedurally generated** unique episode — product names, prices, customer IDs, exchange rates, and dates are randomised using a seeded RNG.

The environment rewards real-world decision-making: **denying bad requests**, **escalating fraud**, **choosing cost-efficient resolutions**, and **asking the customer the right question** all score higher than blindly approving everything.

---

## Why This Environment?

Customer support operations is a genuine, high-stakes, multi-step decision workflow that tests seven real agent capabilities:

1. **Investigate before deciding.** The agent cannot correctly resolve a ticket without inspecting evidence first — resolution actions before any inspect are penalised (`-0.10`).
2. **Approve vs. deny.** Not every customer claim is valid. Expired return windows and policy violations must be denied, not refunded.
3. **Fraud detection.** A `fraud_risk` task requires identifying suspicious signals (new account, high refund velocity, address mismatch) and escalating rather than refunding. Refunding costs **-0.40 score penalty**.
4. **Contextual resolution.** The correct answer for `wrong_item_sent` changes based on live inventory state — exchange is correct if stock exists, otherwise refund.
5. **Cost efficiency.** When multiple valid resolutions exist, the cheaper option (`store_credit` vs `refund`) scores higher — modelling the real business trade-off.
6. **Multi-source reconciliation.** The `extreme_chargeback` task requires reasoning across 4 conflicting data sources simultaneously with 5 different per-item resolutions.
7. **Multi-turn dialogue.** The `ask_customer` action triggers a real customer follow-up response, giving the agent new information mid-episode to inform its decision.

---

## Project Structure

```
return_desk_env/
├── README.md                  # This file
├── SUBMISSION_SUMMARY.md      # Engineering summary for human reviewers
├── inference.py               # Baseline agent (LLM + deterministic policy + belief state)
├── trainer.py                 # Multi-episode training harness + learning curve plot
├── openenv.yaml               # OpenEnv manifest (v0.3.0)
├── pyproject.toml             # Project metadata and dependencies
├── Dockerfile                 # Container image for deployment
├── deploy_to_hf.py            # Helper script to deploy to HuggingFace Spaces
├── .env.example               # Environment variable template (safe to commit)
├── models.py                  # Typed Action, Observation, and State models (Pydantic v2)
├── graders.py                 # 9-component deterministic grader with fraud penalty
├── rewards.py                 # Per-step shaped reward functions
├── client.py                  # Python client for connecting to the server
├── outputs/
│   └── baseline_scores.json   # 70-episode deterministic baseline run (mean score: 0.9053)
├── tasks/
│   ├── __init__.py
│   └── catalog.py             # 7 procedurally-generated tasks across 4 difficulty tiers
├── server/
│   ├── __init__.py
│   ├── app.py                 # FastAPI server + interactive Web UI + all API endpoints
│   └── environment.py         # Core environment state machine + CurriculumState
├── tests/
│   ├── test_environment_logic.py   # 12 environment tests (fraud, extreme, curriculum, sentiment)
│   ├── test_graders.py             # 18 grader tests (semantic, cost, fraud penalty)
│   └── test_inference_smoke.py
└── scripts/
    └── validate-submission.sh
```

---

## Task Catalog — 7 Tasks, 4 Difficulty Tiers

Each task is **procedurally generated** on every `reset()`. The gold-standard grading logic is deterministic — the correct resolution type, tags, and priority are always the same regardless of which product name or price was generated.

| Task ID | Difficulty | Correct Action | What Makes It Hard |
|---|---|---|---|
| `easy_refund` | 🟢 Easy | `refund` | Damaged item within return window — baseline task |
| `medium_exchange` | 🟡 Medium | `store_credit` | Out of stock + coupon = no full exchange; must read inventory |
| `expired_return` | 🟡 Medium | **`deny`** | Customer is WRONG — outside 30-day window; refunding is a violation |
| `wrong_item_sent` | 🟡 Medium | `exchange` or `refund` | Correct answer depends on live inventory state |
| `hard_partial_resolution` | 🔴 Hard | `partial_refund` | 3 items, VIP exception, contradictory dates across 2 data sources |
| `fraud_risk` | 🔴 Hard | **`escalate`** | Fraud signals present; refunding costs **-0.40 penalty** |
| `extreme_chargeback` | ⚫ Extreme | `partial_refund` (5 items) | Cross-currency, insurance limits, 4 data sources, 1 fraud claim |

### Task Details

#### 🟢 `easy_refund`

A customer received a single damaged product within the return window.

**Agent must:** inspect order + policy → set priority `high` → add tags `damaged`, `refund_request` → refund item-1 → ticket resolution `refund` → draft reply mentioning the refund and timeline → submit.

---

#### 🟡 `medium_exchange`

Customer wants to exchange; exact replacement is out of stock; order used a coupon code.

**Agent must:** inspect order + inventory + policy → set priority `medium` → add tags `exchange_request`, `inventory_issue`, `coupon_order` → resolution `store_credit` (exact exchange unavailable + coupon order = store credit, not refund) → draft reply acknowledging the amount actually paid → submit.

---

#### 🟡 `expired_return`

Customer requests a refund for a product delivered 35–60 days ago (outside 30-day window). **The customer is wrong.** Issuing a refund is a policy violation.

**Agent must:** inspect order (confirm delivery date) + policy → calculate days since delivery → set priority `medium` → add tags `return_window_exceeded`, `policy_violation` → **deny** item and ticket → draft polite denial explaining the policy → submit.

---

#### 🟡 `wrong_item_sent`

The customer received the wrong product. The correct resolution depends on live inventory.

**Agent must:** inspect order + inventory + policy → if correct item available → `exchange`; if out of stock → `refund`. The agent cannot know the answer without inspecting inventory first.

---

#### 🔴 `hard_partial_resolution`

Three items in one ticket; each requires a different resolution:
- **Item 1:** VIP customer + reported defect "within 5 days" — but delivery records show 15 days. Prior support note says 5 days; agent must trust the original contact date → `refund` (VIP exception)
- **Item 2:** Arrived visibly damaged → `refund`
- **Item 3:** Personalized/monogrammed item — non-returnable by policy → `deny`

**What makes it hard:** Two data sources have contradictory dates. The agent must reconcile them, apply VIP exception logic, and set three different item resolutions before a single ticket-level `partial_refund` resolution.

---

#### 🔴 `fraud_risk`

A customer claims non-delivery on a high-value electronic item. But:
- Carrier records show delivery confirmed with GPS + signature
- Customer account is 3 days old
- 5 prior refund attempts in the account history
- Delivery address does not match the billing address

**The correct action is to escalate, not refund.** If the agent refunds, the grader subtracts **-0.40** from the raw score.

**Agent must:** inspect order + customer + policy → call `flag_fraud` (+0.10 bonus) → add tags `fraud_flag`, `escalation_required` → set priority `urgent` → `escalate` all items → ticket resolution `escalate` → draft reply explaining escalation → submit.

---

#### ⚫ `extreme_chargeback`

A corporate client filed a formal bank chargeback on a 5-item bulk order. Currency is randomised (EUR/GBP/CAD) with a dynamic exchange rate to USD.

**Item breakdown:**
- Items 1–2: Confirmed damaged in transit, covered by insurance → `partial_refund`
- Items 3–4: Delivered fine, customer changed mind — corporate bulk orders are non-refundable for change-of-mind → `deny`
- Item 5: Customer claims non-delivery, but carrier has GPS confirmation + signature on file → `escalate` (fraud flag required)

**Agent must:** inspect all 4 sections → `flag_fraud` → set priority `urgent` → add 5 tags → set 5 different item resolutions → ticket resolution `partial_refund` → draft a formal response citing policy per item and the 48-hour chargeback deadline → submit within 18 steps.

---

## Action Space

The agent emits one typed JSON action per step.

| Action | Required Fields | What it does |
|---|---|---|
| `inspect_order` | — | Reveals order details: items, prices, dates, currency, carrier evidence |
| `inspect_customer` | — | Reveals customer history, VIP status, fraud signals, prior tickets |
| `inspect_policy` | — | Reveals the return/refund/exchange/insurance policy rules |
| `inspect_inventory` | — | Reveals current stock, available replacements |
| `flag_fraud` | — | Raises a fraud flag (+0.10 if correct; -0.08 if false positive) |
| `ask_customer` | `question` (optional) | Sends a follow-up to the customer; receives a simulated response (+0.02 if new info needed; -0.01 if unnecessary) |
| `set_priority` | `priority` | Sets the ticket priority level |
| `add_tag` | `tag` | Adds a canonical tag to the ticket |
| `set_item_resolution` | `item_id`, `resolution` | Sets a resolution for one specific item |
| `set_ticket_resolution` | `resolution` | Sets the overall ticket resolution (using `request_info` automatically triggers a customer follow-up) |
| `draft_reply` | `message` | Writes the customer-facing reply |
| `submit` | — | Finalises the episode and triggers grading |

**Priority values:** `low`, `medium`, `high`, `urgent`

**Resolution values:** `refund`, `exchange`, `store_credit`, `deny`, `escalate`, `request_info`, `partial_refund`

> **Forced investigation gate:** Calling `set_item_resolution`, `set_ticket_resolution`, or `submit` without having first called at least one `inspect_*` action returns `reward = -0.10` and a warning. Agents cannot blindly guess the answer.

> **Multi-turn dialogue:** Each task has a pool of 3 customer follow-up messages. Calling `ask_customer` or setting resolution to `request_info` appends a customer reply to `customer_messages` in the observation — giving the agent new information to work with.

---

## Observation Space

After every `reset()` and `step()`, the agent receives a `ReturnDeskObservation` with 23 typed fields:

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Which task is active |
| `difficulty` | `str` | `easy`, `medium`, `hard`, or `extreme` |
| `objective` | `str` | Natural language description of what the agent must do |
| `customer_ticket` | `dict` | The customer's message, channel, and item list |
| `available_actions` | `list[str]` | JSON examples of all 12 valid actions |
| `allowed_priorities` | `list[str]` | Valid priority values |
| `allowed_resolutions` | `list[str]` | Valid resolution values |
| `available_tags` | `list[str]` | 14 canonical tags the agent can apply |
| `visible_sections` | `list[str]` | Which sections have been inspected so far |
| `order_summary` | `dict \| null` | Order details — only populated after `inspect_order` |
| `customer_summary` | `dict \| null` | Customer record — only after `inspect_customer` |
| `policy_summary` | `dict \| null` | Policy rules — only after `inspect_policy` |
| `inventory_summary` | `dict \| null` | Stock levels — only after `inspect_inventory` |
| `current_priority` | `str \| null` | Priority currently set on this ticket |
| `current_tags` | `list[str]` | Tags currently applied |
| `item_resolutions` | `dict[str, str]` | Per-item resolutions set so far |
| `ticket_resolution` | `str \| null` | Overall ticket resolution currently set |
| `drafted_reply` | `str` | Current draft of the customer reply |
| `customer_messages` | `list[dict]` | **Multi-turn dialogue:** customer follow-up messages (role + text) |
| `customer_sentiment` | `float` | Dynamic: −1.0 (furious) to +1.0 (satisfied). Decays each step, rises on correct actions |
| `fraud_flagged` | `bool` | Whether the agent has raised a fraud flag |
| `reward_breakdown` | `dict[str, float]` | **Live component scores at every step** — agent can course-correct before submitting |
| `history` | `list[str]` | Last 8 actions taken in this episode |
| `steps_remaining` | `int` | How many steps remain before auto-submit |
| `latest_note` | `str` | Plain-language feedback on the last action |
| `final_score` | `float \| null` | Final grader score — only after `submit` |
| `grader_breakdown` | `dict[str, float]` | Per-component scores — only after `submit` |

> **Important:** `order_summary`, `customer_summary`, `policy_summary`, and `inventory_summary` are all `null` until the agent calls the corresponding inspect action.

> **Sentiment dynamics:** `customer_sentiment` starts at a task-specific baseline (e.g. -0.6 for extreme_chargeback, +0.4 for medium_exchange) and changes dynamically: decays -0.04/step after step 5, rises +0.06 on correct item resolution, drops -0.04 on wrong resolution, drops -0.08 when fraud is flagged.

---

## Reward Design

Dense per-step rewards — agents can learn from trajectories, not just final outcomes.

| Action | Reward |
|---|---|
| Inspect a **required** section (first time) | `+0.05` |
| Inspect a non-required section (first time) | `+0.01` |
| Inspect any section **again** (duplicate) | `−0.02` |
| `flag_fraud` on a genuine fraud task | `+0.10` |
| `flag_fraud` on a non-fraud task (false positive) | `−0.08` |
| `ask_customer` when info is still missing | `+0.02` |
| `ask_customer` when info already available | `−0.01` |
| Set **correct** item resolution | `+0.08` |
| Set **wrong** item resolution | `−0.06` |
| Set **correct** ticket resolution | `+0.07` |
| Set **wrong** ticket resolution | `−0.05` |
| Add a **gold** tag (first time) | `+0.03` |
| Add a **wrong** tag | `−0.01` |
| Draft a reply covering more required slots | `+0.10 × improvement` |
| **Resolve before inspecting anything** | `−0.10` |
| `submit` with all evidence gathered | `final_score` |
| `submit` with incomplete evidence | `final_score − 0.10` |
| **Refund a fraud-risk ticket (final grader)** | **−0.40 on raw score** |

---

## Grading Rubric

On `submit`, the grader computes a final score from 9 weighted components. All scores are clamped to `(0.01, 0.99)`.

| Component | Weight | How it is measured |
|---|---|---|
| `item_resolution_accuracy` | 22% | Exact match per item, averaged across all items |
| `ticket_resolution_accuracy` | 13% | Exact match against gold ticket resolution |
| `policy_compliance` | 12% | Whether decisions fall within policy-allowed resolutions |
| `reply_quality` | 10% | **Jaccard word-overlap** similarity to gold requirements |
| `tag_quality` | 10% | F1 score between predicted tags and gold tags |
| `evidence_coverage` | 10% | Fraction of required sections that were inspected |
| `priority_accuracy` | 8% | Exact match against gold priority |
| `efficiency` | 8% | Fewer steps → higher score (full at ≤60% of budget) |
| `cost_efficiency` | 7% | When multiple valid resolutions exist, cheapest one scores highest |

**Fraud penalty:** If the agent refunds a fraud-risk order, `−0.40` is subtracted from the weighted raw score before clamping.

### Cost Efficiency Resolution Costs

| Resolution | Cost | Meaning |
|---|---|---|
| `deny` | 0.00 | No cost — customer's claim rejected |
| `request_info` | 0.10 | Near-free — defer decision |
| `escalate` | 0.30 | Specialist time required |
| `store_credit` | 0.45 | Deferred, likely < full redemption |
| `partial_refund` | 0.60 | Partial financial cost |
| `exchange` | 0.75 | Shipping new item + return logistics |
| `refund` | 1.00 | Full financial cost to company |

When `["refund", "store_credit"]` are both allowed, an agent choosing `store_credit` scores higher on `cost_efficiency` — exactly how a real operations team evaluates agent decisions.

### Reply Quality — Semantic Grading

Reply grading uses **Jaccard word-overlap similarity**, not binary keyword matching:
- Exact phrase present in reply → full credit for that slot
- Word-level overlap ≥ 25% → full credit (agent paraphrased correctly)
- Partial overlap → graduated partial credit
- Completely irrelevant reply → 0 credit

This rewards agents that write semantically correct but differently-worded replies.

---

## Baseline Performance

### Deterministic Policy — 70 Episodes (`RETURN_DESK_USE_LLM=false`)

Measured by running `python trainer.py --episodes 70 --mode round_robin`. Full results in `outputs/baseline_scores.json`.

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

The mean score of **0.9053** across all 7 tasks proves the environment is fully solvable. The curriculum tracker correctly advances to "extreme" difficulty after ~20 episodes.

### Belief State Tracking

The inference agent maintains a **per-episode belief state** updated at every step:
```python
belief_state = {
    "suspected_resolution": "escalate",   # Updated from observation data
    "fraud_suspected": True,               # Set when customer fraud features detected
    "sections_still_needed": ["policy"],  # What the agent still needs to inspect
    "confidence": 0.67,                    # Fraction of required sections seen
}
```
This is injected into the LLM prompt at every step, enabling smarter multi-step reasoning for LLM-based agents.

---

## Curriculum Training

The environment supports automatic difficulty progression through `CurriculumState`:

```python
from server.environment import CurriculumState, ReturnDeskEnvironment

cs = CurriculumState(window=10)
env = ReturnDeskEnvironment()

for episode in range(100):
    obs = env.reset(mode="curriculum", curriculum_state=cs)
    # ... run episode ...
    cs.record(final_score)
    print(cs.summary())  # rolling_mean, current_difficulty, episode_count
```

**Difficulty ladder:**
- Rolling mean < 0.55 → easy tasks
- Rolling mean 0.55–0.70 → medium tasks
- Rolling mean 0.70–0.82 → hard tasks
- Rolling mean ≥ 0.82 → extreme tasks

The `trainer.py` script wraps this into a full training harness with logging and matplotlib plots.

---

## Training Harness

`trainer.py` runs multi-episode training loops using the deterministic policy:

```bash
# Run 70 episodes across all 7 tasks (round-robin, no API key needed)
python trainer.py --episodes 70 --mode round_robin

# Curriculum mode (auto-advances difficulty as agent improves)
python trainer.py --episodes 100 --mode curriculum

# Random task sampling
python trainer.py --episodes 50 --mode random --seed 123
```

Outputs:
- `outputs/baseline_scores.json` — full episode results with grader breakdowns
- `outputs/training_curve.png` — learning curve coloured by task difficulty (requires `pip install matplotlib`)

---

## Setup and Usage

### Option 1 — Docker (Recommended)

```bash
docker build -t return-desk-env .
docker run -p 8000:8000 return-desk-env
```

Then open http://localhost:8000/web for the interactive Web UI.

---

### Option 2 — Local with uv

```bash
pip install uv
uv run server
```

---

### Option 3 — Local with pip

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## Running the Inference Script

```bash
# 1. Copy and fill in your environment variables
cp .env.example .env

# 2. Start the server (terminal 1)
uv run server

# 3. Run inference across all 7 tasks (terminal 2)
python inference.py

# Run with deterministic policy only (no API key needed)
RETURN_DESK_USE_LLM=false python inference.py   # Linux/Mac
set RETURN_DESK_USE_LLM=false && python inference.py  # Windows
```

**Environment variables (`.env`):**
```
HF_TOKEN=hf_your_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
RETURN_DESK_BASE_URL=http://localhost:8000
RETURN_DESK_USE_LLM=true
```

**Alternative free inference providers (when HF free tier limit is hit):**
```
# Groq — fast, 14,400 req/day free at console.groq.com
API_BASE_URL=https://api.groq.com/openai/v1
MODEL_NAME=llama-3.3-70b-versatile

# Google AI Studio — free at aistudio.google.com
API_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
MODEL_NAME=gemini-2.0-flash

# Ollama — fully local, no limits
API_BASE_URL=http://localhost:11434/v1
MODEL_NAME=llama3.2:3b
```

### Expected output format

```
[START] task=fraud_risk env=return_desk_env model=Qwen/Qwen2.5-72B-Instruct
[STEP]  step=1 action=inspect_order reward=0.05 done=false error=null
[STEP]  step=2 action=inspect_customer reward=0.05 done=false error=null
[STEP]  step=3 action=flag_fraud reward=0.10 done=false error=null
...
[END]   success=true steps=11 score=0.937 rewards=0.05,0.05,0.10,...
```

---

## Deploy to HuggingFace Spaces

```bash
HF_TOKEN=hf_... python deploy_to_hf.py --space-id your-username/return-desk-env
```

The script runs the test suite before uploading, prepends HF Space metadata to README, and uploads all project files. Use `--skip-tests` to force upload.

---

## Available Endpoints

| Endpoint | Description |
|---|---|
| `GET /health` | Health check → `{"status": "healthy"}` |
| `POST /reset` | Start a new episode |
| `POST /step` | Take one action |
| `GET /state` | Get the current non-leaky state snapshot |
| `POST /api/reset` | Stateful reset (for curl/Swagger chaining) |
| `POST /api/step` | Stateful step (for curl/Swagger chaining) |
| `GET /api/hint` | Returns optimal next action from deterministic policy |
| `GET /api/replay/{episode_id}` | **Full step-by-step trajectory replay** for a completed episode |
| `GET /api/replay` | List all stored episode IDs (max 100, most recent last) |
| `GET /api/curriculum` | Current curriculum state: rolling mean, difficulty, episode count |
| `GET /web` | Interactive Web UI (all 7 tasks, auto-play) |
| `GET /docs` | Auto-generated API documentation (Swagger) |

---

## Canonical Tags (14)

`damaged` · `refund_request` · `exchange_request` · `inventory_issue` · `coupon_order` · `vip_exception` · `partial_resolution` · `non_returnable` · `fraud_flag` · `policy_violation` · `return_window_exceeded` · `wrong_item` · `duplicate_charge` · `escalation_required`

---

## OpenEnv Compliance

- `reset()` → returns `ReturnDeskObservation`
- `step(action)` → returns observation with `reward`, `done`
- `state()` → returns `ReturnDeskState` (no gold leakage)
- All models are typed Pydantic v2 classes with `model_validator`
- `openenv.yaml` manifest present at repo root (v0.3.0)
- Score strictly in `(0.01, 0.99)` — validator-safe
- **36 automated tests** — `python -m pytest tests/ -v`

---

## Learn More

- [OpenEnv Documentation](https://github.com/meta-pytorch/OpenEnv)
- [OpenEnv Environment Design Guide](https://github.com/meta-pytorch/OpenEnv/blob/main/README.md)
- [HuggingFace Router](https://huggingface.co/docs/inference-providers/en/index) — free API access to open models
- [Groq](https://console.groq.com) — fast free inference alternative (14,400 req/day)
- [Google AI Studio](https://aistudio.google.com) — Gemini free API
