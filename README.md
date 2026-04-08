---
title: ReturnDeskEnv
emoji: 📦
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
app_port: 8000
base_path: /web
tags:
  - openenv
  - customer-operations
  - returns
  - refunds
---

# ReturnDeskEnv

**ReturnDeskEnv** is an [OpenEnv](https://github.com/meta-pytorch/OpenEnv)-compatible reinforcement learning environment that simulates the work of a real e-commerce customer support specialist handling returns, refunds, and exchanges.

The agent reads a customer ticket, gathers evidence by inspecting order details, customer history, company policy, and inventory, then makes a series of structured decisions — priority, tags, resolution per item, overall ticket resolution, and a drafted customer reply — before submitting for grading.

Every episode is graded deterministically: there is a single correct answer for each task and the grader measures how closely the agent's decisions match it.

---

## Why This Environment?

Customer support operations is a genuine, high-stakes, multi-step decision workflow:

- **Information gathering is required before deciding.** An agent cannot correctly resolve a ticket without inspecting the relevant evidence first.
- **Decisions are structured and typed.** The correct resolution type, priority level, and tags are objectively verifiable.
- **Policy constraints matter.** A refund outside the return window, or an exchange on a non-returnable item, is wrong regardless of how polite the reply sounds.
- **The workflow has natural difficulty progression.** A single damaged item refund is easy. A multi-item ticket with VIP exceptions, coupon constraints, and a non-returnable personalized item is hard.

---

## Project Structure

```
return_desk_env/
├── README.md               # This file
├── inference.py            # Baseline agent script (LLM + deterministic)
├── openenv.yaml            # OpenEnv manifest
├── pyproject.toml          # Project metadata and dependencies
├── Dockerfile              # Container image for deployment
├── .env.example            # Environment variable template (safe to commit)
├── models.py               # Typed Action, Observation, and State models
├── graders.py              # Deterministic grading logic
├── rewards.py              # Per-step shaped reward functions
├── client.py               # Python client for connecting to the server
├── __init__.py             # Package exports
├── tasks/
│   ├── __init__.py
│   └── catalog.py          # Task definitions with gold standards
├── server/
│   ├── __init__.py
│   ├── app.py              # FastAPI server (HTTP + WebSocket + Web UI)
│   └── environment.py      # Core environment state machine
├── tests/
│   ├── test_environment_logic.py
│   ├── test_graders.py
│   └── test_inference_smoke.py
└── scripts/
    └── validate-submission.sh
```

---

## Tasks

There are three tasks of increasing difficulty. Each task has a fixed gold standard; scoring is fully deterministic.

### 1. `easy_refund` — Easy

A customer received a single damaged product within the return window and asks for a refund.

**What the agent must do:**
- Inspect the order to confirm the product and damage claim
- Inspect the policy to verify eligibility
- Set priority to `high`
- Add tags: `damaged`, `refund`
- Set item resolution to `refund`
- Set ticket resolution to `refund`
- Draft a policy-compliant reply acknowledging the damage and confirming the refund
- Submit

**Expected score:** 1.000 with a well-tuned agent. A baseline LLM achieves ~0.842.

---

### 2. `medium_exchange` — Medium

A customer wants to exchange a product. The exact replacement is out of stock, and the original order used a coupon code.

**What the agent must do:**
- Inspect the order (reveals the coupon code and item details)
- Inspect the inventory (reveals the replacement is unavailable)
- Inspect the policy (defines exchange eligibility rules)
- Set the correct priority
- Set item resolution to `exchange`
- Set ticket resolution to `exchange`
- Draft a reply that acknowledges the coupon, the inventory situation, and offers the exchange
- Submit

**Expected score:** 1.000 deterministic. A baseline LLM achieves ~0.762.

---

### 3. `hard_partial_resolution` — Hard

A multi-item ticket with mixed eligibility. Three items; each requires a different resolution:

- **Item 1:** VIP customer with a defective product — qualifies for a policy exception refund
- **Item 2:** Damaged on arrival — standard refund
- **Item 3:** Personalized/custom item — non-returnable, must be denied

**What the agent must do:**
- Inspect order, customer (to discover VIP status), policy, and inventory
- Set priority to `urgent`
- Add tags: `vip`, `damaged`, `partial`
- Set a different resolution for each of the three items
- Set an overall ticket resolution that covers all three cases
- Draft a reply that addresses all three cases individually
- Submit

**Expected score:** 1.000 deterministic. A baseline LLM achieves ~0.902.

---

## Action Space

The agent emits one typed JSON action per step. All actions follow the `ReturnDeskAction` schema.

| Action | Required Fields | What it does |
|---|---|---|
| `inspect_order` | — | Reveals order details: items, prices, dates, coupon |
| `inspect_customer` | — | Reveals customer history, VIP status, previous tickets |
| `inspect_policy` | — | Reveals the return/refund/exchange policy rules |
| `inspect_inventory` | — | Reveals current stock and available replacements |
| `set_priority` | `priority` | Sets the ticket priority level |
| `add_tag` | `tag` | Adds a canonical tag to the ticket |
| `set_item_resolution` | `item_id`, `resolution` | Sets a resolution for one specific item |
| `set_ticket_resolution` | `resolution` | Sets the overall ticket resolution |
| `draft_reply` | `message` | Writes the customer-facing reply |
| `submit` | — | Finalises the episode and triggers grading |

**Priority values:** `low`, `medium`, `high`, `urgent`

**Resolution values:** `refund`, `exchange`, `store_credit`, `deny`, `escalate`, `request_info`, `partial_refund`

**Example actions (JSON):**

```json
{"action_type": "inspect_order"}
{"action_type": "set_priority", "priority": "high"}
{"action_type": "add_tag", "tag": "damaged"}
{"action_type": "set_item_resolution", "item_id": "item-1", "resolution": "refund"}
{"action_type": "set_ticket_resolution", "resolution": "refund"}
{"action_type": "draft_reply", "message": "We are sorry to hear about the damage..."}
{"action_type": "submit"}
```

---

## Observation Space

After every `reset()` and `step()`, the agent receives a `ReturnDeskObservation` with the following fields:

| Field | Type | Description |
|---|---|---|
| `task_id` | `str` | Which task is active |
| `difficulty` | `str` | `easy`, `medium`, or `hard` |
| `objective` | `str` | Natural language description of what the agent must do |
| `customer_ticket` | `dict` | The customer's original message and contact details |
| `available_actions` | `list[str]` | Actions valid at the current step |
| `allowed_priorities` | `list[str]` | Valid priority values |
| `allowed_resolutions` | `list[str]` | Valid resolution values |
| `available_tags` | `list[str]` | Canonical tags the agent can apply |
| `visible_sections` | `list[str]` | Which sections have already been inspected |
| `order_summary` | `dict \| null` | Order details — only populated after `inspect_order` |
| `customer_summary` | `dict \| null` | Customer history — only after `inspect_customer` |
| `policy_summary` | `dict \| null` | Policy rules — only after `inspect_policy` |
| `inventory_summary` | `dict \| null` | Stock levels — only after `inspect_inventory` |
| `current_priority` | `str \| null` | Priority currently set on this ticket |
| `current_tags` | `list[str]` | Tags currently applied |
| `item_resolutions` | `dict[str, str]` | Per-item resolutions set so far |
| `ticket_resolution` | `str \| null` | Overall ticket resolution currently set |
| `drafted_reply` | `str` | Current draft of the customer reply |
| `history` | `list[str]` | Log of all actions taken in this episode |
| `steps_remaining` | `int` | How many steps remain before the episode times out |
| `latest_note` | `str` | Plain-language feedback on the last action |
| `final_score` | `float \| null` | Final grader score — only populated after `submit` |
| `grader_breakdown` | `dict[str, float]` | Per-component scores — only after `submit` |

> **Important:** `order_summary`, `customer_summary`, `policy_summary`, and `inventory_summary` are all `null` until the agent takes the corresponding inspect action. The agent must gather its own evidence — it is not given everything upfront.

---

## State Space

`state()` returns a `ReturnDeskState` — a minimal, non-leaky snapshot of the environment's internal state. It does **not** expose gold-standard answers.

| Field | Type | Description |
|---|---|---|
| `task_id` | `str \| null` | Active task identifier |
| `difficulty` | `str \| null` | Task difficulty level |
| `visible_sections` | `list[str]` | Which sections have been inspected |
| `submitted` | `bool` | Whether the agent has already submitted |
| `current_priority` | `str \| null` | Current ticket priority |
| `current_tags` | `list[str]` | Current applied tags |
| `item_resolutions` | `dict[str, str]` | Current per-item resolutions |
| `ticket_resolution` | `str \| null` | Current overall resolution |
| `steps_remaining` | `int` | Steps left before timeout |

---

## Reward Design

The reward function provides feedback at every step, not just at submission. This allows an RL agent to learn which actions are useful and which are not.

| Action | Reward Signal |
|---|---|
| Inspect a **required** section (first time) | `+0.05` |
| Inspect a non-required section (first time) | `+0.01` |
| Inspect any section **again** (duplicate) | `−0.02` |
| Set **correct** priority (was wrong before) | `+0.10` |
| Set **wrong** priority | `−0.05` |
| Set correct priority **again** (no change) | `−0.01` |
| Add a **gold** tag (first time) | `+0.03` |
| Add a **wrong** tag | `−0.01` |
| Add an existing tag (duplicate) | `−0.02` |
| Set **correct** item resolution | `+0.08` |
| Set **wrong** item resolution | `−0.06` |
| Set **correct** ticket resolution | `+0.07` |
| Set **wrong** ticket resolution | `−0.05` |
| Draft a reply with more correct content | `+0.10 × improvement` |
| Draft a reply with same or less content | `−0.01` |
| `submit` with all evidence gathered | `final_score` |
| `submit` with missing evidence | `final_score − 0.10` |

The episode terminates when:
- The agent calls `submit` (done = `True`)
- The step limit is reached (done = `True`, score is whatever was earned)

---

## Grading Rubric

On `submit`, the grader computes a final score in `[0.0, 1.0]` as a weighted average of seven components:

| Component | Weight | How it is measured |
|---|---|---|
| `item_resolution_accuracy` | 30% | Exact match per item, averaged across all items |
| `reply_quality` | 15% | Fraction of required reply slots that are covered |
| `ticket_resolution_accuracy` | 15% | Exact match against the gold ticket resolution |
| `priority_accuracy` | 10% | Exact match against the gold priority |
| `tag_quality` | 10% | F1 score between predicted tags and gold tags |
| `evidence_coverage` | 10% | Fraction of required sections that were inspected |
| `policy_compliance` | 10% | Whether decisions are within policy-allowed resolutions |

The grader is fully deterministic. Running the same submission twice always produces the same score.

---

## Baseline Performance

### Deterministic Policy (`RETURN_DESK_USE_LLM=false`)

The built-in rule-based policy follows the optimal action sequence for each task:

| Task | Score | Steps |
|---|---|---|
| `easy_refund` | 1.000 | 9 |
| `medium_exchange` | 1.000 | 11 |
| `hard_partial_resolution` | 1.000 | 14 |
| **Mean** | **1.000** | — |

### LLM Policy — `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router

| Task | Score | Steps |
|---|---|---|
| `easy_refund` | 0.842 | 7 |
| `medium_exchange` | 0.762 | 7 |
| `hard_partial_resolution` | 0.902 | 11 |
| **Mean** | **0.836** | — |

The gap between 0.836 (LLM) and 1.000 (deterministic) shows the environment is genuinely challenging: a large frontier model still makes meaningful mistakes on sequencing, tag selection, and reply content.

---

## Setup and Usage

### Option 1 — Docker (Recommended)

```bash
# Build the image
docker build -t return-desk-env .

# Run the server
docker run -p 8000:8000 return-desk-env
```

Then open http://localhost:8000/web for the interactive Web UI.

---

### Option 2 — Local with uv

```bash
pip install uv
uv run server
```

This installs all dependencies and starts the server at http://localhost:8000.

---

### Option 3 — Local with pip

```bash
pip install -e .
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

---

## Running the Inference Script

The inference script connects to the running server and evaluates the agent across all three tasks.

### Step 1 — Set up your environment variables

Copy the template and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:
```
HF_TOKEN=hf_your_token_here
API_BASE_URL=https://router.huggingface.co/v1
MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
RETURN_DESK_BASE_URL=http://localhost:8000
RETURN_DESK_USE_LLM=true
```

Your HF token is available at https://huggingface.co/settings/tokens. The `.env` file is in `.gitignore` and will never be committed.

### Step 2 — Start the server (in one terminal)

```bash
uv run server
```

### Step 3 — Run inference (in another terminal)

```bash
python inference.py
```

**To run with the built-in deterministic policy (no API key needed):**

```bash
# Windows
set RETURN_DESK_USE_LLM=false
python inference.py

# Linux / Mac
RETURN_DESK_USE_LLM=false python inference.py
```

### Expected output format

```
[START] task=easy_refund env=return_desk_env model=Qwen/Qwen2.5-72B-Instruct
[STEP] step=1 action=inspect_order reward=0.05 done=false error=null
[STEP] step=2 action=inspect_policy reward=0.05 done=false error=null
...
[END] success=true steps=7 score=0.84 rewards=0.05,0.05,...
```

---

## Available Endpoints

Once the server is running:

| Endpoint | Description |
|---|---|
| `GET /health` | Health check — returns `{"status": "healthy"}` |
| `POST /reset` | Start a new episode |
| `POST /step` | Take one action |
| `GET /state` | Get the current state snapshot |
| `GET /web` | Interactive Web UI |
| `GET /docs` | Auto-generated API documentation (Swagger) |

---

## OpenEnv Compliance

The environment implements the full OpenEnv interface:

- `reset()` → returns `ReturnDeskObservation`
- `step(action)` → returns `(observation, reward, done, info)`
- `state()` → returns `ReturnDeskState`
- All models are typed Pydantic classes
- `openenv.yaml` manifest is present at the repo root

---

## Learn More

- [OpenEnv Documentation](https://github.com/meta-pytorch/OpenEnv)
- [OpenEnv Environment Design Guide](https://github.com/meta-pytorch/OpenEnv/blob/main/README.md)
- [HuggingFace Router](https://huggingface.co/docs/inference-providers/en/index) — free API access to Qwen, Llama, and other open models
