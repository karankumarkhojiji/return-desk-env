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

ReturnDeskEnv is a deterministic OpenEnv benchmark for **e-commerce returns, refunds, exchanges, and policy exceptions**. It models the work a human support operations specialist actually performs: inspect order facts, inspect customer history, inspect policy, inspect inventory, make item-level and ticket-level decisions, draft a customer reply, and submit the case for grading.

This environment is optimized for hackathon judging:

- real-world workflow, not a toy problem
- typed actions, observations, and state
- 3 benchmark tasks (easy / medium / hard)
- deterministic grader with scores in `[0.0, 1.0]`
- shaped reward with partial progress
- root-level `inference.py` that uses the OpenAI client
- Dockerized HF Space deployment

## Tasks

### 1. `easy_refund`
Standard damaged-item refund. The customer received a broken product inside the return window and explicitly asks for a refund.

### 2. `medium_exchange`
Exchange request with inventory and coupon constraints. The exact replacement is unavailable, and the order used a coupon.

### 3. `hard_partial_resolution`
Multi-item mixed-eligibility ticket. One item needs a VIP defect exception, one item is a damaged-arrival refund, and one personalized item must be denied.

## Action space

The agent must emit a typed JSON action with one of these action types:

- `inspect_order`
- `inspect_customer`
- `inspect_policy`
- `inspect_inventory`
- `set_priority`
- `add_tag`
- `set_item_resolution`
- `set_ticket_resolution`
- `draft_reply`
- `submit`

Examples are exposed directly in the observation via `available_actions`.

## Observation space

Each observation includes:

- task id and difficulty
- objective
- visible customer ticket
- revealed sections (`order`, `customer`, `policy`, `inventory`) if already inspected
- current priority, tags, item resolutions, ticket resolution
- drafted reply
- step history
- steps remaining
- latest note from the environment
- final score and grader breakdown once the episode is done

## State space

`state()` returns a non-leaky state snapshot:

- episode id
- step count
- task id / difficulty
- visible sections
- current decisions
- steps remaining

It does **not** expose the gold labels or grader answer key.

## Reward design

The environment provides shaped reward across the trajectory:

- positive reward for first-time relevant inspections
- positive reward for correct priority / tags / resolutions
- small penalties for repeated or noisy actions
- delta reward for improving the drafted reply
- final reward on submit based on the deterministic grader score

This gives useful learning signal beyond sparse terminal reward.

## Grading rubric

Final score is the weighted sum of:

- item resolution accuracy: `0.30`
- ticket resolution accuracy: `0.15`
- priority accuracy: `0.10`
- tag quality (F1): `0.10`
- evidence coverage: `0.10`
- policy compliance: `0.10`
- reply quality: `0.15`

The grader is fully deterministic.

## Project structure

```text
return_desk_env/
├── README.md
├── __init__.py
├── client.py
├── models.py
├── openenv.yaml
├── pyproject.toml
├── uv.lock
├── Dockerfile
├── inference.py
├── graders.py
├── rewards.py
├── tasks/
│   ├── __init__.py
│   └── catalog.py
├── server/
│   ├── __init__.py
│   ├── app.py
│   ├── environment.py
│   └── Dockerfile
├── tests/
│   ├── test_environment_logic.py
│   ├── test_graders.py
│   └── test_inference_smoke.py
└── scripts/
    └── prep_submission.sh
```

## Local development

Install dependencies:

```bash
pip install -e .
```

Run the server:

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Or, if you have `uv` set up:

```bash
uv run server
```

## Docker build

Root-level Docker build:

```bash
docker build -t return-desk-env:latest .
```

Alternative build using the server Dockerfile from repo root:

```bash
docker build -t return-desk-env:latest -f server/Dockerfile .
```

Run locally:

```bash
docker run --rm -p 8000:8000 return-desk-env:latest
```

## OpenEnv validation

From the repo root:

```bash
openenv validate
openenv validate --verbose
```

## Inference script

The mandatory hackathon baseline lives at the repo root in `inference.py`.

It reads these environment variables:

- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

Optional environment variables:

- `RETURN_DESK_BASE_URL` (default: `http://localhost:8000`)
- `RETURN_DESK_TASKS` (comma-separated task ids)
- `RETURN_DESK_MAX_STEPS` (default: `12`)

Example:

```bash
export API_BASE_URL="https://router.huggingface.co/v1"
export MODEL_NAME="your-model"
export HF_TOKEN="your-token"
export RETURN_DESK_BASE_URL="http://localhost:8000"
python inference.py
```

## Hugging Face Space deployment

This repo already includes the YAML front matter required for a Docker Space. The server listens on port `8000`, so `app_port: 8000` is set at the top of this README.

Recommended flow:

```bash
openenv validate
openenv push
```

Or create a Docker Space manually and push this repo.

## Submission checklist

Before final hackathon submission:

1. run `uv lock` locally to replace the placeholder `uv.lock`
2. run `openenv validate`
3. build Docker successfully
4. deploy the HF Space
5. confirm `POST /reset` returns HTTP 200 on the live Space
6. run `python inference.py` and record baseline scores in the README

## Baseline scores

Measured with `python inference.py` using the built-in deterministic policy (`RETURN_DESK_USE_LLM=false`):

| Task | Score | Steps |
|---|---|---|
| `easy_refund` | 1.000 | 9 |
| `medium_exchange` | 1.000 | 11 |
| `hard_partial_resolution` | 1.000 | 14 |
| **Mean** | **1.000** | — |

Measured with `Qwen/Qwen2.5-72B-Instruct` via HuggingFace Router (`RETURN_DESK_USE_LLM=true`):

| Task | Score | Steps |
|---|---|---|
| `easy_refund` | 0.842 | 7 |
| `medium_exchange` | 0.762 | 7 |
| `hard_partial_resolution` | 0.902 | 11 |
| **Mean** | **0.836** | — |

The deterministic policy provides a perfect 1.000 reference implementation.
The LLM baseline demonstrates genuine multi-step reasoning across all three difficulty levels.

## Suggested baseline score targets

A simple general-purpose LLM baseline should usually:

- solve `easy_refund` reliably
- do reasonably on `medium_exchange`
- be meaningfully challenged by `hard_partial_resolution`

That spread is deliberate: it shows task progression and non-trivial difficulty.
