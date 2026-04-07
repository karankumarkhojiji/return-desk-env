from __future__ import annotations

import threading
from typing import Any, Dict, Optional

from fastapi import HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from openenv.core.env_server.http_server import create_app

try:
    from ..models import ReturnDeskAction, ReturnDeskObservation
    from .environment import ReturnDeskEnvironment
except ImportError:  # pragma: no cover - supports running from repo root
    from models import ReturnDeskAction, ReturnDeskObservation
    from server.environment import ReturnDeskEnvironment


app = create_app(
    ReturnDeskEnvironment,
    ReturnDeskAction,
    ReturnDeskObservation,
    env_name="return_desk_env",
)

# ---------------------------------------------------------------------------
# Stateful HTTP session — lets Swagger UI and curl work across reset→step
# ---------------------------------------------------------------------------
_env_lock = threading.Lock()
_stateful_env: Optional[ReturnDeskEnvironment] = None


class DemoResetRequest(BaseModel):
    task_id: Optional[str] = "easy_refund"
    difficulty: Optional[str] = None


class DemoStepRequest(BaseModel):
    action: Dict[str, Any]


def _build_response(obs: ReturnDeskObservation, reward=None, done=False) -> Dict[str, Any]:
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
    }


@app.post(
    "/api/reset",
    summary="[Stateful] Reset environment",
    description=(
        "Starts a new episode. Call this before /api/step.\n\n"
        "Unlike the built-in /reset, this endpoint keeps state between calls "
        "so you can chain multiple /api/step calls after it."
    ),
    tags=["Stateful Demo API"],
)
def api_reset(request: DemoResetRequest) -> Dict[str, Any]:
    global _stateful_env
    with _env_lock:
        if _stateful_env is not None:
            try:
                _stateful_env.close()
            except Exception:
                pass
        _stateful_env = ReturnDeskEnvironment()
        kwargs: Dict[str, Any] = {}
        if request.task_id:
            kwargs["task_id"] = request.task_id
        elif request.difficulty:
            kwargs["difficulty"] = request.difficulty
        obs = _stateful_env.reset(**kwargs)
    return _build_response(obs, reward=None, done=False)


@app.post(
    "/api/step",
    summary="[Stateful] Execute one action",
    description=(
        "Executes one action in the running episode.\n\n"
        "You MUST call /api/reset first.\n\n"
        "**action_type options:** `inspect_order`, `inspect_customer`, `inspect_policy`, "
        "`inspect_inventory`, `set_priority`, `add_tag`, `set_item_resolution`, "
        "`set_ticket_resolution`, `draft_reply`, `submit`\n\n"
        "**Example bodies:**\n"
        "```json\n"
        '{\"action\": {\"action_type\": \"inspect_order\"}}\n'
        '{\"action\": {\"action_type\": \"set_priority\", \"priority\": \"high\"}}\n'
        '{\"action\": {\"action_type\": \"add_tag\", \"tag\": \"damaged\"}}\n'
        '{\"action\": {\"action_type\": \"set_item_resolution\", \"item_id\": \"item-1\", \"resolution\": \"refund\"}}\n'
        '{\"action\": {\"action_type\": \"set_ticket_resolution\", \"resolution\": \"refund\"}}\n'
        '{\"action\": {\"action_type\": \"draft_reply\", \"message\": \"We are sorry...\"}}\n'
        '{\"action\": {\"action_type\": \"submit\"}}\n'
        "```"
    ),
    tags=["Stateful Demo API"],
)
def api_step(request: DemoStepRequest) -> Dict[str, Any]:
    global _stateful_env
    with _env_lock:
        if _stateful_env is None:
            raise HTTPException(
                status_code=400,
                detail="No active episode. Call POST /api/reset first.",
            )
        try:
            action = ReturnDeskAction(**request.action)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc))

        obs = _stateful_env.step(action)
        reward = obs.reward
        done = obs.done
    return _build_response(obs, reward=reward, done=done)


# ---------------------------------------------------------------------------
# Interactive Web UI  — visit http://localhost:8000/web
# ---------------------------------------------------------------------------
@app.get("/web", response_class=HTMLResponse, include_in_schema=False)
def web_ui() -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>ReturnDeskEnv — Interactive Demo</title>
<link rel="preconnect" href="https://fonts.googleapis.com"/>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  :root {
    --bg: #0d0f14; --surface: #151821; --surface2: #1c2030;
    --border: #252a3a; --accent: #6366f1; --accent2: #8b5cf6;
    --green: #22c55e; --red: #ef4444; --yellow: #f59e0b; --blue: #3b82f6;
    --text: #e2e8f0; --muted: #64748b; --tag-bg: #1e2a3a;
  }
  body { font-family: 'Inter', sans-serif; background: var(--bg); color: var(--text);
    min-height: 100vh; display: flex; flex-direction: column; }
  header { background: linear-gradient(135deg, #1a1f35 0%, #0d1120 100%);
    border-bottom: 1px solid var(--border); padding: 18px 32px;
    display: flex; align-items: center; gap: 16px; }
  header .logo { font-size: 22px; font-weight: 700;
    background: linear-gradient(135deg, var(--accent), var(--accent2));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  header .badge { font-size: 11px; font-weight: 600; padding: 3px 10px;
    border-radius: 20px; background: rgba(99,102,241,0.2); color: var(--accent);
    border: 1px solid rgba(99,102,241,0.3); letter-spacing: 0.5px; }
  .main { display: grid; grid-template-columns: 360px 1fr; gap: 0; flex: 1; overflow: hidden; }
  .panel { padding: 24px; overflow-y: auto; }
  .left-panel { border-right: 1px solid var(--border); background: var(--surface); display: flex; flex-direction: column; gap: 20px; }
  .right-panel { background: var(--bg); display: flex; flex-direction: column; gap: 0; }
  .card { background: var(--surface2); border: 1px solid var(--border);
    border-radius: 12px; padding: 18px; }
  .card-title { font-size: 11px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1px; color: var(--muted); margin-bottom: 14px; }
  select, input[type=text] {
    width: 100%; background: var(--bg); border: 1px solid var(--border);
    color: var(--text); border-radius: 8px; padding: 10px 12px;
    font-family: inherit; font-size: 13px; outline: none; transition: border-color .2s; }
  select:focus, input[type=text]:focus { border-color: var(--accent); }
  .btn { width: 100%; padding: 11px; border: none; border-radius: 8px; font-family: inherit;
    font-size: 13px; font-weight: 600; cursor: pointer; transition: all .2s; }
  .btn-primary { background: linear-gradient(135deg, var(--accent), var(--accent2));
    color: white; }
  .btn-primary:hover { opacity: 0.9; transform: translateY(-1px); box-shadow: 0 4px 20px rgba(99,102,241,0.4); }
  .btn-secondary { background: var(--surface); border: 1px solid var(--border); color: var(--text); }
  .btn-secondary:hover { border-color: var(--accent); color: var(--accent); }
  .btn:disabled { opacity: 0.4; cursor: not-allowed; transform: none; }
  .btn-submit { background: linear-gradient(135deg, var(--green), #16a34a); color: white; }
  .btn-submit:hover { opacity: 0.9; transform: translateY(-1px); }
  .action-group { display: flex; flex-direction: column; gap: 8px; }
  .action-row { display: grid; grid-template-columns: 1fr auto; gap: 8px; align-items: end; }
  label { font-size: 12px; color: var(--muted); display: block; margin-bottom: 5px; }
  .status-bar { background: var(--surface2); border-bottom: 1px solid var(--border);
    padding: 12px 24px; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }
  .stat { display: flex; align-items: center; gap: 8px; }
  .stat-label { font-size: 11px; color: var(--muted); }
  .stat-val { font-size: 13px; font-weight: 600; }
  .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--muted); }
  .dot.active { background: var(--green); box-shadow: 0 0 8px var(--green); }
  .log-area { flex: 1; padding: 20px 24px; overflow-y: auto; display: flex; flex-direction: column; gap: 10px; }
  .log-entry { border-radius: 10px; padding: 14px 16px; border: 1px solid transparent; animation: fadeIn .3s ease; }
  @keyframes fadeIn { from { opacity: 0; transform: translateY(8px); } to { opacity: 1; transform: translateY(0); } }
  .log-system { background: rgba(99,102,241,0.08); border-color: rgba(99,102,241,0.2); }
  .log-pos { background: rgba(34,197,94,0.07); border-color: rgba(34,197,94,0.2); }
  .log-neg { background: rgba(239,68,68,0.07); border-color: rgba(239,68,68,0.2); }
  .log-done { background: linear-gradient(135deg, rgba(34,197,94,0.12), rgba(59,130,246,0.12)); border-color: rgba(34,197,94,0.3); }
  .log-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
  .log-action { font-size: 13px; font-weight: 600; font-family: 'JetBrains Mono', monospace; }
  .log-reward { font-size: 12px; font-weight: 700; padding: 2px 8px; border-radius: 5px; }
  .reward-pos { background: rgba(34,197,94,0.2); color: var(--green); }
  .reward-neg { background: rgba(239,68,68,0.2); color: var(--red); }
  .log-note { font-size: 12px; color: var(--muted); }
  .score-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 10px; }
  .score-item { background: var(--surface); border-radius: 8px; padding: 10px; }
  .score-name { font-size: 10px; color: var(--muted); margin-bottom: 4px; }
  .score-bar-wrap { background: var(--border); border-radius: 3px; height: 4px; margin: 4px 0; }
  .score-bar { height: 4px; border-radius: 3px; background: linear-gradient(90deg, var(--accent), var(--green)); transition: width .6s ease; }
  .score-val { font-size: 12px; font-weight: 600; color: var(--green); }
  .ticket-preview { font-size: 12px; line-height: 1.6; color: var(--muted);
    background: var(--bg); border-radius: 8px; padding: 12px; border-left: 3px solid var(--accent); }
  .tags-display { display: flex; flex-wrap: wrap; gap: 6px; }
  .tag-chip { font-size: 11px; padding: 3px 9px; border-radius: 20px;
    background: var(--tag-bg); color: #94a3b8; border: 1px solid var(--border); font-family: 'JetBrains Mono', monospace; }
  .empty-state { text-align: center; padding: 60px 20px; color: var(--muted); }
  .empty-state .icon { font-size: 48px; margin-bottom: 16px; }
  .empty-state p { font-size: 14px; }
  .final-score { text-align: center; padding:20px 0 10px; }
  .final-score .number { font-size: 52px; font-weight: 700;
    background: linear-gradient(135deg, var(--green), var(--accent));
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .divider { height: 1px; background: var(--border); }
  .inline-btn { background: var(--accent); color: white; border: none; border-radius: 6px;
    padding: 8px 14px; font-family: inherit; font-size: 12px; font-weight: 600;
    cursor: pointer; white-space: nowrap; transition: opacity .2s; }
  .inline-btn:hover { opacity: 0.85; }
  select option { background: var(--surface2); }
</style>
</head>
<body>
<header>
  <div class="logo">📦 ReturnDeskEnv</div>
  <div class="badge">OpenEnv · Meta PyTorch Hackathon</div>
</header>

<div class="main">
  <!-- LEFT: Controls -->
  <div class="left-panel panel">

    <div class="card">
      <div class="card-title">1 — Start Episode</div>
      <label>Choose Task</label>
      <select id="taskSelect">
        <option value="easy_refund">🟢 easy_refund — Damaged item refund</option>
        <option value="medium_exchange">🟡 medium_exchange — Size exchange</option>
        <option value="hard_partial_resolution">🔴 hard_partial_resolution — Multi-item mixed</option>
      </select>
      <br/><br/>
      <button class="btn btn-primary" onclick="doReset()">⚡ Reset / Start New Episode</button>
    </div>

    <div class="card">
      <div class="card-title">2 — Take Action</div>
      <div class="action-group">

        <div>
          <label>Action Type</label>
          <select id="actionType" onchange="updateFields()">
            <option value="inspect_order">inspect_order</option>
            <option value="inspect_customer">inspect_customer</option>
            <option value="inspect_policy">inspect_policy</option>
            <option value="inspect_inventory">inspect_inventory</option>
            <option value="set_priority">set_priority</option>
            <option value="add_tag">add_tag</option>
            <option value="set_item_resolution">set_item_resolution</option>
            <option value="set_ticket_resolution">set_ticket_resolution</option>
            <option value="draft_reply">draft_reply</option>
            <option value="submit">submit</option>
          </select>
        </div>

        <div id="priorityField" style="display:none">
          <label>Priority</label>
          <select id="priorityVal">
            <option value="low">low</option>
            <option value="medium">medium</option>
            <option value="high" selected>high</option>
            <option value="urgent">urgent</option>
          </select>
        </div>

        <div id="tagField" style="display:none">
          <label>Tag</label>
          <select id="tagVal">
            <option value="damaged">damaged</option>
            <option value="refund_request">refund_request</option>
            <option value="exchange_request">exchange_request</option>
            <option value="inventory_issue">inventory_issue</option>
            <option value="coupon_order">coupon_order</option>
            <option value="vip_exception">vip_exception</option>
            <option value="partial_resolution">partial_resolution</option>
            <option value="non_returnable">non_returnable</option>
          </select>
        </div>

        <div id="itemIdField" style="display:none">
          <label>Item ID</label>
          <select id="itemIdVal">
            <option value="item-1">item-1</option>
            <option value="item-2">item-2</option>
            <option value="item-3">item-3</option>
          </select>
        </div>

        <div id="resolutionField" style="display:none">
          <label>Resolution</label>
          <select id="resolutionVal">
            <option value="refund">refund</option>
            <option value="exchange">exchange</option>
            <option value="store_credit">store_credit</option>
            <option value="deny">deny</option>
            <option value="escalate">escalate</option>
            <option value="request_info">request_info</option>
            <option value="partial_refund">partial_refund</option>
          </select>
        </div>

        <div id="messageField" style="display:none">
          <label>Reply Message</label>
          <textarea id="messageVal"
            style="width:100%;background:var(--bg);border:1px solid var(--border);color:var(--text);
              border-radius:8px;padding:10px 12px;font-family:inherit;font-size:12px;
              resize:vertical;min-height:90px;outline:none;"
            placeholder="Type your customer reply here..."></textarea>
        </div>

        <button class="btn btn-secondary" id="stepBtn" onclick="doStep()" disabled>
          ▶ Send Action
        </button>

        <!-- Quick-fill perfect solution button -->
        <button class="btn" id="autoBtn" onclick="autoPlay()"
          style="background:rgba(245,158,11,0.15);border:1px solid rgba(245,158,11,0.3);color:var(--yellow);" disabled>
          ⚡ Auto-play Perfect Solution
        </button>
      </div>
    </div>

    <div class="card" id="ticketCard" style="display:none">
      <div class="card-title">Customer Ticket</div>
      <div class="ticket-preview" id="ticketText"></div>
    </div>

    <div class="card" id="stateCard" style="display:none">
      <div class="card-title">Current State</div>
      <div style="display:flex;flex-direction:column;gap:8px;font-size:12px;">
        <div><span style="color:var(--muted)">Priority: </span><span id="stPriority" style="font-family:'JetBrains Mono',monospace">—</span></div>
        <div><span style="color:var(--muted)">Ticket Resolution: </span><span id="stTicketRes" style="font-family:'JetBrains Mono',monospace">—</span></div>
        <div><span style="color:var(--muted)">Tags: </span><div class="tags-display" id="stTags" style="margin-top:4px"></div></div>
        <div><span style="color:var(--muted)">Item Resolutions: </span><div class="tags-display" id="stItemRes" style="margin-top:4px"></div></div>
        <div id="stReplySec" style="display:none"><span style="color:var(--muted)">Reply drafted: </span><span style="color:var(--green)">✓</span></div>
      </div>
    </div>

  </div>

  <!-- RIGHT: Log -->
  <div class="right-panel">
    <div class="status-bar">
      <div class="stat">
        <div class="dot" id="statusDot"></div>
        <span class="stat-label">Status:</span>
        <span class="stat-val" id="statusText">No active episode</span>
      </div>
      <div class="stat">
        <span class="stat-label">Task:</span>
        <span class="stat-val" id="statusTask">—</span>
      </div>
      <div class="stat">
        <span class="stat-label">Steps left:</span>
        <span class="stat-val" id="statusSteps">—</span>
      </div>
      <div class="stat" style="margin-left:auto">
        <button class="inline-btn" onclick="clearLog()">Clear Log</button>
      </div>
    </div>
    <div class="log-area" id="logArea">
      <div class="empty-state">
        <div class="icon">📋</div>
        <p>Click <strong>Reset / Start New Episode</strong> to begin.</p>
        <br/>
        <p style="font-size:12px">Then send actions one by one and watch your score build up.</p>
      </div>
    </div>
  </div>
</div>

<script>
const BASE = window.location.origin;
let episodeActive = false;
let taskId = 'easy_refund';
let autoPlaying = false;

const perfectSolutions = {
  easy_refund: [
    {action_type:'inspect_order'},
    {action_type:'inspect_policy'},
    {action_type:'set_priority', priority:'high'},
    {action_type:'add_tag', tag:'damaged'},
    {action_type:'add_tag', tag:'refund_request'},
    {action_type:'set_item_resolution', item_id:'item-1', resolution:'refund'},
    {action_type:'set_ticket_resolution', resolution:'refund'},
    {action_type:'draft_reply', message:'We are sorry your BrewMaster Coffee Grinder arrived damaged. We will process a refund, and you should see it in 3-5 business days. No return required for the damaged unit.'},
    {action_type:'submit'},
  ],
  medium_exchange: [
    {action_type:'inspect_order'},
    {action_type:'inspect_policy'},
    {action_type:'inspect_inventory'},
    {action_type:'set_priority', priority:'medium'},
    {action_type:'add_tag', tag:'exchange_request'},
    {action_type:'add_tag', tag:'inventory_issue'},
    {action_type:'add_tag', tag:'coupon_order'},
    {action_type:'set_item_resolution', item_id:'item-1', resolution:'store_credit'},
    {action_type:'set_ticket_resolution', resolution:'store_credit'},
    {action_type:'draft_reply', message:'We are sorry the exact blue size L hoodie is unavailable. We can issue store credit for the amount paid, $51.00, since the order used a coupon.'},
    {action_type:'submit'},
  ],
  hard_partial_resolution: [
    {action_type:'inspect_order'},
    {action_type:'inspect_customer'},
    {action_type:'inspect_policy'},
    {action_type:'set_priority', priority:'high'},
    {action_type:'add_tag', tag:'damaged'},
    {action_type:'add_tag', tag:'vip_exception'},
    {action_type:'add_tag', tag:'partial_resolution'},
    {action_type:'add_tag', tag:'non_returnable'},
    {action_type:'set_item_resolution', item_id:'item-1', resolution:'refund'},
    {action_type:'set_item_resolution', item_id:'item-2', resolution:'refund'},
    {action_type:'set_item_resolution', item_id:'item-3', resolution:'deny'},
    {action_type:'set_ticket_resolution', resolution:'partial_refund'},
    {action_type:'draft_reply', message:'We are sorry for the issues with your order. For the AirFry Pro we will process a refund under a VIP exception. For the Glass Storage Set we will process a refund because it arrived damaged. The Monogram Apron is personalized, so it is a partial resolution covering two items.'},
    {action_type:'submit'},
  ],
};

function updateFields() {
  const t = document.getElementById('actionType').value;
  document.getElementById('priorityField').style.display = t==='set_priority' ? '' : 'none';
  document.getElementById('tagField').style.display = t==='add_tag' ? '' : 'none';
  document.getElementById('itemIdField').style.display = t==='set_item_resolution' ? '' : 'none';
  document.getElementById('resolutionField').style.display = (t==='set_item_resolution'||t==='set_ticket_resolution') ? '' : 'none';
  document.getElementById('messageField').style.display = t==='draft_reply' ? '' : 'none';
}

function buildAction() {
  const t = document.getElementById('actionType').value;
  const a = {action_type: t};
  if (t==='set_priority') a.priority = document.getElementById('priorityVal').value;
  if (t==='add_tag')      a.tag = document.getElementById('tagVal').value;
  if (t==='set_item_resolution') {
    a.item_id = document.getElementById('itemIdVal').value;
    a.resolution = document.getElementById('resolutionVal').value;
  }
  if (t==='set_ticket_resolution') a.resolution = document.getElementById('resolutionVal').value;
  if (t==='draft_reply')  a.message = document.getElementById('messageVal').value;
  return a;
}

function addLog(html, cls='log-system') {
  const area = document.getElementById('logArea');
  const empty = area.querySelector('.empty-state');
  if (empty) empty.remove();
  const div = document.createElement('div');
  div.className = 'log-entry ' + cls;
  div.innerHTML = html;
  area.appendChild(div);
  area.scrollTop = area.scrollHeight;
}

function clearLog() {
  document.getElementById('logArea').innerHTML =
    '<div class="empty-state"><div class="icon">📋</div><p>Log cleared. Start a new episode!</p></div>';
}

function updateStatusBar(obs, done) {
  document.getElementById('statusDot').className = 'dot' + (episodeActive && !done ? ' active' : '');
  document.getElementById('statusText').textContent = done ? '✅ Episode complete' : 'Running';
  document.getElementById('statusTask').textContent = obs.task_id || '—';
  document.getElementById('statusSteps').textContent = obs.steps_remaining ?? '—';
}

function updateStatePanel(obs) {
  document.getElementById('stateCard').style.display = '';
  document.getElementById('stPriority').textContent = obs.current_priority || '—';
  document.getElementById('stTicketRes').textContent = obs.ticket_resolution || '—';

  const tagsEl = document.getElementById('stTags');
  tagsEl.innerHTML = obs.current_tags && obs.current_tags.length
    ? obs.current_tags.map(t=>`<span class="tag-chip">${t}</span>`).join('')
    : '<span style="color:var(--muted);font-size:11px">none</span>';

  const itemsEl = document.getElementById('stItemRes');
  const resEntries = Object.entries(obs.item_resolutions || {});
  itemsEl.innerHTML = resEntries.length
    ? resEntries.map(([id,r])=>`<span class="tag-chip">${id}: ${r}</span>`).join('')
    : '<span style="color:var(--muted);font-size:11px">none</span>';

  document.getElementById('stReplySec').style.display = obs.drafted_reply ? '' : 'none';
}

async function doReset() {
  taskId = document.getElementById('taskSelect').value;
  try {
    const r = await fetch(`${BASE}/api/reset`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({task_id: taskId}),
    });
    const data = await r.json();
    if (!r.ok) { addLog(`<div style="color:var(--red)">❌ Reset error: ${JSON.stringify(data)}</div>`); return; }

    const obs = data.observation;
    episodeActive = true;
    document.getElementById('stepBtn').disabled = false;
    document.getElementById('autoBtn').disabled = false;

    document.getElementById('ticketCard').style.display = '';
    document.getElementById('ticketText').textContent = obs.customer_ticket?.message || '';
    updateStatusBar(obs, false);
    updateStatePanel(obs);

    addLog(`
      <div class="log-header">
        <span class="log-action">⚡ Episode started — ${obs.task_id}</span>
        <span style="font-size:11px;color:var(--muted)">${obs.difficulty} · ${obs.steps_remaining} steps</span>
      </div>
      <div class="log-note">${obs.objective}</div>
    `, 'log-system');
  } catch(e) { addLog(`<div style="color:var(--red)">❌ ${e}</div>`); }
}

async function doStep() {
  const action = buildAction();
  await sendAction(action);
}

async function sendAction(action) {
  try {
    const r = await fetch(`${BASE}/api/step`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({action}),
    });
    const data = await r.json();
    if (!r.ok) {
      addLog(`<div class="log-header"><span class="log-action" style="color:var(--red)">❌ ${action.action_type}</span></div><div class="log-note">${JSON.stringify(data.detail)}</div>`, 'log-neg');
      return false;
    }

    const obs = data.observation;
    const reward = data.reward ?? 0;
    const done = data.done;
    const cls = done ? 'log-done' : reward >= 0 ? 'log-pos' : 'log-neg';
    const rewardCls = reward >= 0 ? 'reward-pos' : 'reward-neg';
    const sign = reward >= 0 ? '+' : '';

    addLog(`
      <div class="log-header">
        <span class="log-action">${action.action_type}</span>
        <span class="log-reward ${rewardCls}">${sign}${reward.toFixed(3)}</span>
      </div>
      <div class="log-note">${obs.latest_note}</div>
    `, cls);

    updateStatusBar(obs, done);
    updateStatePanel(obs);

    if (done) {
      episodeActive = false;
      document.getElementById('stepBtn').disabled = true;
      document.getElementById('autoBtn').disabled = true;

      const bd = obs.grader_breakdown || {};
      const items = Object.entries(bd).map(([k,v])=>`
        <div class="score-item">
          <div class="score-name">${k.replace(/_/g,' ')}</div>
          <div class="score-bar-wrap"><div class="score-bar" style="width:${v*100}%"></div></div>
          <div class="score-val">${v.toFixed(2)}</div>
        </div>`).join('');

      addLog(`
        <div class="final-score">
          <div style="font-size:13px;color:var(--muted);margin-bottom:4px">Final Score</div>
          <div class="number">${(obs.final_score||0).toFixed(3)}</div>
          <div style="font-size:12px;color:var(--muted)">${obs.latest_note}</div>
        </div>
        <div class="divider" style="margin:12px 0"></div>
        <div class="score-grid">${items}</div>
      `, 'log-done');
    }
    return done;
  } catch(e) {
    addLog(`<div style="color:var(--red)">❌ ${e.message}</div>`);
    return false;
  }
}

async function autoPlay() {
  if (autoPlaying) return;
  autoPlaying = true;
  document.getElementById('autoBtn').disabled = true;
  document.getElementById('stepBtn').disabled = true;

  const steps = perfectSolutions[taskId] || perfectSolutions['easy_refund'];
  for (const action of steps) {
    const done = await sendAction(action);
    await new Promise(r => setTimeout(r, 500));
    if (done) break;
  }
  autoPlaying = false;
}

updateFields();
</script>
</body>
</html>"""


def main() -> None:
    import uvicorn
    uvicorn.run(app, port=8000)


if __name__ == "__main__":
    main()
