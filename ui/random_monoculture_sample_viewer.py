#!/usr/bin/env python3
"""Small HTTP viewer for random-monoculture sample rollouts.

This intentionally avoids Streamlit so it stays stable behind a simple SSH
tunnel on login nodes.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import traceback
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = (
    PROJECT_ROOT
    / "experiments/results/full_games123_random_monoculture_control_20260628_014357"
)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def tail_text(path: Path | None, max_bytes: int = 24000) -> str:
    if path is None or not path.exists():
        return ""
    size = path.stat().st_size
    with path.open("rb") as handle:
        if size > max_bytes:
            handle.seek(size - max_bytes)
        data = handle.read()
    text = data.decode("utf-8", errors="replace")
    if size > max_bytes:
        return "[tail excerpt]\n" + text
    return text


def resolve_output_dir(raw_output_dir: str) -> Path:
    candidate = Path(raw_output_dir)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def compact_text(value: Any, limit: int = 260) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "..."


def sum_utilities(value: Any) -> float | None:
    if not isinstance(value, dict):
        return None
    total = 0.0
    for item in value.values():
        try:
            total += float(item)
        except (TypeError, ValueError):
            return None
    return total


def result_path_for(root: Path, config: dict[str, Any], status: dict[str, Any]) -> Path | None:
    raw_status_path = status.get("result_path")
    if raw_status_path:
        path = Path(str(raw_status_path))
        if path.exists():
            return path

    raw_output_dir = config.get("output_dir")
    if isinstance(raw_output_dir, str):
        output_dir = resolve_output_dir(raw_output_dir)
        for name in ("experiment_results.json", "run_1_experiment_results.json"):
            candidate = output_dir / name
            if candidate.exists():
                return candidate

    config_id = str(config.get("config_id", ""))
    matches = sorted((root / "runs").glob(f"{config_id}_*/experiment_results.json"))
    return matches[-1] if matches else None


def interactions_path_for(result_path: Path | None) -> Path | None:
    if result_path is None:
        return None
    for name in ("all_interactions.json", "run_1_all_interactions.json"):
        candidate = result_path.with_name(name)
        if candidate.exists():
            return candidate
    return None


@dataclass
class RunRecord:
    config_id: str
    config_path: Path
    status_path: Path | None
    result_path: Path | None
    interactions_path: Path | None
    log_path: Path | None
    attempt_log_path: Path | None
    row: dict[str, Any]


class BatchIndex:
    def __init__(self, results_root: Path):
        self.results_root = results_root.resolve()
        self.records: dict[str, RunRecord] = {}
        self.rows: list[dict[str, Any]] = []
        self.reload()

    def reload(self) -> None:
        records: dict[str, RunRecord] = {}
        rows: list[dict[str, Any]] = []
        configs_dir = self.results_root / "configs"
        for config_path in sorted(configs_dir.glob("config_*.json")):
            try:
                config = read_json(config_path)
            except Exception:
                continue

            config_id = str(config.get("config_id") or config_path.stem)
            status_path = self.results_root / "status" / f"{config_id}.json"
            status: dict[str, Any] = {}
            if status_path.exists():
                try:
                    status = read_json(status_path)
                except Exception:
                    status = {}

            state = str(status.get("state") or "NOT_STARTED")
            result_path = result_path_for(self.results_root, config, status)
            result: dict[str, Any] = {}
            if result_path is not None:
                try:
                    result = read_json(result_path)
                    if status.get("result_validation_error") is None:
                        state = "SUCCESS"
                except Exception:
                    result = {}

            game_type = config.get("game_type") or result.get("config", {}).get("game_type")
            final_utilities = result.get("final_utilities")
            vote_integrity = result.get("vote_integrity") or {}
            conversation_logs = result.get("conversation_logs") or []
            row = {
                "config_id": config_id,
                "state": state,
                "game_label": config.get("game_label"),
                "game_type": game_type,
                "n_agents": config.get("n_agents") or config.get("num_agents"),
                "model": config.get("monoculture_model") or config.get("baseline_model"),
                "model_elo": config.get("model_elo"),
                "competition_id": config.get("competition_id"),
                "final_round": result.get("final_round"),
                "consensus": result.get("consensus_reached"),
                "conversation_logs": len(conversation_logs) if isinstance(conversation_logs, list) else 0,
                "utility_sum": sum_utilities(final_utilities),
                "synthetic_votes": vote_integrity.get("synthetic_vote_count", 0),
                "vote_contaminated": bool(vote_integrity.get("contaminated")),
                "vote_hard_failed": bool(vote_integrity.get("hard_failed")),
                "duration_seconds": status.get("duration_seconds"),
                "started_at": status.get("started_at"),
                "finished_at": status.get("finished_at"),
                "result_validation_error": status.get("result_validation_error"),
                "run_dir": str(result_path.parent) if result_path else None,
            }
            record = RunRecord(
                config_id=config_id,
                config_path=config_path,
                status_path=status_path if status_path.exists() else None,
                result_path=result_path,
                interactions_path=interactions_path_for(result_path),
                log_path=Path(str(status["log_path"])) if status.get("log_path") else None,
                attempt_log_path=Path(str(status["attempt_log_path"])) if status.get("attempt_log_path") else None,
                row=row,
            )
            records[config_id] = record
            rows.append(row)

        self.records = records
        self.rows = sorted(rows, key=lambda item: item["config_id"])

    def summary(self) -> dict[str, Any]:
        states: dict[str, int] = {}
        games: dict[str, int] = {}
        for row in self.rows:
            states[str(row.get("state"))] = states.get(str(row.get("state")), 0) + 1
            games[str(row.get("game_label"))] = games.get(str(row.get("game_label")), 0) + 1
        return {
            "results_root": str(self.results_root),
            "total": len(self.rows),
            "states": states,
            "games": games,
        }

    def get(self, config_id: str) -> RunRecord:
        if config_id not in self.records:
            raise KeyError(config_id)
        return self.records[config_id]


INDEX: BatchIndex


HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Random Monoculture Sample Viewer</title>
  <style>
    :root {
      --bg: #f7f8fa;
      --panel: #ffffff;
      --line: #d8dde6;
      --text: #18202a;
      --muted: #5c6675;
      --accent: #0f766e;
      --accent2: #1d4ed8;
      --bad: #b91c1c;
      --warn: #b45309;
      --good: #047857;
      --mono: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace;
      --sans: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: var(--sans);
      color: var(--text);
      background: var(--bg);
    }
    .app {
      display: grid;
      grid-template-columns: 380px minmax(0, 1fr);
      min-height: 100vh;
    }
    aside {
      border-right: 1px solid var(--line);
      background: var(--panel);
      height: 100vh;
      overflow: auto;
      padding: 14px;
    }
    main {
      height: 100vh;
      overflow: auto;
      padding: 18px 22px 48px;
    }
    h1 { font-size: 19px; margin: 0 0 6px; }
    h2 { font-size: 18px; margin: 22px 0 10px; }
    h3 { font-size: 15px; margin: 16px 0 8px; }
    .muted { color: var(--muted); }
    .small { font-size: 12px; }
    .mono { font-family: var(--mono); }
    .topline {
      display: flex;
      gap: 10px;
      align-items: center;
      justify-content: space-between;
      margin-bottom: 12px;
    }
    .stats {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px;
      margin: 12px 0;
    }
    .stat {
      border: 1px solid var(--line);
      background: #fbfcfd;
      padding: 9px 10px;
      border-radius: 6px;
    }
    .stat b { display: block; font-size: 17px; }
    label { display: block; font-size: 12px; font-weight: 650; color: #354052; margin: 10px 0 4px; }
    input, select, button {
      width: 100%;
      border: 1px solid var(--line);
      background: white;
      color: var(--text);
      border-radius: 6px;
      padding: 8px 9px;
      font: inherit;
      font-size: 13px;
    }
    button {
      cursor: pointer;
      font-weight: 650;
      background: #f4f7fb;
    }
    button:hover { border-color: #9aa7ba; }
    .filters {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 8px;
    }
    .filters .wide { grid-column: 1 / -1; }
    .run-list {
      margin-top: 12px;
      border-top: 1px solid var(--line);
    }
    .run-row {
      display: grid;
      grid-template-columns: 86px minmax(0, 1fr);
      gap: 8px;
      padding: 9px 2px;
      border-bottom: 1px solid #edf0f4;
      cursor: pointer;
    }
    .run-row:hover { background: #f7fafc; }
    .run-row.active {
      background: #e9f5f3;
      border-left: 3px solid var(--accent);
      padding-left: 6px;
    }
    .badge {
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 2px 7px;
      font-size: 11px;
      font-weight: 700;
      background: #e5e7eb;
      color: #374151;
      white-space: nowrap;
    }
    .badge.good { background: #d1fae5; color: var(--good); }
    .badge.bad { background: #fee2e2; color: var(--bad); }
    .badge.warn { background: #fef3c7; color: var(--warn); }
    .summary-grid {
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 10px;
      margin: 14px 0;
    }
    .section {
      border: 1px solid var(--line);
      background: var(--panel);
      border-radius: 8px;
      padding: 14px;
      margin: 14px 0;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      text-align: left;
      vertical-align: top;
      border-bottom: 1px solid #edf0f4;
      padding: 7px 8px;
    }
    th {
      color: #334155;
      background: #f8fafc;
      font-weight: 750;
    }
    pre {
      white-space: pre-wrap;
      overflow-wrap: anywhere;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.45;
      background: #f8fafc;
      border: 1px solid #e5e7eb;
      border-radius: 6px;
      padding: 10px;
      max-height: 440px;
      overflow: auto;
    }
    .message {
      border: 1px solid #e1e6ee;
      background: #fff;
      border-radius: 7px;
      margin: 9px 0;
      overflow: hidden;
    }
    .message-head {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      padding: 8px 10px;
      background: #f8fafc;
      border-bottom: 1px solid #e7ebf0;
      font-size: 12px;
      color: #334155;
    }
    .message-body {
      padding: 10px;
      white-space: pre-wrap;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }
    .tabs {
      display: flex;
      gap: 8px;
      margin: 10px 0;
      border-bottom: 1px solid var(--line);
    }
    .tab {
      width: auto;
      border: 0;
      border-bottom: 3px solid transparent;
      border-radius: 0;
      background: transparent;
      padding: 9px 10px;
    }
    .tab.active {
      border-bottom-color: var(--accent);
      color: var(--accent);
    }
    .two-col {
      display: grid;
      grid-template-columns: minmax(0, 1fr) minmax(0, 1fr);
      gap: 12px;
    }
    .hidden { display: none; }
    .error { color: var(--bad); font-weight: 700; }
    @media (max-width: 900px) {
      .app { grid-template-columns: 1fr; }
      aside { height: auto; max-height: 55vh; border-right: 0; border-bottom: 1px solid var(--line); }
      main { height: auto; }
      .summary-grid, .two-col { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="app">
    <aside>
      <div class="topline">
        <div>
          <h1>Sample Viewer</h1>
          <div id="rootPath" class="muted small mono"></div>
        </div>
        <button style="width:auto" onclick="reloadAll()">Reload</button>
      </div>
      <div class="stats">
        <div class="stat"><span class="small muted">Total</span><b id="statTotal">-</b></div>
        <div class="stat"><span class="small muted">Visible</span><b id="statVisible">-</b></div>
        <div class="stat"><span class="small muted">Success</span><b id="statSuccess">-</b></div>
        <div class="stat"><span class="small muted">Failed</span><b id="statFailed">-</b></div>
      </div>
      <div class="filters">
        <div class="wide">
          <label for="search">Search</label>
          <input id="search" type="search" placeholder="config, model, cell">
        </div>
        <div>
          <label for="gameFilter">Game</label>
          <select id="gameFilter"></select>
        </div>
        <div>
          <label for="statusFilter">Status</label>
          <select id="statusFilter"></select>
        </div>
        <div class="wide">
          <label for="modelFilter">Model</label>
          <select id="modelFilter"></select>
        </div>
      </div>
      <div id="runList" class="run-list"></div>
    </aside>
    <main>
      <div id="mainEmpty" class="section">Loading...</div>
      <div id="mainContent" class="hidden">
        <div class="topline">
          <div>
            <h1 id="runTitle"></h1>
            <div id="runSubtitle" class="muted mono small"></div>
          </div>
          <div id="runBadges"></div>
        </div>
        <div id="summaryGrid" class="summary-grid"></div>
        <div class="tabs">
          <button class="tab active" data-tab="rollout" onclick="setTab('rollout')">Rollout</button>
          <button class="tab" data-tab="outcome" onclick="setTab('outcome')">Outcome</button>
          <button class="tab" data-tab="config" onclick="setTab('config')">Config</button>
          <button class="tab" data-tab="interactions" onclick="setTab('interactions')">Interactions</button>
          <button class="tab" data-tab="raw" onclick="setTab('raw')">Raw</button>
        </div>
        <section id="tab-rollout" class="tabPanel"></section>
        <section id="tab-outcome" class="tabPanel hidden"></section>
        <section id="tab-config" class="tabPanel hidden"></section>
        <section id="tab-interactions" class="tabPanel hidden"></section>
        <section id="tab-raw" class="tabPanel hidden"></section>
      </div>
    </main>
  </div>
<script>
let runs = [];
let selectedId = null;
let selectedPayload = null;
let currentTab = 'rollout';

function esc(value) {
  return String(value ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  }[c]));
}

function fmt(value) {
  if (value === null || value === undefined || value === '') return '-';
  if (typeof value === 'number') {
    if (Number.isInteger(value)) return String(value);
    return String(Math.round(value * 1000) / 1000);
  }
  return String(value);
}

function badge(label, kind='') {
  return `<span class="badge ${kind}">${esc(label)}</span>`;
}

function stateKind(state) {
  if (state === 'SUCCESS') return 'good';
  if (state === 'FAILED') return 'bad';
  return 'warn';
}

async function getJson(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
  return await response.json();
}

async function reloadAll() {
  document.getElementById('mainEmpty').textContent = 'Loading...';
  document.getElementById('mainContent').classList.add('hidden');
  const payload = await getJson('/api/runs');
  runs = payload.runs;
  document.getElementById('rootPath').textContent = payload.results_root;
  buildFilters();
  renderRuns();
  const preferred = runs.find(r => r.state === 'SUCCESS') || runs[0];
  if (preferred) loadRun(preferred.config_id);
}

function buildFilters() {
  const games = [...new Set(runs.map(r => r.game_label).filter(Boolean))].sort();
  const states = [...new Set(runs.map(r => r.state).filter(Boolean))].sort();
  const models = [...new Set(runs.map(r => r.model).filter(Boolean))].sort();
  setOptions('gameFilter', ['All', ...games]);
  setOptions('statusFilter', ['All', ...states]);
  setOptions('modelFilter', ['All', ...models]);
  for (const id of ['search', 'gameFilter', 'statusFilter', 'modelFilter']) {
    document.getElementById(id).oninput = renderRuns;
  }
}

function setOptions(id, options) {
  const select = document.getElementById(id);
  const old = select.value;
  select.innerHTML = options.map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join('');
  if (options.includes(old)) select.value = old;
}

function filteredRuns() {
  const q = document.getElementById('search').value.trim().toLowerCase();
  const game = document.getElementById('gameFilter').value;
  const state = document.getElementById('statusFilter').value;
  const model = document.getElementById('modelFilter').value;
  return runs.filter(r => {
    if (game !== 'All' && r.game_label !== game) return false;
    if (state !== 'All' && r.state !== state) return false;
    if (model !== 'All' && r.model !== model) return false;
    if (!q) return true;
    const hay = [r.config_id, r.game_label, r.game_type, r.model, r.competition_id, r.state].join(' ').toLowerCase();
    return hay.includes(q);
  });
}

function renderRuns() {
  const visible = filteredRuns();
  const success = runs.filter(r => r.state === 'SUCCESS').length;
  const failed = runs.filter(r => r.state === 'FAILED').length;
  document.getElementById('statTotal').textContent = runs.length;
  document.getElementById('statVisible').textContent = visible.length;
  document.getElementById('statSuccess').textContent = success;
  document.getElementById('statFailed').textContent = failed;
  document.getElementById('runList').innerHTML = visible.map(r => `
    <div class="run-row ${r.config_id === selectedId ? 'active' : ''}" onclick="loadRun('${esc(r.config_id)}')">
      <div>
        <div class="mono small">${esc(r.config_id)}</div>
        ${badge(r.state, stateKind(r.state))}
      </div>
      <div>
        <div><b>${esc(r.game_label)}</b> ${esc(r.competition_id)}</div>
        <div class="muted small">${esc(r.model)} | N=${fmt(r.n_agents)} | round ${fmt(r.final_round)}</div>
      </div>
    </div>
  `).join('');
}

async function loadRun(configId) {
  selectedId = configId;
  renderRuns();
  document.getElementById('mainEmpty').textContent = `Loading ${configId}...`;
  document.getElementById('mainEmpty').classList.remove('hidden');
  document.getElementById('mainContent').classList.add('hidden');
  try {
    selectedPayload = await getJson(`/api/run?config_id=${encodeURIComponent(configId)}`);
    renderSelected();
    document.getElementById('mainEmpty').classList.add('hidden');
    document.getElementById('mainContent').classList.remove('hidden');
  } catch (err) {
    document.getElementById('mainEmpty').innerHTML = `<span class="error">${esc(err.message)}</span>`;
  }
}

function renderSelected() {
  const row = selectedPayload.row;
  document.getElementById('runTitle').textContent = `${row.config_id} | ${row.game_label} | ${row.model}`;
  document.getElementById('runSubtitle').textContent = row.run_dir || selectedPayload.config_path;
  const badges = [
    badge(row.state, stateKind(row.state)),
    badge(row.consensus === true ? 'consensus' : row.consensus === false ? 'no consensus' : 'no result', row.consensus === true ? 'good' : row.consensus === false ? 'warn' : ''),
    row.vote_contaminated || row.vote_hard_failed || row.synthetic_votes ? badge('vote issue', 'bad') : badge('votes clean', 'good'),
  ];
  document.getElementById('runBadges').innerHTML = badges.join(' ');
  document.getElementById('summaryGrid').innerHTML = [
    stat('Game', row.game_type || row.game_label),
    stat('Agents', row.n_agents),
    stat('Cell', row.competition_id),
    stat('Final Round', row.final_round ? `${row.final_round}/10` : '-'),
    stat('Utility Sum', row.utility_sum),
    stat('Logs', row.conversation_logs),
    stat('Duration', row.duration_seconds ? `${Math.round(row.duration_seconds)}s` : '-'),
    stat('Finished', row.finished_at || '-'),
  ].join('');
  renderRollout();
  renderOutcome();
  renderConfig();
  renderInteractionsPlaceholder();
  renderRaw();
  setTab(currentTab);
}

function stat(label, value) {
  return `<div class="stat"><span class="small muted">${esc(label)}</span><b>${esc(fmt(value))}</b></div>`;
}

function renderRollout() {
  const result = selectedPayload.result || {};
  const logs = Array.isArray(result.conversation_logs) ? result.conversation_logs : [];
  const failed = selectedPayload.row.state === 'FAILED';
  let html = `<div class="section"><h2>Rollout</h2>`;
  if (failed) {
    html += `<p class="error">This config has no completed result file.</p><pre>${esc(selectedPayload.error_tail || '')}</pre></div>`;
    document.getElementById('tab-rollout').innerHTML = html;
    return;
  }
  if (!logs.length) {
    html += `<p class="muted">No conversation logs found.</p></div>`;
    document.getElementById('tab-rollout').innerHTML = html;
    return;
  }
  const byRound = {};
  for (const log of logs) {
    const round = log.round ?? 'setup';
    if (!byRound[round]) byRound[round] = [];
    byRound[round].push(log);
  }
  for (const round of Object.keys(byRound).sort((a, b) => Number(a) - Number(b))) {
    html += `<h3>Round ${esc(round)}</h3>`;
    for (const log of byRound[round]) {
      html += messageBlock(log.phase, log.from, log.content, `turn ${fmt(log.discussion_turn)} | speaker ${fmt(log.speaker_order)}/${fmt(log.total_speakers)}`);
    }
  }
  html += `</div>`;
  document.getElementById('tab-rollout').innerHTML = html;
}

function messageBlock(phase, from, content, meta='') {
  return `<div class="message">
    <div class="message-head">
      ${badge(phase || 'phase')} <b>${esc(from || 'system')}</b> <span class="muted">${esc(meta)}</span>
    </div>
    <div class="message-body">${esc(content || '')}</div>
  </div>`;
}

function renderOutcome() {
  const result = selectedPayload.result || {};
  const row = selectedPayload.row;
  let html = `<div class="section"><h2>Outcome</h2>`;
  html += `<div class="two-col"><div><h3>Final Utilities</h3><pre>${esc(JSON.stringify(result.final_utilities ?? {}, null, 2))}</pre></div>`;
  html += `<div><h3>Final Allocation</h3><pre>${esc(JSON.stringify(result.final_allocation ?? null, null, 2))}</pre></div></div>`;
  html += `<h3>Vote Integrity</h3><pre>${esc(JSON.stringify(result.vote_integrity ?? {
    synthetic_votes: row.synthetic_votes,
    contaminated: row.vote_contaminated,
    hard_failed: row.vote_hard_failed
  }, null, 2))}</pre>`;
  html += `<h3>Strategic Behaviors</h3><pre>${esc(JSON.stringify(result.strategic_behaviors ?? {}, null, 2))}</pre>`;
  html += `</div>`;
  document.getElementById('tab-outcome').innerHTML = html;
}

function configRows(obj) {
  const keys = [
    'config_id', 'game_label', 'game_type', 'n_agents', 'num_agents', 'monoculture_model',
    'model_elo', 'competition_id', 'competition_level', 'rho', 'theta', 'sigma', 'alpha',
    'max_rounds', 'discussion_turns', 'gamma_discount', 'parallel_phases', 'random_seed',
    'model_order'
  ];
  const rows = keys.filter(k => obj[k] !== undefined).map(k => `<tr><th>${esc(k)}</th><td>${esc(JSON.stringify(obj[k]))}</td></tr>`);
  return `<table>${rows.join('')}</table>`;
}

function renderConfig() {
  const cfg = selectedPayload.config || {};
  let html = `<div class="section"><h2>Config</h2>${configRows(cfg)}`;
  html += `<h3>Models</h3><pre>${esc(JSON.stringify(cfg.models ?? cfg.agent_model_map ?? {}, null, 2))}</pre>`;
  html += `<h3>Full Config JSON</h3><pre>${esc(JSON.stringify(cfg, null, 2))}</pre></div>`;
  document.getElementById('tab-config').innerHTML = html;
}

function renderInteractionsPlaceholder() {
  const has = selectedPayload.has_interactions;
  const button = has ? `<button style="max-width:260px" onclick="loadInteractions()">Load full prompts and responses</button>` : '';
  document.getElementById('tab-interactions').innerHTML = `<div class="section"><h2>Interactions</h2>${button}<div id="interactionsBody" class="muted">${has ? '' : 'No all_interactions.json file found.'}</div></div>`;
}

async function loadInteractions() {
  const body = document.getElementById('interactionsBody');
  body.textContent = 'Loading interactions...';
  try {
    const payload = await getJson(`/api/interactions?config_id=${encodeURIComponent(selectedId)}`);
    const interactions = payload.interactions || [];
    let html = `<p class="muted small">${interactions.length} interactions</p>`;
    for (const item of interactions) {
      html += `<div class="message">
        <div class="message-head">${badge(item.phase || 'phase')} <b>${esc(item.agent_id || 'agent')}</b> <span class="muted">round ${fmt(item.round)} | ${esc(item.model_name || '')}</span></div>
        <div class="message-body"><b>Response</b>\n${esc(item.response || '')}</div>
        <details><summary>Prompt</summary><pre>${esc(item.prompt || '')}</pre></details>
      </div>`;
    }
    body.innerHTML = html;
  } catch (err) {
    body.innerHTML = `<span class="error">${esc(err.message)}</span>`;
  }
}

function renderRaw() {
  const raw = {
    row: selectedPayload.row,
    status: selectedPayload.status,
    result: selectedPayload.result,
  };
  document.getElementById('tab-raw').innerHTML = `<div class="section"><h2>Raw</h2><pre>${esc(JSON.stringify(raw, null, 2))}</pre></div>`;
}

function setTab(name) {
  currentTab = name;
  for (const node of document.querySelectorAll('.tab')) {
    node.classList.toggle('active', node.dataset.tab === name);
  }
  for (const node of document.querySelectorAll('.tabPanel')) {
    node.classList.add('hidden');
  }
  document.getElementById(`tab-${name}`).classList.remove('hidden');
}

reloadAll().catch(err => {
  document.getElementById('mainEmpty').innerHTML = `<span class="error">${esc(err.stack || err.message)}</span>`;
});
</script>
</body>
</html>
"""


class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "RandomMonocultureSampleViewer/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write("%s - - [%s] %s\n" % (self.client_address[0], self.log_date_time_string(), fmt % args))

    def send_bytes(self, body: bytes, content_type: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, payload: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
        self.send_bytes(body, "application/json; charset=utf-8", status=status)

    def send_error_json(self, status: HTTPStatus, message: str) -> None:
        self.send_json({"error": message}, status=status)

    def do_HEAD(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path in {"/", "/api/status", "/api/runs"}:
            self.send_response(HTTPStatus.OK)
            content_type = "text/html; charset=utf-8" if parsed.path == "/" else "application/json; charset=utf-8"
            self.send_header("Content-Type", content_type)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        self.send_response(HTTPStatus.NOT_FOUND)
        self.end_headers()

    def do_GET(self) -> None:
        try:
            parsed = urlparse(self.path)
            if parsed.path == "/":
                self.send_bytes(HTML.encode("utf-8"), "text/html; charset=utf-8")
                return
            if parsed.path == "/api/status":
                self.send_json({"ok": True, **INDEX.summary()})
                return
            if parsed.path == "/api/runs":
                self.send_json({"results_root": str(INDEX.results_root), "runs": INDEX.rows})
                return
            if parsed.path == "/api/reload":
                INDEX.reload()
                self.send_json({"ok": True, **INDEX.summary()})
                return
            if parsed.path == "/api/run":
                self.handle_run(parsed.query)
                return
            if parsed.path == "/api/interactions":
                self.handle_interactions(parsed.query)
                return
            self.send_error_json(HTTPStatus.NOT_FOUND, f"Unknown path: {parsed.path}")
        except Exception as exc:
            traceback.print_exc()
            self.send_error_json(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def handle_run(self, query: str) -> None:
        params = parse_qs(query)
        config_id = (params.get("config_id") or [""])[0]
        if not config_id:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "Missing config_id")
            return
        try:
            record = INDEX.get(config_id)
        except KeyError:
            self.send_error_json(HTTPStatus.NOT_FOUND, f"Unknown config_id: {config_id}")
            return

        config = read_json(record.config_path)
        status = read_json(record.status_path) if record.status_path else {}
        result = read_json(record.result_path) if record.result_path else {}
        error_tail = ""
        if record.row.get("state") == "FAILED":
            error_tail = tail_text(record.attempt_log_path or record.log_path)

        self.send_json(
            {
                "row": record.row,
                "config_path": str(record.config_path),
                "status_path": str(record.status_path) if record.status_path else None,
                "result_path": str(record.result_path) if record.result_path else None,
                "interactions_path": str(record.interactions_path) if record.interactions_path else None,
                "has_interactions": record.interactions_path is not None,
                "config": config,
                "status": status,
                "result": result,
                "error_tail": error_tail,
            }
        )

    def handle_interactions(self, query: str) -> None:
        params = parse_qs(query)
        config_id = (params.get("config_id") or [""])[0]
        if not config_id:
            self.send_error_json(HTTPStatus.BAD_REQUEST, "Missing config_id")
            return
        try:
            record = INDEX.get(config_id)
        except KeyError:
            self.send_error_json(HTTPStatus.NOT_FOUND, f"Unknown config_id: {config_id}")
            return
        if record.interactions_path is None:
            self.send_error_json(HTTPStatus.NOT_FOUND, "No interactions file")
            return
        self.send_json({"config_id": config_id, "interactions": read_json(record.interactions_path)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8002)
    return parser.parse_args()


def main() -> None:
    global INDEX
    args = parse_args()
    INDEX = BatchIndex(args.results_root)
    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    print(
        f"Serving {len(INDEX.rows)} configs from {INDEX.results_root} "
        f"on http://{args.host}:{args.port}",
        flush=True,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()
