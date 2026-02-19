#!/usr/bin/env python3
"""Live trading dashboard — web UI for monitoring IB paper/live account.

Shows open positions, P&L, trade history, and account summary.
Connects to IB Gateway read-only on a separate clientId.

Usage:
    PYTHONPATH=src python scripts/dashboard.py --port 4002
    Then open http://localhost:8080
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import ib_insync.util as ib_util

ib_util.patchAsyncio()

from flask import Flask, jsonify, render_template_string
from ib_insync import IB

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

STATE_FILE = Path(__file__).resolve().parents[1] / "data" / "live_state.json"

app = Flask(__name__)

# Shared state updated by background thread
_state = {
    "account": {},
    "positions": [],
    "orders": [],
    "trades": [],
    "bot_positions": {},
    "last_update": None,
    "connected": False,
}
_lock = threading.Lock()


# ── HTML Template ──────────────────────────────────────────────────

DASHBOARD_HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ICT Trading Bot</title>
<style>
  :root {
    --bg-primary: #0b0e14;
    --bg-secondary: #12161f;
    --bg-tertiary: #1a1f2e;
    --bg-hover: #1e2436;
    --border: #252b3b;
    --border-light: #2d3548;
    --text-primary: #e1e4ea;
    --text-secondary: #8890a4;
    --text-muted: #565e73;
    --accent: #4c8dff;
    --accent-dim: #2a4a8a;
    --green: #00c48c;
    --green-dim: rgba(0,196,140,0.12);
    --red: #ff4757;
    --red-dim: rgba(255,71,87,0.12);
    --yellow: #ffc107;
    --yellow-dim: rgba(255,193,7,0.12);
  }

  * { margin: 0; padding: 0; box-sizing: border-box; }

  body {
    font-family: 'SF Mono', 'Fira Code', 'JetBrains Mono', 'Cascadia Code', Consolas, monospace;
    background: var(--bg-primary);
    color: var(--text-primary);
    line-height: 1.5;
    min-height: 100vh;
  }

  /* ── Header ── */
  .header {
    background: var(--bg-secondary);
    border-bottom: 1px solid var(--border);
    padding: 14px 28px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
  }

  .header-left { display: flex; align-items: center; gap: 16px; }

  .logo {
    font-size: 1.1em;
    font-weight: 700;
    color: var(--text-primary);
    letter-spacing: -0.02em;
  }
  .logo span { color: var(--accent); }

  .conn-badge {
    display: flex;
    align-items: center;
    gap: 7px;
    font-size: 0.75em;
    color: var(--text-secondary);
    background: var(--bg-tertiary);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 4px 10px;
  }
  .conn-dot {
    width: 7px; height: 7px;
    border-radius: 50%;
    background: var(--red);
    transition: background 0.3s;
  }
  .conn-dot.live {
    background: var(--green);
    box-shadow: 0 0 6px rgba(0,196,140,0.5);
    animation: pulse 2s infinite;
  }
  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.5; }
  }

  .header-right {
    display: flex;
    align-items: center;
    gap: 16px;
    font-size: 0.75em;
    color: var(--text-muted);
  }

  .update-timer { font-variant-numeric: tabular-nums; }

  /* ── Main layout ── */
  .main { padding: 20px 28px; max-width: 1440px; margin: 0 auto; }

  /* ── Cards grid ── */
  .cards {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(190px, 1fr));
    gap: 10px;
    margin-bottom: 24px;
  }

  .card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
    transition: border-color 0.2s;
  }
  .card:hover { border-color: var(--border-light); }
  .card-label {
    font-size: 0.65em;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    margin-bottom: 6px;
  }
  .card-value {
    font-size: 1.35em;
    font-weight: 600;
    font-variant-numeric: tabular-nums;
    transition: color 0.3s;
  }
  .card-sub {
    font-size: 0.7em;
    color: var(--text-muted);
    margin-top: 2px;
  }

  .val-pos { color: var(--green); }
  .val-neg { color: var(--red); }
  .val-warn { color: var(--yellow); }
  .val-neutral { color: var(--text-primary); }

  /* ── Sections ── */
  .section {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 16px;
    overflow: hidden;
  }
  .section-head {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    border-bottom: 1px solid var(--border);
  }
  .section-title {
    font-size: 0.8em;
    font-weight: 600;
    color: var(--text-secondary);
    text-transform: uppercase;
    letter-spacing: 0.06em;
  }
  .section-count {
    font-size: 0.7em;
    color: var(--text-muted);
    background: var(--bg-tertiary);
    padding: 2px 8px;
    border-radius: 4px;
  }

  /* ── Tables ── */
  table { width: 100%; border-collapse: collapse; }
  thead th {
    font-size: 0.65em;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    padding: 8px 14px;
    text-align: left;
    border-bottom: 1px solid var(--border);
    background: var(--bg-tertiary);
    position: sticky;
    top: 0;
  }
  thead th.r { text-align: right; }
  tbody td {
    padding: 9px 14px;
    font-size: 0.82em;
    border-bottom: 1px solid rgba(37,43,59,0.5);
    font-variant-numeric: tabular-nums;
  }
  tbody td.r { text-align: right; }
  tbody td.mono { font-family: inherit; }
  tbody tr { transition: background 0.15s; }
  tbody tr:hover { background: var(--bg-hover); }
  tbody tr:last-child td { border-bottom: none; }

  .empty-state {
    padding: 28px 16px;
    text-align: center;
    color: var(--text-muted);
    font-size: 0.8em;
  }

  /* ── Tags ── */
  .tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.75em;
    font-weight: 700;
    letter-spacing: 0.04em;
  }
  .tag-long { background: rgba(76,141,255,0.15); color: var(--accent); }
  .tag-short { background: var(--red-dim); color: var(--red); }
  .tag-buy { background: rgba(76,141,255,0.15); color: var(--accent); }
  .tag-sell { background: var(--red-dim); color: var(--red); }

  /* ── P&L bar ── */
  .pnl-cell { display: flex; align-items: center; gap: 8px; justify-content: flex-end; }
  .pnl-bar {
    width: 40px; height: 4px;
    border-radius: 2px;
    background: var(--bg-tertiary);
    overflow: hidden;
  }
  .pnl-bar-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.5s ease;
  }

  /* ── Flash animation ── */
  @keyframes flash-green { 0% { background: var(--green-dim); } 100% { background: transparent; } }
  @keyframes flash-red { 0% { background: var(--red-dim); } 100% { background: transparent; } }
  .flash-up { animation: flash-green 0.8s ease-out; }
  .flash-down { animation: flash-red 0.8s ease-out; }

  /* ── Footer ── */
  .footer {
    text-align: center;
    padding: 20px;
    font-size: 0.65em;
    color: var(--text-muted);
  }

  /* ── Risk progress ── */
  .risk-meter {
    width: 100%;
    height: 3px;
    background: var(--bg-tertiary);
    border-radius: 2px;
    margin-top: 6px;
    overflow: hidden;
  }
  .risk-meter-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.5s;
  }

  /* ── Responsive ── */
  @media (max-width: 768px) {
    .header { padding: 12px 16px; }
    .main { padding: 12px 16px; }
    .cards { grid-template-columns: repeat(2, 1fr); gap: 8px; }
    .card { padding: 10px 12px; }
    .card-value { font-size: 1.1em; }
  }
</style>
</head>
<body>

<!-- Header -->
<div class="header">
  <div class="header-left">
    <div class="logo"><span>ICT</span> Trading Bot</div>
    <div class="conn-badge">
      <div class="conn-dot" id="connDot"></div>
      <span id="connText">Connecting...</span>
    </div>
  </div>
  <div class="header-right">
    <span id="accountId"></span>
    <span class="update-timer" id="updateTimer">--</span>
  </div>
</div>

<!-- Main content -->
<div class="main">

  <!-- Account summary cards -->
  <div class="cards" id="cardsGrid">
    <div class="card">
      <div class="card-label">Net Liquidation</div>
      <div class="card-value val-neutral" id="cardNLV">--</div>
    </div>
    <div class="card">
      <div class="card-label">Unrealized P&amp;L</div>
      <div class="card-value" id="cardUPnL">--</div>
    </div>
    <div class="card">
      <div class="card-label">Realized P&amp;L</div>
      <div class="card-value" id="cardRPnL">--</div>
    </div>
    <div class="card">
      <div class="card-label">Open Positions</div>
      <div class="card-value val-neutral" id="cardPosCount">0</div>
    </div>
    <div class="card">
      <div class="card-label">Available Funds</div>
      <div class="card-value val-neutral" id="cardFunds">--</div>
    </div>
    <div class="card">
      <div class="card-label">Margin Used</div>
      <div class="card-value val-neutral" id="cardMargin">--</div>
    </div>
    <div class="card">
      <div class="card-label">Max Risk (all SL hit)</div>
      <div class="card-value val-warn" id="cardMaxRisk">--</div>
      <div class="risk-meter"><div class="risk-meter-fill" id="riskMeterFill" style="width:0%;background:var(--yellow)"></div></div>
    </div>
    <div class="card">
      <div class="card-label">Total Risk Pips</div>
      <div class="card-value val-neutral" id="cardRiskPips">--</div>
    </div>
  </div>

  <!-- Open Positions -->
  <div class="section">
    <div class="section-head">
      <div class="section-title">Open Positions</div>
      <div class="section-count" id="posCount">0</div>
    </div>
    <div id="positionsTable"></div>
  </div>

  <!-- Open Orders -->
  <div class="section">
    <div class="section-head">
      <div class="section-title">Open Orders</div>
      <div class="section-count" id="ordersCount">0</div>
    </div>
    <div id="ordersTable"></div>
  </div>

  <!-- Recent Fills -->
  <div class="section">
    <div class="section-head">
      <div class="section-title">Recent Fills</div>
      <div class="section-count" id="fillsCount">0</div>
    </div>
    <div id="fillsTable"></div>
  </div>

</div>

<div class="footer">
  Live data from IB Gateway &middot; Updates every 10s
</div>

<script>
(function() {
  "use strict";

  let lastData = null;
  let lastUpdateTs = 0;
  let prevPnlMap = {};

  // ── Helpers ──

  function fmt(v, dec) {
    if (v == null || v === '' || v === 'N/A') return '--';
    const n = parseFloat(v);
    if (isNaN(n)) return v;
    return n.toLocaleString('en-US', {minimumFractionDigits: dec || 0, maximumFractionDigits: dec || 0});
  }

  function fmtPrice(v) {
    if (v == null) return '--';
    const n = parseFloat(v);
    if (isNaN(n)) return '--';
    return n.toFixed(5);
  }

  function pnlClass(v) {
    const n = parseFloat(v);
    if (isNaN(n) || n === 0) return 'val-neutral';
    return n >= 0 ? 'val-pos' : 'val-neg';
  }

  function tagHtml(dir) {
    const cls = (dir === 'long' || dir === 'BUY') ? 'tag-long' : 'tag-short';
    const label = (dir === 'long' || dir === 'BUY')
      ? (dir === 'BUY' ? 'BUY' : 'LONG')
      : (dir === 'SELL' ? 'SELL' : 'SHORT');
    return `<span class="tag ${cls}">${label}</span>`;
  }

  function flashCell(el, direction) {
    el.classList.remove('flash-up', 'flash-down');
    void el.offsetWidth; // trigger reflow
    el.classList.add(direction > 0 ? 'flash-up' : 'flash-down');
  }

  // ── Card updaters ──

  function updateCard(id, value, cssClass) {
    const el = document.getElementById(id);
    if (!el) return;
    const prev = el.textContent;
    el.textContent = value;
    if (cssClass !== undefined) el.className = 'card-value ' + cssClass;
    if (prev !== '--' && prev !== value) {
      const dir = parseFloat(value.replace(/,/g,'')) >= parseFloat(prev.replace(/,/g,'')) ? 1 : -1;
      flashCell(el, dir);
    }
  }

  // ── Render positions table ──

  function renderPositions(positions) {
    const container = document.getElementById('positionsTable');
    if (!positions || positions.length === 0) {
      container.innerHTML = '<div class="empty-state">No open positions</div>';
      document.getElementById('posCount').textContent = '0';
      document.getElementById('cardPosCount').textContent = '0';
      return;
    }
    document.getElementById('posCount').textContent = positions.length;
    document.getElementById('cardPosCount').textContent = positions.length;

    // Find max abs PnL for bar scaling
    const maxPnl = Math.max(...positions.map(p => Math.abs(p.unrealized_pnl || 0)), 1);

    let html = `<table>
      <thead><tr>
        <th>Pair</th><th>Side</th><th class="r">Units</th>
        <th class="r">Entry</th><th class="r">Market</th>
        <th class="r">P&amp;L</th>
        <th class="r">Stop Loss</th><th class="r">Take Profit</th>
        <th class="r">Risk</th>
      </tr></thead><tbody>`;

    for (const p of positions) {
      const pnl = p.unrealized_pnl || 0;
      const pnlPct = Math.min(Math.abs(pnl) / maxPnl * 100, 100);
      const barColor = pnl >= 0 ? 'var(--green)' : 'var(--red)';
      const pnlKey = p.pair;
      const prevPnl = prevPnlMap[pnlKey];
      const flashDir = (prevPnl !== undefined && pnl !== prevPnl) ? (pnl > prevPnl ? 1 : -1) : 0;
      prevPnlMap[pnlKey] = pnl;

      html += `<tr${flashDir ? ` class="${flashDir > 0 ? 'flash-up' : 'flash-down'}"` : ''}>
        <td><strong>${p.pair}</strong></td>
        <td>${tagHtml(p.direction)}</td>
        <td class="r">${fmt(Math.abs(p.units), 0)}</td>
        <td class="r">${fmtPrice(p.entry_price)}</td>
        <td class="r">${fmtPrice(p.market_price)}</td>
        <td class="r"><div class="pnl-cell">
          <span class="${pnlClass(pnl)}">${pnl >= 0 ? '+' : ''}${fmt(pnl, 2)}</span>
          <div class="pnl-bar"><div class="pnl-bar-fill" style="width:${pnlPct}%;background:${barColor}"></div></div>
        </div></td>
        <td class="r">${fmtPrice(p.stop_loss)}</td>
        <td class="r">${fmtPrice(p.take_profit)}</td>
        <td class="r">${p.risk_pips != null ? fmt(p.risk_pips, 1) + 'p' : '--'}</td>
      </tr>`;
    }
    html += '</tbody></table>';
    container.innerHTML = html;
  }

  // ── Render orders table ──

  function renderOrders(orders) {
    const container = document.getElementById('ordersTable');
    document.getElementById('ordersCount').textContent = orders ? orders.length : 0;
    if (!orders || orders.length === 0) {
      container.innerHTML = '<div class="empty-state">No open orders</div>';
      return;
    }
    let html = `<table>
      <thead><tr>
        <th>Pair</th><th>Type</th><th>Action</th>
        <th class="r">Units</th><th class="r">Price</th><th>Status</th>
      </tr></thead><tbody>`;

    for (const o of orders) {
      html += `<tr>
        <td><strong>${o.pair}</strong></td>
        <td>${o.order_type}</td>
        <td>${tagHtml(o.action)}</td>
        <td class="r">${fmt(o.units, 0)}</td>
        <td class="r">${fmtPrice(o.price)}</td>
        <td>${o.status}</td>
      </tr>`;
    }
    html += '</tbody></table>';
    container.innerHTML = html;
  }

  // ── Render fills table ──

  function renderFills(fills) {
    const container = document.getElementById('fillsTable');
    document.getElementById('fillsCount').textContent = fills ? fills.length : 0;
    if (!fills || fills.length === 0) {
      container.innerHTML = '<div class="empty-state">No fills today</div>';
      return;
    }
    let html = `<table>
      <thead><tr>
        <th>Time</th><th>Pair</th><th>Action</th>
        <th class="r">Units</th><th class="r">Price</th>
        <th class="r">Realized P&amp;L</th><th class="r">Commission</th>
      </tr></thead><tbody>`;

    for (const f of fills) {
      const rpnl = f.realized_pnl || 0;
      html += `<tr>
        <td>${f.time || '--'}</td>
        <td><strong>${f.pair}</strong></td>
        <td>${tagHtml(f.action)}</td>
        <td class="r">${fmt(f.units, 0)}</td>
        <td class="r">${fmtPrice(f.price)}</td>
        <td class="r ${pnlClass(rpnl)}">${rpnl !== 0 ? (rpnl >= 0 ? '+' : '') + fmt(rpnl, 2) : '--'}</td>
        <td class="r">${f.commission ? fmt(f.commission, 2) : '--'}</td>
      </tr>`;
    }
    html += '</tbody></table>';
    container.innerHTML = html;
  }

  // ── Compute max risk ──

  function computeMaxRisk(positions) {
    if (!positions || positions.length === 0) return {amount: 0, pips: 0};
    let totalPips = 0;
    let totalAmount = 0;
    for (const p of positions) {
      if (p.stop_loss == null || p.entry_price == null) continue;
      const riskPips = p.risk_pips || 0;
      totalPips += riskPips;
      // Approximate USD loss: units * |entry - SL|
      const slDist = Math.abs(p.entry_price - p.stop_loss);
      totalAmount += Math.abs(p.units) * slDist;
    }
    return {amount: totalAmount, pips: totalPips};
  }

  // ── Timer ──

  function updateTimerDisplay() {
    const el = document.getElementById('updateTimer');
    if (!lastUpdateTs) { el.textContent = '--'; return; }
    const ago = Math.floor((Date.now() - lastUpdateTs) / 1000);
    el.textContent = ago < 2 ? 'just now' : ago + 's ago';
  }
  setInterval(updateTimerDisplay, 1000);

  // ── Main fetch loop ──

  async function fetchAndUpdate() {
    try {
      const resp = await fetch('/api/status');
      if (!resp.ok) throw new Error('HTTP ' + resp.status);
      const data = await resp.json();
      lastData = data;
      lastUpdateTs = Date.now();

      // Connection
      const dot = document.getElementById('connDot');
      const txt = document.getElementById('connText');
      if (data.connected) {
        dot.classList.add('live');
        txt.textContent = 'Live';
      } else {
        dot.classList.remove('live');
        txt.textContent = 'Disconnected';
      }

      // Account
      const acc = data.account || {};
      document.getElementById('accountId').textContent = acc.Account || '';

      updateCard('cardNLV', fmt(acc.NetLiquidation, 2), 'val-neutral');

      const upnl = parseFloat(acc.UnrealizedPnL_raw || 0);
      updateCard('cardUPnL', (upnl >= 0 ? '+' : '') + fmt(upnl, 2), pnlClass(upnl));

      const rpnl = parseFloat(acc.RealizedPnL_raw || 0);
      updateCard('cardRPnL', (rpnl >= 0 ? '+' : '') + fmt(rpnl, 2), pnlClass(rpnl));

      updateCard('cardFunds', fmt(acc.AvailableFunds, 2), 'val-neutral');
      updateCard('cardMargin', fmt(acc.InitMarginReq, 2), 'val-neutral');

      // Max risk
      const risk = computeMaxRisk(data.positions);
      updateCard('cardMaxRisk', '-' + fmt(risk.amount, 2), 'card-value val-warn');
      updateCard('cardRiskPips', fmt(risk.pips, 1) + 'p', 'val-neutral');

      // Risk meter: scale to NLV
      const nlv = parseFloat(acc.NetLiquidation || 0);
      if (nlv > 0) {
        const pct = Math.min(risk.amount / nlv * 100, 100);
        const fill = document.getElementById('riskMeterFill');
        fill.style.width = pct + '%';
        fill.style.background = pct > 10 ? 'var(--red)' : pct > 5 ? 'var(--yellow)' : 'var(--green)';
      }

      // Tables
      renderPositions(data.positions);
      renderOrders(data.orders);
      renderFills(data.fills);

    } catch (err) {
      const dot = document.getElementById('connDot');
      const txt = document.getElementById('connText');
      dot.classList.remove('live');
      txt.textContent = 'Error';
    }
  }

  // Initial fetch, then poll every 10s
  fetchAndUpdate();
  setInterval(fetchAndUpdate, 10000);

})();
</script>
</body>
</html>
"""


# ── Data classes for template ──────────────────────────────────────

class PosView:
    def __init__(self, pair, direction, units, entry_price, market_price,
                 unrealized_pnl, stop_loss, take_profit, risk_pips):
        self.pair = pair
        self.direction = direction
        self.units = units
        self.entry_price = entry_price
        self.market_price = market_price
        self.unrealized_pnl = unrealized_pnl
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.risk_pips = risk_pips


class OrderView:
    def __init__(self, pair, order_type, action, units, price, status):
        self.pair = pair
        self.order_type = order_type
        self.action = action
        self.units = units
        self.price = price
        self.status = status


class FillView:
    def __init__(self, time, pair, action, units, price, realized_pnl, commission):
        self.time = time
        self.pair = pair
        self.action = action
        self.units = units
        self.price = price
        self.realized_pnl = realized_pnl
        self.commission = commission


# ── IB Data Fetcher (background thread) ───────────────────────────

PAIR_MAP = {
    "EURUSD": "EUR_USD", "GBPUSD": "GBP_USD", "USDJPY": "USD_JPY",
    "AUDUSD": "AUD_USD", "USDCAD": "USD_CAD", "NZDUSD": "NZD_USD",
    "EURGBP": "EUR_GBP",
}


def symbol_to_pair(contract) -> str:
    """Convert IB contract to our pair name."""
    key = f"{contract.symbol}_{contract.currency}"
    if key in PAIR_MAP.values():
        return key
    combined = f"{contract.symbol}{contract.currency}"
    return PAIR_MAP.get(combined, f"{contract.symbol}/{contract.currency}")


def load_bot_state() -> dict:
    """Load the bot's position state file."""
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    except Exception:
        pass
    return {}


def run_ib_poller(host: str, port: int, client_id: int):
    """Background thread: connect to IB and poll account data."""
    global _state

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    ib = IB()

    while True:
        try:
            if not ib.isConnected():
                ib.connect(host, port, clientId=client_id, readonly=True)
                logger.info("Dashboard connected to IB Gateway (clientId=%d)", client_id)

            # Account summary
            account_info = {}
            for av in ib.accountValues():
                if av.currency in ("BASE", "") or av.tag in (
                    "NetLiquidation", "UnrealizedPnL", "RealizedPnL",
                    "AvailableFunds", "InitMarginReq", "MaintMarginReq",
                    "TotalCashValue", "GrossPositionValue",
                ):
                    if av.tag not in account_info or av.currency == "BASE":
                        account_info[av.tag] = av.value

            # Get account ID
            accounts = ib.managedAccounts()
            if accounts:
                account_info["Account"] = accounts[0]

            # Store raw numeric values for API
            try:
                account_info["UnrealizedPnL_raw"] = float(account_info.get("UnrealizedPnL", 0))
            except (ValueError, TypeError):
                account_info["UnrealizedPnL_raw"] = 0
            try:
                account_info["RealizedPnL_raw"] = float(account_info.get("RealizedPnL", 0))
            except (ValueError, TypeError):
                account_info["RealizedPnL_raw"] = 0

            # Positions from IB
            ib_positions = []
            bot_state = load_bot_state()

            portfolio = ib.portfolio()
            for item in portfolio:
                pair = symbol_to_pair(item.contract)
                bot_pos = bot_state.get(pair, {})

                direction = "long" if item.position > 0 else "short"
                ib_positions.append(PosView(
                    pair=pair,
                    direction=bot_pos.get("direction", direction),
                    units=item.position,
                    entry_price=item.averageCost,
                    market_price=item.marketPrice,
                    unrealized_pnl=item.unrealizedPNL,
                    stop_loss=bot_pos.get("stop_loss"),
                    take_profit=bot_pos.get("take_profit"),
                    risk_pips=bot_pos.get("risk_pips"),
                ))

            # Open orders
            open_orders = []
            for trade in ib.openTrades():
                pair = symbol_to_pair(trade.contract)
                order = trade.order
                price = order.auxPrice if order.auxPrice else order.lmtPrice
                open_orders.append(OrderView(
                    pair=pair,
                    order_type=order.orderType,
                    action=order.action,
                    units=order.totalQuantity,
                    price=price,
                    status=trade.orderStatus.status,
                ))

            # Recent fills
            fills = []
            for fill in ib.fills():
                pair = symbol_to_pair(fill.contract)
                exec_ = fill.execution
                comm = fill.commissionReport
                fills.append(FillView(
                    time=exec_.time.strftime("%H:%M:%S") if exec_.time else "",
                    pair=pair,
                    action=exec_.side.replace("SLD", "SELL").replace("BOT", "BUY"),
                    units=exec_.shares,
                    price=exec_.price,
                    realized_pnl=comm.realizedPNL if comm.realizedPNL else 0,
                    commission=comm.commission if comm.commission else 0,
                ))

            # Sort fills by time (most recent first)
            fills.sort(key=lambda f: f.time, reverse=True)

            with _lock:
                _state["account"] = account_info
                _state["positions"] = [p for p in ib_positions if p.units != 0]
                _state["orders"] = open_orders
                _state["trades"] = fills
                _state["connected"] = True
                _state["last_update"] = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")

        except Exception as exc:
            logger.warning("Dashboard poller error: %s", exc)
            with _lock:
                _state["connected"] = False
            try:
                ib.disconnect()
            except Exception:
                pass

        time.sleep(10)


# ── Flask Routes ──────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/status")
def api_status():
    """JSON API endpoint — full state for live JS updates."""
    with _lock:
        return jsonify({
            "connected": _state["connected"],
            "last_update": _state["last_update"],
            "account": _state["account"],
            "positions": [
                {
                    "pair": p.pair,
                    "direction": p.direction,
                    "units": p.units,
                    "entry_price": p.entry_price,
                    "market_price": p.market_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "risk_pips": p.risk_pips,
                }
                for p in _state["positions"]
            ],
            "orders": [
                {
                    "pair": o.pair,
                    "order_type": o.order_type,
                    "action": o.action,
                    "units": o.units,
                    "price": o.price,
                    "status": o.status,
                }
                for o in _state["orders"]
            ],
            "fills": [
                {
                    "time": t.time,
                    "pair": t.pair,
                    "action": t.action,
                    "units": t.units,
                    "price": t.price,
                    "realized_pnl": t.realized_pnl,
                    "commission": t.commission,
                }
                for t in _state["trades"]
            ],
        })


def main():
    parser = argparse.ArgumentParser(description="ICT Trading Bot Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="IB Gateway host")
    parser.add_argument("--port", type=int, default=4002, help="IB Gateway port")
    parser.add_argument("--client-id", type=int, default=99, help="IB client ID (separate from bot)")
    parser.add_argument("--web-port", type=int, default=8080, help="Dashboard web port")
    args = parser.parse_args()

    # Start IB poller in background thread
    poller = threading.Thread(
        target=run_ib_poller,
        args=(args.host, args.port, args.client_id),
        daemon=True,
    )
    poller.start()
    logger.info("Starting dashboard at http://localhost:%d", args.web_port)

    # Start Flask
    app.run(host="0.0.0.0", port=args.web_port, debug=False)


if __name__ == "__main__":
    main()
