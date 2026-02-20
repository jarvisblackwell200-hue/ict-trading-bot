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

from ict_bot.trading.config import PIP_SIZES
from ict_bot.trading.news_filter import NewsFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

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
    "news_blocked_pairs": [],
    "news_events": [],
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
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.7/dist/chart.umd.min.js"></script>
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

  /* ── Ticker tape ── */
  .ticker-wrap {
    background: var(--bg-tertiary);
    border-bottom: 1px solid var(--border);
    overflow: hidden;
    height: 28px;
    position: relative;
  }
  .ticker {
    display: flex;
    gap: 40px;
    animation: scroll-ticker 30s linear infinite;
    position: absolute;
    white-space: nowrap;
    padding: 5px 0;
  }
  .ticker-item {
    font-size: 0.72em;
    color: var(--text-secondary);
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .ticker-pair { color: var(--text-primary); font-weight: 600; }
  .ticker-price { font-variant-numeric: tabular-nums; }
  .ticker-pnl { font-weight: 600; }
  @keyframes scroll-ticker {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
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

  /* ── Charts grid ── */
  .charts-grid {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 16px;
    margin-bottom: 24px;
  }

  .chart-panel {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    position: relative;
  }
  .chart-panel-title {
    font-size: 0.7em;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--text-muted);
    margin-bottom: 12px;
  }
  .chart-canvas-wrap {
    position: relative;
    width: 100%;
  }

  /* ── Position gauges ── */
  .gauges-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
    gap: 10px;
    margin-bottom: 24px;
  }
  .gauge-card {
    background: var(--bg-secondary);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 14px 16px;
  }
  .gauge-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 10px;
  }
  .gauge-pair { font-weight: 700; font-size: 0.9em; }
  .gauge-pnl { font-weight: 600; font-size: 0.85em; }
  .gauge-track {
    position: relative;
    height: 8px;
    background: var(--bg-tertiary);
    border-radius: 4px;
    margin: 8px 0;
    overflow: visible;
  }
  .gauge-zone {
    position: absolute;
    height: 100%;
    border-radius: 4px;
  }
  .gauge-marker {
    position: absolute;
    top: -4px;
    width: 3px;
    height: 16px;
    border-radius: 2px;
    transform: translateX(-50%);
    transition: left 0.8s ease;
  }
  .gauge-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.62em;
    color: var(--text-muted);
    margin-top: 4px;
  }
  .gauge-labels .sl { color: var(--red); }
  .gauge-labels .tp { color: var(--green); }
  .gauge-labels .entry { color: var(--text-secondary); }

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

  /* ── News alert banner ── */
  .news-alert {
    background: var(--red-dim);
    border: 1px solid var(--red);
    border-radius: 8px;
    padding: 12px 20px;
    margin-bottom: 16px;
    display: none;
    align-items: center;
    gap: 12px;
    animation: pulse-red 2s infinite;
  }
  .news-alert.active { display: flex; }
  .news-alert-icon {
    font-size: 1.2em;
    color: var(--red);
    font-weight: 700;
  }
  .news-alert-text {
    font-size: 0.82em;
    color: var(--red);
    font-weight: 600;
  }
  .news-alert-pairs {
    font-size: 0.75em;
    color: var(--text-secondary);
    font-weight: 400;
  }
  @keyframes pulse-red {
    0%, 100% { border-color: var(--red); }
    50% { border-color: rgba(255,71,87,0.3); }
  }

  /* ── News event rows ── */
  tbody tr.news-active { background: rgba(255,71,87,0.08); }
  .impact-high { color: var(--red); font-weight: 700; }
  .impact-medium { color: var(--yellow); font-weight: 600; }
  .impact-low { color: var(--text-muted); }
  .impact-holiday { color: var(--accent); }

  /* ── Responsive ── */
  @media (max-width: 900px) {
    .charts-grid { grid-template-columns: 1fr; }
    .gauges-grid { grid-template-columns: 1fr; }
  }
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

<!-- Scrolling ticker tape -->
<div class="ticker-wrap">
  <div class="ticker" id="tickerTape"></div>
</div>

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

  <!-- News blackout alert banner -->
  <div class="news-alert" id="newsAlert">
    <span class="news-alert-icon">&#9888;</span>
    <span class="news-alert-text">TRADING PAUSED</span>
    <span class="news-alert-pairs" id="newsAlertPairs">News blackout active</span>
  </div>

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

  <!-- Charts row -->
  <div class="charts-grid">
    <div class="chart-panel">
      <div class="chart-panel-title">Unrealized P&amp;L Timeline</div>
      <div class="chart-canvas-wrap"><canvas id="pnlChart" height="180"></canvas></div>
    </div>
    <div class="chart-panel">
      <div class="chart-panel-title">P&amp;L by Position</div>
      <div class="chart-canvas-wrap"><canvas id="posBarChart" height="180"></canvas></div>
    </div>
  </div>

  <!-- Position gauges: SL ←── Entry ──→ TP -->
  <div class="gauges-grid" id="gaugesGrid"></div>

  <!-- Upcoming News Events -->
  <div class="section">
    <div class="section-head">
      <div class="section-title">Upcoming News Events</div>
      <div class="section-count" id="newsCount">0</div>
    </div>
    <div id="newsTable"></div>
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

  const MAX_HISTORY = 360; // ~1h at 10s intervals
  let lastData = null;
  let lastUpdateTs = 0;
  let prevPnlMap = {};
  let pnlHistory = [];      // {t, total, perPair:{pair: pnl}}
  let pnlLineChart = null;
  let posBarChart = null;

  // ── Chart.js global config ──
  Chart.defaults.color = '#8890a4';
  Chart.defaults.borderColor = '#252b3b';
  Chart.defaults.font.family = "'SF Mono','Fira Code',Consolas,monospace";
  Chart.defaults.font.size = 11;

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
    void el.offsetWidth;
    el.classList.add(direction > 0 ? 'flash-up' : 'flash-down');
  }

  function timeLabel() {
    const d = new Date();
    return d.getHours().toString().padStart(2,'0') + ':' +
           d.getMinutes().toString().padStart(2,'0') + ':' +
           d.getSeconds().toString().padStart(2,'0');
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

  // ── Ticker tape ──

  function updateTicker(positions) {
    const tape = document.getElementById('tickerTape');
    if (!positions || positions.length === 0) {
      tape.innerHTML = '<span class="ticker-item" style="color:var(--text-muted)">Waiting for data...</span>';
      return;
    }
    // Duplicate items for seamless scroll loop
    let items = '';
    for (let dup = 0; dup < 4; dup++) {
      for (const p of positions) {
        const pnl = p.unrealized_pnl || 0;
        const color = pnl >= 0 ? 'var(--green)' : 'var(--red)';
        const arrow = pnl >= 0 ? '\u25B2' : '\u25BC';
        items += `<span class="ticker-item">
          <span class="ticker-pair">${p.pair}</span>
          <span class="ticker-price">${fmtPrice(p.market_price)}</span>
          <span class="ticker-pnl" style="color:${color}">${arrow} ${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}</span>
        </span>`;
      }
    }
    tape.innerHTML = items;
  }

  // ── P&L Timeline Chart ──

  function initPnlChart() {
    const ctx = document.getElementById('pnlChart').getContext('2d');
    pnlLineChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Total Unrealized P&L',
          data: [],
          borderColor: '#4c8dff',
          backgroundColor: 'rgba(76,141,255,0.08)',
          borderWidth: 2,
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          pointHitRadius: 8,
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 600, easing: 'easeOutQuart' },
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1a1f2e',
            borderColor: '#252b3b',
            borderWidth: 1,
            titleColor: '#e1e4ea',
            bodyColor: '#8890a4',
            callbacks: {
              label: function(ctx) { return 'P&L: ' + (ctx.parsed.y >= 0 ? '+' : '') + ctx.parsed.y.toFixed(2); }
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(37,43,59,0.4)' },
            ticks: { maxTicksLimit: 8, font: { size: 10 } }
          },
          y: {
            grid: { color: 'rgba(37,43,59,0.4)' },
            ticks: {
              font: { size: 10 },
              callback: function(v) { return (v >= 0 ? '+' : '') + v.toFixed(0); }
            }
          }
        }
      }
    });
  }

  function updatePnlChart(totalPnl) {
    pnlHistory.push({ t: timeLabel(), v: totalPnl });
    if (pnlHistory.length > MAX_HISTORY) pnlHistory.shift();

    const chart = pnlLineChart;
    chart.data.labels = pnlHistory.map(h => h.t);
    const ds = chart.data.datasets[0];
    ds.data = pnlHistory.map(h => h.v);

    // Color the line and fill based on current P&L
    if (totalPnl >= 0) {
      ds.borderColor = '#00c48c';
      ds.backgroundColor = 'rgba(0,196,140,0.08)';
    } else {
      ds.borderColor = '#ff4757';
      ds.backgroundColor = 'rgba(255,71,87,0.08)';
    }
    chart.update('none'); // skip animation for smooth feel, data animates via tension
  }

  // ── Position P&L Bar Chart ──

  function initPosBarChart() {
    const ctx = document.getElementById('posBarChart').getContext('2d');
    posBarChart = new Chart(ctx, {
      type: 'bar',
      data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderRadius: 4, barThickness: 22 }] },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 800, easing: 'easeOutQuart' },
        plugins: {
          legend: { display: false },
          tooltip: {
            backgroundColor: '#1a1f2e',
            borderColor: '#252b3b',
            borderWidth: 1,
            callbacks: {
              label: function(ctx) { return 'P&L: ' + (ctx.parsed.x >= 0 ? '+' : '') + ctx.parsed.x.toFixed(2); }
            }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(37,43,59,0.4)' },
            ticks: {
              font: { size: 10 },
              callback: function(v) { return (v >= 0 ? '+' : '') + v; }
            }
          },
          y: {
            grid: { display: false },
            ticks: { font: { size: 11, weight: 'bold' } }
          }
        }
      }
    });
  }

  function updatePosBarChart(positions) {
    if (!positions || positions.length === 0) {
      posBarChart.data.labels = ['No positions'];
      posBarChart.data.datasets[0].data = [0];
      posBarChart.data.datasets[0].backgroundColor = ['#252b3b'];
      posBarChart.update();
      return;
    }
    const sorted = [...positions].sort((a, b) => (b.unrealized_pnl || 0) - (a.unrealized_pnl || 0));
    posBarChart.data.labels = sorted.map(p => p.pair);
    posBarChart.data.datasets[0].data = sorted.map(p => p.unrealized_pnl || 0);
    posBarChart.data.datasets[0].backgroundColor = sorted.map(p =>
      (p.unrealized_pnl || 0) >= 0 ? 'rgba(0,196,140,0.7)' : 'rgba(255,71,87,0.7)'
    );
    posBarChart.update();
  }

  // ── Position Gauges (SL ←── Price ──→ TP) ──

  function renderGauges(positions) {
    const container = document.getElementById('gaugesGrid');
    if (!positions || positions.length === 0) {
      container.innerHTML = '';
      return;
    }

    let html = '';
    for (const p of positions) {
      if (p.stop_loss == null || p.take_profit == null || p.market_price == null) continue;

      const sl = p.stop_loss;
      const tp = p.take_profit;
      const entry = p.entry_price;
      const price = p.market_price;
      const pnl = p.unrealized_pnl || 0;

      // For the gauge, figure out min/max range
      const lo = Math.min(sl, tp);
      const hi = Math.max(sl, tp);
      const range = hi - lo || 1;

      // Position as percentage along the SL-TP range
      const pricePct = Math.max(0, Math.min(100, ((price - lo) / range) * 100));
      const entryPct = Math.max(0, Math.min(100, ((entry - lo) / range) * 100));

      // Determine if SL is on left or right
      const slIsLeft = sl < tp;
      const slLabel = fmtPrice(sl);
      const tpLabel = fmtPrice(tp);

      // Color zone from entry to current price
      const zonePct1 = Math.min(entryPct, pricePct);
      const zonePct2 = Math.max(entryPct, pricePct);
      const zoneColor = pnl >= 0 ? 'rgba(0,196,140,0.25)' : 'rgba(255,71,87,0.25)';
      const markerColor = pnl >= 0 ? '#00c48c' : '#ff4757';

      html += `<div class="gauge-card">
        <div class="gauge-header">
          <span class="gauge-pair">${p.pair} ${tagHtml(p.direction)}</span>
          <span class="gauge-pnl ${pnlClass(pnl)}">${pnl >= 0 ? '+' : ''}${pnl.toFixed(2)}</span>
        </div>
        <div class="gauge-track">
          <div class="gauge-zone" style="left:${zonePct1}%;width:${zonePct2-zonePct1}%;background:${zoneColor}"></div>
          <div class="gauge-marker" style="left:${entryPct}%;background:var(--text-muted)" title="Entry ${fmtPrice(entry)}"></div>
          <div class="gauge-marker" style="left:${pricePct}%;background:${markerColor}" title="Current ${fmtPrice(price)}"></div>
        </div>
        <div class="gauge-labels">
          <span class="${slIsLeft ? 'sl' : 'tp'}">${slIsLeft ? 'SL ' + slLabel : 'TP ' + tpLabel}</span>
          <span class="entry">Entry ${fmtPrice(entry)}</span>
          <span class="${slIsLeft ? 'tp' : 'sl'}">${slIsLeft ? 'TP ' + tpLabel : 'SL ' + slLabel}</span>
        </div>
      </div>`;
    }
    container.innerHTML = html;
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
      const slDist = Math.abs(p.entry_price - p.stop_loss);
      totalAmount += Math.abs(p.units) * slDist;
    }
    return {amount: totalAmount, pips: totalPips};
  }

  // ── Render news events table ──

  function renderNewsEvents(events, blockedPairs) {
    const container = document.getElementById('newsTable');
    document.getElementById('newsCount').textContent = events ? events.length : 0;
    if (!events || events.length === 0) {
      container.innerHTML = '<div class="empty-state">No upcoming events</div>';
      return;
    }
    const blockedSet = new Set(blockedPairs || []);
    let html = `<table>
      <thead><tr>
        <th>Time (UTC)</th><th>Currency</th><th>Event</th><th>Impact</th>
        <th class="r">Forecast</th><th class="r">Previous</th><th>Affected Pairs</th>
      </tr></thead><tbody>`;

    for (const e of events) {
      const d = new Date(e.date);
      const timeStr = d.getUTCFullYear() + '-' +
        String(d.getUTCMonth()+1).padStart(2,'0') + '-' +
        String(d.getUTCDate()).padStart(2,'0') + ' ' +
        String(d.getUTCHours()).padStart(2,'0') + ':' +
        String(d.getUTCMinutes()).padStart(2,'0');
      const impactCls = e.impact === 'High' ? 'impact-high' :
                         e.impact === 'Medium' ? 'impact-medium' :
                         e.impact === 'Holiday' ? 'impact-holiday' : 'impact-low';
      const isActive = e.affected_pairs && e.affected_pairs.some(p => blockedSet.has(p));
      const rowCls = isActive ? ' class="news-active"' : '';
      const pairs = (e.affected_pairs || []).join(', ');
      html += `<tr${rowCls}>
        <td>${timeStr}</td>
        <td><strong>${e.country}</strong></td>
        <td>${e.title}</td>
        <td><span class="${impactCls}">${e.impact}</span></td>
        <td class="r">${e.forecast || '--'}</td>
        <td class="r">${e.previous || '--'}</td>
        <td style="font-size:0.75em">${pairs}</td>
      </tr>`;
    }
    html += '</tbody></table>';
    container.innerHTML = html;
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

      // Risk meter
      const nlv = parseFloat(acc.NetLiquidation || 0);
      if (nlv > 0) {
        const pct = Math.min(risk.amount / nlv * 100, 100);
        const fill = document.getElementById('riskMeterFill');
        fill.style.width = pct + '%';
        fill.style.background = pct > 10 ? 'var(--red)' : pct > 5 ? 'var(--yellow)' : 'var(--green)';
      }

      // Charts
      updatePnlChart(upnl);
      updatePosBarChart(data.positions);

      // Ticker
      updateTicker(data.positions);

      // Gauges
      renderGauges(data.positions);

      // News alert banner
      const blockedPairs = data.news_blocked_pairs || [];
      const newsAlert = document.getElementById('newsAlert');
      if (blockedPairs.length > 0) {
        newsAlert.classList.add('active');
        document.getElementById('newsAlertPairs').textContent =
          'News blackout: ' + blockedPairs.join(', ');
      } else {
        newsAlert.classList.remove('active');
      }

      // News events table
      renderNewsEvents(data.news_events || [], blockedPairs);

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

  // ── Init ──
  initPnlChart();
  initPosBarChart();
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


def run_ib_poller(host: str, port: int, client_id: int):
    """Background thread: connect to IB and poll account data."""
    global _state

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    ib = IB()
    news_filter = NewsFilter()

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

            # Collect open orders first so we can match SL/TP to positions
            open_orders = []
            # pair -> {"stop": price, "limit": price} for SL/TP matching
            order_map: dict[str, dict[str, float]] = {}
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
                # Track stop/limit orders per pair for SL/TP derivation
                if pair not in order_map:
                    order_map[pair] = {}
                if order.orderType == "STP" and price:
                    order_map[pair]["stop"] = price
                elif order.orderType == "LMT" and price:
                    order_map[pair]["limit"] = price

            # Positions from IB — derive SL/TP from open orders
            ib_positions = []
            portfolio = ib.portfolio()
            for item in portfolio:
                pair = symbol_to_pair(item.contract)
                direction = "long" if item.position > 0 else "short"

                # Match SL/TP from open orders for this pair
                pair_orders = order_map.get(pair, {})
                stop_price = pair_orders.get("stop")
                limit_price = pair_orders.get("limit")

                # For longs: SL is the stop (below entry), TP is the limit (above)
                # For shorts: SL is the stop (above entry), TP is the limit (below)
                stop_loss = stop_price
                take_profit = limit_price

                # Calculate risk pips from entry to SL
                risk_pips = None
                if stop_loss is not None:
                    pip_size = PIP_SIZES.get(pair, 0.0001)
                    risk_pips = abs(item.averageCost - stop_loss) / pip_size

                ib_positions.append(PosView(
                    pair=pair,
                    direction=direction,
                    units=item.position,
                    entry_price=item.averageCost,
                    market_price=item.marketPrice,
                    unrealized_pnl=item.unrealizedPNL,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    risk_pips=risk_pips,
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

            # Refresh news calendar
            news_status = news_filter.get_status_dict()

            with _lock:
                _state["account"] = account_info
                _state["positions"] = [p for p in ib_positions if p.units != 0]
                _state["orders"] = open_orders
                _state["trades"] = fills
                _state["connected"] = True
                _state["last_update"] = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
                _state["news_blocked_pairs"] = news_status["news_blocked_pairs"]
                _state["news_events"] = news_status["news_events"]

        except Exception as exc:
            logger.warning("Dashboard poller error: %s", exc, exc_info=True)
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
            "news_blocked_pairs": _state["news_blocked_pairs"],
            "news_events": _state["news_events"],
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
