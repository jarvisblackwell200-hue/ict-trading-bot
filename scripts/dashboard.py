#!/usr/bin/env python3
"""Live trading dashboard — professional terminal UI for monitoring IB paper/live account.

Shows live market data for all 7 pairs, positions, P&L, risk gauges, trade history,
news calendar, sparkline charts, and equity curve.

Connects to IB Gateway read-only on a separate clientId.

Usage:
    PYTHONPATH=src python scripts/dashboard.py --port 4002 --client-id 99 --web-port 8080
    Then open http://localhost:8080
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path

import ib_insync.util as ib_util

ib_util.patchAsyncio()

from flask import Flask, jsonify, render_template_string
from ib_insync import IB, CFD

from ict_bot.trading.config import PAIR_TO_IB, PAIR_TO_CFD, PIP_SIZES
from ict_bot.trading.news_filter import NewsFilter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

ALL_PAIRS = list(PAIR_TO_IB.keys())
RISK_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "risk_state.json"
LIVE_STATE_PATH = Path(__file__).resolve().parent.parent / "data" / "live_state.json"
SPARKLINE_INTERVAL = 900  # 15 minutes
PARQUET_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Shared state updated by background thread
_state = {
    "account": {},
    "positions": [],
    "orders": [],
    "trades": [],
    "connected": False,
    "last_update": None,
    "news_blocked_pairs": [],
    "news_events": [],
    "market_data": {},
    "sparklines": {},
    "risk": {},
    "equity_history": [],
    "latency_ms": 0,
}
_lock = threading.Lock()
_equity_history: deque = deque(maxlen=720)
_sparkline_cache: dict = {}
_sparkline_last_ts: float = 0


# ── Helpers ───────────────────────────────────────────────────────

def _safe(v):
    """Convert IB ticker value to JSON-safe (NaN/None → None)."""
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or v <= 0):
        return None
    return v


def _read_risk_state() -> dict:
    """Read risk manager state from JSON file."""
    try:
        if RISK_STATE_PATH.exists():
            data = json.loads(RISK_STATE_PATH.read_text())
            balance = data.get("balance", 0)
            peak = data.get("peak_balance", 0)
            dd_pct = ((peak - balance) / peak * 100) if peak > 0 else 0
            return {
                "balance": balance,
                "peak_balance": peak,
                "daily_pnl": data.get("daily_pnl", 0),
                "consecutive_losses": data.get("consecutive_losses", 0),
                "killed": data.get("killed", False),
                "circuit_broken": data.get("circuit_broken", False),
                "drawdown_pct": round(dd_pct, 2),
                "exposure_pct": 0,
            }
    except Exception:
        pass
    return {}


def _read_live_state() -> dict:
    """Read bot position metadata from JSON file."""
    try:
        if LIVE_STATE_PATH.exists():
            return json.loads(LIVE_STATE_PATH.read_text())
    except Exception:
        pass
    return {}


# ── Data view classes ─────────────────────────────────────────────

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
    def __init__(self, pair, order_type, action, units, price, status, time=None):
        self.pair = pair
        self.order_type = order_type
        self.action = action
        self.units = units
        self.price = price
        self.status = status
        self.time = time


class FillView:
    def __init__(self, time, pair, action, units, price, realized_pnl, commission):
        self.time = time
        self.pair = pair
        self.action = action
        self.units = units
        self.price = price
        self.realized_pnl = realized_pnl
        self.commission = commission


# ── IB Data Fetcher (background thread) ──────────────────────────

PAIR_MAP = {
    "EURUSD": "EUR_USD", "GBPUSD": "GBP_USD", "USDJPY": "USD_JPY",
    "AUDUSD": "AUD_USD", "USDCAD": "USD_CAD", "NZDUSD": "NZD_USD",
    "EURGBP": "EUR_GBP",
}


def symbol_to_pair(contract) -> str:
    key = f"{contract.symbol}_{contract.currency}"
    if key in PAIR_MAP.values():
        return key
    combined = f"{contract.symbol}{contract.currency}"
    return PAIR_MAP.get(combined, f"{contract.symbol}/{contract.currency}")


def _fetch_sparklines() -> dict:
    """Read last 50 M15 bars per pair from parquet files. Cached for SPARKLINE_INTERVAL seconds."""
    global _sparkline_cache, _sparkline_last_ts
    now = time.time()
    if now - _sparkline_last_ts < SPARKLINE_INTERVAL and _sparkline_cache:
        return _sparkline_cache

    try:
        import pandas as pd
    except ImportError:
        return _sparkline_cache

    sparklines = {}
    for pair in ALL_PAIRS:
        path = PARQUET_DIR / f"{pair}_M15.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path, columns=["close"])
            tail = df.tail(50)
            sparklines[pair] = [
                {"t": str(idx), "c": float(row["close"])} for idx, row in tail.iterrows()
            ]
        except Exception as e:
            logger.warning("Sparkline read failed for %s: %s", pair, e)

    if sparklines:
        _sparkline_cache = sparklines
        _sparkline_last_ts = now
    return _sparkline_cache


def _get_daily_opens(sparklines: dict) -> dict[str, float]:
    """Derive today's opening price per pair from sparkline data.

    Uses the first M15 bar of the current UTC day as the daily open.
    Falls back to the earliest bar in the sparkline if no today match.
    """
    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    opens = {}
    for pair, bars in sparklines.items():
        # Find first bar of today
        for bar in bars:
            if bar["t"].startswith(today_str):
                opens[pair] = bar["c"]
                break
        # Fallback: use first bar in sparkline as approximate "previous close"
        if pair not in opens and bars:
            opens[pair] = bars[0]["c"]
    return opens


def run_ib_poller(host: str, port: int, client_id: int, account: str = ""):
    """Background thread: connect to IB and poll account data."""
    global _state

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    ib = IB()
    news_filter = NewsFilter()
    subscribed = False
    tickers: dict = {}        # pair -> Ticker
    contracts: dict = {}       # pair -> CFD Contract (for live data)

    while True:
        try:
            if not ib.isConnected():
                ib.connect(host, port, clientId=client_id, readonly=True,
                           account=account or "")
                logger.info("Dashboard connected to IB Gateway (clientId=%d, account=%s)",
                            client_id, account or "auto")
                subscribed = False

            # Subscribe to market data for all pairs (once after connect)
            if not subscribed:
                for pair, (sym, cur) in PAIR_TO_CFD.items():
                    try:
                        # CFD for live market data
                        contract = CFD(symbol=sym, currency=cur)
                        ib.qualifyContracts(contract)
                        ticker = ib.reqMktData(contract, '', False, False)
                        contracts[pair] = contract
                        tickers[pair] = ticker
                    except Exception as e:
                        logger.warning("Failed to subscribe %s: %s", pair, e)
                subscribed = True
                logger.info("Subscribed to market data for %d pairs", len(tickers))

            t0 = time.time()

            # Allow IB to process pending events
            ib.sleep(0.5)

            # ── Account summary ──
            account_info = {}
            base_currency = "USD"
            usd_rate = 1.0  # base_currency per 1 USD

            ACCT_TAGS = {
                "NetLiquidation", "UnrealizedPnL", "RealizedPnL",
                "AvailableFunds", "InitMarginReq", "MaintMarginReq",
                "TotalCashValue", "GrossPositionValue", "BuyingPower",
            }
            for av in ib.accountValues():
                if av.currency in ("BASE", "") or av.tag in ACCT_TAGS:
                    # Prefer non-BASE, non-zero values over BASE (ISK accounts
                    # report BASE as 0 while the real value is in local currency)
                    if av.tag not in account_info:
                        account_info[av.tag] = av.value
                    elif av.currency == "BASE":
                        # Only overwrite with BASE if current value is 0
                        try:
                            if float(account_info[av.tag]) == 0 and float(av.value) != 0:
                                account_info[av.tag] = av.value
                        except (ValueError, TypeError):
                            pass
                    elif av.currency not in ("BASE", ""):
                        # Non-BASE currency value — prefer if current is 0
                        try:
                            if float(account_info[av.tag]) == 0 and float(av.value) != 0:
                                account_info[av.tag] = av.value
                        except (ValueError, TypeError):
                            pass
                # Detect base currency and USD exchange rate
                if av.tag == "NetLiquidation" and av.currency not in ("BASE", "USD", ""):
                    base_currency = av.currency
                    # Use this non-zero NLV as the primary value
                    try:
                        if float(av.value) > 0:
                            account_info[av.tag] = av.value
                    except (ValueError, TypeError):
                        pass
                if av.tag == "ExchangeRate" and av.currency == "USD":
                    try:
                        usd_rate = float(av.value)
                    except (ValueError, TypeError):
                        pass

            if account:
                account_info["Account"] = account
            else:
                accounts = ib.managedAccounts()
                if accounts:
                    account_info["Account"] = accounts[0]
            account_info["base_currency"] = base_currency
            account_info["usd_rate"] = usd_rate

            try:
                account_info["UnrealizedPnL_raw"] = float(account_info.get("UnrealizedPnL", 0))
            except (ValueError, TypeError):
                account_info["UnrealizedPnL_raw"] = 0
            try:
                account_info["RealizedPnL_raw"] = float(account_info.get("RealizedPnL", 0))
            except (ValueError, TypeError):
                account_info["RealizedPnL_raw"] = 0

            # ── Sparklines (refreshed every 15 min from parquet) ──
            sparklines = _fetch_sparklines()

            # ── Market data from tickers + portfolio + parquet fallback ──
            market_data = {}
            portfolio_prices: dict[str, float] = {}
            for item in ib.portfolio():
                pair = symbol_to_pair(item.contract)
                if item.position != 0 and item.marketPrice:
                    portfolio_prices[pair] = item.marketPrice

            daily_opens = _get_daily_opens(sparklines)

            for pair in ALL_PAIRS:
                ticker = tickers.get(pair)
                bid = _safe(ticker.bid) if ticker else None
                ask = _safe(ticker.ask) if ticker else None
                last = _safe(ticker.last) if ticker else None
                close = _safe(ticker.close) if ticker else None
                high = _safe(ticker.high) if ticker else None
                low = _safe(ticker.low) if ticker else None

                # Fallback to portfolio marketPrice if ticker has no data
                if not last and not bid and pair in portfolio_prices:
                    last = portfolio_prices[pair]

                # Fallback to last sparkline close from parquet
                if not last and not bid:
                    sp = sparklines.get(pair, [])
                    if sp:
                        last = sp[-1]["c"]

                # Fallback: use daily open from sparkline as "previous close"
                # so ticker tape can compute % change without market data sub
                if not close and pair in daily_opens:
                    close = daily_opens[pair]

                market_data[pair] = {
                    "bid": bid, "ask": ask, "last": last,
                    "close": close, "high": high, "low": low,
                }

            # ── Open orders and SL/TP mapping (from ALL client IDs) ──
            open_orders = []
            order_map: dict[str, dict[str, float]] = {}

            # reqAllOpenOrders() fetches orders from ALL API clients
            # (works with clientId=0/master; graceful fallback otherwise)
            all_order_trades = []
            try:
                all_order_trades = ib.reqAllOpenOrders()
                ib.sleep(1)  # wait for IB to deliver order data
            except Exception:
                pass
            # Also include own client's trades as fallback
            seen_ids = {t.order.orderId for t in all_order_trades if t.order}
            for t in ib.openTrades():
                if t.order and t.order.orderId not in seen_ids:
                    all_order_trades.append(t)

            for trade in all_order_trades:
                pair = symbol_to_pair(trade.contract)
                order = trade.order
                status = trade.orderStatus.status
                price = order.auxPrice if order.auxPrice else order.lmtPrice
                # Get order time from log or fall back to current time
                order_time = None
                if trade.log and len(trade.log) > 0:
                    order_time = trade.log[0].time.strftime("%Y-%m-%d %H:%M:%S") if trade.log[0].time else None
                if not order_time:
                    order_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                open_orders.append(OrderView(
                    pair=pair, order_type=order.orderType,
                    action=order.action, units=order.totalQuantity,
                    price=price, status=status,
                    time=order_time,
                ))
                # Only use ACTIVE orders for SL/TP mapping (ignore Filled/Cancelled)
                if status not in ("PreSubmitted", "Submitted"):
                    continue
                if pair not in order_map:
                    order_map[pair] = {}
                if order.orderType == "STP" and price:
                    order_map[pair]["stop"] = price
                elif order.orderType == "LMT" and price:
                    order_map[pair]["limit"] = price

            # ── Positions (merge IB portfolio + bot state + live tickers) ──
            ib_positions = []
            bot_state = _read_live_state()
            for item in ib.portfolio():
                pair = symbol_to_pair(item.contract)
                direction = "long" if item.position > 0 else "short"
                pair_orders = order_map.get(pair, {})
                stop_loss = pair_orders.get("stop")
                take_profit = pair_orders.get("limit")

                # Fall back to bot's live_state.json for SL/TP + metadata
                bot_pos = bot_state.get(pair, {})
                if stop_loss is None and bot_pos.get("stop_loss"):
                    stop_loss = bot_pos["stop_loss"]
                if take_profit is None and bot_pos.get("take_profit"):
                    take_profit = bot_pos["take_profit"]

                # Entry price: prefer bot state (actual fill), fall back to IB averageCost
                entry_price = bot_pos.get("entry_price") or item.averageCost

                risk_pips = bot_pos.get("risk_pips")
                if risk_pips is None and stop_loss is not None:
                    pip_size = PIP_SIZES.get(pair, 0.0001)
                    risk_pips = abs(entry_price - stop_loss) / pip_size

                # Market price: prefer live ticker mid, fall back to portfolio
                market_price = item.marketPrice
                if pair in tickers:
                    t = tickers[pair]
                    if _safe(t.bid) and _safe(t.ask):
                        market_price = (t.bid + t.ask) / 2
                    elif _safe(t.last):
                        market_price = t.last

                ib_positions.append(PosView(
                    pair=pair, direction=direction, units=item.position,
                    entry_price=entry_price, market_price=market_price,
                    unrealized_pnl=item.unrealizedPNL,
                    stop_loss=stop_loss, take_profit=take_profit,
                    risk_pips=risk_pips,
                ))

            # ── Fills ──
            fills = []
            for fill in ib.fills():
                pair = symbol_to_pair(fill.contract)
                exec_ = fill.execution
                comm = fill.commissionReport
                fills.append(FillView(
                    time=exec_.time.strftime("%H:%M:%S") if exec_.time else "",
                    pair=pair,
                    action=exec_.side.replace("SLD", "SELL").replace("BOT", "BUY"),
                    units=exec_.shares, price=exec_.price,
                    realized_pnl=comm.realizedPNL if comm.realizedPNL else 0,
                    commission=comm.commission if comm.commission else 0,
                ))
            fills.sort(key=lambda f: f.time, reverse=True)

            # ── Risk state ──
            risk = _read_risk_state()
            # Compute exposure % from positions
            nlv = float(account_info.get("NetLiquidation", 0))
            margin_used = float(account_info.get("InitMarginReq", 0) or 0)
            if nlv > 0 and risk:
                risk["exposure_pct"] = round(margin_used / nlv * 100, 2)

            # ── Equity history ──
            if nlv > 0:
                _equity_history.append({
                    "t": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                    "v": nlv,
                })

            # ── News ──
            news_status = news_filter.get_status_dict()

            # ── Latency ──
            latency_ms = int((time.time() - t0) * 1000)

            # ── Update shared state ──
            active_positions = [p for p in ib_positions if p.units != 0]
            with _lock:
                _state["account"] = account_info
                _state["positions"] = active_positions
                _state["orders"] = open_orders
                _state["trades"] = fills
                _state["connected"] = True
                _state["last_update"] = datetime.now(timezone.utc).strftime("%H:%M:%S UTC")
                _state["news_blocked_pairs"] = news_status["news_blocked_pairs"]
                _state["news_events"] = news_status["news_events"]
                _state["market_data"] = market_data
                _state["sparklines"] = sparklines
                _state["risk"] = risk
                _state["equity_history"] = list(_equity_history)
                _state["latency_ms"] = latency_ms

        except Exception as exc:
            logger.warning("Dashboard poller error: %s", exc, exc_info=True)
            with _lock:
                _state["connected"] = False
            try:
                ib.disconnect()
            except Exception:
                pass
            subscribed = False
            tickers.clear()
            contracts.clear()

        time.sleep(10)


# ── Flask Routes ──────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


@app.route("/api/status")
def api_status():
    """JSON API — full state for live JS updates."""
    with _lock:
        return jsonify({
            "connected": _state["connected"],
            "last_update": _state["last_update"],
            "account": _state["account"],
            "positions": [
                {
                    "pair": p.pair, "direction": p.direction,
                    "units": p.units, "entry_price": p.entry_price,
                    "market_price": p.market_price,
                    "unrealized_pnl": p.unrealized_pnl,
                    "stop_loss": p.stop_loss, "take_profit": p.take_profit,
                    "risk_pips": p.risk_pips,
                }
                for p in _state["positions"]
            ],
            "orders": [
                {
                    "pair": o.pair, "order_type": o.order_type,
                    "action": o.action, "units": o.units,
                    "price": o.price, "status": o.status,
                    "time": o.time,
                }
                for o in _state["orders"]
            ],
            "fills": [
                {
                    "time": t.time, "pair": t.pair, "action": t.action,
                    "units": t.units, "price": t.price,
                    "realized_pnl": t.realized_pnl, "commission": t.commission,
                }
                for t in _state["trades"]
            ],
            "news_blocked_pairs": _state["news_blocked_pairs"],
            "news_events": _state["news_events"],
            "market_data": _state["market_data"],
            "sparklines": _state["sparklines"],
            "risk": _state["risk"],
            "equity_history": _state["equity_history"],
            "latency_ms": _state["latency_ms"],
        })


# ── HTML Template ─────────────────────────────────────────────────

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
    --bg-primary: #0a0d12;
    --bg-secondary: #111620;
    --bg-tertiary: #181d2a;
    --bg-hover: #1e2436;
    --bg-card: #141924;
    --border: #222940;
    --border-light: #2d3548;
    --text-primary: #e4e8f0;
    --text-secondary: #8892a8;
    --text-muted: #555e74;
    --accent: #4c8dff;
    --accent-dim: rgba(76,141,255,0.12);
    --green: #00d4a1;
    --green-dim: rgba(0,212,161,0.10);
    --red: #ff475a;
    --red-dim: rgba(255,71,90,0.10);
    --yellow: #ffb800;
    --yellow-dim: rgba(255,184,0,0.10);
    --shadow: 0 2px 12px rgba(0,0,0,0.3);
    --shadow-lg: 0 4px 24px rgba(0,0,0,0.4);
  }
  * { margin:0; padding:0; box-sizing:border-box; }
  body {
    font-family: 'SF Mono','Fira Code','JetBrains Mono','Cascadia Code',Consolas,monospace;
    background: var(--bg-primary); color: var(--text-primary);
    line-height: 1.5; min-height: 100vh;
  }

  /* ── Ticker tape ── */
  .ticker-wrap {
    background: linear-gradient(90deg, var(--bg-tertiary), var(--bg-secondary));
    border-bottom: 1px solid var(--border);
    overflow: hidden; height: 32px; position: relative;
  }
  .ticker {
    display: flex; gap: 48px;
    animation: scroll-ticker 40s linear infinite;
    position: absolute; white-space: nowrap; padding: 6px 0;
  }
  .ticker-item {
    font-size: 0.72em; color: var(--text-secondary);
    display: flex; align-items: center; gap: 8px;
  }
  .ticker-pair { color: var(--text-primary); font-weight: 600; }
  .ticker-price { font-variant-numeric: tabular-nums; }
  .ticker-spread { color: var(--text-muted); font-size: 0.9em; }
  .ticker-change { font-weight: 600; }
  .ticker-badge {
    font-size: 0.7em; padding: 1px 5px; border-radius: 3px;
    font-weight: 700; letter-spacing: 0.03em;
    background: var(--green-dim); color: var(--green);
  }
  @keyframes scroll-ticker {
    0% { transform: translateX(0); }
    100% { transform: translateX(-50%); }
  }

  /* ── Header (glassmorphism) ── */
  .header {
    background: rgba(17,22,32,0.82);
    backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
    border-bottom: 1px solid var(--border);
    padding: 12px 28px; display: flex; align-items: center;
    justify-content: space-between;
    position: sticky; top: 0; z-index: 100;
  }
  .header-left { display: flex; align-items: center; gap: 16px; }
  .logo { font-size: 1.1em; font-weight: 700; letter-spacing: -0.02em; }
  .logo span { color: var(--accent); }
  .conn-badge {
    display: flex; align-items: center; gap: 7px;
    font-size: 0.73em; color: var(--text-secondary);
    background: var(--bg-tertiary); border: 1px solid var(--border);
    border-radius: 6px; padding: 4px 10px;
  }
  .conn-dot {
    width: 7px; height: 7px; border-radius: 50%; background: var(--red);
    transition: background 0.3s;
  }
  .conn-dot.live {
    background: var(--green);
    box-shadow: 0 0 6px rgba(0,212,161,0.5);
    animation: pulse 2s infinite;
  }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:0.5} }
  .latency-badge {
    font-size: 0.85em; padding: 2px 6px; border-radius: 4px;
    font-weight: 600; font-variant-numeric: tabular-nums;
    background: var(--green-dim); color: var(--green);
  }
  .latency-badge.warn { background: var(--yellow-dim); color: var(--yellow); }
  .latency-badge.bad { background: var(--red-dim); color: var(--red); }
  .header-right {
    display: flex; align-items: center; gap: 16px;
    font-size: 0.75em; color: var(--text-muted);
  }
  .update-timer { font-variant-numeric: tabular-nums; }

  /* ── Main ── */
  .main { padding: 20px 28px; max-width: 1480px; margin: 0 auto; }

  /* ── News alert ── */
  .news-alert {
    background: var(--red-dim); border: 1px solid var(--red);
    border-radius: 8px; padding: 10px 20px; margin-bottom: 16px;
    display: none; align-items: center; gap: 12px;
    animation: pulse-red 2s infinite;
  }
  .news-alert.active { display: flex; }
  .news-alert-icon { font-size: 1.1em; color: var(--red); font-weight: 700; }
  .news-alert-text { font-size: 0.82em; color: var(--red); font-weight: 600; }
  .news-alert-pairs { font-size: 0.75em; color: var(--text-secondary); }
  @keyframes pulse-red { 0%,100%{border-color:var(--red)} 50%{border-color:rgba(255,71,90,0.3)} }

  /* ── Hero section ── */
  .hero {
    display: flex; align-items: stretch; gap: 20px; margin-bottom: 16px;
  }
  .hero-left {
    flex: 0 0 320px; background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 10px; padding: 20px 24px;
    box-shadow: var(--shadow); display: flex; flex-direction: column; justify-content: center;
  }
  .hero-label { font-size: 0.65em; text-transform: uppercase; letter-spacing: 0.1em; color: var(--text-muted); margin-bottom: 4px; }
  .hero-value { font-size: 2.2em; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1.1; }
  .hero-change { font-size: 0.85em; font-weight: 600; margin-top: 6px; }
  .hero-chart {
    flex: 1; background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 10px; padding: 14px 16px; box-shadow: var(--shadow);
    height: 160px; position: relative; overflow: hidden;
  }
  .hero-chart-title { font-size: 0.65em; text-transform: uppercase; letter-spacing: 0.06em; color: var(--text-muted); margin-bottom: 6px; }

  /* ── Stat cards ── */
  .cards {
    display: grid; grid-template-columns: repeat(6, 1fr);
    gap: 10px; margin-bottom: 16px;
  }
  .card {
    background: var(--bg-card); border: 1px solid var(--border);
    border-radius: 8px; padding: 14px 16px;
    box-shadow: var(--shadow); transition: border-color 0.2s, transform 0.15s;
  }
  .card:hover { border-color: var(--border-light); transform: translateY(-1px); }
  .card-label { font-size: 0.6em; text-transform: uppercase; letter-spacing: 0.08em; color: var(--text-muted); margin-bottom: 4px; }
  .card-value { font-size: 1.2em; font-weight: 700; font-variant-numeric: tabular-nums; }
  .card-sub { font-size: 0.65em; color: var(--text-muted); margin-top: 2px; }
  .val-pos { color: var(--green); }
  .val-neg { color: var(--red); }
  .result-badge { font-size: 0.7em; font-weight: 600; padding: 2px 6px; border-radius: 4px; margin-left: 6px; vertical-align: middle; }
  .result-badge.win { background: rgba(0,212,161,0.2); color: var(--green); }
  .result-badge.loss { background: rgba(255,71,90,0.2); color: var(--red); }
  .val-warn { color: var(--yellow); }
  .val-neutral { color: var(--text-primary); }

  /* ── Risk panel ── */
  .risk-panel {
    display: flex; align-items: center; gap: 20px; margin-bottom: 16px;
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 10px; padding: 16px 24px; box-shadow: var(--shadow);
  }
  .risk-gauges { display: flex; gap: 24px; flex: 1; }
  .risk-gauge {
    display: flex; flex-direction: column; align-items: center; gap: 4px;
    position: relative; width: 110px;
  }
  .risk-gauge canvas { width: 100px !important; height: 100px !important; }
  .gauge-center {
    position: absolute; top: 30px; left: 50%; transform: translateX(-50%);
    text-align: center;
  }
  .gauge-center-val { font-size: 1.1em; font-weight: 700; }
  .gauge-center-label { font-size: 0.55em; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.06em; }
  .risk-status {
    display: flex; flex-direction: column; gap: 10px;
    padding-left: 20px; border-left: 1px solid var(--border);
  }
  .status-row { display: flex; align-items: center; gap: 8px; font-size: 0.75em; }
  .status-dot-sm {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--green); flex-shrink: 0;
  }
  .status-dot-sm.triggered { background: var(--red); box-shadow: 0 0 6px rgba(255,71,90,0.5); }
  .loss-dots { display: flex; gap: 5px; margin-top: 4px; }
  .loss-dot {
    width: 8px; height: 8px; border-radius: 50%;
    background: var(--bg-tertiary); border: 1px solid var(--border);
  }
  .loss-dot.filled { background: var(--red); border-color: var(--red); }

  /* ── Market overview grid ── */
  .market-grid {
    display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
    gap: 10px; padding: 12px 16px;
  }
  .market-card {
    background: var(--bg-tertiary); border: 1px solid var(--border);
    border-radius: 8px; padding: 10px 12px;
    transition: border-color 0.2s, box-shadow 0.2s; cursor: default;
  }
  .market-card:hover { border-color: var(--border-light); box-shadow: var(--shadow); }
  .mc-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 2px; }
  .mc-pair { font-weight: 700; font-size: 0.82em; }
  .mc-badges { display: flex; gap: 4px; }
  .mc-badge {
    font-size: 0.55em; padding: 1px 5px; border-radius: 3px;
    font-weight: 700; letter-spacing: 0.04em;
  }
  .mc-badge-live { background: var(--green-dim); color: var(--green); }
  .mc-badge-news { background: var(--red-dim); color: var(--red); }
  .mc-price { font-size: 1.05em; font-weight: 600; font-variant-numeric: tabular-nums; }
  .mc-row { display: flex; justify-content: space-between; font-size: 0.65em; color: var(--text-muted); }
  .mc-sparkline { height: 36px; margin-top: 4px; }

  /* ── Position gauges ── */
  .gauges-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
    gap: 10px; padding: 12px 16px;
  }
  .gauge-card {
    background: var(--bg-tertiary); border-radius: 8px;
    padding: 14px 16px; border-left: 3px solid var(--border);
    transition: border-color 0.3s;
  }
  .gauge-card.profit { border-left-color: var(--green); }
  .gauge-card.loss { border-left-color: var(--red); }
  .gauge-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px; }
  .gauge-pair { font-weight: 700; font-size: 0.88em; }
  .gauge-pnl { font-weight: 600; font-size: 0.85em; }
  .gauge-meta { display: flex; gap: 12px; font-size: 0.65em; color: var(--text-muted); margin-bottom: 6px; flex-wrap: wrap; }
  .gauge-meta .sl { color: var(--red); font-weight: 600; }
  .gauge-meta .tp { color: var(--green); font-weight: 600; }
  .gauge-track {
    position: relative; height: 8px; background: var(--bg-primary);
    border-radius: 4px; margin: 6px 0; overflow: visible;
  }
  .gauge-zone { position: absolute; height: 100%; border-radius: 4px; }
  .gauge-marker {
    position: absolute; top: -4px; width: 3px; height: 16px;
    border-radius: 2px; transform: translateX(-50%); transition: left 0.8s ease;
  }
  .gauge-labels {
    display: flex; justify-content: space-between;
    font-size: 0.6em; color: var(--text-muted); margin-top: 4px;
  }
  .gauge-labels .sl { color: var(--red); }
  .gauge-labels .tp { color: var(--green); }
  .gauge-labels .entry { color: var(--text-secondary); }

  /* ── Section ── */
  .section {
    background: var(--bg-secondary); border: 1px solid var(--border);
    border-radius: 10px; margin-bottom: 16px; overflow: hidden;
    box-shadow: var(--shadow);
  }
  .section-head {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 16px; border-bottom: 1px solid var(--border);
  }
  .section-title {
    font-size: 0.75em; font-weight: 600; color: var(--text-secondary);
    text-transform: uppercase; letter-spacing: 0.06em;
  }
  .section-count {
    font-size: 0.7em; color: var(--text-muted);
    background: var(--bg-tertiary); padding: 2px 8px; border-radius: 4px;
  }

  /* ── Two-column bottom ── */
  .bottom-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }

  /* ── Tables ── */
  table { width: 100%; border-collapse: collapse; }
  thead th {
    font-size: 0.62em; text-transform: uppercase; letter-spacing: 0.06em;
    color: var(--text-muted); padding: 8px 14px; text-align: left;
    border-bottom: 1px solid var(--border); background: var(--bg-tertiary);
    position: sticky; top: 0;
  }
  thead th.r { text-align: right; }
  tbody td {
    padding: 8px 14px; font-size: 0.78em;
    border-bottom: 1px solid rgba(34,41,64,0.5);
    font-variant-numeric: tabular-nums;
  }
  tbody td.r { text-align: right; }
  tbody tr { transition: background 0.15s; }
  tbody tr:hover { background: var(--bg-hover); }
  tbody tr:last-child td { border-bottom: none; }
  .empty-state { padding: 24px 16px; text-align: center; color: var(--text-muted); font-size: 0.78em; }

  /* ── Tags ── */
  .tag { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 0.72em; font-weight: 700; letter-spacing: 0.04em; }
  .tag-long { background: var(--accent-dim); color: var(--accent); }
  .tag-short { background: var(--red-dim); color: var(--red); }
  .tag-buy { background: var(--accent-dim); color: var(--accent); }
  .tag-sell { background: var(--red-dim); color: var(--red); }

  /* ── P&L bar ── */
  .pnl-cell { display: flex; align-items: center; gap: 8px; justify-content: flex-end; }
  .pnl-bar { width: 40px; height: 4px; border-radius: 2px; background: var(--bg-tertiary); overflow: hidden; }
  .pnl-bar-fill { height: 100%; border-radius: 2px; transition: width 0.5s ease; }

  /* ── Flash ── */
  @keyframes flash-green { 0%{background:var(--green-dim)} 100%{background:transparent} }
  @keyframes flash-red { 0%{background:var(--red-dim)} 100%{background:transparent} }
  .flash-up { animation: flash-green 0.8s ease-out; }
  .flash-down { animation: flash-red 0.8s ease-out; }

  /* ── News rows ── */
  tbody tr.news-active { background: rgba(255,71,90,0.06); }
  .impact-high { color: var(--red); font-weight: 700; }
  .impact-medium { color: var(--yellow); font-weight: 600; }
  .impact-low { color: var(--text-muted); }
  .impact-holiday { color: var(--accent); }
  .countdown { font-variant-numeric: tabular-nums; font-weight: 600; }
  .countdown.imminent { color: var(--red); animation: pulse 1.5s infinite; }

  /* ── Footer ── */
  .footer { text-align: center; padding: 16px; font-size: 0.6em; color: var(--text-muted); }

  /* ── Responsive ── */
  @media (max-width: 1100px) {
    .hero { flex-direction: column; }
    .hero-left { flex: none; }
    .cards { grid-template-columns: repeat(3, 1fr); }
    .risk-panel { flex-direction: column; align-items: stretch; }
    .risk-status { padding-left: 0; border-left: none; padding-top: 12px; border-top: 1px solid var(--border); }
  }
  @media (max-width: 768px) {
    .header { padding: 10px 16px; }
    .main { padding: 12px 16px; }
    .cards { grid-template-columns: repeat(2, 1fr); }
    .bottom-grid { grid-template-columns: 1fr; }
    .market-grid { grid-template-columns: repeat(2, 1fr); }
    /* News calendar mobile */
    #newsTable { overflow-x: auto; -webkit-overflow-scrolling: touch; }
    #newsTable table { font-size: 0.75em; min-width: 500px; }
    #newsTable th:nth-child(2), #newsTable td:nth-child(2) { display: none; } /* Hide Time column */
    #newsTable th:nth-child(6), #newsTable td:nth-child(6) { display: none; } /* Hide Prev column */
    #newsTable th:nth-child(7), #newsTable td:nth-child(7) { display: none; } /* Hide Pairs column */
  }
  @media (max-width: 480px) {
    .cards { grid-template-columns: 1fr 1fr; gap: 8px; }
    .card { padding: 10px; }
    .card-value { font-size: 1em; }
    /* News calendar very small screens */
    #newsTable table { min-width: 320px; font-size: 0.7em; }
    #newsTable th:nth-child(5), #newsTable td:nth-child(5) { display: none; } /* Also hide Impact on tiny screens */
  }
</style>
</head>
<body>

<!-- Ticker tape -->
<div class="ticker-wrap"><div class="ticker" id="tickerTape"></div></div>

<!-- Header -->
<div class="header">
  <div class="header-left">
    <div class="logo"><span>ICT</span> Trading Bot</div>
    <div class="conn-badge">
      <div class="conn-dot" id="connDot"></div>
      <span id="connText">Connecting...</span>
      <span class="latency-badge" id="latencyBadge">--</span>
    </div>
  </div>
  <div class="header-right">
    <span id="accountId"></span>
    <span class="update-timer" id="updateTimer">--</span>
  </div>
</div>

<div class="main">
  <!-- News alert -->
  <div class="news-alert" id="newsAlert">
    <span class="news-alert-icon">&#9888;</span>
    <span class="news-alert-text">TRADING PAUSED</span>
    <span class="news-alert-pairs" id="newsAlertPairs">News blackout active</span>
  </div>

  <!-- Hero section -->
  <div class="hero">
    <div class="hero-left">
      <div class="hero-label">Portfolio Value</div>
      <div class="hero-value val-neutral" id="heroNLV">--</div>
      <div class="hero-change" id="heroChange">&nbsp;</div>
    </div>
    <div class="hero-chart">
      <div class="hero-chart-title">Session P&amp;L</div>
      <div style="position:relative;height:120px">
        <canvas id="equityChart"></canvas>
      </div>
    </div>
  </div>

  <!-- Stat cards -->
  <div class="cards">
    <div class="card"><div class="card-label">Daily P&amp;L</div><div class="card-value" id="cardDailyPnl">--</div></div>
    <div class="card"><div class="card-label">Unrealized P&amp;L</div><div class="card-value" id="cardUPnL">--</div></div>
    <div class="card"><div class="card-label">Margin Used</div><div class="card-value val-neutral" id="cardMargin">--</div></div>
    <div class="card"><div class="card-label">Buying Power</div><div class="card-value val-neutral" id="cardBuyPow">--</div></div>
    <div class="card"><div class="card-label">Open Positions</div><div class="card-value val-neutral" id="cardPosCount">0</div></div>
    <div class="card"><div class="card-label">Consec. Losses</div><div class="card-value val-neutral" id="cardConsecLoss">0</div></div>
  </div>

  <!-- Risk panel -->
  <div class="risk-panel">
    <div class="risk-gauges">
      <div class="risk-gauge">
        <canvas id="gaugeDD"></canvas>
        <div class="gauge-center"><div class="gauge-center-val" id="gaugeDDVal">0%</div><div class="gauge-center-label">Drawdown</div></div>
      </div>
      <div class="risk-gauge">
        <canvas id="gaugeDL"></canvas>
        <div class="gauge-center"><div class="gauge-center-val" id="gaugeDLVal">0%</div><div class="gauge-center-label">Daily Loss</div></div>
      </div>
      <div class="risk-gauge">
        <canvas id="gaugeEX"></canvas>
        <div class="gauge-center"><div class="gauge-center-val" id="gaugeEXVal">0%</div><div class="gauge-center-label">Exposure</div></div>
      </div>
    </div>
    <div class="risk-status">
      <div class="status-row"><div class="status-dot-sm" id="cbDot"></div><span>Circuit Breaker</span></div>
      <div class="status-row"><div class="status-dot-sm" id="ksDot"></div><span>Kill Switch</span></div>
      <div class="loss-dots" id="lossDots"></div>
    </div>
  </div>

  <!-- Market overview -->
  <div class="section">
    <div class="section-head">
      <div class="section-title">Market Overview</div>
      <div class="section-count">7 pairs</div>
    </div>
    <div class="market-grid" id="marketGrid"></div>
  </div>

  <!-- Positions -->
  <div class="section">
    <div class="section-head">
      <div class="section-title">Open Positions</div>
      <div class="section-count" id="posCount">0</div>
    </div>
    <div id="positionsArea"></div>
  </div>

  <!-- Two-column bottom -->
  <div class="bottom-grid">
    <div class="section">
      <div class="section-head">
        <div class="section-title">Trade History</div>
        <div class="section-count" id="activityCount">0</div>
      </div>
      <div id="activityFeed"></div>
    </div>
    <div class="section">
      <div class="section-head">
        <div class="section-title">News Calendar</div>
        <div class="section-count" id="newsCount">0</div>
      </div>
      <div id="newsTable"></div>
    </div>
  </div>
</div>

<div class="footer">Live data from IB Gateway &middot; Updates every 10s</div>

<script>
(function(){
"use strict";

const ALL_PAIRS = ['EUR_USD','GBP_USD','USD_JPY','AUD_USD','USD_CAD','NZD_USD','EUR_GBP'];
const PIP_SIZES = {EUR_USD:0.0001,GBP_USD:0.0001,USD_JPY:0.01,AUD_USD:0.0001,USD_CAD:0.0001,NZD_USD:0.0001,EUR_GBP:0.0001};
let lastUpdateTs = 0;
let prevPnlMap = {};
let equityChart = null;
let riskGauges = {};       // {dd, dl, ex}
let sparklineCharts = {};  // pair -> Chart
let marketGridBuilt = false;

Chart.defaults.color = '#8892a8';
Chart.defaults.borderColor = '#222940';
Chart.defaults.font.family = "'SF Mono','Fira Code',Consolas,monospace";
Chart.defaults.font.size = 11;

// ── Helpers ──
function fmt(v,d){ if(v==null||v===''||v==='N/A')return'--'; const n=parseFloat(v); return isNaN(n)?v:n.toLocaleString('en-US',{minimumFractionDigits:d||0,maximumFractionDigits:d||0}); }
function fmtP(v){ if(v==null)return'--'; const n=parseFloat(v); return isNaN(n)?'--':n.toFixed(5); }
function fmtPJ(v,pair){ if(v==null)return'--'; const n=parseFloat(v); if(isNaN(n))return'--'; return pair==='USD_JPY'?n.toFixed(3):n.toFixed(5); }
function pnlCls(v){ const n=parseFloat(v); return isNaN(n)||n===0?'val-neutral':n>=0?'val-pos':'val-neg'; }
function tagH(d){ const c=(d==='long'||d==='BUY')?'tag-long':'tag-short'; const l=(d==='long'||d==='BUY')?(d==='BUY'?'BUY':'LONG'):(d==='SELL'?'SELL':'SHORT'); return '<span class="tag '+c+'">'+l+'</span>'; }
function flashEl(el,dir){ el.classList.remove('flash-up','flash-down'); void el.offsetWidth; el.classList.add(dir>0?'flash-up':'flash-down'); }
function timeNow(){ const d=new Date(); return d.getHours().toString().padStart(2,'0')+':'+d.getMinutes().toString().padStart(2,'0')+':'+d.getSeconds().toString().padStart(2,'0'); }

function gaugeColor(pct){
  if(pct<50) return 'var(--green)';
  if(pct<80) return 'var(--yellow)';
  return 'var(--red)';
}

function updateCard(id, value, cls) {
  const el = document.getElementById(id);
  if(!el) return;
  const prev = el.textContent;
  el.textContent = value;
  if(cls !== undefined) el.className = 'card-value ' + cls;
  if(prev !== '--' && prev !== value) {
    const dir = parseFloat(value.replace(/,/g,'')) >= parseFloat(prev.replace(/,/g,'')) ? 1 : -1;
    flashEl(el, dir);
  }
}

// ── Equity Chart ──
const EQUITY_WINDOW = 120; // show last 20 min (120 x 10s)
function initEquityChart() {
  const ctx = document.getElementById('equityChart').getContext('2d');
  equityChart = new Chart(ctx, {
    type: 'line',
    data: { labels: [], datasets: [{ data: [], borderColor: '#4c8dff', borderWidth: 1.5, fill: false, tension: 0.3, pointRadius: 0, pointHitRadius: 8 }] },
    options: {
      responsive: true, maintainAspectRatio: false,
      animation: false,
      interaction: { mode: 'index', intersect: false },
      plugins: { legend: { display: false }, tooltip: { backgroundColor:'#181d2a', borderColor:'#222940', borderWidth:1, titleColor:'#e4e8f0', bodyColor:'#8892a8', callbacks: { label: function(c){ return (c.parsed.y>=0?'+':'')+c.parsed.y.toLocaleString('en-US',{minimumFractionDigits:2,maximumFractionDigits:2}); } } } },
      scales: {
        x: { grid:{color:'rgba(34,41,64,0.3)'}, ticks:{maxTicksLimit:6,font:{size:9}} },
        y: { grid:{color:'rgba(34,41,64,0.3)'}, suggestedMin: -5, suggestedMax: 5, ticks:{font:{size:9}, callback:function(v){return (v>=0?'+':'')+v.toFixed(2);}} }
      }
    }
  });
}

function updateEquityChart(history) {
  const ds = equityChart.data.datasets[0];
  if(!history || history.length === 0) {
    // Pre-fill with flat zero line so chart is always full-width
    equityChart.data.labels = new Array(EQUITY_WINDOW).fill('');
    ds.data = new Array(EQUITY_WINDOW).fill(0);
    equityChart.update('none');
    return;
  }
  const win = history.slice(-EQUITY_WINDOW);
  const base = win[0].v;
  // Pad left with zeros so chart is always fixed-width
  const padLen = EQUITY_WINDOW - win.length;
  equityChart.data.labels = new Array(padLen).fill('').concat(win.map(h => h.t));
  ds.data = new Array(padLen).fill(0).concat(win.map(h => h.v - base));
  const last = ds.data[ds.data.length - 1];
  if(last >= 0) { ds.borderColor='#00d4a1'; }
  else { ds.borderColor='#ff475a'; }
  equityChart.update('none');
}

// ── Risk Gauges (doughnut arcs) ──
function initRiskGauges() {
  ['DD','DL','EX'].forEach(function(key) {
    const ctx = document.getElementById('gauge'+key).getContext('2d');
    riskGauges[key] = new Chart(ctx, {
      type: 'doughnut',
      data: { datasets: [{ data: [0, 100], backgroundColor: ['var(--green)', 'rgba(255,255,255,0.04)'], borderWidth: 0 }] },
      options: { circumference: 270, rotation: -135, cutout: '78%', plugins: { legend:{display:false}, tooltip:{enabled:false} }, animation: false, responsive: true, maintainAspectRatio: true }
    });
  });
}

function updateRiskGauges(risk, acc) {
  if(!risk || !risk.balance) return;
  const dd = Math.min(risk.drawdown_pct || 0, 10);
  const ddPct = dd / 10 * 100;
  // Use IB's RealizedPnL instead of bot's daily_pnl for accuracy
  const dpnl = acc ? parseFloat(acc.RealizedPnL_raw || 0) : (risk.daily_pnl || 0);
  const nlv = acc ? parseFloat(acc.NetLiquidation || 0) : risk.balance;
  const dlPct = Math.min(Math.abs(dpnl < 0 ? dpnl / nlv * 100 : 0) / 3 * 100, 100);
  const exPct = Math.min((risk.exposure_pct || 0) / 50 * 100, 100);

  [{k:'DD',v:ddPct,label:dd.toFixed(1)+'%'},{k:'DL',v:dlPct,label:(Math.abs(dpnl<0&&nlv>0?dpnl/nlv*100:0)).toFixed(1)+'%'},{k:'EX',v:exPct,label:(risk.exposure_pct||0).toFixed(1)+'%'}].forEach(function(g){
    const chart = riskGauges[g.k];
    if(!chart) return;
    const c = gaugeColor(g.v);
    chart.data.datasets[0].data = [g.v, 100 - g.v];
    chart.data.datasets[0].backgroundColor = [c, 'rgba(255,255,255,0.04)'];
    chart.update('none');
    document.getElementById('gauge'+g.k+'Val').textContent = g.label;
    document.getElementById('gauge'+g.k+'Val').style.color = c;
  });
}

// ── Ticker Tape ──
function updateTicker(marketData, positions) {
  const tape = document.getElementById('tickerTape');
  const posPairs = new Set((positions||[]).map(p=>p.pair));
  let items = '';
  for(let dup=0; dup<4; dup++){
    for(const pair of ALL_PAIRS){
      const md = marketData[pair] || {};
      const price = md.bid || md.last || md.close;
      const priceStr = price ? fmtPJ(price, pair) : '--';
      const spread = (md.bid && md.ask) ? ((md.ask - md.bid) / PIP_SIZES[pair]).toFixed(1) : '--';
      const chg = (md.last && md.close) ? ((md.last - md.close) / md.close * 100) : (md.bid && md.close) ? ((md.bid - md.close) / md.close * 100) : null;
      const chgStr = chg !== null ? (chg>=0?'+':'')+chg.toFixed(2)+'%' : '';
      const chgColor = chg !== null ? (chg>=0?'var(--green)':'var(--red)') : 'var(--text-muted)';
      const badge = posPairs.has(pair) ? ' <span class="ticker-badge">LIVE</span>' : '';
      items += '<span class="ticker-item"><span class="ticker-pair">'+pair.replace('_','/')+'</span><span class="ticker-price">'+priceStr+'</span><span class="ticker-spread">'+spread+'p</span><span class="ticker-change" style="color:'+chgColor+'">'+chgStr+'</span>'+badge+'</span>';
    }
  }
  tape.innerHTML = items;
}

// ── Market Overview Grid ──
function buildMarketGrid() {
  const container = document.getElementById('marketGrid');
  let html = '';
  for(const pair of ALL_PAIRS) {
    const id = pair.replace('_','');
    html += '<div class="market-card" id="mc-'+id+'">'+
      '<div class="mc-header"><span class="mc-pair">'+pair.replace('_','/')+'</span><div class="mc-badges" id="mcb-'+id+'"></div></div>'+
      '<div class="mc-price" id="mcp-'+id+'">--</div>'+
      '<div class="mc-row"><span id="mcspread-'+id+'">--</span><span class="mc-change" id="mcchg-'+id+'">--</span></div>'+
      '<div class="mc-sparkline"><canvas id="spark-'+id+'" height="36"></canvas></div>'+
    '</div>';
  }
  container.innerHTML = html;
  marketGridBuilt = true;
}

function updateMarketGrid(marketData, sparklines, positions, blockedPairs) {
  if(!marketGridBuilt) buildMarketGrid();
  const posPairs = new Set((positions||[]).map(p=>p.pair));
  const blocked = new Set(blockedPairs||[]);

  for(const pair of ALL_PAIRS) {
    const id = pair.replace('_','');
    const md = marketData[pair] || {};
    const price = md.bid || md.last || md.close;
    const el = document.getElementById('mcp-'+id);
    if(el) el.textContent = price ? fmtPJ(price, pair) : '--';

    const spread = (md.bid && md.ask) ? ((md.ask-md.bid)/PIP_SIZES[pair]).toFixed(1)+'p' : '--';
    const spreadEl = document.getElementById('mcspread-'+id);
    if(spreadEl) spreadEl.textContent = 'Spread: '+spread;

    const chg = (md.bid && md.close) ? ((md.bid-md.close)/md.close*100) : null;
    const chgEl = document.getElementById('mcchg-'+id);
    if(chgEl) {
      chgEl.textContent = chg!==null ? (chg>=0?'+':'')+chg.toFixed(2)+'%' : '--';
      chgEl.style.color = chg!==null ? (chg>=0?'var(--green)':'var(--red)') : 'var(--text-muted)';
    }

    // Badges
    let badges = '';
    if(posPairs.has(pair)) badges += '<span class="mc-badge mc-badge-live">LIVE</span>';
    if(blocked.has(pair)) badges += '<span class="mc-badge mc-badge-news">NEWS</span>';
    const badgeEl = document.getElementById('mcb-'+id);
    if(badgeEl) badgeEl.innerHTML = badges;

    // Sparkline
    const sparkData = (sparklines||{})[pair] || [];
    if(sparkData.length > 1) {
      const canvasId = 'spark-'+id;
      const canvas = document.getElementById(canvasId);
      if(!canvas) continue;
      const values = sparkData.map(d=>d.c);
      const isUp = values[values.length-1] >= values[0];
      const color = isUp ? '#00d4a1' : '#ff475a';
      const bg = isUp ? 'rgba(0,212,161,0.08)' : 'rgba(255,71,90,0.08)';

      if(sparklineCharts[pair] && sparklineCharts[pair].canvas === canvas) {
        const sc = sparklineCharts[pair];
        sc.data.labels = sparkData.map(d=>d.t);
        sc.data.datasets[0].data = values;
        sc.data.datasets[0].borderColor = color;
        sc.data.datasets[0].backgroundColor = bg;
        sc.update('none');
      } else {
        if(sparklineCharts[pair]) sparklineCharts[pair].destroy();
        sparklineCharts[pair] = new Chart(canvas.getContext('2d'), {
          type:'line',
          data:{ labels:sparkData.map(d=>d.t), datasets:[{ data:values, borderColor:color, backgroundColor:bg, borderWidth:1.5, fill:true, tension:0.3, pointRadius:0 }] },
          options:{ responsive:true, maintainAspectRatio:false, animation:false, plugins:{legend:{display:false},tooltip:{enabled:false}}, scales:{x:{display:false},y:{display:false}} }
        });
      }
    }
  }
}

// ── Position Gauges ──
function renderPositions(positions, ccySym, usdRate) {
  ccySym = ccySym || '';
  usdRate = usdRate || 1;
  const container = document.getElementById('positionsArea');
  const countEl = document.getElementById('posCount');
  const cardCountEl = document.getElementById('cardPosCount');
  if(!positions || positions.length === 0) {
    container.innerHTML = '<div class="empty-state">No open positions</div>';
    countEl.textContent = '0';
    cardCountEl.textContent = '0';
    return;
  }
  countEl.textContent = positions.length;
  cardCountEl.textContent = positions.length;

  let html = '<div class="gauges-grid">';
  for(const p of positions) {
    if(p.market_price==null) continue;
    const sl=p.stop_loss, tp=p.take_profit, entry=p.entry_price, price=p.market_price;
    const pnl = (p.unrealized_pnl||0) * usdRate;  // Convert USD to base currency
    const hasGauge = sl!=null && tp!=null;

    // R-multiple (if SL known)
    let rStr = '--';
    if(sl!=null) {
      const riskDist = Math.abs(entry - sl);
      const rMult = riskDist > 0 ? ((price - entry) * (p.direction==='long'?1:-1) / riskDist) : 0;
      rStr = (rMult>=0?'+':'')+rMult.toFixed(2)+'R';
    }

    html += '<div class="gauge-card '+(pnl>=0?'profit':'loss')+'">'+
      '<div class="gauge-header"><span class="gauge-pair">'+p.pair+' '+tagH(p.direction)+'</span><span class="gauge-pnl '+pnlCls(pnl)+'">'+ccySym+(pnl>=0?'+':'')+fmt(pnl,2)+'</span></div>'+
      '<div class="gauge-meta"><span>'+rStr+'</span><span class="sl">SL '+(sl!=null?fmtP(sl):'NONE')+'</span><span class="tp">TP '+(tp!=null?fmtP(tp):'NONE')+'</span><span>Risk: '+(p.risk_pips!=null?fmt(p.risk_pips,1)+'p':'--')+'</span><span>Units: '+fmt(Math.abs(p.units),0)+'</span></div>';

    if(hasGauge) {
      const lo=Math.min(sl,tp), hi=Math.max(sl,tp), range=hi-lo||1;
      const pricePct=Math.max(0,Math.min(100,((price-lo)/range)*100));
      const entryPct=Math.max(0,Math.min(100,((entry-lo)/range)*100));
      const slIsLeft=sl<tp;
      const zonePct1=Math.min(entryPct,pricePct), zonePct2=Math.max(entryPct,pricePct);
      const zoneColor=pnl>=0?'rgba(0,212,161,0.2)':'rgba(255,71,90,0.2)';
      const markerColor=pnl>=0?'#00d4a1':'#ff475a';
      html += '<div class="gauge-track">'+
        '<div class="gauge-zone" style="left:'+zonePct1+'%;width:'+(zonePct2-zonePct1)+'%;background:'+zoneColor+'"></div>'+
        '<div class="gauge-marker" style="left:'+entryPct+'%;background:var(--text-muted)" title="Entry '+fmtP(entry)+'"></div>'+
        '<div class="gauge-marker" style="left:'+pricePct+'%;background:'+markerColor+'" title="Current '+fmtP(price)+'"></div>'+
      '</div>'+
      '<div class="gauge-labels">'+
        '<span class="'+(slIsLeft?'sl':'tp')+'">'+(slIsLeft?'SL '+fmtP(sl):'TP '+fmtP(tp))+'</span>'+
        '<span class="entry">Entry '+fmtP(entry)+'</span>'+
        '<span class="'+(slIsLeft?'tp':'sl')+'">'+(slIsLeft?'TP '+fmtP(tp):'SL '+fmtP(sl))+'</span>'+
      '</div>';
    }
    html += '</div>';
  }
  html += '</div>';
  container.innerHTML = html;
}

// ── Trade History (fills only, no pending orders) ──
function renderActivityFeed(fills, orders) {
  const container = document.getElementById('activityFeed');
  // Only show fills - pending SL/TP orders are already visible in Open Positions
  const items = fills || [];
  document.getElementById('activityCount').textContent = items.length;

  if(items.length === 0) {
    container.innerHTML = '<div class="empty-state">No trades yet</div>';
    return;
  }
  let html = '<table><thead><tr><th>Time</th><th>Pair</th><th>Action</th><th class="r">Units</th><th class="r">Price</th><th class="r">P&amp;L</th></tr></thead><tbody>';
  for(const d of items) {
    const rpnl = d.realized_pnl||0;
    let pnlCell = '--';
    let resultBadge = '';
    if(rpnl !== 0) {
      const isWin = rpnl > 0;
      pnlCell = (isWin ? '+' : '') + fmt(rpnl, 2);
      resultBadge = isWin 
        ? '<span class="result-badge win">WIN</span>' 
        : '<span class="result-badge loss">LOSS</span>';
    }
    html += '<tr><td>'+d.time+'</td><td><strong>'+d.pair+'</strong> '+resultBadge+'</td><td>'+tagH(d.action)+'</td><td class="r">'+fmt(d.units,0)+'</td><td class="r">'+fmtP(d.price)+'</td><td class="r '+pnlCls(rpnl)+'">'+pnlCell+'</td></tr>';
  }
  html += '</tbody></table>';
  container.innerHTML = html;
}

// ── News Calendar ──
function renderNews(events, blockedPairs) {
  const container = document.getElementById('newsTable');
  document.getElementById('newsCount').textContent = events ? events.length : 0;
  if(!events || events.length===0) {
    container.innerHTML = '<div class="empty-state">No upcoming events</div>';
    return;
  }
  const blockedSet = new Set(blockedPairs||[]);
  const now = Date.now();
  let html = '<table><thead><tr><th>Countdown</th><th>Time (CET)</th><th>Currency</th><th>Event</th><th>Impact</th><th class="r">Prev</th><th>Pairs</th></tr></thead><tbody>';
  for(const e of events) {
    const d = new Date(e.date);
    const diff = d.getTime() - now;
    let cdStr = '';
    if(diff > 0) {
      const hrs = Math.floor(diff/3600000);
      const mins = Math.floor((diff%3600000)/60000);
      cdStr = hrs > 0 ? hrs+'h '+mins+'m' : mins+'m';
    } else { cdStr = 'PAST'; }
    const imminent = diff > 0 && diff < 1800000;
    const cdCls = imminent ? 'countdown imminent' : 'countdown';
    const cet = new Date(d.toLocaleString('en-US',{timeZone:'Europe/Stockholm'}));
    const timeStr = cet.getFullYear()+'-'+String(cet.getMonth()+1).padStart(2,'0')+'-'+String(cet.getDate()).padStart(2,'0')+' '+String(cet.getHours()).padStart(2,'0')+':'+String(cet.getMinutes()).padStart(2,'0');
    const impCls = e.impact==='High'?'impact-high':e.impact==='Medium'?'impact-medium':e.impact==='Holiday'?'impact-holiday':'impact-low';
    const isActive = e.affected_pairs && e.affected_pairs.some(function(p){return blockedSet.has(p);});
    const rowCls = isActive ? ' class="news-active"' : '';
    html += '<tr'+rowCls+'><td><span class="'+cdCls+'">'+cdStr+'</span></td><td>'+timeStr+'</td><td><strong>'+e.country+'</strong></td><td>'+e.title+'</td><td><span class="'+impCls+'">'+e.impact+'</span></td><td class="r">'+(e.previous||'--')+'</td><td style="font-size:0.7em">'+(e.affected_pairs||[]).join(', ')+'</td></tr>';
  }
  html += '</tbody></table>';
  container.innerHTML = html;
}

// ── Timer ──
function updateTimerDisplay() {
  const el = document.getElementById('updateTimer');
  if(!lastUpdateTs) { el.textContent = '--'; return; }
  const ago = Math.floor((Date.now() - lastUpdateTs) / 1000);
  el.textContent = ago < 2 ? 'just now' : ago + 's ago';
}
setInterval(updateTimerDisplay, 1000);

// ── Main Fetch Loop ──
async function fetchAndUpdate() {
  try {
    const resp = await fetch('/api/status');
    if(!resp.ok) throw new Error('HTTP '+resp.status);
    const data = await resp.json();
    lastUpdateTs = Date.now();

    // Connection
    const dot = document.getElementById('connDot');
    const txt = document.getElementById('connText');
    if(data.connected) { dot.classList.add('live'); txt.textContent='Live'; }
    else { dot.classList.remove('live'); txt.textContent='Disconnected'; }

    // Latency
    const lb = document.getElementById('latencyBadge');
    const lat = data.latency_ms || 0;
    lb.textContent = lat+'ms';
    lb.className = 'latency-badge' + (lat>500?' bad':lat>200?' warn':'');

    // Account + currency
    const acc = data.account || {};
    const baseCcy = acc.base_currency || 'USD';
    const usdRate = parseFloat(acc.usd_rate) || 1;
    const isUSD = baseCcy === 'USD';
    const ccySym = isUSD ? '$' : baseCcy + ' ';
    document.getElementById('accountId').textContent = (acc.Account || '') + ' (' + baseCcy + ')';

    // Hero NLV — show in base currency + USD equivalent
    const nlv = parseFloat(acc.NetLiquidation || 0);
    const heroEl = document.getElementById('heroNLV');
    if(nlv) {
      heroEl.textContent = ccySym + fmt(nlv, 0);
      if(!isUSD && usdRate > 0) {
        heroEl.textContent += '  ($' + fmt(nlv / usdRate, 0) + ')';
      }
    } else { heroEl.textContent = '--'; }

    // Hero daily change — use IB's RealizedPnL (accurate) instead of bot's tracking
    const risk = data.risk || {};
    const dpnl = parseFloat(acc.RealizedPnL_raw || 0);  // IB's actual realized P&L today
    const nlv = parseFloat(acc.NetLiquidation || 0);
    const dpnlPct = nlv > 0 ? (dpnl / nlv * 100) : 0;
    const changeEl = document.getElementById('heroChange');
    if(dpnl !== 0) {
      changeEl.textContent = (dpnl>=0?'+':'')+fmt(dpnl,2)+' ('+(dpnlPct>=0?'+':'')+dpnlPct.toFixed(2)+'%)';
      changeEl.className = 'hero-change '+(dpnl>=0?'val-pos':'val-neg');
    }

    // Stat cards — show base currency values (use IB realized P&L)
    updateCard('cardDailyPnl', (dpnl>=0?'+':'')+fmt(dpnl,2), pnlCls(dpnl));
    const upnl = parseFloat(acc.UnrealizedPnL_raw||0);
    updateCard('cardUPnL', ccySym+(upnl>=0?'+':'')+fmt(upnl,2), pnlCls(upnl));
    updateCard('cardMargin', ccySym+fmt(acc.InitMarginReq,0), 'val-neutral');
    updateCard('cardBuyPow', ccySym+fmt(acc.BuyingPower || acc.AvailableFunds,0), 'val-neutral');
    updateCard('cardConsecLoss', ''+(risk.consecutive_losses||0), (risk.consecutive_losses||0)>=3?'val-warn':'val-neutral');

    // Risk panel
    updateRiskGauges(risk, acc);

    // Status lights
    document.getElementById('cbDot').className = 'status-dot-sm'+(risk.circuit_broken?' triggered':'');
    document.getElementById('ksDot').className = 'status-dot-sm'+(risk.killed?' triggered':'');

    // Loss dots
    const cl = risk.consecutive_losses || 0;
    let dots = '';
    for(let i=0;i<5;i++) dots += '<div class="loss-dot'+(i<cl?' filled':'')+'"></div>';
    document.getElementById('lossDots').innerHTML = dots;

    // Equity chart
    updateEquityChart(data.equity_history);

    // Ticker
    updateTicker(data.market_data || {}, data.positions);

    // Market grid
    updateMarketGrid(data.market_data||{}, data.sparklines||{}, data.positions, data.news_blocked_pairs);

    // Positions
    renderPositions(data.positions, ccySym, usdRate);

    // Activity feed
    renderActivityFeed(data.fills, data.orders);

    // News
    const blockedPairs = data.news_blocked_pairs || [];
    const newsAlert = document.getElementById('newsAlert');
    if(blockedPairs.length > 0) {
      newsAlert.classList.add('active');
      document.getElementById('newsAlertPairs').textContent = 'News blackout: '+blockedPairs.join(', ');
    } else { newsAlert.classList.remove('active'); }
    renderNews(data.news_events||[], blockedPairs);

  } catch(err) {
    document.getElementById('connDot').classList.remove('live');
    document.getElementById('connText').textContent = 'Error';
  }
}

// ── Init ──
initEquityChart();
initRiskGauges();
buildMarketGrid();
fetchAndUpdate();
setInterval(fetchAndUpdate, 10000);
})();
</script>
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser(description="ICT Trading Bot Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="IB Gateway host")
    parser.add_argument("--port", type=int, default=4002, help="IB Gateway port")
    parser.add_argument("--client-id", type=int, default=99, help="IB client ID")
    parser.add_argument("--web-port", type=int, default=8080, help="Dashboard web port")
    parser.add_argument("--account", default="U24347050", help="IB account ID to monitor")
    args = parser.parse_args()

    poller = threading.Thread(
        target=run_ib_poller,
        args=(args.host, args.port, args.client_id, args.account),
        daemon=True,
    )
    poller.start()
    logger.info("Starting dashboard at http://localhost:%d", args.web_port)
    app.run(host="0.0.0.0", port=args.web_port, debug=False)


if __name__ == "__main__":
    main()
