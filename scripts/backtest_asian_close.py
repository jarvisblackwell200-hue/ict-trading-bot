#!/usr/bin/env python3
"""Backtest: Close ALL positions at Asian KZ start (19:00 ET) vs baseline."""
import logging, sys
logging.basicConfig(level=logging.WARNING)
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from ict_bot.backtest.engine import BacktestConfig, simulate_trades
from ict_bot.signals.detector import generate_signals

ASIAN_START = pd.Timestamp("19:00").time()
ASIAN_END = pd.Timestamp("22:00").time()
ALL_PAIRS = ["EUR_USD","GBP_USD","USD_JPY","AUD_USD","USD_CAD","NZD_USD","EUR_GBP"]
data_dir = Path(__file__).resolve().parent.parent / "data" / "processed"

def rr(t):
    return t["pnl_pips"] / t["risk_pips"] if t["risk_pips"] > 0 else 0

all_b = []; all_a = []

for pair in ALL_PAIRS:
    pq = data_dir / f"{pair}_M15.parquet"
    if not pq.exists(): continue
    df = pd.read_parquet(pq)
    if "date" in df.columns: df["date"]=pd.to_datetime(df["date"],utc=True);df=df.set_index("date")
    if df.index.tz is None: df.index=df.index.tz_localize("UTC")
    # Last 90 days
    df = df[df.index >= df.index.max() - pd.Timedelta(days=90)]

    dpq = data_dir / f"{pair}_D.parquet"
    htf = None
    if dpq.exists():
        htf = pd.read_parquet(dpq)
        if "date" in htf.columns: htf["date"]=pd.to_datetime(htf["date"],utc=True);htf=htf.set_index("date")
        if htf.index.tz is None: htf.index=htf.index.tz_localize("UTC")

    pip = 0.01 if "JPY" in pair else 0.0001

    sigs = generate_signals(ohlc=df, htf_ohlc=htf, pair=pair, swing_length=10,
        confluence_threshold=4, min_rr=2.0, sl_buffer_pips=2.0, skip_days=[0,4],
        use_displacement=True, fvg_lookback=16, pullback_window=20, compute_ob=False)
    if not sigs:
        print(f"  {pair}: 0 signals", flush=True); continue

    cfg = BacktestConfig(pair=pair, confluence_threshold=4, min_rr=2.0, sl_buffer_pips=2.0,
        skip_days=[0,4], use_displacement=True, fvg_lookback=16, pullback_window=20,
        compute_ob=False, use_breakeven=True, be_threshold_r=1.5, use_partial_tp=True,
        starting_balance=10000.0, risk_per_trade=0.01, max_bars=200)

    baseline = simulate_trades(sigs, df, cfg)
    if not baseline:
        print(f"  {pair}: 0 trades", flush=True); continue

    # Build asian-close variant
    ac = []
    for t in baseline:
        try: ei = df.index.get_loc(t["entry_time"])
        except: ac.append(t); continue
        found = False
        for j in range(ei+1, len(df)):
            ct = df.index[j]
            if ct >= t["exit_time"]: break
            et = ct.tz_convert("US/Eastern").time()
            if et >= ASIAN_START and et < ASIAN_END and j > ei+1:
                pet = df.index[j-1].tz_convert("US/Eastern").time()
                if pet < ASIAN_START or pet >= ASIAN_END:
                    cp = df.iloc[j]["open"]
                    pp = (cp-t["entry_price"])/pip if t["direction"]=="long" else (t["entry_price"]-cp)/pip
                    m = dict(t)
                    m["exit_price"]=cp; m["exit_time"]=ct; m["exit_reason"]="asian_close"
                    m["pnl_pips"]=pp
                    # Scale pnl_amount proportionally
                    orig_rr = rr(t)
                    new_rr = pp/t["risk_pips"] if t["risk_pips"]>0 else 0
                    if orig_rr != 0:
                        m["pnl_amount"] = t["pnl_amount"] * (new_rr / orig_rr)
                    else:
                        m["pnl_amount"] = 0
                    ac.append(m); found=True; break
        if not found: ac.append(t)

    all_b.extend(baseline); all_a.extend(ac)
    acn = sum(1 for x in ac if x["exit_reason"]=="asian_close")
    bpnl = sum(x["pnl_amount"] for x in baseline)
    apnl = sum(x["pnl_amount"] for x in ac)
    bwr = sum(1 for x in baseline if x["pnl_amount"]>0)/len(baseline)*100
    awr = sum(1 for x in ac if x["pnl_amount"]>0)/len(ac)*100
    print(f"  {pair}: B ${bpnl:+.0f} ({bwr:.0f}% WR, {len(baseline)} trades) | AC ${apnl:+.0f} ({awr:.0f}% WR) | {acn} closed at Asian open", flush=True)

# Overall
print(f"\n{'='*60}")
for lbl, tds in [("BASELINE", all_b), ("ASIAN-CLOSE (19:00 ET)", all_a)]:
    if not tds: continue
    pnl = sum(t["pnl_amount"] for t in tds)
    w = [t for t in tds if t["pnl_amount"]>0]
    l = [t for t in tds if t["pnl_amount"]<=0]
    wr = len(w)/len(tds)*100
    exp = np.mean([rr(t) for t in tds])
    aw = np.mean([rr(t) for t in w]) if w else 0
    al = np.mean([rr(t) for t in l]) if l else 0
    eq=0;pk=0;mdd=0
    for t in tds: eq+=t["pnl_amount"];pk=max(pk,eq);mdd=max(mdd,pk-eq)
    print(f"\n{lbl}:")
    print(f"  Trades: {len(tds)}  WR: {wr:.1f}%  P&L: ${pnl:+,.2f}")
    print(f"  Expectancy: {exp:+.3f}R  Avg W: {aw:+.2f}R  Avg L: {al:+.2f}R")
    print(f"  Max DD: ${mdd:,.2f}")

# Exit reason breakdown
if all_a:
    print(f"\nAsian-close exit reasons:")
    reasons = {}
    for t in all_a:
        r = t["exit_reason"]
        if r not in reasons: reasons[r]={"n":0,"pnl":0,"rrs":[]}
        reasons[r]["n"]+=1; reasons[r]["pnl"]+=t["pnl_amount"]; reasons[r]["rrs"].append(rr(t))
    for r,d in sorted(reasons.items()):
        print(f"  {r:20s}: {d['n']:3d} trades  ${d['pnl']:+10,.2f}  avg {np.mean(d['rrs']):+.2f}R")

# Trades that were closed at asian — what would they have been?
asian_closed = [(b,a) for b,a in zip(all_b,all_a) if a["exit_reason"]=="asian_close"]
if asian_closed:
    print(f"\n{'─'*60}")
    print(f"TRADES CLOSED AT ASIAN OPEN ({len(asian_closed)} trades):")
    print(f"  Original outcome if held:")
    orig_reasons = {}
    for b,a in asian_closed:
        r = b["exit_reason"]
        if r not in orig_reasons: orig_reasons[r]=0
        orig_reasons[r]+=1
    for r,n in sorted(orig_reasons.items()):
        print(f"    {r}: {n}")
    
    orig_pnl = sum(b["pnl_amount"] for b,_ in asian_closed)
    asian_pnl = sum(a["pnl_amount"] for _,a in asian_closed)
    print(f"  Original P&L: ${orig_pnl:+,.2f}")
    print(f"  Asian-close P&L: ${asian_pnl:+,.2f}")
    print(f"  Difference: ${asian_pnl - orig_pnl:+,.2f} ({'BETTER' if asian_pnl > orig_pnl else 'WORSE'})")
