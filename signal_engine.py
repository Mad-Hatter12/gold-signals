"""
Live gold signals for the phone app.  Run:  python3 signal_engine.py      (writes site/ and data/history.json)
Fetches recent gold prices, runs the 3 strategies of the tested combo, writes site/signals.json,
and rebuilds site/index.html (the phone app page). Runs on GitHub Actions every 15 minutes.

  v3 (your rules)         limit order at the neckline / zone edge, risk 2%
  NY opening-range break   market order 10:00-13:00 New York, risk 1%, exit by 16:00 NY
  Session drift            market buy at 18:00 New York, risk 0.5%, exit 08:00 NY next morning
All three only trade with the Monthly/Weekly storyline (flips excepted) and skip FOMC/NFP days.

Prices: Yahoo gold futures (GC=F), ~10-15 min delayed. Futures sit a few dollars above spot XAUUSD,
so the app lets you type your MT5 price and shifts every level by the difference.
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

from news import events
from strategy_v2 import Config2, candidates, daily_bias, ltf_arrays, tday

ROOT = Path(__file__).parent
APP = ROOT / "app"                                         # holds template.html
OUT = ROOT / os.environ.get("OUT_DIR", "site")             # the published website
HIST = ROOT / os.environ.get("HIST_FILE", "data/history.json")

HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<meta name="robots" content="noindex,nofollow">
<meta name="apple-mobile-web-app-capable" content="yes"><meta name="mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-title" content="Gold Signals"><meta http-equiv="refresh" content="300">
<style>:root{padding:env(safe-area-inset-top,0) 0 env(safe-area-inset-bottom,0)}body{margin:0}[hidden]{display:none!important}</style>
</head><body>"""
V3 = Config2(min_rr=1.0, news_days="skip", skip_double=True, no_entry_utc=(17, 18, 19, 20, 21))
RISK = {"v3": 2.0, "orb": 1.0, "drift": 0.5}
NY = "America/New_York"


def fetch(symbol="GC=F"):
    def get(interval, period):
        d = yf.download(symbol, period=period, interval=interval, progress=False, auto_adjust=False)
        d.columns = d.columns.get_level_values(0)
        d = d[["Open", "High", "Low", "Close"]].dropna()
        d.index = d.index.tz_convert("UTC")
        return d
    m15 = get("15m", "60d")
    h1 = get("1h", "730d")
    # drop the bar still forming
    now = pd.Timestamp.now(tz="UTC")
    m15 = m15[m15.index + pd.Timedelta(minutes=15) <= now]
    h1 = h1[h1.index + pd.Timedelta(hours=1) <= now]
    m30 = m15.resample("30min").agg(dict(Open="first", High="max", Low="min", Close="last")).dropna()
    return m15, m30, h1


def daily_atr(h1):
    d = h1.resample("D").agg(dict(High="max", Low="min", Close="last")).dropna()
    tr = pd.concat([d.High - d.Low, (d.High - d.Close.shift()).abs(), (d.Low - d.Close.shift()).abs()], axis=1).max(axis=1)
    return float(tr.rolling(14).mean().iloc[-2])


# Conviction flags, from the 2016-2026 backtest (each held in both 2016-22 and 2023-26):
#   HIGH    v3 rule-8 flips (56% accuracy, +0.56R avg incl. big-target v3 trades) and v3 trades whose target
#           is >= 2.5x the stop  -> size x1.5
#   CAUTION NY opening-range breakouts on Fridays (42% accuracy, -0.07R)  -> half size or skip
#   NORMAL  everything else
CONVICTION = {
    "HIGH": (1.5, "High conviction: this setup type averaged +0.56R per trade over 10 years (56% accuracy), better in both halves of the test."),
    "NORMAL": (1.0, "Normal: standard setup for this strategy."),
    "CAUTION": (0.5, "Caution: Friday opening-range breakouts lost money over 10 years (42% accuracy). Half size, or skip it."),
}


def conviction(strategy, kind, rr, when_ny):
    if strategy.startswith("Your rules") and (kind == "flip" or (rr or 0) >= 2.5):
        return "HIGH"
    if strategy.startswith("NY opening") and when_ny.weekday() == 4:
        return "CAUTION"
    return "NORMAL"


def sig(sid, strategy, side, kind, entry, sl, tp, risk_pct, valid_until, why, setup_kind=""):
    s = 1 if side == "BUY" else -1
    rr = abs(tp - entry) / abs(entry - sl) if tp is not None else None
    level = conviction(strategy, setup_kind, rr, pd.Timestamp.now(tz=NY))
    mult, note = CONVICTION[level]
    return dict(conviction=level, size_mult=mult, conviction_note=note,id=sid, strategy=strategy, side=side, order=kind, entry=round(entry, 2), sl=round(sl, 2),
                tp=round(tp, 2) if tp is not None else None, sl_dist=round(abs(entry - sl), 2),
                tp_dist=round(abs(tp - entry), 2) if tp is not None else None,
                rr=round(abs(tp - entry) / abs(entry - sl), 2) if tp is not None else None,
                risk_pct=risk_pct, valid_until=valid_until, why=why)


def v3_signals(m15, m30, h1, bias_today):
    armed = []
    cands, _ = candidates(h1, {"M15": ltf_arrays(m15, V3), "M30": ltf_arrays(m30, V3)}, V3, armed=armed)
    out, now = [], pd.Timestamp.now(tz="UTC")
    today = tday(now)
    for a in armed:
        side = "BUY" if a["dir"] == "buy" else "SELL"
        s = 1 if side == "BUY" else -1
        risk = (a["entry"] - a["sl"]) * s
        rew = (a["tp"] - a["entry"]) * s
        if risk <= 0 or rew < V3.min_rr * risk:
            continue                                   # target too small vs stop (filter 1)
        if tday(a["since"]) != today:
            continue                                   # rule 9: orders don't carry over to the next day
        if a["kind"] == "double":
            continue
        if a["kind"] != "flip" and bias_today != s:
            continue
        what = {"flip": "Rule 8 flip: price broke through the zone, trade the other way at the zone edge",
                "higher-low": f"{a['tf']} breakout zone + {'higher low' if s == 1 else 'lower high'} pattern; limit at the neckline"}[a["kind"]]
        day_end = (today + pd.Timedelta(hours=22)).isoformat()   # trading day ends ~22:00 UTC
        out.append(sig(f"v3-{a['tf']}-{a['since']:%Y%m%d%H%M}-{side}", "Your rules (v3)", side,
                       f"{side} LIMIT", a["entry"], a["sl"], a["tp"], RISK["v3"], day_end, what, setup_kind=a["kind"]))
    return out


def orb_signal(m15, bias_today):
    nyt = m15.index.tz_convert(NY)
    now_ny = pd.Timestamp.now(tz=NY)
    today = m15[(nyt.date == now_ny.date())]
    tny = today.index.tz_convert(NY)
    mins = tny.hour * 60 + tny.minute
    rng = today[(mins >= 570) & (mins < 600)]
    if len(rng) < 2 or bias_today == 0:
        return []
    hi, lo = rng.High.max(), rng.Low.min()
    after = today[(mins >= 600) & (mins < 780)]
    for t, r in after.iterrows():
        side = "BUY" if r.Close > hi else "SELL" if r.Close < lo else None
        if side is None:
            continue
        s = 1 if side == "BUY" else -1
        if s != bias_today:
            return []                                  # first break went against the storyline: no trade
        brk_ny = t.tz_convert(NY)
        if now_ny - brk_ny > pd.Timedelta(minutes=45):
            return []                                  # signal is stale
        entry = r.Close
        sl = lo if s == 1 else hi
        tp = entry + s * 2 * abs(entry - sl)
        until = now_ny.normalize() + pd.Timedelta(hours=16)
        return [sig(f"orb-{brk_ny:%Y%m%d}", "NY opening-range breakout", side, f"{side} NOW (market)",
                    entry, sl, tp, RISK["orb"], until.tz_convert("UTC").isoformat(),
                    "Price broke the 09:30-10:00 New York range in the trend direction. Close the trade by 16:00 New York if neither SL nor TP is hit.")]
    return []


def drift_signal(m15, h1, bias_today):
    now_ny = pd.Timestamp.now(tz=NY)
    mins = now_ny.hour * 60 + now_ny.minute
    if bias_today != 1 or not (18 * 60 <= mins < 19 * 60 + 30) or now_ny.weekday() == 4:
        return []
    entry = float(m15.Close.iloc[-1])
    sl = entry - 0.6 * daily_atr(h1)
    until = (now_ny.normalize() + pd.Timedelta(days=1, hours=8))
    return [sig(f"drift-{now_ny:%Y%m%d}", "Session drift", "BUY", "BUY NOW (market)", entry, sl, None,
                RISK["drift"], until.tz_convert("UTC").isoformat(),
                "Gold has tended to rise through Asian and London hours in uptrends. No take profit: close the trade at 08:00 New York tomorrow, or at the stop.")]


def main():
    m15, m30, h1 = fetch()
    now = datetime.now(timezone.utc)
    bias, parts = daily_bias(h1, Config2())
    td = tday(pd.Timestamp(now))
    b = int(bias.get(td, bias.iloc[-1]))
    p = parts.loc[td] if td in parts.index else parts.iloc[-1]
    news_today = td in set(tday(events()))
    sigs = [] if news_today else v3_signals(m15, m30, h1, b) + orb_signal(m15, b) + drift_signal(m15, h1, b)

    hist = json.loads(HIST.read_text()) if HIST.exists() else []
    known = {h["id"] for h in hist}
    for s_ in sigs:
        if s_["id"] not in known:
            hist.append(dict(s_, first_seen=now.isoformat()))
    hist = hist[-200:]
    HIST.write_text(json.dumps(hist, indent=1))

    word = {1: "UP", -1: "DOWN", 0: "MIXED"}
    data = dict(generated_at=now.isoformat(), price=round(float(m15.Close.iloc[-1]), 2),
                price_time=m15.index[-1].isoformat(), source="Gold futures GC=F (Yahoo, ~15 min delayed)",
                storyline=dict(monthly=word[int(p["M"])], weekly=word[int(p["W"])], daily=word[int(p["D"])],
                               bias={1: "BUY only", -1: "SELL only", 0: "No trend: stand aside"}[b]),
                news_today=news_today, signals=sigs, history=hist[::-1][:30])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "signals.json").write_text(json.dumps(data, indent=1))
    page = (APP / "template.html").read_text().replace("/*__DATA__*/null", json.dumps(data))
    (OUT / "index.html").write_text(HEAD + page + "</body></html>")
    (OUT / "robots.txt").write_text("User-agent: *\nDisallow: /\n")
    print(json.dumps({k: data[k] for k in ("generated_at", "price", "storyline", "news_today")}), f"{len(sigs)} signal(s)")


if __name__ == "__main__":
    main()
