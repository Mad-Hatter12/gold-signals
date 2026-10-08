"""
v2: top-down "storyline" strategy (your rules 1-10).

  1/9  BIAS      Monthly + Weekly trend (Daily must not disagree) decides BUY-only or SELL-only for the day.
                 Re-checked every day from scratch.
  2    SETUP     On H1 and H4: a tight consolidation, then a strong breakout candle in the bias direction.
                 Breakout line = edge of the consolidation. Zone = the breakout candle's body.
  3/4  CONFIRM   Price pulls back into the zone. On M15 (for H1) / M30 (for H4) look for a pattern at the zone:
                 double bottom / higher low / inverse H&S for buys (two lows, the 2nd not lower than the 1st),
                 double top / lower high / H&S for sells. Neckline = the swing between the two lows (highs).
  5    ENTRY     Price breaks the neckline, then pulls back to it -> limit order fills at the neckline.
                 TP = highest (lowest) price reached after the neckline break, before the pullback.
                 SL = beyond the pattern's 2nd low (high).   <- your rules don't specify SL; this is my assumption
  6    OVERLAP   H1 and H4 trades in the same direction within a few hours -> one "strong" trade,
                 conservative entry (better price) and conservative (nearer) TP.
  7    No pattern -> no trade.
  8    FLIP      If price closes through the far side of the zone, the setup flips: trade the other way at the
                 zone edge (no pattern needed), TP = extreme reached before the pullback, SL beyond the zone.
  10   NEWS      Option to skip FOMC / NFP days entirely.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

from news import events


@dataclass
class Config2:
    # 1/9 bias
    ema_m: int = 10
    ema_w: int = 20
    ema_d: int = 50
    require_daily: bool = False      # True = Daily must agree too (not just "not disagree")
    # 2 setup
    cons_bars: int = 10              # consolidation length (HTF bars)
    cons_atr: float = 3.0            # consolidation range <= this x ATR
    brk_body_atr: float = 1.0        # breakout candle body >= this x ATR
    setup_life: int = 30             # HTF bars a zone stays valid
    near_atr: float = 0.5            # pattern lows must be within zone +/- this x HTF ATR
    # 3-5 confirmation
    piv: int = 2                     # LTF swing = lowest/highest of +/- piv bars
    dbl_tol_atr: float = 0.3         # 2nd low may undercut 1st by at most this x LTF ATR
    sl_buf_atr: float = 0.2          # SL buffer beyond the pattern low, x LTF ATR
    min_rr: float = 0.0              # 0 = take every trade your rules give
    inval_atr: float = 0.25          # close beyond far side of zone by this x HTF ATR = break-through (rule 8)
    allow_flip: bool = True
    overlap_h: float = 4.0
    news_days: str = "trade"         # "trade" | "skip"
    timeframes: tuple = ("H1", "H4")
    skip_double: bool = False        # v3: drop double top/bottom confirmations
    no_entry_utc: tuple = ()         # v3: e.g. (17, 18, 19, 20, 21) = no new entries in late NY hours
    # costs / sizing
    cost: float = 0.45               # spread + slippage, USD/oz
    risk_pct: float = 0.01
    start_equity: float = 1000.0


LTF_OF = {"H1": "M15", "H4": "M30"}
TD_OFFSET = pd.Timedelta(hours=2)    # gold "day" rolls over ~22:00 UTC


def tday(ts):
    return (ts + TD_OFFSET).normalize()


def atr(df, n=14):
    pc = df.Close.shift()
    tr = pd.concat([df.High - df.Low, (df.High - pc).abs(), (df.Low - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def resample(h1, rule):
    g = h1.resample(rule)
    out = g.agg(dict(Open="first", High="max", Low="min", Close="last"))
    out["close_time"] = h1.index.to_series().resample(rule).max() + pd.Timedelta(hours=1)
    return out.dropna()


# ---------------- 1/9 bias ----------------
def trend(df, n):
    e = df.Close.ewm(span=n, adjust=False).mean()
    up = (df.Close > e) & (e > e.shift(3))
    dn = (df.Close < e) & (e < e.shift(3))
    return pd.Series(np.where(up, 1, np.where(dn, -1, 0)), index=df.close_time)


def daily_bias(h1, cfg):
    days = pd.DatetimeIndex(sorted(set(tday(h1.index))))
    starts = days - TD_OFFSET                                   # moment each trading day begins
    parts = {}
    for name, rule, n in (("M", "MS", cfg.ema_m), ("W", "W-FRI", cfg.ema_w), ("D", "D", cfg.ema_d)):
        tr = trend(resample(h1, rule), n).sort_index()
        parts[name] = tr.reindex(starts, method="ffill").fillna(0).values   # only bars closed before day start
    m, w, d = parts["M"], parts["W"], parts["D"]
    agree = (m == w) & (m != 0)
    ok_d = (d == m) if cfg.require_daily else (d != -m)
    bias = np.where(agree & ok_d, m, 0)
    return pd.Series(bias, index=days), pd.DataFrame(parts, index=days)


# ---------------- 3-5 confirmation on the lower timeframe ----------------
def pivots(l, k):
    n = len(l.High)
    H, L = l.High.values, l.Low.values
    ph = np.zeros(n, bool); pl = np.zeros(n, bool)
    hi = pd.Series(H).rolling(2 * k + 1, center=True).max().values
    lo = pd.Series(L).rolling(2 * k + 1, center=True).min().values
    ph[:] = H == hi; pl[:] = L == lo
    return ph, pl


def scan(setup, L, cfg):
    """Walk LTF bars for one zone. Returns a trade candidate dict or None."""
    s = 1 if setup["dir"] == "buy" else -1
    # mirror sells into buys: work on negated prices so one code path handles both
    H = L["H"] if s == 1 else -L["L"]
    Lo = L["L"] if s == 1 else -L["H"]
    C = L["C"] if s == 1 else -L["C"]
    ph, pl = (L["ph"], L["pl"]) if s == 1 else (L["pl"], L["ph"])
    zlo, zhi = (setup["zlo"], setup["zhi"]) if s == 1 else (-setup["zhi"], -setup["zlo"])
    A, la = setup["atr"], L["atr"]
    k = cfg.piv
    b0, b1 = setup["b0"], setup["b1"]
    touched, lows, highs = False, [], []
    pat = None            # (neck, low2, low2_bar)
    brk = None            # bar where neckline broke
    peak = -np.inf
    for b in range(b0, b1):
        # rule 8: break through the far side of the zone -> flip
        if C[b] < zlo - cfg.inval_atr * A:
            if cfg.allow_flip and not setup.get("flipped"):
                return dict(flip=True, bar=b)
            return None
        if not touched and Lo[b] <= zhi:
            touched = True
            setup["touched_bar"] = b                   # recorded for the chart view
        j = b - k
        if j >= b0:
            if ph[j]: highs.append(j)
            if pl[j]:
                lows.append(j)
                if pat is None and touched and len(lows) >= 2:
                    l1, l2 = lows[-2], lows[-1]
                    between = [h for h in highs if l1 < h < l2]
                    near = lambda p: zlo - cfg.near_atr * A <= p <= zhi + cfg.near_atr * A
                    if between and near(Lo[l1]) and near(Lo[l2]) and Lo[l2] >= Lo[l1] - cfg.dbl_tol_atr * la[l2]:
                        neck = max(H[h] for h in between)
                        if neck > Lo[l2]:
                            kind = "double" if abs(Lo[l2] - Lo[l1]) <= cfg.dbl_tol_atr * la[l2] else "higher-low"
                            pat = (neck, Lo[l2], l2, kind)
                            setup["pattern"] = dict(l1=int(l1), p1=s * Lo[l1], l2=int(l2), p2=s * Lo[l2],
                                                    neck=s * neck, kind=kind)
        if pat is None:
            continue
        neck, low2, l2, kind = pat
        if brk is None:
            if Lo[b] < low2:                         # pattern failed before breaking out
                pat = None; setup.pop("pattern", None); continue
            if C[b] > neck:
                brk, peak = b, H[b]
                setup["brk_bar"] = b
            continue
        if Lo[b] <= neck and b > brk:               # pullback fills the limit at the neckline
            if tday(L["t"][b]) != tday(L["t"][brk]):  # rule 9: pending orders don't survive the day
                return None
            entry, tp = neck, peak
            sl = low2 - cfg.sl_buf_atr * la[l2]
            risk = entry - sl
            if risk <= 0 or tp <= entry or (tp - entry) < cfg.min_rr * risk:
                return None
            return dict(flip=False, dir=setup["dir"], fill_bar=b, entry=s * entry, tp=s * tp, sl=s * sl,
                        kind=kind, tf=setup["tf"])
        peak = max(peak, H[b])
    if pat is not None and brk is not None:            # live use: limit order waiting at the neckline
        neck, low2, l2, kind = pat
        return dict(armed=True, flip=False, dir=setup["dir"], entry=s * neck, tp=s * peak,
                    sl=s * (low2 - cfg.sl_buf_atr * la[l2]), kind=kind, tf=setup["tf"], since=L["t"][brk])
    return None


def scan_flip(setup, bar, L, cfg):
    """Rule 8: price broke through the zone. Trade the other way at the zone edge, no pattern needed."""
    s = -1 if setup["dir"] == "buy" else 1          # new direction
    H = L["H"] if s == 1 else -L["L"]
    Lo = L["L"] if s == 1 else -L["H"]
    zlo, zhi = (setup["zlo"], setup["zhi"]) if s == 1 else (-setup["zhi"], -setup["zlo"])
    edge = zhi
    peak = H[bar]
    for b in range(bar + 1, setup["b1"]):
        if Lo[b] <= edge:
            sl = zlo - cfg.inval_atr * setup["atr"]
            risk, rew = edge - sl, peak - edge
            if rew <= 0 or rew < cfg.min_rr * risk:
                return None
            return dict(flip=False, dir="buy" if s == 1 else "sell", fill_bar=b, entry=s * edge, tp=s * peak,
                        sl=s * sl, kind="flip", tf=setup["tf"])
        peak = max(peak, H[b])
    if peak > edge:                                    # live use: waiting for the pullback to the zone edge
        return dict(armed=True, flip=False, dir="buy" if s == 1 else "sell", entry=s * edge, tp=s * peak,
                    sl=s * (zlo - cfg.inval_atr * setup["atr"]), kind="flip", tf=setup["tf"], since=L["t"][bar])
    return None


# ---------------- 2 setups on H1 / H4 ----------------
def find_setups(htf, tfname, bias, cfg):
    a = atr(htf).values
    O, H, Lw, C = htf.Open.values, htf.High.values, htf.Low.values, htf.Close.values
    n = cfg.cons_bars
    rhi = pd.Series(H).shift(1).rolling(n).max().values
    rlo = pd.Series(Lw).shift(1).rolling(n).min().values
    days = tday(pd.DatetimeIndex(htf.close_time))
    b = bias.reindex(days).fillna(0).values
    out = []
    for i in range(n + 15, len(htf)):
        if np.isnan(a[i]) or rhi[i] - rlo[i] > cfg.cons_atr * a[i - 1]:
            continue
        body = abs(C[i] - O[i])
        if body < cfg.brk_body_atr * a[i]:
            continue
        if C[i] > rhi[i] and C[i] > O[i] and b[i] == 1:
            out.append(dict(tf=tfname, dir="buy", t=htf.close_time.iloc[i], zlo=O[i], zhi=C[i],
                            line=rhi[i], atr=a[i], life_end=htf.close_time.iloc[min(i + cfg.setup_life, len(htf) - 1)]))
        elif C[i] < rlo[i] and C[i] < O[i] and b[i] == -1:
            out.append(dict(tf=tfname, dir="sell", t=htf.close_time.iloc[i], zlo=C[i], zhi=O[i],
                            line=rlo[i], atr=a[i], life_end=htf.close_time.iloc[min(i + cfg.setup_life, len(htf) - 1)]))
    return out


def ltf_arrays(df, cfg):
    ph, pl = pivots(df, cfg.piv)
    return dict(t=df.index, H=df.High.values, L=df.Low.values, C=df.Close.values, atr=atr(df).values,
                ph=ph, pl=pl)


def candidates(h1, ltfs, cfg, armed=None, trace=None):
    """armed: pass a list to also collect setups still waiting for their fill (live signals).
    trace: pass a list to collect every setup with its outcome (for the chart view)."""
    bias, _ = daily_bias(h1, cfg)
    htfs = {"H1": resample(h1, "1h"), "H4": resample(h1, "4h")}
    cands = []
    for tf in cfg.timeframes:
        L = ltfs[LTF_OF[tf]]
        for st in find_setups(htfs[tf], tf, bias, cfg):
            st["b0"] = int(np.searchsorted(L["t"], st["t"]))
            st["b1"] = int(np.searchsorted(L["t"], st["life_end"]))
            if st["b1"] - st["b0"] < 5:
                continue
            r = scan(st, L, cfg)
            if r and r["flip"]:
                st["flipped"] = True
                st["flip_bar"] = r["bar"]
                r = scan_flip(st, r["bar"], L, cfg)
            if trace is not None:
                trace.append((st, r))
            if r and r.get("armed"):
                if armed is not None and st["b1"] >= len(L["t"]) - 1:   # setup window still open
                    armed.append(dict(r, setup_time=st["t"]))
                continue
            if r:
                r["time"] = L["t"][r["fill_bar"]]
                r["setup_time"] = st["t"]
                # rule 9: on the fill day, the storyline must still agree (flips are the exception)
                if cfg.skip_double and r["kind"] == "double":
                    continue
                if r["time"].hour in cfg.no_entry_utc:
                    continue
                if r["kind"] != "flip" and bias.get(tday(r["time"]), 0) != (1 if r["dir"] == "buy" else -1):
                    continue
                cands.append(r)
    return sorted(cands, key=lambda r: r["time"]), bias


def merge_overlaps(cands, cfg):
    """Rule 6: H1 + H4 trades in the same direction close in time -> one strong trade, conservative levels."""
    out, used = [], set()
    w = pd.Timedelta(hours=cfg.overlap_h)
    for i, a in enumerate(cands):
        if i in used:
            continue
        a = dict(a, strong=False)
        for j in range(i + 1, len(cands)):
            b = cands[j]
            if b["time"] - a["time"] > w:
                break
            if j not in used and b["dir"] == a["dir"] and b["tf"] != a["tf"]:
                pick = min if a["dir"] == "buy" else max
                near = min if a["dir"] == "buy" else max
                a.update(entry=pick(a["entry"], b["entry"]), tp=near(a["tp"], b["tp"]), sl=pick(a["sl"], b["sl"]),
                         time=max(a["time"], b["time"]), strong=True, tf="H1+H4", kind=f'{a["kind"]}+{b["kind"]}')
                used.add(j)
                break
        out.append(a)
    return out


def simulate(cands, m15, cfg):
    """Fill + manage on M15 bars, one position at a time. Worst case if SL and TP hit in the same bar."""
    t, H, L = m15.index, m15.High.values, m15.Low.values
    news_days = set(tday(events()))
    eq, rows, busy_until = cfg.start_equity, [], t[0]
    for c in cands:
        if c["time"] < busy_until:
            continue
        if cfg.news_days == "skip" and tday(c["time"]) in news_days:
            continue
        s = 1 if c["dir"] == "buy" else -1
        entry, tp, sl = c["entry"], c["tp"], c["sl"]
        risk = (entry - sl) * s
        if risk <= 0 or (tp - entry) * s <= 0:
            continue
        i = int(np.searchsorted(t, c["time"]))
        # find the actual fill (merged trades may need a deeper pullback); same trading day only
        day = tday(t[i]) if i < len(t) else None
        while i < len(t) and tday(t[i]) == day and not (L[i] <= entry if s == 1 else H[i] >= entry):
            i += 1
        if i >= len(t) or tday(t[i]) != day:
            continue
        x, exit_px = i, None
        while x < len(t):
            hit_sl = L[x] <= sl if s == 1 else H[x] >= sl
            hit_tp = (H[x] >= tp if s == 1 else L[x] <= tp) and x > i
            if hit_sl: exit_px = sl; break
            if hit_tp: exit_px = tp; break
            x += 1
        if exit_px is None:
            break
        move = (exit_px - entry) * s - cfg.cost
        size = eq * cfg.risk_pct / risk
        eq += move * size
        rows.append(dict(entry_time=t[i], exit_time=t[x], tf=c["tf"], dir=c["dir"], kind=c["kind"],
                         strong=c.get("strong", False), entry=round(entry, 2), sl=round(sl, 2), tp=round(tp, 2),
                         rr=round((tp - entry) * s / risk, 2), R=round(move / risk, 2), equity=round(eq, 2),
                         news_day=tday(t[i]) in news_days))
        busy_until = t[x]
    return pd.DataFrame(rows)
