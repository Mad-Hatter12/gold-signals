"""
High-impact USD news that moves gold: FOMC rate decisions and US jobs report (NFP).

FOMC dates are the Fed's published schedule. NFP is approximated as the first Friday
of the month at 8:30 New York time (a few releases were moved, e.g. the late-2025
government shutdown). CPI is NOT included yet -- for live trading, plug in a real
economic-calendar feed instead of this hand-made list.
"""
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd

NY = ZoneInfo("America/New_York")

FOMC = ["2016-01-27", "2016-03-16", "2016-04-27", "2016-06-15", "2016-07-27", "2016-09-21", "2016-11-02",
        "2016-12-14", "2017-02-01", "2017-03-15", "2017-05-03", "2017-06-14", "2017-07-26", "2017-09-20",
        "2017-11-01", "2017-12-13", "2018-01-31", "2018-03-21", "2018-05-02", "2018-06-13", "2018-08-01",
        "2018-09-26", "2018-11-08", "2018-12-19", "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
        "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11", "2020-01-29", "2020-04-29", "2020-06-10",
        "2020-07-29", "2020-09-16", "2020-11-05", "2020-12-16", "2021-01-27", "2021-03-17", "2021-04-28",
        "2021-06-16", "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15", "2022-01-26", "2022-03-16",
        "2022-05-04", "2022-06-15", "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14", "2023-02-01",
        "2023-03-22", "2023-05-03", "2023-06-14", "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12", "2024-07-31", "2024-09-18", "2024-11-07",
        "2024-12-18", "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18", "2025-07-30", "2025-09-17",
        "2025-10-29", "2025-12-10", "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17", "2026-07-29",
        "2026-09-16", "2026-10-28", "2026-12-09"]


def events():
    ev = [datetime.fromisoformat(d).replace(hour=14, tzinfo=NY) for d in FOMC]
    d = datetime(2016, 1, 1)
    while d.year < 2027:
        first = d + timedelta(days=(4 - d.weekday()) % 7)        # first Friday
        ev.append(first.replace(hour=8, minute=30, tzinfo=NY))
        d = (d.replace(day=28) + timedelta(days=4)).replace(day=1)
    return pd.DatetimeIndex(sorted(e.astimezone(ZoneInfo("UTC")) for e in ev))


def news_mask(index, window_h):
    """True for bars within +/- window_h hours of a news event."""
    idx = index.tz_convert("UTC") if index.tz is not None else index.tz_localize("UTC")
    ev = events().values
    pos = np.searchsorted(ev, idx.values)
    w = np.timedelta64(int(window_h * 3600), "s")
    near = np.zeros(len(idx), bool)
    for off in (-1, 0):
        p = np.clip(pos + off, 0, len(ev) - 1)
        near |= np.abs(idx.values - ev[p]) <= w
    return near
