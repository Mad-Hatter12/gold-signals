# Gold Signals

Personal XAUUSD trade signals. A GitHub Action runs `signal_engine.py` every 15 minutes (Sunday–Friday),
computes signals from three backtested strategies, and publishes the phone page with GitHub Pages.

- `signal_engine.py` – fetches gold prices (Yahoo GC=F, ~15 min delayed), builds `site/index.html`
- `strategy_v2.py` – the top-down rules (storyline → H1/H4 zone → M15/M30 confirmation)
- `news.py` – FOMC / NFP calendar (signals are skipped on those days)
- `app/template.html` – the phone page
- `data/history.json` – log of every signal shown

Not financial advice. Test on a demo account first.
