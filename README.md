# NFL Model — 2026

Opponent-adjusted power ratings, a four-statistic regression, and a possession-level
Monte Carlo. Fitted weekly by GitHub Actions, simulated live in the browser.

- **Site:** `docs/` → GitHub Pages. Two pages: the game model, and a
  pick'em pool optimiser that pre-fills each week's matchups from the baked-in
  2026 schedule — you only type spreads. See [DEPLOY.md](DEPLOY.md).
- **Method:** [MODEL.md](MODEL.md).
- **Engine:** `nfl_model.py` — `ratings`, `predict`, `backtest`, `demo`.
- **Pool strategy:** `pickem.py` + [PICKEM.md](PICKEM.md) — which games to pick
  against the field in a large pick'em league.
- **Spreadsheet:** `NFL_Model_2026.xlsx` — the same maths in live formulas.

```bash
pip install -r requirements.txt
python nfl_model.py demo                 # proves the install, synthetic data
python scripts/export_web.py --placeholder   # writes docs/data/model.json
python pickem.py demo                    # the pool optimiser, on a sample week
```

Built on the methods in Excel LADZ's *Building an NFL Model* series and Austin
Streitmatter's *How I Built a Competitive NFL Prediction Model with Only Five
Statistics* (Samford Sports Analytics). See MODEL.md for what changed and why.

For research and entertainment. Bet only what you can afford to lose, if at all.
