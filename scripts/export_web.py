#!/usr/bin/env python3
"""
Fits the model and writes docs/data/model.json for the static site.

Run locally, or let .github/workflows/update-model.yml run it every Tuesday.
The site does its own simulation in the browser - this only ships the ratings
and the parameters the browser needs.

  python scripts/export_web.py --games data/games.csv --week 3
  python scripts/export_web.py --placeholder        # synthetic, for first deploy
"""

import argparse
import json
import os
import sys
from dataclasses import asdict
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import nfl_model as M


def build(games_csv=None, priors_csv=None, week=1, placeholder=False, slate_csv=None):
    df = M.make_demo_games(n_seasons=2) if placeholder else M.load_games(games_csv)
    long = M.to_long(df)
    priors = None
    if priors_csv and os.path.exists(priors_csv):
        import pandas as pd
        p = pd.read_csv(priors_csv)
        priors = dict(zip(p["team"].str.upper(), p["prior_rating"].astype(float)))

    r = M.fit_ratings(long, M.CFG, priors)
    bm = M.fit_box_model(long, M.CFG)

    teams = {}
    for t in sorted(r.attack):
        teams[t] = {
            "att": round(r.attack[t], 5),
            "def": round(r.defense[t], 5),
            "passAtt": round(r.pass_att.get(t, 1.0), 5),
            "passDef": round(r.pass_def.get(t, 1.0), 5),
            "rushAtt": round(r.rush_att.get(t, 1.0), 5),
            "rushDef": round(r.rush_def.get(t, 1.0), 5),
            "give": round(r.give.get(t, r.league["to"]), 4),
            "take": round(r.take.get(t, r.league["to"]), 4),
            "power": round(r.power(t), 3),
            "n": int(r.n_games.get(t, 0)),
        }

    cfg = asdict(M.CFG)
    slate = []
    if slate_csv and os.path.exists(slate_csv):
        import pandas as pd
        s = pd.read_csv(slate_csv)
        slate = json.loads(s.to_json(orient="records"))

    return {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "placeholder": bool(placeholder),
        "week": week,
        "gamesUsed": int(len(df)),
        "league": {k: round(v, 4) for k, v in r.league.items()},
        "teams": teams,
        "coef": {k: round(v, 6) for k, v in bm.coef.items()},
        "intercept": round(bm.intercept, 5),
        "residSd": round(bm.resid_sd, 4),
        "r2": round(bm.r2, 4),
        "statSd": {k: round(v, 3) for k, v in bm.sd.items()},
        "params": {k: cfg[k] for k in (
            "hfa_points", "bye_week_bonus", "short_week_penalty", "long_travel_penalty",
            "drives_per_team", "drive_sd", "env_sigma", "base_td_rate", "base_fg_rate",
            "base_safety_rate", "td_elasticity", "fg_elasticity",
            "two_point_attempt_rate", "two_point_success", "xp_success",
            "w_ratings", "w_boxscore", "market_blend_week1", "market_blend_floor",
            "market_blend_halflife_weeks", "kelly_fraction", "max_stake_pct",
            "min_edge_ml", "min_edge_spread", "min_edge_total", "bankroll",
            "ot_length_drives",
        )},
        "empiricalMargin": M.EMPIRICAL_MARGIN_PCT,
        "slate": slate,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", default="data/games.csv")
    ap.add_argument("--priors", default="data/priors.csv")
    ap.add_argument("--slate", default="data/slate.csv")
    ap.add_argument("--week", type=int, default=1)
    ap.add_argument("--placeholder", action="store_true")
    ap.add_argument("--out", default="docs/data/model.json")
    a = ap.parse_args()

    use_placeholder = a.placeholder or not os.path.exists(a.games)
    if use_placeholder and not a.placeholder:
        print(f"No {a.games} found - writing placeholder ratings instead.")

    payload = build(a.games, a.priors, a.week, use_placeholder, a.slate)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"Wrote {a.out}  ({len(payload['teams'])} teams, "
          f"{payload['gamesUsed']} games, placeholder={payload['placeholder']})")


if __name__ == "__main__":
    main()
