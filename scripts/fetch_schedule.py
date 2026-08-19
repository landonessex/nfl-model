#!/usr/bin/env python3
"""
Writes docs/data/schedule.json so the pick'em page can pre-fill each week's
matchups. Run it once — the schedule doesn't change — or let the weekly job
do it.

  python scripts/fetch_schedule.py --season 2026

Source is nflverse, which publishes the full season the day the NFL releases it,
including games not yet played. If the package import fails, check which
nflverse Python package is current; it has been renamed once already. You can
also supply your own CSV instead:

  python scripts/fetch_schedule.py --from-csv myschedule.csv

  week,away_team,home_team,neutral
  1,NE,SEA,0
  1,SF,LAR,1
"""

import argparse
import json
import os
import sys

import pandas as pd

ALIASES = {"WSH": "WAS", "JAC": "JAX", "LA": "LAR", "SD": "LAC", "OAK": "LV"}


def norm(t):
    t = str(t).upper().strip()
    return ALIASES.get(t, t)


def from_nflverse(season):
    try:
        import nflreadpy as nfl
        df = nfl.load_schedules(seasons=[season]).to_pandas()
    except Exception as e1:
        try:
            import nfl_data_py as nfl
            df = nfl.import_schedules([season])
        except Exception as e2:
            sys.exit(f"Could not load schedules.\n  nflreadpy: {e1}\n  nfl_data_py: {e2}\n"
                     "Use --from-csv instead.")
    if "game_type" in df.columns:
        df = df[df["game_type"] == "REG"]
    df["neutral"] = 0
    if "location" in df.columns:
        df["neutral"] = (df["location"].astype(str).str.lower() != "home").astype(int)
    return df[["week", "away_team", "home_team", "neutral"]]


def build(df, season):
    weeks = {}
    for _, r in df.sort_values("week").iterrows():
        w = str(int(r["week"]))
        weeks.setdefault(w, []).append({
            "away": norm(r["away_team"]),
            "home": norm(r["home_team"]),
            "neutral": int(r.get("neutral", 0)),
        })
    return {"season": int(season), "weeks": weeks,
            "games": sum(len(v) for v in weeks.values())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--from-csv", default=None)
    ap.add_argument("--out", default="docs/data/schedule.json")
    a = ap.parse_args()

    df = pd.read_csv(a.from_csv) if a.from_csv else from_nflverse(a.season)
    payload = build(df, a.season)

    n = payload["games"]
    if n < 250:
        print(f"WARNING: only {n} games found. A full NFL season is 272 - "
              "check the source before trusting the week fills.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(payload, f, indent=1)
    print(f"Wrote {a.out}: {n} games across {len(payload['weeks'])} weeks")


if __name__ == "__main__":
    main()
