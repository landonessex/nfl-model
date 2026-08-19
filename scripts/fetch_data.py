#!/usr/bin/env python3
"""
Pulls completed games into data/games.csv, in the schema nfl_model.py expects.

Uses nflverse play-by-play, aggregated to one row per team-game. That is heavier
than scraping a stats table, but it is the only source where passing yards are
reliably NET of sacks, which is what the regression was calibrated on.

  python scripts/fetch_data.py --season 2026 --history 2

Run this locally once before you trust the scheduled job. The nflverse Python
package has been renamed once already; if the import fails, check which package
is current and adjust the two calls below.
"""

import argparse
import os
import sys

import pandas as pd


def load_pbp(years):
    """
    Fetch one season at a time and skip whichever the installed library
    rejects, instead of one batched call that dies if ANY season is invalid.

    This matters specifically for the season that hasn't started yet: nflverse
    packages cap their allowed season range and only extend it once games are
    underway (sometimes not until Week 1 has actually been played), so asking
    for the current year alongside past years can take the whole request down
    even though the past years are perfectly fetchable. A plain `pip install`
    picks up whatever the library's latest allowed range is on each run, so
    this self-heals once the season starts - no code change needed later.
    """
    frames, skipped = [], []
    for yr in years:
        got = None
        try:
            import nflreadpy as nfl
            got = nfl.load_pbp(seasons=[yr]).to_pandas()
        except Exception as e1:
            try:
                import nfl_data_py as nfl
                got = nfl.import_pbp_data([yr], downcast=True)
            except Exception as e2:
                skipped.append((yr, f"nflreadpy: {e1} | nfl_data_py: {e2}"))
                continue
        if got is not None and len(got):
            frames.append(got)
        else:
            skipped.append((yr, "no rows returned"))

    for yr, reason in skipped:
        print(f"  skipping {yr}: {reason}")

    if not frames:
        sys.exit(
            "Could not load play-by-play data for any requested season.\n"
            "Install nflreadpy or nfl_data_py, or drop a hand-built games.csv into data/."
        )
    return pd.concat(frames, ignore_index=True)


def build(years) -> pd.DataFrame:
    pbp = load_pbp(years)

    need = {"game_id", "season", "week", "home_team", "away_team",
            "posteam", "home_score", "away_score", "season_type"}
    missing = need - set(pbp.columns)
    if missing:
        sys.exit(f"Play-by-play is missing columns: {sorted(missing)}")

    pbp = pbp[pbp["season_type"] == "REG"].copy()

    # Net passing yards: passing yards less yards lost to sacks.
    pbp["pass_net"] = (pbp.get("passing_yards", 0).fillna(0)
                       + pbp.get("yards_gained", 0).where(pbp.get("sack", 0) == 1, 0).fillna(0))
    pbp["rush_y"] = pbp.get("rushing_yards", 0).fillna(0)
    pbp["giveaway"] = (pbp.get("interception", 0).fillna(0)
                       + (pbp.get("fumble_lost", 0).fillna(0)))

    off = (pbp.groupby(["game_id", "posteam"], dropna=True)
              .agg(pass_yds=("pass_net", "sum"),
                   rush_yds=("rush_y", "sum"),
                   giveaways=("giveaway", "sum"))
              .reset_index())

    meta = (pbp.groupby("game_id")
               .agg(season=("season", "first"), week=("week", "first"),
                    home_team=("home_team", "first"), away_team=("away_team", "first"),
                    home_score=("home_score", "first"), away_score=("away_score", "first"))
               .reset_index())

    m = meta.merge(off.rename(columns=lambda c: "home_" + c if c not in ("game_id", "posteam") else c),
                   left_on=["game_id", "home_team"], right_on=["game_id", "posteam"], how="left")
    m = m.drop(columns=["posteam"])
    m = m.merge(off.rename(columns=lambda c: "away_" + c if c not in ("game_id", "posteam") else c),
                left_on=["game_id", "away_team"], right_on=["game_id", "posteam"], how="left")
    m = m.drop(columns=["posteam"])

    m["neutral"] = 0
    cols = ["season", "week", "home_team", "away_team", "home_score", "away_score",
            "home_pass_yds", "away_pass_yds", "home_rush_yds", "away_rush_yds",
            "home_giveaways", "away_giveaways", "neutral"]
    m = m.dropna(subset=["home_score", "away_score"])
    for c in cols[4:12]:
        m[c] = m[c].fillna(0).round(0).astype(int)
    return m[cols].sort_values(["season", "week"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--history", type=int, default=2,
                    help="how many prior seasons to include")
    ap.add_argument("--out", default="data/games.csv")
    a = ap.parse_args()

    years = list(range(a.season - a.history, a.season + 1))
    df = build(years)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"Wrote {a.out}: {len(df)} games, seasons {years[0]}-{years[-1]}")


if __name__ == "__main__":
    main()
