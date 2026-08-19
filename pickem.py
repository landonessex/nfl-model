#!/usr/bin/env python3
"""
================================================================================
 PICK'EM POOL OPTIMISER  -  straight pick'em, large field
================================================================================
 Companion to nfl_model.py. That file answers "who wins this game?". This one
 answers the question that actually decides a 150-person pool, which is
 different and less obvious:

     Which games should I pick DIFFERENTLY from everyone else?

 THE CORE IDEA

 Games where the whole pool picks the same side are irrelevant to your standing.
 Everyone gains together or loses together; the leaderboard doesn't move. Your
 finishing position is decided entirely by the handful of games where you
 deviate from the field.

 That reframes everything. You are not trying to maximise correct picks. You are
 trying to maximise the probability of finishing first, and those are different
 objectives whenever the field is large.

 THE EXPLOITABLE PATTERN

 The public picks favourites more often than favourites win. A 3-point home
 favourite wins about 58% of the time; roughly 67% of a public pool picks it.
 A 7-point favourite wins about 70%; about 85% of the pool takes it. That gap
 is systematic, it is large, and it means the contrarian value in a pick'em pool
 sits almost entirely on underdogs.

 SEASON AND WEEK PULL OPPOSITE WAYS

 Over 270 picks, skill compounds and variance averages out - the season-long
 objective is close to "maximise expected correct", with a light tilt.
 Over 15 picks, the weekly winner needs 13 or 14 and you cannot get there by
 picking what everyone else picked. Weekly wants variance; season wants accuracy.
 Running both objectives and showing you the trade is most of what this file does.

 USAGE
   python pickem.py demo
   python pickem.py optimise --slate week_picks.csv --mode both
   python pickem.py calibrate --history field_history.csv

 SLATE SCHEMA (only home_team and away_team are required)
   home_team, away_team, spread_home, public_home_pct, model_home_prob
   - spread_home: -3 means the home team gives 3
   - public_home_pct: 0-1. Leave blank and it is estimated from the spread.
   - model_home_prob: leave blank and it is taken from the ratings file.
================================================================================
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required:  pip install pandas")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass
class PoolConfig:
    n_entrants: int = 150            # including you
    weeks_remaining: int = 18

    # How the field behaves. chalk_spread is the standard deviation of how
    # chalky each entrant is: 0 would make every opponent identical to the
    # public average, which no real pool looks like.
    chalk_spread: float = 0.38
    max_pick_rate: float = 0.985

    # A real pool is not 149 shades of the same person. A large bloc submits the
    # favourite in every game, week after week, and a small group is genuinely
    # good. Without these two, a chalk entry looks unbeatable over a season
    # because it is the only entry with no noise in it.
    chalk_bloc_frac: float = 0.22    # pick every favourite, deterministically
    sharp_frac: float = 0.08         # about as good as your model

    # Public pick rate as a function of the spread. Calibrated so a 3-point
    # favourite draws about 67% of picks and a 7-point favourite about 85%.
    # Replace with your own pool's numbers once you have a few weeks logged.
    public_beta: float = 0.236

    # Simulation
    n_sims: int = 20000
    opt_sims: int = 6000             # smaller during the search, for speed
    seed: int = 20260909

    # How much you care about each prize. Both add to 1.
    w_weekly: float = 0.5
    w_season: float = 0.5


CFG = PoolConfig()


# ==============================================================================
# THE FIELD
# ==============================================================================

def public_from_spread(spread_home: float, cfg: PoolConfig = CFG) -> float:
    """
    Estimated share of the pool picking the home team.

    A logistic in the spread. Crude, but it is right where it matters: it
    reproduces the public's well-documented over-backing of favourites, which
    is the whole source of contrarian value.
    """
    if spread_home is None or (isinstance(spread_home, float) and math.isnan(spread_home)):
        return 0.5
    return 1.0 / (1.0 + math.exp(-cfg.public_beta * (-spread_home)))


class Field:
    """
    The other 149 entrants, with fixed identities for the whole season.

    Three kinds of people, because that is what a pick'em pool contains:
      - a chalk bloc who submit the favourite in every game, every week
      - a small sharp group whose picks look like a decent model's
      - everyone else, leaning chalky or contrarian by temperament, with
        week-to-week noise on top
    """

    def __init__(self, n_opponents, rng, cfg=CFG):
        self.n = n_opponents
        self.cfg = cfg
        self.z = rng.normal(0, cfg.chalk_spread, (n_opponents, 1))
        kind = rng.random(n_opponents)
        self.is_chalk = kind < cfg.chalk_bloc_frac
        self.is_sharp = ((kind >= cfg.chalk_bloc_frac) &
                         (kind < cfg.chalk_bloc_frac + cfg.sharp_frac))

    def week(self, public, p_home, rng):
        """One week of picks. True = took the home team."""
        lean = 0.5 + (public[None, :] - 0.5) * (1.0 + self.z)
        lean = np.clip(lean, 1 - self.cfg.max_pick_rate, self.cfg.max_pick_rate)
        picks = rng.random((self.n, len(public))) < lean
        picks[self.is_chalk] = (public > 0.5)[None, :]
        picks[self.is_sharp] = (p_home > 0.5)[None, :]
        return picks


# ==============================================================================
# SCORING A PICK SET AGAINST THE POOL
# ==============================================================================

def evaluate_week(picks: np.ndarray, p_home: np.ndarray, public: np.ndarray,
                  cfg: PoolConfig = CFG, n_sims: int | None = None,
                  seed: int | None = None) -> dict:
    """
    One week, simulated over both unknowns: how the games go, and how the field
    picked. Ties are split rather than ignored - at the top of a 150-person pool
    a tie is common, and a shared prize is worth exactly its share.
    """
    n_sims = n_sims or cfg.n_sims
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    n_opp, g = cfg.n_entrants - 1, len(p_home)

    wins = 0.0
    scores = np.empty(n_sims)
    done = 0
    while done < n_sims:
        s_ = min(600, n_sims - done)
        outcomes = rng.random((s_, g)) < p_home[None, :]
        you = (outcomes == picks[None, :]).sum(axis=1).astype(float)
        field = Field(n_opp, rng, cfg).week(public, p_home, rng)
        opp = (outcomes[:, None, :] == field[None, :, :]).sum(axis=2)

        best = opp.max(axis=1)
        n_tied = (opp == you[:, None]).sum(axis=1)
        wins += (you > best).sum() + ((you == best) / (n_tied + 1.0)).sum()
        scores[done:done + s_] = you
        done += s_

    return {"p_win": wins / n_sims,
            "exp_correct": float(scores.mean()),
            "sd_correct": float(scores.std())}


def evaluate_season(picks: np.ndarray, p_home: np.ndarray, public: np.ndarray,
                    cfg: PoolConfig = CFG, n_sims: int = 3000,
                    seed: int | None = None) -> dict:
    """
    The whole season, with this week's picks played now and the same policy
    repeated for the weeks remaining. Opponent styles are fixed for the season.

    This is an approximation - future weeks reuse this week's shape - but it
    captures the thing that matters: over 270 picks the field's spread widens,
    the chalk bloc stops being a wall of ties, and one bold week barely moves you.
    """
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    n_opp, g = cfg.n_entrants - 1, len(p_home)
    weeks = max(1, cfg.weeks_remaining)

    wins = 0.0
    totals = np.empty(n_sims)
    done = 0
    while done < n_sims:
        s_ = min(200, n_sims - done)
        field_obj = Field(n_opp, rng, cfg)
        you_tot = np.zeros(s_)
        opp_tot = np.zeros((s_, n_opp))
        for w in range(weeks):
            outcomes = rng.random((s_, g)) < p_home[None, :]
            you_tot += (outcomes == picks[None, :]).sum(axis=1)
            field = field_obj.week(public, p_home, rng)
            opp_tot += (outcomes[:, None, :] == field[None, :, :]).sum(axis=2)

        best = opp_tot.max(axis=1)
        n_tied = (opp_tot == you_tot[:, None]).sum(axis=1)
        wins += (you_tot > best).sum() + ((you_tot == best) / (n_tied + 1.0)).sum()
        totals[done:done + s_] = you_tot
        done += s_

    return {"p_win": wins / n_sims,
            "exp_correct": float(totals.mean()),
            "sd_correct": float(totals.std())}


# ==============================================================================
# CHOOSING THE PICKS
# ==============================================================================

def deviation_ladder(p_home: np.ndarray, public: np.ndarray, cfg: PoolConfig = CFG,
                     max_dev: int = 6, verbose: bool = False) -> list[np.ndarray]:
    """
    Build a nested sequence of pick sets: the best 0-deviation card, the best
    1-deviation card, and so on.

    "Deviation" means picking against the field's majority. Each step adds the
    single flip that most improves weekly win probability, so the k-th entry is
    a sensible k-contrarian card rather than an exhaustive optimum. Exhaustive
    would be 2^15 evaluations for a benefit you cannot measure through the noise.

    Candidates share a random seed. Two 6,000-sim estimates differ by more than
    the effect being measured otherwise, and the search wanders instead of climbing.
    """
    chalk = public > 0.5
    picks = chalk.copy()
    ladder = [picks.copy()]
    used = set()

    for k in range(max_dev):
        base, _ = _score(picks, p_home, public, cfg)
        best_gain, best_i = -1e9, None
        for i in range(len(p_home)):
            if i in used:
                continue
            trial = picks.copy()
            trial[i] = ~trial[i]
            val, _ = _score(trial, p_home, public, cfg)
            if val - base > best_gain:
                best_gain, best_i = val - base, i
        if best_i is None:
            break
        picks[best_i] = ~picks[best_i]
        used.add(best_i)
        ladder.append(picks.copy())
        if verbose:
            print(f"  deviation {k+1}: game {best_i}, weekly win {best_gain:+.4%}")
    return ladder


def _score(pk, p_home, public, cfg):
    r = evaluate_week(pk, p_home, public, cfg, cfg.opt_sims, seed=cfg.seed)
    return r["p_win"], r


def frontier(ladder, p_home, public, cfg: PoolConfig = CFG,
             weekly_prize: float = 50.0, season_prize: float = 500.0) -> pd.DataFrame:
    """
    Score every rung of the ladder on both prizes, in money.

    This is the only honest way to blend the two objectives. A weekly win and a
    season win are not the same unit, and no abstract weighting makes them one.
    Expected winnings does, once you say what each pays.
    """
    rows = []
    for k, pk in enumerate(ladder):
        w = evaluate_week(pk, p_home, public, cfg, cfg.n_sims)
        s = evaluate_season(pk, p_home, public, cfg, n_sims=2500)
        ev = weekly_prize * w["p_win"] * cfg.weeks_remaining + season_prize * s["p_win"]
        rows.append({
            "deviations": k,
            "exp_correct": round(w["exp_correct"], 2),
            "P(win week)": round(w["p_win"], 5),
            "P(win season)": round(s["p_win"], 5),
            "EV_$": round(ev, 2),
        })
    return pd.DataFrame(rows)


# ==============================================================================
# SLATE HANDLING
# ==============================================================================

def load_slate(path: str, ratings_json: str | None = None,
               cfg: PoolConfig = CFG) -> pd.DataFrame:
    df = pd.read_csv(path)
    for c in ("home_team", "away_team"):
        df[c] = df[c].astype(str).str.upper().str.strip()
    for c in ("spread_home", "moneyline_home", "public_home_pct", "model_home_prob"):
        if c not in df.columns:
            df[c] = np.nan

    # A moneyline gives the win probability directly, with no round-trip
    # through the spread-to-probability approximation - prefer it when present.
    def market_prob(row):
        if pd.notna(row.get("moneyline_home")):
            return _prob_from_ml(row["moneyline_home"])
        return _prob_from_spread(row.get("spread_home"))

    need_model = df["model_home_prob"].isna()
    if need_model.any():
        if ratings_json and os.path.exists(ratings_json):
            df.loc[need_model, "model_home_prob"] = [
                _prob_from_ratings(r.home_team, r.away_team, r.spread_home, ratings_json)
                for r in df[need_model].itertuples()
            ]
        else:
            df.loc[need_model, "model_home_prob"] = [
                market_prob(r) for _, r in df.loc[need_model].iterrows()
            ]

    # Fill a display spread from the moneyline where only the moneyline was given,
    # so downstream reporting always has one to show.
    need_spread = df["spread_home"].isna() & df["moneyline_home"].notna()
    if need_spread.any():
        df.loc[need_spread, "spread_home"] = [
            _spread_from_prob(_prob_from_ml(m)) for m in df.loc[need_spread, "moneyline_home"]
        ]

    pub = df["public_home_pct"].isna()
    df.loc[pub, "public_home_pct"] = [
        public_from_spread(s, cfg) for s in df.loc[pub, "spread_home"]
    ]
    return df


def _prob_from_spread(spread_home) -> float:
    """Win probability implied by a point spread. Roughly a normal on a 13.5 sd."""
    if spread_home is None or (isinstance(spread_home, float) and math.isnan(spread_home)):
        return 0.5
    z = (-spread_home) / 13.5
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def _prob_from_ml(odds) -> float:
    """Raw implied win probability from a single American price (vig still in it)."""
    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return 0.5
    return (-odds) / (-odds + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def _spread_from_prob(p: float) -> float:
    """Invert _prob_from_spread by bisection, for display when only a moneyline
    was given (Python's stdlib has no erfinv)."""
    p = min(max(p, 0.001), 0.999)
    lo, hi = -30.0, 30.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if _prob_from_spread(mid) > p:
            lo = mid
        else:
            hi = mid
    return round((lo + hi) / 2, 1)


def _prob_from_ratings(home, away, spread, ratings_json) -> float:
    """Run the full possession simulation, if nfl_model and ratings are available."""
    try:
        import json
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import nfl_model as M
        data = json.load(open(ratings_json))
        t = data["teams"]
        if home not in t or away not in t:
            return _prob_from_spread(spread)
        L = data["league"]
        mu_h = L["ppg"] * t[home]["att"] * t[away]["def"] + data["params"]["hfa_points"] / 2
        mu_a = L["ppg"] * t[away]["att"] * t[home]["def"] - data["params"]["hfa_points"] / 2
        rng = np.random.default_rng(7)
        cfg = M.Config(n_sims=12000)
        hp, ap = M.simulate_drives(mu_h, mu_a, cfg, rng)
        m = hp - ap
        return float((m > 0).mean() + 0.5 * (m == 0).mean())
    except Exception:
        return _prob_from_spread(spread)


# ==============================================================================
# REPORTING
# ==============================================================================

def report(df: pd.DataFrame, picks: np.ndarray, res: dict, cfg: PoolConfig = CFG) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(df.itertuples()):
        took_home = bool(picks[i])
        side = r.home_team if took_home else r.away_team
        my_p = r.model_home_prob if took_home else 1 - r.model_home_prob
        pub = r.public_home_pct if took_home else 1 - r.public_home_pct
        chalk_side = r.home_team if r.public_home_pct >= 0.5 else r.away_team
        rows.append({
            "game": f"{r.away_team} @ {r.home_team}",
            "spread": r.spread_home,
            "pick": side,
            "model_%": round(100 * my_p, 1),
            "field_%": round(100 * pub, 1),
            "leverage": round(100 * (my_p - pub), 1),
            "against_field": "YES" if side != chalk_side else "",
        })
    out = pd.DataFrame(rows)
    return out


# ==============================================================================
# CALIBRATION FROM YOUR OWN POOL
# ==============================================================================

def calibrate(path: str) -> None:
    """
    Fit public_beta to your actual pool.

    You can see the field's picks after the deadline. Log them - week, spread,
    and the share of the pool that took the home team - and this recovers how
    your 150 people behave, which is more useful than any national average.
    """
    df = pd.read_csv(path)
    need = {"spread_home", "public_home_pct"}
    if not need.issubset(df.columns):
        sys.exit(f"history file needs columns: {sorted(need)}")
    d = df.dropna(subset=["spread_home", "public_home_pct"]).copy()
    d = d[(d["public_home_pct"] > 0.01) & (d["public_home_pct"] < 0.99)]
    if len(d) < 20:
        print(f"Only {len(d)} usable rows. Keep logging - 60+ gives a stable fit.")
    y = np.log(d["public_home_pct"] / (1 - d["public_home_pct"]))
    x = -d["spread_home"].to_numpy(dtype=float)
    beta = float((x @ y) / (x @ x))
    print(f"\n=== YOUR POOL ===")
    print(f"  games logged      {len(d)}")
    print(f"  public_beta       {beta:.4f}   (default {CFG.public_beta})")
    print(f"  3-pt favourite    {100/(1+math.exp(-beta*3)):.1f}% of the pool takes it")
    print(f"  7-pt favourite    {100/(1+math.exp(-beta*7)):.1f}%")
    print(f"  10-pt favourite   {100/(1+math.exp(-beta*10)):.1f}%")
    if beta > CFG.public_beta * 1.1:
        print("\n  Your pool is chalkier than average. Good news: more contrarian value.")
    elif beta < CFG.public_beta * 0.9:
        print("\n  Your pool is sharper than average. Deviations are worth less here;")
        print("  lean closer to your model's straight favourites.")
    print(f"\n  Set public_beta = {beta:.4f} in PoolConfig.")


# ==============================================================================
# CLI
# ==============================================================================

def demo_slate() -> pd.DataFrame:
    """A plausible 15-game week, with a couple of live underdogs in it."""
    data = [
        ("KC", "BUF", -2.5), ("SF", "SEA", -3.0), ("PHI", "DAL", -6.5),
        ("BAL", "CIN", -4.5), ("DET", "GB", -3.5), ("HOU", "IND", -5.5),
        ("MIA", "NYJ", -1.5), ("TB", "NO", -2.5), ("LAC", "DEN", -3.0),
        ("MIN", "CHI", -6.0), ("ATL", "CAR", -7.5), ("PIT", "CLE", -2.0),
        ("LAR", "ARI", -4.0), ("NE", "TEN", -9.5), ("JAX", "LV", -3.5),
    ]
    return pd.DataFrame(
        [{"home_team": h, "away_team": a, "spread_home": s,
          "public_home_pct": np.nan, "model_home_prob": np.nan} for h, a, s in data])


def cmd_optimise(args):
    cfg = PoolConfig(n_entrants=args.entrants, weeks_remaining=args.weeks)

    if args.demo or not args.slate:
        df = demo_slate()
        df["public_home_pct"] = [public_from_spread(s, cfg) for s in df["spread_home"]]
        # Your model agreeing with the spread on every game would make the demo
        # strategically empty, so a few games carry a real disagreement.
        edges = [0, 0, .06, 0, 0, -.05, 0, .07, 0, 0, -.04, 0, .05, 0, 0]
        df["model_home_prob"] = [
            min(0.97, max(0.03, _prob_from_spread(sp) + e))
            for sp, e in zip(df["spread_home"], edges)]
    else:
        df = load_slate(args.slate, args.ratings, cfg)

    p = df["model_home_prob"].to_numpy(dtype=float)
    q = df["public_home_pct"].to_numpy(dtype=float)

    print(f"Building the deviation ladder: {len(df)} games, "
          f"{cfg.n_entrants} entrants, {cfg.weeks_remaining} weeks left...")
    ladder = deviation_ladder(p, q, cfg, max_dev=args.max_dev, verbose=args.verbose)
    fr = frontier(ladder, p, q, cfg, args.weekly_prize, args.season_prize)

    print("\n=== HOW MANY GAMES TO PICK AGAINST THE FIELD ===")
    print(fr.to_string(index=False))
    best_k = int(fr.loc[fr["EV_$"].idxmax(), "deviations"])
    print(f"\n  Best expected return: {best_k} contrarian pick"
          f"{'' if best_k == 1 else 's'}"
          f"  (weekly pays ${args.weekly_prize:.0f}, season pays ${args.season_prize:.0f})")
    print(f"  Random entry wins {1/cfg.n_entrants:.2%} of the time; "
          f"a pure chalk card wins the week {fr.loc[0, 'P(win week)']:.2%}.")
    print("  Chalk does worse than random because thirty other people submit the")
    print("  identical card and you split the prize with all of them.")

    k = args.deviations if args.deviations is not None else best_k
    picks = ladder[min(k, len(ladder) - 1)]
    table = report(df, picks, {}, cfg)
    print(f"\n=== YOUR CARD  ({k} against the field) ===")
    print(table.to_string(index=False))
    print("\n  'leverage' is your probability minus the field's pick rate on the same")
    print("  side. Games where everyone agrees cannot move you up the leaderboard,")
    print("  whichever way they land. Only the YES rows can.")

    if args.out:
        table.to_csv(args.out, index=False)
        fr.to_csv(args.out.replace(".csv", "_frontier.csv"), index=False)
        print(f"\nSaved -> {args.out}")


def cmd_calibrate(args):
    calibrate(args.history)


def cmd_demo(args):
    args.demo = True
    args.slate = None
    args.out = None
    cmd_optimise(args)


def main():
    p = argparse.ArgumentParser(description="Pick'em pool optimiser")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("optimise", cmd_optimise), ("calibrate", cmd_calibrate),
                     ("demo", cmd_demo)):
        s = sub.add_parser(name)
        s.add_argument("--slate", default=None)
        s.add_argument("--ratings", default="docs/data/model.json")
        s.add_argument("--history", default="data/field_history.csv")
        s.add_argument("--entrants", type=int, default=150)
        s.add_argument("--weeks", type=int, default=18)
        s.add_argument("--max-dev", type=int, default=6,
                       help="how many contrarian picks to consider")
        s.add_argument("--deviations", type=int, default=None,
                       help="force a specific number instead of the best EV")
        s.add_argument("--weekly-prize", type=float, default=50.0)
        s.add_argument("--season-prize", type=float, default=500.0)
        s.add_argument("--out", default=None)
        s.add_argument("--demo", action="store_true")
        s.add_argument("-v", "--verbose", action="store_true")
        s.set_defaults(func=fn)
    a = p.parse_args()
    a.func(a)


if __name__ == "__main__":
    main()
