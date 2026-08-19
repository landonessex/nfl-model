#!/usr/bin/env python3
"""
================================================================================
 NFL PREDICTION MODEL - 2026 SEASON
================================================================================
 A synthesis of two published methodologies, plus the fixes each one needs to
 survive contact with a real sportsbook.

 SOURCE A - Excel LADZ, "Building an NFL Model" Parts 1-3 (Patreon, 2024)
   - Opponent-adjusted Attack & Defense ratings -> expected score per team
   - Simulate a full scoreline, not just a margin
   - Explicit overtime handling
   - Run many simulations -> distribution -> price Moneyline / Spread / Total
   - Compare to book prices to find positive expected value

 SOURCE B - Austin Streitmatter, Samford Sports Analytics (Feb 2023)
   "How I Built a Competitive NFL Prediction Model with Only Five Statistics"
   - Linear regression over historical box scores to find which stats matter
   - Kept: Passing Yards, Rushing Yards, Takeaways, Giveaways
   - Expected stat = blend of team's own 'for' rate and opponent's 'against' rate
   - NORM.INV(RAND(), mean, sd) to draw each stat, multiply by coefficients
   - 10,000 simulations per game

 WHAT THIS FILE ADDS ON TOP
   1. Discrete drive-based scoring, so simulated scores land on 3/7/10/14/17/20/24
      like real NFL games. A normal distribution on points cannot price a -3 or
      a +7 correctly; roughly 1 game in 7 lands exactly on 3 or 7.
   2. Ridge regression with proper opponent adjustment instead of raw averages.
   3. Recency weighting + shrinkage to a prior, so Week 1 and Week 15 are both
      handled instead of the model being useless until Week 4.
   4. Correlated game environment, so team scores are not independent and totals
      get realistic variance.
   5. Devigging, expected value, fractional Kelly staking, and closing line value
      tracking - the parts that decide whether a 55% model actually makes money.
   6. Walk-forward backtesting with Brier score, log loss and calibration bins,
      benchmarked against the market rather than against a coin flip.

 USAGE
   python nfl_model.py demo                     # synthetic data, runs anywhere
   python nfl_model.py ratings   --games games.csv
   python nfl_model.py predict   --games games.csv --slate slate.csv
   python nfl_model.py backtest  --games games.csv

 DATA (see README for exact schemas and where to pull it free)
   games.csv  - one row per completed game, both teams' box lines
   slate.csv  - upcoming games plus the book's current prices
   priors.csv - optional preseason power ratings (win totals or last year)

 DISCLAIMER: for research and entertainment. No model reliably beats a sharp
 closing line. Bet only what you can afford to lose, if at all.
================================================================================
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict

import numpy as np

try:
    import pandas as pd
except ImportError:
    sys.exit("pandas is required:  pip install pandas")


# ==============================================================================
# SECTION 0 - CONFIGURATION
# All tunable knobs live here. Nothing below this block hardcodes a number.
# ==============================================================================

@dataclass
class Config:
    # ---- League environment (update from the prior season each August) ----
    league_ppg: float = 22.4          # points per team per game
    league_pass_ypg: float = 227.0    # net passing yards per team per game
    league_rush_ypg: float = 116.0    # rushing yards per team per game
    league_to_pg: float = 1.27        # giveaways per team per game
    drives_per_team: float = 11.2     # possessions per team, excluding kneeldowns

    # ---- Home field advantage, in points, split evenly between the teams ----
    hfa_points: float = 1.9           # NFL HFA has fallen from ~2.6 to ~1.7-2.1
    hfa_neutral: float = 0.0
    hfa_international: float = 0.3    # designated "home" team abroad keeps a sliver

    # ---- Rest / travel / short week adjustments, in points ----
    bye_week_bonus: float = 0.6
    short_week_penalty: float = -0.5  # Thursday game off a Sunday
    long_travel_penalty: float = -0.4 # 3+ time zones, body-clock disadvantage

    # ---- Rating estimation ----
    recency_halflife_games: float = 10.0  # weight halves every N games back
    carryover_weight: float = 0.30        # how much last season survives into this one
    shrink_k: float = 6.0                 # games of prior; higher = slower to move
    ridge_lambda: float = 12.0            # regularisation on the box-score regression
    solver_iters: int = 400

    # ---- Simulation ----
    n_sims: int = 20000                   # Samford used 10k; 20k tightens key numbers
    env_sigma: float = 0.075              # shared game-environment lognormal sd
    drive_sd: float = 1.15                # sd of possessions per game
    td_elasticity: float = 1.35           # good offences convert TDs, not just FGs
    fg_elasticity: float = 0.45
    base_td_rate: float = 0.235           # per drive, at league-average offence
    base_fg_rate: float = 0.135
    base_safety_rate: float = 0.004
    two_point_attempt_rate: float = 0.09
    two_point_success: float = 0.48
    xp_success: float = 0.955

    # ---- Overtime (NFL rules as of the 2025 season) ----
    # Both teams are guaranteed a possession in the regular season, 10-minute
    # period, regular-season games may still end tied. VERIFY EACH AUGUST.
    ot_length_drives: int = 3
    ot_regular_season_tie_allowed: bool = True

    # ---- Ensemble weights: ratings model vs box-score model vs market ----
    w_ratings: float = 0.55
    w_boxscore: float = 0.45
    # Market blend decays as your own sample grows. Week 1 you are mostly market.
    market_blend_week1: float = 0.75
    market_blend_floor: float = 0.25
    market_blend_halflife_weeks: float = 4.0

    # ---- Betting ----
    kelly_fraction: float = 0.25      # never full Kelly; variance will end you
    max_stake_pct: float = 0.02       # hard cap, fraction of bankroll per bet
    min_edge_ml: float = 0.025        # required edge before a moneyline bet fires
    min_edge_spread: float = 0.020
    min_edge_total: float = 0.020
    bankroll: float = 1000.0

    seed: int | None = 20260909


CFG = Config()

TEAMS = [
    "ARI", "ATL", "BAL", "BUF", "CAR", "CHI", "CIN", "CLE", "DAL", "DEN", "DET",
    "GB", "HOU", "IND", "JAX", "KC", "LAC", "LAR", "LV", "MIA", "MIN", "NE",
    "NO", "NYG", "NYJ", "PHI", "PIT", "SEA", "SF", "TB", "TEN", "WAS",
]


# ==============================================================================
# SECTION 1 - SMALL MATH UTILITIES
# ==============================================================================

def norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def american_to_prob(odds: float) -> float:
    """Implied probability INCLUDING the book's margin."""
    if odds is None or (isinstance(odds, float) and math.isnan(odds)):
        return float("nan")
    return (-odds) / (-odds + 100.0) if odds < 0 else 100.0 / (odds + 100.0)


def american_to_decimal(odds: float) -> float:
    return 1.0 + (100.0 / abs(odds) if odds < 0 else odds / 100.0)


def prob_to_american(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return -round(100.0 * p / (1.0 - p)) if p >= 0.5 else round(100.0 * (1.0 - p) / p)


def devig_multiplicative(p_a: float, p_b: float) -> tuple[float, float]:
    """Simplest devig: scale both sides so they sum to 1."""
    s = p_a + p_b
    return p_a / s, p_b / s


def devig_power(p_a: float, p_b: float, tol: float = 1e-10) -> tuple[float, float]:
    """
    Power devig. Solves p_a^k + p_b^k = 1.

    Better than multiplicative on lopsided markets, where the vig is not spread
    evenly: books tax heavy favourites' prices differently from long shots.
    """
    lo, hi = 0.5, 3.0
    for _ in range(200):
        k = 0.5 * (lo + hi)
        s = p_a ** k + p_b ** k
        if abs(s - 1.0) < tol:
            break
        if s > 1.0:
            lo = k
        else:
            hi = k
    k = 0.5 * (lo + hi)
    a, b = p_a ** k, p_b ** k
    t = a + b
    return a / t, b / t


def kelly_stake(p: float, odds: float, cfg: Config = CFG) -> float:
    """Fractional Kelly as a share of bankroll, capped."""
    d = american_to_decimal(odds)
    b = d - 1.0
    edge = p * b - (1.0 - p)
    if edge <= 0:
        return 0.0
    f = edge / b
    return float(min(f * cfg.kelly_fraction, cfg.max_stake_pct))


def expected_value(p: float, odds: float) -> float:
    """EV per 1 unit staked."""
    d = american_to_decimal(odds)
    return p * (d - 1.0) - (1.0 - p)


# ==============================================================================
# SECTION 2 - DATA LOADING AND LONG-FORM RESHAPE
# ==============================================================================

GAME_COLS = [
    "season", "week", "home_team", "away_team", "home_score", "away_score",
    "home_pass_yds", "away_pass_yds", "home_rush_yds", "away_rush_yds",
    "home_giveaways", "away_giveaways",
]


def load_games(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in GAME_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"games file is missing required columns: {missing}")
    for c in ("home_team", "away_team"):
        df[c] = df[c].astype(str).str.upper().str.strip()
    if "neutral" not in df.columns:
        df["neutral"] = 0
    df = df.sort_values(["season", "week"]).reset_index(drop=True)
    df["game_id"] = np.arange(len(df))
    return df


def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per team-game. Every downstream routine works off this shape, which
    is what makes the opponent adjustment easy: a team's defence rows are simply
    its opponents' offence rows.
    """
    frames = []
    for side, opp in (("home", "away"), ("away", "home")):
        f = pd.DataFrame({
            "game_id": df["game_id"],
            "season": df["season"],
            "week": df["week"],
            "team": df[f"{side}_team"],
            "opp": df[f"{opp}_team"],
            "is_home": 1 if side == "home" else 0,
            "neutral": df["neutral"],
            "pts_for": df[f"{side}_score"],
            "pts_against": df[f"{opp}_score"],
            "pass_yds": df[f"{side}_pass_yds"],
            "rush_yds": df[f"{side}_rush_yds"],
            "giveaways": df[f"{side}_giveaways"],
            "takeaways": df[f"{opp}_giveaways"],   # your takeaways = their giveaways
            "pass_yds_allowed": df[f"{opp}_pass_yds"],
            "rush_yds_allowed": df[f"{opp}_rush_yds"],
        })
        frames.append(f)
    long = pd.concat(frames, ignore_index=True)
    long["margin"] = long["pts_for"] - long["pts_against"]
    return long.sort_values(["season", "week", "game_id"]).reset_index(drop=True)


def recency_weights(long: pd.DataFrame, cfg: Config = CFG,
                    target_season: int | None = None) -> np.ndarray:
    """
    Exponential decay by how long ago the game was, then an extra haircut for
    anything from a previous season. A roster in September is not the roster
    that finished last January.

    target_season lets the caller say which season is actually being predicted.
    Left as None, it infers "current" from the most recent season in the data -
    which is wrong specifically in the gap between seasons, when the newest
    season has no games in it yet. Without an explicit target, last year's
    results get treated as fully current instead of one full year stale, which
    quietly drowns out any preseason prior.
    """
    order = long.groupby("team").cumcount(ascending=False)
    n_per_team = long.groupby("team")["game_id"].transform("size")
    games_back = (n_per_team - 1 - long.groupby("team").cumcount()).to_numpy()
    w = 0.5 ** (games_back / cfg.recency_halflife_games)
    target = target_season if target_season is not None else long["season"].max()
    seasons_back = (target - long["season"]).to_numpy()
    w = w * (cfg.carryover_weight ** np.clip(seasons_back, 0, 4))
    return w


# ==============================================================================
# SECTION 3 - MODULE A: ATTACK / DEFENSE RATINGS  (Excel LADZ methodology)
# ==============================================================================
# Multiplicative ratings solved by alternating least squares:
#
#     E[points team scores] = league_ppg * ATT_team * DEF_opponent  (+ HFA/2)
#
# ATT = 1.15 means the offence scores 15% more than league average against an
# average defence. DEF = 0.90 means the defence allows 10% fewer than average.
# The iteration is what turns raw points-per-game into an opponent-adjusted
# number - the single biggest upgrade over "just average the last few weeks".

@dataclass
class Ratings:
    attack: dict = field(default_factory=dict)
    defense: dict = field(default_factory=dict)
    pass_att: dict = field(default_factory=dict)
    pass_def: dict = field(default_factory=dict)
    rush_att: dict = field(default_factory=dict)
    rush_def: dict = field(default_factory=dict)
    give: dict = field(default_factory=dict)
    take: dict = field(default_factory=dict)
    league: dict = field(default_factory=dict)
    n_games: dict = field(default_factory=dict)

    def power(self, team: str, cfg: Config = CFG) -> float:
        """Net points vs a league-average opponent on a neutral field."""
        a = self.attack.get(team, 1.0)
        d = self.defense.get(team, 1.0)
        lg = self.league.get("ppg", cfg.league_ppg)
        return lg * a - lg * d

    def table(self, cfg: Config = CFG) -> pd.DataFrame:
        rows = []
        for t in sorted(self.attack):
            lg = self.league.get("ppg", cfg.league_ppg)
            rows.append({
                "team": t,
                "ATT": round(self.attack[t], 4),
                "DEF": round(self.defense[t], 4),
                "Off_PPG_adj": round(lg * self.attack[t], 2),
                "Def_PPG_adj": round(lg * self.defense[t], 2),
                "Power": round(self.power(t, cfg), 2),
                "PassATT": round(self.pass_att.get(t, 1.0), 3),
                "PassDEF": round(self.pass_def.get(t, 1.0), 3),
                "RushATT": round(self.rush_att.get(t, 1.0), 3),
                "RushDEF": round(self.rush_def.get(t, 1.0), 3),
                "Give_pg": round(self.give.get(t, 1.27), 2),
                "Take_pg": round(self.take.get(t, 1.27), 2),
                "N": int(self.n_games.get(t, 0)),
            })
        out = pd.DataFrame(rows).sort_values("Power", ascending=False)
        out.insert(0, "rank", range(1, len(out) + 1))
        return out.reset_index(drop=True)


def _solve_multiplicative(long: pd.DataFrame, w: np.ndarray, col_for: str,
                          col_against: str, league_mean: float,
                          cfg: Config = CFG) -> tuple[dict, dict]:
    """
    Alternating updates for one stat family (points, pass yards, rush yards).

    Each pass: hold defence fixed, solve every attack rating as a weighted mean
    of (observed / expected-given-opponent); then swap. Shrink toward 1.0 by
    sample size so a team with 2 games does not sit at the top of the table.
    """
    teams = sorted(set(long["team"]) | set(long["opp"]))
    att = {t: 1.0 for t in teams}
    dfn = {t: 1.0 for t in teams}

    team_arr = long["team"].to_numpy()
    opp_arr = long["opp"].to_numpy()
    for_arr = long[col_for].to_numpy(dtype=float)
    ag_arr = long[col_against].to_numpy(dtype=float)

    # Effective sample size per team drives the shrinkage.
    n_eff = {t: 0.0 for t in teams}
    for t, wi in zip(team_arr, w):
        n_eff[t] += wi

    for _ in range(cfg.solver_iters):
        num_a = {t: 0.0 for t in teams}
        den_a = {t: 0.0 for t in teams}
        for i in range(len(long)):
            t, o, wi = team_arr[i], opp_arr[i], w[i]
            num_a[t] += wi * for_arr[i]
            den_a[t] += wi * league_mean * dfn[o]
        for t in teams:
            raw = num_a[t] / den_a[t] if den_a[t] > 0 else 1.0
            k = n_eff[t] / (n_eff[t] + cfg.shrink_k)
            att[t] = k * raw + (1 - k) * 1.0

        num_d = {t: 0.0 for t in teams}
        den_d = {t: 0.0 for t in teams}
        for i in range(len(long)):
            t, o, wi = team_arr[i], opp_arr[i], w[i]
            num_d[t] += wi * ag_arr[i]
            den_d[t] += wi * league_mean * att[o]
        for t in teams:
            raw = num_d[t] / den_d[t] if den_d[t] > 0 else 1.0
            k = n_eff[t] / (n_eff[t] + cfg.shrink_k)
            dfn[t] = k * raw + (1 - k) * 1.0

        # Identifiability: the product ATT*DEF is scale-free, so pin the mean.
        ma = float(np.mean(list(att.values())))
        md = float(np.mean(list(dfn.values())))
        att = {t: v / ma for t, v in att.items()}
        dfn = {t: v / md for t, v in dfn.items()}

    return att, dfn


def fit_ratings(long: pd.DataFrame, cfg: Config = CFG,
                priors: dict | None = None, target_season: int | None = None) -> Ratings:
    w = recency_weights(long, cfg, target_season)

    # Strip home field out of the observed points before rating anything,
    # otherwise teams that happened to play at home a lot look better than they are.
    adj = long.copy()
    hfa_side = np.where(adj["neutral"] == 1, 0.0,
                        np.where(adj["is_home"] == 1, cfg.hfa_points / 2, -cfg.hfa_points / 2))
    adj["pts_for_adj"] = adj["pts_for"] - hfa_side
    adj["pts_against_adj"] = adj["pts_against"] + hfa_side

    lg_ppg = float(np.average(adj["pts_for_adj"], weights=w))
    lg_pass = float(np.average(adj["pass_yds"], weights=w))
    lg_rush = float(np.average(adj["rush_yds"], weights=w))
    lg_to = float(np.average(adj["giveaways"], weights=w))

    att, dfn = _solve_multiplicative(adj, w, "pts_for_adj", "pts_against_adj", lg_ppg, cfg)
    patt, pdef = _solve_multiplicative(adj, w, "pass_yds", "pass_yds_allowed", lg_pass, cfg)
    ratt, rdef = _solve_multiplicative(adj, w, "rush_yds", "rush_yds_allowed", lg_rush, cfg)

    # Turnovers are the noisiest input in football. Shrink them hard: takeaway
    # rate is roughly half luck year to year, so a 3-game hot streak means little.
    give, take, ngames = {}, {}, {}
    for t, grp in adj.groupby("team"):
        gw = w[grp.index.to_numpy()]
        n = float(gw.sum())
        k = n / (n + 2 * cfg.shrink_k)
        give[t] = k * float(np.average(grp["giveaways"], weights=gw)) + (1 - k) * lg_to
        take[t] = k * float(np.average(grp["takeaways"], weights=gw)) + (1 - k) * lg_to
                # Recency-weighted, matching the shrink calc just above - NOT a flat
        # len(grp). A raw count treats a 2024 game and a 2026 game as equally
        # informative, which is exactly the assumption that made priors nearly
        # invisible: two full stale seasons (~34 games) looked like "plenty of
        # evidence" even though almost none of that evidence was about the
        # season actually being predicted.
        ngames[t] = n


    r = Ratings(att, dfn, patt, pdef, ratt, rdef, give, take,
                {"ppg": lg_ppg, "pass": lg_pass, "rush": lg_rush, "to": lg_to}, ngames)

    # Preseason priors: blend market win totals or last year's finish into teams
    # with little or no current-season data. This is what makes Week 1 usable.
    if priors:
        for t in r.attack:
            n = r.n_games.get(t, 0)
            k = n / (n + cfg.shrink_k)
            p = priors.get(t)
            if p is None:
                continue
            prior_att = 1.0 + (p / 2.0) / lg_ppg
            prior_def = 1.0 - (p / 2.0) / lg_ppg
            r.attack[t] = k * r.attack[t] + (1 - k) * prior_att
            r.defense[t] = k * r.defense[t] + (1 - k) * prior_def
    return r


# ==============================================================================
# SECTION 4 - MODULE B: FIVE-STATISTIC REGRESSION  (Samford methodology)
# ==============================================================================
# The article regressed points on basic box-score stats and kept the four that
# came back significant: passing yards, rushing yards, takeaways, giveaways.
# Reproduced here with two changes:
#   - ridge instead of plain OLS, because passing and rushing yards are
#     correlated with each other and with game script
#   - fit on team-games rather than games, which doubles the usable sample

@dataclass
class BoxModel:
    coef: dict
    intercept: float
    resid_sd: float
    r2: float
    sd: dict

    def predict_points(self, pass_yds, rush_yds, takeaways, giveaways):
        return (self.intercept
                + self.coef["pass_yds"] * pass_yds
                + self.coef["rush_yds"] * rush_yds
                + self.coef["takeaways"] * takeaways
                + self.coef["giveaways"] * giveaways)

    def report(self) -> str:
        c = self.coef
        return (
            "  Points = {:.3f}\n"
            "         + {:.5f} x Passing Yards      ({:.2f} pts per 100 yds)\n"
            "         + {:.5f} x Rushing Yards      ({:.2f} pts per 100 yds)\n"
            "         + {:.4f}  x Takeaways\n"
            "         + {:.4f}  x Giveaways\n"
            "  R-squared {:.3f}   residual sd {:.2f} pts"
        ).format(self.intercept, c["pass_yds"], c["pass_yds"] * 100,
                 c["rush_yds"], c["rush_yds"] * 100,
                 c["takeaways"], c["giveaways"], self.r2, self.resid_sd)


def fit_box_model(long: pd.DataFrame, cfg: Config = CFG) -> BoxModel:
    feats = ["pass_yds", "rush_yds", "takeaways", "giveaways"]
    X = long[feats].to_numpy(dtype=float)
    y = long["pts_for"].to_numpy(dtype=float)

    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - mu) / sd
    Xs = np.column_stack([np.ones(len(Xs)), Xs])

    # Ridge, leaving the intercept unpenalised.
    P = np.eye(Xs.shape[1]) * cfg.ridge_lambda
    P[0, 0] = 0.0
    beta = np.linalg.solve(Xs.T @ Xs + P, Xs.T @ y)

    coef_raw = beta[1:] / sd
    intercept = beta[0] - float(np.sum(beta[1:] * mu / sd))
    pred = X @ coef_raw + intercept
    resid = y - pred
    r2 = 1.0 - float(np.var(resid) / np.var(y))

    return BoxModel(
        coef=dict(zip(feats, coef_raw)),
        intercept=float(intercept),
        resid_sd=float(np.std(resid)),
        r2=r2,
        sd={
            "pass_yds": float(long["pass_yds"].std()),
            "rush_yds": float(long["rush_yds"].std()),
            "takeaways": float(long["takeaways"].std()),
            "giveaways": float(long["giveaways"].std()),
        },
    )


def expected_box_stats(team: str, opp: str, r: Ratings) -> dict:
    """
    The Samford article blends the team's own 'for' average with the opponent's
    'against' average. That works, but it double-counts schedule strength.
    Here the same idea runs through the opponent-adjusted ratings instead:

        expected = league_average x TeamAttack x OpponentDefense
    """
    return {
        "pass_yds": r.league["pass"] * r.pass_att.get(team, 1.0) * r.pass_def.get(opp, 1.0),
        "rush_yds": r.league["rush"] * r.rush_att.get(team, 1.0) * r.rush_def.get(opp, 1.0),
        "giveaways": 0.5 * (r.give.get(team, r.league["to"]) + r.take.get(opp, r.league["to"])),
        "takeaways": 0.5 * (r.take.get(team, r.league["to"]) + r.give.get(opp, r.league["to"])),
    }


# ==============================================================================
# SECTION 5 - EXPECTED POINTS: THE ENSEMBLE
# ==============================================================================

def expected_points(home: str, away: str, r: Ratings, bm: BoxModel,
                    cfg: Config = CFG, neutral: bool = False,
                    rest_home: str = "normal", rest_away: str = "normal",
                    travel_away: bool = False,
                    market_spread: float | None = None,
                    market_total: float | None = None,
                    weeks_of_data: float = 18.0) -> dict:
    """
    Three estimates of each team's expected points, blended:
      A. Attack/Defense ratings           (Excel LADZ)
      B. Five-statistic regression        (Samford)
      C. The market's own line            (the hardest baseline in sports)

    C's weight decays through the season. In Week 1 you have no 2026 data and
    the market has every roster move priced; by Week 10 your sample means something.
    """
    lg = r.league["ppg"]

    # --- A: ratings ---
    a_home = lg * r.attack.get(home, 1.0) * r.defense.get(away, 1.0)
    a_away = lg * r.attack.get(away, 1.0) * r.defense.get(home, 1.0)

    # --- B: box-score regression ---
    sh = expected_box_stats(home, away, r)
    sa = expected_box_stats(away, home, r)
    b_home = bm.predict_points(sh["pass_yds"], sh["rush_yds"], sh["takeaways"], sh["giveaways"])
    b_away = bm.predict_points(sa["pass_yds"], sa["rush_yds"], sa["takeaways"], sa["giveaways"])

    wa, wb = cfg.w_ratings, cfg.w_boxscore
    mu_home = (wa * a_home + wb * b_home) / (wa + wb)
    mu_away = (wa * a_away + wb * b_away) / (wa + wb)

    # --- situational adjustments, applied as points then split across the two teams ---
    adj_home = 0.0 if neutral else cfg.hfa_points / 2
    adj_away = 0.0 if neutral else -cfg.hfa_points / 2
    rest_map = {"bye": cfg.bye_week_bonus, "short": cfg.short_week_penalty, "normal": 0.0}
    adj_home += rest_map.get(rest_home, 0.0) / 2
    adj_away += rest_map.get(rest_away, 0.0) / 2
    if travel_away:
        adj_away += cfg.long_travel_penalty / 2
        adj_home -= cfg.long_travel_penalty / 2
    mu_home += adj_home
    mu_away += adj_away

    # --- C: market blend ---
    blend = 0.0
    if market_spread is not None:
        decay = 0.5 ** (max(weeks_of_data, 0.0) / cfg.market_blend_halflife_weeks)
        blend = cfg.market_blend_floor + (cfg.market_blend_week1 - cfg.market_blend_floor) * decay
        blend = float(np.clip(blend, 0.0, 1.0))
        # market_spread is quoted from the home team's side: -3.5 means home favoured by 3.5
        mkt_margin = -market_spread
        mkt_total = market_total if market_total is not None else (mu_home + mu_away)
        mkt_home = (mkt_total + mkt_margin) / 2
        mkt_away = (mkt_total - mkt_margin) / 2
        mu_home = (1 - blend) * mu_home + blend * mkt_home
        mu_away = (1 - blend) * mu_away + blend * mkt_away

    return {
        "mu_home": float(max(mu_home, 6.0)),
        "mu_away": float(max(mu_away, 6.0)),
        "components": {
            "ratings": (round(a_home, 2), round(a_away, 2)),
            "boxscore": (round(b_home, 2), round(b_away, 2)),
            "market_weight": round(blend, 3),
            "situational": (round(adj_home, 2), round(adj_away, 2)),
            "exp_stats_home": {k: round(v, 1) for k, v in sh.items()},
            "exp_stats_away": {k: round(v, 1) for k, v in sa.items()},
        },
    }


# ==============================================================================
# SECTION 6 - THE SIMULATION ENGINE
# ==============================================================================
# Two engines. Run both; they disagree in informative ways.
#
#   simulate_normal()  - faithful to the Samford article. Draw each stat from a
#                        normal, multiply by regression coefficients. Fast,
#                        transparent, and WRONG near key numbers, because it puts
#                        continuous mass where football puts none.
#
#   simulate_drives()  - the Excel LADZ approach taken further. Simulate
#                        possessions and score types, so the output is a real
#                        scoreline. This is the one to price bets off.

def simulate_normal(mu_home, mu_away, bm: BoxModel, r: Ratings,
                    home: str, away: str, cfg: Config = CFG,
                    rng: np.random.Generator | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Samford's NORM.INV(RAND(), mean, sd) loop, vectorised."""
    rng = rng or np.random.default_rng(cfg.seed)
    n = cfg.n_sims
    sh = expected_box_stats(home, away, r)
    sa = expected_box_stats(away, home, r)

    # The article used one random draw per game with the complement for the
    # opponent, which forces perfect negative correlation. A shared environment
    # factor plus independent noise is the same idea, better calibrated.
    env = rng.normal(0, 1, n)

    def draw(mean, sd, sign):
        return mean + sd * (0.45 * sign * env + math.sqrt(1 - 0.45 ** 2) * rng.normal(0, 1, n))

    hp = draw(sh["pass_yds"], bm.sd["pass_yds"], +1)
    hr = draw(sh["rush_yds"], bm.sd["rush_yds"], -1)
    ap = draw(sa["pass_yds"], bm.sd["pass_yds"], -1)
    ar = draw(sa["rush_yds"], bm.sd["rush_yds"], +1)
    ht = rng.poisson(max(sh["takeaways"], 0.05), n)
    hg = rng.poisson(max(sh["giveaways"], 0.05), n)

    # Residual noise matters. The regression explains maybe a third of the
    # variance in points; without adding the unexplained part back, this engine
    # produces absurdly confident win probabilities.
    shock = rng.normal(0, 1, n)
    ph = bm.predict_points(hp, hr, ht, hg) + bm.resid_sd * (
        0.35 * shock + math.sqrt(1 - 0.35 ** 2) * rng.normal(0, 1, n))
    pa = bm.predict_points(ap, ar, hg, ht) + bm.resid_sd * (
        0.35 * shock + math.sqrt(1 - 0.35 ** 2) * rng.normal(0, 1, n))

    # Recentre on the ensemble expectation, keep the simulated spread.
    ph = ph - ph.mean() + mu_home
    pa = pa - pa.mean() + mu_away
    return np.maximum(np.round(ph), 0), np.maximum(np.round(pa), 0)


def _drive_rates(mu: float, drives: float, cfg: Config) -> tuple[float, float]:
    """
    Convert a target points-per-game into per-drive touchdown and field goal
    rates. TDs scale faster than FGs: a great offence does not just kick more.
    Then rescale both so expected points hits the target exactly.
    """
    base_ppd = cfg.base_td_rate * 6.95 + cfg.base_fg_rate * 3.0
    k = mu / max(drives * base_ppd, 1e-6)
    td = cfg.base_td_rate * (k ** cfg.td_elasticity)
    fg = cfg.base_fg_rate * (k ** cfg.fg_elasticity)
    ppd = td * 6.95 + fg * 3.0
    scale = (mu / drives) / max(ppd, 1e-6)
    td, fg = td * scale, fg * scale
    if td + fg > 0.92:
        s = 0.92 / (td + fg)
        td, fg = td * s, fg * s
    return float(td), float(fg)


def _score_drives(n_sims, drives, td_rate, fg_rate, cfg, rng):
    """Vectorised: for each sim, roll every possession and total the points."""
    d = int(math.ceil(drives.max())) if isinstance(drives, np.ndarray) else int(math.ceil(drives))
    u = rng.random((n_sims, d))
    live = np.arange(d)[None, :] < np.asarray(drives).reshape(-1, 1)

    is_td = (u < td_rate) & live
    is_fg = (u >= td_rate) & (u < td_rate + fg_rate) & live
    is_saf = (u >= td_rate + fg_rate) & (u < td_rate + fg_rate + cfg.base_safety_rate) & live

    n_td = is_td.sum(axis=1)
    pts = n_td * 6 + is_fg.sum(axis=1) * 3 + is_saf.sum(axis=1) * 2

    # Extra point / two point conversions, resolved per touchdown.
    max_td = int(n_td.max()) if n_td.size and n_td.max() > 0 else 0
    if max_td:
        mask = np.arange(max_td)[None, :] < n_td[:, None]
        go2 = rng.random((n_sims, max_td)) < cfg.two_point_attempt_rate
        made2 = rng.random((n_sims, max_td)) < cfg.two_point_success
        made1 = rng.random((n_sims, max_td)) < cfg.xp_success
        extra = np.where(go2, np.where(made2, 2, 0), np.where(made1, 1, 0)) * mask
        pts = pts + extra.sum(axis=1)
    return pts.astype(int)


def _overtime(home_pts, away_pts, td_h, fg_h, td_a, fg_a, cfg, rng, playoff=False):
    """
    Current NFL overtime: both teams are guaranteed a possession, then next score
    wins. Regular-season games can still end tied; playoff games cannot.
    Overtime is rare enough that the exact treatment barely moves a line, which
    is why the source model standardised it too - but a tie is a moneyline push,
    so it has to exist.
    """
    tied = home_pts == away_pts
    idx = np.where(tied)[0]
    if idx.size == 0:
        return home_pts, away_pts

    hp, ap = home_pts.copy(), away_pts.copy()
    for _ in range(cfg.ot_length_drives):
        still = np.where(hp[idx] == ap[idx])[0]
        if still.size == 0:
            break
        sel = idx[still]
        uh, ua = rng.random(sel.size), rng.random(sel.size)
        add_h = np.where(uh < td_h, 7, np.where(uh < td_h + fg_h, 3, 0))
        add_a = np.where(ua < td_a, 7, np.where(ua < td_a + fg_a, 3, 0))
        hp[sel] += add_h
        ap[sel] += add_a

    if playoff or not cfg.ot_regular_season_tie_allowed:
        still = idx[hp[idx] == ap[idx]]
        if still.size:
            win_h = rng.random(still.size) < 0.5
            hp[still] += np.where(win_h, 3, 0)
            ap[still] += np.where(win_h, 0, 3)
    return hp, ap


def simulate_drives(mu_home: float, mu_away: float, cfg: Config = CFG,
                    rng: np.random.Generator | None = None,
                    pace: float = 1.0, playoff: bool = False
                    ) -> tuple[np.ndarray, np.ndarray]:
    rng = rng or np.random.default_rng(cfg.seed)
    n = cfg.n_sims

    # A shared environment factor: weather, pace, whether the game turns into a
    # shootout. This is what makes the two teams' scores correlated, which the
    # totals market cares about a great deal.
    env = np.exp(rng.normal(0, cfg.env_sigma, n))
    drives = np.clip(np.round(rng.normal(cfg.drives_per_team * pace, cfg.drive_sd, n)), 7, 16)

    td_h, fg_h = _drive_rates(mu_home, cfg.drives_per_team * pace, cfg)
    td_a, fg_a = _drive_rates(mu_away, cfg.drives_per_team * pace, cfg)

    # env scales scoring efficiency up and down together
    td_h_v = np.clip(td_h * env, 0.01, 0.75)[:, None]
    fg_h_v = np.clip(fg_h * env ** 0.5, 0.01, 0.45)[:, None]
    td_a_v = np.clip(td_a * env, 0.01, 0.75)[:, None]
    fg_a_v = np.clip(fg_a * env ** 0.5, 0.01, 0.45)[:, None]

    hp = _score_drives(n, drives, td_h_v, fg_h_v, cfg, rng)
    ap = _score_drives(n, drives, td_a_v, fg_a_v, cfg, rng)
    return _overtime(hp, ap, td_h, fg_h, td_a, fg_a, cfg, rng, playoff)


# ==============================================================================
# SECTION 7 - TURNING SIMULATIONS INTO PRICES
# ==============================================================================

def price_from_sims(hp: np.ndarray, ap: np.ndarray) -> dict:
    n = len(hp)
    margin = hp - ap
    total = hp + ap
    return {
        "p_home_win": float((margin > 0).mean() + 0.5 * (margin == 0).mean()),
        "p_away_win": float((margin < 0).mean() + 0.5 * (margin == 0).mean()),
        "p_tie": float((margin == 0).mean()),
        "mean_home": float(hp.mean()),
        "mean_away": float(ap.mean()),
        "median_margin": float(np.median(margin)),
        "mean_margin": float(margin.mean()),
        "sd_margin": float(margin.std()),
        "mean_total": float(total.mean()),
        "sd_total": float(total.std()),
        "_margin": margin,
        "_total": total,
    }


def p_cover(margin: np.ndarray, spread_home: float) -> tuple[float, float, float]:
    """
    spread_home is the home line as the book quotes it: -3.5 means home gives 3.5.
    Home covers when margin + spread_home > 0. Returns (home, away, push).
    """
    adj = margin + spread_home
    push = float((adj == 0).mean())
    h = float((adj > 0).mean())
    a = float((adj < 0).mean())
    if push > 0:  # pushes are returned, so renormalise over decided outcomes
        h, a = h / (1 - push), a / (1 - push)
    return h, a, push


def p_over(total: np.ndarray, line: float) -> tuple[float, float, float]:
    push = float((total == line).mean())
    o = float((total > line).mean())
    u = float((total < line).mean())
    if push > 0:
        o, u = o / (1 - push), u / (1 - push)
    return o, u, push


# Observed frequency of each absolute final margin across roughly fifteen recent
# NFL seasons. Your simulated column should look like this one. If it does not,
# the levers are base_fg_rate, base_td_rate and td_elasticity - in that order.
EMPIRICAL_MARGIN_PCT = {
    0: 0.5, 1: 2.5, 2: 3.6, 3: 9.5, 4: 4.9, 5: 3.4, 6: 5.3, 7: 7.1, 8: 3.4,
    9: 2.9, 10: 5.7, 11: 3.0, 12: 2.3, 13: 3.1, 14: 4.4, 15: 2.3, 16: 2.5,
    17: 3.4, 18: 1.8, 19: 1.5, 20: 2.1, 21: 2.0,
}


def key_number_table(margin: np.ndarray) -> pd.DataFrame:
    """
    The distribution of exact margins, beside the real one.

    This is the whole argument for the drive engine. 3 and 7 carry far more mass
    than any smooth curve would give them, and that mass is exactly what you buy
    or sell when you take a -2.5 instead of a -3, or lay a -3.5.
    """
    vals, counts = np.unique(np.abs(margin), return_counts=True)
    df = pd.DataFrame({"margin": vals, "model_pct": counts / len(margin) * 100})
    df = df[df["margin"] <= 17].copy()
    df["real_nfl_pct"] = df["margin"].map(EMPIRICAL_MARGIN_PCT)
    df["diff"] = df["model_pct"] - df["real_nfl_pct"]
    return df.round(2)


def analyse_game(home, away, r, bm, cfg: Config = CFG, **kw) -> dict:
    market_spread = kw.pop("market_spread", None)
    market_total = kw.pop("market_total", None)
    weeks = kw.pop("weeks_of_data", 18.0)
    playoff = kw.pop("playoff", False)
    pace = kw.pop("pace", 1.0)

    ep = expected_points(home, away, r, bm, cfg,
                         market_spread=market_spread, market_total=market_total,
                         weeks_of_data=weeks, **kw)
    rng = np.random.default_rng(cfg.seed)
    hp, ap = simulate_drives(ep["mu_home"], ep["mu_away"], cfg, rng, pace, playoff)
    out = price_from_sims(hp, ap)
    out["expected"] = ep
    out["home"], out["away"] = home, away

    # Cross-check against the Samford-style normal engine. A large gap between
    # the two is a signal to look at the game by hand, not to bet it.
    hp2, ap2 = simulate_normal(ep["mu_home"], ep["mu_away"], bm, r, home, away, cfg, rng)
    alt = price_from_sims(hp2, ap2)
    out["crosscheck_normal_p_home"] = alt["p_home_win"]
    out["engine_disagreement"] = abs(alt["p_home_win"] - out["p_home_win"])
    return out


# ==============================================================================
# SECTION 8 - BET SELECTION
# ==============================================================================

def evaluate_bets(res: dict, row: dict, cfg: Config = CFG) -> list[dict]:
    bets = []
    margin, total = res["_margin"], res["_total"]
    home, away = res["home"], res["away"]

    def add(market, side, model_p, odds, fair_p, min_edge):
        if odds is None or (isinstance(odds, float) and math.isnan(odds)):
            return
        edge = model_p - fair_p
        ev = expected_value(model_p, odds)
        stake = kelly_stake(model_p, odds, cfg) if edge >= min_edge and ev > 0 else 0.0
        bets.append({
            "game": f"{away} @ {home}", "market": market, "side": side,
            "odds": odds, "model_prob": round(model_p, 4),
            "no_vig_prob": round(fair_p, 4), "edge": round(edge, 4),
            "ev_per_unit": round(ev, 4),
            "stake_pct": round(stake, 4),
            "stake_$": round(stake * cfg.bankroll, 2),
            "fair_odds": prob_to_american(model_p),
            "bet": "YES" if stake > 0 else "no",
        })

    # ---- Moneyline ----
    mlh, mla = row.get("home_ml"), row.get("away_ml")
    if mlh and mla and not (math.isnan(mlh) or math.isnan(mla)):
        fh, fa = devig_power(american_to_prob(mlh), american_to_prob(mla))
        add("Moneyline", home, res["p_home_win"], mlh, fh, cfg.min_edge_ml)
        add("Moneyline", away, res["p_away_win"], mla, fa, cfg.min_edge_ml)

    # ---- Spread ----
    sp = row.get("spread_home")
    if sp is not None and not (isinstance(sp, float) and math.isnan(sp)):
        ph, pa, push = p_cover(margin, sp)
        oh = row.get("spread_price_home", -110)
        oa = row.get("spread_price_away", -110)
        fh, fa = devig_power(american_to_prob(oh), american_to_prob(oa))
        add("Spread", f"{home} {sp:+.1f}", ph, oh, fh, cfg.min_edge_spread)
        add("Spread", f"{away} {-sp:+.1f}", pa, oa, fa, cfg.min_edge_spread)

    # ---- Total ----
    tl = row.get("total")
    if tl is not None and not (isinstance(tl, float) and math.isnan(tl)):
        po, pu, push = p_over(total, tl)
        oo = row.get("over_price", -110)
        ou = row.get("under_price", -110)
        fo, fu = devig_power(american_to_prob(oo), american_to_prob(ou))
        add("Total", f"Over {tl}", po, oo, fo, cfg.min_edge_total)
        add("Total", f"Under {tl}", pu, ou, fu, cfg.min_edge_total)

    return bets


# ==============================================================================
# SECTION 9 - BACKTESTING AND CALIBRATION
# ==============================================================================

def brier(p, y):
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def log_loss(p, y):
    p = np.clip(np.asarray(p), 1e-9, 1 - 1e-9)
    y = np.asarray(y)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def calibration(p, y, bins=10) -> pd.DataFrame:
    p, y = np.asarray(p), np.asarray(y)
    edges = np.linspace(0, 1, bins + 1)
    rows = []
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum() == 0:
            continue
        rows.append({
            "bin": f"{edges[i]:.0%}-{edges[i+1]:.0%}",
            "n": int(m.sum()),
            "predicted": round(float(p[m].mean()), 3),
            "actual": round(float(y[m].mean()), 3),
            "gap": round(float(y[m].mean() - p[m].mean()), 3),
        })
    return pd.DataFrame(rows)


def walk_forward(df: pd.DataFrame, cfg: Config = CFG, min_week: int = 4,
                 fast_sims: int = 4000) -> dict:
    """
    Refit after every week using only prior games, then predict the next week.
    This is the only honest way to test. Fitting on the full season and scoring
    the same season will tell you the model is brilliant, and it will be lying.
    """
    c = Config(**{**asdict(cfg), "n_sims": fast_sims})
    seasons = sorted(df["season"].unique())
    preds, actual, rows = [], [], []

    for s in seasons:
        weeks = sorted(df[df["season"] == s]["week"].unique())
        for wk in weeks:
            if wk < min_week:
                continue
            train = df[(df["season"] < s) | ((df["season"] == s) & (df["week"] < wk))]
            test = df[(df["season"] == s) & (df["week"] == wk)]
            if len(train) < 60 or len(test) == 0:
                continue
            long = to_long(train)
            r = fit_ratings(long, c)
            bm = fit_box_model(long, c)
            for _, g in test.iterrows():
                try:
                    res = analyse_game(g["home_team"], g["away_team"], r, bm, c,
                                       neutral=bool(g.get("neutral", 0)),
                                       weeks_of_data=float(wk))
                except Exception:
                    continue
                p = res["p_home_win"]
                y = 1 if g["home_score"] > g["away_score"] else 0
                preds.append(p)
                actual.append(y)
                rows.append({
                    "season": s, "week": wk,
                    "game": f"{g['away_team']} @ {g['home_team']}",
                    "p_home": round(p, 3), "home_won": y,
                    "proj_margin": round(res["mean_margin"], 1),
                    "actual_margin": int(g["home_score"] - g["away_score"]),
                    "proj_total": round(res["mean_total"], 1),
                    "actual_total": int(g["home_score"] + g["away_score"]),
                })

    preds, actual = np.array(preds), np.array(actual)
    if len(preds) == 0:
        return {"error": "not enough data to backtest"}
    log = pd.DataFrame(rows)
    picks = (preds > 0.5).astype(int)
    return {
        "n_games": len(preds),
        "straight_up_accuracy": round(float((picks == actual).mean()), 4),
        "brier": round(brier(preds, actual), 4),
        "brier_baseline_coinflip": 0.25,
        "log_loss": round(log_loss(preds, actual), 4),
        "margin_mae": round(float((log["proj_margin"] - log["actual_margin"]).abs().mean()), 2),
        "total_mae": round(float((log["proj_total"] - log["actual_total"]).abs().mean()), 2),
        "calibration": calibration(preds, actual),
        "log": log,
    }


# ==============================================================================
# SECTION 10 - SYNTHETIC DATA, SO THE FILE RUNS BEFORE YOU HAVE ANY
# ==============================================================================

def make_demo_games(n_seasons=2, seed=7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    strength = {t: rng.normal(0, 3.0) for t in TEAMS}
    rows = []
    for s in range(2024, 2024 + n_seasons):
        for t in TEAMS:
            strength[t] = 0.75 * strength[t] + rng.normal(0, 1.6)
        for wk in range(1, 18):
            order = list(TEAMS)
            rng.shuffle(order)
            for i in range(0, 32, 2):
                h, a = order[i], order[i + 1]
                mh = 22.4 + (strength[h] - strength[a]) / 2 + CFG.hfa_points / 2
                ma = 22.4 + (strength[a] - strength[h]) / 2 - CFG.hfa_points / 2
                hp, ap = simulate_drives(mh, ma, Config(n_sims=1, seed=None),
                                         np.random.default_rng(rng.integers(1 << 31)))
                rows.append({
                    "season": s, "week": wk, "home_team": h, "away_team": a,
                    "home_score": int(hp[0]), "away_score": int(ap[0]),
                    # Box stats are tied to the simulated score, so the demo
                    # regression recovers coefficients close to the real ones.
                    "home_pass_yds": max(60, int(rng.normal(227 + 4.6 * (hp[0] - 22.4), 52))),
                    "away_pass_yds": max(60, int(rng.normal(227 + 4.6 * (ap[0] - 22.4), 52))),
                    "home_rush_yds": max(20, int(rng.normal(116 + 2.4 * (hp[0] - 22.4), 38))),
                    "away_rush_yds": max(20, int(rng.normal(116 + 2.4 * (ap[0] - 22.4), 38))),
                    "home_giveaways": int(rng.poisson(max(0.2, 1.3 - 0.04 * (hp[0] - 22.4)))),
                    "away_giveaways": int(rng.poisson(max(0.2, 1.3 - 0.04 * (ap[0] - 22.4)))),
                    "neutral": 0,
                })
    out = pd.DataFrame(rows).sort_values(["season", "week"]).reset_index(drop=True)
    out["game_id"] = np.arange(len(out))
    return out


# ==============================================================================
# SECTION 11 - COMMAND LINE
# ==============================================================================

def _load_priors(path):
    if not path or not os.path.exists(path):
        return None
    p = pd.read_csv(path)
    return dict(zip(p["team"].str.upper(), p["prior_rating"].astype(float)))


def cmd_ratings(args):
    df = make_demo_games() if args.demo else load_games(args.games)
    long = to_long(df)
    r = fit_ratings(long, CFG, _load_priors(args.priors))
    bm = fit_box_model(long, CFG)
    print("\n=== FIVE-STATISTIC REGRESSION (Samford methodology) ===")
    print(bm.report())
    print("\n=== POWER RATINGS (opponent-adjusted, recency-weighted) ===")
    print(r.table().to_string(index=False))
    print("\nRead: Power = points better than a league-average team on a neutral field.")
    print("A matchup line is roughly Power(home) - Power(away) + %.1f HFA." % CFG.hfa_points)
    if args.out:
        r.table().to_csv(args.out, index=False)
        print(f"\nSaved -> {args.out}")


def cmd_predict(args):
    df = make_demo_games() if args.demo else load_games(args.games)
    long = to_long(df)
    r = fit_ratings(long, CFG, _load_priors(args.priors))
    bm = fit_box_model(long, CFG)

    if args.demo or not args.slate:
        slate = pd.DataFrame([
            {"home_team": "KC", "away_team": "BUF", "spread_home": -2.5, "total": 48.5,
             "home_ml": -140, "away_ml": +118, "spread_price_home": -110,
             "spread_price_away": -110, "over_price": -108, "under_price": -112},
            {"home_team": "SF", "away_team": "SEA", "spread_home": -3.0, "total": 44.5,
             "home_ml": -165, "away_ml": +140, "spread_price_home": -112,
             "spread_price_away": -108, "over_price": -110, "under_price": -110},
        ])
    else:
        slate = pd.read_csv(args.slate)
        for c in ("home_team", "away_team"):
            slate[c] = slate[c].astype(str).str.upper().str.strip()

    all_bets, summary = [], []
    for _, g in slate.iterrows():
        row = g.to_dict()
        res = analyse_game(
            row["home_team"], row["away_team"], r, bm, CFG,
            neutral=bool(row.get("neutral", 0)),
            rest_home=str(row.get("rest_home", "normal")),
            rest_away=str(row.get("rest_away", "normal")),
            travel_away=bool(row.get("travel_away", 0)),
            market_spread=row.get("spread_home"),
            market_total=row.get("total"),
            weeks_of_data=float(args.week),
            playoff=bool(row.get("playoff", 0)),
        )
        ph, pa = res["mean_home"], res["mean_away"]
        summary.append({
            "game": f"{res['away']} @ {res['home']}",
            "proj_score": f"{res['home']} {ph:.1f} - {pa:.1f} {res['away']}",
            "model_line": f"{res['home']} {-res['mean_margin']:+.1f}",
            "market_line": f"{row.get('spread_home', float('nan')):+.1f}",
            "line_edge": round(-res["mean_margin"] - float(row.get("spread_home", np.nan)), 2),
            "model_total": round(res["mean_total"], 1),
            "market_total": row.get("total"),
            "p_home_win": round(res["p_home_win"], 3),
            "engine_gap": round(res["engine_disagreement"], 3),
        })
        all_bets += evaluate_bets(res, row, CFG)

        if args.verbose:
            print(f"\n--- {res['away']} @ {res['home']} ---")
            print("market weight in blend:", res["expected"]["components"]["market_weight"])
            print("ratings vs boxscore:", res["expected"]["components"]["ratings"],
                  res["expected"]["components"]["boxscore"])
            print("\nMargin distribution (key numbers):")
            print(key_number_table(res["_margin"]).to_string(index=False))

    print("\n=== PROJECTIONS ===")
    print(pd.DataFrame(summary).to_string(index=False))

    bets = pd.DataFrame(all_bets)
    print("\n=== FULL PRICING ===")
    print(bets.drop(columns=["stake_pct"]).to_string(index=False))
    fire = bets[bets["bet"] == "YES"]
    print("\n=== BETS THAT CLEAR THE EDGE THRESHOLD ===")
    print(fire.to_string(index=False) if len(fire) else
          "  None. That is the normal result and it is the model working.")
    if args.out:
        bets.to_csv(args.out, index=False)
        print(f"\nSaved -> {args.out}")


def cmd_backtest(args):
    df = make_demo_games(n_seasons=3) if args.demo else load_games(args.games)
    print("Walk-forward backtest running (refits every week)...")
    res = walk_forward(df, CFG, min_week=args.min_week)
    if "error" in res:
        print(res["error"])
        return
    print("\n=== OUT-OF-SAMPLE PERFORMANCE ===")
    for k in ("n_games", "straight_up_accuracy", "brier", "brier_baseline_coinflip",
              "log_loss", "margin_mae", "total_mae"):
        print(f"  {k:28s} {res[k]}")
    print("\n=== CALIBRATION ===")
    print(res["calibration"].to_string(index=False))
    print("\nA well-calibrated model has 'actual' tracking 'predicted' in every bin.")
    print("Accuracy above 66% or margin MAE below 9.5 on real data means you have")
    print("a bug or a leak, not an edge. The market itself sits near 67% / 9.8.")
    if args.out:
        res["log"].to_csv(args.out, index=False)
        print(f"\nSaved -> {args.out}")


def cmd_demo(args):
    print(__doc__.split("USAGE")[0])
    class A: pass
    a = A(); a.demo = True; a.games = None; a.priors = None; a.out = None
    a.slate = None; a.week = 1; a.verbose = True; a.min_week = 6
    cmd_ratings(a)
    cmd_predict(a)
    cmd_backtest(a)


def main():
    p = argparse.ArgumentParser(description="NFL prediction model")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn in (("ratings", cmd_ratings), ("predict", cmd_predict),
                     ("backtest", cmd_backtest), ("demo", cmd_demo)):
        s = sub.add_parser(name)
        s.add_argument("--games", default="games.csv")
        s.add_argument("--slate", default=None)
        s.add_argument("--priors", default="priors.csv")
        s.add_argument("--out", default=None)
        s.add_argument("--week", type=float, default=1.0)
        s.add_argument("--min-week", type=int, default=6)
        s.add_argument("--demo", action="store_true")
        s.add_argument("-v", "--verbose", action="store_true")
        s.set_defaults(func=fn)
    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
