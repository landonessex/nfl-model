# NFL Model — 2026 Season

Built from the two sources you sent, with the gaps in each one filled in.

**Season starts Wednesday 9 September 2026** (Patriots at Seahawks, a Super Bowl LX rematch). That gives you about three weeks to load data and sanity-check the thing before it matters.

---

## What's in the box

| File | What it does |
|---|---|
| `nfl_model.py` | The engine. Ratings, regression, simulation, pricing, staking, backtesting. |
| `NFL_Model_2026.xlsx` | Live-formula version. Same maths, spreadsheet-native, no Python needed. |
| `games_TEMPLATE.csv` | Completed games — feeds the ratings. |
| `slate_TEMPLATE.csv` | Upcoming games plus the book's prices. |
| `priors_TEMPLATE.csv` | Preseason power ratings, so Week 1 isn't a coin flip. |

Start with the spreadsheet if you want to see the machinery. Move to Python once you're betting more than a couple of games a week — it does the opponent adjustment properly, and the spreadsheet can't.

```bash
python nfl_model.py demo          # runs on synthetic data, proves the install
python nfl_model.py ratings  --games games.csv
python nfl_model.py predict  --games games.csv --slate slate.csv --week 5 -v
python nfl_model.py backtest --games games.csv
```

---

## How the two sources map onto this

### From Excel LADZ — the skeleton

Attack and Defense ratings produce an expected score for each team, that score gets simulated into a full scoreline, overtime is handled explicitly, thousands of simulations become a distribution, and the distribution prices Moneyline, Spread and Total against the book. That is the architecture here, unchanged.

Two things got upgraded:

**Ratings are solved, not averaged.** Excel LADZ divides points scored by league average. That's readable and it's wrong by Week 6, because a team that has played the four worst defences in football looks like an offensive juggernaut. This engine solves attack and defense simultaneously against opponent quality, iterating until they stop moving. A team's rating is what it would do against a league-average opponent, which is the only version that transfers to next week's game.

**Simulation counts possessions.** Both source models produce a points total and stop. This one rolls each drive: touchdown, field goal, or nothing, then resolves the extra point or the two-point try. That's why a simulated game comes out 24–20 and not 23.7–19.4.

### From Samford — the five statistics

Streitmatter's regression found four stats carried the signal: passing yards, rushing yards, takeaways, giveaways. That regression is here, refit on whatever data you supply, and its output is a genuinely independent second estimate of each team's score.

Three changes:

**Ridge, not plain OLS.** Passing and rushing yards are correlated with each other and with game script — teams that lead run the ball. Unpenalised OLS splits credit between them unstably; add a season of data and the coefficients swing. Ridge keeps them steady.

**Expected stats route through the ratings.** The article blends a team's own passing average with its opponent's passing-allowed average. Reasonable, but both numbers already have schedule baked in, so blending them keeps the bias rather than cancelling it. Here it's `league average × team's pass attack × opponent's pass defense`, which is schedule-free on both sides. (The spreadsheet keeps the original blend, because the iterative solve doesn't fit in a cell.)

**Residual noise is added back.** The regression explains roughly a third of the variance in points. Simulate without the other two thirds and you get win probabilities of 94% on a three-point favourite. The `Residual SD` parameter puts the unexplained part back.

### The part neither source has

**Key numbers.** The Samford model draws points from a normal distribution. NFL margins are not normal — they pile up on 3 and 7, which together account for about one game in six, and they leave 2.5 and 3.5 comparatively empty. A smooth curve smears that mass across numbers football rarely lands on, which is precisely the error that decides whether buying a -3 down to -2.5 is worth the price. Run `predict -v` and you'll get your simulated margin distribution beside the real one.

**Devigging.** A -110/-110 pair implies 52.4% on each side, summing to 104.8%. Compare your 54% to the raw 52.4% and you'll bet things you shouldn't. The engine strips the margin out with a power devig, which handles lopsided markets better than simple scaling.

**Kelly staking.** Quarter Kelly, capped at 2% of bankroll, and only when the edge clears a threshold. Flat betting throws away information about how big each edge is; full Kelly on estimated probabilities finds ruin faster than you'd believe.

**Market blending.** In Week 1 you have no 2026 data and the market has every roster move, every coaching change and every injury priced. So the model starts at 75% market weight and decays toward 25% as your own sample grows. This will feel like cheating. It is the single most valuable line in the file.

**Walk-forward backtesting.** Refits after every week and predicts the next one, using only prior games. Fitting on a full season and grading the same season will tell you the model is brilliant. It will be lying.

---

## Data schemas

**games.csv** — one row per completed game.

```
season, week, home_team, away_team, home_score, away_score,
home_pass_yds, away_pass_yds, home_rush_yds, away_rush_yds,
home_giveaways, away_giveaways, neutral
```

Team codes are the standard three-letter abbreviations (`KC`, `SF`, `LAR`, `WAS`…). Passing yards should be **net** — after sacks — because that's what the regression was calibrated on. Takeaways aren't a column: your takeaways are your opponent's giveaways, and the engine derives them.

Load two full prior seasons plus the current one. The engine decays old games automatically, so more history costs you nothing.

**slate.csv** — upcoming games. Only `home_team`, `away_team` are required; every price column is optional, and any market you leave blank simply isn't priced. `spread_home` is quoted from the home side: `-2.5` means the home team gives 2.5. `rest_home` and `rest_away` take `normal`, `bye`, or `short`.

**priors.csv** — `team, prior_rating` in points versus average. Derive them from win totals: a 10.5-win team is roughly +3, an 11.5-win team roughly +5, a 6.5-win team roughly -3. Or use last season's final power ratings regressed a third of the way toward zero.

### Where to get the data, free

- **nflverse** — `github.com/nflverse/nflverse-data`. Complete play-by-play and team game logs, R and Python packages. This is what serious people use.
- **Pro Football Reference** — `pro-football-reference.com/years/2026/`. Every table has a "Share & Export → Get as CSV" link.
- **The Odds API** — `the-odds-api.com`. Free tier covers a weekly NFL slate across most books.

---

## Weekly routine

**Tuesday.** Update `games.csv` with the weekend's results. Rerun `ratings`. Read the table before you read anything else — if a team has moved more than about 2 points in a week, either something real happened (a quarterback went down) or you have a data error. Find out which.

**Wednesday.** Run `predict` against opening numbers, without the market blend if you want to see your raw opinion. Note every game where you disagree by more than a point and a half.

**Thursday–Saturday.** Re-run as prices move. Bet where the edge survives.

**Sunday.** Record the closing line for every bet you placed, whether it won or lost.

That last step is the one people skip and it's the one that matters. Over a single season, win rate tells you almost nothing — a 55% bettor loses money over 200 bets about one season in six. Closing line value tells you something after about thirty bets. If you're consistently beating the number the market settles on, you have an edge, and the profit will find you. If you're not, no amount of winning weekends means you do.

---

## Tuning

Everything lives in the `Config` block at the top of `nfl_model.py`, and on the `Coefficients` sheet in the workbook.

**Check first, every August:**

- `league_ppg` — set from last season's actual scoring
- `hfa_points` — currently 1.9. NFL home field has drifted down from about 2.6 a decade ago. Set it to 0 for neutral sites and use `hfa_international` for the London and Melbourne games.
- Overtime rules — the file assumes both teams are guaranteed a possession in the regular season and a tie is still possible. Verify this hasn't changed before Week 1.

**The main calibration dial** is the key-number table from `predict -v`. If your 3s and 7s come in light against the real column, raise `base_fg_rate` and lower `td_elasticity`. As shipped the model runs a little flat — about 7.5% on 3 against a real 9.5% — because it doesn't model the endgame, where a trailing team kicks the field goal that makes it a three-point game. Worth knowing before you bet a lot of -3s.

**Leave alone unless you have a reason:** `env_sigma`, `drive_sd`, the elasticities. They're set so simulated margin and total variance land near the real values (13.9 and 15.0 against actual 13.6 and 13.9). Changing them changes every price in the file.

---

## What good looks like

Run `backtest` on real data and expect roughly:

| Metric | Good | What it means |
|---|---|---|
| Straight-up accuracy | 63–67% | The closing spread gets about 67%. Matching it is a real result. |
| Brier score | 0.19–0.21 | 0.25 is a coin flip. Below 0.185 on real data, look for a leak. |
| Margin MAE | 9.8–10.5 | The market sits near 9.8. |
| Calibration gap | within 0.03 per bin | Games you call 60% should win about 60% of the time. |

If you're seeing 72% and a margin MAE of 8, you have not built a better model than every professional in the industry. You have leaked future information into training — usually by fitting on games you're also grading. Run the walk-forward backtest instead.

The Samford article reports a 70.7% success rate over a partial season. That's a real result on a small sample, and worth reading as encouragement rather than a target: over ten weeks of NFL, the gap between 70% and 58% is about nine games, which is comfortably inside the range luck produces.

---

## The honest part

An NFL season is 272 games. If you bet a hundred of them at a genuine 54% — which would be a very good model — your expected profit at -110 is about three units, with a standard deviation of ten. Losing seasons happen to winning models routinely. That maths doesn't change no matter how good the code is.

Most weeks this model should find nothing. When it finds eight bets on a thirteen-game slate, it isn't sharp, it's broken — go look at your inputs.

This is for research and entertainment. Sports betting is legal in some places and not others, it's designed to take money from the people playing, and it's genuinely addictive for a meaningful fraction of them. Bet only what you can afford to lose, if at all.
