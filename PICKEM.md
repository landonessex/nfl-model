# Winning a 150-person pick'em pool

The model tells you who wins each game. That is not the same as telling you what to pick.

## The one idea that matters

**Games where the whole pool picks the same side cannot move you up the leaderboard.** Everyone gains together or loses together. Your finishing position is decided entirely by the handful of games where you pick differently from the field.

So you are not maximising correct picks. You are maximising the probability of finishing first, and with 150 entrants those are different objectives.

Here is what that looks like in numbers, from a demo week:

| Contrarian picks | Expected correct | P(win the week) | P(win the season) |
|---|---|---|---|
| 0 — pure chalk | 9.46 | 0.03% | 0.60% |
| 1 | 9.11 | 1.03% | 1.50% |
| 3 | 8.75 | 1.63% | 0.67% |
| 6 | 8.29 | 1.52% | 0.26% |

A random card wins 0.67% of the time. **Pure chalk does worse than random** — 0.03% — because thirty other people submit the identical card and you split the prize with all of them. The most accurate strategy available to you is close to the worst strategy available to you.

Note the other thing in that table: the first deviation improves your *season* odds too, from 0.60% to 1.50%. Escaping the tie-bloc is free. Only past two or three does accuracy start costing you more than differentiation buys.

The gap between 1 and 5 deviations is inside the simulation's noise. Treat "two or three" as the answer and don't agonise over which.

## Why underdogs, specifically

The public picks favourites more often than favourites win, and the gap widens with the spread:

| Spread | Favourite wins | Pool picks the favourite | Gap |
|---|---|---|---|
| −3 | ~58% | ~67% | 9 pts |
| −7 | ~70% | ~85% | 15 pts |
| −10 | ~77% | ~91% | 14 pts |

That gap is systematic and it is the whole source of contrarian value. It also means the best deviations are usually mid-sized underdogs — around a touchdown — rather than the near coin flips, because that is where the public is most wrong relative to the true probability.

## Two prizes, opposite instructions

**Weekly** rewards variance. Fifteen games, 150 people, the winner needs 13 or 14. You cannot get there on a card that thirty people also submitted.

**Season** rewards accuracy. Over 270 picks, luck averages out and skill compounds. There is no room for a variance play to survive that long.

The tool resolves this by pricing both in money rather than blending them with an abstract weight — a weekly win and a season win are not the same unit, and no weighting makes them one. Tell it what each prize pays and it maximises expected winnings:

```bash
python pickem.py optimise --slate week_picks.csv \
    --entrants 150 --weeks 18 \
    --weekly-prize 50 --season-prize 500
```

Change those two numbers and the recommended number of deviations moves. If the season pot dwarfs the weekly pot, it will tell you to play close to chalk. That is correct.

## The easy way

Open `pickem.html` on your phone. It picks the current week from today's date and fills in the
matchups, so all you type is the spreads:

```
BUF @ KC -2.5
SEA @ SF -140
DAL @ PHI -6.5
```

Either a spread or a moneyline works on each line — whichever the book shows you first. A
moneyline is the more precise of the two, since it skips the rounding step of converting a
spread into a probability; use it when you have it. There's no field for the total: a straight
pick'em only rewards picking the winner, and the total doesn't change who's favoured to win, so
it can't change your card either.

Press **Build my card**. Refilling a week keeps any spreads you already typed, so you can update
one line without redoing the rest. If `schedule.json` hasn't been loaded yet, type the matchups
too — same format. It estimates the field's pick rates from the spreads, simulates your
pool a few thousand times, and hands back the card with the contrarian picks flagged. Copy
picks puts the list on your clipboard in slate order.

The spread is optional — drop it and it uses the ratings alone. If you happen to know the
field's actual pick rate on a game, put it on the end (`BUF @ KC -2.5 29`) and it uses that
instead of estimating.

Your slate, settings and pool history stay on the device. Nothing is sent anywhere.

After the deadline, open **Teach it your pool**, paste the same lines with the real pick
percentages, and hit Add to history. Twelve games gets a first fit; sixty makes it stable.
That's the whole weekly loop — two pastes, no files.

## The command-line way

**Before the deadline.** Build `week_picks.csv`. Only two columns are required; the rest improve it.

```csv
home_team,away_team,spread_home,moneyline_home,public_home_pct,model_home_prob
KC,BUF,-2.5,,,
PHI,DAL,,-140,0.79,
```

`moneyline_home` is optional and preferred over `spread_home` when both are present, for the
same reason as the paste box: it's the book's probability with no conversion step in between.

Leave `public_home_pct` blank and it is estimated from the spread. Leave `model_home_prob` blank and it comes from your ratings file. Then:

```bash
python pickem.py optimise --slate week_picks.csv
```

**After the deadline,** when the pool's picks become visible: log them. This is the highest-value fifteen minutes of your week.

```csv
week,home_team,away_team,spread_home,public_home_pct
1,KC,BUF,-2.5,0.71
```

After six or seven weeks:

```bash
python pickem.py calibrate --history data/field_history.csv
```

That fits how *your* 150 people behave, rather than a national average. If your pool is chalkier than typical, there is more contrarian value in it and the tool will find more of it. If they're sharp, deviations are worth less and it will tell you to stay closer to your model's favourites. Either way you are no longer guessing.

## What to expect

Your realistic ceiling is somewhere around 2% per week and 2–3% for the season, against a 0.67% random baseline. That is three or four times better than a coin flip and it still means you lose almost every week. A 150-person pool is mostly a lottery with a small skill component, and no amount of modelling changes that.

What it does change: over a full season you should expect roughly one weekly win rather than roughly zero, and a real shot at the season prize instead of a rounding error. For a fun experiment against 150 people, that's about as good as the maths allows.

## The failure mode to avoid

After a week where your three contrarian picks all lose and a chalk-picker wins, the instinct is to go chalk. Don't. The frontier table is a statement about the strategy, not about last Sunday. Three underdogs at 40% each miss all three about 22% of the time, which will happen to you four times this season.

The corresponding trap in the other direction: after a contrarian week hits, taking six deviations next week. The table already says six is worse than three.
