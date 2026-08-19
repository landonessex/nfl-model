# Putting this on GitHub Pages

## The approach, and why

GitHub Pages serves static files. It will not run Python. That single constraint decides the whole architecture, and the good news is it decides it in a nice direction:

**Split the work by how often it changes.**

| Job | Where it runs | How often |
|---|---|---|
| Pull results, fit ratings, fit the regression | GitHub Actions | Once a week |
| Simulate a matchup, price it, size a stake | The visitor's browser | Every keystroke |

Actions is a real Linux box with network access and a cron schedule — it can do everything your laptop does. It runs on Tuesday, refits, and commits `docs/data/model.json`. Pages then serves that JSON as a plain file.

The 20,000-simulation Monte Carlo is maybe 40ms of JavaScript, so there's no reason to precompute it. The browser does it live, which means you can drag the spread half a point and watch the cover probability move. That interactivity is the whole point of having a site rather than a spreadsheet.

It also means nothing you type is transmitted anywhere. Your bankroll and your bet sizes stay in the tab.

### What this rules out, and the workarounds

**Live odds.** A static site can't hold an API key — anything in the JS is public, and anyone can drain your Odds API quota. Two options:

1. *Bake them in.* Add a step to the workflow that fetches odds with a repo secret and writes them into `model.json`. Free, simple, and up to a week stale. Fine if you're using the site to think, and typing the current number in by hand when you're actually betting.
2. *Proxy them.* A Cloudflare Worker or Vercel function holds the key and forwards requests. Free tier, about twenty lines, and gets you genuinely live prices. Worth doing if you check the site more than a couple of times a week.

**Privacy.** A public repo means a public site and a public model. Private repos need GitHub Pro for Pages. If you'd rather not publish your ratings: Cloudflare Pages or Netlify both serve private repos free and take the same folder.

**Pool history.** The pick'em page keeps your logged field percentages in the browser's own
storage on whichever device you used. That's deliberate — no accounts, no backend — but it does
mean clearing site data loses it. If you'd rather it followed you between phone and laptop, log
the numbers into `data/field_history.csv` in the repo and run `pickem.py calibrate` instead.

**Bet history.** No database. Keep using the `BetLog` sheet, or add a `bets.json` the workflow reads from a Gist. Don't put it in `localStorage` and call it a record — you'll clear your browser one day.

---

## Setup, about fifteen minutes

```
nfl-model/
├── docs/                    ← Pages serves this folder
│   ├── index.html           ← game model: ratings, sims, pricing
│   ├── pickem.html          ← pool optimiser: paste a slate, get a card
│   ├── app.js  pickem.js
│   ├── styles.css  pickem.css
│   └── data/
│       ├── model.json      ← written by the workflow
│       └── schedule.json   ← full 2026 slate, baked in once (see below)
├── data/
│   ├── games.csv            ← results, fetched weekly
│   ├── priors.csv           ← preseason ratings, you write these once
│   └── slate.csv            ← optional
├── scripts/
│   ├── export_web.py
│   ├── fetch_data.py
│   └── fetch_schedule.py
├── .github/workflows/update-model.yml
├── nfl_model.py
└── requirements.txt
```

**1. Create the repo and push this folder.**

```bash
git init && git add . && git commit -m "NFL model"
git remote add origin git@github.com:YOU/nfl-model.git
git push -u origin main
```

**2. Turn on Pages.** Settings → Pages → Source: *Deploy from a branch* → Branch `main`, folder `/docs`. Live at `https://YOU.github.io/nfl-model/` in about a minute.

**3. Let Actions write to the repo.** Settings → Actions → General → Workflow permissions → *Read and write*. Without this the weekly commit fails with a 403, which is the single most common way this setup breaks.

**4. Load real data.** The repo ships with placeholder ratings so the page renders immediately — you'll see an amber banner saying so. To replace them:

```bash
pip install -r requirements.txt
python scripts/fetch_data.py --season 2026 --history 2   # test this locally first
python scripts/export_web.py --games data/games.csv --week 1
git add data docs/data && git commit -m "Real ratings" && git push
```

Test `fetch_data.py` on your own machine before trusting the schedule. The nflverse Python package has been renamed once already, and a rename will fail silently in Actions until you notice the site hasn't updated in three weeks.

**4b. Load the schedule, once.** This is what lets the pick'em page pre-fill each week's
matchups so you only type spreads.

```bash
python scripts/fetch_schedule.py --season 2026
git add docs/data/schedule.json && git commit -m "2026 schedule" && git push
```

The weekly job also runs it, so you can skip this and let Tuesday handle it. If the nflverse
package won't load, `--from-csv` takes a four-column file (`week,away_team,home_team,neutral`).
Without `schedule.json` the page falls back to typing matchups by hand — everything else works.

**5. Write `data/priors.csv` before Week 1.** Preseason win totals converted to points: a 10.5-win team is roughly +3, an 11.5-win team roughly +5, a 6.5-win team roughly -3. Without priors the model has no opinion in September, and the market blend will be doing all the work.

**6. Run it once by hand.** Actions tab → *Refit model* → Run workflow. Watch it go green before you rely on the cron.

---

## Alternatives, briefly

**Streamlit Community Cloud** — if you'd rather write the whole thing in Python and skip the JavaScript port. Free, runs the real `nfl_model.py`, and you get backtesting in the browser too. Costs you: it sleeps after inactivity and takes thirty seconds to wake, and the UI will look like every other Streamlit app.

**Observable Framework** — static build, excellent charts, designed for exactly this shape of problem. Genuinely nice. More build tooling than the plain HTML here.

**Just keep the spreadsheet.** If you're the only user, the workbook already does everything the site does and you don't have to maintain a deployment. The site earns its keep when you want to check a line from your phone, or show someone else the ratings without emailing an xlsx.

My recommendation is the setup in this folder. It's free forever, it has no runtime to break, and the only moving part is a weekly cron you can watch succeed or fail in one tab.

---

## When something breaks

**Site shows the placeholder banner** — `model.json` still has `"placeholder": true`. Run the export with a real `games.csv`.

**Workflow fails with a 403 on push** — step 3 above.

**Site doesn't update after a successful run** — Pages caches. Hard refresh; the app already cache-busts the JSON, but not itself.

**Ratings look wrong after an update** — check `gamesUsed` on the page. If it jumped or collapsed, `fetch_data.py` changed shape. Diff `data/games.csv` against the previous commit; the whole point of committing it is that you can.


## Keeping schedule.json current

`docs/data/schedule.json` has all 272 games for the 2026 regular season, baked in once — the
NFL doesn't change matchups mid-season, only occasionally the day/time for flex scheduling,
which doesn't matter here. The pick'em page uses it to pre-fill each week's matchups so you
only type spreads.

Nothing to maintain. If a future season needs it regenerated, pull the schedule from
Pro Football Reference (`pro-football-reference.com/years/YYYY/games.htm`) and reshape it to:

```json
{"weeks": {"1": [{"away": "NE", "home": "SEA"}, ...], "2": [...], ...}}
```
