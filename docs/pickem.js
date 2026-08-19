/* =============================================================================
   PICK'EM POOL OPTIMISER — browser engine
   A port of pickem.py. Everything runs locally; nothing is sent anywhere.

   The trick that makes this fast enough for a phone: the field's picks don't
   depend on YOUR picks. So the opponents get simulated once per slate, and each
   candidate card is then scored in a single pass over the stored outcomes.
   Without that, hill-climbing 15 games would be a hundred full pool simulations.
   ========================================================================== */

const WSIMS = 6000;      // weekly simulations
const SSIMS = 1200;      // season simulations (each one plays every week)
const MAX_DEV = 6;

const DEFAULTS = {
  entrants: 150, weeksLeft: 18, wPrize: 50, sPrize: 500,
  chalkSpread: 0.38, maxPickRate: 0.985,
  chalkBloc: 0.22,   // submit the favourite in every game, every week
  sharpFrac: 0.08,   // about as good as your model
  publicBeta: 0.236  // 3-pt fav → 67% of picks; 7-pt fav → 85%
};

const ALIASES = {
  WSH:'WAS', WFT:'WAS', JAC:'JAX', LA:'LAR', STL:'LAR', SD:'LAC', OAK:'LV',
  TAM:'TB', KAN:'KC', GNB:'GB', NWE:'NE', NOR:'NO', SFO:'SF', ARZ:'ARI',
  CLV:'CLE', BLT:'BAL', HST:'HOU'
};

let MODEL = null, SCHED = null, SLATE = null, POOL = null, LADDER = null, FRONT = null;
const $ = id => document.getElementById(id);

/* ---------- small maths ---------- */
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const erf = x => {                        // Abramowitz & Stegun 7.1.26
  const s = Math.sign(x); x = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * x);
  const y = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t
            - 0.284496736) * t + 0.254829592) * t * Math.exp(-x * x);
  return s * y;
};
const normCdf = z => 0.5 * (1 + erf(z / Math.SQRT2));
const probFromSpread = sp => normCdf((-sp) / 13.5);
const spreadFromProb = p => {
  // invert the above by bisection; only used when no spread was given
  let lo = -25, hi = 25;
  for (let i = 0; i < 60; i++){
    const m = (lo + hi) / 2;
    probFromSpread(m) > p ? lo = m : hi = m;
  }
  return (lo + hi) / 2;
};
const publicFromSpread = (sp, beta) => 1 / (1 + Math.exp(-beta * (-sp)));
// Raw implied probability from a single American price. Still has the book's
// vig baked in (no opposing price to devig against) - same as reading a
// probability off a spread does; both are market-implied, not fair.
const impliedProbML = odds => odds < 0 ? -odds / (-odds + 100) : 100 / (odds + 100);
const pctS = x => (100 * x).toFixed(1) + '%';

let spare = null;
function gauss(){
  if (spare !== null){ const s = spare; spare = null; return s; }
  let u, v, s;
  do { u = Math.random() * 2 - 1; v = Math.random() * 2 - 1; s = u * u + v * v; }
  while (s >= 1 || s === 0);
  const f = Math.sqrt(-2 * Math.log(s) / s);
  spare = v * f; return u * f;
}

/* ---------- settings, remembered on this device ---------- */
function settings(){
  const s = { ...DEFAULTS };
  ['entrants','weeksLeft','wPrize','sPrize'].forEach(k => {
    const v = parseFloat($(k).value);
    if (isFinite(v)) s[k] = v;
  });
  s.publicBeta = calibratedBeta() ?? DEFAULTS.publicBeta;
  return s;
}
function store(k, v){ try { localStorage.setItem('nflpick.' + k, JSON.stringify(v)); } catch(e){} }
function recall(k, d){ try { const v = localStorage.getItem('nflpick.' + k); return v ? JSON.parse(v) : d; } catch(e){ return d; } }

/* ---------- your pool's behaviour, learned from logged weeks ---------- */
function calibratedBeta(){
  const h = recall('history', []);
  const use = h.filter(r => isFinite(r.spread) && r.pub > 0.01 && r.pub < 0.99 && r.spread !== 0);
  if (use.length < 12) return null;
  let xy = 0, xx = 0;
  use.forEach(r => {
    const x = -r.spread, y = Math.log(r.pub / (1 - r.pub));
    xy += x * y; xx += x * x;
  });
  return xx > 0 ? xy / xx : null;
}

/* ---------- parsing ---------- */
function normTeam(t){
  t = t.toUpperCase().replace(/[^A-Z]/g, '');
  return ALIASES[t] || t;
}

function parseSlate(text, cfg){
  const games = [], bad = [];
  text.split('\n').forEach((raw, n) => {
    const line = raw.trim();
    if (!line) return;
    const parts = line.replace(/[,;]/g, ' ').split(/\s+/);
    const at = parts.findIndex(p => /^(@|at|vs\.?|v)$/i.test(p));
    if (at < 1 || at >= parts.length - 1){
      bad.push(`line ${n + 1}: "${line}" — needs an @ between the teams`);
      return;
    }
    const away = normTeam(parts[at - 1]);
    const home = normTeam(parts[at + 1]);
    if (!away || !home){ bad.push(`line ${n + 1}: couldn't read the team names`); return; }

    const nums = parts.slice(at + 2)
      .map(p => parseFloat(p))
      .filter(v => isFinite(v));
    // The first number is either a point spread or an American moneyline -
    // whichever the book shows first. |value| >= 100 can only be odds; NFL
    // spreads don't reach 30, and odds are never quoted closer to zero than 100.
    let spread = null, mlHome = null;
    if (nums.length){
      if (Math.abs(nums[0]) >= 100) mlHome = nums[0];
      else spread = nums[0];
    }
    let pub = nums.length > 1 ? nums[1] : null;
    if (pub !== null && pub > 1) pub /= 100;           // 71 and 0.71 both work

    games.push({ home, away, spread, mlHome, pubGiven: pub });
  });
  return { games, bad };
}

/* ---------- model probability ---------- */
function ratingsProb(home, away){
  if (!MODEL || MODEL.placeholder) return null;
  const t = MODEL.teams;
  if (!t[home] || !t[away]) return null;
  const L = MODEL.league, P = MODEL.params;
  const muH = L.ppg * t[home].att * t[away].def + P.hfa_points / 2;
  const muA = L.ppg * t[away].att * t[home].def - P.hfa_points / 2;

  // A light version of the possession simulation — enough for a win probability.
  const rate = mu => {
    const D = P.drives_per_team;
    const base = P.base_td_rate * 6.95 + P.base_fg_rate * 3;
    const k = mu / (D * base);
    let td = P.base_td_rate * Math.pow(k, P.td_elasticity);
    let fg = P.base_fg_rate * Math.pow(k, P.fg_elasticity);
    const sc = (mu / D) / (td * 6.95 + fg * 3);
    return [td * sc, fg * sc];
  };
  const [tdH, fgH] = rate(muH), [tdA, fgA] = rate(muA);
  const N = 8000; let win = 0;
  for (let i = 0; i < N; i++){
    const env = Math.exp(gauss() * P.env_sigma);
    const d = clamp(Math.round(P.drives_per_team + gauss() * P.drive_sd), 7, 16);
    const roll = (td, fg) => {
      let pts = 0;
      for (let j = 0; j < d; j++){
        const u = Math.random();
        if (u < td) pts += Math.random() < 0.09
          ? (Math.random() < 0.48 ? 8 : 6)
          : (Math.random() < 0.955 ? 7 : 6);
        else if (u < td + fg) pts += 3;
      }
      return pts;
    };
    const h = roll(clamp(tdH * env, .01, .75), clamp(fgH * Math.sqrt(env), .01, .45));
    const a = roll(clamp(tdA * env, .01, .75), clamp(fgA * Math.sqrt(env), .01, .45));
    if (h > a) win++; else if (h === a) win += 0.5;
  }
  return win / N;
}

function buildSlate(games, cfg){
  return games.map(g => {
    let p = ratingsProb(g.home, g.away);
    let spread = g.spread;
    // A moneyline gives the win probability directly - no need to round-trip
    // it through the spread-to-probability approximation, which gets shakier
    // the bigger the spread.
    const marketP = g.mlHome !== null
      ? impliedProbML(g.mlHome)
      : (spread !== null ? probFromSpread(spread) : null);

    if (marketP === null && p === null) p = 0.5;
    // A market price is a very strong predictor on its own. When both exist,
    // lean on the market — it has injuries and weather in it that the
    // ratings do not.
    else if (p === null) p = marketP;
    else if (marketP !== null) p = 0.35 * p + 0.65 * marketP;

    // Keep a spread around too, for the public-pick-rate estimate and for
    // display, even when the input was a moneyline.
    if (spread === null) spread = spreadFromProb(marketP !== null ? marketP : p);

    const pub = g.pubGiven !== null && g.pubGiven !== undefined
      ? g.pubGiven
      : publicFromSpread(spread, cfg.publicBeta);

    return { ...g, spread, p: clamp(p, 0.03, 0.97), pub: clamp(pub, 0.02, 0.98) };
  });
}

/* ---------- the pool ----------
   Simulated once per slate. Opponent scores never depend on your card, so the
   expensive part happens here and every candidate afterwards is nearly free. */
async function simulatePool(slate, cfg, onProgress){
  const G = slate.length;
  const p = Float64Array.from(slate.map(g => g.p));
  const pub = Float64Array.from(slate.map(g => g.pub));
  const chalk = Uint8Array.from(slate.map(g => g.pub > 0.5 ? 1 : 0));
  const sharp = Uint8Array.from(slate.map(g => g.p > 0.5 ? 1 : 0));

  const nOpp = Math.max(1, Math.round(cfg.entrants) - 1);
  const nChalk = Math.round(nOpp * cfg.chalkBloc);
  const nSharp = Math.round(nOpp * cfg.sharpFrac);
  const nNoisy = Math.max(0, nOpp - nChalk - nSharp);
  const W = Math.max(1, Math.round(cfg.weeksLeft));

  const outW = new Uint8Array(WSIMS * G);
  const bestW = new Int16Array(WSIMS), tieW = new Int16Array(WSIMS);
  const outS = new Uint8Array(SSIMS * W * G);
  const bestS = new Int16Array(SSIMS), tieS = new Int16Array(SSIMS);
  const z = new Float64Array(nNoisy);

  // score one deterministic bloc against an outcome slice
  const blocScore = (out, off, pick) => {
    let s = 0;
    for (let g = 0; g < G; g++) if (out[off + g] === pick[g]) s++;
    return s;
  };

  // ---- weekly ----
  let i = 0;
  while (i < WSIMS){
    const stop = Math.min(i + 400, WSIMS);
    for (; i < stop; i++){
      const off = i * G;
      for (let g = 0; g < G; g++) outW[off + g] = Math.random() < p[g] ? 1 : 0;
      for (let o = 0; o < nNoisy; o++) z[o] = gauss() * cfg.chalkSpread;

      let best = -1, tie = 0;
      const consider = (score, count) => {
        if (count <= 0) return;
        if (score > best){ best = score; tie = count; }
        else if (score === best) tie += count;
      };
      consider(blocScore(outW, off, chalk), nChalk);
      consider(blocScore(outW, off, sharp), nSharp);
      for (let o = 0; o < nNoisy; o++){
        let s = 0;
        for (let g = 0; g < G; g++){
          let lean = 0.5 + (pub[g] - 0.5) * (1 + z[o]);
          if (lean > cfg.maxPickRate) lean = cfg.maxPickRate;
          else if (lean < 1 - cfg.maxPickRate) lean = 1 - cfg.maxPickRate;
          if ((Math.random() < lean ? 1 : 0) === outW[off + g]) s++;
        }
        consider(s, 1);
      }
      bestW[i] = best; tieW[i] = tie;
    }
    onProgress(0.5 * i / WSIMS);
    await new Promise(r => setTimeout(r, 0));
  }

  // ---- season: opponent styles persist for the whole year ----
  let j = 0;
  while (j < SSIMS){
    const stop = Math.min(j + 60, SSIMS);
    for (; j < stop; j++){
      for (let o = 0; o < nNoisy; o++) z[o] = gauss() * cfg.chalkSpread;
      const base = j * W * G;
      let cScore = 0, sScore = 0;
      const noisy = new Int16Array(nNoisy);

      for (let w = 0; w < W; w++){
        const off = base + w * G;
        for (let g = 0; g < G; g++) outS[off + g] = Math.random() < p[g] ? 1 : 0;
        cScore += blocScore(outS, off, chalk);
        sScore += blocScore(outS, off, sharp);
        for (let o = 0; o < nNoisy; o++){
          let s = 0;
          for (let g = 0; g < G; g++){
            let lean = 0.5 + (pub[g] - 0.5) * (1 + z[o]);
            if (lean > cfg.maxPickRate) lean = cfg.maxPickRate;
            else if (lean < 1 - cfg.maxPickRate) lean = 1 - cfg.maxPickRate;
            if ((Math.random() < lean ? 1 : 0) === outS[off + g]) s++;
          }
          noisy[o] += s;
        }
      }
      let best = -1, tie = 0;
      const consider = (score, count) => {
        if (count <= 0) return;
        if (score > best){ best = score; tie = count; }
        else if (score === best) tie += count;
      };
      consider(cScore, nChalk);
      consider(sScore, nSharp);
      for (let o = 0; o < nNoisy; o++) consider(noisy[o], 1);
      bestS[j] = best; tieS[j] = tie;
    }
    onProgress(0.5 + 0.5 * j / SSIMS);
    await new Promise(r => setTimeout(r, 0));
  }

  return { G, W, outW, bestW, tieW, outS, bestS, tieS, chalk, sharp };
}

function evalWeek(picks, P){
  const { G, outW, bestW, tieW } = P;
  let wins = 0, correct = 0;
  for (let i = 0; i < WSIMS; i++){
    const off = i * G;
    let s = 0;
    for (let g = 0; g < G; g++) if (outW[off + g] === picks[g]) s++;
    correct += s;
    if (s > bestW[i]) wins += 1;
    else if (s === bestW[i]) wins += 1 / (1 + tieW[i]);
  }
  return { pWin: wins / WSIMS, expCorrect: correct / WSIMS };
}

function evalSeason(picks, P){
  const { G, W, outS, bestS, tieS } = P;
  let wins = 0;
  for (let i = 0; i < SSIMS; i++){
    let tot = 0;
    const base = i * W * G;
    for (let w = 0; w < W; w++){
      const off = base + w * G;
      for (let g = 0; g < G; g++) if (outS[off + g] === picks[g]) tot++;
    }
    if (tot > bestS[i]) wins += 1;
    else if (tot === bestS[i]) wins += 1 / (1 + tieS[i]);
  }
  return { pWin: wins / SSIMS };
}

/* ---------- choosing the card ----------
   Each rung adds the single flip that most improves weekly win probability, so
   the k-th card is a sensible k-contrarian card. Exhaustive search would be
   2^15 evaluations for a difference you couldn't measure through the noise. */
function ladder(P, cfg){
  const G = P.G;
  const picks = Uint8Array.from(P.chalk);
  const rungs = [Uint8Array.from(picks)];
  const used = new Set();

  for (let k = 0; k < Math.min(MAX_DEV, G); k++){
    const base = evalWeek(picks, P).pWin;
    let bestGain = -1e9, bestI = -1;
    for (let i = 0; i < G; i++){
      if (used.has(i)) continue;
      picks[i] ^= 1;
      const gain = evalWeek(picks, P).pWin - base;
      picks[i] ^= 1;
      if (gain > bestGain){ bestGain = gain; bestI = i; }
    }
    if (bestI < 0) break;
    picks[bestI] ^= 1;
    used.add(bestI);
    rungs.push(Uint8Array.from(picks));
  }
  return rungs;
}

function frontier(rungs, P, cfg){
  return rungs.map((pk, k) => {
    const w = evalWeek(pk, P), s = evalSeason(pk, P);
    return {
      k,
      expCorrect: w.expCorrect,
      pWeek: w.pWin,
      pSeason: s.pWin,
      ev: cfg.wPrize * w.pWin * cfg.weeksLeft + cfg.sPrize * s.pWin
    };
  });
}

/* ---------- rendering ---------- */
function renderCard(k){
  const picks = LADDER[Math.min(k, LADDER.length - 1)];
  let devs = 0;
  const html = SLATE.map((g, i) => {
    const tookHome = picks[i] === 1;
    const side = tookHome ? g.home : g.away;
    const other = tookHome ? g.away : g.home;
    const myP = tookHome ? g.p : 1 - g.p;
    const fld = tookHome ? g.pub : 1 - g.pub;
    const chalkSide = g.pub > 0.5 ? g.home : g.away;
    const dev = side !== chalkSide;
    if (dev) devs++;
    const lev = myP - fld;
    return `<li class="${dev ? 'dev' : ''}">
      <span class="pickname">${side}</span>
      <span class="meta">over ${other} · <b>${g.mlHome !== null
        ? (g.mlHome > 0 ? '+' : '') + g.mlHome
        : (g.spread > 0 ? '+' : '') + g.spread.toFixed(1)}</b><br>
        you ${pctS(myP)} · field ${pctS(fld)}</span>
      <span class="lev ${lev > 0 ? 'up' : 'down'}">${lev > 0 ? '+' : ''}${(100 * lev).toFixed(1)}
        ${dev ? '<span class="flagword">against field</span>' : ''}</span>
    </li>`;
  }).join('');
  $('card').innerHTML = html;
  $('devCount').textContent = devs;
}

function renderFrontier(cfg){
  const bestK = FRONT.reduce((a, b) => b.ev > a.ev ? b : a).k;
  $('frontier').querySelector('tbody').innerHTML = FRONT.map(r => `
    <tr class="${r.k === bestK ? 'best' : ''}">
      <td>${r.k}${r.k === 0 ? ' (chalk)' : ''}</td>
      <td>${r.expCorrect.toFixed(2)}</td>
      <td>${pctS(r.pWeek)}</td>
      <td>${pctS(r.pSeason)}</td>
      <td>$${r.ev.toFixed(2)}</td>
    </tr>`).join('');

  const rnd = 1 / cfg.entrants;
  $('frontNote').innerHTML =
    `A random card wins ${pctS(rnd)} of the time. Chalk wins the week ${pctS(FRONT[0].pWeek)} — `
    + `worse than random, because the people who submit the identical card split the prize with you. `
    + `The rows within a dollar or two of the best are inside the simulation's noise; `
    + `treat "two or three" as the answer rather than agonising over which.`;

  $('devSel').innerHTML = FRONT.map(r =>
    `<option value="${r.k}" ${r.k === bestK ? 'selected' : ''}>${r.k}</option>`).join('');
  renderCard(bestK);
}

function copyPicks(){
  const k = parseInt($('devSel').value, 10);
  const picks = LADDER[Math.min(k, LADDER.length - 1)];
  const text = SLATE.map((g, i) => picks[i] === 1 ? g.home : g.away).join('\n');
  navigator.clipboard?.writeText(text).then(
    () => { $('copy').textContent = 'Copied'; setTimeout(() => $('copy').textContent = 'Copy picks', 1600); },
    () => { $('copy').textContent = 'Copy failed'; }
  );
}

/* ---------- the schedule ----------
   Optional. Without schedule.json the page works exactly as before; with it,
   picking a week fills the matchups so the only thing left to type is spreads. */
function currentWeek(){
  // Week 1 of the 2026 season opens Wednesday 9 September.
  const opener = new Date(2026, 8, 9);
  const days = Math.floor((Date.now() - opener) / 86400000);
  return clamp(Math.floor(days / 7) + 1, 1, 18);
}

function fillWeek(){
  if (!SCHED) return;
  const wk = $('weekSel').value;
  const games = SCHED.weeks[wk] || [];
  if (!games.length) return;

  // Keep any spreads already typed for the same matchup, so refilling after a
  // line move doesn't wipe your work.
  const known = {};
  parseSlate($('slateBox').value, settings()).games.forEach(g => {
    if (g.spread !== null || g.mlHome !== null) known[g.away + '@' + g.home] = g;
  });

  $('slateBox').value = games.map(g => {
    const prev = known[g.away + '@' + g.home];
    const line = prev && prev.mlHome !== null ? ' ' + prev.mlHome
               : prev && prev.spread !== null ? ' ' + prev.spread : '';
    const pub = prev && prev.pubGiven ? ' ' + Math.round(100 * prev.pubGiven) : '';
    return `${g.away} @ ${g.home}${line}${pub}`;
  }).join('\n');
  store('slate', $('slateBox').value);
  $('status').textContent = `Week ${wk}: ${games.length} games. Add a spread or moneyline to each, then build.`;
}

/* ---------- actions ---------- */
async function build(){
  const cfg = settings();
  const text = $('slateBox').value;
  store('slate', text);
  ['entrants','weeksLeft','wPrize','sPrize'].forEach(k => store(k, $(k).value));

  const { games, bad } = parseSlate(text, cfg);
  $('parseErr').hidden = bad.length === 0;
  $('parseErr').textContent = bad.join('\n');
  if (!games.length){
    $('status').textContent = 'Nothing to build — paste a slate above.';
    return;
  }

  $('build').disabled = true;
  $('status').textContent = 'Simulating the pool…';
  await new Promise(r => setTimeout(r, 20));

  SLATE = buildSlate(games, cfg);
  POOL = await simulatePool(SLATE, cfg, f => {
    $('status').textContent = `Simulating the pool… ${Math.round(100 * f)}%`;
  });
  LADDER = ladder(POOL, cfg);
  FRONT = frontier(LADDER, POOL, cfg);

  $('cardPanel').hidden = false;
  $('frontPanel').hidden = false;
  renderFrontier(cfg);
  $('status').textContent = `${SLATE.length} games · ${cfg.entrants} entrants`;
  $('build').disabled = false;
}

function addHistory(){
  const cfg = { ...DEFAULTS };
  const { games, bad } = parseSlate($('histBox').value, cfg);
  const rows = recall('history', []);
  let added = 0;
  games.forEach(g => {
    if (g.pubGiven === null || g.pubGiven === undefined || g.spread === null) return;
    rows.push({ spread: g.spread, pub: g.pubGiven });
    added++;
  });
  store('history', rows);
  $('histBox').value = '';
  $('calibOut').innerHTML = added
    ? `Added <strong>${added}</strong> games.` + renderCalib()
    : `Nothing added — each line needs both a spread and a pick percentage.`;
  updateBadge();
}

function renderCalib(){
  const rows = recall('history', []);
  const beta = calibratedBeta();
  if (beta === null){
    return `<br>Logged <strong>${rows.length}</strong> games so far. `
         + `Twelve is the minimum for a fit; sixty gives a stable one.`;
  }
  const at = s => (100 / (1 + Math.exp(-beta * s))).toFixed(0);
  const cmp = beta > DEFAULTS.publicBeta * 1.08
    ? 'Your pool is chalkier than typical — good news, there is more contrarian value in it.'
    : beta < DEFAULTS.publicBeta * 0.92
      ? 'Your pool is sharper than typical. Deviations are worth less here; lean closer to your favourites.'
      : 'Your pool behaves close to a typical one.';
  return `<br>Logged <strong>${rows.length}</strong> games. Fitted from your pool:
    a 3-point favourite draws <strong>${at(3)}%</strong> of picks,
    a 7-point favourite <strong>${at(7)}%</strong>,
    a 10-point favourite <strong>${at(10)}%</strong>.<br>${cmp}`;
}

function updateBadge(){
  const n = recall('history', []).length;
  const fitted = calibratedBeta() !== null;
  $('calibBadge').textContent = n ? (fitted ? `· fitted on ${n} games` : `· ${n} logged`) : '';
}

/* ---------- boot ---------- */
Promise.all([
  fetch('data/model.json?v=' + Date.now()).then(r => r.ok ? r.json() : null).catch(() => null),
  fetch('data/schedule.json?v=' + Date.now()).then(r => r.ok ? r.json() : null).catch(() => null),
])
  .then(([m, s]) => { MODEL = m; SCHED = s; })
  .finally(() => {
    if (MODEL){
      $('genDate').textContent = 'ratings ' + MODEL.generated;
      if (MODEL.placeholder) $('stale').hidden = false;
    } else {
      $('genDate').textContent = 'no ratings loaded — using spreads only';
      $('stale').hidden = false;
    }
    ['entrants','weeksLeft','wPrize','sPrize'].forEach(k => {
      const v = recall(k, null);
      if (v !== null) $(k).value = v;
    });
    $('slateBox').value = recall('slate', '');

    if (SCHED && SCHED.weeks && Object.keys(SCHED.weeks).length){
      const weeks = Object.keys(SCHED.weeks).sort((a, b) => a - b);
      $('weekSel').innerHTML = weeks.map(w =>
        `<option value="${w}">${w}</option>`).join('');
      const wk = String(clamp(currentWeek(), +weeks[0], +weeks[weeks.length - 1]));
      $('weekSel').value = SCHED.weeks[wk] ? wk : weeks[0];
      $('weekBar').hidden = false;
      $('weeksLeft').value = recall('weeksLeft', Math.max(1, 19 - currentWeek()));
      $('loadWeek').addEventListener('click', fillWeek);
      if (!$('slateBox').value.trim()) fillWeek();
    }
    updateBadge();
    $('calibOut').innerHTML = renderCalib();

    $('build').addEventListener('click', build);
    $('clear').addEventListener('click', () => {
      $('slateBox').value = ''; store('slate', '');
      $('cardPanel').hidden = true; $('frontPanel').hidden = true;
      $('status').textContent = ''; $('parseErr').hidden = true;
    });
    $('copy').addEventListener('click', copyPicks);
    $('devSel').addEventListener('change', e => renderCard(parseInt(e.target.value, 10)));
    $('addHist').addEventListener('click', addHistory);
    $('resetHist').addEventListener('click', () => {
      store('history', []); updateBadge();
      $('calibOut').innerHTML = 'History cleared.';
    });
  });
