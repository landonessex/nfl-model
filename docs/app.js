/* =============================================================================
   NFL Model — browser engine
   A port of the possession simulation, pricing and staking from nfl_model.py.
   The ratings are fitted server-side by a scheduled job and loaded as JSON;
   everything below runs locally, so nothing you type leaves the page.
   ========================================================================== */

const N_SIMS = 20000;
const CHUNK  = 2500;

let M = null;                       // model.json
let margins = new Int16Array(N_SIMS);
let totals  = new Int16Array(N_SIMS);
let done = 0, runToken = 0;

const $ = id => document.getElementById(id);
const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
const fmtSigned = x => (x > 0 ? '+' : '') + x.toFixed(1);
const pct = x => (100 * x).toFixed(1) + '%';

/* ---------- odds maths ---------- */
const impliedProb = o => o < 0 ? -o / (-o + 100) : 100 / (o + 100);
const toDecimal   = o => 1 + (o < 0 ? 100 / -o : o / 100);
const evPerUnit   = (p, o) => p * (toDecimal(o) - 1) - (1 - p);

function fairOdds(p){
  p = clamp(p, 1e-4, 1 - 1e-4);
  return p >= 0.5 ? -Math.round(100 * p / (1 - p)) : Math.round(100 * (1 - p) / p);
}

/* Power devig: solve a^k + b^k = 1. Handles lopsided markets better than
   simply scaling both sides, because books don't spread the margin evenly. */
function devig(a, b){
  let lo = 0.5, hi = 3;
  for (let i = 0; i < 120; i++){
    const k = (lo + hi) / 2;
    (Math.pow(a, k) + Math.pow(b, k) > 1) ? lo = k : hi = k;
  }
  const k = (lo + hi) / 2, A = Math.pow(a, k), B = Math.pow(b, k);
  return [A / (A + B), B / (A + B)];
}

function kelly(p, o, P){
  const b = toDecimal(o) - 1;
  const edge = p * b - (1 - p);
  if (edge <= 0) return 0;
  return Math.min((edge / b) * P.kelly_fraction, P.max_stake_pct);
}

/* ---------- random ---------- */
let spare = null;
function gauss(){
  if (spare !== null){ const s = spare; spare = null; return s; }
  let u, v, s;
  do { u = Math.random() * 2 - 1; v = Math.random() * 2 - 1; s = u * u + v * v; }
  while (s >= 1 || s === 0);
  const f = Math.sqrt(-2 * Math.log(s) / s);
  spare = v * f;
  return u * f;
}

/* ---------- expected points ---------- */
function expectedPoints(){
  const P = M.params, L = M.league;
  const h = M.teams[$('homeSel').value], a = M.teams[$('awaySel').value];
  const neutral = $('neutral').checked;

  // A: attack and defense ratings
  let muH = L.ppg * h.att * a.def;
  let muA = L.ppg * a.att * h.def;

  // B: the four-statistic regression, fed opponent-adjusted expected stats
  const c = M.coef;
  const box = (o, d) => M.intercept
    + c.pass_yds * (L.pass * o.passAtt * d.passDef)
    + c.rush_yds * (L.rush * o.rushAtt * d.rushDef)
    + c.takeaways * ((o.take + d.give) / 2)
    + c.giveaways * ((o.give + d.take) / 2);
  const bH = box(h, a), bA = box(a, h);

  const wr = P.w_ratings, wb = P.w_boxscore;
  muH = (wr * muH + wb * bH) / (wr + wb);
  muA = (wr * muA + wb * bA) / (wr + wb);

  // situational
  const rest = { normal: 0, bye: P.bye_week_bonus, short: P.short_week_penalty };
  let adjH = neutral ? 0 :  P.hfa_points / 2;
  let adjA = neutral ? 0 : -P.hfa_points / 2;
  adjH += rest[$('restH').value] / 2;
  adjA += rest[$('restA').value] / 2;
  muH += adjH; muA += adjA;

  // C: the market's own line, weighted heavily early and decaying
  const spread = parseFloat($('spread').value);
  const total  = parseFloat($('total').value);
  let blend = 0;
  if ($('useMarket').checked && isFinite(spread)){
    const wk = clamp(parseFloat($('week').value) || 1, 0, 22);
    const decay = Math.pow(0.5, wk / P.market_blend_halflife_weeks);
    blend = P.market_blend_floor + (P.market_blend_week1 - P.market_blend_floor) * decay;
    const T = isFinite(total) ? total : muH + muA;
    const mkH = (T - spread) / 2, mkA = (T + spread) / 2;
    muH = (1 - blend) * muH + blend * mkH;
    muA = (1 - blend) * muA + blend * mkA;
  }
  $('blendPct').textContent = $('useMarket').checked ? `now ${Math.round(blend * 100)}%` : 'off';

  return { muH: Math.max(muH, 6), muA: Math.max(muA, 6), power: [h.power, a.power] };
}

/* ---------- per-drive scoring rates ----------
   Touchdowns scale faster than field goals: a good offence converts, it doesn't
   merely kick more. Both rates are then rescaled to hit the expected score. */
function driveRates(mu){
  const P = M.params, D = P.drives_per_team;
  const basePpd = P.base_td_rate * 6.95 + P.base_fg_rate * 3;
  const k = mu / (D * basePpd);
  let td = P.base_td_rate * Math.pow(k, P.td_elasticity);
  let fg = P.base_fg_rate * Math.pow(k, P.fg_elasticity);
  const scale = (mu / D) / (td * 6.95 + fg * 3);
  td *= scale; fg *= scale;
  if (td + fg > 0.92){ const s = 0.92 / (td + fg); td *= s; fg *= s; }
  return [td, fg];
}

function scoreDrives(n, td, fg, P){
  let pts = 0;
  for (let i = 0; i < n; i++){
    const u = Math.random();
    if (u < td){
      pts += 6;
      if (Math.random() < P.two_point_attempt_rate){
        if (Math.random() < P.two_point_success) pts += 2;
      } else if (Math.random() < P.xp_success) pts += 1;
    } else if (u < td + fg){
      pts += 3;
    } else if (u < td + fg + P.base_safety_rate){
      // conceded safety shows up as two points the other way; close enough at this scale
    }
  }
  return pts;
}

function simulateChunk(from, to, muH, muA){
  const P = M.params;
  const [tdH, fgH] = driveRates(muH);
  const [tdA, fgA] = driveRates(muA);

  for (let i = from; i < to; i++){
    // one shared environment shock, so the two scores move together:
    // weather, pace, whether it turns into a shootout
    const env = Math.exp(gauss() * P.env_sigma);
    const drives = clamp(Math.round(P.drives_per_team + gauss() * P.drive_sd), 7, 16);

    const th = clamp(tdH * env, .01, .75), fh = clamp(fgH * Math.sqrt(env), .01, .45);
    const ta = clamp(tdA * env, .01, .75), fa = clamp(fgA * Math.sqrt(env), .01, .45);

    let hp = scoreDrives(drives, th, fh, P);
    let ap = scoreDrives(drives, ta, fa, P);

    // Overtime: both teams get a possession, then next score wins. A regular
    // season game may still end tied, which pushes the moneyline.
    if (hp === ap){
      for (let d = 0; d < P.ot_length_drives && hp === ap; d++){
        const uh = Math.random(), ua = Math.random();
        hp += uh < th ? 7 : uh < th + fh ? 3 : 0;
        ap += ua < ta ? 7 : ua < ta + fa ? 3 : 0;
      }
    }
    margins[i] = hp - ap;
    totals[i]  = hp + ap;
  }
}

/* ---------- pricing off the simulated distribution ---------- */
function priceAll(){
  const n = done;
  let hw = 0, tie = 0, st = 0;
  for (let i = 0; i < n; i++){
    const m = margins[i];
    if (m > 0) hw++; else if (m === 0) tie++;
    st += totals[i];
  }
  const pHome = (hw + 0.5 * tie) / n;

  const spread = parseFloat($('spread').value);
  let cH = 0, cA = 0, push = 0;
  for (let i = 0; i < n; i++){
    const v = margins[i] + spread;
    if (v > 0) cH++; else if (v < 0) cA++; else push++;
  }
  const decided = n - push || 1;

  const line = parseFloat($('total').value);
  let ov = 0, un = 0, tp = 0;
  for (let i = 0; i < n; i++){
    if (totals[i] > line) ov++; else if (totals[i] < line) un++; else tp++;
  }
  const tDecided = n - tp || 1;

  return {
    pHome, pAway: 1 - pHome, meanTotal: st / n,
    pCoverH: cH / decided, pCoverA: cA / decided,
    pOver: ov / tDecided, pUnder: un / tDecided
  };
}

/* ---------- rendering ---------- */
function renderNumbers(ep){
  const n = done || 1;
  let sh = 0, sa = 0;
  for (let i = 0; i < n; i++){ sh += (totals[i] + margins[i]) / 2; sa += (totals[i] - margins[i]) / 2; }
  const mH = sh / n, mA = sa / n;

  $('homeScore').textContent = mH.toFixed(1);
  $('awayScore').textContent = mA.toFixed(1);
  $('homePower').textContent = 'power ' + fmtSigned(ep.power[0]);
  $('awayPower').textContent = 'power ' + fmtSigned(ep.power[1]);

  const line = -(mH - mA);
  $('modelLine').textContent  = `${$('homeSel').value} ${fmtSigned(line)}`;
  $('modelTotal').textContent = (mH + mA).toFixed(1);

  const sp = parseFloat($('spread').value), tt = parseFloat($('total').value);
  $('marketLineOut').textContent  = isFinite(sp) ? `${$('homeSel').value} ${fmtSigned(sp)}` : '—';
  $('marketTotalOut').textContent = isFinite(tt) ? tt.toFixed(1) : '—';
  $('simCount').textContent = done.toLocaleString();
}

function renderHistogram(){
  const cv = $('hist'), ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height, pad = 34, base = H - 30;
  ctx.clearRect(0, 0, W, H);
  if (!done) return;

  const LO = -35, HI = 35, span = HI - LO + 1;
  const counts = new Int32Array(span);
  for (let i = 0; i < done; i++) counts[clamp(margins[i], LO, HI) - LO]++;
  const max = Math.max(...counts) || 1;
  const bw = (W - pad * 2) / span;
  const x = m => pad + (m - LO) * bw;

  // baseline
  ctx.strokeStyle = '#2a3947'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad, base + .5); ctx.lineTo(W - pad, base + .5); ctx.stroke();

  // hash marks on the key numbers
  ctx.setLineDash([3, 5]); ctx.strokeStyle = 'rgba(238,242,246,.22)';
  [3, 7, -3, -7].forEach(k => {
    const px = x(k) + bw / 2;
    ctx.beginPath(); ctx.moveTo(px, 18); ctx.lineTo(px, base); ctx.stroke();
  });
  ctx.setLineDash([]);

  // the market's spread
  const sp = parseFloat($('spread').value);
  if (isFinite(sp)){
    const px = x(-sp) + bw / 2;
    ctx.setLineDash([6, 4]); ctx.strokeStyle = '#6e93c4'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(px, 12); ctx.lineTo(px, base); ctx.stroke();
    ctx.setLineDash([]);
  }

  // bars
  for (let i = 0; i < span; i++){
    const m = i + LO, h = (counts[i] / max) * (base - 40);
    ctx.fillStyle = m > 0 ? '#e0a33c' : m < 0 ? '#5d7ea8' : '#8ea0b3';
    if (Math.abs(m) === 3 || Math.abs(m) === 7) ctx.fillStyle = m > 0 ? '#f2be62' : '#7fa3cd';
    ctx.fillRect(x(m) + 1, base - h, Math.max(bw - 2, 1), h);
  }

  // axis labels
  ctx.fillStyle = '#61748a';
  ctx.font = '500 11px "IBM Plex Mono", monospace';
  ctx.textAlign = 'center';
  for (let m = LO + 5; m <= HI; m += 7){
    ctx.fillText(String(Math.abs(m)), x(m) + bw / 2, base + 18);
  }
  ctx.textAlign = 'left';
  ctx.fillText('away wins by', pad, 16);
  ctx.textAlign = 'right';
  ctx.fillText('home wins by', W - pad, 16);
}

function renderKeyTable(){
  const n = done || 1;
  const abs = new Int32Array(40);
  for (let i = 0; i < n; i++){ const a = Math.abs(margins[i]); if (a < 40) abs[a]++; }
  const rows = [1, 3, 4, 6, 7, 10, 14, 17].map(k => {
    const mine = 100 * abs[k] / n;
    const real = M.empiricalMargin[k];
    return `<div class="keycell${k === 3 || k === 7 ? ' hot' : ''}">
      <span>margin ${k}</span><b>${mine.toFixed(1)}%</b>
      <em>real ${real}%</em></div>`;
  });
  $('keyTable').innerHTML = rows.join('');
}

function renderPricing(p){
  const P = M.params;
  const bank = parseFloat($('bank').value) || P.bankroll;
  const H = $('homeSel').value, A = $('awaySel').value;
  const sp = parseFloat($('spread').value);

  const rows = [];
  const push = (market, side, odds, other, model, minEdge) => {
    if (!isFinite(odds) || !isFinite(other)) return;
    const [fair] = devig(impliedProb(odds), impliedProb(other));
    const edge = model - fair;
    const ev = evPerUnit(model, odds);
    const stake = edge >= minEdge && ev > 0 ? kelly(model, odds, P) : 0;
    rows.push({ market, side, odds, fair, model, edge, ev, stake: stake * bank });
  };

  const mlH = +$('mlH').value, mlA = +$('mlA').value;
  push('Moneyline', H, mlH, mlA, p.pHome, P.min_edge_ml);
  push('Moneyline', A, mlA, mlH, p.pAway, P.min_edge_ml);
  push('Spread', `${H} ${fmtSigned(sp)}`, +$('spH').value, +$('spA').value, p.pCoverH, P.min_edge_spread);
  push('Spread', `${A} ${fmtSigned(-sp)}`, +$('spA').value, +$('spH').value, p.pCoverA, P.min_edge_spread);
  push('Total', `Over ${$('total').value}`, +$('ovP').value, +$('unP').value, p.pOver, P.min_edge_total);
  push('Total', `Under ${$('total').value}`, +$('unP').value, +$('ovP').value, p.pUnder, P.min_edge_total);

  $('priceTable').querySelector('tbody').innerHTML = rows.map(r => `
    <tr>
      <td>${r.market} · ${r.side}</td>
      <td>${r.odds > 0 ? '+' : ''}${r.odds}</td>
      <td>${pct(r.fair)}</td>
      <td class="model">${pct(r.model)}</td>
      <td class="${r.edge >= 0 ? 'pos' : 'neg'}">${(r.edge >= 0 ? '+' : '') + (100 * r.edge).toFixed(1)}%</td>
      <td class="${r.ev >= 0 ? 'pos' : 'neg'}">${(r.ev >= 0 ? '+' : '') + r.ev.toFixed(3)}</td>
      <td>${r.stake > 0 ? '$' + r.stake.toFixed(2) : '—'}</td>
      <td>${r.stake > 0
        ? '<span class="tag bet">bet</span>'
        : `<span class="tag pass">fair ${fairOdds(r.model) > 0 ? '+' : ''}${fairOdds(r.model)}</span>`}</td>
    </tr>`).join('');
}

function renderRatings(){
  const L = M.league;
  const rows = Object.entries(M.teams)
    .map(([t, v]) => ({ t, ...v }))
    .sort((a, b) => b.power - a.power);
  $('ratingsTable').querySelector('tbody').innerHTML = rows.map((r, i) => `
    <tr>
      <td>${i + 1}</td><td><strong>${r.t}</strong></td>
      <td class="${r.power >= 0 ? 'pos' : 'neg'}">${fmtSigned(r.power)}</td>
      <td>${r.att.toFixed(3)}</td><td>${r.def.toFixed(3)}</td>
      <td>${(L.ppg * r.att).toFixed(1)}</td><td>${(L.ppg * r.def).toFixed(1)}</td>
      <td>${r.n}</td>
    </tr>`).join('');
}

/* ---------- run loop ---------- */
function run(){
  const token = ++runToken;
  const ep = expectedPoints();
  done = 0;

  const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const step = () => {
    if (token !== runToken) return;
    const to = Math.min(done + (reduced ? N_SIMS : CHUNK), N_SIMS);
    simulateChunk(done, to, ep.muH, ep.muA);
    done = to;
    renderNumbers(ep);
    renderHistogram();
    if (done < N_SIMS) requestAnimationFrame(step);
    else { renderKeyTable(); renderPricing(priceAll()); }
  };
  step();
}

/* ---------- boot ---------- */
fetch('data/model.json?v=' + Date.now())
  .then(r => { if (!r.ok) throw new Error('model.json missing'); return r.json(); })
  .then(data => {
    M = data;
    $('genDate').textContent  = 'fitted ' + M.generated;
    $('gameCount').textContent = M.gamesUsed.toLocaleString();
    if (M.placeholder) $('stale').hidden = false;
    $('week').value = M.week || 1;

    const teams = Object.keys(M.teams).sort();
    const opts = teams.map(t => `<option value="${t}">${t}</option>`).join('');
    $('homeSel').innerHTML = opts; $('awaySel').innerHTML = opts;
    $('homeSel').value = teams.includes('KC') ? 'KC' : teams[0];
    $('awaySel').value = teams.includes('BUF') ? 'BUF' : teams[1];

    renderRatings();
    document.querySelectorAll('select,input').forEach(el => {
      el.addEventListener('change', run);
      if (el.type === 'number') el.addEventListener('input', debounce(run, 220));
    });
    run();
  })
  .catch(err => {
    document.querySelector('main').insertAdjacentHTML('afterbegin',
      `<section class="panel"><div class="panel-label">No ratings loaded</div>
       <p class="footnote top">${err.message}. Run
       <code>python scripts/export_web.py --placeholder</code> and commit
       <code>docs/data/model.json</code>.</p></section>`);
  });

function debounce(fn, ms){
  let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); };
}
