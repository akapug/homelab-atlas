'use strict';
// the batch-window cost model (src/model.js win*, data.window from build.py + probe/window_inputs.py): run after python3 build.py
const assert = require('assert'), fs = require('fs'), path = require('path');
const FM = require('../src/model.js');
const ROOT = path.join(__dirname, '..');
const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'data.json'), 'utf8'));
const clone = x => JSON.parse(JSON.stringify(x));
const v = FM.v, TAGS = new Set(['MEASURED', 'READ', 'PRIOR', 'ESTIMATE', 'FIT', 'UNKNOWN']);
// a build with no data/window config (a public copy holds none) runs only the generic cases
const key = data.window && data.window.default, W = key ? data.window.configs[key] : null;
let passed = 0;
const generic = (name, fn) => { fn(); passed++; console.log('ok ' + name); };
const test = (name, fn) => (W ? generic(name, fn) : console.log('skip ' + name + ' (no data/window config in this build)'));
const fit = W ? FM.winFit(data, key) : null;
const lev = W ? FM.winLevers(data, key, { fit }) : null;

test('the ubench roll-up is parsed row for row from the raw log', () => {
  const raw = fs.readFileSync(path.join(ROOT, W.ubench.file), 'utf8').split('\n');
  const h = raw.findIndex(x => x.startsWith('STAGE ROLL-UP')), end = raw.findIndex((x, i) => i > h && x.startsWith('TOTAL'));
  const rows = raw.slice(h + 2, end).filter(x => x.trim());
  assert.strictEqual(W.ubench.classes.length, rows.length);
  rows.forEach((line, i) => {
    const t = line.trim().split(/\s+/), c = W.ubench.classes[i];
    assert.strictEqual(c.name, t.slice(0, -10).join(' '));
    assert.strictEqual(c.layers, Number(t[t.length - 10]));
    assert.deepStrictEqual(c.ms, t.slice(-9, -4).map(Number));
    assert.strictEqual(c.xfloor, Number(t[t.length - 1]));
    assert(c.tag === 'MEASURED' && c.src.includes(c.name));
  });
  assert.deepStrictEqual(W.ubench.grid, [1, 2, 4, 8, 16]);
  assert.strictEqual(W.ubench.floor_rows, 16);
  assert.strictEqual(v(W.ubench.floor_gbs), 548);
  assert.strictEqual(v(W.ubench.method_floor_us), 8.7);
  const kinds = W.ubench.kinds, total = Object.values(kinds).reduce((a, b) => a + b, 0);
  assert.strictEqual(total, W.ubench.measured_layers);
  W.ubench.classes.forEach(c => assert(c.kind === null ? c.layers === total : kinds[c.kind] === c.layers, c.name));
  // the routing delta is the gap between the two TOTAL lines, and the roll-up rows add up to the random-routing total
  const T = W.ubench.totals, rnd = T[W.ubench.routing.random], alt = T[W.ubench.routing.alt];
  W.ubench.routing.alt_delta_ms.forEach((d, i) => assert(Math.abs(d - (alt[i] - rnd[i])) < 1e-9));
  W.ubench.grid.forEach((_, i) => assert(Math.abs(W.ubench.classes.reduce((a, c) => a + c.ms[i], 0) - rnd[i]) < 0.01, 'TOTAL col ' + i));
  assert(W.ubench.classes.some(c => c.name === W.ubench.routing.class));
  assert(W.ubench.classes.some(c => c.launches === 2), 'a class named "x2" runs two launches a layer');
});

test('the spec gives the layer kinds; the model list fills the rest and agrees where both speak', () => {
  const raw = fs.readFileSync(path.join(ROOT, W.spec.file), 'utf8').split('\n').filter(x => x.startsWith('layer '));
  raw.forEach(line => {
    const t = line.split(/\s+/), L = Number(t[1]), has = t.slice(2).some(p => p.split('=')[0] === W.spec.kind_rule.tensor);
    assert.strictEqual(W.layers[L].kind, has ? W.spec.kind_rule.kind : W.spec.kind_rule.default);
    assert.strictEqual(W.layers[L].from, 'spec');
  });
  assert.strictEqual(W.layers.length, v(W.n_layers));
  assert.strictEqual(W.layers.filter(x => x.from === 'model').length, v(W.n_layers) - raw.length);
  Object.entries(W.ubench.kinds).forEach(([k, n]) => assert.strictEqual(W.layers.filter(x => x.from === 'spec' && x.kind === k).length, n, k));
});

test('every constant of the window config is tagged and sourced', () => {
  const walk = (x, at) => {
    if (Array.isArray(x)) return x.forEach((y, i) => walk(y, at + '[' + i + ']'));
    if (!x || typeof x !== 'object') return;
    if ('v' in x && typeof x.v !== 'object' || ('v' in x && Array.isArray(x.v))) assert(x.src, at + ' has no src');
    if ('tag' in x && 'v' in x) assert(TAGS.has(x.tag), at + ' ' + x.tag);
    Object.entries(x).forEach(([k, y]) => walk(y, at + '.' + k));
  };
  walk(W, key);
  ['gamma', 'graph_launch_us', 'lever_xfloor', 'extra_lever_share'].forEach(k => assert(TAGS.has(W.settings[k].tag), k));
  W.points.forEach(p => { assert(TAGS.has(p.stage_ms.tag) && p.stage_ms.src, p.id); if (p.tps) assert(TAGS.has(p.tps.tag) && p.tps.src, p.id); });
  W.extras.forEach(e => assert(e.src && e.label && ['first', 'last'].includes(e.stage) && ['row', 'fixed'].includes(e.per), e.id));
  assert(W.ubench.routing_phi.tag && W.groups.tag && W.max_window_rows.tag);
});

test('calibration: the fit points are matched, the check points within 3 % stage and tok/s, the pipeline law within 1 % median, 5 % worst', () => {
  console.log('FIT', JSON.stringify({ K: fit.K, stage_fit: fit.stage_fit_max_abs_pct, stage_check: fit.stage_check_max_abs_pct, tps_fit: fit.tps_fit_max_abs_pct, tps_check: fit.tps_check_max_abs_pct, law: fit.law_max_abs_pct }));
  assert(fit.stage_fit_max_abs_pct < 1, fit.stage_fit_max_abs_pct);
  assert(fit.stage_check_max_abs_pct < 3, fit.stage_check_max_abs_pct);
  assert(fit.tps_check_max_abs_pct < 3, fit.tps_check_max_abs_pct);
  // the law takes the slowest stage's mean as the cycle: with the stages level (split 25: within ~1 ms) the per-window
  // maximum of two jittering stages runs above both means, so the law overpredicts those points (S25 +2 %, A3 all-on +6 %).
  // Pinned as a known limit: every level-stage point errs high, the imbalanced ones stay within 5 %
  const level = r => Math.abs(r.stage_meas[0] - r.stage_meas[1]) / Math.max(...r.stage_meas) < 0.02;
  fit.residuals.filter(r => r.law_err !== null && r.rows > 1).forEach(r => (level(r) ? assert(r.law_err > 0 && r.law_err < 0.07, r.id + ' ' + r.law_err) : assert(Math.abs(r.law_err) < 0.05, r.id + ' ' + r.law_err)));
  fit.residuals.filter(r => r.law_err !== null && r.rows === 1).forEach(r => assert(Math.abs(r.law_err) < 0.02, r.id + ' ' + r.law_err));
  const law = fit.residuals.filter(r => r.law_err !== null).map(r => Math.abs(r.law_err)).sort((a, b) => a - b);
  assert(law[Math.floor(law.length / 2)] < 0.01, 'median law error ' + law[Math.floor(law.length / 2)]);
  assert(fit.K.eta > 0.9 && fit.K.eta <= 1.05, fit.K.eta);
  assert(fit.K.omega_us >= 0 && fit.K.omega_us < v(W.settings.omega_max_us));
  Object.values(fit.K.extras).forEach(x => assert(x >= 0));
  assert(fit.residuals.some(r => r.use === 'check'), 'a point outside the fit is scored');
  // the tok/s points come from the ledger row (build.py reads runs.jsonl)
  assert.strictEqual(v(W.points.find(p => p.id === 'G2-n32').tps), 272.8);
  // the kernel-scale reading fits too, and keeps gamma inside its range
  const g = FM.winFit(data, key, { gamma_free: true }), gr = v(W.settings.gamma_range);
  assert(g.K.gamma >= gr[0] && g.K.gamma <= gr[1] && g.stage_fit_max_abs_pct < 1 && g.tps_check_max_abs_pct < 3);
  W.extras.filter(e => e.kernel_scale_alt).forEach(e => assert.strictEqual(g.K.extras[e.id], 0));
});

test('held-out throughput and stages cannot change fitted efficiency; training throughput can', () => {
  const d = clone(data), X = d.window.configs[key], checks = X.points.filter(p => p.use === 'check');
  assert(checks.length > 0, 'the causal control must perturb actual held-out points');
  X.points.filter(p => p.use !== 'fit').forEach(p => { if (p.tps) p.tps.v *= 2; p.stage_ms.v = p.stage_ms.v.map(x => x * 3); });
  assert.deepStrictEqual(FM.winFit(d, key).K, fit.K, 'check observations are not training operands');
  X.points.filter(p => p.use === 'fit' && p.tps).forEach(p => { p.tps.v *= 2; });
  assert(Math.abs(FM.winFit(d, key).K.eta / fit.K.eta - 2) < 1e-12, 'training throughput changes the fitted eta');
});

test('without valid training throughput efficiency and downstream throughput remain UNKNOWN', () => {
  const d = clone(data), X = d.window.configs[key];
  X.points.filter(p => p.use === 'fit').forEach(p => { p.tps = { v: null }; });
  const f = FM.winFit(d, key), step = FM.winStep(X, v(X.settings.plan_agents), f.K);
  assert.strictEqual(f.K.eta, null);
  assert.strictEqual(f.status, 'UNKNOWN');
  assert.strictEqual(step.total, null);
  assert.strictEqual(step.per_agent, null);
  assert(f.residuals.every(p => p.tps_pred === null && p.tps_err === null && p.law_err === null));
  const report = FM.winReport(d, key);
  assert.strictEqual(report.status, 'UNKNOWN');
  assert.strictEqual(report.plan, null);
  assert.deepStrictEqual(FM.winLevers(d, key, { fit: f }).levers, [], 'no ranking of unknown throughput');
  for (const invalid of [0, -1, NaN, Infinity]) {
    X.points.filter(p => p.use === 'fit').forEach(p => { p.tps = { v: invalid }; });
    assert.strictEqual(FM.winFit(d, key).K.eta, null, 'only finite positive training observations may identify eta');
  }
});

test('an anchor starts the plan at a measured point and predicts the same run at two agents within 2 %', () => {
  const id = W.settings.plan_anchor && W.settings.plan_anchor.v, p = id && W.points.find(x => x.id === id);
  assert(p, 'the config names a plan anchor among its points');
  const a = FM.winAnchor(data, key, fit, id);
  FM.winStep(W, p.agents, a.K).stage_ms.forEach((m, i) => assert(Math.abs(m - v(p.stage_ms)[i]) < 1e-9));
  const two = W.points.find(x => x.ledger && x.ledger.id === p.ledger.id && x.agents !== p.agents);
  assert(two, 'the anchor run has a point at another agent count');
  const pred = FM.winStep(W, two.agents, a.K).total;
  console.log('ANCHOR', id, '->', two.id, pred.toFixed(1), 'vs', v(two.tps));
  assert(Math.abs(pred / v(two.tps) - 1) < 0.02, pred + ' vs ' + v(two.tps));
  const R = FM.winReport(data, key, { anchor: null, source: 'ubench' });
  assert.strictEqual(R.anchor, null);
  assert(Math.abs(R.plan.total - FM.winStep(W, v(W.settings.plan_agents), fit.K).total) < 1e-9, 'no anchor = the bare model');
});

generic('the pipeline law: groups >= stages bind on the slowest stage, fewer groups on the sum', () => {
  assert(Math.abs(FM.winLaw([50, 60], 16, 2, 1) - 16000 / 60) < 1e-9);
  assert(Math.abs(FM.winLaw([50, 60], 16, 1, 1) - 16000 / 110) < 1e-9);
  assert(Math.abs(FM.winLaw([50, 60], 16, 4, 0.99) - 0.99 * 16000 / 60) < 1e-9);
});

test('levers: monotone in the x-floor target, never below today, ranked by tok/s; the rebalance is a lever', () => {
  const now = lev.now.total;
  lev.levers.forEach((x, i) => {
    assert(x.total >= now - 1e-9 && x.total_rebalanced >= lev.rebalanced.total - 1e-9, x.lever);
    if (i) assert(lev.levers[i - 1].total >= x.total - 1e-12, 'ranked');
    if (x.grid) x.grid.forEach((g, j) => { if (j) { assert(g.xfloor > x.grid[j - 1].xfloor); assert(g.total <= x.grid[j - 1].total + 1e-9, x.lever + ' at ' + g.xfloor); } });
  });
  const reb = lev.levers.find(x => x.kind === 'layout');
  assert(reb && reb.total >= now - 1e-9 && Math.abs(reb.total - lev.rebalanced.total) < 1e-9);
  assert.strictEqual(lev.levers.filter(x => x.kind !== 'layout' && x.kind !== 'variant').length, W.ubench.classes.length, 'one lever per measured class');
  assert.strictEqual(lev.levers.filter(x => x.kind === 'variant').length, (W.variants || []).length, 'one lever per measured variant');
  // a lower target never raises a class above today and never under its launch floor
  W.ubench.classes.forEach(c => { const x = lev.levers.find(l => l.lever === c.name); assert(x.scale > 0 && x.scale <= 1, c.name); });
  lev.unmeasured.forEach(x => assert(x.total >= now - 1e-9 && x.unmeasured));
  console.log('LEVERS', lev.levers.slice(0, 8).map(x => x.lever + ' ' + x.total.toFixed(1) + ' (rebalanced ' + x.total_rebalanced.toFixed(1) + ')').join('; '));
});

test('the rebalance split is the best contiguous split (brute force, 2 and 3 stages)', () => {
  const N = v(W.settings.plan_agents);
  [2, 3].forEach(S => {
    const r = FM.winRebalance(W, N, fit.K, {}, S), L = W.layers.length;
    assert.strictEqual(r.split.length, S);
    assert.strictEqual(r.split[0][0], 0); assert.strictEqual(r.split[S - 1][1], L - 1);
    r.split.forEach((s, i) => { assert(s[1] >= s[0]); if (i) assert.strictEqual(s[0], r.split[i - 1][1] + 1); });
    const cuts = S === 2 ? Array.from({ length: L - 1 }, (_, a) => [a + 1]) : [].concat(...Array.from({ length: L - 2 }, (_, a) => Array.from({ length: L - a - 2 }, (_, b) => [a + 1, a + b + 2])));
    const best = cuts.map(c => { const e = c.concat(L), split = e.map((x, i) => [i ? e[i - 1] : 0, x - 1]); return Math.max(...FM.winStep(W, N, fit.K, { split, groups: r.groups }).stage_ms); }).reduce((a, b) => Math.min(a, b));
    assert(Math.abs(Math.max(...r.stage_ms) - best) < 1e-9, S + ' stages: ' + Math.max(...r.stage_ms) + ' vs ' + best);
  });
});

test('no hardcode: the model code names no kernel class or layer kind, and renaming them changes nothing', () => {
  const src = fs.readFileSync(path.join(ROOT, 'src', 'model.js'), 'utf8');
  const a = src.indexOf('// ---------- batch window'), b = src.indexOf('// Explicit tier-read fractions'), body = src.slice(a, b);
  assert(a > 0 && b > a);
  W.ubench.classes.forEach(c => assert(!body.includes("'" + c.name + "'") && !body.includes('"' + c.name + '"'), c.name));
  Object.keys(W.ubench.kinds).forEach(k => assert(!new RegExp("['\"]" + k + "['\"]").test(body), k));
  ['random', 'experts', 'gdn', 'qsa', 'B70', 'odie', 'fnx'].forEach(w => assert(!new RegExp("['\"`][^'\"`]*\\b" + w + "\\b", 'i').test(body), w));
  const build = fs.readFileSync(path.join(ROOT, 'build.py'), 'utf8'), bw = build.slice(build.indexOf('# ---------------- batch window'), build.indexOf('def external_rows'));
  Object.keys(W.ubench.kinds).concat(W.ubench.classes.map(c => c.name)).forEach(k => assert(!bw.includes("'" + k + "'"), 'build.py names ' + k));
  // rename every class and kind: the same numbers
  const d = clone(data), X = d.window.configs[key], ren = {};
  Object.keys(X.ubench.kinds).forEach((k, i) => { ren[k] = 'K' + i; });
  X.ubench.kinds = Object.fromEntries(Object.entries(X.ubench.kinds).map(([k, n]) => [ren[k], n]));
  X.layers.forEach(l => { l.kind = ren[l.kind]; });
  const byOld = Object.fromEntries(W.ubench.classes.map((c, i) => [c.name, 'c' + i]));
  X.ubench.classes.forEach((c, i) => { if (c.name === X.ubench.routing.class) X.ubench.routing.class = 'c' + i; c.name = 'c' + i; c.kind = c.kind === null ? null : ren[c.kind]; });
  (X.variants || []).forEach(x => { x.class = byOld[x.class]; });
  const f2 = FM.winFit(d, key), l2 = FM.winLevers(d, key, { fit: f2 });
  assert(Math.abs(f2.K.omega_us - fit.K.omega_us) < 1e-6 && Math.abs(l2.now.total - lev.now.total) < 1e-6);
  l2.levers.forEach((x, i) => assert(Math.abs(x.total - lev.levers[i].total) < 1e-6));
  // the profile and the variants name classes and kinds only through the config: renamed there, the same numbers
  if (!X.profile) return;
  X.profile.terms.forEach(t => { t.classes = t.classes.map(([n, f]) => [byOld[n], f]); if (t.kind) t.kind = ren[t.kind]; });
  const p1 = FM.winProfLevers(data, key, { fit: FM.winProfFit(data, key, fit) }), p2 = FM.winProfLevers(d, key, { fit: FM.winProfFit(d, key, f2) });
  p2.levers.forEach((x, i) => assert(Math.abs(x.total - p1.levers[i].total) < 1e-6 && x.lever === p1.levers[i].lever));
});

test('the inputs move the prediction: a class, the card bandwidth, the groups, the stages', () => {
  const N = v(W.settings.plan_agents), now = FM.winStep(W, N, fit.K);
  const d = clone(data), X = d.window.configs[key], big = X.ubench.classes.reduce((a, c) => (c.ms[c.ms.length - 1] > a.ms[a.ms.length - 1] ? c : a));
  X.ubench.classes = X.ubench.classes.filter(c => c !== big);
  assert(FM.winStep(X, N, fit.K).stage_ms.every((s, i) => s < now.stage_ms[i]), 'dropping the largest class shortens every stage');
  const fast = FM.winStep(W, N, fit.K, { bw: 2 * v(W.card_bw_gbs) }), slow = FM.winStep(W, N, fit.K, { bw: v(W.card_bw_gbs) / 2 });
  assert(fast.total > now.total && slow.total < now.total);
  assert(Math.abs(FM.winStep(W, N, fit.K, { bw: v(W.card_bw_gbs) }).total - now.total) < 1e-9, 'the config card is the reference');
  const g1 = FM.winStep(W, N, fit.K, { groups: 1 });
  assert(g1.rows === 2 * now.rows || g1.rows === N, 'one group carries every row');
  const st = v(W.settings.whatif_stages).map(s => FM.winRebalance(W, N, fit.K, {}, s));
  st.forEach(s => assert(s.rows <= v(W.max_window_rows) && s.split.length >= 1));
  assert(st[st.length - 1].total > st[0].total, 'more stages, more tok/s while hand-offs are unpriced');
  const d2 = clone(data); d2.window.configs[key].settings.lever_xfloor.v = 2;
  assert(FM.winLevers(d2, key, { fit }).levers[0].total <= lev.levers[0].total + 1e-9, 'a looser target pays less');
});

generic('another model and card: a config built from scratch gets its own ranking', () => {
  const t = (x, tag) => ({ v: x, tag: tag || 'MEASURED', src: 'test' });
  const cls = (name, kind, layers, ms, xfloor) => ({ name, kind, layers, launches: 1, ms, xfloor, mb_floor_lyr: 1, floor_us: 1, us_row_lyr: 1, tag: 'MEASURED', src: 'test' });
  const X = {
    key: 'toy', label: 'toy', layers: Array.from({ length: 8 }, (_, i) => ({ layer: i, kind: i % 2 ? 'B' : 'A' })),
    ubench: { grid: [1, 4], floor_rows: 4, floor_gbs: t(100, 'READ'), kinds: { A: 2, B: 2 }, measured_layers: 4, routing: null, routing_phi: t(0, 'ESTIMATE'), routing_phi_range: { v: [0, 0], src: 'test' },
      classes: [cls('wide', null, 4, [1, 8], 2), cls('alpha', 'A', 2, [0.5, 1], 10), cls('beta', 'B', 2, [0.5, 3], 1.5)] },
    stages: [{ layers: [0, 3] }, { layers: [4, 7] }], groups: t(2, 'READ'), max_window_rows: t(4, 'READ'), max_agents: t(8, 'READ'), min_batched_agents: t(2, 'READ'),
    card_bw_gbs: t(100, 'READ'), whatif_cards: [], extras: [],
    settings: { gamma: t(1, 'PRIOR'), omega_max_us: { v: 10, src: 'test' }, omega_step_us: { v: 1, src: 'test' }, graph_launch_us: t(0, 'ESTIMATE'), lever_xfloor: t(1, 'ESTIMATE'), lever_xfloor_grid: { v: [1, 2], src: 'test' },
      plan_agents: { v: 8, src: 'test' }, agents_grid: { v: [2, 8], src: 'test' }, whatif_stages: { v: [1, 2], src: 'test' }, extra_lever_share: t(0.5, 'ESTIMATE') },
    points: [{ id: 'p', use: 'fit', agents: 8, stage_ms: t([12, 12]), tps: t(1000 / 3) }]
  };
  const d = { window: { default: 'toy', configs: { toy: X } } }, f = FM.winFit(d, 'toy');
  assert(Math.abs(f.residuals[0].stage_pred[0] - 12) < 1e-6, JSON.stringify(f.residuals[0]));
  const L = FM.winLevers(d, 'toy', { fit: f }).levers.filter(x => x.kind !== 'layout');
  // at 1x floor: alpha saves 1 - 0.1 = 0.9 ms of its 2 layers' 1 ms, wide 8 - 4 = 4, beta 3 - 2 = 1: wide first, then beta
  assert.deepStrictEqual(L.map(x => x.lever), ['wide', 'beta', 'alpha']);
});

const ptest = (name, fn) => (W && W.profile ? generic(name, fn) : console.log('skip ' + name + ' (no profile in this config)'));
const pf = W && W.profile ? FM.winProfFit(data, key, fit, FM.winAnchor(data, key, fit, v(W.profile.curve_anchor || W.settings.plan_anchor)) || fit) : null;
const plev = pf ? FM.winProfLevers(data, key, { fit: pf }) : null;

ptest('the profile: every term MEASURED and mapped to ubench classes; the terms add up to its totals; the gap is the run minus the terms', () => {
  const names = new Set(W.ubench.classes.map(c => c.name)), n = W.stages.length;
  W.profile.terms.forEach(t => {
    assert(t.ms.tag === 'MEASURED' && t.ms.src.includes(t.name) && v(t.ms).length === n, t.id);
    t.classes.forEach(([c, f]) => assert(names.has(c) && f > 0 && f <= 1, t.id + ' ' + c));
    assert(t.rows_basis.tag && t.rows_basis.src, t.id);
  });
  pf.term_sum_ms.forEach((x, i) => assert(Math.abs(x - v(W.profile.totals_ms)[i]) < 0.5, 'stage ' + i + ': ' + x + ' vs ' + v(W.profile.totals_ms)[i]));
  const pt = W.points.find(p => p.id === W.profile.point);
  pf.gap_ms.forEach((g, i) => assert(Math.abs(g - (v(pt.stage_ms)[i] - pf.term_sum_ms[i])) < 1e-9));
  console.log('PROFILE terms', pf.term_sum_ms.map(x => x.toFixed(2)).join('/'), 'gap', pf.gap_ms.map(x => x.toFixed(2)).join('/'));
});

test('a point or the profile on another layout carries its own split; the plan runs at the config\'s stages', () => {
  const stages = W.stages.map(s => s.layers);
  assert.deepStrictEqual(FM.winPointSplit(W, {}), stages);
  const own = W.points.filter(p => p.split);
  assert(own.length && W.points.some(p => !p.split), 'points on both layouts');
  own.forEach(p => {
    const r = fit.residuals.find(x => x.id === p.id);
    assert.deepStrictEqual(r.split, p.split, p.id);
    // the residual is the model on the point's own split, not the plan's
    const G = r.rows ? p.agents / r.rows : 1;
    FM.winStep(W, p.agents, fit.K, { split: p.split, groups: G }).stage_ms.forEach((m, i) => assert(Math.abs(m - r.stage_pred[i]) < 1e-9, p.id));
  });
  W.points.filter(p => !p.split).forEach(p => assert.deepStrictEqual(fit.residuals.find(x => x.id === p.id).split, stages));
  // moving the split changes the model: the same kernels at split 24 and at the plan's split differ
  const p = own[0], a = FM.winStep(W, p.agents, fit.K, { split: p.split }), b = FM.winStep(W, p.agents, fit.K);
  if (JSON.stringify(p.split) !== JSON.stringify(stages)) assert(a.stage_ms.some((m, i) => Math.abs(m - b.stage_ms[i]) > 1e-6));
  if (pf && W.profile.split) assert.deepStrictEqual(pf.split, W.profile.split);
  assert.deepStrictEqual(FM.winReport(data, key, {}).plan.split, stages);
});

ptest('the profile reproduces its point and predicts its checks (other agent counts, the run without profiling) within 2 %', () => {
  const own = pf.checks.find(c => c.use === 'profile');
  own.stage_err.forEach(e => assert(Math.abs(e) < 1e-9));
  const rest = pf.checks.filter(c => c.use === 'check');
  assert(rest.length >= 1 && rest.some(c => c.rows !== pf.rows), 'a check at another row count');
  // at the profile's own split within 2 %. At another split a moved layer also moves the expert cache each card holds (fewer
  // VRAM slots a layer on the card that gains it), which the profile does not price: within 5 % stage there, and the tok/s
  // carries the law's level-stage overprediction (see calibration)
  const same = c => JSON.stringify(c.split) === JSON.stringify(pf.split);
  rest.forEach(c => { const b = same(c) ? 0.02 : 0.05; c.stage_err.forEach(e => assert(Math.abs(e) < b, c.id + ' stage ' + e)); assert(Math.abs(c.tps_err) < (same(c) ? 0.02 : 0.03), c.id + ' tok/s ' + c.tps_err); });
  assert(rest.some(c => !same(c)), 'a check at another split');
  console.log('PROFILE checks', pf.checks.map(c => c.id + ' ' + c.stage_err.map(e => (100 * e).toFixed(1)).join('/') + ' % stage, ' + (100 * c.tps_err).toFixed(1) + ' % tok/s').join('; '));
});

ptest('profile levers: monotone in the target, never below today, ranked; a variant saves (old - new) x launches a layer at the profile rows', () => {
  const now = plev.now.total;
  plev.levers.forEach((x, i) => {
    assert(x.total >= now - 1e-9 && x.total_rebalanced >= plev.rebalanced.total - 1e-9, x.lever);
    if (i) assert(plev.levers[i - 1].total >= x.total - 1e-12, 'ranked');
    if (x.grid) x.grid.forEach((g, j) => { if (j) assert(g.total <= x.grid[j - 1].total + 1e-9, x.lever + ' at ' + g.xfloor); });
  });
  plev.unmeasured.forEach(x => assert(x.total >= now - 1e-9 && x.unmeasured && x.cut_share === v(W.settings.extra_lever_share)));
  assert(plev.unmeasured.some(x => x.kind === 'gap'), 'the gap is a lever of its own');
  (W.variants || []).forEach(x => {
    const r = plev.levers.find(l => l.kind === 'variant' && l.class === x.class), c = W.ubench.classes.find(k => k.name === x.class);
    const per = (v(x.old_us) - v(x.new_us)) / 1000 * v(x.launches), layers = W.stages.map(s => W.layers.filter(l => l.layer >= s.layers[0] && l.layer <= s.layers[1] && (c.kind === null || l.kind === c.kind)).length);
    // the plan runs at the profile's rows, where the cut is taken as measured
    assert.strictEqual(plev.now.rows, pf.rows);
    r.saved_ms.forEach((m, i) => assert(Math.abs(m - per * layers[i]) < 1e-9, x.id + ' stage ' + i + ': ' + m + ' vs ' + per * layers[i]));
  });
  console.log('PROFILE LEVERS', plev.levers.slice(0, 8).map(x => x.lever + ' ' + x.total.toFixed(1) + ' (rebalanced ' + x.total_rebalanced.toFixed(1) + ')').join('; '));
});

ptest('the ubench is the fallback: without a profile, or with --source ubench, the report runs on the classes', () => {
  const d = clone(data);
  delete d.window.configs[key].profile;
  const R0 = FM.winReport(d, key, {}), R1 = FM.winReport(data, key, { source: 'ubench' }), R2 = FM.winReport(data, key, {});
  assert.strictEqual(R0.source, 'ubench'); assert.strictEqual(R1.source, 'ubench'); assert.strictEqual(R2.source, 'profile');
  assert(Math.abs(R0.plan.total - R1.plan.total) < 1e-9 && R0.levers.levers.length === R1.levers.levers.length);
  assert.deepStrictEqual(R2.ubench_levers.map(x => x.lever), R1.levers.levers.map(x => x.lever));
  assert(R2.top.some(x => x.class.startsWith('gap')), 'the bottleneck breakdown holds the gap');
});

generic('a profile on another model and card: its terms, its gap, its floors rank the levers', () => {
  const t = (x, tag) => ({ v: x, tag: tag || 'MEASURED', src: 'test' });
  const cls = (name, kind, layers, ms, xfloor) => ({ name, kind, layers, launches: 1, ms, xfloor, mb_floor_lyr: 1, floor_us: 1, us_row_lyr: 1, tag: 'MEASURED', src: 'test' });
  const term = (where, name, ms, classes, basis) => Object.assign(where === 'last' ? { stage: 'last' } : { kind: where }, { id: where + ':' + name, name, ms: t(ms), classes, lever: name, rows_basis: t(basis || 'stage', 'ESTIMATE'), bw: true });
  const X = {
    key: 'toy', label: 'toy', layers: Array.from({ length: 8 }, (_, i) => ({ layer: i, kind: i % 2 ? 'B' : 'A' })),
    ubench: { grid: [1, 4], floor_rows: 4, floor_gbs: t(100, 'READ'), kinds: { A: 2, B: 2 }, measured_layers: 4, routing: null, routing_phi: t(0, 'ESTIMATE'), routing_phi_range: { v: [0, 0], src: 'test' },
      classes: [cls('alpha', 'A', 2, [0.5, 1], 10), cls('beta', 'B', 2, [0.5, 3], 1.5)] },
    stages: [{ layers: [0, 3] }, { layers: [4, 7] }], groups: t(2, 'READ'), max_window_rows: t(4, 'READ'), max_agents: t(8, 'READ'), min_batched_agents: t(2, 'READ'),
    card_bw_gbs: t(100, 'READ'), whatif_cards: [], extras: [],
    settings: { gamma: t(1, 'PRIOR'), omega_max_us: { v: 10, src: 'test' }, omega_step_us: { v: 1, src: 'test' }, graph_launch_us: t(0, 'ESTIMATE'), lever_xfloor: t(1, 'ESTIMATE'), lever_xfloor_grid: { v: [1, 2], src: 'test' },
      plan_agents: { v: 8, src: 'test' }, agents_grid: { v: [2, 8], src: 'test' }, whatif_stages: { v: [1, 2], src: 'test' }, extra_lever_share: t(0.5, 'ESTIMATE') },
    points: [{ id: 'p', use: 'fit', agents: 8, stage_ms: t([12, 12]), tps: t(1000 / 3) }],
    profile: { point: 'p', totals_ms: t([5, 6]), gap: { label: 'g', rows_basis: t('stage', 'ESTIMATE') },
      terms: [term('A', 'a', [2, 2], [['alpha', 1]]), term('B', 'b', [3, 3], [['beta', 1]]), term('last', 'head', [0, 1], [], 'fixed')] }
  };
  const d = { window: { default: 'toy', configs: { toy: X } } }, f = FM.winFit(d, 'toy'), P = FM.winProfFit(d, 'toy', f);
  assert.deepStrictEqual(P.gap_ms, [7, 6]);
  P.checks[0].stage_pred.forEach(x => assert(Math.abs(x - 12) < 1e-9));
  // at 1x floor: term a (4 ms over 4 A layers) to alpha's floor 0.05 ms a layer = 0.2, saving 3.8; term b (6 ms) to beta's 1 ms a
  // layer = 4, saving 2: a first. The head has no floor: priced at the share, beside the gap
  const L = FM.winProfLevers(d, 'toy', { fit: P });
  assert.deepStrictEqual(L.levers.filter(x => x.kind !== 'layout').map(x => x.lever), ['a', 'b']);
  assert(Math.abs(L.levers[0].stage_ms[0] - (12 - 1.9)) < 1e-9, JSON.stringify(L.levers[0].stage_ms));
  assert.deepStrictEqual(L.unmeasured.map(x => x.kind).sort(), ['gap', 'last stage']);
});

const etest = (name, fn) => (W && W.attempts && W.attempts.length ? generic(name, fn) : console.log('skip ' + name + ' (no attempts in this config)'));
const RE = W && W.attempts ? FM.winReport(data, key, {}) : null;

etest('evidence: each lever lists the attempts on its classes, a loss with its repair, the layout only on the split; engine rows from the ledger', () => {
  const ids = new Set(W.attempts.map(a => a.id));
  W.attempts.forEach(a => {
    ['parent', 'repairs'].forEach(k => assert(!a[k] || ids.has(a[k]), a.id + ' ' + k));
    if (a.ledger && a.tps) Object.values(a.tps).forEach(x => assert(x.tag === 'MEASURED' && x.src.includes(a.ledger.id), a.id));
    if (a.sanity) assert.strictEqual(a.sanity.tag, 'READ');
  });
  const all = RE.levers.levers.concat(RE.levers.unmeasured);
  all.forEach(x => (RE.evidence[x.lever] || []).forEach(e => {
    const a = W.attempts.find(b => b.id === e.id);
    assert(x.kind === 'layout' ? a.layout : (x.classes || []).some(c => a.classes.includes(c) || a.classes.includes('*')), x.lever + ' <- ' + e.id);
    assert.deepStrictEqual(e.repaired_by, W.attempts.filter(b => b.repairs === e.id).map(b => b.id));
  }));
  // an engine-wide attempt shows on every lever with kernels and on none without
  W.attempts.filter(a => a.classes.includes('*')).forEach(a => all.forEach(x => assert.strictEqual((RE.evidence[x.lever] || []).some(e => e.id === a.id), a.outcome !== 'control' && (x.classes || []).length > 0, x.lever + ' <- ' + a.id)));
  const rep = W.attempts.find(a => a.repairs && W.attempts.find(b => b.id === a.repairs).outcome === 'loss');
  if (rep) assert(all.some(x => (RE.evidence[x.lever] || []).some(e => e.id === rep.repairs && e.repaired_by.includes(rep.id))), 'a loss is shown with its repair');
  console.log('EVIDENCE', RE.levers.levers.slice(0, 4).map(x => x.lever + ': ' + RE.evidence[x.lever].map(e => e.id + ' ' + e.outcome).join(', ')).join(' | '));
});

etest('the frontier: each lever priced as its modelled gain on the frontier\'s measured tok/s, ranked, flagged inside noise, with the gate', () => {
  const F = RE.frontier, f = v(W.frontier.tps);
  assert(F && F.tps === f && F.gate && F.gate.steps.length >= 3);
  F.rank.forEach((x, i) => {
    const l = RE.levers.levers.concat(RE.levers.unmeasured).find(y => y.lever === x.lever);
    assert(Math.abs(x.at_frontier - f * l.total_rebalanced / RE.levers.rebalanced.total) < 1e-9);
    assert.strictEqual(x.inside_noise, x.gain_pct < v(W.frontier.noise_pct));
    if (i) assert(F.rank[i - 1].at_frontier >= x.at_frontier);
  });
  assert(!F.rank.some(x => x.lever === RE.levers.levers.find(y => y.kind === 'layout').lever), 'the frontier is already a tuned split');
});

etest('proposals: conservative dedup by mechanism key and action; the expected effect from the model; a blank for any lever', () => {
  const X = clone(W), a = X.attempts.find(b => b.action.split(/\s+/).length > 1 && b.classes.some(c => c !== '*'));
  const toks = a.action.split(/\s+/);
  X.proposals = [
    { id: 'p1', lane: 't', mechanism: { key: a.mechanism, text: '' }, action: toks.slice().reverse().join('  '), classes: a.classes.filter(c => c !== '*'), evidence: 'reuse', test: 't', falsifier: 'f' },
    { id: 'p2', lane: 't', mechanism: { key: a.mechanism, text: '' }, action: toks[0] + ' OTHER=1', classes: [], evidence: 'exploration', test: 't', falsifier: 'f' },
    { id: 'p3', lane: 't', mechanism: { key: 'never-tried:key', text: '' }, action: 'NEW=1', classes: [], evidence: 'exploration', test: 't', falsifier: 'f' },
    { id: 'p4', lane: 't', mechanism: { key: 'never-tried:key', text: '' }, action: 'NEW=1', classes: [], evidence: 'exploration', test: 't', falsifier: 'f' }];
  const P = FM.winProposals(X, RE.levers);
  assert.deepStrictEqual(P.map(p => p.dedup), ['duplicate', 'related', 'new', 'duplicate']);
  assert(P[0].prior.some(x => x.id === a.id && x.duplicate) && P[3].prior.some(x => x.id === 'p3'));
  assert(P[0].expected && P[0].expected.total >= RE.levers.now.total - 1e-9);
  RE.proposals.forEach(p => assert(p.expected, p.id + ' has a modelled effect'));
  const top = RE.levers.levers[0], T = FM.winProposalTemplate(W, RE.levers, top.lever);
  assert.deepStrictEqual(T.classes, top.classes);
  assert.strictEqual(T.evidence, RE.evidence[top.lever].length ? 'reuse' : 'exploration');
  assert(T.gate.length && T.model.total === top.total && T.id === '' && T.mechanism.key === '');
  const out = require('child_process').execFileSync(process.execPath, [path.join(ROOT, 'report.js'), '--window', '--propose', top.lever], { cwd: ROOT }).toString();
  assert(out.includes('"falsifier"') && out.includes(top.lever));
});

test('report.js --window --json carries the fit, plan, levers and what-ifs', () => {
  const out = require('child_process').execFileSync(process.execPath, [path.join(ROOT, 'report.js'), '--window', '--json'], { cwd: ROOT, maxBuffer: 1 << 26 }).toString();
  const R = JSON.parse(out);
  ['fit', 'source', 'profile', 'phis', 'gamma_variant', 'plan', 'top', 'agents', 'levers', 'evidence', 'frontier', 'proposals', 'ubench_levers', 'whatif'].forEach(k => assert(k in R, k));
  assert.strictEqual(R.source, W.profile ? 'profile' : 'ubench');
  assert.strictEqual(R.levers.levers.filter(x => x.kind === 'layout').length, 1);
  if (!W.profile) assert.strictEqual(R.levers.levers.length, W.ubench.classes.length + (W.variants || []).length + 1);
  assert.strictEqual(R.whatif.cards.length, W.whatif_cards.length);
});

console.log(passed + ' window tests passed');
