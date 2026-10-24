'use strict';
// agents vs speed (src/model.js curve*, data.curves from build.py + data/curves.json): run after python3 build.py
const assert = require('assert'), fs = require('fs'), path = require('path');
const FM = require('../src/model.js');
const ROOT = path.join(__dirname, '..');
const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'data.json'), 'utf8'));
const clone = x => JSON.parse(JSON.stringify(x));
const v = FM.v, S = data.curves.settings;
let passed = 0;
const test = (name, fn) => { fn(); passed++; console.log('ok ' + name); };
const skip = name => console.log('skip ' + name + ' (its ledger rows are not in this build)');
const tag = (x, t) => ({ v: x, tag: t || 'MEASURED', src: 'test' });
// a synthetic config: (n, total) pairs, groups, stages, slots; the fitted curve must not need any ledger row
const cfg = (pts, o) => Object.assign({ key: 'k', id: 'synthetic', label: 'synthetic', box: 'odie', model: 'fnx', engine: 'strata', ctx: 131072, draft: true, runs: ['r'],
  groups: tag(2, 'READ'), stages: tag(2, 'READ'), slots: tag(32, 'READ'), serving: { kv_resident: 4096, kv: 'int8', max_context: 131072, batch_groups: 2, batch_mtp: false },
  points: pts.map(([n, t, w]) => ({ n, run: 'r', used: true, why: null, tps: tag(t), win_ms: w ? tag(w, 'READ') : null, watts: null })) }, o || {});
// the measured 2026-10-23 pair of the all-on 2x B70 run (A3), as plain numbers, so the case runs without its ledger row
const A3 = cfg([[2, 102.4, 9.70], [32, 325.9, 45.87]]);

test('two points: the law passes through both, and A, B solve n / total = A max(G,S) + B max(G,S) n / G', () => {
  const f = FM.curveFit(data, A3);
  assert(f.ok, f.why);
  f.errs.forEach(e => assert(Math.abs(e.err) < 1e-12, JSON.stringify(e)));
  assert(Math.abs(2 * f.A + 2 * f.B - 2 / 102.4) < 1e-12);
  assert(Math.abs(2 * f.A + 32 * f.B - 32 / 325.9) < 1e-12);
});

test('the A3 readouts: 32 agents at >= 10 each (325.9 total), 13 at >= 20, asymptote ~381', () => {
  const R = FM.curveReport(data, A3, { min: 10, second: 20 });
  assert.strictEqual(R.headline.n, 32);
  assert(Math.abs(R.headline.total - 325.9) < 1e-9);
  assert.strictEqual(R.second.n, 13);
  assert(R.second.each >= 20 && R.rows[13].each < 20, 'n = 14 drops under 20 each');
  assert(Math.abs(R.asymptote - 381.4) < 0.1, R.asymptote);
  assert.strictEqual(R.rows.length, v(S.agents_max));
  assert(R.rows.every((r, i) => i === 0 || r.total > R.rows[i - 1].total), 'the total rises with every agent');
  assert(R.rows.every(r => r.total < R.asymptote));
});

test('the readout thresholds and the curve length are settings: data and opts both move them', () => {
  const d = clone(data);
  d.curves.settings.min_each_tps.v = 20; d.curves.settings.second_each_tps.v = 10; d.curves.settings.agents_max.v = 40;
  const R = FM.curveReport(d, A3);
  assert.strictEqual(R.headline.n, 13);
  assert.strictEqual(R.second.n, 32);
  assert.strictEqual(R.rows.length, 40);
  assert.strictEqual(FM.curveReport(data, A3, { min: 5 }).headline.n, v(S.agents_max), 'every count up to agents_max holds 5 each');
  assert(FM.curveReport(data, A3, { min: 5 }).headline.at_max);
  assert.strictEqual(FM.curveReport(data, A3, { min: 1000 }).headline.n, 0);
});

test('uneven groups need no special case: the window is linear in r, so n need not divide by G', () => {
  const f = FM.curveFit(data, A3), t33 = f.at(33);
  // 33 agents in 2 groups (17 + 16): two windows a + 17b and a + 16b on the slowest stage = 2a + 33b, the same as r = 16.5
  const a = f.A, b = f.B, cyc = (a + 17 * b) + (a + 16 * b);
  assert(Math.abs(t33 - 33 / cyc) < 1e-9);
});

test('fewer groups than stages leave a stage idle: max(G, S) windows a cycle, so one agent gets half of two in two groups', () => {
  const f = FM.curveFit(data, A3);
  assert(Math.abs(f.at(1) - f.at(2) / 2) < 1e-12, 'G = 1 < S = 2 at n = 1');
  const one = cfg([[2, 102.4], [32, 325.9]], { groups: tag(1, 'READ') }), g1 = FM.curveFit(data, one);
  assert(g1.ok && Math.abs(g1.at(2) - 102.4) < 1e-9 && Math.abs(g1.at(32) - 325.9) < 1e-9);
});

test('the stage window: eff per point from the run note, and the line t = a + b r through two counts', () => {
  const w = FM.curveFit(data, A3).win;
  assert(Math.abs(w.pts[0].eff - 102.4 / (1000 * 2 / (2 * 9.70))) < 1e-12);
  assert(Math.abs(w.pts[1].eff - 325.9 / (1000 * 32 / (2 * 45.87))) < 1e-12);
  assert(Math.abs(w.b - (45.87 - 9.70) / 15) < 1e-9 && Math.abs(w.a - (9.70 - w.b)) < 1e-9);
  assert(w.eff > 0.9 && w.eff <= 1, 'the window law holds the measured totals within 10 %');
  assert.strictEqual(FM.curveFit(data, cfg([[2, 102.4], [32, 325.9]])).win, null, 'no window without a note');
});

test('a fit that misses a measured point by more than fit_warn_pct is flagged', () => {
  const c = cfg([[2, 102.4], [16, 250], [32, 325.9]]), R = FM.curveReport(data, c), d = clone(data);
  assert.strictEqual(R.fit_warn, 100 * R.fit.err_max > v(S.fit_warn_pct));
  d.curves.settings.fit_warn_pct.v = 0;
  assert(FM.curveReport(d, c).fit_warn);
  assert(!FM.curveReport(data, A3).fit_warn, 'two points fit exactly');
});

test('points that fall as agents rise, or one agent count, give no curve and say why', () => {
  const bad = FM.curveReport(data, cfg([[1, 60.4], [2, 58.4], [4, 37.4]], { groups: tag(1, 'READ'), stages: tag(1, 'READ') }));
  assert(!bad.ok && /do not follow the window law/.test(bad.why), bad.why);
  const one = FM.curveReport(data, cfg([[32, 254.4], [32, 254.0]]));
  assert(!one.ok && /1 distinct agent count/.test(one.why), one.why);
  const unused = cfg([[1, 66.98], [4, 134.65], [8, 181.62]]);
  unused.points[0].used = false;
  assert.strictEqual(FM.curveFit(data, unused).pts.length, 2, 'a point kept out of the fit stays out');
});

test('watts: tok/s per watt at measured counts, straight-line ESTIMATE between them, none past them', () => {
  const c = cfg([[2, 102.4], [32, 325.9]]);
  c.points[0].watts = tag(400); c.points[1].watts = tag(700);
  const R = FM.curveReport(data, c, { min: 10, second: 20 });
  assert.deepStrictEqual(R.power.map(p => +p.tps_per_w.toFixed(4)), [+(102.4 / 400).toFixed(4), +(325.9 / 700).toFixed(4)]);
  assert.strictEqual(R.headline.watts.tag, 'MEASURED');
  assert(Math.abs(R.headline.tps_per_w - 325.9 / 700) < 1e-12);
  assert.strictEqual(R.second.watts.tag, 'ESTIMATE');
  assert(Math.abs(R.second.watts.w - (400 + 300 * 11 / 30)) < 1e-9);
  assert.strictEqual(FM.curveWatts(c, 40), null);
});

const hostOk = data.models.fnx && data.models.fnx.kv_host && data.checks.host_pinned && data.checks.host_pinned.terms['fnx/strata'];
(hostOk ? test : skip)('the host-memory ceiling: the host-pin check passes at the ceiling and fails one slot above; lend drop raises it', () => {
  const R = FM.curveReport(data, A3);
  const hc = R.host;
  assert(hc.ceiling >= 1 && hc.ceiling < v(S.agents_max), JSON.stringify(hc).slice(0, 300));
  assert.strictEqual(FM.curveHost(data, A3, hc.ceiling).status, 'PASS');
  assert.strictEqual(FM.curveHost(data, A3, hc.ceiling + 1).status, 'FAIL');
  assert.strictEqual(hc.first_over.n, hc.ceiling + 1);
  assert(hc.with_lend_drop > hc.ceiling, 'dropping the stage-lend duplicate frees room for more slots');
  assert(R.rows.every(r => r.fits_host === (r.n <= hc.ceiling)));
  // the headline past the ceiling also reads out the best count inside it
  if (R.headline.n > hc.ceiling) assert(R.headline.past_host && R.headline_in_memory.n === hc.ceiling);
  const d = clone(data);
  d.boxes.odie.dram.total_gib.v += 64;
  assert(FM.curveReport(d, A3).host.ceiling > hc.ceiling, 'the box DRAM drives the ceiling');
  assert.strictEqual(FM.curveHost(data, Object.assign({}, A3, { slots: null }), 8).status, 'UNKNOWN', 'no --batch, no serving shape');
});

// the build's configs from the ledger
const C = data.curves.configs;
test('every config point is a MEASURED ledger value; groups and stages are READ from the run flags', () => {
  C.forEach(c => {
    assert(c.runs.length && c.points.every(p => p.tps.tag === 'MEASURED' && c.runs.indexOf(p.run) >= 0), c.id);
    assert(c.groups.tag === 'READ' && c.stages.tag === 'READ' && v(c.groups) >= 1 && v(c.stages) >= 1, c.id);
    c.points.filter(p => !p.used).forEach(p => assert(p.why, c.id + ' n=' + p.n + ' kept out without a reason'));
    if (c.slots) c.points.filter(p => p.n > v(c.slots)).forEach(p => assert(!p.used, c.id + ' n=' + p.n + ' above the slots'));
  });
});

const a3 = C.find(c => c.runs.indexOf('odie-fnx-iq3xxs-2xb70-split-batch32-groups2-all3-v4s25-allon-quick-1023') >= 0);
(a3 ? test : skip)('the A3 ledger config: 2 groups, 2 stages, 32 slots READ from its flags; its readout matches the plain numbers', () => {
  assert.deepStrictEqual([v(a3.groups), v(a3.stages), v(a3.slots)], [2, 2, 32]);
  assert.strictEqual(a3.serving.max_context, 131072);
  assert.strictEqual(a3.serving.kv, 'int8');
  const R = FM.curveReport(data, a3, { min: 10, second: 20 }), T = FM.curveReport(data, A3, { min: 10, second: 20 });
  assert.deepStrictEqual([R.headline.n, R.second.n, R.host.ceiling], [T.headline.n, T.second.n, T.host.ceiling]);
  assert(Math.abs(R.fit.win.a - T.fit.win.a) < 1e-9, 'the stage windows come from its layout note');
  assert.strictEqual(data.curves.default, a3.key, 'the default config holds the window frontier run');
});

const fr2 = C.find(c => c.runs.indexOf('odie-fnx-iq3xxs-2xb70-split-batch8-nobmtp-groups2-franken2-1023') >= 0);
(fr2 ? test : skip)('n = 1 with a drafter and --no-batch-mtp runs the single-stream path: kept out; the split stage count comes from the isolation note', () => {
  const p1 = fr2.points.find(p => p.n === 1);
  assert(!p1.used && /single-stream/.test(p1.why), JSON.stringify(p1));
  assert.strictEqual(v(fr2.stages), 2);
});

// the 2026-10-23 uniform agent-count sweep: one fresh server per agent count ('--batch=agents'), every slot 131,072 cells, STRATA_STAGE_LEND_DROP=1;
// the host-pin model at each measured point stays within PIN_TOL_GIB of the engine's own init-line pins (conservative side
// mostly: the model keeps the brief's 2 MiB a copy, the sweep build showed whole cells)
const PIN_TOL_GIB = 2.5;
const SWEEP = { 'odie-fnx-iq3xxs-2xb70-sweep-u32i-1023': { kv: 'int8', pins: { 24: 45.63, 32: 60.99 } }, 'odie-fnx-iq3xxs-2xb70-sweep-u32k-1023': { kv: 'k8v4', pins: { 32: 47.72 } } };
Object.entries(SWEEP).forEach(([run, want]) => {
  const c = C.find(x => x.runs.indexOf(run) >= 0);
  (c ? test : skip)('the uniform sweep config of ' + run + ': a serving shape per agent count, the measured lend-drop pins, a host ceiling and tok/s per W', () => {
    assert(!c.slots && v(c.slots_each_point) === true && c.slots_each_point.tag === 'READ', JSON.stringify(c.slots_each_point));
    assert.deepStrictEqual([c.serving.kv, c.serving.max_context, c.serving.stage_lend_drop, c.serving.batch_groups], [want.kv, 131072, true, 2]);
    Object.entries(want.pins).forEach(([n, gib]) => {
      const h = FM.curveHost(data, c, +n);
      assert.strictEqual(h.status, 'PASS', h.detail);
      assert(Math.abs(h.pred.pinned_gib - gib) < PIN_TOL_GIB, 'n=' + n + ' model ' + h.pred.pinned_gib.toFixed(2) + ' vs measured ' + gib);
      assert.strictEqual(h.pred.terms.stage_lend, 0, 'the lend copy is dropped');
    });
    const R = FM.curveReport(data, c);
    assert(R.host.ceiling >= Math.max(...Object.keys(want.pins).map(Number)), 'every measured count fits: ' + JSON.stringify(R.host).slice(0, 300));
    assert(R.power.length && R.power.every(p => p.tps_per_w > 0), 'tok/s per W from measured.power_w');
  });
});

console.log(passed + ' passed');
