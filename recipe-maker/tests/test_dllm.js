#!/usr/bin/env node
// the dLLM (block-diffusion) term in src/model.js: coverage, TPF cells, the kernel-gap fit and its pooling, the UNMODELLED
// rules, the held-out run and --check on a dLLM recipe. Run: node tests/test_dllm.js (needs data.json: python3 build.py)
'use strict';
const assert = require('assert');
const fs = require('fs'), path = require('path'), cp = require('child_process');
const ROOT = path.join(__dirname, '..');
const FM = require(path.join(ROOT, 'src', 'model.js'));
const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'data.json'), 'utf8'));
FM.applyCal(data);
const D = data.dllm;
const clone = x => JSON.parse(JSON.stringify(x));
const recipe = f => JSON.parse(fs.readFileSync(path.join(ROOT, 'recipes', f), 'utf8'));
let n = 0;
const t = (name, fn) => { fn(); n++; console.log('ok ' + name); };

// a synthetic dLLM block on the real odie SYCL line: one MoE model, rows generated from the formula with a known gap
const synth = (eff, opts) => {
  const d = clone(data), o = opts || {};
  d.dllm.holdout = { v: [] };
  d.dllm.models = { mx: { arch: 'x', moe: true, L: 20, E: 256, k: 8, shape: '2048x512', experts_gib: { v: 10 }, nonexpert_gib: { v: 1 }, total_gib: { v: 11 } },
    my: { arch: 'y', moe: true, L: 20, E: 256, k: 8, shape: '4096x1024', experts_gib: { v: 40 }, nonexpert_gib: { v: 2 }, total_gib: { v: 42 } } };
  const row = (id, model, b, tpf, e, extra) => {
    const r = Object.assign({ id, when: 'x', box: 'odie', model, backend: 'sycl', path: 'per-pair', cards: 1, quant: 'q4_0', schedule: 's', mode: 'fuse', block: 32, workload: 'w', placement: 'all_gpu', ok: true, tpf, b }, extra || {});
    const L = FM.dllmLine(d, 'odie', 'sycl'), m = d.dllm.models[model], GB = FM.GiB / 1e9, rate = FM.v(L.rate);
    const ms = FM.v(L.host) + FM.v(m.nonexpert_gib) * GB / rate * 1000 + FM.v(m.experts_gib) * GB * FM.dllmCover(m, b, FM.v(d.dllm.routing)) / rate * 1000 / e;
    r.ms_per_fwd = ms; r.decode_tps = tpf * 1000 / ms;
    return r;
  };
  d.dllm.rows = [row('a1', 'mx', 36, 3.0, eff[0]), row('a2', 'mx', 40, 3.2, eff[0]), row('a3', 'mx', 38, 3.1, eff[0]), row('b1', 'my', 38, 4.0, eff[1]), row('b2', 'my', 40, 4.2, eff[1])].concat(o.extra ? o.extra(row) : []);
  return d;
};
const rec = (model, extra) => Object.assign({ kind: 'dllm', box: 'odie', model, backend: 'sycl', moe_path: 'per-pair', cards: ['b70_2'], quant: 'q4_0', schedule: { name: 's', mode: 'fuse' }, block: 32, workload: 'w', placement: 'all_gpu', ctx: 512 }, extra || {});

t('coverage: 1 - (1 - k/E)^B, 1 for dense', () => {
  const m = { moe: true, k: 8, E: 256 };
  assert.ok(Math.abs(FM.dllmCover(m, 1) - 8 / 256) < 1e-12);
  assert.ok(Math.abs(FM.dllmCover(m, 32) - (1 - Math.pow(248 / 256, 32))) < 1e-12);
  assert.ok(FM.dllmCover(m, 32) < 0.7, 'uniform routing: a 32-row block touches ~64 % of the experts, not nearly all');
  assert.strictEqual(FM.dllmCover({ moe: false }, 32), 1);
  assert.ok(Math.abs(FM.dllmCover(m, 32, 0.5) - FM.dllmCover(m, 16)) < 1e-12, 'locality rho: B rows touch as many experts as rho x B uniform picks');
});

t('routing locality is measured: rho from the fixed-probe ledger, a 32-row block well under uniform', () => {
  assert.strictEqual(D.routing.tag, 'MEASURED');
  assert.strictEqual(D.routing.evidence, 'seen');
  const rho = FM.v(D.routing), c = FM.dllmCover({ moe: true, k: 8, E: 256 }, 32, rho);
  assert.ok(rho > 0.15 && rho < 0.4, 'LLaDA2.1-mini block forwards: rho ' + rho);
  assert.ok(c > 0.15 && c < 0.35, '32-row coverage ' + c + ' (uniform 0.64; the probe bug read 1.0)');
  assert.ok(/coverage-2026-10-19/.test(D.routing.src));
});

t('fit recovers the kernel gap and pools groups that agree', () => {
  const d = synth([0.12, 0.11]), T = FM.dllmTerms(d), K = T.eff['odie|sycl|per-pair'];
  assert.ok(Math.abs(K.groups['2048x512|q4_0'].v - 0.12) < 1e-9);
  assert.ok(Math.abs(K.groups['4096x1024|q4_0'].v - 0.11) < 1e-9);
  assert.ok(K.pools, 'spread 9 % <= eff_spread_max');
  assert.strictEqual(K.b_min, 36); assert.strictEqual(K.b_max, 40);
  assert.strictEqual(T.tpf['mx|s|32|w'].v, 3.1); assert.strictEqual(T.tpf['mx|s|32|w'].tag, 'MEASURED');
  const p = FM.predictDllm(d, rec('mx'));
  assert.strictEqual(p.status, 'PREDICTED');
  const own = d.dllm.rows.find(r => r.id === 'a3');
  assert.ok(Math.abs(p.ms_per_fwd / own.ms_per_fwd - 1) < 1e-9, 'B 38 = the cell median: the forward is reproduced exactly');
  assert.ok(Math.abs(p.decode_tps - 3.1 * 1000 / p.ms_per_fwd) < 1e-9, 'decode = TPF x 1000 / ms per forward');
});

t('a lone group borrows the pooled gap only when the groups agree', () => {
  const q5 = e => synth([0.12, 0.12], { extra: row => [row('c1', 'my', 38, 4.0, e, { quant: 'q5_K_M' })] });
  const r = rec('my', { quant: 'q5_K_M' });
  // c1's own group holds 1 row < eff_min_rows: it takes the pooled gap when the groups agree within eff_spread_max
  const p = FM.predictDllm(q5(0.13), r);
  assert.strictEqual(p.status, 'PREDICTED', p.why);
  assert.ok(p.assumptions.some(a => /pooled over/.test(a)));
  const q = FM.predictDllm(q5(0.3), r);
  assert.strictEqual(q.status, 'UNMODELLED'); assert.ok(/differs by expert shape/.test(q.why), q.why);
  assert.strictEqual(FM.predictDllm(q5(0.3), rec('my')).status, 'PREDICTED', 'my|q4_0 holds 2 rows: its own group stands');
});

t('TPF rows that spread past tpf_spread_max are CONFOUNDED', () => {
  const d = synth([0.12, 0.12]);
  d.dllm.rows.find(r => r.id === 'a1').tpf = 2.0;
  assert.strictEqual(FM.dllmTerms(d).tpf['mx|s|32|w'].tag, 'MEASURED, CONFOUNDED');
});

t('UNMODELLED: spec, full forwards, CPU experts beside a GPU, B out of range, no line, dense', () => {
  const d = synth([0.12, 0.11]);
  const why = r => { const p = FM.predictDllm(d, r); assert.strictEqual(p.status, 'UNMODELLED', JSON.stringify(r)); assert.ok(p.why); assert.ok(p.evidence.length, 'carries an evidence line'); return p.why; };
  assert.ok(/two models/.test(why(rec('mx', { schedule: { name: 's', mode: 'spec' } }))));
  assert.ok(/whole sequence/.test(why(rec('mx', { schedule: { name: 's', mode: 'full' } }))));
  assert.ok(/CPU-held experts/.test(why(rec('mx', { placement: 'n_cpu_moe:18' }))));
  assert.ok(/no measured line/.test(why(rec('mx', { box: 'nobox-x' }))));
  d.dllm.rows.push(Object.assign({}, d.dllm.rows[0], { id: 'z', model: 'mx', schedule: 'wide', b: 400, tpf: 3, ok: true, mode: 'cli' }));
  assert.ok(/outside the fitted/.test(why(rec('mx', { schedule: { name: 'wide', mode: 'fuse' } }))));
  d.dllm.models.dz = { moe: false, L: 28, shape: 'dense', experts_gib: { v: 0 }, nonexpert_gib: { v: 4 }, total_gib: { v: 4 } };
  d.dllm.rows.push(Object.assign({}, d.dllm.rows[0], { id: 'dz1', model: 'dz' }));
  assert.ok(/dense/.test(why(rec('dz'))));
});

t('the held-out run is out of every term and predicted within 10 %', () => {
  const ho = D.holdout.v, T = FM.dllmTerms(data);
  assert.ok(ho.length >= 1);
  ho.forEach(id => {
    assert.ok(T.used.indexOf(id) < 0, id + ' fits no gap');
    Object.values(T.tpf).forEach(x => assert.ok(x.runs.indexOf(id) < 0, id + ' sets no TPF'));
  });
  const V = FM.dllmValidation(data);
  V.holdout.forEach(x => { assert.strictEqual(x.status, 'PREDICTED', x.why); assert.ok(Math.abs(x.err) < 0.1, x.id + ' ' + x.err); });
  assert.ok(V.loo_summary.n >= 10 && V.loo_summary.median_abs_pct < 10, JSON.stringify(V.loo_summary));
});

t('--check on a dLLM recipe prints the gate and the trust lines', () => {
  const f = path.join(ROOT, 'recipes', 'odie-dllm-llada22flash-2xb70-perpair-lev-s-fuse.json');
  const out = cp.execFileSync('node', [path.join(ROOT, 'report.js'), '--check', f], { encoding: 'utf8' });
  assert.ok(/trust\s+decode/.test(out), out.slice(-2000));
  assert.ok(/trust\s+ms per forward/.test(out), out.slice(-2000));
  const c = FM.checkRecipe(data, recipe('odie-dllm-llada22flash-2xb70-perpair-lev-s-fuse.json'));
  assert.ok(c.checks.some(x => x.id === 'vram_fit' && x.status === 'PASS'));
  assert.ok(c.checks.some(x => x.id === 'conditions'));
  assert.strictEqual(c.prediction.status, 'PREDICTED');
  const bad = FM.checkRecipe(data, Object.assign(recipe('odie-dllm-llada22flash-2xb70-perpair-lev-s-fuse.json'), { cards: ['b70_2'] }));
  assert.ok(bad.checks.some(x => x.id === 'vram_fit' && x.status === 'FAIL'), 'a 100B Q4_0 does not fit one 32 GB card');
});

t('the AR control predicts from the bench line alone', () => {
  const r = D.rows.find(x => x.id === 'odie-dllm-n1-ling-q4-ar-full');
  const p = FM.predictDllm(data, rec(r.model, { schedule: { name: r.schedule, mode: 'ar' }, workload: r.workload, quant: r.quant }));
  assert.strictEqual(p.status, 'PREDICTED'); assert.strictEqual(p.tpf, 1);
});

console.log(`${n} tests passed`);
