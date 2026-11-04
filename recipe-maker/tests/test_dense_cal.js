#!/usr/bin/env node
// the dense-path calibration (data/dense-cal.json, model.js denseCal): kernel rates per quant type from their microbenchmarks, the
// fitted terms reproduce their fit points, the held-out points stay inside the stated error, the plain line (before) is the old
// dense path, and predictDense widens its stated error where no measurement covers a term. Run: node tests/test_dense_cal.js
// (needs data.json: python3 build.py)
'use strict';
const assert = require('assert');
const fs = require('fs'), path = require('path');
const ROOT = path.join(__dirname, '..');
const FM = require(path.join(ROOT, 'src', 'model.js'));
const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'data.json'), 'utf8'));
FM.applyCal(data);
const clone = x => JSON.parse(JSON.stringify(x));
let n = 0;
const t = (name, fn) => { fn(); n++; console.log('ok ' + name); };
const C = FM.denseCal(data), row = id => C.rows.find(r => r.id === id), term = id => C.terms.find(x => x.id === id);

t('every point names its source and a use; fit_on points exist and are use=fit', () => {
  C.points.forEach(p => { assert.ok(p.src && p.src.length > 20, p.id); assert.ok(['fit', 'holdout', 'context', 'line'].includes(p.use), p.id); });
  C.terms.forEach(x => x.fit_on.forEach(id => assert.strictEqual(C.points.find(p => p.id === id).use, 'fit', id)));
  assert.ok(C.rows.filter(r => r.use === 'holdout').length >= 2);
});

t('kernel rates are the microbenchmark GB/s over the same bench reference', () => {
  const k = C.kernels.find(x => x.backend === 'sycl' && x.eff.IQ4_XS !== undefined);
  assert.ok(Math.abs(k.eff.IQ4_XS - 324 / 585) < 1e-9);
  assert.ok(Math.abs(k.eff.IQ4_NL - 98 / 585) < 1e-9);
  const b = C.kernels.find(x => x.eff.IQ3_XXS !== undefined);
  assert.strictEqual(b.eff.IQ3_XXS, k.eff.IQ3_S);          // borrowed, and tagged so
  assert.strictEqual(b.tag, 'ESTIMATE');
});

t('fitted terms reproduce their fit points and are physical', () => {
  ['sycl-qwen27-udq4kxl-e4077b9-d1k5', 'sycl-qwen27-udq4kxl-e4077b9-d16k5', 'cuda-turing-gsq-iq3s-b11175'].forEach(id => assert.ok(Math.abs(row(id).err_after) < 0.005, id + ' ' + row(id).err_after));
  const kv = term('sycl_kv_read_eff').v, lm = term('sycl_linear_layer_ms').v, wm = term('turing_iq_ms_per_gweight').v;
  assert.ok(kv > 0.1 && kv < 1, 'kv eff ' + kv);
  // the 2026-09-23 profile: DeltaNet state + conv ~3.5 ms, tiny Q8_0 projections ~1.2, small ops a few ms over 48 layers
  assert.ok(lm > 0.08 && lm < 0.25, 'linear layer ms ' + lm);
  // Turing IQ: ~22 G weights in ~35 ms of a 40.8 ms step
  assert.ok(wm > 1.2 && wm < 2.0, 'IQ ms per G weights ' + wm);
  C.terms.forEach(x => { assert.strictEqual(x.tag, 'FIT'); assert.ok(/^FIT: /.test(x.src)); });
});

t('held-out points: within the stated tolerance after, and closer than before', () => {
  const H = C.rows.filter(r => r.use === 'holdout');
  H.forEach(r => assert.ok(Math.abs(r.err_after) <= Math.abs(r.err_before) + 1e-9, r.id));
  H.filter(r => r.backend === 'sycl').forEach(r => assert.ok(Math.abs(r.err_after) < 0.15, r.id + ' ' + r.err_after));
  H.filter(r => r.backend === 'cuda').forEach(r => assert.ok(Math.abs(r.err_after) < 0.05, r.id + ' ' + r.err_after));
  // the error the task names: qwen27 Q4_K_M on a B70 was +38 % (34.2 tok/s) against the power-scaled measurement
  const q = row('sycl-qwen27-q4km-ext-150w');
  assert.ok(Math.abs(q.before - 34.2) < 0.15, 'before ' + q.before);
  assert.ok(q.err_before > 0.3 && Math.abs(q.err_after) < 0.12);
  // Turing IQ2_XS: +117 % before (the bytes line), within 5 % after
  const i = row('cuda-turing-gsq-iq2xs-b11175');
  assert.ok(i.err_before > 1 && Math.abs(i.err_after) < 0.05);
  // the plain and Ampere rows do not move: no factor applies to them
  ['cuda-turing-llama2-7b-q40-ext', 'cuda-ampere-qwen27-udq4kxl-screen'].forEach(id => assert.ok(Math.abs(row(id).after - row(id).before) < 1e-9, id));
  assert.ok(C.base_err.sycl > 0 && C.base_err.cuda > 0);
});

t('a holdout never sets a term: moving its measurement leaves every term unchanged', () => {
  const d = clone(data);
  d.dense_cal.points.find(p => p.id === 'sycl-qwen27-q4km-ext-150w').tps = 5;
  const C2 = FM.denseCal(d);
  C.terms.forEach(x => assert.strictEqual(C2.terms.find(y => y.id === x.id).v, x.v));
  assert.ok(C2.base_err.sycl > C.base_err.sycl);
});

// a synthetic hybrid spec: 48 linear + 16 full layers of qwen27's Q4_K_M bytes (the ext holdout point's composition)
const P = C.points.find(p => p.id === 'sycl-qwen27-q4km-ext-150w');
const spec = (types, ctx) => {
  const kinds = [], lt = [];
  for (let i = 0; i < 64; i++) {
    const k = i % 4 === 3 ? 'full' : 'linear', L = P.layers[k];
    kinds.push(k);
    lt.push(Object.fromEntries(Object.entries(types ? types(L.types) : L.types).map(([x, b]) => [x, b / L.n])));
  }
  const out = P.layers.output.types;
  return { model: 'syn', arch: 'x', ctx, kv_type: 'q8_0', depths: [0, ctx / 2, ctx],
    kv_layers: kinds.map(k => ({ f16_bytes_per_token: k === 'full' ? 4096 : 0, window: null, kind: k })),
    quants: [{ label: 'Q4_K_M', engine: 'llamacpp', layer_bytes: lt.map(x => Object.values(x).reduce((a, b) => a + b, 0)), layer_types: lt,
      output_bytes: Object.values(out).reduce((a, b) => a + b, 0), output_types: out, embd_bytes: 0.715e9 }] };
};

t('predictDense on odie: the calibrated qwen27 Q4_K_M lands on the measured point; uncal is the old dense path', () => {
  const s = spec(null, 8192), q = s.quants[0];
  const a = FM.predictDense(data, s, 'odie', q, {}), b = FM.predictDense(data, s, 'odie', q, { uncal: true });
  const ms = 1000 / a.decode[0].tps, pt = FM.densePointMs(data, C, Object.assign({}, P, { depth: 0 }));
  assert.ok(Math.abs(ms - pt) < 0.05, ms + ' vs ' + pt);
  assert.ok(b.decode[0].tps > a.decode[0].tps * 1.2);
  assert.ok(a.terms.some(x => /ms a linear layer adds/.test(x.term) && x.tag === 'FIT'));
  assert.ok(a.error && a.error.plus_pct >= 10 && a.error.minus_pct >= a.error.plus_pct);
});

t('stated error widens for a kernel-bound type without a measured rate and for an unfitted layer kind', () => {
  const s = spec(T => ({ IQ3_XXS: Object.values(T).reduce((a, b) => a + b, 0) }), 8192), q = s.quants[0];
  const o = FM.predictDense(data, s, 'odie', q, {});                        // IQ3_XXS on SYCL: borrowed rate
  assert.ok(o.error.why.some(w => /IQ3_XXS \(borrowed rate\)/.test(w)), o.error.why.join(' | '));
  const sn = FM.predictDense(data, s, 'snoopy', q, { knobs: { freeDesktop: true } });   // no IQ rate on the 3090 Ti
  assert.ok(sn.error.why.some(w => /no measured kernel rate: IQ3_XXS on snoopy/.test(w)));
  const dr = FM.predictDense(data, spec(null, 8192), 'droopy', q, {});      // linear layers on CUDA: assumed 0
  assert.ok(dr.error.why.some(w => /no fitted cost: linear layers on droopy/.test(w)));
  const sa = FM.predictDense(data, spec(null, 8192), 'snoopy', spec(null, 8192).quants[0], { knobs: { freeDesktop: true } });   // measured on the 3090 Ti
  assert.ok(!sa.error.why.some(w => /linear layers on snoopy/.test(w)), sa.error.why.join(' | '));
});

t('llama.cpp VRAM headroom: --fit on keeps llamacpp_fit free, -ngl pinned keeps llamacpp_ngl', () => {
  const M = data.checks.vram_peak.margin;
  assert.ok(M.llamacpp_fit.v > M.llamacpp_fit.lo && M.llamacpp_fit.lo >= 1.0);
  assert.ok(M.llamacpp_ngl.hi > 0.4 && M.llamacpp_ngl.hi < 1.0);
  const s = spec(null, 8192), q = s.quants[0];
  const fit = FM.predictDense(data, s, 'odie', q, {}), ngl = FM.predictDense(data, s, 'odie', q, { knobs: { ngl: 99 } });
  assert.strictEqual(fit.cards[0].margin_gib, M.llamacpp_fit.v);
  assert.strictEqual(ngl.cards[0].margin_gib, M.llamacpp_ngl.hi);
  assert.ok(fit.terms.some(x => /--fit on/.test(x.term)));
});

t('fit_on that names a holdout leaves the term unfitted (a holdout can never set a term, by construction)', () => {
  const d = clone(data);
  d.dense_cal.terms.find(x => x.id === 'sycl_linear_layer_ms').fit_on = ['sycl-qwen27-q4km-ext-150w'];
  const x = FM.denseCal(d).terms.find(y => y.id === 'sycl_linear_layer_ms');
  assert.strictEqual(x.v, null);
  assert.strictEqual(x.tag, 'UNKNOWN');
});

t('odie test cards: the resident on B70 #2 is planned stopped and named; freeTestCards false plans around it', () => {
  // 64 layers of ~0.8 GB: more than one B70 holds beside the --fit headroom, so the plan needs B70 #2
  const s = spec(T => Object.fromEntries(Object.entries(T).map(([x, b]) => [x, b * 2.3])), 8192), q = s.quants[0];
  const free = FM.predictDense(data, s, 'odie', q, {}), keep = FM.predictDense(data, s, 'odie', q, { knobs: { freeTestCards: false } });
  assert.strictEqual(free.layers_cpu, 0);
  assert.ok(free.cards.some(c => c.card === 'odie b70_2' && c.layers > 0 && c.reserved_gib === 0));
  assert.strictEqual(free.displaces.length, 1);
  assert.ok(/qwenlocal/.test(free.displaces[0].what) && free.displaces[0].card === 'odie b70_2' && free.displaces[0].gib > 20);
  assert.ok(free.terms.some(x => /resident service displaced \(odie b70_2\)/.test(x.term)));
  assert.ok(keep.layers_cpu > 0, 'kept resident: the plan spills to the CPU');
  assert.strictEqual(keep.displaces.length, 0);
  // a plan that fits one card displaces nothing; the old path's default (freeHousehold false) is untouched
  const small = FM.predictDense(data, spec(null, 8192), 'odie', spec(null, 8192).quants[0], {});
  assert.strictEqual(small.displaces.length, 0);
  assert.strictEqual(data.knobs.freeHousehold, false);
});

t('a borrowed kernel rate states a faster side: every borrowed byte at the fastest measured rate on the card', () => {
  const s = spec(T => ({ IQ3_XXS: Object.values(T).reduce((a, b) => a + b, 0) }), 8192), q = s.quants[0];
  const o = FM.predictDense(data, s, 'odie', q, {});
  const w = o.error.why.find(x => /borrowed rate: at the fastest measured kernel-bound rate \(IQ4_XS on odie b70_1\)/.test(x));
  assert.ok(w, o.error.why.join(' | '));
  // nearly every byte is IQ3_XXS at 105 GB/s; re-timed at IQ4_XS 324 the step is ~2x+ faster
  assert.ok(o.error.plus_pct > 100, String(o.error.plus_pct));
  const m = FM.predictDense(data, spec(null, 8192), 'odie', spec(null, 8192).quants[0], {});   // K-quants only: no borrowed side
  assert.ok(!m.error.why.some(x => /borrowed rate/.test(x)));
});

console.log(n + ' tests passed');
