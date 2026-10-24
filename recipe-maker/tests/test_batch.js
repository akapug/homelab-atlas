'use strict';
// the batched-decode term (src/model.js batchStep / batchFit / batchTable, data.batch from build.py): run after python3 build.py
const assert = require('assert'), fs = require('fs');
const FM = require('../src/model.js');
const data = JSON.parse(fs.readFileSync('data.json', 'utf8'));
const clone = x => JSON.parse(JSON.stringify(x));
const B = data.batch, v = FM.v;
const TAGS = new Set(['MEASURED', 'READ', 'PRIOR', 'ESTIMATE', 'FIT']);
let passed = 0;
const test = (name, fn) => { fn(); passed++; console.log('ok ' + name); };
const fit = FM.batchFit(data, B.miss_modes.v[0]);
const table = FM.batchTable(data, fit);
const row = (d, cfg, variant, ctx) => FM.batchTable(d, FM.batchFit(d, d.batch.miss_modes.v[0])).find(t => t.config === cfg && t.variant === variant && t.ctx === ctx);

test('distinct experts: k at one position, E in the limit, never more than the picks', () => {
  assert.strictEqual(FM.batchDistinct(512, 10, 1), 10);
  assert(Math.abs(FM.batchDistinct(512, 10, 1e4) - 512) < 1e-6);
  [2, 4, 8, 32].forEach(n => { const d = FM.batchDistinct(512, 10, n); assert(d < 10 * n && d > FM.batchDistinct(512, 10, n / 2)); });
});

test('every batch constant is tagged and sourced; the free ones are not in the data', () => {
  const nums = Object.entries(B).filter(([, x]) => x && typeof x === 'object' && 'tag' in x);
  assert(nums.length >= 15, nums.length);
  nums.forEach(([k, x]) => { assert(TAGS.has(x.tag), k + ' ' + x.tag); assert(x.src, k); });
  Object.entries(B.variants).forEach(([k, x]) => assert(x.row_ms === null || (TAGS.has(x.row_ms.tag) && x.row_ms.src), k));
  ['streams', 'contexts', 'target_tps', 'fit_rounds', 'fit_bisect_steps', 'miss_modes'].forEach(k => assert(B[k].src, k));
  assert.strictEqual(B.variants.fit.row_ms, null, 'the measured variant takes the fitted ms per row');
});

test('the config values match the HF config fixture (the build carries them so a public copy builds without tests/)', () => {
  const c = JSON.parse(fs.readFileSync('tests/fixtures/hf-qfn.json', 'utf8')).config.text_config;
  const lin = c.layer_types.filter(x => x === 'linear_attention').length, full = c.layer_types.length - lin;
  assert.strictEqual(v(B.attn_budget_tokens), c.indexer_budget);
  assert.strictEqual(v(B.index_bytes_per_token), full * c.indexer_head_dim / c.indexer_compress_ratio);
  assert(Math.abs(v(B.state_mib_per_row) - lin * c.linear_num_value_heads * c.linear_key_head_dim * c.linear_value_head_dim * 4 / 2 ** 20) < 0.05);
  assert.deepStrictEqual([B.L, B.E, B.k], [c.num_hidden_layers, c.num_experts, c.num_experts_per_tok]);
});

test('the ledger rows feed the fit: the r20k series and the split twin are read from runs.jsonl', () => {
  const r = B.rows.find(x => x.id === 'odie-fnx-iq3xxs-1xb70-batch-262k-r20k-xe0145-1022');
  assert.deepStrictEqual(r.tps, { 1: 61.25, 2: 72.12, 4: 82.2, 8: 75.34 });
  assert.strictEqual(r.kv_resident, 20480);
  assert.strictEqual(B.twin.tps.split, 94.6);
  assert(Math.abs(v(B.experts_gib) - 39.96) < 0.01);
});

test('both miss modes fit within 10 % median and 30 % worst, with non-negative constants', () => {
  B.miss_modes.v.forEach(m => {
    const f = FM.batchFit(data, m);
    assert(f.median_abs_pct < 10, m + ' median ' + f.median_abs_pct);
    assert(f.max_abs_pct < 30, m + ' max ' + f.max_abs_pct);
    assert(f.K.fixed >= 0 && f.K.row >= 0 && f.K.fixed + f.K.row > 0 && f.F.mpg >= 0 && f.F.tau > 0, JSON.stringify([f.K, f.F]));
    console.log('FIT', m, JSON.stringify({ K: f.K, F: f.F, median: f.median_abs_pct, max: f.max_abs_pct, twin: f.twin }));
  });
});

test('the split has no misses; one card misses more with every stream', () => {
  const one = table.find(t => t.config === 'one' && t.variant === 'fit' && t.ctx === 40960), split = table.find(t => t.config === 'split' && t.variant === 'fit' && t.ctx === 40960);
  split.steps.forEach(s => assert.strictEqual(s.miss, 0));
  one.steps.slice(1).forEach((s, i) => assert(s.miss > one.steps[i].miss));
});

test('the kernel variants order: bandwidth-only >= ~2 ms per row >= fitted, and bandwidth-only binds on bytes', () => {
  B.contexts.v.forEach(ctx => {
    const g = k => table.find(t => t.config === 'split' && t.variant === k && t.ctx === ctx).steps;
    g('ideal').forEach((s, i) => {
      assert(s.total >= g('row2')[i].total - 1e-9 && g('row2')[i].total >= g('fit')[i].total - 1e-9, ctx + ' N ' + s.N);
      assert(!/kernel/.test(s.binding), s.binding);
    });
  });
});

test('a variant only lowers the ms per row: never above the fitted one', () => {
  B.miss_modes.v.forEach(m => {
    const f = FM.batchFit(data, m), t = FM.batchTable(data, f);
    const g = k => t.find(x => x.config === 'split' && x.variant === k && x.ctx === 40960).steps;
    g('row2').forEach((s, i) => assert(s.total >= g('fit')[i].total - 1e-9, m + ' N ' + s.N));
  });
});

test('the per-row setting moves the prediction (no hidden constant)', () => {
  const d = clone(data);
  d.batch.variants.row2.row_ms.v = 4;
  assert(row(d, 'split', 'row2', 40960).steps[3].total < row(data, 'split', 'row2', 40960).steps[3].total);
  d.batch.card_bw_gbs.v = 2 * v(B.card_bw_gbs);
  assert(row(d, 'split', 'ideal', 40960).steps[5].total > row(data, 'split', 'ideal', 40960).steps[5].total);
});

test('VRAM: KV past the free VRAM streams from RAM; the int4 TP2 store does not fit two cards', () => {
  const s = table.find(t => t.config === 'split' && t.variant === 'fit' && t.ctx === 131072).steps;
  const last = s[s.length - 1];
  assert(last.stream && last.ram_gib > 0, 'N 32 at 128k streams its KV');
  assert(!s[0].stream && s[0].row_gib > 0.9);
  const tp = table.find(t => t.config === 'tp2_int4' && t.ctx === 16384).steps[0];
  assert(tp.weights_gib > 2 * v(B.card_vram_gib) - 2 * v(B.configs.tp2_int4.overhead_gib), tp.weights_gib);
  assert(tp.free_gib < 0 && !tp.fits);
});

test('first N at the target is the first stream count whose total reaches it', () => {
  table.forEach(t => {
    const i = t.steps.findIndex(s => s.total >= B.target_tps.v);
    assert.strictEqual(t.first_n_at_target, i < 0 ? null : t.steps[i].N);
  });
});

console.log(passed + ' batch tests passed');
