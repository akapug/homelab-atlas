'use strict';
// the kv_host check (src/model.js servingOf / kvHostPlan / kvHostCheck; engines.strata.serving, models.fnx.kv_host and
// checks.kv_host from build.py): run after python3 build.py
const assert = require('assert'), fs = require('fs');
const FM = require('../src/model.js');
const data = JSON.parse(fs.readFileSync('data.json', 'utf8'));
const v = FM.v, GiB = FM.GiB;
const recipe = f => JSON.parse(fs.readFileSync('recipes/' + f, 'utf8'));
const MIXED = recipe('odie-fnx-xestrata-slotctx-b32.json'), UNIFORM = recipe('odie-fnx-xestrata-uniform-b32-262k.json');
const check = (r, id) => FM.checkRecipe(data, r, null, { predict: false }).checks.find(c => c.id === (id || 'kv_host'));
const withServing = (r, s) => Object.assign({}, r, { serving: Object.assign({}, r.serving, s) });
let passed = 0;
const test = (name, fn) => { fn(); passed++; console.log('ok ' + name); };

test('bytes per cell come from the HF config fixture through the engine block layout', () => {
  const c = JSON.parse(fs.readFileSync('tests/fixtures/hf-qfn.json', 'utf8')).config.text_config, K = data.models.fnx.kv_host;
  const full = c.layer_types.filter(x => x !== 'linear_attention').length, half = c.num_key_value_heads * c.head_dim;
  assert.strictEqual(v(K.layers), full);
  assert.strictEqual(v(K.draft_layers), c.mtp_num_hidden_layers);
  assert.strictEqual(v(K.bytes_per_cell_layer.fp16), 2 * half * 2);
  assert(v(K.bytes_per_cell_layer.int8) > half * 2 && v(K.bytes_per_cell_layer.int8) < 2 * half * 2, 'INT8 codes + scales sit between one and two bytes per value');
  Object.values(K.bytes_per_cell_layer).forEach(x => assert(x.tag && x.src));
  assert.strictEqual(data.models.fnxc.kv_host.layers.v, v(K.layers), 'the Coder variant shares the attention layers');
});

test('the pinned copy follows bytes_per_cell x (max_context x (mains + K) + slot_context x (batch - K)) + an overhead a copy, drafters apart', () => {
  [MIXED, UNIFORM].forEach(r => {
    const p = FM.kvHostPlan(data, Object.assign({}, r, { draft: false })), s = p.sv.s, K = s.slot_context ? s.big_slots : s.batch;
    const cells = s.max_context * (p.mains + K) + (s.slot_context || s.max_context) * (s.batch - K);
    assert.strictEqual(p.copies, (p.mains + s.batch) * p.layers);
    assert(Math.abs(p.main_gib - (cells * p.bpc + p.copies / p.layers * p.overhead_mib * 2 ** 20) * p.layers / GiB) < 1e-9, r.id);
    assert.strictEqual(p.draft_gib, 0, 'no drafter without a draft');
  });
});

test('32 x 262,144 uniform FAILs on odie; the mixed slot contexts PASS', () => {
  const u = check(UNIFORM), m = check(MIXED);
  assert.strictEqual(u.status, 'FAIL', u.detail);
  assert.strictEqual(m.status, 'PASS', m.detail);
  assert(u.pred.pinned_gib > m.pred.pinned_gib * 1.8, u.pred.pinned_gib + ' vs ' + m.pred.pinned_gib);
  assert(/refuses to start/.test(u.detail));
  assert.strictEqual(check(MIXED, 'inputs').status, 'PASS', check(MIXED, 'inputs').detail);
});

test('the reserve and MemAvailable are settings: the recipe overrides both', () => {
  const m = check(MIXED), left = m.pred && parseFloat(/leaves ([\d.]+) GiB/.exec(m.detail)[1]);
  assert.strictEqual(check(withServing(MIXED, { kv_host_reserve_gib: Math.ceil(left) + 1 })).status, 'FAIL');
  assert.strictEqual(check(withServing(MIXED, { mem_available_gib: Math.floor(m.pred.pinned_gib) })).status, 'FAIL');
  assert.strictEqual(check(withServing(UNIFORM, { mem_available_gib: Math.ceil(check(UNIFORM).pred.pinned_gib) + v(data.checks.kv_host.reserve_gib) })).status, 'PASS');
  const d2 = JSON.parse(JSON.stringify(data));
  d2.checks.kv_host.reserve_gib.v = 0;
  d2.boxes.odie.dram.host_floor_gib.v = v(d2.boxes.odie.dram.total_gib) - check(UNIFORM).pred.pinned_gib - 1;
  assert.strictEqual(FM.checkRecipe(d2, UNIFORM, null, { predict: false }).checks.find(c => c.id === 'kv_host').status, 'PASS', 'the data reserve and box floor drive the verdict');
});

test('no streaming pins nothing; a session inside the VRAM ring is not streamed', () => {
  assert.strictEqual(check(withServing(UNIFORM, { kv_resident: 0 })).status, 'PASS');
  const small = withServing(UNIFORM, { max_context: 16384 });
  small.ctx = 16384;
  const p = FM.kvHostPlan(data, small);
  assert.strictEqual(p.streamed, 0, 'the 20,480-cell minimum ring holds 16,384 cells');
  assert.strictEqual(check(small).status, 'PASS');
});

test('the engine defaults fill what a recipe leaves out; batch_mtp follows the draft', () => {
  const D = data.engines.strata.serving.defaults, bare = Object.assign({}, MIXED, { serving: { kv_resident: 4096 } });
  const s = FM.servingOf(data, bare).s;
  assert.strictEqual(s.max_context, MIXED.ctx, 'max_context falls back to ctx');
  ['batch', 'batch_groups', 'slot_context', 'big_slots', 'kv', 'mtp_window'].forEach(k => assert.strictEqual(s[k], v(D[k]), k));
  assert.strictEqual(s.batch_mtp, !!MIXED.draft);
  assert.strictEqual(FM.servingOf(data, Object.assign({}, MIXED, { draft: false })).s.batch_mtp, false);
  const on = FM.kvHostPlan(data, MIXED), off = FM.kvHostPlan(data, withServing(MIXED, { batch_mtp: false }));
  assert(on.draft_gib > off.draft_gib && off.draft_gib > 0, 'the main session drafts either way');
});

test('a bad serving field is a FAIL in inputs; serving on an engine without a serving model is UNKNOWN', () => {
  ['slot_context', 'big_slots', 'batch'].forEach(k => assert.strictEqual(check(withServing(MIXED, { [k]: -1 }), 'inputs').status, 'FAIL', k));
  assert.strictEqual(check(withServing(MIXED, { slots: 4 }), 'inputs').status, 'FAIL', 'an unknown field');
  assert.strictEqual(check(withServing(MIXED, { max_context: 131072 }), 'inputs').status, 'FAIL', 'max_context disagrees with ctx');
  assert.strictEqual(check(withServing(MIXED, { kv: 'fp8' })).status, 'UNKNOWN');
  const ik = Object.assign({}, recipe('droopy-glm-be1-q3.json'), { serving: { batch: 2 } });
  assert.strictEqual(check(ik, 'kv_host'), undefined, 'no kv_host check without a serving model');
  assert(/recipe.serving given/.test(check(ik, 'inputs').detail), check(ik, 'inputs').detail);
});

// ---------- host_pinned: every host-pinned buffer against DRAM total - household floor - pinned margin ----------
const LIVE = recipe('odie-fnx-xestrata-live-b32-131k.json');
const hp = r => FM.hostPinnedCheck(data, r), HP = data.checks.host_pinned;

test('the live qfn2 config reproduces the measured 397 copies, 51.95 GiB of KV and ~84.6 GiB pinned, and FAILs on odie', () => {
  const p = FM.hostPinnedPlan(data, LIVE), c = hp(LIVE);
  assert.strictEqual(p.kv.copies, 397);
  assert(Math.abs(p.terms.find(x => x.id === 'kv_copies').gib - 51.95) < 0.01, p.terms[0].gib);
  assert(Math.abs(p.pinned_gib - v(HP.total_measured_gib)) / v(HP.total_measured_gib) < 0.01, p.pinned_gib);
  assert.strictEqual(c.status, 'FAIL', c.detail);
  assert(/stage_lend_drop/.test(c.detail) && /other host use/.test(c.detail));
  assert(FM.checkRecipe(data, LIVE, null, { predict: false }).checks.some(x => x.id === 'host_pinned' && x.status === 'FAIL'), 'the gate carries it');
  Object.values(HP.terms['fnx/strata']).forEach(x => assert(x.tag && x.src));
});

test('the levers: --slot-context at the VRAM ring saves ~48.7 GiB, at 65,536 ~24, STRATA_STAGE_LEND_DROP the lend copy; each PASSes', () => {
  const base = FM.hostPinnedPlan(data, LIVE).pinned_gib, at = s => FM.hostPinnedPlan(data, withServing(LIVE, s)).pinned_gib;
  const ring = v(data.engines.strata.serving.kv_resident_min);
  assert(Math.abs(base - at({ slot_context: ring, big_slots: 1 }) - 48.7) < 0.5, base - at({ slot_context: ring, big_slots: 1 }));
  assert(Math.abs(base - at({ slot_context: 65536, big_slots: 1 }) - 24) < 1.5, base - at({ slot_context: 65536, big_slots: 1 }));
  assert(Math.abs(base - at({ stage_lend_drop: true }) - v(HP.terms['fnx/strata'].stage_lend_gib)) < 1e-9);
  [{ slot_context: ring, big_slots: 1 }, { slot_context: 65536 }, { stage_lend_drop: true }].forEach(s => assert.strictEqual(hp(withServing(LIVE, s)).status, 'PASS', JSON.stringify(s)));
});

test('the reserve is a setting: the box floor and margin, or the recipe\'s own; a one-card recipe has no lend copy', () => {
  const d2 = JSON.parse(JSON.stringify(data));
  d2.boxes.odie.dram.pinned_margin_gib.v = 0; d2.boxes.odie.dram.host_floor_gib.v = 0;
  assert.strictEqual(FM.hostPinnedCheck(d2, LIVE).status, 'PASS', 'the box settings drive the verdict');
  assert.strictEqual(hp(withServing(LIVE, { host_reserve_gib: 0 })).status, 'PASS');
  assert.strictEqual(hp(withServing(LIVE, { stage_lend_drop: true, host_reserve_gib: 64 })).status, 'FAIL');
  const one = Object.assign({}, LIVE, { gpu_split: undefined });
  assert.strictEqual(FM.hostPinnedPlan(data, one).terms.find(x => x.id === 'stage_lend').gib, 0);
  assert.strictEqual(FM.hostPinnedPlan(data, withServing(LIVE, { host_experts_gib: 0 })).terms.find(x => x.id === 'experts_offgpu').gib, 0);
  const d3 = JSON.parse(JSON.stringify(data)); delete d3.checks.host_pinned.terms['fnx/strata'];
  assert.strictEqual(FM.hostPinnedCheck(d3, LIVE).status, 'UNKNOWN');
});

console.log(passed + ' kv_host tests passed');
