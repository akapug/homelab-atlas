'use strict';
const assert = require('assert'), fs = require('fs');
const FM = require('../src/model.js');
const load = p => JSON.parse(fs.readFileSync(p, 'utf8'));
const clone = x => JSON.parse(JSON.stringify(x));
const data = load('data.json');
FM.applyCal(data);
const files = {
  fs1c: 'droopy-v41-fs1c.json', fs1p: 'droopy-v41-fs1p.json',
  nohost: 'droopy-v41-lo-nohost.json', follow: 'droopy-v41-lo-follow.json',
  plain: 'droopy-v41-lo-ba.json', draftgpu: 'droopy-v41-lo-draftgpu.json', alloff: 'droopy-v41-alloff.json'
};
const recipes = Object.fromEntries(Object.entries(files).map(([k, p]) => [k, load('recipes/' + p)]));
const predict = r => FM.predictRecipe(data, r);
const predictions = Object.fromEntries(Object.entries(recipes).map(([k, r]) => [k, predict(r)]));
let passed = 0;
const test = (name, fn) => { fn(); passed++; console.log('ok ' + name); };
const near = (a, b) => assert(Math.abs(a - b) < 1e-9, a + ' != ' + b);
const predicted = p => { assert.strictEqual(p.status, 'PREDICTED', p.why); assert(Number.isFinite(p.decode_tps)); };
const unknown = r => assert.doesNotThrow(() => { const p = predict(r); assert.strictEqual(p.status, 'UNMODELLED'); assert.strictEqual(p.decode_tps, null); assert(p.why); });
const change = fn => { const r = clone(recipes.fs1c); fn(r); return r; };

test('fs1p is held out and both decode widths predict within ten percent', () => {
  const p = predictions.fs1p;
  predicted(p);
  assert(Math.abs(p.decode_tps / 12.13 - 1) < .10);
  assert(Math.abs(p.decode_tps_nodraft / 10 - 1) < .10);
  console.log('HOLDOUT', JSON.stringify({n4:p.decode_tps, n0:p.decode_tps_nodraft}));
});

test('all six training arms remain within ten percent without arm speed constants', () => {
  const rates = {fs1c:[11.88,9.72], nohost:[10.99,9.04], follow:[11.92,9.78], plain:[11.31,9.31], draftgpu:[11.65,9.40], alloff:[10.70,8.80]};
  Object.entries(rates).forEach(([k, [n4,n0]]) => {
    const p = predictions[k]; predicted(p);
    assert(Math.abs(p.decode_tps / n4 - 1) < .10, k);
    assert(Math.abs(p.decode_tps_nodraft / n0 - 1) < .10, k);
    console.log('ARM', k, p.decode_tps, p.decode_tps_nodraft);
  });
});

test('observed lever signs and sizes survive: no-host, bank-aware, combined draft/banks, flat follow', () => {
  const gain = k => predictions.fs1c.decode_tps / predictions[k].decode_tps - 1;
  assert(gain('nohost') > .06 && gain('nohost') < .10);
  assert(gain('plain') > .035 && gain('plain') < .07);
  assert(gain('draftgpu') > .01 && gain('draftgpu') < .04);
  near(gain('follow'), 0);
  assert(predictions.fs1c.decode_tps > predictions.alloff.decode_tps);
});

test('real runtime lever names are operands, not prose or recipe identifiers', () => {
  const nohost = change(r => { r.conditions.levers.LLAMA_NO_HOST_WEIGHTS = 0; });
  near(predict(nohost).decode_tps, predictions.nohost.decode_tps);
  const follow = change(r => { r.conditions.levers.GGML_SCHED_INPLACE_FOLLOW = 0; });
  near(predict(follow).decode_tps, predictions.fs1c.decode_tps);
  const renamed = change(r => { r.id = 'unrelated'; r.label = 'plain'; r.numa.src = 'not a selector'; r.conditions.package = 'not a build coefficient'; });
  near(predict(renamed).decode_tps, predictions.fs1c.decode_tps);
});

test('routing fractions are ALL reads, conserve one, and affect timing without moving capacity or quality', () => {
  const r = change(r => { delete r.decode_profile; });
  const measured = predict(r), legacy = predict(Object.assign({}, r, {decode_reads:undefined}));
  predicted(measured); predicted(legacy);
  near(measured.tiers.reduce((s, x) => s + x.read_share, 0), 1);
  near(measured.tiers.find(x => x.id === 'pmem').read_share, .070);
  assert(measured.ms_per_token.cpu < legacy.ms_per_token.cpu);
  assert.deepStrictEqual(measured.tiers.map(x => x.experts_gib), legacy.tiers.map(x => x.experts_gib));
  near(measured.lowbit_share, legacy.lowbit_share);
  near(measured.prefill_tps, legacy.prefill_tps);
  near(measured.vram_post_gib, legacy.vram_post_gib);
  const lanes = measured.ms_per_lane;
  near(lanes.dram_bw + lanes.pmem_bw + lanes.nvme_bw + lanes.cpu, measured.ms_per_token.cpu);
});

test('missing, malformed, unsupported and unrun operands fail closed', () => {
  [r=>delete r.decode_reads, r=>r.decode_reads=null, r=>r.decode_reads.shares.pmem=-.1,
   r=>r.decode_reads.shares.pmem='0.070', r=>r.decode_reads.shares.pmem=NaN,
   r=>delete r.decode_reads.shares.dram, r=>r.decode_reads.shares.unknown=0,
   r=>r.decode_reads.shares.pmem=.070*.8, r=>r.decode_reads.src='',
   r=>r.banks_gib=7.6, r=>r.banks_gib=2.2, r=>r.threads=20, r=>r.ub=512,
   r=>r.box='lassie', r=>r.model='glm', r=>r.store.cold_type='q3_K',
   r=>r.conditions.workload.e1.gen_tokens=1000, r=>r.conditions.sampling.temperature=1, r=>r.conditions.draft_node='0',
   r=>r.files[0].access='mmap', r=>r.numa.nodes=[{experts_gib:1}], r=>r.numa.pinned_host=true,
   r=>r.placement={mode:'coldest',pmem_gib:10}, r=>r.engine_build='unmeasured', r=>r.expert_chunking=true,
   r=>r.conditions.cpu_reserve='none', r=>r.draft='true', r=>r.prefill_tokens=100,
   r=>r.conditions.levers.GGML_SCHED_INPLACE_FOLLOW=2, r=>r.conditions.levers.LLAMA_NO_HOST_WEIGHTS='1',
   r=>r.knobs={draftMult:2}, r=>r.decode_profile='missing'].forEach(fn=>unknown(change(fn)));
});

test('profile names must be strings and own keys in both tables', () => {
  const inheritedSpecs = Object.assign({}, data, {decode_fits:Object.create(data.decode_fits)});
  const inheritedFits = Object.assign({}, data, {cal:Object.assign({}, data.cal, {recipe_decode:Object.create(data.cal.recipe_decode)})});
  [inheritedSpecs, inheritedFits].forEach(d => {
    const p = FM.predictRecipe(d, recipes.fs1c);
    assert.strictEqual(p.status, 'UNMODELLED', 'inherited profile entries must not authorize a fit');
    assert.strictEqual(p.decode_tps, null);
    assert.strictEqual(p.decode_tps_nodraft, null);
  });
  ['constructor', 'toString', '__proto__', null, 1, [], [recipes.fs1c.decode_profile],
   {toString:()=>recipes.fs1c.decode_profile}].forEach(key => unknown(change(r=>{r.decode_profile=key;})));
});

test('a missing measured knob envelope never calibrates', () => {
  const d = load('data.json');
  delete d.decode_fits[recipes.fs1c.decode_profile].knobs;
  FM.applyCal(d);
  assert.strictEqual(FM.predictRecipe(d, recipes.fs1c).status, 'UNMODELLED');
  assert(!Object.prototype.hasOwnProperty.call(d.cal.recipe_decode, recipes.fs1c.decode_profile));
});

test('profile and evidence replace the confounded old host plus split, never double-count them', () => {
  const p = predictions.fs1c;
  assert(p.decode_fit);
  assert(p.ms_per_token.host > 0 && p.ms_per_token.host < 30);
  const decode = p.evidence.find(x => x.metric === 'decode') || p.evidence[0];
  const text = JSON.stringify(decode);
  assert(text.includes('scoped decode residual'));
  assert(text.includes('within-tier expert type mix'));
  assert(!text.includes('split GGUF host ms/token'));
  assert(data.decode_fits[recipes.fs1c.decode_profile].training.every(x => !x.id.includes('fs1p')));
});

test('holdout never enters either fit and mutations of its retained measurements cannot change calibration', () => {
  const spec = data.decode_fits[recipes.fs1p.decode_profile];
  assert.strictEqual(spec.training.length, 6);
  assert(spec.training.every(x => !spec.holdout.includes(x.id)));
  assert(Object.values(data.draft_twins).flat().every(x => !spec.holdout.includes(x.id)));
  // Reload the wire input: JSON-cloning warmed data loses its Float64Array heat caches.
  const other = load('data.json');
  other.recipes.filter(x => spec.holdout.includes(x.run_id)).forEach(x => { x.outcome = 'absurd'; x.observed = {decode_tps:1000}; });
  FM.applyCal(other);
  near(FM.predictRecipe(other, recipes.fs1p).decode_tps, predictions.fs1p.decode_tps);
  FM.applyCal(other);
  near(FM.predictRecipe(other, recipes.fs1p).decode_tps, predictions.fs1p.decode_tps);
});

test('current page replay and standalone recipes share operands without rewriting saved predictions', () => {
  Object.entries(recipes).forEach(([k,r]) => {
    const row = data.recipes.find(x=>x.recipe.id===r.id);
    assert(row, k);
    near(predict(row.recipe).decode_tps, predictions[k].decode_tps);
    assert.deepStrictEqual(row.recipe.decode_reads, Object.assign({}, r.decode_reads, {evidence:'inferred'}));
  });
});

test('legacy recipes and non-decode quantities keep their path; unsupported composed modes stay unknown', () => {
  const r = change(r => { delete r.decode_profile; delete r.decode_reads; });
  predicted(predict(r));
  const noDraft = predict(change(r=>{r.draft=false;}));
  predicted(noDraft); near(noDraft.decode_tps, noDraft.decode_tps_nodraft);
  unknown(change(r=>{delete r.decode_profile; r.trim=[{layers:[0,1],keep:1}];}));
  const modified = load('data.json'); modified.knobs.kernel2x = true;
  FM.applyCal(modified);
  assert.strictEqual(FM.predictRecipe(modified,recipes.fs1c).status,'UNMODELLED');
  const overrides = load('data.json'); overrides.overrides = {kappa:{},kld:{}};
  FM.applyCal(overrides);
  near(FM.predictRecipe(overrides,recipes.fs1c).decode_tps, predictions.fs1c.decode_tps);
  overrides.overrides.kappa.mxfp4 = .1;
  FM.applyCal(overrides);
  assert.strictEqual(FM.predictRecipe(overrides,recipes.fs1c).status,'UNMODELLED');
  const serialized = clone(data);
  FM.applyCal(serialized);
  near(FM.predictRecipe(serialized, recipes.fs1c).decode_tps, predictions.fs1c.decode_tps);
});

console.log(passed + ' tests passed');
