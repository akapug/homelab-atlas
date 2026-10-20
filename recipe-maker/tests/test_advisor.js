#!/usr/bin/env node
// The layer map (per-layer sensitivity, data/layer-maps.json), the route-trim proposer and the upgrade advisor
// (data/upgrades.json, src/advisor.js). Run: node tests/test_advisor.js (needs data.json: python3 build.py)
'use strict';
const assert = require('assert');
const fs = require('fs'), path = require('path');
const ROOT = path.join(__dirname, '..');
const FM = require(path.join(ROOT, 'src', 'model.js'));
const ADV = require(path.join(ROOT, 'src', 'advisor.js'));
const load = () => { const d = JSON.parse(fs.readFileSync(path.join(ROOT, 'data.json'), 'utf8')); FM.applyCal(d); return d; };
const data = load();
const runs = ADV.ledger(path.join(ROOT, 'runs', 'runs.jsonl'));
const Q = data.checks.quality;
const recipeOf = (d, id) => d.recipes.find(x => x.run_id === id).recipe;
const quality = (d, r) => FM.checkRecipe(d, r, null, { predict: false }).checks.find(c => c.id === 'quality');
const effOf = (d, mk, dm) => { const w = FM.layerWeights(d, mk); return dm.reduce((a, x, l) => a + x * w[l], 0) / dm.length; };
// 2-bit damage on planner layers whose model ids fall in [lo, hi] (every routed read of those layers at 2 bits)
const cut = (m, lo, hi) => Float64Array.from({ length: m.L }, (_, l) => { const id = (m.layer_ids || [])[l]; return id >= lo && id <= hi ? 1 : 0; });
let n = 0;
const t = (name, fn) => { fn(); n++; console.log('ok ' + name); };

t('layer map: GLM calibrates on the three measured gates; provenance per band; Maya bands CLAIMED with a url', () => {
  const lm = FM.layerMap(data, 'glm'), M = data.layer_maps.models.glm;
  assert.strictEqual(lm.status, 'calibrated', lm.why);
  assert.deepStrictEqual(lm.points.map(p => p.run), ['snoopy-glm-p3-1503', 'snoopy-glm-p4-s25-2047', 'snoopy-glm-p4-s15-2047']);
  assert.strictEqual(lm.dof, 1, 'three points, two unknowns (the CLAIMED scale and the fitted 8-22 band)');
  M.bands.filter(b => b.tag === 'CLAIMED').forEach(b => assert.ok(/^https:\/\/github\.com\/mw00\/project-maya\//.test(b.url), JSON.stringify(b)));
  M.bands.filter(b => b.fit).forEach(b => assert.strictEqual(b.tag, 'INFERRED-BY-US'));
  lm.points.forEach(p => assert.ok(Math.abs(p.pred_kl - p.kl) <= lm.residual_max.kl + 1e-12));
  // sensitivity falls with depth: early > fitted middle > 23-29 > late
  const w = l => lm.w[lm.layer_ids.indexOf(l)];
  assert.ok(w(5) > w(15) && w(15) > w(25) && w(25) > w(40), [5, 15, 25, 40].map(w).join(' '));
  assert.ok(lm.w.every(x => x > 0), 'no layer weighs zero');
});

t('layer map: V4.1 has no published map (UNKNOWN to Maya at abc09c7); our two points fit two bands, exactly determined', () => {
  const lm = FM.layerMap(data, 'v41'), M = data.layer_maps.models.v41;
  assert.ok(M.bands.every(b => b.fit && b.tag === 'INFERRED-BY-US') && !M.claims.length && !M.trim);
  assert.strictEqual(lm.status, 'calibrated', lm.why);
  assert.strictEqual(lm.dof, 0);
  const tp = ADV.trimProposals(data, runs).find(x => x.model === 'v41');
  assert.ok(!tp || tp.status === 'UNKNOWN', 'no V4.1 trim proposal without a trim unit');
});

t('layer map: a layer in no band is UNKNOWN, never zero (it weighs as the most sensitive layer; a trim on it has no KL)', () => {
  const d = load();
  d.layer_maps.models.glm.bands = d.layer_maps.models.glm.bands.filter(b => b.layers[0] !== 23);
  delete d._lm;
  const lm = FM.layerMap(d, 'glm'), m = d.models.glm;
  assert.strictEqual(lm.status, 'calibrated', lm.why);
  const unk = lm.layer_ids.map((id, l) => l).filter(l => lm.band[l] < 0);
  assert.deepStrictEqual(unk.map(l => lm.layer_ids[l]), [23, 24, 25, 26, 27, 28, 29]);
  assert.ok(unk.every(l => lm.prov[l] === 'UNKNOWN' && lm.s[l] === null && lm.c[l] === null));
  assert.ok(unk.every(l => lm.w[l] === Math.max.apply(null, lm.w)), 'unknown layers weigh as the most sensitive known layer');
  const keep = FM.trimKeeps(m, [{ layers: [25, 26], keep: 6 }]);
  const r = FM.trimKl(d, 'glm', keep, new Float64Array(m.L).fill(0.5));
  assert.strictEqual(r.kl, null);
  assert.ok(/25,26 have no map value \(UNKNOWN\)/.test(r.why), r.why);
  assert.deepStrictEqual(ADV.mapSummary(d).find(x => x.model === 'glm').unknown_layers, [23, 24, 25, 26, 27, 28, 29]);
});

t('P3-style middle-layer cut is a PREDICTED FAIL, in sample and with P3 left out of the calibration', () => {
  const r = recipeOf(data, 'snoopy-glm-p3-1503'), q = quality(data, r);
  assert.strictEqual(q.status, 'FAIL', q.detail);
  assert.ok(/PREDICTED \(layer map, 3 calibration points\)/.test(q.detail), q.detail);
  // leave P3 out: S25 and S15 are both coldest-expert layouts (the same shape over layers), so they cannot separate the CLAIMED
  // scale from the fitted 8-22 band; the map says so (no physical fit, UNKNOWN) and the check falls back to the plain share line,
  // which still fails the P3 layout
  const d = load();
  d.checks.quality.families.glm.points = d.checks.quality.families.glm.points.filter(p => p.run !== 'snoopy-glm-p3-1503');
  delete d._lm;
  const lm = FM.layerMap(d, 'glm');
  assert.strictEqual(lm.status, 'UNKNOWN');
  assert.ok(/no physical fit/.test(lm.why), lm.why);
  assert.strictEqual(FM.layerWeights(d, 'glm'), null, 'an uncalibrated map weighs every layer the same');
  const q2 = quality(d, recipeOf(d, 'snoopy-glm-p3-1503'));
  assert.strictEqual(q2.status, 'FAIL', q2.detail);
  assert.ok(!/layer map/.test(q2.detail), q2.detail);
});

t('the same share of 2-bit layers costs more in the middle than late: 15 middle layers FAIL, the last 15 PASS', () => {
  const m = data.models.glm, mid = effOf(data, 'glm', cut(m, 8, 22)), late = effOf(data, 'glm', cut(m, 30, 44));
  assert.ok(mid > late * 10, mid + ' vs ' + late);
  const a = FM.mapQuality(data, 'glm', mid, 0), b = FM.mapQuality(data, 'glm', late, 0);
  assert.ok(a.kl > Q.bar_kl, 'middle 8-22 at 2 bits: PREDICTED KL ' + a.kl);
  assert.ok(b.kl < Q.bar_kl && b.top1 > Q.bar_top1, 'late 30-44 at 2 bits: PREDICTED KL ' + b.kl + ' top-1 ' + b.top1);
  // the plain share line cannot tell them apart (same 2-bit share)
  assert.strictEqual(cut(m, 8, 22).reduce((x, y) => x + y, 0), cut(m, 30, 44).reduce((x, y) => x + y, 0));
});

t('route-trim proposer: the map proposes a late-layer trim on snoopy GLM under the KL bar, engine supported', () => {
  const tp = ADV.trimProposals(data, runs).find(x => x.box === 'snoopy' && x.model === 'glm');
  assert.ok(tp && tp.status === 'proposed', JSON.stringify(tp && { status: tp.status, why: tp.why }));
  assert.ok(tp.trim.length && tp.trim.every(b => b.layers[0] >= 23), 'late layers only: ' + JSON.stringify(tp.trim));
  assert.ok(tp.trim.every(b => b.keep >= FM.v(data.layer_maps.settings.trim_min_keep) && b.keep < data.models.glm.k));
  const x = tp.predicted;
  assert.ok(x.gain > 0.05 && x.decode_planner > x.decode_planner_before, 'trim speeds decode: ' + x.gain);
  assert.ok(x.kl <= tp.budget.kl && tp.budget.kl < Q.bar_kl, 'KL under the bar by the map margin: ' + x.kl);
  assert.ok(x.top1 >= tp.budget.top1 && tp.budget.top1 > Q.bar_top1);
  assert.ok(/^PREDICTED/.test(x.label));
  assert.strictEqual(tp.quality_check.status, 'PASS', tp.quality_check.detail);
  assert.strictEqual(tp.engine_support.status, 'yes', 'glmlite expert-trim e1160e7, measured on snoopy');
  // the base run already trims (run B): the scaled figure divides by that run's own prediction, never the untrimmed one
  const own = FM.predictRecipe(data, tp.base_recipe || recipeOf(data, tp.base_run));
  assert.ok(Math.abs(x.decode_scaled - tp.base_measured * x.decode_planner / own.decode_tps) < 1e-9, 'scaled ' + x.decode_scaled);
  assert.ok(x.decode_scaled < tp.base_measured * 1.1, 'a mild re-trim of a trimmed run cannot scale like a fresh trim: ' + x.decode_scaled);
  // a trim on the sensitive early band blows the budget
  const early = Object.assign({}, tp.recipe, { trim: [{ layers: [3, 7], keep: 4 }] });
  assert.ok(quality(data, early).status !== 'PASS', quality(data, early).detail);
});

const advice = ADV.upgradeAdvice(data, runs);
t('trim quality: transfer and the top-1 rate are MEASURED on runs A, B and C and predict each within 0.01 top-1', () => {
  const T = data.layer_maps.models.glm.trim, mk = 'glm', m = data.models.glm;
  assert.strictEqual(T.transfer.tag, 'MEASURED'); assert.strictEqual(T.top1_per_kl.tag, 'MEASURED');
  const pred = id => {
    const r = recipeOf(data, id), miss = FM.predictRecipe(data, Object.assign({}, r, { trim: [] })).trim.miss_by_layer;
    return FM.mapQuality(data, mk, 0, FM.trimKl(data, mk, FM.trimKeeps(m, r.trim), miss).kl);
  };
  const a = pred('snoopy-glm-glm-coder-s1w-trim-a'), b = pred('snoopy-glm-glm-coder-s1w-trim-b');
  assert.ok(a.top1 < 0.981 && Math.abs(a.top1 - 0.96875) < 0.01, 'A (measured 0.96875; 0.981 before the rate): ' + a.top1);
  assert.ok(Math.abs(b.top1 - 0.9838) < 0.01, 'B (measured 0.9838): ' + b.top1);
  const c = pred('snoopy-glm-glm-coder-s1w-trim-c');
  assert.ok(Math.abs(c.top1 - 0.9752) < 0.01, 'C (measured 0.9752): ' + c.top1);
  assert.ok(a.kl > b.kl, 'the deeper trim costs more KL');
});

t('upgrade advisor: the droopy PMem buy (+15-18 %) does NOT trigger; every box reads its best quality-passing run', () => {
  const v41 = advice.boxes.droopy.models.find(m => m.model === 'v41');
  assert.ok(v41 && v41.quality.status === 'PASS', JSON.stringify(v41 && v41.quality));
  const pm = v41.parts.find(p => p.id === 'droopy-pmem100-4x128');
  assert.ok(pm.gain[0] > 0.12 && pm.gain[1] < 0.2, 'PMem gain ' + pm.gain);
  assert.strictEqual(pm.suggested, false);
  assert.ok(pm.why_not.some(w => /< suggest_min_gain 20 %/.test(w)), pm.why_not.join('; '));
  assert.ok(!advice.suggestions.some(s => s.id === 'droopy-pmem100-4x128'));
  assert.strictEqual(pm.price_date, '2026-10-17');
  assert.deepStrictEqual(pm.price_usd, [276, 516], '4 x $69-129');
  const dlt = pm.tok_s.scaled_after.map(x => x - pm.tok_s.measured_before);
  assert.ok(Math.abs(pm.usd_per_tok_s[0] - 276 / dlt[1]) < 1e-9 && Math.abs(pm.usd_per_tok_s[1] - 516 / dlt[0]) < 1e-9);
  // quality FAIL runs never anchor advice
  Object.values(advice.boxes).forEach(b => b.models.forEach(m => assert.strictEqual(m.quality.status, 'PASS')));
});

t('upgrade advisor: a suggestion needs the binding lane and the setting threshold (no hardcode)', () => {
  const v41 = advice.boxes.droopy.models.find(m => m.model === 'v41'), dd = v41.parts.find(p => p.id === 'droopy-ddr4-s1-6x16');
  assert.strictEqual(v41.binding, 'dram_cap');
  assert.ok(dd.suggested && dd.gain[0] >= data.upgrade.settings.suggest_min_gain, JSON.stringify(dd.why_not));
  assert.ok(dd.caveat && /NUMA-layout-blind/.test(dd.caveat));
  const d = load();
  d.upgrade.settings.suggest_min_gain = 0.3;
  const A = ADV.upgradeAdvice(d, runs);
  assert.ok(!A.suggestions.length, 'a 30 % threshold silences the +25 % DDR4 part');
});

t('upgrade advisor: a synthetic PMem-bound recipe triggers the PMem part', () => {
  const run = JSON.parse(JSON.stringify(runs.find(r => r.id === 'droopy-v41-v41r1b-1354')));
  run.id = 'synthetic-pmem-bound';
  run.recipe.ram_tier_gib = 40;   // most cold experts forced onto PMem
  const A = ADV.upgradeAdvice(data, [run]), m = A.boxes.droopy.models[0];
  assert.strictEqual(m.binding, 'pmem_bw', m.lanes.map(l => l.lane + ' ' + l.gain.toFixed(3)).join(', '));
  const pm = m.parts.find(p => p.id === 'droopy-pmem100-4x128');
  assert.ok(pm.suggested && pm.gain[0] >= 0.2, JSON.stringify(pm.why_not) + ' ' + pm.gain);
  assert.ok(A.suggestions.some(s => s.id === 'droopy-pmem100-4x128' && s.box === 'droopy'));
  assert.ok(pm.tok_s.scaled_after[0] > pm.tok_s.measured_before && pm.usd_per_tok_s[0] > 0);
});

t('upgrade advisor: snoopy GLM binds on DRAM read ceiling + CPU together (co-binding); no part prices it', () => {
  const g = advice.boxes.snoopy.models.find(m => m.model === 'glm'), gain = l => (g.lanes.find(x => x.lane === l) || { gain: 0 }).gain;
  assert.strictEqual(g.binding, 'dram_bw+cpu');
  assert.ok(gain('dram_bw') < 0.01 && gain('cpu') < 0.01 && gain('dram_bw+cpu') > 0.3, g.lanes.map(l => l.lane + ' ' + l.gain.toFixed(3)).join(', '));
  assert.ok(advice.boxes.snoopy.unpriced.some(u => u.lane === 'dram_bw+cpu'));
  // droopy's PMem and CPU lanes only compound: no co-binding pair
  assert.ok(!advice.boxes.droopy.models.find(m => m.model === 'v41').lanes.some(l => l.lane === 'pmem_bw+cpu'));
});

t('lane probes move only their lane: ms_per_lane splits the token time before the GPU/CPU overlap', () => {
  const r = recipeOf(data, 'snoopy-glm-glm-coder-s1w'), p = FM.predictRecipe(data, r), L = p.ms_per_lane, T = p.ms_per_token;
  const cpuSide = L.dram_bw + L.pmem_bw + L.nvme_bw + L.cpu;
  assert.ok(Math.abs(cpuSide - T.cpu) < 1e-6 && Math.abs(L.vram_bw - T.gpu) < 1e-9 && L.host === T.host, JSON.stringify(L));
  const nv = FM.predictRecipe(data, Object.assign({}, r, { knobs: { lane: { scale: { nvme_bw: 4 } } } }));
  assert.ok(Math.abs(nv.decode_tps - p.decode_tps) < 1e-9, 'no NVMe tier on this recipe');
});

console.log(n + ' tests passed');
