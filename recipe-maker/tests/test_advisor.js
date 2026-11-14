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

t('runQuality KL borrowing: PASS borrows only on matching commit/build/patches/trim/kv_type, never across arithmetic changes', () => {
  const baseRun = {
    id: 'ref-kl-pass', when: '2026-10-20T12:00Z', box: 'snoopy', model: 'glm', engine: 'glmlite',
    commit: 'f20ef3f', patches: ['p512-exact'],
    recipe: {
      box: 'snoopy', model: 'glm', engine: 'glmlite', engine_build: 'p512-lane', rtr: false, trim: [],
      store: { split: false, types: ['exl3_3.05'] },
      conditions: { kv_type: 'f16' },
    },
    measured: { outcome: 'ok', kl_verdict: 'PASS', kl: 0.0045, top1: 0.9914 },
  };
  const cloneCand = mut => {
    const c = JSON.parse(JSON.stringify(baseRun));
    c.id = 'cand-run';
    c.when = '2026-10-21T12:00Z';
    c.measured = { outcome: 'ok', decode_tps: { single: 25.0 } };
    mut(c);
    return c;
  };
  // Identical arithmetic metadata (even with reordered store keys) -> borrows PASS from ref-kl-pass
  const exact = cloneCand(c => { c.recipe.store = { types: ['exl3_3.05'], split: false }; });
  const qExact = ADV.runQuality(data, exact, [baseRun, exact]);
  assert.strictEqual(qExact.status, 'PASS');
  assert.ok(/same store and build \(ref-kl-pass\)/.test(qExact.basis), qExact.basis);

  // Changing commit, engine_build, patches, trim, -rtr, kv_type, or package must NOT borrow ref-kl-pass
  const mutations = [
    ['commit', c => { c.commit = 'deadbeef'; }],
    ['engine_commit', c => { delete c.commit; c.recipe.engine_commit = 'cafebabe'; }],
    ['engine_build', c => { c.recipe.engine_build = 'avx2-lane'; }],
    ['patches', c => { c.patches = ['p512-exact', 'unverified-fast-math']; }],
    ['trim', c => { c.recipe.trim = [{ layers: [29, 44], keep: 4 }]; }],
    ['rtr', c => { c.recipe.rtr = true; }],
    ['kv_type', c => { c.recipe.conditions.kv_type = 'q4_0'; }],
    ['package', c => { delete c.recipe.engine_build; c.recipe.conditions.package = 'optane-bin/custom-fork'; }],
  ];
  for (const [label, fn] of mutations) {
    const cand = cloneCand(fn);
    assert.notStrictEqual(ADV.klEquivKey(cand), ADV.klEquivKey(baseRun), 'equiv key must differ on ' + label);
    const q = ADV.runQuality(data, cand, [baseRun, cand]);
    assert.ok(!/ref-kl-pass/.test(q.basis), label + ' silently borrowed PASS: ' + q.basis);
  }

  // Real ledger regression check: untrimmed avx2-lane snoopy GLM run must not borrow trimmed p512-lane trim-c gate
  const arma = runs.find(r => r.id === 'snoopy-glm-arma-0719');
  if (arma) {
    const qa = ADV.runQuality(data, arma, runs);
    assert.ok(!/trim-c/.test(qa.basis), 'untrimmed avx2-lane borrowed trimmed p512-lane gate: ' + qa.basis);
  }
});

t('EARf behavioral fidelity gate: >= 0.98 default rejects GSQ IQ3_XXS (0.894) and cross-engine Q8_0 (0.975), warns on unmeasured sub-4-bit', () => {
  const fnxRecipe = storeTypes => ({
    box: 'odie', model: 'fnx', engine: 'strata', rtr: false, draft: false, files: [],
    store: { split: false, types: storeTypes },
  });
  const earfChk = (r, opts) => FM.checkRecipe(data, r, null, Object.assign({ predict: false }, opts || {})).checks.find(c => c.id === 'earf');

  // 1. GSQ IQ3_XXS collapses EARf to 0.894 -> hard FAIL
  const iq3 = earfChk(fnxRecipe(['gsq_iq3xxs']));
  assert.strictEqual(iq3.status, 'FAIL', iq3.detail);
  assert.strictEqual(iq3.earf, 0.894);
  assert.strictEqual(iq3.min_earf, 0.98);
  assert.ok(/0\.894 < 0\.98/.test(iq3.detail), iq3.detail);

  // 2. Unverified cross-engine Q8_0 port at 0.975 and UD-Q4_K_XL at 0.955 both FAIL the 0.98 bar
  const q8 = earfChk(fnxRecipe(['q8_0']));
  assert.strictEqual(q8.status, 'FAIL', q8.detail);
  assert.strictEqual(q8.earf, 0.975);
  const q4 = earfChk({ box: 'odie', model: 'fnx', engine: 'strata', rtr: false, draft: false, files: [], store: { name: 'ud_q4kxl', split: false, types: ['q4k_nl'] } });
  assert.strictEqual(q4.status, 'FAIL', q4.detail);
  assert.strictEqual(q4.earf, 0.955);
  // Explicit looser minEarf option overrides the bar cleanly when requested
  assert.strictEqual(earfChk(fnxRecipe(['q8_0']), { minEarf: 0.97 }).status, 'PASS');

  // 3. Unmeasured sub-4-bit quant (UD-IQ4_XS mixes IQ3_XXS; V4.1 q2_K) -> UNKNOWN with explicit sub-4-bit warning
  const iq4xs = earfChk({ box: 'odie', model: 'fnx', engine: 'strata', rtr: false, draft: false, files: [], store: { name: 'ud_iq4xs', split: false, types: ['q4k_nl'] } });
  assert.strictEqual(iq4xs.status, 'UNKNOWN', iq4xs.detail);
  assert.strictEqual(iq4xs.sub4bit, true);
  assert.ok(/WARNING: EARf unmeasured on sub-4-bit/.test(iq4xs.detail), iq4xs.detail);

  const v41Sub4 = earfChk({ box: 'droopy', model: 'v41', engine: 'ik', rtr: false, draft: false, files: [], store: { split: false, types: ['q2_K'] } });
  assert.strictEqual(v41Sub4.status, 'UNKNOWN', v41Sub4.detail);
  assert.ok(/sub-4-bit/.test(v41Sub4.detail), v41Sub4.detail);

  // 4. Measured EARf >= 0.98 passes both checkRecipe and advisor.runQuality; EARf < 0.98 overrides even a PASS KL gate in advisor
  const good = earfChk(Object.assign(fnxRecipe(['q8_0']), { earf: 0.985, earf_src: 'verified same-engine panel' }));
  assert.strictEqual(good.status, 'PASS', good.detail);
  assert.strictEqual(good.earf, 0.985);

  const badRun = {
    id: 'odie-fnx-iq3xxs-kl-pass-but-earf-fail', when: '2026-11-12T00:00Z', box: 'odie', model: 'fnx', engine: 'strata',
    recipe: fnxRecipe(['gsq_iq3xxs']),
    measured: { outcome: 'ok', kl_verdict: 'PASS', kl: 0.01, top1: 0.98, decode_tps: { single: 90.0 } },
  };
  const rq = ADV.runQuality(data, badRun, [badRun]);
  assert.strictEqual(rq.status, 'FAIL', JSON.stringify(rq));
  assert.strictEqual(rq.earf, 0.894);
  assert.ok(/EARf 0\.894 < 0\.98/.test(rq.basis), rq.basis);
  const bp = ADV.bestPassing(data, [badRun]);
  assert.strictEqual(bp.best.length, 0, 'EARf FAIL run must never pass bestPassing');
  assert.ok(bp.skipped.some(s => s.run === badRun.id && /EARf 0\.894 < 0\.98/.test(s.why)), JSON.stringify(bp.skipped));
});

console.log(n + ' tests passed');
