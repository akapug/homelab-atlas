#!/usr/bin/env node
// Planner advice built on src/model.js: the layer map's calibration summary, the route-trim proposer (the layer map plans a
// late-layer trim on a measured recipe) and the upgrade advisor (which lane binds each box's best quality-passing recipe, and
// which catalog part would buy >= suggest_min_gain of decode). report.js prints these; build.py stores them in data.json
// (node src/advisor.js --json prints {layer_maps, trim, upgrades}).
'use strict';
(function () {
  const fs = require('fs'), path = require('path');
  const FM = require('./model.js'), ACC = require('./accuracy.js');
  const v = FM.v;

  function ledger(file) {
    return require('./ledger.js').rows(file);
  }
  // the KL / top-1 gate a run stands on (as the box map reads it): its own measured gate, else the newest gate measured on the
  // same store (model, engine and recipe.store equal), else today's planner quality check on its recipe
  function ownGate(m) {
    const g = m.kl_gate || {}, x = String(m.kl_verdict || g.verdict || '');
    return x ? { status: x.startsWith('PASS') ? 'PASS' : x.startsWith('FAIL') ? 'FAIL' : 'UNKNOWN', kl: m.kl !== undefined ? m.kl : g.kl_mean, top1: m.top1 !== undefined ? m.top1 : g.top1 } : null;
  }
  function runQuality(data, run, runs) {
    const own = ownGate(run.measured || {});
    if (own) return Object.assign({ basis: "this run's KL / top-1 gate (" + run.id + ')' }, own);
    const same = runs.filter(x => x.id !== run.id && x.model === run.model && x.engine === run.engine && ownGate(x.measured || {}) &&
      JSON.stringify((x.recipe || {}).store) === JSON.stringify((run.recipe || {}).store)).sort((a, b) => String(b.when).localeCompare(String(a.when)))[0];
    if (same) return Object.assign({ basis: 'KL / top-1 gate of the same store (' + same.id + ')' }, ownGate(same.measured));
    const c = FM.checkRecipe(data, run.recipe, null, { predict: false }).checks.find(x => x.id === 'quality');
    return { status: c ? c.status : 'UNKNOWN', kl: null, top1: null, basis: 'planner quality check: ' + (c ? c.detail : 'none') };
  }
  // per box and model: the fastest measured run that went fine and carries no confounded caveat, is predicted by the planner and passes quality
  function bestPassing(data, runs) {
    const skip = new Set(data.upgrade.settings.exclude_tags), out = {}, why = [];
    runs.forEach(run => {
      const r = run.recipe || {};
      if (r.kind === 'dllm' || (run.tags || []).some(t => skip.has(t)) || ACC.wentBad(run).length || require('./ledger.js').caveated(run).length) return;
      const ph = ACC.primaryPhase(run), meas = ph && ((run.measured || {}).decode_tps || {})[ph];
      if (!(meas > 0)) return;
      const k = run.box + '|' + run.model;
      if (out[k] && out[k].measured >= meas) return;
      const q = runQuality(data, run, runs);
      if (q.status !== 'PASS') { why.push({ run: run.id, why: 'quality ' + q.status + ' (' + q.basis + ')' }); return; }
      const p = FM.predictRecipe(data, r);
      if (p.status !== 'PREDICTED' || !(p.decode_tps > 0)) { why.push({ run: run.id, why: p.status !== 'PREDICTED' ? 'UNMODELLED: ' + p.why : 'draft speed UNMODELLED: ' + p.draft_why }); return; }
      out[k] = { run, recipe: r, phase: ph, measured: meas, quality: q, pred: p };
    });
    return { best: Object.values(out), skipped: why };
  }

  // ---------- layer map ----------
  function mapSummary(data) {
    return Object.keys((data.layer_maps || {}).models || {}).map(mk => {
      const lm = FM.layerMap(data, mk), M = data.layer_maps.models[mk];
      return { model: mk, status: lm.status, why: lm.why, unit: M.unit, dof: lm.dof, alpha: lm.alpha, residual_max: lm.residual_max || null,
        // derived copies name their catalog tag as provenance (data.json counts each tagged catalog value once)
        bands: M.bands.map((b, i) => ({ layers: b.layers, provenance: b.tag, value: b.fit ? (lm.beta || []).find(x => x.layers === b.layers) || null : b.kl, fit: !!b.fit, src: b.src, url: b.url || null })),
        unknown_layers: lm.layer_ids.filter((_, l) => lm.band[l] < 0),
        line: lm.line ? { base_kl: lm.line.base_kl, slope_kl: lm.line.slope_kl, base_top1: lm.line.base_top1, slope_top1: lm.line.slope_top1, max_eff: lm.line.max_eff, binds: lm.line.binds } : null,
        points: lm.points, claims: (lm.claims || []).map(({ tag, ...c }) => Object.assign(c, { provenance: tag })), weights: lm.w || null, layer_ids: lm.layer_ids };
    });
  }
  // the 2-bit damage of a recipe weighted by the map (0 when nothing is 2-bit), or null
  function recipeEff(data, r) {
    const w = FM.layerWeights(data, r.model), dm = w && FM.recipeDamage(data, r);
    return dm && dm.d ? dm.d.reduce((a, x, l) => a + x * w[l], 0) / dm.d.length : null;
  }
  // contiguous planner layers with the same keep -> trim bands in model layer ids
  function keepBands(m, keep) {
    const ids = m.layer_ids || Array.from({ length: m.L }, (_, l) => l), out = [];
    for (let l = 0; l < m.L; l++) {
      if (keep[l] >= m.k) continue;
      const last = out[out.length - 1];
      if (last && last.keep === keep[l] && last.layers[1] === ids[l - 1] && l > 0 && keep[l - 1] === keep[l]) last.layers[1] = ids[l];
      else out.push({ layers: [ids[l], ids[l]], keep: keep[l] });
    }
    return out;
  }
  // Route-trim proposer: from a measured recipe, trim the least sensitive layers first (one rank at a time, down to
  // layer_maps.settings.trim_min_keep) while the map's predicted KL stays under the bar by the map's largest KL residual and
  // its predicted top-1 over the bar by its largest top-1 residual (settings.plan_margin = max_residual)
  function proposeTrim(data, best) {
    const r = best.recipe, mk = r.model, m = data.models[mk], lm = FM.layerMap(data, mk), M = ((data.layer_maps || {}).models || {})[mk];
    const head = { box: r.box, model: mk, engine: r.engine, base_run: best.run.id, base_measured: best.measured };
    if (!M || !M.trim) return Object.assign(head, { status: 'UNKNOWN', why: 'no trim unit in the layer map of ' + mk });
    if (!lm || lm.status !== 'calibrated') return Object.assign(head, { status: 'UNKNOWN', why: 'layer map ' + (lm ? lm.why : 'missing') });
    const ts = FM.trimSupport(data, r);
    if (ts.status === 'no') return Object.assign(head, { status: 'none', why: ts.src });
    const S = data.layer_maps.settings, Q = data.checks.quality;
    if (v(S.plan_margin) !== 'max_residual') return Object.assign(head, { status: 'UNKNOWN', why: 'plan_margin rule ' + v(S.plan_margin) + ' is not known' });
    const base = FM.predictRecipe(data, Object.assign({}, r, { trim: [] }));
    if (base.status !== 'PREDICTED') return Object.assign(head, { status: 'UNKNOWN', why: base.why });
    const miss = base.trim.miss_by_layer, eff = recipeEff(data, r) || 0, mg = lm.residual_max;
    const budget = { kl: Q.bar_kl - mg.kl, top1: Q.bar_top1 + mg.top1 };
    const keep = new Int16Array(m.L).fill(m.k), minKeep = v(S.trim_min_keep);
    const ok = () => { const t = FM.trimKl(data, mk, keep, miss); const q = t.kl === null ? null : FM.mapQuality(data, mk, eff, t.kl); return q && q.kl <= budget.kl && q.top1 >= budget.top1 ? q : null; };
    // least sensitive first (map value per layer), later layers first on a tie
    const order = lm.layer_ids.map((id, l) => l).filter(l => lm.s[l] !== null).sort((a, b) => lm.s[a] - lm.s[b] || b - a);
    order.forEach(l => { while (keep[l] > minKeep) { keep[l]--; if (!ok()) { keep[l]++; break; } } });
    const bands = keepBands(m, keep);
    if (!bands.length) return Object.assign(head, { status: 'none', why: 'no trim step fits the budget (KL <= ' + budget.kl.toFixed(4) + ', top-1 >= ' + budget.top1.toFixed(3) + ')' });
    const rec = Object.assign({}, r, { id: (r.id || best.run.id) + '-trim', label: 'PROPOSED by the layer map: ' + (r.label || r.id) + ' + route trim ' + bands.map(b => 'keep ' + b.keep + ' on ' + b.layers.join('-')).join(', ') + ' (picks that miss VRAM)', trim: bands, proposed_by: 'fit-viz src/advisor.js proposeTrim (layer map)' });
    const p = FM.predictRecipe(data, rec), t = FM.trimKl(data, mk, keep, miss), q = FM.mapQuality(data, mk, eff, t.kl);
    // scale from the measured run's own prediction: a base run that already trims must not count its trim twice
    const own = r.trim && r.trim.length ? FM.predictRecipe(data, r) : base;
    const ck = FM.checkRecipe(data, rec, null, { predict: false }).checks.find(c => c.id === 'quality');
    return Object.assign(head, { status: 'proposed', engine_support: ts, trim: bands, recipe: rec, budget, margin: mg,
      predicted: { label: 'PREDICTED (layer map; trim KL from CLAIMED + INFERRED map values)', decode_planner_before: base.decode_tps, decode_planner: p.decode_tps,
        decode_scaled: best.measured * p.decode_tps / own.decode_tps, gain: p.decode_tps / base.decode_tps - 1, kl: q.kl, top1: q.top1, trim_kl: t.kl, reads_dropped: p.trim.reads_dropped },
      quality_check: ck && { status: ck.status, detail: ck.detail } });
  }

  // ---------- upgrade advisor ----------
  const LANES = ['vram_bw', 'vram_cap', 'dram_bw', 'dram_cap', 'pmem_bw', 'pmem_cap', 'nvme_bw', 'cpu'];
  // a CPU-side tier reads at min(read ceiling, CPU kernel rate): where the two sit together neither alone moves decode, so the
  // pair is probed too and binds as one lane when its gain beats its members' gains compounded by settings.co_binding_min_extra
  // (lanes on different tiers just add up; lanes that sit together gain far more as a pair)
  const COMBOS = [['dram_bw', 'cpu'], ['pmem_bw', 'cpu']];
  const tierGiB = (p, id) => (p.tiers || []).filter(t => t.id === id || (id === 'gpu' && /^gpu/.test(t.id))).reduce((a, t) => a + t.experts_gib, 0);
  const merge = (r, knobs, lane) => {
    const kn = Object.assign({}, r.knobs || {}, knobs || {}), L0 = kn.lane || {}, L = lane || {};
    kn.lane = { scale: Object.assign({}, L0.scale, L.scale), add_gib: Object.assign({}, L0.add_gib) };
    Object.entries(L.add_gib || {}).forEach(([k, x]) => kn.lane.add_gib[k] = (kn.lane.add_gib[k] || 0) + x);
    return Object.assign({}, r, { knobs: kn });
  };
  // one lane made p x faster or bigger, nothing else; null when the box has no such lane
  function probe(data, r, base, lane, p) {
    const box = data.boxes[r.box];
    if (/^pmem/.test(lane) && !box.pmem) return null;
    if (lane.indexOf('+') > 0) return merge(r, null, { scale: Object.fromEntries(lane.split('+').map(x => [x, p])) });
    if (/_bw$|^cpu$/.test(lane)) return merge(r, null, { scale: { [lane]: p } });
    if (lane === 'vram_cap') {
      if (r.banks_experts || r.banks_gib) return Object.assign({}, r, r.banks_experts ? { banks_experts: Math.round(r.banks_experts * p) } : {}, r.banks_gib ? { banks_gib: r.banks_gib * p } : {});
      return merge(r, null, { add_gib: { vram: (p - 1) * tierGiB(base, 'gpu') } });
    }
    if (lane === 'dram_cap') return merge(r, null, { add_gib: { dram: (p - 1) * (tierGiB(base, 'dram') || v(box.dram.usable_gib)) } });
    if (lane === 'pmem_cap') return merge(r, null, { add_gib: { pmem: (p - 1) * v(box.pmem.gib) } });
    return null;
  }
  const priceOf = part => part.price_usd ? v(part.price_usd) : part.price_each_usd ? v(part.price_each_usd).map(x => x * (part.qty || 1)) : null;
  function partEffect(data, b, part, min) {
    const r = b.recipe, br = part.bracket;
    const arms = br ? [br.lo, br.hi] : [null];
    const preds = arms.map(x => FM.predictRecipe(data, merge(r, Object.assign({}, part.knobs || {}, br && x !== null && x !== undefined ? { [br.knob]: x } : {}), part.lane)));
    const bad = preds.find(p => p.status !== 'PREDICTED');
    if (bad) return { id: part.id, part: part.part, status: 'UNMODELLED', why: bad.why };
    const after = preds.map(p => p.decode_tps), gain = after.map(x => x / b.pred.decode_tps - 1);
    const scaled = after.map(x => b.measured * x / b.pred.decode_tps), d = scaled.map(x => x - b.measured), price = priceOf(part);
    const lo = Math.min.apply(null, gain), hi = Math.max.apply(null, gain);
    const why = [];
    if (lo < min) why.push('gain ' + (100 * lo).toFixed(1) + ' % < suggest_min_gain ' + (100 * min).toFixed(0) + ' %');
    if (!b.binding.split('+').every(l => part.lanes.indexOf(l) >= 0)) why.push('lane ' + part.lanes.join('/') + ' is not the binding lane (' + b.binding + ')');
    if (b.quality.status !== 'PASS') why.push('quality ' + b.quality.status);
    return { id: part.id, part: part.part, lanes: part.lanes, status: 'PREDICTED', suggested: !why.length, why_not: why,
      gain: [lo, hi], tok_s: { planner_before: b.pred.decode_tps, planner_after: [Math.min.apply(null, after), Math.max.apply(null, after)], measured_before: b.measured, scaled_after: [Math.min.apply(null, scaled), Math.max.apply(null, scaled)] },
      price_usd: price, price_date: part.price_date || null, price_src: (part.price_usd || part.price_each_usd || {}).src || null,
      usd_per_tok_s: price && Math.max.apply(null, d) > 0 ? [price[0] / Math.max.apply(null, d), Math.min.apply(null, d) > 0 ? price[1] / Math.min.apply(null, d) : null] : null,
      bracket: br ? br.src : null, caveat: part.caveat || null };
  }
  function upgradeAdvice(data, runs) {
    const U = data.upgrade, S = U.settings, min = S.suggest_min_gain, P = S.probe_scale;
    const B = bestPassing(data, runs), boxes = {};
    Object.keys(data.boxes).forEach(bk => boxes[bk] = { box: bk, models: [], suggestions: [], unpriced: [] });
    B.best.forEach(b => {
      const r = b.recipe, one = l => {
        const x = probe(data, r, b.pred, l, P), p = x && FM.predictRecipe(data, x);
        return p && p.status === 'PREDICTED' ? { lane: l, gain: p.decode_tps / b.pred.decode_tps - 1 } : null;
      };
      const single = LANES.map(one).filter(Boolean);
      const pairs = COMBOS.filter(c => c.every(l => single.some(x => x.lane === l))).map(c => one(c.join('+')))
        .filter(x => x && (1 + x.gain) / x.lane.split('+').reduce((a, l) => a * (1 + single.find(y => y.lane === l).gain), 1) - 1 > S.co_binding_min_extra);
      const lanes = single.concat(pairs).sort((x, y) => y.gain - x.gain);
      b.binding = lanes.length && lanes[0].gain > 1e-3 ? lanes[0].lane : 'none';
      const parts = U.parts.filter(x => x.box === r.box).map(x => partEffect(data, b, x, min));
      const row = { model: r.model, engine: r.engine, run: b.run.id, recipe: r.id || null, phase: b.phase, measured_tok_s: b.measured, predicted_tok_s: b.pred.decode_tps,
        quality: b.quality, ms_per_lane: b.pred.ms_per_lane, probe_scale: P, lanes, binding: b.binding, parts };
      boxes[r.box].models.push(row);
      parts.filter(x => x.suggested).forEach(x => boxes[r.box].suggestions.push(Object.assign({ box: r.box, model: r.model, run: b.run.id }, x)));
      const top = lanes[0];
      if (top && top.gain >= min && !U.parts.some(x => x.box === r.box && top.lane.split('+').every(l => x.lanes.indexOf(l) >= 0)))
        boxes[r.box].unpriced.push({ box: r.box, model: r.model, run: b.run.id, lane: top.lane, probe_gain: top.gain, why: 'the binding lane gains >= ' + (100 * min).toFixed(0) + ' % at ' + P + 'x, but no catalog part (data/upgrades.json parts) prices it' });
    });
    const all = [].concat.apply([], Object.values(boxes).map(x => x.suggestions));
    return { settings: S, built: data.meta.built, boxes, suggestions: all, skipped: B.skipped };
  }
  function trimProposals(data, runs) {
    return bestPassing(data, runs).best.map(b => proposeTrim(data, b)).filter(x => x.status !== 'none');
  }
  function all(data, runs) {
    return { layer_maps: mapSummary(data), trim: trimProposals(data, runs), upgrades: upgradeAdvice(data, runs) };
  }

  const api = { ledger, runQuality, bestPassing, mapSummary, recipeEff, keepBands, proposeTrim, probe, partEffect, upgradeAdvice, trimProposals, all, LANES };
  module.exports = api;
  if (require.main === module) {
    const ROOT = path.join(__dirname, '..');
    const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'data.json'), 'utf8'));
    FM.applyCal(data);
    const out = all(data, ledger(path.join(ROOT, 'runs', 'runs.jsonl')));
    out.calibration_split = require('./accuracy.js').calibrationSplit(data);
    fs.writeSync(1, JSON.stringify(out, null, process.argv.includes('--json') ? 0 : 1) + '\n');
  }
})();
