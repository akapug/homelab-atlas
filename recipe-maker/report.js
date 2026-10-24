#!/usr/bin/env node
// Node report: best projected point at 20 and 25 tok/s per model x box x engine, calibration residuals,
// owner's droopy layout, and a knob-combination search. Usage: node report.js [--json out.json] [--quick]
// Pre-launch check: node report.js --check recipe.json [--json] [--strict]. Run ledger accuracy: node report.js --accuracy [runs.jsonl] [--json] [--bucket day|week]
// External validation (community rows, runs/external.jsonl): node report.js --external [--json]
// Screening rounds against the card conditions (runs/screen.jsonl, data/cards): node report.js --screen [--json]
// Batched decode (N concurrent streams, data.batch): node report.js --batch [--json]
// Batch window (bottom-up kernel cost per pipeline stage, data.window; lever table): node report.js --window [CONFIG] [--agents N] [--xfloor K] [--bw GBS] [--stages S] [--anchor ID | --no-anchor] [--source ubench] [--propose LEVER] [--json]
// Layer map + trim proposals + upgrade advisor only (src/advisor.js; also part of the full and --quick reports): node report.js --advice [--json]
'use strict';
const fs = require('fs'), path = require('path');
const FM = require('./src/model.js');
const ACC = require('./src/accuracy.js');
const data = JSON.parse(fs.readFileSync(path.join(__dirname, 'data.json'), 'utf8'));
const args = process.argv.slice(2);
const quick = args.includes('--quick');
const t0 = Date.now();
FM.applyCal(data);
const v = FM.v, GiB = FM.GiB;

// ---------- pre-launch check of one recipe: node report.js --check recipe.json [--json] [--strict] ----------
// exit 0 = no FAIL (UNKNOWN or UNMODELLED allowed unless --strict), 1 = a hard check FAILs (refuse the launch), 2 = bad input, 3 = UNKNOWN or UNMODELLED with --strict
const ci = args.indexOf('--check');
if (ci >= 0) {
  const os = require('os');
  let rs;
  try { rs = JSON.parse(fs.readFileSync(args[ci + 1], 'utf8')); } catch (e) { console.error('cannot read recipe ' + args[ci + 1] + ': ' + e.message); process.exit(2); }
  const list = Array.isArray(rs) ? rs : [rs];
  let worst = 0;
  const outAll = [], gaps = [];
  for (const r of list) {
    // resolve symlinks only on the box itself (a link elsewhere says nothing about the box's disks)
    const here = os.hostname().split('.')[0] === r.box;
    const resolve = p => { const x = String(p).replace(/^~(?=\/)/, os.homedir()); try { return here ? fs.realpathSync(x) : p; } catch (e) { return p; } };
    const res = FM.checkRecipe(data, r, resolve);
    const pr = res.prediction || null;
    // the recipe fields the planner keys on travel with the gate output, so a ledger row read later shows what was predicted
    const carried = Object.fromEntries(['engine_build', 'banks_experts', 'ram_tier_gib'].filter(k => r[k] !== undefined).map(k => [k, r[k]]));
    outAll.push({ id: r.id, label: r.label, box: r.box, model: r.model, engine: r.engine, ...carried, paths_resolved: here, checks: res.checks.map(c => ({ id: c.id, status: c.status, detail: c.detail })), notes: res.notes, facts: res.facts,
      prediction: pr && Object.assign({ basis: 'launch', at: new Date().toISOString().slice(0, 19) + 'Z', data_built: data.meta.built }, pr) });
    if (!args.includes('--json')) {
      console.log(`== ${r.id || '(no id)'}: ${r.label || ''} [${r.box} / ${r.model} / ${r.engine}]` + (here ? ' (paths resolved on this box)' : ' (paths taken as given: run on the box to resolve symlinks)'));
      res.checks.forEach(c => console.log(`  ${c.status.padEnd(7)} ${c.id.padEnd(9)} ${c.detail}`));
      if (pr) console.log('  predict ' + ACC.fmtPrediction(pr));
      if (pr && pr.evidence) pr.evidence.forEach(e => console.log('  trust   ' + FM.fmtRollup(e)));
      res.notes.forEach(n => console.log('  note: ' + n));
    }
    // what the gate could not judge: UNKNOWN checks and speeds it has no number for; GO never covers them
    const gap = [...res.checks.filter(c => c.status === 'UNKNOWN').map(c => c.id),
      ...(!pr || pr.status !== 'PREDICTED' ? ['speed UNMODELLED'] : Number.isFinite(pr.prefill_tps) ? [] : ['prefill UNMODELLED'])];
    gaps.push(...gap);
    // --strict holds on anything unjudged: an UNKNOWN check or a speed the planner has no number for
    const s = res.bad_input ? 2 : res.checks.some(c => c.status === 'FAIL') ? 1 : (args.includes('--strict') && gap.length) ? 3 : 0;
    if (s === 2 || (s === 1 && worst !== 2) || (s === 3 && worst === 0)) worst = s;
  }
  // writeSync: console.log on a pipe is asynchronous and process.exit() would cut the JSON at 64 KiB
  if (args.includes('--json')) fs.writeSync(1, JSON.stringify(outAll.length === 1 ? outAll[0] : outAll, null, 1) + '\n');
  else console.log(`verdict: ${['GO (no FAIL)', 'REFUSE (a hard check FAILs)', 'BAD INPUT', 'HOLD (not judged, with --strict)'][worst]}` +
    (gaps.length ? `; not judged, so not passed: ${[...new Set(gaps)].join(', ')}` : ''));
  process.exit(worst);
}

// ---------- external validation: node report.js --external [--json] ----------
// our physics on each community row (runs/external.jsonl, MEASURED-BY-THEM = inferred) and the gap per hardware class
if (args.includes('--external')) {
  const S = FM.externalSummary(data);
  if (args.includes('--json')) { fs.writeSync(1, JSON.stringify({ rows: S.rows.map(r => ({ id: r.id, hw_class: r.hw_class, status: r.status, why: r.why, placement: r.placement, ours: r.decode_tps, theirs: r.measured, err: r.err })), by_class: S.by_class, by_class_placement: S.by_class_placement, unmodelled: S.unmodelled }, null, 1) + '\n'); process.exit(0); }
  const L = FM.fmtExternal(S), miss = data.external.rows.filter(r => r.missing.length);
  console.log('external rows: ' + S.rows.length + ' (' + data.external.file + '; every row is inferred: a validation set, never a fitted term)');
  L.rows.forEach(x => console.log('  ' + x));
  console.log('per hardware class:'); L.by_class.forEach(x => console.log('  ' + x));
  console.log('per class and placement:'); L.by_class_placement.forEach(x => console.log('  ' + x));
  console.log('rows with undeclared fields: ' + miss.length + ' of ' + data.external.rows.length + ' (' + miss.map(r => r.id + ' [' + r.missing.join(' ') + ']').slice(0, 5).join('; ') + (miss.length > 5 ? '; ...' : '') + ')');
  process.exit(0);
}

// ---------- screening rounds: node report.js --screen [--json] ----------
// every quality-round row (runs/screen.jsonl) against its model's card conditions (data/cards): the verdict-ready check,
// each departure, and any keep/delete verdict the round gave on a run that was not verdict-ready
if (args.includes('--screen')) {
  const S = FM.screenChecks(data);
  if (args.includes('--json')) { fs.writeSync(1, JSON.stringify(S.map(x => ({ id: x.row.id, card: x.row.card, arm: x.row.arm, shapes: x.row.shapes, settings: x.row.settings, check: x.check })), null, 1) + '\n'); process.exit(0); }
  console.log(`screening runs: ${S.length} (${data.screen.file}); verdict-ready refuses a keep/delete verdict over ${(100 * v(data.checks.verdict_ready.shape_rate_max)).toFixed(0)} % failure shapes or a card departure without a recorded best-config sweep`);
  S.forEach(({ row, check }) => {
    console.log(`== ${row.id} [${row.card} on ${row.box} / ${row.engine}]${row.verdict ? ' verdict given: ' + row.verdict.call : ''}`);
    console.log(`  ${check.status.padEnd(7)} ${check.detail}`);
    if (check.refused) console.log(`  REFUSED  the ${row.verdict.call} verdict waits for a run under the card conditions or a best-config sweep (${row.verdict.src})`);
  });
  const cs = Object.keys(data.cards).filter(k => !S.some(x => x.row.card === k));
  if (cs.length) console.log('cards with no screening run: ' + cs.map(k => k + ' (' + data.cards[k].name + ')').join(', '));
  process.exit(0);
}

// ---------- batch window: node report.js --window [CONFIG] [--agents N] [--xfloor K] [--bw GBS] [--stages S] [--anchor ID | --no-anchor] [--source ubench] [--propose LEVER] [--json] ----------
// The bottom-up cost of one batched-decode window per pipeline stage (data.window, src/model.js win*): the calibration and its
// errors, tok/s per agent count, the bottleneck stage's largest terms, the lever table (each kernel class at k x its bandwidth
// floor, ranked by tok/s at the plan's agents; the layer-split rebalance; the unmeasured extras) and the what-ifs. A measured
// profile in the config is the source of the plan and the levers; the ubench classes are the fallback (--source ubench)
const wi = args.indexOf('--window');
if (wi >= 0) {
  const W0 = data.window || { configs: {} }, num = f => { const i = args.indexOf(f); return i >= 0 ? Number(args[i + 1]) : undefined; };
  const key = args[wi + 1] && !args[wi + 1].startsWith('--') ? args[wi + 1] : W0.default;
  if (!key || !W0.configs[key]) { console.log('batch window: no config ' + (key || '') + ' in this build (data/window/*.json; configs: ' + (Object.keys(W0.configs).join(', ') || 'none') + ')'); process.exit(key ? 2 : 0); }
  const R = FM.winReport(data, key, { agents: num('--agents'), xfloor: num('--xfloor'), bw: num('--bw'), stages: num('--stages'), anchor: args.includes('--no-anchor') ? null : (args.indexOf('--anchor') >= 0 ? args[args.indexOf('--anchor') + 1] : undefined), source: args.indexOf('--source') >= 0 ? args[args.indexOf('--source') + 1] : undefined, propose: args.indexOf('--propose') >= 0 ? args[args.indexOf('--propose') + 1] : undefined }), W = W0.configs[key];
  if (args.includes('--json')) { fs.writeSync(1, JSON.stringify(R, null, 1) + '\n'); process.exit(0); }
  if (args.includes('--propose')) {
    if (!R.template) { console.log('no lever named ' + args[args.indexOf('--propose') + 1] + '; levers: ' + R.levers.levers.concat(R.levers.unmeasured).map(x => x.lever).join(', ')); process.exit(2); }
    console.log('// a proposal for ' + R.template.model.lever + ': fill the blanks and add it to the config\'s proposals (the model and gate fields are for reading)');
    fs.writeSync(1, JSON.stringify(R.template, null, 1) + '\n'); process.exit(0);
  }
  const f = R.fit, K = f.K, ms = xs => xs.map(x => x.toFixed(2)).join('/'), pct = x => (x === null ? '-' : (x >= 0 ? '+' : '') + (100 * x).toFixed(1) + ' %');
  const tps = x => x.toFixed(1), sp = s => s.map(([a, b]) => a + '-' + b).join(' | ');
  console.log(`batch window: ${R.label}`);
  console.log(`  ubench: ${W.ubench.src}; ${W.ubench.classes.length} kernel classes over ${W.ubench.measured_layers} layers (${Object.entries(W.ubench.kinds).map(([k, c]) => c + ' ' + k).join(', ')}); floor at ${W.ubench.floor_rows} rows / ${FM.v(W.ubench.floor_gbs)} GB/s`);
  console.log(`  spec: ${W.spec.src}; ${W.layers.length} layers (${W.layers.filter(x => x.from === 'model').length} kinds from the model's layer list)` + (R.unmeasured_kinds.length ? `; kinds the ubench did not measure: ${R.unmeasured_kinds.join(', ')}` : ''));
  console.log(`calibration (FIT on ${f.equations} stage times, ${f.free} free constants${f.equations <= f.free ? ': exact in sample by construction' : ''}): ${K.omega_us.toFixed(1)} us taken off each isolated launch; gamma ${K.gamma.toFixed(2)} (${W.settings.gamma.tag}); routing share ${K.phi}; ` +
    W.extras.map(e => `${e.id} ${K.extras[e.id].toFixed(3)} ms${e.per === 'row' ? ' a row' : ''}`).join(', ') + `; pipeline eta ${K.eta.toFixed(4)}`);
  f.residuals.forEach(r => console.log(`  ${r.use.padEnd(5)} ${r.id} (${r.agents} agents, ${r.rows} rows a window${r.change ? '; ' + r.change : ''}): stage ms measured ${ms(r.stage_meas)}` +
    (r.use === 'law' ? '' : `, predicted ${ms(r.stage_pred)} (${r.stage_err.map(pct).join(', ')})`) + (r.tps_meas === null ? '' : `; tok/s measured ${tps(r.tps_meas)}` + (r.tps_pred === null ? '' : `, predicted ${tps(r.tps_pred)} (${pct(r.tps_err)})`) + `, law on the measured stages ${tps(r.law_pred)} (${pct(r.law_err)})`)));
  const p1 = x => (x === null ? '-' : x.toFixed(2) + ' %');
  console.log(`  error: stage ms max |err| fit ${p1(f.stage_fit_max_abs_pct)}, check ${p1(f.stage_check_max_abs_pct)}; tok/s fit ${p1(f.tps_fit_max_abs_pct)}, check ${p1(f.tps_check_max_abs_pct)}; pipeline law ${p1(f.law_max_abs_pct)}`);
  R.phis.forEach(x => console.log(`  routing share ${x.phi}: refit ${x.K.omega_us.toFixed(1)} us, ${W.extras.map(e => e.id + ' ' + x.K.extras[e.id].toFixed(3)).join(', ')}; plan ${tps(x.plan.total)} tok/s`));
  const g = R.gamma_variant;
  console.log(`  kernel-scale reading (gamma fitted ${g.fit.K.gamma.toFixed(2)}, ${g.fit.K.omega_us.toFixed(1)} us, extras ${W.extras.map(e => e.id + ' ' + g.fit.K.extras[e.id].toFixed(3)).join(', ')}; check tok/s ${p1(g.fit.tps_check_max_abs_pct)}): top levers ` + g.levers.slice(0, 4).map(x => `${x.lever} ${tps(x.total)}`).join('; '));
  const PF = R.profile;
  if (PF) {
    console.log(`profile: ${W.profile.src}; ${PF.rows} rows a window at point ${PF.point}; term sums ${ms(PF.term_sum_ms)} ms (the profile's totals ${ms(PF.totals_ms)}), gap (${W.profile.gap.label}) ${ms(PF.gap_ms)} ms; terms follow ${FM.v(W.profile.rows_basis)} row curves (${W.profile.rows_basis.tag})`);
    PF.checks.forEach(r => console.log(`  ${r.use.padEnd(7)} ${r.id} (${r.agents} agents, ${r.rows} rows a window): stage ms measured ${ms(r.stage_meas)}, predicted ${ms(r.stage_pred)} (${r.stage_err.map(pct).join(', ')})` +
      (r.tps_meas === null ? '' : `; tok/s measured ${tps(r.tps_meas)}, predicted ${tps(r.tps_pred)} (${pct(r.tps_err)})`)));
  }
  console.log(`source of the plan and levers: ${R.source === 'profile' ? 'the measured profile (--source ubench for the ubench classes)' : 'the ubench classes' + (PF ? ' (--source ubench)' : '')}`);
  if (R.anchor && R.source === 'ubench') console.log(`anchored on ${R.anchor.id} (its measured stage times; offsets ${ms(R.anchor.offset_ms)} ms a row on the first/last stage hold what changed since the ubench): plan, levers and what-ifs start there (--no-anchor for the bare model, --anchor ID for another point)`);
  const P = R.plan;
  console.log(`plan: ${P.N} agents in ${P.groups} row groups, ${P.rows} rows a window, split ${sp(P.split)}: stage ms ${ms(P.stage_ms)}, bottleneck stage ${P.bottleneck + 1}; ${tps(P.total)} tok/s in total, ${P.per_agent.toFixed(2)} per agent`);
  console.log('  bottleneck stage, largest terms (ms a window): ' + R.top.slice(0, 6).map(x => `${x.class} ${x.ms.toFixed(2)}`).join('; '));
  console.log('  agents: ' + R.agents.map(a => `${a.N}: ${tps(a.total)} (${a.per_agent.toFixed(1)} each; stages ${ms(a.stage_ms)})${a.beyond_grid ? ' [rows past the ubench grid]' : ''}`).join('; '));
  const L = R.levers;
  console.log(`levers at ${L.xfloor}x floor (${L.source === 'profile' ? 'each term group at k x its mapped classes\' floor' : 'in the graph'}; never below ${FM.v(W.settings.graph_launch_us)} us a launch), tok/s at ${L.agents} agents (now ${tps(L.now.total)}; with the split rebalanced ${tps(L.rebalanced.total)} at ${sp(L.rebalanced.split)}):`);
  const tag = a => a.id + ' ' + a.outcome + (a.repaired_by.length ? '>' + a.repaired_by.join('+') : '');
  const tried = name => (R.evidence[name] || []).filter(a => !a.engine_wide).map(tag).join(', ') || '-';
  const wide = (W.attempts || []).filter(a => !a.network_recipe && a.classes.includes('*'));
  if (wide.length) console.log('  tried on every kernel lever (engine-wide, not repeated in the column): ' + wide.map(a => tag(Object.assign({ repaired_by: W.attempts.filter(b => b.repairs === a.id).map(b => b.id) }, a))).join(', '));
  console.log(`  rank  lever                                 kind     ${L.source === 'profile' ? 'x floor now      ' : 'x floor iso/graph'}  ms in stages     tok/s  (gain)    rebalanced  ` + FM.v(W.settings.lever_xfloor_grid).map(x => x + 'x').join('/').padEnd(17) + 'tried (attempt outcome>repaired by)');
  const xf = x => (x.xfloor_iso !== undefined ? x.xfloor_iso + '/' + (x.xfloor_graph === null ? '-' : x.xfloor_graph.toFixed(1)) : x.xfloor_now !== undefined ? x.xfloor_now.toFixed(2) : '');
  L.levers.forEach((x, i) => console.log(`  ${String(i + 1).padStart(4)}  ${x.lever.padEnd(37)} ${String(x.kind === null ? 'all' : x.kind).padEnd(9)}` +
    `${xf(x).padEnd(19)}` +
    `${(x.saved_ms ? x.saved_ms.map(y => '-' + y.toFixed(2)).join('/') : x.ms_in_stage ? ms(x.ms_in_stage) : ms(x.stage_ms)).padEnd(16)} ${tps(x.total).padStart(6)} (${(x.gain >= 0 ? '+' : '') + x.gain.toFixed(1)})`.padEnd(10) +
    `  ${tps(x.total_rebalanced).padStart(8)}    ${(x.grid ? x.grid.map(y => y.total.toFixed(0)).join('/') : '').padEnd(17)}${tried(x.lever)}`));
  console.log(`  ${L.source === 'profile' ? 'no floor' : 'unmeasured'} (priced at ${100 * FM.v(W.settings.extra_lever_share)} % off): ` + L.unmeasured.map(x => `${x.lever} ${ms(x.ms_in_stage)} ms -> ${tps(x.total)} (rebalanced ${tps(x.total_rebalanced)})`).join('; '));
  const A = W.attempts || [];
  if (A.length) {
    console.log(`attempts (what was already tried; the evidence graph behind the tried column; engine rows read from the ledger):`);
    A.forEach(a => console.log(`  ${a.id.padEnd(4)} ${a.outcome.padEnd(8)} ${a.scope.padEnd(12)}${a.label}: ${a.action}` + (a.parent ? `; parent ${a.parent}` : '') + (a.repairs ? `; repairs ${a.repairs}` : '') +
      (A.some(b => b.repairs === a.id) ? `; repaired by ${A.filter(b => b.repairs === a.id).map(b => b.id).join(', ')}` : '') +
      (a.tps ? `; ${Object.entries(a.tps).map(([k, x]) => k + ' ' + FM.v(x)).join(', ')} tok/s; sanity ${a.sanity === null ? 'unknown' : FM.v(a.sanity) === true ? 'passed' : FM.v(a.sanity) === false ? 'none' : 'unknown'}` : '') + (a.result ? `; ${a.result}` : '')));
  }
  if (R.ubench_levers) console.log('  the ubench reading of the levers (the fallback; anchored): ' + R.ubench_levers.slice(0, 6).map(x => `${x.lever} ${tps(x.total)}`).join('; '));
  const FR = R.frontier;
  if (FR) {
    console.log(`against the frontier ${FR.label} (${tps(FR.tps)} tok/s measured, split ${sp(FR.split)}; noise ${FR.noise_pct} %): each lever's modelled gain on its rebalanced split applied to the frontier; no-floor rows at the share off:`);
    FR.rank.slice(0, 10).forEach((x, i) => console.log(`  ${String(i + 1).padStart(4)}  ${x.lever.padEnd(37)} ${tps(x.at_frontier).padStart(6)} (${(x.gain_pct >= 0 ? '+' : '') + x.gain_pct.toFixed(1)} %)${x.inside_noise ? ' inside noise' : ''}${x.no_floor ? ' [no floor]' : ''}; tried ${x.tried ? Object.entries(x.tried).map(([k, n]) => n + ' ' + k).join(', ') : 'never'}`));
    if (FR.gate) console.log(`  correctness gate (nothing is accepted before it; ${FR.gate.src}):\n` + FR.gate.steps.map((g, i) => `    ${i + 1}. ${g}`).join('\n'));
  }
  if (R.network.length) {
    console.log('shared kernel/network frontier: ' + (FR ? FR.shared_rank.length : 0) + ' comparable rows; one-box headline unchanged');
    (FR ? FR.candidates : R.network).forEach(x => console.log('  ' + (x.eligible ? tps(x.at_frontier) : 'UNRANKED') + ' ' + x.lever + ' [' + x.status + '] ' + (x.comparison ? x.comparison.why.join('; ') : '') + (x.checks ? '; ' + x.checks.map(c => c.id + ' ' + c.status).join(', ') : '')));
  }
  if (R.proposals.length) {
    console.log('proposals (node report.js --window --propose LEVER prints a blank one; dedup: same mechanism key + same action = duplicate, same key = related):');
    R.proposals.forEach(p => console.log(`  ${p.id} [${p.dedup}${p.prior.length ? ': ' + p.prior.map(x => x.id + ' ' + x.outcome).join(', ') : ''}] ${p.lane}: ${p.mechanism.key}: ${p.action} (${p.evidence})` +
      (p.expected ? `; expected ${tps(p.expected.total)} tok/s (${(p.expected.gain >= 0 ? '+' : '') + p.expected.gain.toFixed(1)}; rebalanced ${tps(p.expected.total_rebalanced)}) via ${p.expected.lever}` : '; no lever matches its classes') +
      `\n      reuses: ${(p.reuse || []).join('; ') || '-'}; explores: ${p.exploration || '-'}; prior attempts on its targets: ${p.evidence_on_targets.map(a => a.id + ' ' + a.outcome).join(', ') || 'none'}` +
      `\n      cheapest test: ${p.test}; falsifier: ${p.falsifier}`));
  }
  console.log(`what-if, cards (the config's stages, each card's bandwidth; kernel efficiency carried over, extras, waits and the gap unchanged: ESTIMATE): ` + R.whatif.cards.map(c => `${c.name} ${c.bw} GB/s: ${tps(c.step.total)} (rebalanced ${tps(c.rebalanced.total)}); top levers ${c.levers.join(', ')}`).join('; '));
  console.log(`what-if, stages (one card each, as many row groups as the config's ratio, split rebalanced; extra hand-offs unpriced, VRAM not checked): ` + R.whatif.stages.map(s => `${s.split.length}: ${tps(s.total)} tok/s (${s.rows.toFixed(1)} rows, ${sp(s.split)}, stage ms ${ms(s.stage_ms)})`).join('; '));
  process.exit(0);
}

// ---------- agents vs speed: node report.js --curve [RUN_ID | all] [--min X] [--second X] [--max N] [--json] ----------
// Per config (ledger rows with aggregate tok/s at two or more agent counts, pooled by box/model/engine/build/flags/ctx; data.curves,
// src/model.js curve*): the fitted window law, total and per-agent tok/s from 1 to --max agents, the largest agent count with
// each agent at >= --min tok/s (headline) and >= --second, the asymptotic total, the host-memory ceiling and tok/s per watt.
// No RUN_ID: the default config (the window frontier's run). all: one line per config. A RUN_ID picks the config holding that run
const cvi = args.indexOf('--curve');
if (cvi >= 0) {
  const C = data.curves, num = f => { const i = args.indexOf(f); return i >= 0 ? Number(args[i + 1]) : undefined; };
  const opts = { min: num('--min'), second: num('--second'), max: num('--max') };
  const want = args[cvi + 1] && !args[cvi + 1].startsWith('--') ? args[cvi + 1] : null;
  const t1 = x => x.toFixed(1), pct = x => (x >= 0 ? '+' : '') + (100 * x).toFixed(1) + ' %';
  if (want === 'all') {
    const all = FM.curveReports(data, opts);
    if (args.includes('--json')) { fs.writeSync(1, JSON.stringify(all.map(r => Object.assign({}, r, { cfg: r.cfg.id, fit: r.ok ? { A: r.fit.A, B: r.fit.B, err_max: r.fit.err_max, win: r.fit.win } : { why: r.why } })), null, 1) + '\n'); process.exit(0); }
    // measured-range readouts first (an extrapolated headline sits past every agent count the runs measured), each by its total
    const ok = all.filter(r => r.ok).sort((a, b) => a.headline.extrapolated - b.headline.extrapolated || b.headline.total - a.headline.total), S = C.settings;
    console.log(`agents vs speed: ${all.length} configs from the ledger, ${ok.length} with a curve; most agents at >= ${v(S.min_each_tps)} tok/s each (total there; x = past the largest measured count, m = past the host-memory ceiling), at >= ${v(S.second_each_tps)}, the asymptotic total, the host-memory ceiling (slots), the fit's max error, tok/s per W at the headline (measured.${v(S.power_field)})`);
    const mk = h => (h.extrapolated ? 'x' : ' ') + (h.past_host ? 'm' : ' ');
    const pw = x => x ? ` ${x.toFixed(3)} tok/s per W` : '';
    ok.forEach(r => console.log(`  ${String(r.headline.n).padStart(3)} @ ${t1(r.headline.total).padStart(6)}${mk(r.headline)} ${String(r.second.n).padStart(3)} @ ${t1(r.second.total).padStart(6)}${mk(r.second)} asym ${t1(r.asymptote).padStart(6)}  host ${r.host.ceiling === null ? '  ?' : String(r.host.ceiling).padStart(3)}  fit ${(100 * r.fit.err_max).toFixed(1).padStart(4)} %${r.fit_warn ? '!' : ' '} measured n ${[...new Set(r.fit.pts.map(p => p.n))].join('/')}${pw(r.headline.tps_per_w)}  ${r.cfg.id}`));
    all.filter(r => !r.ok).forEach(r => console.log(`  no curve: ${r.cfg.id}: ${r.why}; host ${r.host.ceiling === null ? '?' : r.host.ceiling}${r.power.map(p => `; n=${p.n} ${t1(p.tps)} @ ${t1(p.watts)} W${pw(p.tps_per_w)}`).join('')}`));
    process.exit(0);
  }
  const c = want ? C.configs.find(x => x.runs.indexOf(want) >= 0 || x.id === want) : C.configs.find(x => x.key === C.default);
  if (!c) { console.log('agents vs speed: ' + (want ? 'no config holds run ' + want : 'no default config') + ' (node report.js --curve all lists them)'); process.exit(want ? 2 : 0); }
  const R = FM.curveReport(data, c, opts);
  if (args.includes('--json')) { fs.writeSync(1, JSON.stringify(Object.assign({}, R, { fit: R.ok ? Object.assign({}, R.fit, { at: undefined }) : R.fit }), null, 1) + '\n'); process.exit(0); }
  console.log(`agents vs speed: ${c.label}`);
  console.log(`  ${c.box} ${c.model} ${c.engine}; ${v(c.groups)} batch group${v(c.groups) > 1 ? 's' : ''} (${c.groups.src}), ${v(c.stages)} stage${v(c.stages) > 1 ? 's' : ''} (${c.stages.src}), ${c.slots ? v(c.slots) + ' slots' : c.slots_each_point ? 'one server per agent count (--batch n)' : 'slots not given'}; runs ${c.runs.join(', ')}`);
  console.log('  measured: ' + c.points.map(p => `n=${p.n} ${t1(v(p.tps))}${p.used ? '' : ' (not fitted: ' + p.why + ')'}${p.watts ? ' at ' + t1(v(p.watts)) + ' W' : ''}`).join('; '));
  if (!R.ok) { console.log('  no curve: ' + R.why); process.exit(0); }
  const f = R.fit;
  console.log(`  fit (FIT): n / total = ${f.A.toPrecision(4)} x max(G, S) + ${f.B.toPrecision(4)} x max(G, S) x n / G; max |err| ${(100 * f.err_max).toFixed(1)} % ` + f.errs.map(e => `(n=${e.n} ${t1(e.meas)} vs ${t1(e.pred)}, ${pct(e.err)})`).join(' '));
  if (R.fit_warn) console.log(`  WARNING: the fit misses a measured point by more than ${v(C.settings.fit_warn_pct)} %: the points do not follow the window law well (a step in the kernels, or mixed conditions)`);
  if (f.win) console.log(`  stage window (READ from the run notes, slowest stage): ` + f.win.pts.map(x => `n=${x.n} r=${x.r} ${x.t.toFixed(2)} ms, eff ${x.eff.toFixed(3)}`).join('; ') + (f.win.a !== null ? `; t(r) = ${f.win.a.toFixed(2)} + ${f.win.b.toFixed(3)} x r ms` : '') + `; mean eff ${f.win.eff.toFixed(3)}`);
  const H = R.headline, I = R.headline_in_memory, S2 = R.second, hc = R.host;
  const w = x => x.watts ? `; ${t1(x.watts.w)} W (${x.watts.tag}), ${x.tps_per_w.toFixed(2)} tok/s per W` : '';
  console.log(`  HEADLINE: ${H.n}${H.at_max ? '+' : ''} agents at >= ${R.min} tok/s each, ${t1(H.total)} tok/s total (${t1(H.each)} each)${H.extrapolated ? ', extrapolated past the measured counts' : ''}${H.past_host ? `, PAST the host-memory ceiling` : ''}${w(H)}`);
  if (I) console.log(`  within host memory: ${I.n} agents, ${t1(I.total)} tok/s total (${t1(I.each)} each)${w(I)}`);
  console.log(`  at >= ${R.second_min} tok/s each: ${S2.n} agents, ${t1(S2.total)} tok/s total${w(S2)}`);
  console.log(`  asymptotic total (agents without limit): ${t1(R.asymptote)} tok/s`);
  console.log(`  host-memory ceiling: ` + (hc.ceiling === null ? `unknown (${hc.first_over ? hc.first_over.detail : 'no check'})` : hc.all_fit ? `all ${R.max} slots fit` : `${hc.ceiling} slots (at ${hc.first_over.n}: ${hc.first_over.status}: ${hc.first_over.detail.slice(0, 260)}...)`) + (hc.with_lend_drop !== undefined ? `; ${hc.with_lend_drop} with STRATA_STAGE_LEND_DROP=1 (serving.stage_lend_drop)` : ''));
  if (R.power.length) console.log('  power: ' + R.power.map(p => `n=${p.n} ${t1(p.watts)} W, ${p.tps_per_w.toFixed(2)} tok/s per W`).join('; '));
  else console.log(`  power: no watts recorded for these runs (measured.${v(C.settings.power_field)})`);
  const pick = new Set(v(C.settings.table_n).concat([H.n, S2.n, I ? I.n : 0, hc.ceiling || 0]).filter(n => n >= 1 && n <= R.max));
  console.log('      n   total  each  ' + ['', 'measured'].join(''));
  R.rows.filter(r => pick.has(r.n)).forEach(r => console.log(`  ${String(r.n).padStart(5)} ${t1(r.total).padStart(7)} ${t1(r.each).padStart(5)}  ${[r.measured.map(m => t1(m.tps) + (m.used ? '' : '*')).join('/'), r.extrapolated ? 'extrapolated' : '', r.beyond_slots ? 'needs --batch ' + r.n : '', r.fits_host === false ? 'over host memory' : '', r.n === H.n ? '<- headline' : '', r.n === S2.n ? '<- >= ' + R.second_min + ' each' : ''].filter(Boolean).join(', ')}`));
  process.exit(0);
}

// ---------- batched decode: node report.js --batch [--json] ----------
// N concurrent streams of one model (data.batch): the fit of the free constants per miss mode, its residuals on the ledger rows,
// the hit-rate sensitivity, the one-card vs split twin, and the planning table (config x kernel variant x context x streams)
if (args.includes('--batch')) {
  if (!data.batch) { console.log('batched decode: no data.batch in this build (its ledger rows are not here)'); process.exit(0); }
  const B = data.batch, fits = B.miss_modes.v.map(m => FM.batchFit(data, m));
  const sens = [].concat(...B.miss_modes.v.map(m => B.hit_four_range.v.map(h => FM.batchFit(data, m, h))));
  const tables = fits.map(x => ({ mode: x.mode, table: FM.batchTable(data, x) }));
  if (args.includes('--json')) { fs.writeSync(1, JSON.stringify({ fits, sensitivity: sens.map(s => ({ mode: s.mode, hit4: s.hit4, K: s.K, F: s.F, median_abs_pct: s.median_abs_pct })), tables }, null, 1) + '\n'); process.exit(0); }
  const pct = x => (x >= 0 ? '+' : '') + (100 * x).toFixed(1) + ' %';
  const fk = x => `fixed ${x.K.fixed.toFixed(2)} ms + ${x.K.row.toFixed(2)} ms per row; MTP ${x.F.tau.toFixed(2)} tokens per one-stream step; misses ${(100 * x.F.miss1).toFixed(1)} % + ${(100 * x.F.churn).toFixed(2)} % per extra row + ${(100 * x.F.mpg).toFixed(2)} % per GiB the other rows take`;
  console.log(`batched decode: ${data.models[B.model].name} ${B.store} on ${B.box} (${B.engine}); ${B.rows.length} ledger rows, constants in data.batch`);
  fits.forEach(x => {
    console.log(`fit, misses ${x.mode}: ${fk(x)} (FIT); median |err| ${x.median_abs_pct.toFixed(1)} %, max ${x.max_abs_pct.toFixed(1)} %`);
    x.residuals.forEach(e => console.log(`  ${e.use.padEnd(6)} ${e.id} N ${e.N}: measured ${e.meas.toFixed(1)}, predicted ${e.pred.toFixed(1)} (${pct(e.err)}; ${e.binding})`));
    console.log(`  twin  split / one card at one stream: measured ${x.twin.meas.toFixed(3)}, predicted ${x.twin.pred.toFixed(3)}`);
  });
  const key = B.sensitivity_row;
  sens.forEach(s => {
    const t = FM.batchTable(data, s).find(x => x.config === key.config && x.variant === key.variant && x.ctx === key.ctx);
    console.log(`sensitivity, misses ${s.mode}, four-stream hit ${s.hit4}: ${fk(s)}; median |err| ${s.median_abs_pct.toFixed(1)} %, twin ${s.twin.pred.toFixed(3)}; ${key.config}/${key.variant}/ctx ${key.ctx}: ` + t.steps.map(x => x.N + ': ' + x.total.toFixed(0)).join(', '));
  });
  tables.forEach(({ mode, table }) => {
    console.log(`table, misses ${mode}: total tok/s (per stream) [binding] per stream count ${B.streams.v.join('/')}; target ${B.target_tps.v} tok/s`);
    table.forEach(t => {
      const s0 = t.steps[0], last = t.steps[t.steps.length - 1];
      console.log(`== ${t.config} / ${t.variant} (${t.variant_label}) / ctx ${t.ctx}: first N at ${B.target_tps.v}: ${t.first_n_at_target || 'none'}; row ${s0.row_gib.toFixed(2)} GiB in VRAM` +
        (s0.free_gib === null ? ' (taken from the expert cache)' : `, free ${s0.free_gib.toFixed(1)} GiB`) + `; weights ${s0.weights_gib.toFixed(1)} GiB; RAM at N ${last.N}: ${last.ram_gib.toFixed(1)} GiB`);
      console.log('  ' + t.steps.map(s => `${s.N}: ${s.total.toFixed(0)} (${s.per_stream.toFixed(1)}) [${s.binding}${s.stream ? ', KV streamed' : ''}${s.fits ? '' : ', DOES NOT FIT'}]`).join('; '));
    });
  });
  process.exit(0);
}

// ---------- run-ledger accuracy: node report.js --accuracy [runs.jsonl] [--json] ----------
// A launch-time prediction/check (captured by record_run.py from --check --json) is used as recorded; backfilled runs are
// predicted and checked post hoc with today's planner and are labelled so. Speed errors are scored only on runs that went fine
// and carry no confounded caveat (runs/run_caveats.json, read by src/ledger.js).
const ai = args.indexOf('--accuracy');
if (ai >= 0) {
  const file = args[ai + 1] && !args[ai + 1].startsWith('--') ? args[ai + 1] : path.join(__dirname, 'runs', 'runs.jsonl');
  const LED = require('./src/ledger.js'), runs = LED.rows(file);
  // cause labels beside the ledger (runs/labels.json): read from each row's notes and logs, never guessed
  const lf = path.join(path.dirname(file), 'labels.json'), labels = fs.existsSync(lf) ? JSON.parse(fs.readFileSync(lf, 'utf8')) : {};
  const replay = data.accuracy.replay_checks.v;
  const rows = runs.map(run => {
    const launch = run.prediction && run.prediction.basis === 'launch';
    // today's gate as --check runs it (with the prediction, so mem_cap / ram_fit apply)
    const now = FM.checkRecipe(data, run.recipe, null), today = now.checks;
    const pred = launch ? run.prediction : now.prediction || FM.predictRecipe(data, run.recipe);
    // a launch capture keeps its own verdicts; a check added after it (accuracy.replay_checks) is replayed post hoc and labelled
    const ck = run.check ? run.check.checks.concat(today.filter(c => replay.indexOf(c.id) >= 0 && !run.check.checks.some(x => x.id === c.id)).map(c => Object.assign({ replayed: true }, c))) : today;
    const lab = labels[run.id], bad = ACC.wentBad(run, lab);
    const predNow = now.prediction || FM.predictRecipe(data, run.recipe);
    return { id: run.id, when: run.when, box: run.box, model: run.model, engine: run.engine, run, pred, pred_basis: launch ? 'launch' : 'post-hoc', pairs_now: ACC.pairs(run, predNow),
      checks: ck, today, launch_checks: run.check ? run.check.checks : null, check_basis: run.check ? 'launch' : 'post-hoc', bad, scored: !bad.length && !LED.caveated(run).length, pairs: ACC.pairs(run, pred),
      causes: [...new Set(ACC.causesOf(lab).map(c => c.class))], disturbed: ACC.disturbed(lab),
      in_sample: (run.tags || []).filter(t => t.startsWith('anchor:')).map(t => t.slice(7)),
      bound_source: (run.tags || []).filter(t => t.startsWith('bound-source:')).map(t => t.slice(13)) };
  });
  const pct = x => x === null ? '-' : ((x >= 0 ? '+' : '') + (100 * x).toFixed(1) + '%');
  const n2 = x => x === null || x === undefined ? '-' : x.toFixed(2);
  const out = { file, built: data.meta.built, runs: [], summary: {}, verdicts: null, unmodelled: [] };
  console.log(`== Run ledger accuracy (${rows.length} runs from ${path.relative(process.cwd(), file) || file}; planner data built ${data.meta.built}) ==`);
  console.log('decode: the draft prediction pairs with the primary draft phase (n4), the no-draft prediction with n0. post-hoc = predicted today from the recorded recipe; in-sample = the run is a calibration anchor.');
  console.log('\n-- per run --');
  rows.forEach(r => {
    const d = r.pairs.filter(p => p.metric.startsWith('decode'));
    const tag = [r.pred_basis, r.in_sample.length ? 'in-sample ' + r.in_sample.join('/') : '', r.bad.length ? 'went bad: ' + r.bad.join(', ') : '', !r.bad.length && !r.scored ? 'confounded: not scored' : ''].filter(Boolean).join('; ');
    if (r.pred.status !== 'PREDICTED') { console.log(`${r.id.padEnd(30)} UNMODELLED: ${r.pred.why}` + (d.length ? ` | measured ${d.map(p => p.phase + ' ' + n2(p.meas)).join(', ')}` : '') + ` [${tag}]`); out.unmodelled.push({ id: r.id, why: r.pred.why }); }
    else console.log(`${r.id.padEnd(30)} ` + (r.pairs.length ? r.pairs.map(p => `${p.metric}${p.phase ? '[' + p.phase + ']' : ''} ${n2(p.pred)} vs ${n2(p.meas)}` + (p.pred === null ? ' (UNMODELLED)' : p.pct !== null ? ` ${pct(p.pct)}` : ` (${p.err >= 0 ? '+' : ''}${n2(p.err)})`)).join(', ') : 'nothing measured') + ` [${tag}]`);
    out.runs.push({ id: r.id, box: r.box, model: r.model, engine: r.engine, pred_basis: r.pred_basis, status: r.pred.status, why: r.pred.why, scored: r.scored, bad: r.bad, in_sample: r.in_sample, pairs: r.pairs });
  });
  console.log('\n-- decode error per box / model / engine (runs that went fine; error = (pred - meas) / meas) --');
  const METRICS = ['decode_tps', 'decode_tps_nodraft', 'prefill_tps', 'vram_post_gib'];
  for (const metric of METRICS) {
    const s = ACC.summarize(rows, r => r.box + '/' + r.model + '/' + r.engine, metric);
    out.summary[metric] = s;
    s.forEach(x => console.log(`${metric.padEnd(19)} ${x.key.padEnd(22)} n ${String(x.n).padStart(2)}  median |err| ${x.median_abs_pct.toFixed(1).padStart(5)} %  bias ${(x.bias_pct >= 0 ? '+' : '') + x.bias_pct.toFixed(1)} %`));
  }
  // the same summary with today's planner on every run (launch rows replayed from their recorded recipe): how far the model
  // is off now, beside how far it was off when each run launched
  console.log('\n-- the same with today\'s planner on every run (launch-time rows replayed post hoc) --');
  const NOW = rows.map(r => Object.assign({}, r, { pairs: r.pairs_now }));
  out.summary_now = {};
  for (const metric of METRICS.slice(0, 2)) {
    const s = ACC.summarize(NOW, r => r.box + '/' + r.model + '/' + r.engine, metric);
    out.summary_now[metric] = s;
    s.forEach(x => console.log(`${metric.padEnd(19)} ${x.key.padEnd(22)} n ${String(x.n).padStart(2)}  median |err| ${x.median_abs_pct.toFixed(1).padStart(5)} %  bias ${(x.bias_pct >= 0 ? '+' : '') + x.bias_pct.toFixed(1)} %`));
  }
  out.draft_ratios = ACC.draftRatios(rows, r => r.box + '/' + r.model + '/' + r.engine);
  out.draft_ratios.forEach(x => console.log(`draft speed-up ${x.key.padEnd(22)} measured in-load x${x.median.toFixed(3)} median of ${x.n} (` + x.runs.map(y => y.id.replace(/^droopy-|^snoopy-|^odie-/, '') + ' ' + y.meas.toFixed(2)).join(', ') + `) vs planner x${x.pred === null ? '-' : x.pred.toFixed(3)}`));
  const dist = rows.filter(r => r.disturbed.length);
  out.disturbed = dist.map(r => ({ id: r.id, phases: r.disturbed }));
  console.log(`excluded as disturbed (runs/labels.json, from the run notes): ${dist.length}` + (dist.length ? ' (' + dist.map(r => r.id + ' ' + r.disturbed.join('/')).join(', ') + ')' : ''));
  const V = ACC.verdicts(rows, data.causes);
  out.verdicts = V;
  const line = e => `${e.id}: ` + (e.fail.length ? 'FAIL ' + e.fail.join('/') : 'no FAIL') + (e.unknown.length ? ', UNKNOWN ' + e.unknown.join('/') : '') + (e.bad.length ? ' | went bad: ' + e.bad.join(', ') : '') + (e.causes.length ? ' | cause ' + e.causes.join('/') + (e.matched.length ? ' (decided by ' + e.matched.join('/') + ')' : '') : '') + ` [checks ${e.basis}${e.replayed.length ? ' + ' + e.replayed.join('/') + ' replayed post hoc' : ''}${e.bound_source.length ? '; set the ' + e.bound_source.join('/') + ' bound (hindsight)' : ''}]`;
  console.log('\n-- pre-launch checks vs outcome --');
  console.log(`true positives (FAILed and went bad): ${V.true_positive.length}`); V.true_positive.forEach(e => console.log('   ' + line(e)));
  console.log(`misses (no FAIL but went bad: the next checks to add): ${V.miss.length}`); V.miss.forEach(e => console.log('   ' + line(e)));
  console.log(`false alarms (FAILed but ran fine): ${V.false_alarm.length}`); V.false_alarm.forEach(e => console.log('   ' + line(e)));
  const tpm = V.true_positive.filter(e => e.matched.length);
  console.log(`cause-matched true positives (a FAILing check decides a labelled cause, data.causes): ${tpm.length} of ${V.true_positive.length}` + (tpm.length ? ' (' + tpm.map(e => e.id + ' ' + e.matched.join('/')).join(', ') + ')' : ''));
  // crash/went-bad cause taxonomy: launch-time verdict vs today's gate for the deciding checks
  const TX = ACC.taxonomy(rows, data.causes);
  out.taxonomy = TX;
  console.log('cause taxonomy (runs/labels.json x data.causes; then = the run\'s own checks, now = today\'s gate, hindsight where the term came from that crash):');
  TX.classes.forEach(c => console.log(`   ${c.cls.padEnd(19)} n ${c.n}  decided by ${c.decided_by.join('/') || 'NOTHING before launch'}  caught then ${c.caught_then} / now ${c.caught_now}  (` +
    TX.items.filter(x => x.cls === c.cls).map(x => x.id.replace(/^droopy-|^snoopy-|^odie-/, '') + (c.decided_by.length ? ' [' + x.basis + ': ' + x.then.join(', ') + '; now ' + x.now.join(', ') + ']' : '')).join(', ') + ')'));
  console.log(`   no pre-launch check (stay UNKNOWN causes): ${TX.undecided.join(', ') || 'none'}`);
  // what the replayed checks change: the verdicts on the launch captures alone vs with the replay
  const V0 = ACC.verdicts(rows.map(r => Object.assign({}, r, { checks: r.launch_checks || r.checks.filter(c => replay.indexOf(c.id) < 0) })), data.causes);
  const was = new Set(V0.miss.map(e => e.id)), fa0 = new Set(V0.false_alarm.map(e => e.id));
  out.replay = { checks: replay, misses_now_tp: V.true_positive.filter(e => was.has(e.id)).map(e => ({ id: e.id, matched: e.matched })), new_false_alarms: V.false_alarm.filter(e => !fa0.has(e.id)).map(e => e.id) };
  console.log(`replayed ${replay.join('/')}: misses that become true positives ${out.replay.misses_now_tp.length}` + (out.replay.misses_now_tp.length ? ' (' + out.replay.misses_now_tp.map(e => e.id + (e.matched.length ? ' cause-matched' : ' other cause')).join(', ') + ')' : '') +
    `; new false alarms ${out.replay.new_false_alarms.length}` + (out.replay.new_false_alarms.length ? ' (' + out.replay.new_false_alarms.join(', ') + ')' : ''));
  out.per_check = ACC.perCheck(rows);
  console.log('per check (FAIL & went bad / FAIL & ran fine / passed & went bad):');
  out.per_check.forEach(x => console.log(`   ${x.id.padEnd(9)} ${x.fail_bad.length} / ${x.fail_fine.length} / ${x.pass_bad.length}` + (x.fail_fine.length ? `  false alarms: ${x.fail_fine.join(', ')}` : '')));
  console.log(`clean passes (no FAIL, ran fine): ${V.clean_pass.length}` + (V.clean_pass.length ? ' (' + V.clean_pass.map(e => e.id + (e.unknown.length ? ' UNKNOWN ' + e.unknown.join('/') : '')).join(', ') + ')' : ''));
  // trend: error and gate outcome per time bucket (--bucket day|week; default data.accuracy.trend_bucket)
  const bi = args.indexOf('--bucket'), mode = bi >= 0 ? args[bi + 1] : data.accuracy.trend_bucket.v;
  const T = ACC.trend(rows, r => r.box + '/' + r.model + '/' + r.engine, mode, METRICS);
  const TL = ACC.trend(rows.filter(r => r.pred_basis === 'launch'), r => r.box + '/' + r.model + '/' + r.engine, mode, METRICS);
  out.trend = { all: T, launch_only: TL };
  const cellOf = s => s ? `n ${s.n} ${s.median_abs_pct.toFixed(1)}% (${(s.bias_pct >= 0 ? '+' : '') + s.bias_pct.toFixed(1)})` : '-';
  const ptsOf = x => x.points.map(p => `${p.bucket} ${p.y.toFixed(1)}%` + (p.n !== undefined ? ` n${p.n}` : '')).join(' -> ');
  console.log(`\n-- error trend per ${mode} (median |err| % (bias) of scored runs; gate counts; is the gap shrinking?) --`);
  console.log('bucket'.padEnd(14) + 'runs launch | ' + METRICS.map(k => k.padEnd(24)).join(' ') + ' | gate TP/miss/FA/clean');
  T.buckets.forEach(b => console.log(b.bucket.padEnd(14) + String(b.runs).padStart(4) + String(b.launch).padStart(7) + ' | ' + METRICS.map(k => cellOf(b.metrics[k]).padEnd(24)).join(' ') + ` | ${b.gate.tp}/${b.gate.miss}/${b.gate.false_alarm}/${b.gate.clean}`));
  [['all rows', T], ['launch-time predictions only (post-hoc rows are today\'s planner on old runs)', TL]].forEach(([lab, X]) => {
    console.log(`verdict, ${lab}:`);
    X.overall.filter(o => o.points.length).forEach(o => console.log(`   ${o.metric.padEnd(19)} ${o.verdict}: ${ptsOf(o)}`));
    console.log(`   ${'gate miss share'.padEnd(19)} ${X.gate.verdict}: ${ptsOf(X.gate)}`);
  });
  console.log('per box / model / engine (all rows):');
  T.groups.forEach(g => console.log(`   ${g.metric.padEnd(19)} ${g.key.padEnd(22)} ${g.verdict}: ` + g.points.map(p => `${p.bucket} n${p.n} ${p.y.toFixed(1)}% (${(p.bias >= 0 ? '+' : '') + p.bias.toFixed(1)})`).join(' -> ')));
  if (args.includes('--json')) { const j = args.indexOf('--json'); const p = args[j + 1] && !args[j + 1].startsWith('--') ? args[j + 1] : null; if (p) fs.writeFileSync(p, JSON.stringify(out, null, 1)); else fs.writeSync(1, JSON.stringify(out, null, 1) + '\n'); }
  process.exit(0);
}

// ---------- layer map, route-trim proposals and the upgrade advisor (src/advisor.js) ----------
const pct = x => (x >= 0 ? '+' : '') + (100 * x).toFixed(1) + ' %';
const rng = (a, f) => a[0].toFixed(f) + (Math.abs(a[1] - a[0]) > 1e-9 ? '-' + a[1].toFixed(f) : '');
function advice(P) {
  const ADV = require('./src/advisor.js');
  const A = ADV.all(data, ADV.ledger(path.join(__dirname, 'runs', 'runs.jsonl')));
  P('\n== Layer map (per-layer sensitivity; PREDICTED quality, calibrated on our KL points) ==');
  A.layer_maps.forEach(m => {
    P(`${data.models[m.model].short}: ${m.status}${m.why ? ' (' + m.why + ')' : ''}, dof ${m.dof}` + (m.alpha !== undefined && m.alpha !== null ? `, CLAIMED scale ${m.alpha.toFixed(3)}` : '') + (m.residual_max ? `, largest residual KL ${m.residual_max.kl.toFixed(4)} / top-1 ${m.residual_max.top1.toFixed(4)}` : ''));
    m.bands.forEach(b => P(`   layers ${b.layers.join('-')}: ${b.fit ? 'fit ' + (b.value ? b.value.kl_per_layer.toFixed(5) + ' KL/layer in gate units' : 'UNKNOWN') : b.value} ${b.provenance}${b.url ? ' ' + b.url : ''}`));
    if (m.unknown_layers.length) P(`   UNKNOWN layers (no band): ${m.unknown_layers.join(', ')}`);
    (m.points || []).forEach(x => P(`   point ${x.run}: 2-bit share ${x.share.toFixed(3)}, map-weighted ${x.eff.toFixed(3)}; MEASURED KL ${x.kl.toFixed(4)} top-1 ${x.top1.toFixed(3)} -> PREDICTED ${x.pred_kl.toFixed(4)} / ${x.pred_top1.toFixed(3)}`));
    (m.claims || []).forEach(x => P(`   claim ${x.what}: CLAIMED KL ${x.claimed_kl} -> map ${x.pred_kl === null ? 'UNKNOWN' : x.pred_kl.toFixed(4)}`));
  });
  P('\n== Route-trim proposals (layer map, least sensitive layers first) ==');
  if (!A.trim.length) P('(none)');
  A.trim.forEach(t => {
    if (t.status !== 'proposed') { P(`${t.box} ${t.model} ${t.base_run}: ${t.status} (${t.why})`); return; }
    const x = t.predicted;
    P(`${t.box} ${t.model} on ${t.base_run} (measured ${t.base_measured.toFixed(2)}): ` + t.trim.map(b => `keep ${b.keep} on ${b.layers[0] === b.layers[1] ? b.layers[0] : b.layers.join('-')}`).join(', '));
    P(`   PREDICTED decode ${x.decode_planner_before.toFixed(2)} -> ${x.decode_planner.toFixed(2)} (${pct(x.gain)}, ~${x.decode_scaled.toFixed(1)} scaled to measured), KL ${x.kl.toFixed(4)}, top-1 ${x.top1.toFixed(3)} (budget KL <= ${t.budget.kl.toFixed(4)}, top-1 >= ${t.budget.top1.toFixed(3)}); quality check ${t.quality_check ? t.quality_check.status : 'none'}; engine ${t.engine_support.status}: ${t.engine_support.src}`);
  });
  const U = A.upgrades;
  P(`\n== Upgrade advisor (binding lane of each box's best quality-passing run; suggest at >= ${(100 * U.settings.suggest_min_gain).toFixed(0)} %) ==`);
  Object.values(U.boxes).forEach(b => {
    if (!b.models.length) { P(`${b.box}: no quality-passing predicted run`); return; }
    b.models.forEach(m => {
      P(`${b.box} ${m.model} ${m.run} (measured ${m.measured_tok_s.toFixed(2)}, planner ${m.predicted_tok_s.toFixed(2)}): binding ${m.binding}; ${m.probe_scale}x probes ` + m.lanes.filter(l => l.gain > 1e-3).map(l => `${l.lane} ${pct(l.gain)}`).join(', '));
      m.parts.forEach(x => P(`   part ${x.id}: ` + (x.status !== 'PREDICTED' ? x.status + ' ' + x.why : `${rng(x.gain.map(g => 100 * g), 1)} % -> ${rng(x.tok_s.scaled_after, 2)} tok/s scaled` + (x.price_usd ? `, $${rng(x.price_usd, 0)} (${x.price_date})` : '') + (x.usd_per_tok_s ? `, $${x.usd_per_tok_s[0].toFixed(0)}-${x.usd_per_tok_s[1] === null ? '?' : x.usd_per_tok_s[1].toFixed(0)} per tok/s` : '') + (x.suggested ? ' SUGGESTED' : ' not suggested: ' + x.why_not.join('; ')) + (x.suggested && x.caveat ? ' | caveat: ' + x.caveat : ''))));
    });
    b.unpriced.forEach(u => P(`   UNPRICED binding lane ${u.lane} ${pct(u.probe_gain)} at ${U.settings.probe_scale}x (${u.model}): no catalog part`));
  });
  P(U.suggestions.length ? 'SUGGESTED: ' + U.suggestions.map(x => `${x.box} ${x.id} ${rng(x.gain.map(g => 100 * g), 1)} %`).join('; ') : 'SUGGESTED: none');
  return A;
}
if (args.includes('--advice')) {
  const A = advice(args.includes('--json') ? () => {} : s => console.log(s));
  if (args.includes('--json')) fs.writeSync(1, JSON.stringify({ layer_map: A.layer_maps, trim_proposals: A.trim, upgrades: A.upgrades }, null, 1) + '\n');
  process.exit(0);
}

function engines(bk, mk, kn) {
  const box = data.boxes[bk];
  const gpus = box.gpus.slice();
  if (kn && kn.moveB70 === bk) gpus.push(data.b70move.gpu);
  const set = new Set();
  gpus.forEach(g => (g.engines || []).forEach(e => set.add(e)));
  return [...set].filter(e => !data.engines[e].no_solver && data.engines[e].types.some(q => !data.types[q].kld || v(data.types[q].kld[data.models[mk].kld_family]) !== null));
}
function run(bk, mk, eng, kn, targets) {
  const ctx = FM.prepare(FM.context(data, bk, mk, eng, kn), data.models[mk].heat.plan_default, data.models[mk].heat.score_default);
  const pts = FM.frontier(ctx);
  const out = { bk, mk, eng, maxTok: Math.max(...pts.filter(p => p.feasible).map(p => p.tokps)), at: {} };
  targets.forEach(T => {
    const r = FM.bestAt(ctx, T, pts);
    out.at[T] = r ? summarize(ctx, r.ev, r.lay) : null;
  });
  out.ctx = ctx;
  return out;
}
function summarize(ctx, ev, lay) {
  const tiers = ctx.tiers.map((t, i) => ({ id: t.id, gib: +(ev.capUse[i] / GiB).toFixed(1), cap: +(t.cap / GiB).toFixed(1), bpw: +ev.avgBpw[i].toFixed(2), heat: +(ev.heatT[i] / ctx.kL * 100).toFixed(1) }));
  let bits = 0, n = 0; for (let e = 0; e < lay.type.length; e++) { bits += v(ctx.data.types[ctx.types[lay.type[e]]].bpw); n++; }
  const checks = FM.checksFor(ctx, lay, ev).map(c => ({ id: c.id, status: c.status, detail: c.detail }));
  return { tokps: +ev.tokps.toFixed(2), quality: +ev.quality.toFixed(2), kld: +ev.kld.toFixed(4), avgBpw: +(bits / n).toFixed(2), lowbit_share: +FM.lowShare(ctx, lay).toFixed(3), tiers, checks };
}
const fmtChecks = a => a && a.checks ? a.checks.map(c => c.id + ' ' + c.status).join(', ') + (a.checks.some(c => c.status !== 'PASS') ? ' | ' + a.checks.filter(c => c.status !== 'PASS').map(c => c.id + ': ' + c.detail).join(' / ') : '') : '';
const fmtAt = a => a ? `${a.tokps.toFixed(1)} tok/s, top-1 ${a.quality.toFixed(1)} %, KLD ${a.kld.toFixed(3)}, ${a.avgBpw} bpw avg, 2-bit reads ${(100 * a.lowbit_share).toFixed(0)} % | ` + a.tiers.filter(t => t.gib > 0).map(t => `${t.id} ${t.gib}G@${t.bpw}b ${t.heat}%`).join(', ') : 'not reachable';

const base = Object.assign({}, data.knobs);
const lines = [];
const P = s => { lines.push(s); console.log(s); };
const result = { built: data.meta.built, best: [], residuals: [], combos: [], owner: null, recipes: [], frontier: [], layer_map: [], trim_proposals: [], upgrades: null };

P('== Calibration (measured -> predicted) ==');
Object.keys(data.cal.boxes).forEach(bk => {
  const c = data.cal.boxes[bk];
  P(`${bk}: CPU efficiency f ${v(c.f).toFixed(3)} (${c.f.tag})` + (c.pmem_bw ? `, PMem rate ${v(c.pmem_bw).toFixed(1)} GB/s FIT` : '') + `, host ms ` + Object.keys(c.host).map(m => `${m} ${v(c.host[m]).toFixed(1)} ${c.host[m].tag}`).join(', ') + (c.split_ms ? `, split GGUF +${v(c.split_ms).toFixed(1)} ms ${c.split_ms.tag}` : '') + (c.draft && Object.keys(c.draft).length ? ', draft x ' + Object.keys(c.draft).map(m => m + ' ' + v(c.draft[m]).toFixed(3)).join(' ') : '') + (c.rms !== undefined ? `, rms log err ${c.rms.toFixed(3)}` : ''));
});
data.cal.rows.forEach(r => { P(`  ${r.id.padEnd(16)} ${r.fit.padEnd(6)} ${r.meas.toFixed(2).padStart(6)} -> ${r.pred.toFixed(2).padStart(6)}  ${(100 * r.resid).toFixed(1).padStart(6)} %  ${r.label}`); result.residuals.push({ id: r.id, fit: r.fit, meas: r.meas, pred: +r.pred.toFixed(3), resid_pct: +(100 * r.resid).toFixed(2), label: r.label }); });

P('\n== Best projected point at 20 and 25 tok/s (default knobs; draft on; host copies kept) ==');
for (const mode of [false, true]) {
  P(mode ? '-- per-expert types (needs the split-expert build) --' : '-- GGUF today: one type per layer tensor, tier per expert --');
  for (const mk of ['v41', 'glm', 'fnx', 'fnxc']) for (const bk of Object.keys(data.boxes)) for (const eng of engines(bk, mk, base)) {
    const kn = Object.assign({}, base, { perExpertTypes: mode });
    const tt = Date.now();
    const r = run(bk, mk, eng, kn, [20, 25]);
    P(`${data.models[mk].short} ${bk} ${eng}: max ${r.maxTok.toFixed(1)} tok/s (${Date.now() - tt} ms)`);
    P(`   @20: ${fmtAt(r.at[20])}`);
    if (r.at[20]) P(`        checks: ${fmtChecks(r.at[20])}`);
    P(`   @25: ${fmtAt(r.at[25])}`);
    if (r.at[25]) P(`        checks: ${fmtChecks(r.at[25])}`);
    if (r.ctx.notes.length) P('   notes: ' + r.ctx.notes.join(' '));
    result.best.push({ model: mk, box: bk, engine: eng, perExpertTypes: mode, max_tokps: +r.maxTok.toFixed(2), at20: r.at[20], at25: r.at[25] });
  }
}

// today's runs and the queued ones against the hard checks
P('\n== Hard checks on 10-17 runs (storage / VRAM peak / -rtr / 2-bit quality gate) ==');
FM.checkRecipes(data).forEach(r => {
  P(`${String(r.when).padEnd(8)} ${r.box}/${r.model} ${r.label}: ${r.outcome}${r.wasted ? ' [cost ' + r.wasted + ']' : ''}`);
  P('   ' + r.checks.map(c => c.id + ' ' + c.status).join(', '));
  r.checks.filter(c => c.status !== 'PASS').forEach(c => P(`     ${c.id}: ${c.detail}`));
  result.recipes.push({ id: r.id, label: r.label, when: r.when, outcome: r.outcome, wasted: r.wasted, checks: r.checks.map(c => ({ id: c.id, status: c.status, detail: c.detail })) });
});

// frontier per box: best quality at each speed target (default knobs; snoopy also with the desktop off the GPU)
P('\n== Frontier per box: best projected top-1 at each speed (all four checks evaluated) ==');
const TG = [6, 8, 10, 15, 20, 25];
for (const bk of Object.keys(data.boxes)) for (const [tag, extra] of [['default knobs', {}], ['desktop off the GPU', { freeDesktop: true }], ['both B70s (the resident service on B70 #2 off)', { freeHousehold: true }]]) {
  if (tag === 'desktop off the GPU' && bk !== 'snoopy') continue;
  if (tag.startsWith('both B70s') && bk !== 'odie') continue;
  for (const mk of ['v41', 'glm', 'fnx', 'fnxc']) for (const pe of [false, true]) {
    const kn = Object.assign({}, base, extra, { perExpertTypes: pe });
    for (const eng of engines(bk, mk, kn)) {
      let r;
      try { r = run(bk, mk, eng, kn, TG); } catch (e) { P(`${bk} ${data.models[mk].short} ${eng}: error ${e.message}`); continue; }
      const name = `${bk} ${data.models[mk].short} ${eng}${pe ? ' per-expert' : ''} (${tag})`;
      if (!isFinite(r.maxTok)) { P(`${name}: no feasible layout`); continue; }
      const cells = TG.filter(T => r.at[T]).map(T => { const a = r.at[T], bad = a.checks.filter(c => c.status !== 'PASS'); return `${T}: ${a.quality.toFixed(1)} % (${a.avgBpw} bpw${bad.length ? ', ' + bad.map(c => c.id + ' ' + c.status).join('/') : ''})`; });
      P(`${name}: max ${r.maxTok.toFixed(1)} tok/s | ` + (cells.join(' · ') || 'none of the targets'));
      result.frontier.push({ box: bk, model: mk, engine: eng, perExpertTypes: pe, knobs: tag, max_tokps: +r.maxTok.toFixed(2), at: Object.fromEntries(TG.filter(T => r.at[T]).map(T => [T, r.at[T]])) });
    }
  }
}

// owner's droopy layout (needs the alloc3 coding set; a build without that input skips it)
if (data.alloc3) {
  const kn = Object.assign({}, base, { perExpertTypes: true });
  const ctx = FM.prepare(FM.context(data, 'droopy', 'v41', 'ik', kn), 'coding_odd', 'coding_even');
  P('\n== Owner layout on droopy (V4.1: hottest coding experts MXFP4 in VRAM, other coding experts in DRAM, the rest MXFP4 in PMem) ==');
  result.owner = {};
  [[false, 'coding experts in DRAM at q3_K'], [true, 'same, leftover DRAM spent on 4-bit']].forEach(([up, label]) => {
    const lay = FM.ownerLayout(ctx, up);
    const s = summarize(ctx, FM.evaluate(ctx, lay), lay);
    P(`   ${label}: ` + fmtAt(s));
    result.owner[up ? 'upgraded' : 'as_stated'] = s;
  });
}

{
  const A = advice(P);
  Object.assign(result, { layer_map: A.layer_maps, trim_proposals: A.trim, upgrades: A.upgrades });
}

if (!quick) {
  P('\n== Knob combinations projecting 20+ (best quality at 20 tok/s) ==');
  const opts = {
    freeDesktop: [false, true], moveB70: ['none', 'snoopy', 'droopy'], addDRAM: [false, true], addOptane: [false, true],
    kernel2x: [false, true], smt: [false, true], perExpertTypes: [false, true]
  };
  const cost = { freeDesktop: 'snoopy desktop off the 3090 Ti (display off, or the desktop on the iGPU)', moveB70: (x, eng) => `move B70 #1 from odie into ${x}` + (eng === 'ik' ? ' + a SYCL/XPU expert worker beside ik (code)' : eng === 'sycl' ? ' (mainline SYCL drives it alone; V4.1 support in mainline UNKNOWN)' : ''), addDRAM: 'buy DRAM', addOptane: 'buy 4 Optane PMem 100 sticks (droopy)', kernel2x: '2x faster CPU expert kernel (code: CLX/AVX-512 kernel work)', smt: 'SMT threads on', perExpertTypes: 'split-expert build (per-expert types, lab notes)' };
  const keys = Object.keys(opts);
  const combos = [];
  const rec = (i, cur) => {
    if (i === keys.length) { combos.push(Object.assign({}, cur)); return; }
    opts[keys[i]].forEach(x => { cur[keys[i]] = x; rec(i + 1, cur); });
  };
  rec(0, {});
  const found = [];
  for (const mk of ['v41', 'glm']) for (const bk of Object.keys(data.boxes)) {
    for (const c of combos) {
      // skip knobs that do nothing on this box
      if (c.freeDesktop && bk !== 'snoopy') continue;
      if (c.addOptane && !(data.boxes[bk].upgrades.pmem)) continue;
      if (c.addDRAM && !(data.boxes[bk].upgrades.dram)) continue;
      if (c.moveB70 !== 'none' && c.moveB70 !== bk) continue;
      if (c.smt && v(data.boxes[bk].cpu.smt_mult) === 1) continue;
      const kn = Object.assign({}, base, c);
      for (const eng of engines(bk, mk, kn)) {
        let r;
        try { r = run(bk, mk, eng, kn, [20]); } catch (e) { continue; }
        if (!r.at[20]) continue;
        const on = keys.filter(k => c[k] && c[k] !== 'none' && !(k === 'perExpertTypes' && !c[k]));
        found.push({ model: mk, box: bk, engine: eng, knobs: on.map(k => k === 'moveB70' ? 'moveB70:' + c[k] : k), n: on.length, takes: on.map(k => typeof cost[k] === 'function' ? cost[k](c[k], eng) : cost[k]), at20: r.at[20] });
      }
    }
  }
  // per model x box: the cheapest combo (fewest knobs) reaching 20, and the best quality
  found.sort((a, b) => b.at20.quality - a.at20.quality);
  const seen = new Set();
  found.forEach(f => {
    const key = f.model + f.box + f.engine + f.knobs.join();
    if (seen.has(key)) return; seen.add(key);
    result.combos.push(f);
  });
  P(`(${found.length} feasible combos)`);
  ['v41', 'glm'].forEach(mk => {
    P(`-- ${data.models[mk].short}: top by quality --`);
    result.combos.filter(f => f.model === mk).slice(0, 6).forEach(f => P(`   ${f.box}/${f.engine} [${f.knobs.join(', ') || 'no changes'}]: ${fmtAt(f.at20)}`));
    P(`-- ${data.models[mk].short}: fewest changes --`);
    result.combos.filter(f => f.model === mk).slice().sort((a, b) => a.n - b.n || b.at20.quality - a.at20.quality).slice(0, 6).forEach(f => P(`   ${f.box}/${f.engine} [${f.knobs.join(', ') || 'no changes'}]: ${fmtAt(f.at20)}`));
  });
}
P(`\n(total ${Date.now() - t0} ms)`);
const ji = args.indexOf('--json');
if (ji >= 0) fs.writeFileSync(args[ji + 1], JSON.stringify(result, null, 1));
