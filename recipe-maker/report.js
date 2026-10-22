#!/usr/bin/env node
// Node report: best projected point at 20 and 25 tok/s per model x box x engine, calibration residuals,
// owner's droopy layout, and a knob-combination search. Usage: node report.js [--json out.json] [--quick]
// Pre-launch check: node report.js --check recipe.json [--json] [--strict]. Run ledger accuracy: node report.js --accuracy [runs.jsonl] [--json] [--bucket day|week]
// External validation (community rows, runs/external.jsonl): node report.js --external [--json]
// Screening rounds against the card conditions (runs/screen.jsonl, data/cards): node report.js --screen [--json]
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
// exit 0 = no FAIL (UNKNOWN allowed unless --strict), 1 = a hard check FAILs (refuse the launch), 2 = bad input, 3 = UNKNOWN with --strict
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
    gaps.push(...res.checks.filter(c => c.status === 'UNKNOWN').map(c => c.id),
      ...(!pr || pr.status !== 'PREDICTED' ? ['speed UNMODELLED'] : Number.isFinite(pr.prefill_tps) ? [] : ['prefill UNMODELLED']));
    const s = res.bad_input ? 2 : res.checks.some(c => c.status === 'FAIL') ? 1 : (args.includes('--strict') && res.checks.some(c => c.status === 'UNKNOWN')) ? 3 : 0;
    if (s === 2 || (s === 1 && worst !== 2) || (s === 3 && worst === 0)) worst = s;
  }
  // writeSync: console.log on a pipe is asynchronous and process.exit() would cut the JSON at 64 KiB
  if (args.includes('--json')) fs.writeSync(1, JSON.stringify(outAll.length === 1 ? outAll[0] : outAll, null, 1) + '\n');
  else console.log(`verdict: ${['GO (no FAIL)', 'REFUSE (a hard check FAILs)', 'BAD INPUT', 'HOLD (UNKNOWN with --strict)'][worst]}` +
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
