// Run-ledger accuracy: predicted vs measured per run, error summaries per box/model/engine, and how the
// pre-launch checks did against what each run actually did. Pure functions; report.js --accuracy drives them.
'use strict';

const median = xs => {
  if (!xs.length) return null;
  const s = xs.slice().sort((a, b) => a - b), h = s.length >> 1;
  return s.length % 2 ? s[h] : (s[h - 1] + s[h]) / 2;
};
const num = x => typeof x === 'number' && isFinite(x);
const f2 = x => num(x) ? x.toFixed(2) : '-';

// the confirmed causes of a run's label (runs/labels.json); confirmed false = a suspicion in the notes, never counted
// a label with an effect note ('none visible', 'load only') names a contact that harmed nothing, so it is not a cause either
const causesOf = lab => ((lab || {}).causes || []).filter(c => c.confirmed !== false && !c.effect);
// the decode phases a confirmed cause disturbed (interference, or a slowdown the notes leave unexplained)
const disturbed = lab => [...new Set(causesOf(lab).reduce((a, c) => a.concat(c.phases || []), []))];
// a run went bad when it did not finish cleanly, finished degraded (memory pressure, placement short, ...), or a label
// names a cause in its timed phases
function wentBad(run, lab) {
  const m = run.measured || {};
  const why = [];
  if (m.outcome && m.outcome !== 'ok') why.push(m.outcome);
  (m.degraded || []).forEach(d => why.push(d));
  const cs = causesOf(lab).filter(c => (c.phases || []).length);
  if (cs.length && !why.length) why.push(cs.map(c => c.class + ' in ' + c.phases.join('/')).join(', ') + ' (runs/labels.json)');
  return why;
}

// the measured decode phase a prediction pairs with: the draft prediction with the primary draft phase, the no-draft one with n0.
// Kits that run one workload per phase group put the width inside the name (std-n4 / std-n0, d600_m3_mean / d600_n0_mean):
// such a phase reads like its bare width
const DRAFT_RX = /(?:^|[-_])(?:n4|m3)(?:b)?(?=$|[-_])/, OFF_RX = /(?:^|[-_])[mn]0(?=$|[-_])/;
function primaryPhase(run) {
  const m = run.measured || {}, d = m.decode_tps || {}, ks = Object.keys(d);
  if (m.primary_phase && d[m.primary_phase] !== undefined) return m.primary_phase;
  const draft = run.recipe && run.recipe.draft;
  for (const k of draft ? ['n4', 'c1', 'all'] : ['n0', 'c1', 'all']) if (d[k] !== undefined) return k;
  return ks.find(k => (draft ? DRAFT_RX : OFF_RX).test(k)) || ks[0] || null;
}
// a no-draft phase: n0 / m0, bare or inside a workload group name
const isOff = k => NODRAFT.indexOf(k) >= 0 || OFF_RX.test(k);
// the no-draft twin of a phase: n0 / m0, or the same workload group with its width set to n0 / m0 (std-n4 -> std-n0)
function offOf(d, ph) {
  const bare = NODRAFT.find(k => d[k] !== undefined);
  if (bare) return bare;
  const m = ph && /(^|[-_])(n4|m3)b?(?=$|[-_])/.exec(ph);
  return m ? ['n0', 'm0'].map(o => ph.slice(0, m.index + m[1].length) + o + ph.slice(m.index + m[0].length)).find(k => d[k] !== undefined) || null : null;
}

// drafts per request the run declared (recipe.conditions.draft_n: a number, or a string led by one); null = not declared
function draftN(run) {
  const x = ((run.recipe || {}).conditions || {}).draft_n;
  return typeof x === 'number' ? x : typeof x === 'string' && /^\s*\d+/.test(x) ? parseInt(x, 10) : null;
}
// the measured no-draft phase names of a draft run: V4.1 kits n0, GLM be1 kits m0 (MTP off)
const NODRAFT = ['n0', 'm0'];

// predicted vs measured pairs for one run; pct = (pred - meas) / meas. A draft recipe whose requests declared draft_n 0
// (q4l/q4i/q4j/q4t: the server loads MTP, every request turns it off) measured no-draft decode: it pairs with the no-draft
// prediction, never with the draft speed-up
function pairs(run, pred) {
  const m = run.measured || {}, out = [];
  const add = (metric, phase, p, x, pctOk) => {
    if (!num(x)) return;
    out.push({ metric, phase, pred: num(p) ? p : null, meas: x, err: num(p) ? p - x : null, pct: num(p) && pctOk && x !== 0 ? (p - x) / x : null });
  };
  const d = m.decode_tps || {}, ph = primaryPhase(run), draft = !!(run.recipe && run.recipe.draft), off = draft && (draftN(run) === 0 || isOff(ph || ''));
  if (ph) add(off ? 'decode_tps_nodraft' : 'decode_tps', ph, pred && (off ? pred.decode_tps_nodraft : pred.decode_tps), d[ph], true);
  const nd = offOf(d, ph);
  if (draft && !off && nd && ph !== nd) add('decode_tps_nodraft', nd, pred && pred.decode_tps_nodraft, d[nd], true);
  add('prefill_tps', null, pred && pred.prefill_tps, m.prefill_tps, true);
  add('vram_post_gib', null, pred && pred.vram_post_gib, m.vram_post_gib, true);
  add('node_memfree_min_gib', null, pred && pred.node_memfree_min_gib, m.node_memfree_min_gib, false);
  return out;
}

// the draft speed-up each run measured inside one load: primary draft phase / no-draft phase, against the planner's draft
// multiplier (pred decode_tps / decode_tps_nodraft). One load holds the layout fixed, so a run that went bad for a layout
// reason still measures it; a run whose drafts were off, or with either phase disturbed (labels), does not
function draftRatios(rows, keyOf) {
  const g = {};
  rows.forEach(r => {
    const d = (r.run.measured || {}).decode_tps || {}, ph = primaryPhase(r.run), nd = offOf(d, ph);
    if (!r.run.recipe.draft || draftN(r.run) === 0 || !ph || !nd || ph === nd || !num(d[ph]) || !num(d[nd]) || [ph, nd].some(k => r.disturbed.indexOf(k) >= 0)) return;
    const p = r.pred && num(r.pred.decode_tps) && num(r.pred.decode_tps_nodraft) ? r.pred.decode_tps / r.pred.decode_tps_nodraft : null;
    (g[keyOf(r)] = g[keyOf(r)] || []).push({ id: r.id, meas: d[ph] / d[nd], pred: p });
  });
  return Object.keys(g).sort().map(k => ({ key: k, n: g[k].length, median: median(g[k].map(x => x.meas)), pred: median(g[k].filter(x => x.pred !== null).map(x => x.pred)), runs: g[k] }));
}

// per group: n, median |pct|, bias (median signed pct), for scored pairs of one metric
function summarize(rows, keyOf, metric) {
  const g = {};
  rows.forEach(r => r.pairs.filter(p => p.metric === metric && p.pct !== null && r.scored).forEach(p => (g[keyOf(r)] = g[keyOf(r)] || []).push(p.pct)));
  return Object.keys(g).sort().map(k => ({ key: k, n: g[k].length, median_abs_pct: 100 * median(g[k].map(Math.abs)), bias_pct: 100 * median(g[k]) }));
}

// pre-launch check verdicts vs outcome. FAIL = the gate would refuse. hold = UNKNOWN only (--strict holds it).
// causes (data.causes) given: a true positive is cause-matched when a FAILing check decides one of the run's labelled causes
function verdicts(rows, causes) {
  const out = { true_positive: [], miss: [], false_alarm: [], clean_pass: [] };
  rows.forEach(r => {
    if (!r.checks) return;
    const fail = r.checks.filter(c => c.status === 'FAIL').map(c => c.id);
    const unk = r.checks.filter(c => c.status === 'UNKNOWN').map(c => c.id);
    const bad = r.bad.length > 0;
    const cls = r.causes || [], by = cls.reduce((a, k) => a.concat(((causes || {})[k] || {}).decided_by || []), []);
    const e = { id: r.id, fail, unknown: unk, bad: r.bad, basis: r.check_basis, bound_source: r.bound_source, causes: cls,
      matched: fail.filter(x => by.indexOf(x) >= 0), replayed: r.checks.filter(c => c.replayed).map(c => c.id) };
    (fail.length ? (bad ? out.true_positive : out.false_alarm) : (bad ? out.miss : out.clean_pass)).push(e);
  });
  return out;
}

// per check: how often its FAIL met a run that went bad, a run that ran fine, and how many bad runs it let pass
function perCheck(rows) {
  const t = {};
  rows.forEach(r => (r.checks || []).forEach(c => {
    const x = t[c.id] = t[c.id] || { id: c.id, fail_bad: [], fail_fine: [], pass_bad: [], unknown: [] };
    const bad = r.bad.length > 0;
    if (c.status === 'FAIL') (bad ? x.fail_bad : x.fail_fine).push(r.id);
    else if (bad) x.pass_bad.push(r.id);
    if (c.status === 'UNKNOWN') x.unknown.push(r.id);
  }));
  return Object.values(t);
}

// time bucket of a run: 'day' = its UTC launch day (one launch batch on this ledger), 'week' = the Monday of its ISO week
function bucketOf(when, mode) {
  const d = String(when || '').slice(0, 10);
  if (mode !== 'week' || !/^\d{4}-\d\d-\d\d$/.test(d)) return d || '?';
  const t = new Date(d + 'T00:00:00Z');
  return 'wk ' + new Date(t.getTime() - ((t.getUTCDay() + 6) % 7) * 864e5).toISOString().slice(0, 10);
}
// least-squares slope of ys over their order
function slope(ys) {
  const n = ys.length, mx = (n - 1) / 2, my = ys.reduce((a, y) => a + y, 0) / n;
  const num = ys.reduce((a, y, i) => a + (i - mx) * (y - my), 0), den = ys.reduce((a, y, i) => a + (i - mx) * (i - mx), 0);
  return den ? num / den : 0;
}
// the gap shrinks when the last bucket is below the first and the fitted slope over every bucket falls
function gapVerdict(pts) {
  if (pts.length < 2) return { verdict: 'too few buckets (' + pts.length + ')', slope: null };
  const ys = pts.map(p => p.y), s = slope(ys);
  return { verdict: s < 0 && ys[ys.length - 1] < ys[0] ? 'shrinking' : 'NOT shrinking', slope: s };
}
// error and gate outcome per time bucket, overall and per group; answers "is the gap between prediction and result shrinking?"
function trend(rows, keyOf, mode, metrics) {
  const by = {};
  rows.forEach(r => (by[bucketOf(r.when, mode)] = by[bucketOf(r.when, mode)] || []).push(r));
  const names = Object.keys(by).sort();
  const buckets = names.map(b => {
    const rs = by[b], V = verdicts(rs);
    return { bucket: b, runs: rs.length, launch: rs.filter(r => r.pred_basis === 'launch').length,
      metrics: Object.fromEntries(metrics.map(k => [k, summarize(rs, () => 'all', k)[0] || null])),
      gate: { tp: V.true_positive.length, miss: V.miss.length, false_alarm: V.false_alarm.length, clean: V.clean_pass.length } };
  });
  const keys = [...new Set(rows.map(keyOf))].sort();
  const groups = [];
  metrics.forEach(k => keys.forEach(g => {
    const pts = names.map(b => ({ bucket: b, s: summarize(by[b].filter(r => keyOf(r) === g), keyOf, k)[0] })).filter(p => p.s)
      .map(p => ({ bucket: p.bucket, n: p.s.n, y: p.s.median_abs_pct, bias: p.s.bias_pct }));
    if (pts.length) groups.push(Object.assign({ key: g, metric: k, points: pts }, gapVerdict(pts)));
  }));
  const overall = metrics.map(k => Object.assign({ metric: k, points: buckets.filter(b => b.metrics[k]).map(b => ({ bucket: b.bucket, n: b.metrics[k].n, y: b.metrics[k].median_abs_pct })) },
    gapVerdict(buckets.filter(b => b.metrics[k]).map(b => ({ y: b.metrics[k].median_abs_pct })))));
  // the gate's gap: the share of runs that went bad with no check failing
  const miss = buckets.filter(b => b.gate.tp + b.gate.miss).map(b => ({ bucket: b.bucket, n: b.gate.tp + b.gate.miss, y: 100 * b.gate.miss / (b.gate.tp + b.gate.miss) }));
  return { mode, buckets, groups, overall, gate: Object.assign({ metric: 'miss share of bad runs', points: miss }, gapVerdict(miss)) };
}

function fmtPrediction(p) {
  if (!p) return 'none';
  if (p.status !== 'PREDICTED') return 'UNMODELLED: ' + p.why;
  return 'decode ' + f2(p.decode_tps) + ' tok/s' + (num(p.decode_tps_nodraft) && Math.abs(p.decode_tps_nodraft - p.decode_tps) > 1e-9 ? ' (no draft ' + f2(p.decode_tps_nodraft) + ')' : '') +
    ', prefill ' + (num(p.prefill_tps) ? f2(p.prefill_tps) + ' tok/s at ' + ((p.prefill && p.prefill.tokens) || 2048) + ' tokens' + (p.prefill_tps_at ? ' (8k ' + f2(p.prefill_tps_at[8192]) + ', 16k ' + f2(p.prefill_tps_at[16384]) + ', 32k ' + f2(p.prefill_tps_at[32768]) + ')' : '') : 'UNMODELLED (' + String(p.prefill_why || '').replace(/^UNMODELLED: /, '') + '; do not trust a number here)') + ', VRAM post-warm-up ' + f2(p.vram_post_gib) + ' GiB' + (p.vram_peak_range_gib ? ', peak ' + f2(p.vram_peak_range_gib[0]) + '-' + f2(p.vram_peak_range_gib[1]) : '') +
    (num(p.node_memfree_min_gib) ? ', min node MemFree ' + f2(p.node_memfree_min_gib) + ' GiB' : '') +
    (p.tiers ? ' | ' + p.tiers.map(t => t.id + ' ' + t.experts_gib.toFixed(1) + 'G ' + (100 * t.read_share).toFixed(1) + '%').join(', ') : '') +
    (p.assumptions && p.assumptions.length ? ' | assumes: ' + p.assumptions.join('; ') : '');
}

// cause taxonomy: every labelled cause of a run that went bad, the checks that decide it (data.causes), their verdict on the
// run's own checks (launch capture or post hoc) and on today's gate. A today-only FAIL on a launch row is hindsight when the
// deciding term was measured from that crash (the catalog src says so); undecided = classes no pre-launch check covers
function taxonomy(rows, causes) {
  const st = (cks, ids) => ids.map(id => { const c = (cks || []).find(x => x.id === id); return id + ' ' + (c ? c.status : 'absent'); });
  const fails = (cks, ids) => ids.some(id => ((cks || []).find(x => x.id === id) || {}).status === 'FAIL');
  const items = [];
  rows.filter(r => r.bad.length).forEach(r => (r.causes || []).forEach(k => {
    const by = ((causes || {})[k] || {}).decided_by || [];
    items.push({ id: r.id, cls: k, decided_by: by, basis: r.check_basis, then: st(r.checks, by), now: st(r.today, by),
      caught_then: fails(r.checks, by), caught_now: fails(r.today, by) });
  }));
  const cls = [...new Set(items.map(x => x.cls))].sort();
  return { items, classes: cls.map(k => ({ cls: k, n: items.filter(x => x.cls === k).length, decided_by: ((causes || {})[k] || {}).decided_by || [],
    caught_then: items.filter(x => x.cls === k && x.caught_then).length, caught_now: items.filter(x => x.cls === k && x.caught_now).length })),
    undecided: cls.filter(k => !(((causes || {})[k] || {}).decided_by || []).length) };
}

// Frozen validation stays on the existing recipe -> --check -> record_run ledger path. A hash is an identity,
// not evidence that two experiments match. Never infer a decision cohort from a box, date or model name.
const FM = require('./model.js'), crypto = require('crypto');
const canonical = x => JSON.stringify(FM.networkCanonical(x));
const digest = x => crypto.createHash('sha256').update(canonical(x)).digest('hex');
const copy = x => JSON.parse(JSON.stringify(x));
const sha = x => typeof x === 'string' && /^[0-9a-f]{64}$/.test(x);
const same = (a, b) => canonical(a) === canonical(b);
const field = (x, p) => typeof p === 'string' && p.split('.').every(k => k && !['__proto__', 'constructor', 'prototype'].includes(k)) ?
  p.split('.').reduce((a, k) => a && Object.prototype.hasOwnProperty.call(a, k) ? a[k] : undefined, x) : undefined;
const sourced = (m, p) => [field(m.src, p), (m.src || {})[p], (m.src || {})[p.split('.')[0]]].some(x => typeof x === 'string' && x.trim());
const bareRecipe = r => Object.fromEntries(Object.entries(r || {}).filter(([k]) => k !== 'validation'));

// Use the actual evaluator for its fitted dLLM membership, rather than maintaining a second eligibility predicate.
function calibrationSplit(data) {
  const D = data.dllm, dt = FM.dllmTerms(data), ids = xs => [...new Set(xs)].sort(), fitted = [];
  const collect = x => {
    if (!x || typeof x !== 'object') return;
    if (Array.isArray(x.runs)) fitted.push(...x.runs);
    Object.values(x).forEach(collect);
  };
  // cal.rows also contains residual-only observations; only fitted term operands belong in fitted_run_ids.
  Object.entries(data.cal || {}).filter(([k]) => k !== 'rows').forEach(([, x]) => collect(x));
  return {
    shared: { anchors: Object.fromEntries((data.anchors || []).map(a => [a.id, { fit: a.fit || null, twin: a.twin || null }])), fitted_run_ids: ids(fitted) },
    observed_runs: ids((data.recipes || []).map(r => r.run_id).concat((D && D.rows || []).map(r => r.id)).filter(Boolean)),
    decode: Object.fromEntries(Object.entries(data.decode_fits || {}).map(([k, x]) => [k, { train: ids(x.train), holdout: ids(x.holdout) }])),
    dllm: { holdout: ids(D && D.holdout.v || []), efficiency_train: ids(dt.used),
      tpf_train: Object.fromEntries(Object.entries(dt.tpf).map(([k, x]) => [k, ids(x.runs)])),
      submitted_train: Object.fromEntries(Object.entries(dt.b).map(([k, x]) => [k, ids(x.runs)])) },
    window: Object.fromEntries(Object.entries(data.window && data.window.configs || {}).map(([k, W]) => [k, {
      fit: ids(W.points.filter(p => p.use === 'fit').map(p => p.id)), eta_train: ids(FM.winFit(data, k).eta_training),
      check: ids(W.points.filter(p => p.use === 'check').map(p => p.id)),
      law: ids(W.points.filter(p => p.use === 'law').map(p => p.id)),
      // Profile/anchor conditioning is not an untouched end-to-end check, even when its point has use: check.
      conditioning: ids([W.profile && W.profile.point, W.profile && W.profile.curve_anchor && W.profile.curve_anchor.v,
        W.settings.plan_anchor && W.settings.plan_anchor.v].filter(Boolean)),
      ledger: Object.fromEntries(W.points.filter(p => p.ledger).map(p => [p.id, p.ledger])) }]))
  };
}
function frozenValid(x) {
  return !!(x && x.status === 'FROZEN' && x.format === 'fit-calibration-v1' && sha(x.data_sha256) && x.evaluator_sha256 &&
    Object.keys(x.evaluator_sha256).length && Object.values(x.evaluator_sha256).every(sha) &&
    x.split && Array.isArray(x.split.observed_runs) && sha(x.split_sha256) && digest(x.split) === x.split_sha256);
}
const time = x => typeof x === 'string' && /Z$/.test(x) && Number.isFinite(Date.parse(x)) ? Date.parse(x) : null;
function cohortProblems(c) {
  if (!c || typeof c !== 'object' || Array.isArray(c)) return ['cohort is missing'];
  const out = [], w = c.workload || {}, o = c.objective || {}, cs = c.constraints;
  if (typeof c.id !== 'string' || !c.id || time(c.declared_at) === null) out.push('cohort id and UTC declaration time required');
  if (!frozenValid(c.frozen)) out.push('valid frozen built-data, evaluator and split identity required');
  if (!sha(w.model_sha256) || !sha(w.prompt_set_sha256) || !sha(w.quality_reference_sha256) || !sha(w.quality_prompt_set_sha256) ||
      !Number.isInteger(w.context_tokens) || w.context_tokens <= 0 || !Number.isInteger(w.agents) || w.agents <= 0 ||
      !w.sampling || typeof w.sampling !== 'object' || Array.isArray(w.sampling)) out.push('exact model/prompt and quality-reference/prompt hashes, context, agents and sampling required');
  if (!c.isolation || !c.isolation.id || !c.isolation.scope || !c.isolation.src) out.push('sourced isolation identity required');
  if (!['single-seat', 'batch-fleet'].includes(o.mode) || (o.mode === 'single-seat' &&
      (!/^decode_tps\.[^.]+$/.test(o.measured || '') || !['decode_tps', 'decode_tps_nodraft'].includes(o.predicted)))) out.push('named workload objective and per-stream decode metric mapping required');
  if (o.mode === 'batch-fleet' && (!o.power_scope || !Array.isArray(o.power_scope.participants) || !o.power_scope.participants.length ||
      !Array.isArray(o.power_scope.components) || !same(o.power_scope.components.slice().sort(), ['cpu', 'gpu']) || !o.power_scope.src)) out.push('batch-fleet needs a sourced all-participant CPU+GPU meter scope');
  if (!Array.isArray(cs) || cs.some(x => !x || typeof x !== 'object') || !['quality', 'memory'].every(k => cs.some(x => x.kind === k)) ||
      cs.some(x => typeof x.id !== 'string' || !x.id || !x.unit || !['quality', 'memory'].includes(x.kind) || typeof x.measured !== 'string' || !x.measured ||
        typeof x.predicted !== 'string' || !x.predicted || !['<=', '>='].includes(x.op) || !num(x.limit) || x.limit < 0) ||
      new Set(cs.map(x => x.id)).size !== cs.length) out.push('distinct finite quality and memory constraints required');
  if (!Array.isArray(c.candidates) || c.candidates.length < 2 || c.candidates.some(x => !x || typeof x !== 'object' ||
      typeof x.id !== 'string' || !x.id || typeof x.run_id !== 'string' || !x.run_id || !x.recipe || typeof x.recipe !== 'object' || x.recipe.validation) ||
      new Set(c.candidates.map(x => x.id)).size !== c.candidates.length || new Set(c.candidates.map(x => x.run_id)).size !== c.candidates.length) out.push('at least two distinct exact candidate recipes/run ids required');
  if (frozenValid(c.frozen) && Array.isArray(c.candidates) && c.candidates.some(x => x && c.frozen.split.observed_runs.includes(x.run_id))) out.push('candidate outcomes already occur in frozen data');
  if (Array.isArray(c.candidates) && c.candidates.every(x => x && x.recipe) && new Set(c.candidates.map(x => canonical(Object.fromEntries(Object.entries(x.recipe).filter(([k]) => !['id', 'label'].includes(k)))))).size !== c.candidates.length) out.push('replicas of one recipe are not distinct alternatives');
  return out;
}
function objectiveValue(x, o, measured) {
  if (o.mode === 'batch-fleet') return num(x.accepted_tps) && x.accepted_tps >= 0 && num(x.gpu_cpu_watts) && x.gpu_cpu_watts > 0 ? x.accepted_tps / x.gpu_cpu_watts : null;
  const n = field(x, measured ? o.measured : o.predicted);
  return num(n) && n >= 0 ? n : null;
}
const constraintPass = (x, cs, measured) => cs.every(c => c.op === '<=' ? field(x, measured ? c.measured : c.predicted) <= c.limit : field(x, measured ? c.measured : c.predicted) >= c.limit);
function captureDecision(data, recipe, predict, capturedAt) {
  const v = recipe.validation, c = v && v.cohort, bad = cohortProblems(c);
  if (bad.length) throw new Error('validation: ' + bad.join('; '));
  if (!same(c.frozen, data.meta.calibration)) throw new Error('validation: cohort is not the current frozen build/split');
  if (time(capturedAt) === null || time(c.declared_at) > time(capturedAt)) throw new Error('validation: declaration must precede capture');
  const member = c.candidates.find(x => x.id === v.candidate);
  if (!member || !same(member.recipe, bareRecipe(recipe))) throw new Error('validation: recipe is not the exact declared candidate');
  const defaults = data.accuracy.workloads[c.objective.mode];
  const objective = { ...defaults, ...c.objective,
    metric: c.objective.mode === 'batch-fleet' ? 'total_accepted_tps_per_gpu_cpu_watt' : 'per_stream_decode_tps',
    unit: c.objective.mode === 'batch-fleet' ? 'accepted tok/s/W' : 'tok/s', direction: 'max' };
  if (objective.mode === 'single-seat' && !(num(objective.decode_target_tps) && objective.decode_target_tps > 0) ||
      objective.mode === 'batch-fleet' && !(num(objective.per_agent_floor_tps) && objective.per_agent_floor_tps > 0)) throw new Error('validation: positive named workload target/floor required');
  if (objective.mode === 'batch-fleet') {
    const boxes = [...new Set(c.candidates.flatMap(x => {
      const network = (data.network && data.network.recipes || []).find(n => n.id === x.recipe.network_recipe);
      return network ? network.boxes : [x.recipe.box];
    }))].sort();
    if (!same(boxes, [...new Set(objective.power_scope.participants)].sort())) throw new Error('validation: meter scope must cover every participating box');
  }
  const predictions = c.candidates.map(x => ({ id: x.id, prediction: predict(copy(x.recipe)) }));
  const core = { cohort: c, objective, predictions };
  return copy({ ...core, cohort_sha256: digest(core), candidate: v.candidate, captured_at: capturedAt });
}

function decisions(rows) {
  const groups = new Map(), excluded = [];
  rows.forEach(r => {
    const v = r.pred && r.pred.validation;
    if (!v || !v.cohort || typeof v.cohort.id !== 'string' || !v.cohort.id) { excluded.push({ id: r.id, reason: 'NO_PROSPECTIVE_CAPTURE' }); return; }
    if (!groups.has(v.cohort.id)) groups.set(v.cohort.id, []);
    groups.get(v.cohort.id).push(r);
  });
  const cohorts = [...groups.entries()].map(([id, rs]) => {
    const v = rs[0].pred.validation, c = v.cohort, o = v.objective;
    const unknown = (reason, detail) => ({ id, status: 'UNKNOWN', reason, detail, regret: null, selected_observed_rank: null });
    if (cohortProblems(c).length || !o || !Array.isArray(v.predictions) || v.predictions.length !== c.candidates.length ||
        c.candidates.some(m => v.predictions.filter(p => p && p.id === m.id).length !== 1) || o.mode !== c.objective.mode ||
        (o.mode === 'single-seat' ? !(num(o.decode_target_tps) && o.decode_target_tps > 0) : !(num(o.per_agent_floor_tps) && o.per_agent_floor_tps > 0))) return unknown('INVALID_CAPTURE', cohortProblems(c));
    const core = x => ({ cohort: x.cohort, objective: x.objective, predictions: x.predictions });
    if (rs.some(r => !sha(r.pred.validation.cohort_sha256) || digest(core(r.pred.validation)) !== r.pred.validation.cohort_sha256 ||
        !same(core(r.pred.validation), core(v)))) return unknown('COHORT_IDENTITY_MISMATCH', 'declaration and frozen predictions must match exactly');
    if (rs.length !== c.candidates.length || c.candidates.some(x => rs.filter(r => r.id === x.run_id).length !== 1)) return unknown('INCOMPLETE_COHORT', 'one actual observation for every predeclared alternative required');
    const firstOutcome = Math.min(...rs.map(r => time(r.when) === null ? Infinity : time(r.when)));
    const badTime = rs.some(r => { const t = time(r.pred.validation.captured_at); return t === null || time(c.declared_at) > t || t >= firstOutcome || time(r.when) === null; });
    if (badTime) return unknown('HINDSIGHT_CAPTURE', 'every full-cohort prediction must precede the first observed run');
    const measured = [];
    for (const member of c.candidates) {
      const r = rs.find(x => x.id === member.run_id), cap = r.pred.validation, m = r.run.measured || {}, mv = m.validation;
      const p = (v.predictions || []).find(x => x.id === member.id);
      if (r.pred_basis !== 'launch' || r.pred.basis !== 'launch' || cap.candidate !== member.id || !same(r.pred.calibration, c.frozen) ||
          !same(((r.run.check || {}).prediction || {}).validation, cap) || !same(bareRecipe(r.run.recipe), member.recipe) ||
          !same((r.run.recipe.validation || {}).cohort, c)) return unknown('MEMBER_MISMATCH', member.id);
      if ((r.causes || []).includes('tool_bug')) return unknown('HARNESS_FAILURE', member.id);
      if ((r.disturbed || []).length || (r.causes || []).includes('interference')) return unknown('INTERFERENCE', member.id);
      if (m.outcome !== 'ok') return unknown('OUTCOME_NOT_OK', member.id + ': ' + (m.outcome || 'not measured'));
      if ((m.degraded || []).length || (r.bad || []).length) return unknown('DEGRADED_OBSERVATION', member.id);
      if (!r.scored || (r.in_sample || []).length) return unknown('CONFOUNDED_OR_TRAINING', member.id);
      if (!mv || !sourced(m, 'validation') || mv.cohort_sha256 !== v.cohort_sha256 || mv.candidate !== member.id || !same(mv.workload, c.workload) ||
          !same(mv.isolation, c.isolation) || (o.mode === 'batch-fleet' && !same(mv.power_scope, o.power_scope))) return unknown('MEASUREMENT_IDENTITY_MISMATCH', member.id);
      const paths = c.constraints.map(x => x.measured).concat(o.mode === 'batch-fleet' ? ['accepted_tps', 'gpu_cpu_watts', 'min_agent_tps'] : [o.measured]);
      if (paths.some(k => !num(field(m, k)) || field(m, k) < 0 || !sourced(m, k)) || objectiveValue(m, o, true) === null) return unknown('MISSING_MEASUREMENT', member.id + ': sourced objective, power (when needed), quality and memory operands required');
      const prediction = p && p.prediction;
      if (!prediction || prediction.status !== 'PREDICTED' || objectiveValue(prediction, o, false) === null ||
          c.constraints.some(x => !num(field(prediction, x.predicted))) || (o.mode === 'batch-fleet' && !num(prediction.min_agent_tps))) return unknown('UNMODELLED_PREDICTION', member.id);
      const qp = constraintPass(prediction, c.constraints.filter(x => x.kind === 'quality'), false), qm = constraintPass(m, c.constraints.filter(x => x.kind === 'quality'), true);
      const fp = constraintPass(prediction, c.constraints, false) && (o.mode !== 'batch-fleet' || prediction.min_agent_tps >= o.per_agent_floor_tps);
      const fm = constraintPass(m, c.constraints, true) && (o.mode !== 'batch-fleet' || m.min_agent_tps >= o.per_agent_floor_tps);
      measured.push({ id: member.id, run_id: r.id, predicted: objectiveValue(prediction, o, false), measured: objectiveValue(m, o, true),
        quality_predicted: qp, quality_measured: qm, feasible_predicted: fp, feasible_measured: fm,
        total_accepted_tps: num(m.accepted_tps) && sourced(m, 'accepted_tps') ? m.accepted_tps : null });
    }
    const best = (xs, k) => xs.reduce((a, x) => !a || x[k] > a[k] ? x : a, null);
    const selected = best(measured.filter(x => x.feasible_predicted), 'predicted'), observed = best(measured.filter(x => x.feasible_measured), 'measured');
    const eligible = selected && selected.feasible_measured && observed;
    const ranked = measured.filter(x => x.feasible_predicted && x.feasible_measured);
    const comparisons = ranked.flatMap((a, i) => ranked.slice(i + 1).filter(b => a.predicted !== b.predicted && a.measured !== b.measured)
      .map(b => (a.predicted - b.predicted) * (a.measured - b.measured) < 0));
    return { id, status: 'SCORED', cohort_sha256: v.cohort_sha256, frozen: c.frozen, objective: o, members: measured,
      selected: selected && selected.id, observed_best: observed && observed.id,
      regret: eligible ? observed.measured - selected.measured : null,
      regret_reason: eligible ? null : !selected ? 'NO_PREDICTED_FEASIBLE_CHOICE' : !selected.feasible_measured ? 'SELECTED_INFEASIBLE' : 'NO_MEASURED_FEASIBLE_ALTERNATIVE',
      selected_observed_rank: eligible ? 1 + measured.filter(x => x.feasible_measured && x.measured > selected.measured).length : null,
      rank_pairs: comparisons.length, rank_inversions: comparisons.length ? comparisons.filter(Boolean).length : null,
      rank_error_fraction: comparisons.length ? comparisons.filter(Boolean).length / comparisons.length : null,
      selected_error: selected ? selected.predicted - selected.measured : null,
      selected_decode_target_met: selected && o.mode === 'single-seat' ? selected.measured >= o.decode_target_tps : null,
      headline_total_accepted_tps: selected ? selected.total_accepted_tps : null,
      quality_misclassifications: measured.filter(x => x.quality_predicted !== x.quality_measured).length,
      feasibility_misclassifications: measured.filter(x => x.feasible_predicted !== x.feasible_measured).length };
  });
  return { status: cohorts.some(x => x.status === 'SCORED') ? 'SCORED' : 'UNKNOWN', regret: null, cohorts, excluded,
    why: cohorts.length ? null : 'no explicit prospective matched-alternative cohort capture; historicals are not decision trials' };
}

module.exports = { captureDecision, decisions, calibrationSplit, digest, frozenValid, taxonomy, median, draftN, draftRatios, causesOf, disturbed, wentBad, primaryPhase, pairs, summarize, verdicts, perCheck, bucketOf, slope, gapVerdict, trend, fmtPrediction };
