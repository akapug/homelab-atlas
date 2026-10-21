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

module.exports = { taxonomy, median, draftN, draftRatios, causesOf, disturbed, wentBad, primaryPhase, pairs, summarize, verdicts, perCheck, bucketOf, slope, gapVerdict, trend, fmtPrediction };
