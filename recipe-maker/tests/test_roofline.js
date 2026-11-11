'use strict';
// The per-phase roofline (src/model.js prefillRoof / decodeRoof / phaseBound, data/roofline.json): planted cases where the
// PCIe stream binds, where compute binds, and where the firmware unlock flips the binding term; the sources of every ceiling;
// the batch landscape; the explain / rearrangement search; and the prediction record that refuses post-hoc evidence.
const assert = require('assert'), fs = require('fs');
const FM = require('../src/model.js');
const data = JSON.parse(fs.readFileSync('data.json', 'utf8'));
FM.applyCal(data);
const clone = x => JSON.parse(JSON.stringify(x));
let passed = 0;
const test = (name, fn) => { fn(); passed++; console.log('ok ' + name); };
const R = data.roofline, eta = FM.rooflineFit(data, 'tqllm').eta;
const tq = (o) => Object.assign({ box: 'snoopy', model: 'fnx', engine: 'tqllm', store: 'gsq_iq3s', unlock: 'stock' }, o);
const snoopyCard = d => d.boxes.snoopy.gpus.find(g => g.ceil && g.ceil.class === 'rtx3090ti');
// the public copy (it holds .fitviz-export) ships a subset of the recipes: a catalog-wide count holds on the full catalog, one on the subset
const recipeFloor = n => fs.existsSync('.fitviz-export') ? 1 : n;

test('every card and CPU ceiling names its source: MEASURED, DATASHEET or ESTIMATE', () => {
  const tags = new Set(['MEASURED', 'DATASHEET', 'ESTIMATE']);
  let n = 0;
  Object.values(data.boxes).forEach(b => [b.cpu].concat(b.gpus).forEach(x => {
    if (!x || !x.ceil) return;
    Object.entries(x.ceil.values).forEach(([k, c]) => { n++; assert(tags.has(c.tag), k + ' ' + c.tag); assert(c.src && c.src.startsWith(c.tag), k); });
  }));
  assert(n > 20, 'only ' + n + ' ceilings attached');
  const g = snoopyCard(data);
  assert.strictEqual(g.ceil.values.bf16_tensor.v, 84.2); assert.strictEqual(g.ceil.values.bf16_tensor.tag, 'MEASURED');
  assert.strictEqual(g.ceil.unlock.by_state.unlocked.bf16_tensor.v, 131.4);
  assert.strictEqual(g.firmware_unlock.v, 'unlocked');
});

test('the firmware unlock is a setting: stock and unlocked read different ceilings, dated runs before it read stock', () => {
  const g = snoopyCard(data);
  assert.strictEqual(FM.ceilingOf(data, g.ceil, 'bf16_tensor', 'stock').value, 84.2);
  assert.strictEqual(FM.ceilingOf(data, g.ceil, 'bf16_tensor', 'unlocked').value, 131.4);
  // INT8 was not measured unlocked: both states read the datasheet value
  assert.strictEqual(FM.ceilingOf(data, g.ceil, 'int8_tensor', 'unlocked').value, FM.ceilingOf(data, g.ceil, 'int8_tensor', 'stock').value);
  assert.strictEqual(FM.unlockState(g, { at: '2026-11-09' }), 'stock');
  assert.strictEqual(FM.unlockState(g, { at: '2026-11-11' }), 'unlocked');
  assert.strictEqual(FM.unlockState(g, { unlock: 'stock' }), 'stock');
});

test('planted PCIe-binds: a short agent read streams every expert for one chunk, so the stream term binds and scales with the link', () => {
  const p = FM.prefillRoof(data, tq({ tokens: 1000 }), eta);
  assert.strictEqual(p.status, 'OK', p.why);
  assert.strictEqual(p.binding.term, 'stream.experts'); assert.strictEqual(p.binding.side, 'stream');
  const d = clone(data), g = snoopyCard(d), geo = g.pcie.v || g.pcie;
  geo.width = (geo.max_width || geo.width) / 2; geo.max_width = geo.width;
  const half = FM.prefillRoof(d, tq({ tokens: 1000 }), eta);
  assert(Math.abs(half.time_s / p.time_s - 2) < 0.01, 'half the lanes: ' + half.time_s + ' vs ' + p.time_s);
  // the unlock does not touch a stream-bound read
  const u = FM.prefillRoof(data, tq({ tokens: 1000, unlock: 'unlocked' }), eta);
  assert(Math.abs(u.time_s / p.time_s - 1) < 1e-9);
});

test('planted compute-binds: full 16K chunks are compute-bound and the trunk term is the largest compute term', () => {
  const p = FM.prefillRoof(data, tq({ tokens: 32768 }), eta);
  assert.strictEqual(p.binding.side, 'compute');
  assert.strictEqual(p.binding.term, 'compute.trunk');
  assert(p.chunk_detail.every(c => c.binds === 'compute'));
  // compute-bound: the stock / unlocked time ratio sits near the trunk ceiling ratio 131.4 / 84.2, diluted by the int8 terms
  const u = FM.prefillRoof(data, tq({ tokens: 32768, unlock: 'unlocked' }), eta), r = p.time_s / u.time_s;
  assert(r > 1.2 && r < 131.4 / 84.2, 'unlock gain ' + r);
});

test('unlock flips the binding: a chunk size between the two crossovers binds on compute stock and on the stream unlocked', () => {
  const side = (n, u) => FM.prefillRoof(data, tq({ tokens: n, chunk: n, unlock: u }), eta).binding.side;
  const cross = u => { let lo = 256, hi = 16384; for (let i = 0; i < 40; i++) { const m = (lo + hi) / 2; if (side(Math.round(m), u) === 'compute') hi = m; else lo = m; } return hi; };
  const cs = cross('stock'), cu = cross('unlocked');
  assert(cu > cs * 1.2, 'crossovers ' + cs + ' / ' + cu);
  const mid = Math.round(Math.sqrt(cs * cu));
  assert.strictEqual(side(mid, 'stock'), 'compute'); assert.strictEqual(side(mid, 'unlocked'), 'stream');
  console.log('CROSSOVER stock ' + cs.toFixed(0) + ' tokens a chunk, unlocked ' + cu.toFixed(0));
});

test('the fit is one eta per engine on its fit cells; no other constant is refitted', () => {
  const f = FM.rooflineFit(data, 'tqllm');
  assert.strictEqual(f.status, 'FIT'); assert(f.eta > 0 && f.eta <= 1, f.eta);
  assert(!('unlock_pair' in R.fits.tqllm) && !('m0' in f), 'a second fitted term is back');
  const cells = R.checks.filter(c => f.cells.includes(c.id));
  const pred = cells.reduce((a, c) => a + FM.prefillRoof(data, c.query, f.eta).time_s, 0), meas = cells.reduce((a, c) => a + c.measured.ttft_s, 0);
  assert(Math.abs(pred / meas - 1) < 1e-6);
});

test('checks: post-hoc cells are labelled so and scored term by term, never refitted', () => {
  const C = FM.rooflineChecks(data), by = id => C.find(c => c.id === id);
  ['tq-snoopy-after-32k-a', 'tq-snoopy-after-16k-a', 'tq-snoopy-before-16k-a'].forEach(id => assert(/^post hoc/.test(by(id).evidence), id + ': ' + by(id).evidence));
  assert(/^fit cell/.test(by('tq-snoopy-before-32k-a').evidence));
  // the pre-unlock 16K cells: no modelled term explains the extra seconds within x1.5, so RM names an unmodelled term
  const a16 = by('tq-snoopy-before-16k-a').attribution;
  assert(/unmodelled term/.test(a16.verdict), a16.verdict);
  // the unlocked 32K cell: the trunk term alone explains it within x1.5
  const a32 = by('tq-snoopy-after-32k-a').attribution;
  assert(/compute\.trunk/.test(a32.verdict), a32.verdict);
  assert.strictEqual(FM.rooflineFit(data, 'tqllm').eta, eta);
});

test('attribution: a short read faster than the PCIe link allows is impossible on the stream term, and nothing is refitted', () => {
  const q = tq({ tokens: 1000 }), p = FM.prefillRoof(data, q, eta);
  const a = FM.rooflineAttribute(data, q, p.time_s * 0.3, eta);
  const t = a.terms.find(x => x.term === 'stream.experts');
  assert(t.k < 0.31 && /IMPOSSIBLE: needs [0-9.]+ GB\/s over a [0-9.]+ GB\/s PCIe link/.test(t.verdict), JSON.stringify(t));
  assert(/unmodelled term/.test(a.verdict), a.verdict);
  // a slower read: the stream term alone explains a x1.3 miss (possible), and the verdict names it
  const b = FM.rooflineAttribute(data, q, p.time_s * 1.3, eta);
  assert(/stream\.experts x1\.30/.test(b.verdict), b.verdict);
  assert.strictEqual(FM.rooflineFit(data, 'tqllm').eta, eta);
});

test('decode: distinct experts follow E (1 - (1 - k / E)^B) and the weight bytes amortise with the batch', () => {
  const q = { box: 'odie', model: 'fnx', engine: 'strata', store: 'gsq_iq3xxs', flags: '--prefill auto; layer split 25; STRATA_PREFILL_STREAM_CPU_SHARE=0.25', ctx: 60000, draft: true, kv_format: 'k8v4' };
  const m = data.models.fnx, E = Number(FM.v(m.E)), k = Number(FM.v(m.k));
  [1, 4, 16, 64].forEach(B => {
    const d = FM.decodeRoof(data, Object.assign({}, q, { batch: B }));
    assert.strictEqual(d.status, 'OK', d.why);
    assert(Math.abs(d.distinct_experts_per_layer - E * (1 - Math.pow(1 - k / E, B))) < 1e-9);
  });
  const d1 = FM.decodeRoof(data, Object.assign({}, q, { batch: 1 })), d16 = FM.decodeRoof(data, Object.assign({}, q, { batch: 16 }));
  assert(d16.weight_bytes_per_step / 16 < d1.weight_bytes_per_step / 2);
  assert(d16.kv_state_bytes_per_step > d1.kv_state_bytes_per_step * 10);
});

test('landscape: R8 at 16 agents lands within 10 % of the measured 83.4 useful tok/s; bandwidth binds through B 128', () => {
  const x = R.landscapes.find(l => l.id === 'r8-odie'), L = FM.batchLandscape(data, x.query, { prefill_query: x.prefill_query });
  const r16 = L.rows.find(r => r.B === 16);
  assert(Math.abs(r16.useful_tps / 83.4 - 1) < 0.10, r16.useful_tps);
  assert.strictEqual(L.compute_overtakes_bandwidth_at, null);
  const fit = L.rows.filter(r => r.fits_vram), past = L.rows.filter(r => !r.fits_vram);
  for (let i = 1; i < fit.length; i++) assert(fit[i].aggregate_tps > fit[i - 1].aggregate_tps, 'aggregate falls inside capacity at B ' + fit[i].B);
  // past the VRAM capacity the KV spills over PCIe and binds
  assert(past.length && past.every(r => /kv_spill/.test(r.binding)), JSON.stringify(past.map(r => r.binding)));
  assert(past[0].aggregate_tps < fit[fit.length - 1].aggregate_tps);
});

test('explain names the binding and next-binding terms and ranks rearrangements by predicted useful tok/s', () => {
  const x = R.landscapes.find(l => l.id === 'tq-snoopy'), ex = FM.rooflineExplain(data, x.query, { prefill_query: x.prefill_query, agents: 1 });
  assert.strictEqual(ex.status, 'OK', ex.why);
  assert.strictEqual(ex.prefill.binding, 'stream.experts'); assert(ex.prefill.next && ex.prefill.next !== ex.prefill.binding);
  assert(ex.decode.binding && ex.decode.next);
  const ok = ex.search.rows.filter(r => r.status === 'OK');
  for (let i = 1; i < ok.length; i++) assert(ok[i].gain <= ok[i - 1].gain + 1e-12);
  ok.forEach(r => assert(/ s \(max\) = /.test(r.arithmetic), r.arithmetic));
  // for short agent reads the unlock buys almost nothing: the stream binds
  const u = ok.find(r => r.id === 'unlock_unlocked');
  assert(u && Math.abs(u.gain) < 0.02, u && u.gain);
  assert(ex.relief && ex.relief.gain > 0);
});

test('phase_bound: a recipe the planner bounds PASSes with the binding term named; an unbounded recipe FAILs its fit check', () => {
  const r = JSON.parse(fs.readFileSync('recipes/droopy-v41-fs1c.json', 'utf8'));
  const c = FM.checkRecipe(data, r).checks.find(x => x.id === 'phase_bound');
  assert.strictEqual(c.status, 'PASS', c.detail); assert(/binds/.test(c.detail) && /decode batch 1/.test(c.detail));
  // plant: the engine loses its kernel metadata, so neither phase can be bounded
  const d = clone(data); delete d.roofline.engines[r.engine]; FM.applyCal(d);
  const cb = FM.checkRecipe(d, r).checks.find(x => x.id === 'phase_bound');
  assert.strictEqual(cb.status, 'FAIL'); assert(/no phase bound/.test(cb.detail));
});

test('prediction record: refuses a run with no hypothesis or tested term, and scores only runs measured after the record', () => {
  const q = tq({ tokens: 36000, unlock: 'unlocked' });
  assert.strictEqual(FM.rooflinePredict(data, { id: 'x', query: q, tests_term: 'compute.trunk', band_pct: 10 }).status, 'REFUSED');
  assert.strictEqual(FM.rooflinePredict(data, { id: 'x', query: q, hypothesis: 'h', band_pct: 10 }).status, 'REFUSED');
  assert.strictEqual(FM.rooflinePredict(data, { id: 'x', query: q, hypothesis: 'h', tests_term: 'no.such.term', band_pct: 10 }).status, 'REFUSED');
  const P = FM.rooflinePredict(data, { id: 'x', query: q, hypothesis: 'the trunk binds unlocked', tests_term: 'compute.trunk', band_pct: 10 });
  assert.strictEqual(P.status, 'OK', P.why);
  const rec = Object.assign({ written_utc: '2026-11-11T02:00:00Z' }, P.record), want = rec.predicted.prefill_tps;
  assert.strictEqual(FM.rooflineScore(data, rec, want, '2026-11-11T01:00:00Z').status, 'REFUSED');
  assert.strictEqual(FM.rooflineScore(data, rec, want, null).status, 'REFUSED');
  assert.strictEqual(FM.rooflineScore(data, null, want, '2026-11-12T00:00:00Z').status, 'REFUSED');
  assert.strictEqual(FM.rooflineScore(data, rec, want * 1.05, '2026-11-11T03:00:00Z').status, 'HOLDS');
  const miss = FM.rooflineScore(data, rec, want * 0.6, '2026-11-11T03:00:00Z');
  assert.strictEqual(miss.status, 'FALSIFIED'); assert(miss.tested_term_needed && miss.tested_term_needed.k > 1);
  assert.strictEqual(FM.rooflineFit(data, 'tqllm').eta, eta, 'scoring refitted the eta');
});

test('every prediction record in data/ carries a hypothesis, a tested term, a band and a written time before any measured block', () => {
  (R.predictions || []).forEach(p => {
    ['hypothesis', 'tests_term', 'band_pct', 'written_utc', 'predicted'].forEach(k => assert(p[k] !== undefined && p[k] !== null, p.id + ' ' + k));
    if (p.measured) assert(FM.rlAfterRecord(p.measured.measured_utc, p.written_utc).ok, p.id + ' measured before its record');
    // a prefill record stores the per-chunk composition of its prediction (or a backfill made at its own RM commit) summing to its time
    if (p.phase === 'prefill' || !p.phase) {
      const comp = p.predicted.composition || (p.composition_backfill && p.composition_backfill.composition);
      assert(comp && comp.groups.length, p.id + ' has no frozen composition');
      const tot = {};
      comp.groups.forEach(g => [g.compute, g.stream].forEach(o => Object.entries(o).forEach(([id, x]) => { tot[id] = (tot[id] || 0) + x; })));
      p.predicted.terms.forEach(t => assert(Math.abs((tot[t.id] || 0) - t.s) <= 1e-9 * Math.max(1, t.s), p.id + ' ' + t.id + ': composition ' + tot[t.id] + ' vs term ' + t.s));
      assert.strictEqual(FM.rooflineAttributeFrozen(p, p.predicted.time_s).status, 'OK', p.id);
    }
  });
});

test('the chronology check reads instants, not strings: offsets order by time, and a time with no zone or a bad date is refused', () => {
  assert.strictEqual(FM.rlInstant('2026-11-11T01:00:00+02:00'), Date.parse('2026-11-10T23:00:00Z'));
  ['zzz', '', null, undefined, 42, '2026-11-11', '2026-11-11T01:00:00', '2026-02-30T00:00:00Z', '2026-11-11T24:00:00Z', '2026-13-01T00:00:00Z', '2026-11-11T01:00:00+25:00']
    .forEach(x => assert.strictEqual(FM.rlInstant(x), null, String(x)));
  const w = '2026-11-11T00:41:04Z';
  assert(!FM.rlAfterRecord('2026-11-11T01:00:00+02:00', w).ok, 'an offset time earlier as an instant read as after');
  assert(!FM.rlAfterRecord('2026-11-11T02:41:04+02:00', w).ok, 'the same instant read as after');
  assert(!FM.rlAfterRecord('zzz', w).ok); assert(!FM.rlAfterRecord(w, w).ok); assert(!FM.rlAfterRecord('2026-11-11T00:41:05Z', 'zzz').ok);
  assert(FM.rlAfterRecord('2026-11-11T03:00:00+02:00', w).ok); assert(FM.rlAfterRecord('2026-11-11T00:41:05Z', w).ok);
  const P = FM.rooflinePredict(data, { id: 'x', query: tq({ tokens: 36000, unlock: 'stock' }), hypothesis: 'h', tests_term: 'compute.trunk', band_pct: 10 });
  const rec = Object.assign({ written_utc: w }, P.record), want = rec.predicted.prefill_tps;
  ['zzz', '2026-11-11T01:00:00+02:00', '2026-11-11T02:41:04+02:00', w, '2026-11-11T01:00:00'].forEach(m => assert.strictEqual(FM.rooflineScore(data, rec, want, m).status, 'REFUSED', m));
  assert.strictEqual(FM.rooflineScore(data, rec, want, '2026-11-11T03:00:00+02:00').status, 'HOLDS');
  [0, -5, NaN, null, undefined].forEach(m => assert.strictEqual(FM.rooflineScore(data, rec, m, '2026-11-12T00:00:00Z').status, 'REFUSED', String(m)));
  // through the CLI: a record in data/ scored with an offset time earlier than it, with text, and with a later offset time
  const rr = (R.predictions || []).find(r => (r.phase || 'prefill') === 'prefill' && !r.measured);
  assert(rr, 'no unmeasured prefill record in data/ to score through the CLI');
  const cli = mu => require('child_process').spawnSync(process.execPath, ['report.js', '--roofline', '--score', rr.id, '--measured', String(rr.predicted.prefill_tps), '--measured-utc', mu, '--json'], { encoding: 'utf8' });
  const at = (ms, off) => { const d = new Date(Date.parse(rr.written_utc) + ms + off * 3600e3); return d.toISOString().replace(/\.\d+Z$/, '') + '+0' + off + ':00'; };
  [at(-60e3, 2), 'zzz', at(0, 2)].forEach(mu => { const x = cli(mu); assert.strictEqual(x.status, 2, mu + ': ' + x.stdout + x.stderr); assert.strictEqual(JSON.parse(x.stdout).status, 'REFUSED', mu); });
  const ok = cli(at(60e3, 2));
  assert.strictEqual(ok.status, 0, ok.stdout + ok.stderr); assert.strictEqual(JSON.parse(ok.stdout).status, 'HOLDS');
});

test('a record is attributed from its frozen terms: a ceiling changed after the record moves only the current-model attribution', () => {
  const P = FM.rooflinePredict(data, { id: 'x', query: tq({ tokens: 36000, unlock: 'stock' }), hypothesis: 'h', tests_term: 'compute.trunk', band_pct: 10 });
  assert.strictEqual(P.status, 'OK', P.why);
  const rec = Object.assign({ written_utc: '2026-11-11T02:00:00Z' }, JSON.parse(JSON.stringify(P.record))), mu = '2026-11-11T03:00:00Z', m = rec.predicted.prefill_tps * 0.7;
  const a = FM.rooflineScore(data, rec, m, mu);
  assert.strictEqual(a.attribution.status, 'OK', a.attribution.why); assert(/^frozen/.test(a.attribution.basis));
  assert(/^current model/.test(a.attribution_current.basis));
  // on unchanged data the frozen composition prices every term exactly as the model does
  assert(Math.abs(a.attribution.predicted_s / rec.predicted.time_s - 1) < 1e-12);
  a.attribution.terms.forEach((t, i) => { const c = a.attribution_current.terms[i]; assert.strictEqual(t.term, c.term); assert(t.k === null ? c.k === null : Math.abs(t.k / c.k - 1) < 1e-6, t.term + ' ' + t.k + ' vs ' + c.k); });
  // mutate: halve the card's BF16 tensor ceiling after the record was written
  const d = clone(data); FM.applyCal(d);
  const g = snoopyCard(d); g.ceil.values.bf16_tensor.v /= 2; if (g.ceil.unlock && g.ceil.unlock.by_state.stock && g.ceil.unlock.by_state.stock.bf16_tensor) g.ceil.unlock.by_state.stock.bf16_tensor.v /= 2;
  const b = FM.rooflineScore(d, rec, m, mu);
  assert.strictEqual(b.status, a.status); assert.strictEqual(b.err, a.err);
  assert.deepStrictEqual(b.attribution, a.attribution, 'the frozen attribution moved with the data');
  assert.deepStrictEqual(b.tested_term_needed, a.tested_term_needed);
  assert(b.attribution_current.predicted_s > a.attribution_current.predicted_s * 1.2, 'the mutation did not reach the current model: ' + a.attribution_current.predicted_s + ' -> ' + b.attribution_current.predicted_s);
  // the reported case: the stored 32K stock record (composition backfilled at its own RM commit) keeps its 14.857 s
  const r32 = (R.predictions || []).find(r => r.id === 'tq-snoopy-unlock-ab-32k-stock');
  if (r32) {
    const s0 = FM.rooflineScore(data, r32, r32.predicted.prefill_tps * 0.7, '2099-01-01T00:00:00Z'), s1 = FM.rooflineScore(d, r32, r32.predicted.prefill_tps * 0.7, '2099-01-01T00:00:00Z');
    assert.strictEqual(s1.attribution.status, 'OK', s1.attribution.why); assert(/backfilled/.test(s1.attribution.basis));
    assert.strictEqual(s1.attribution.predicted_s, r32.predicted.time_s); assert.deepStrictEqual(s1.attribution, s0.attribution);
  }
  // a record with no composition is not attributed as predicted (never silently against the current data)
  const bare = clone(rec); delete bare.predicted.composition;
  const c = FM.rooflineScore(d, bare, m, mu);
  assert.strictEqual(c.attribution.status, 'UNFROZEN'); assert.strictEqual(c.tested_term_needed, null); assert.strictEqual(c.attribution_current.status, 'OK');
  // the CPU placement path stores a composition that sums to its time too
  const cc = (R.checks || []).find(x => x.query && x.query.placement === 'cpu_only');
  assert(cc, 'no cpu_only check in data.roofline.checks');
  const pc = FM.prefillRoof(data, cc.query, { gpu: 1, cpu: 1 });
  assert.strictEqual(pc.status, 'OK', pc.why); assert.strictEqual(pc.placement, 'cpu_only');
  const sum = pc.composition.groups.reduce((acc, gr) => { const cs = Object.values(gr.compute).reduce((x, y) => x + y, 0), ss = Object.values(gr.stream).reduce((x, y) => x + y, 0); return acc + (pc.composition.rule === 'max' ? Math.max(cs, ss) : cs + ss); }, 0);
  assert(Math.abs(sum / pc.time_s - 1) < 1e-12);
});

test('a decode, agents or pipeline record scores without a prefill attribution (a pipeline record has no query)', () => {
  (R.predictions || []).filter(r => r.phase && r.phase !== 'prefill').forEach(r => {
    const want = r.phase === 'agents' ? r.predicted.useful_tps : r.phase === 'pipeline' ? r.predicted.aggregate_tps : r.predicted.decode_tps;
    const x = FM.rooflineScore(data, r, want, '2099-01-01T00:00:00Z');
    assert(x.status === 'HOLDS', r.id + ' ' + x.status + ' ' + x.why); assert.strictEqual(x.attribution, null); assert.strictEqual(x.attribution_current, null);
  });
});

// ---------- shape curves: an expert GEMM is priced by the rows an expert sees, never silently at the square ceiling ----------
const r8q = () => clone(R.checks.find(c => c.id === 'r8-prefill').query);
const b70 = d => d.boxes.odie.gpus.find(g => g.ceil && g.ceil.class === 'b70');
const withCompose = (mode) => { const d = clone(data); d.roofline.settings.shape_compose.v = mode; FM.applyCal(d); return d; };

test('shape curve: MEASURED at a point, a power law in log(rows) between, ESTIMATE beyond, on other dims or another quant', () => {
  const g = b70(data), at = (rows, ctx, path, prec) => FM.shapeRate(data, g, 'experts', path || 'onednn_grouped', prec || 'int8_tensor', rows, ctx || { model: 'fnx' });
  const p24 = at(24), p80 = at(80), mid = at(Math.sqrt(24 * 80));
  assert.strictEqual(p24.tag, 'MEASURED'); assert.strictEqual(p24.value, 24.58); assert.strictEqual(p80.value, 66.62);
  assert(Math.abs(mid.value - Math.sqrt(24.58 * 66.62)) < 1e-9, mid.value); assert.strictEqual(mid.tag, 'ESTIMATE'); assert(/between 24 and 80/.test(mid.where));
  assert(/below 24/.test(at(2).where) && at(2).tag === 'ESTIMATE'); assert(/above 320/.test(at(1000).where));
  assert.strictEqual(at(24, { model: 'orn35' }).tag, 'ESTIMATE');
  assert.strictEqual(at(10, { model: 'fnx', store: 'gsq_iq3xxs' }, 'sycl_mmid', 'fp16_tensor').tag, 'MEASURED');
  assert.strictEqual(at(10, { model: 'fnx', store: 'gsq_iq3s' }, 'sycl_mmid', 'fp16_tensor').tag, 'ESTIMATE');
  // no curve for that path / precision, and none on the 3090 Ti: null (the caller keeps the square ceiling, flagged)
  assert.strictEqual(at(24, null, 'onednn_grouped', 'fp32'), null);
  assert.strictEqual(FM.shapeRate(data, snoopyCard(data), 'experts', 'onednn_grouped', 'int8_tensor', 24, {}), null);
});

test('guard: the R8 read prices its 24-row expert GEMMs by the measured curve, not the square ceiling, in both compose modes', () => {
  const q = r8q(), sq = 269.2e12, flops = 2 * 1202 * FM.v(data.models.fnx.k) * FM.v(data.models.fnx.P) * FM.v(data.models.fnx.L);
  ['excess', 'kernel'].forEach(mode => {
    const p = FM.prefillRoof(withCompose(mode), q, 1), ex = p.terms.find(t => t.id === 'compute.experts');
    assert(ex.s > 1.3 * flops / sq, mode + ': experts priced at the square ceiling (' + ex.s + ' s)');
    const sh = p.shapes.filter(x => x.cls === 'experts');
    assert.strictEqual(sh.length, 2); sh.forEach(x => { assert.strictEqual(x.shape.tag, 'MEASURED'); assert(Math.abs(x.rows - 1202 * 10 / 512) < 0.05, x.rows); });
    if (mode === 'kernel') assert(Math.abs(ex.s / (flops / 24.58e12) - 1) < 0.01, 'kernel mode: ' + ex.s);
  });
  // the kernel mode is the brief's arithmetic: strata's eta refits to ~0.29 (the compute bound 3.4x loose); excess ~0.07
  const ek = FM.rooflineFit(withCompose('kernel'), 'strata').eta, ee = FM.rooflineFit(data, 'strata').eta;
  assert(ek > 0.28 && ek < 0.31, ek); assert(ee > 0.06 && ee < 0.08, ee);
  // a planted card without curves is priced at the square ceiling and the output says unmeasured
  const d = clone(data); d.boxes.odie.gpus.forEach(x => { if (x.ceil) delete x.ceil.shape_curves; }); FM.applyCal(d);
  const u = FM.prefillRoof(d, q, 1);
  assert(Math.abs(u.terms.find(t => t.id === 'compute.experts').s / (flops / sq) - 1) < 1e-6);
  assert(u.shapes.every(x => /unmeasured/.test(x.line)), JSON.stringify(u.shapes.map(x => x.line)));
});

test('the compute-bound MUL_MAT_ID path keeps nearly all its kernel time in the excess mode; the unmeasured 3090 Ti says so', () => {
  const q = { box: 'odie', model: 'fnx', engine: 'sycl', store: 'gsq_iq3xxs', tokens: 512, chunk: 512 };
  const p = FM.prefillRoof(data, q, 1), x = p.shapes.find(s => s.cls === 'experts');
  assert.strictEqual(p.status, 'OK', p.why); assert.strictEqual(x.path, 'sycl_mmid'); assert.strictEqual(x.shape.tag, 'MEASURED');
  assert(x.layout_read_s < 0.15 * x.kernel_s && x.s > 0.85 * x.kernel_s, JSON.stringify(x));
  const t = FM.prefillRoof(data, tq({ tokens: 1000 }), eta);
  assert(t.shapes.length && t.shapes.every(s => /unmeasured/.test(s.line)));
});

test('decode: GEMM rows are B x the draft multiplier, experts see rows x k / distinct; excess adds nothing below the shortest shape', () => {
  const q = Object.assign(clone(R.landscapes.find(l => l.id === 'r8-odie').query), { batch: 16 });
  const d = FM.decodeRoof(data, q), m = data.models.fnx, E = FM.v(m.E), k = FM.v(m.k), T = 16 * 1.5;
  assert.strictEqual(d.tokens_per_step, T); assert(Math.abs(d.rows_per_expert - T * k / (E * (1 - Math.pow(1 - k / E, T)))) < 1e-9);
  const ex = d.terms.find(t => t.id === 'compute.experts'), sh = d.shapes.filter(s => s.cls === 'experts');
  assert(sh.every(s => s.shape && /below 24/.test(s.shape.where)));
  assert(Math.abs(ex.s - sh.reduce((a, s) => a + s.s, 0)) < 1e-12);
  // read-bound below the shortest shape: the kernel time less its own weight read is under the square time, so the square holds
  sh.forEach(s => { assert(s.kernel_s - s.layout_read_s < s.s); assert(Math.abs(s.eff_ops / (s.square * 1e12) - 1) < 1e-9); });
  // the kernel mode prices the 1-4 row expert GEMMs at the (extrapolated) bench kernel time: well above the excess mode
  const dk = FM.decodeRoof(withCompose('kernel'), q);
  assert(dk.terms.find(t => t.id === 'compute.experts').s > 10 * ex.s);
});

test('held-out choice of the compose mode: R8 at 16 agents reads within 10 % in excess, and misses by > 15 % in kernel', () => {
  const x = R.landscapes.find(l => l.id === 'r8-odie');
  const u = mode => FM.batchLandscape(withCompose(mode), x.query, { prefill_query: x.prefill_query }).rows.find(r => r.B === 16).useful_tps;
  const ue = u('excess'), uk = u('kernel');
  assert(Math.abs(ue / 83.4 - 1) < 0.10, ue); assert(uk / 83.4 - 1 < -0.15, uk);
  const L = FM.batchLandscape(data, x.query, { prefill_query: x.prefill_query, batches: Array.from({ length: 128 }, (_, i) => i + 1) });
  assert.strictEqual(L.best_useful.B, L.capacity_streams, 'useful peaks at the VRAM capacity: ' + JSON.stringify(L.best_useful));
  assert(L.binding_changes.length && /kv_spill/.test(L.binding_changes[L.binding_changes.length - 1].to), JSON.stringify(L.binding_changes));
  assert(L.rows.every(r => typeof r.rows_per_expert === 'number' && r.experts_shape));
});

// ---------- CPU placement: dense, attention and experts on the box CPU ----------
const cpuQ = o => Object.assign({ box: 'droopy', model: 'v41', engine: 'ik', store: 'q3kl0_13_q2k', placement: 'cpu_only', chunk: 4096, threads: 38 }, o);
test('cpu_only prefill: every GEMM class runs on the CPU ceilings, no VRAM or PCIe term, the CPU experts bind', () => {
  const p = FM.prefillRoof(data, cpuQ({ tokens: 11569 }), 1);
  assert.strictEqual(p.status, 'OK', p.why); assert.strictEqual(p.placement, 'cpu_only');
  const ids = p.terms.map(t => t.id).sort();
  assert.deepStrictEqual(ids, ['compute.cpu_attention', 'compute.cpu_experts', 'compute.cpu_trunk', 'read.cpu_weights']);
  assert.strictEqual(p.binding.term, 'compute.cpu_experts');
  Object.values(p.ceilings).forEach(c => assert(/_cpu$|^fp32$/.test(c.key), c.key));
  // the arithmetic: 2 x (routed + trunk) FLOPs a token at the int8 ceiling, 40 x 4 x 32768 x 640 at fp32
  const sh = data.roofline.models.v41, routed = 6 * 35389440 * 40, int8 = data.boxes.droopy.cpu.ceil.values.int8_cpu.v * 1e12;
  const want = 11569 * 2 * (routed + sh.trunk_params.v) / int8;
  const got = p.terms.filter(t => /cpu_trunk|cpu_experts/.test(t.id)).reduce((a, t) => a + t.s, 0);
  assert(Math.abs(got / want - 1) < 1e-9, got + ' vs ' + want);
  // the same query on the GPU placement is a different bound (the hidden assumption this removes)
  const g = FM.prefillRoof(data, cpuQ({ tokens: 11569, placement: undefined }), 1);
  assert(g.tps > 10 * p.tps, g.tps + ' vs ' + p.tps);
});

test('cpu_only: the CPU eta is fitted on the droopy CED cells only, holds them within 10 %, and never prices a GPU placement', () => {
  const f = FM.rooflineFit(data, 'ik', 'cpu');
  assert.strictEqual(f.status, 'FIT', f.why); assert.strictEqual(f.path, 'kquant_requant');
  assert(f.eta > 0.03 && f.eta < 0.15, f.eta);
  const C = FM.rooflineChecks(data);
  ['ced-droopy-p12k-off', 'ced-droopy-p30k-off'].forEach(id => { const c = C.find(x => x.id === id); assert(Math.abs(c.err) < 0.1, id + ' ' + c.err); assert.strictEqual(c.eta_side, 'cpu'); });
  // no GPU fit for ik: a GPU-placement recipe stays bound-only, never predicted at the CPU-only eta
  assert.strictEqual(FM.rooflineFit(data, 'ik').status, 'UNMODELLED');
  const r = JSON.parse(fs.readFileSync('recipes/droopy-v41-fs1c.json', 'utf8'));
  const pb = FM.checkRecipe(data, r).checks.find(x => x.id === 'phase_bound');
  assert(/no fitted eta for ik \(bound only\)/.test(pb.detail), pb.detail);
  // held out on another box and cpu path: the miss is scored and named, not absorbed
  const l = C.find(x => x.id === 'lassie-v41-q8-mm7-cpu');
  assert(l.attribution && /NOT re-run/.test(l.attribution.verdict), JSON.stringify(l.attribution));
  assert(l.roof.unmodelled.some(u => /cpu path q8_0_repack where the CPU eta is fitted on kquant_requant/.test(u)), l.roof.unmodelled.join('; '));
});

test('cpu_only: threads below the physical cores scale the CPU ceilings; SMT siblings add none', () => {
  const lq = o => cpuQ(Object.assign({ box: 'lassie', store: 'mxfp4', tokens: 2048 }, o));
  const a = FM.prefillRoof(data, lq({ threads: 64 }), 1), b = FM.prefillRoof(data, lq({ threads: 32 }), 1), c = FM.prefillRoof(data, lq({ threads: 128 }), 1);
  const comp = p => p.terms.filter(t => /^compute/.test(t.id)).reduce((x, t) => x + t.s, 0);
  assert(Math.abs(comp(b) / comp(a) - 2) < 1e-9); assert(Math.abs(comp(c) / comp(a) - 1) < 1e-9);
});

test('cpu_only decode: weights, KV and GEMMs on the CPU side, no VRAM capacity', () => {
  const d = FM.decodeRoof(data, cpuQ({ batch: 1, ctx: 8192 }));
  assert.strictEqual(d.status, 'OK', d.why); assert.strictEqual(d.capacity_streams, null);
  assert(d.terms.every(t => /cpu_|DRAM/.test(t.id + ' ' + t.resource)), JSON.stringify(d.terms));
  assert.strictEqual(d.binding.term, 'read.cpu_weights');
});

test('phase_bound guard: a recipe that puts dense and attention on the CPU is REFUSED, never bounded on the GPU, when no CPU ceiling exists', () => {
  const r = JSON.parse(fs.readFileSync('recipes/lassie-v41-q8-mm7.json', 'utf8'));
  assert.strictEqual(r.roofline_placement, 'cpu_only');
  const ok = FM.checkRecipe(data, r).checks.find(x => x.id === 'phase_bound');
  assert.strictEqual(ok.status, 'PASS', ok.detail); assert(/placement cpu_only/.test(ok.detail) && /CPU eta/.test(ok.detail), ok.detail);
  // plant 1: the box loses its CPU ceilings
  const d1 = clone(data); delete d1.boxes.lassie.cpu.ceil; FM.applyCal(d1);
  const c1 = FM.checkRecipe(d1, r).checks.find(x => x.id === 'phase_bound');
  assert.strictEqual(c1.status, 'FAIL'); assert(/REFUSED/.test(c1.detail) && /no CPU compute ceilings/.test(c1.detail), c1.detail);
  // plant 2: the CPU lacks the precision the engine's CPU trunk runs at
  const d2 = clone(data); delete d2.boxes.lassie.cpu.ceil.values.int8_cpu; delete d2.roofline.precision_fallback.v.int8_cpu; FM.applyCal(d2);
  const c2 = FM.checkRecipe(d2, r).checks.find(x => x.id === 'phase_bound');
  assert.strictEqual(c2.status, 'FAIL'); assert(/has no int8_cpu ceiling/.test(c2.detail), c2.detail);
  // plant 3: an engine with no CPU kernel metadata, and a placement the table does not hold
  const d3 = clone(data); delete d3.roofline.engines.ik.prefill.cpu_trunk; FM.applyCal(d3);
  assert(/REFUSED/.test(FM.checkRecipe(d3, r).checks.find(x => x.id === 'phase_bound').detail));
  const c4 = FM.checkRecipe(data, Object.assign({}, r, { roofline_placement: 'npu_only' })).checks.find(x => x.id === 'phase_bound');
  assert.strictEqual(c4.status, 'FAIL'); assert(/not in data.roofline.settings.placements/.test(c4.detail), c4.detail);
  // CPU experts beside a GPU with no CPU ceiling are refused too, never priced at zero
  const q = { box: 'droopy', model: 'v41', engine: 'ik', tokens: 2210, chunk: 4096, resident_bytes: 0, host_path: 'cpu' };
  const d5 = clone(data); delete d5.boxes.droopy.cpu.ceil;
  const p5 = FM.prefillRoof(d5, q, 1);
  assert.strictEqual(p5.status, 'UNMODELLED'); assert(/^refused: host experts computed on the CPU/.test(p5.why), p5.why);
});

test('every lassie V4.1 recipe that waives vram_peak as a CPU-only build declares roofline_placement cpu_only', () => {
  const cpu = fs.readdirSync('recipes').filter(f => /^lassie-v41-.*\.json$/.test(f) && !/\.check\.json$/.test(f))
    .map(f => JSON.parse(fs.readFileSync('recipes/' + f, 'utf8'))).filter(r => /runs a CPU-only build/.test(JSON.stringify(r.gate_waiver || r.gate_waivers || '')));
  assert(cpu.length >= recipeFloor(12), cpu.length);
  cpu.forEach(r => assert.strictEqual(r.roofline_placement, 'cpu_only', r.id));
});

test('recipe placement is the planner per-card object only: no recipe names a roofline placement in it (that is roofline_placement)', () => {
  const rs = fs.readdirSync('recipes').filter(f => /\.json$/.test(f) && !/\.check\.json$/.test(f))
    .map(f => JSON.parse(fs.readFileSync('recipes/' + f, 'utf8'))).filter(r => r.kind !== 'dllm');
  const str = rs.filter(r => r.placement !== undefined && (r.placement === null || typeof r.placement !== 'object'));
  assert.deepStrictEqual(str.map(r => r.id), []);
  assert(rs.filter(r => typeof r.roofline_placement === 'string').length >= recipeFloor(12));
});

test('a recipe with an object placement and a roofline_placement string bounds by the string only', () => {
  const r = JSON.parse(fs.readFileSync('recipes/lassie-v41-q8-mm7.json', 'utf8'));
  const obj = { mode: 'gpu_layers', layers: [0, 1], cards: {} };
  const both = Object.assign({}, r, { placement: obj });
  assert.strictEqual(FM.recipeRoofQuery(data, both).placement, 'cpu_only');
  const pb = FM.checkRecipe(data, both).checks.find(x => x.id === 'phase_bound');
  assert.strictEqual(pb.status, 'PASS', pb.detail); assert(/placement cpu_only/.test(pb.detail), pb.detail);
  // the object alone names no roofline placement (the GPU one); an unknown roofline word still REFUSES
  const objOnly = Object.assign({}, r, { placement: obj }); delete objOnly.roofline_placement;
  assert.strictEqual(FM.recipeRoofQuery(data, objOnly).placement, undefined);
  const bad = FM.checkRecipe(data, Object.assign({}, both, { roofline_placement: 'npu_only' })).checks.find(x => x.id === 'phase_bound');
  assert.strictEqual(bad.status, 'FAIL'); assert(/REFUSED/.test(bad.detail) && /not in data.roofline.settings.placements/.test(bad.detail), bad.detail);
});

test('phase_bound guard: a recipe whose placement is the planner per-card object stays a GPU placement, never REFUSED as an unknown name', () => {
  const objs = fs.readdirSync('recipes').filter(f => /\.json$/.test(f) && !/\.check\.json$/.test(f))
    .map(f => JSON.parse(fs.readFileSync('recipes/' + f, 'utf8'))).filter(r => r.placement && typeof r.placement === 'object');
  assert(objs.length >= recipeFloor(10), objs.length);
  objs.forEach(r => {
    assert.strictEqual(FM.recipeRoofQuery(data, r).placement, undefined, r.id);
    const pb = FM.checkRecipe(data, r).checks.find(x => x.id === 'phase_bound');
    assert(!/REFUSED|not in data\.roofline\.settings\.placements/.test(pb.detail), r.id + ': ' + pb.detail);
  });
});

test('precision fallback takes the fastest key the card has: the 1080 Ti prices FP16 tensor attention at FP32, and lassie fs3 stays at or above its measured prefill', () => {
  const g = data.boxes.lassie.gpus.find(x => x.ceil && x.ceil.class === 'gtx1080ti');
  const c = FM.ceilingOf(data, g.ceil, 'fp16_tensor', 'stock');
  assert.strictEqual(c.key, 'fp32'); assert(c.fallback);
  assert.strictEqual(FM.ceilingOf(data, g.ceil, 'int8_tensor', 'stock').key, 'int8_dp4a');
  const r = JSON.parse(fs.readFileSync('recipes/lassie-v41-q8-fs3.json', 'utf8'));
  const p = FM.phaseBound(data, FM.recipeRoofQuery(data, r)).prefill.bound;
  assert.strictEqual(p.status, 'OK', p.why);
  assert(p.tps >= 48.2, 'bound ' + p.tps + ' below the measured 48.2 tok/s (runs ledger lassie-v41-q8-fs3)');
});

test('orn35 state: the geometry matches its GGUF header and counts the convolution tail', () => {
  const G = data.roofline.models.orn35.state_geometry.v, H = JSON.parse(fs.readFileSync('src/hdr-orn35.json', 'utf8'));
  const kv = H.kv, conv = H.tensors.find(t => t[0] === 'blk.0.ssm_conv1d.weight');
  const main = JSON.parse(fs.readFileSync('data/models/orn35.json', 'utf8')).layers.v.main;
  assert.strictEqual(G.gdn_layers, main - main / kv['qwen35moe.full_attention_interval']);
  assert.strictEqual(G.value_heads, kv['qwen35moe.ssm.time_step_rank']);
  assert.strictEqual(G.key_dim, kv['qwen35moe.ssm.state_size']); assert.strictEqual(G.value_dim, kv['qwen35moe.ssm.state_size']);
  assert.deepStrictEqual([G.conv_tail + 1, G.conv_channels], conv[1]);
  assert.strictEqual(data.roofline.models.orn35.state_bytes_per_stream, undefined);
  const q = data.roofline.checks.find(c => c.id === 'fitvram-decode').query, d = FM.decodeRoof(data, Object.assign({}, q, { batch: 1, ctx: 0 }));
  assert.strictEqual(d.status, 'OK', d.why); assert.strictEqual(d.kv_state_bytes_per_step, 2 * 65863680);
});

test('QFN state: the per-stream recurrent state is derived from the traced engine geometry (48 value heads + convolution), not stated', () => {
  const G = data.roofline.models.fnx.state_geometry;
  assert.strictEqual(G.tag, 'TRACED'); assert(/layout\.hpp:34-65/.test(G.src) && /session\.cpp/.test(G.src), G.src);
  assert.strictEqual(data.roofline.models.fnx.state_bytes_per_stream, undefined);
  // at context 0 a step moves only the state, read and written: 2 x (36 x 128 x 48 x 128 x 4 + 36 x 10240 x 3 x 4) a stream
  const x = R.landscapes.find(l => l.id === 'r8-odie'), d = FM.decodeRoof(data, Object.assign({}, x.query, { batch: 3, ctx: 0 }));
  assert.strictEqual(d.kv_state_bytes_per_step, 3 * 2 * 117669888);
});

console.log(passed + ' passed');
