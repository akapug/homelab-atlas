'use strict';
// agent workload (src/model.js agent*, cpuPoolCheck; data.workload from build.py + data/workload.json): run after python3 build.py
const assert = require('assert'), fs = require('fs'), path = require('path');
const FM = require('../src/model.js');
const ROOT = path.join(__dirname, '..');
const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'data.json'), 'utf8'));
const clone = x => JSON.parse(JSON.stringify(x));
const v = FM.v, W = data.workload;
let passed = 0;
const test = (name, fn) => { fn(); passed++; console.log('ok ' + name); };
const skip = name => console.log('skip ' + name + ' (its ledger rows are not in this build)');
const tag = (x, t) => ({ v: x, tag: t || 'MEASURED', src: 'test' });
const near = (a, b, e) => assert(Math.abs(a - b) < (e || 1e-9), a + ' != ' + b);

// a synthetic two-stage config like the curve tests' (no ledger row needed), and a synthetic real-agent row of the same build
const cfg = (pts, o) => Object.assign({ key: 'k', id: 'synthetic', label: 'synthetic', box: 'odie', model: 'fnx', engine: 'strata', ctx: 131072, draft: true, runs: ['r'],
  build: 'image abcdef01 (test build)', flags: '--batch=agents --batch-groups 2; layer split 25',
  groups: tag(2, 'READ'), stages: tag(2, 'READ'), slots: null, slots_each_point: tag(true, 'READ'), serving: { kv_resident: 4096, kv: 'k8v4', max_context: 131072, batch_groups: 2, batch_mtp: false, stage_lend_drop: true },
  points: pts.map(([n, t]) => ({ n, run: 'r', used: true, why: null, tps: tag(t), win_ms: null, watts: null })) }, o || {});
const realRow = (o) => Object.assign({ run: 'real-1', when: '2026-01-01T00:00Z', box: 'odie', model: 'fnx', engine: 'strata', build: 'image abcdef01 (test build)', build_key: 'abcdef01', n: 16, label: 'real',
  agent_tps: tag(30), prefill_tps: tag(1000), fresh_per_output: tag(25), prefill_busy_pct: tag(80) }, o || {});
const withReal = rows => { const d = clone(data); d.workload.measured = rows; return d; };

test('the workload term: ceiling min(D, P / F), serial 1 / (F / P + 1 / D), the smaller term binds', () => {
  const pre = FM.agentThroughput({ decode: 300, prefill: 1000, fresh: 25, stall: true });
  near(pre.prefill_cap, 40);
  near(pre.ceiling, 40);
  near(pre.serial, 1 / (25 / 1000 + 1 / 300));
  assert.strictEqual(pre.expect, pre.serial, 'an engine whose reads stall decode runs the two in turn');
  assert.strictEqual(pre.binding, 'prefill');
  near(pre.busy, (25 / 1000) / (25 / 1000 + 1 / 300));
  const dec = FM.agentThroughput({ decode: 30, prefill: 1000, fresh: 10, stall: false });
  assert.strictEqual(dec.binding, 'decode');
  near(dec.expect, 30);
  near(dec.busy, 0.3, 1e-12);
});

test('decode capacity alone (a synthetic short-prompt run, F = 0) never binds on prefill; no P gives no ceiling', () => {
  const syn = FM.agentThroughput({ decode: 300, prefill: 1000, fresh: v(W.profiles['synthetic-short'].fresh_per_output), stall: true });
  assert.strictEqual(syn.binding, 'decode');
  near(syn.expect, 300);
  const none = FM.agentThroughput({ decode: 300, prefill: null, fresh: 24, stall: true });
  assert.strictEqual(none.binding, null);
  assert.strictEqual(none.expect, null);
});

test('the profile is a setting: the default names a profile with its F; --fresh overrides it', () => {
  const p = FM.agentProfile(data, {});
  assert.strictEqual(p.name, v(W.settings.default_profile));
  assert.strictEqual(p.fresh, W.profiles[p.name].fresh_per_output);
  assert.strictEqual(v(FM.agentProfile(data, { fresh: 7 }).fresh), 7);
  const d = clone(data); d.workload.settings.default_profile.v = 'synthetic-short';
  assert.strictEqual(v(FM.agentProfile(d, {}).fresh), 0);
  assert.throws(() => FM.agentProfile(data, { profile: 'no-such' }));
});

test('the coding-agent profile carries its per-turn delta distribution', () => {
  const p = W.profiles['coding-agent'];
  assert(v(p.fresh_per_output) > 1, 'a coding agent reads more fresh prompt than it writes');
  ['median', 'p90'].forEach(k => assert(v(p.delta_tokens)[k] > 0, k));
  assert(v(p.delta_tokens).p90 > v(p.delta_tokens).median);
});

test('agentPlan: D from the fitted curve, P from the build\'s best real row, F from the profile; prefill binds', () => {
  const c = cfg([[16, 270], [32, 356]]), d = withReal([realRow(), realRow({ run: 'real-2', agent_tps: tag(25), prefill_tps: tag(1200) }), realRow({ run: 'other', build_key: 'ffffffff', prefill_tps: tag(5000) })]);
  const P = FM.agentPlan(d, c, { n: [16, 32] });
  assert.strictEqual(P.prefill_run, 'real-1', 'the highest agent tok/s row is the reference');
  near(v(P.prefill), 1000);
  const r16 = P.rows[0], F = v(FM.agentProfile(d, {}).fresh);
  near(r16.decode, 270, 1e-6);
  near(r16.prefill_cap, 1000 / F);
  assert.strictEqual(r16.binding, 'prefill');
  assert.strictEqual(P.decode_capacity_only, true);
  assert.deepStrictEqual(r16.real.map(x => x.run), ['real-1', 'real-2'], 'only the same build is compared; other builds are never borrowed');
  near(r16.real[0].own.prefill_cap, 1000 / 25, 1e-9);
  near(r16.real[0].err_expect, r16.real[0].own.serial / 30 - 1);
  assert.strictEqual(P.rows[1].real.length, 0);
});

test('agentPlan without a real row of the build: P PREDICTED (MODEL) from the box and store, binding named; no store, no P; --prefill overrides', () => {
  const c = cfg([[16, 270], [32, 356]]), d = withReal([realRow({ build_key: 'ffffffff' })]);
  const P = FM.agentPlan(d, c, { n: [16] });
  assert.strictEqual(P.prefill.tag, 'MODEL', 'odie / fnx / strata has a prefill pass model');
  assert.strictEqual(P.prefill_basis, 'model');
  assert(v(P.prefill) > 0 && /prefillPass/.test(P.prefill.src), P.prefill.src);
  assert(['decode', 'prefill'].includes(P.rows[0].binding), 'a predicted P names the binding term before launch');
  assert.strictEqual(P.evidence.other.length, 1);
  const none = FM.agentPlan(d, cfg([[16, 270], [32, 356]], { model: 'v41' }), { n: [16] });
  assert.strictEqual(none.prefill, null, 'a store without expert bytes gets no P');
  assert.strictEqual(none.rows[0].binding, null);
  const Q = FM.agentPlan(d, c, { n: [16], prefill: 2400, fresh: 24 });
  near(Q.rows[0].prefill_cap, 100);
  assert.strictEqual(Q.rows[0].binding, 'prefill');
  assert.strictEqual(Q.prefill_basis, 'setting');
});

// ---------- prefill pass model (data.workload.prefill_pass, src/model.js prefillPass) ----------
const PP = W.prefill_pass, fitQ = { box: PP.engines.strata.fit.box, model: PP.engines.strata.fit.model, engine: 'strata', flags: v(PP.engines.strata.fit.flags) };

test('prefillPass: the one strata factor reproduces its fit rows\' server prefill (fresh tokens / prompt time)', () => {
  const p = FM.prefillPass(data, fitQ), F = PP.engines.strata.fit;
  assert.strictEqual(p.status, 'MODEL');
  near(p.mix_tps, v(F.fresh_tokens) / v(F.prompt_s), 1e-6);
  assert.strictEqual(p.chunk, v(PP.engines.strata.chunk_default), '--prefill auto keeps the engine chunk');
  near(p.cpu_share, 0.25, 1e-12);
  assert.strictEqual(p.stages.length, 2, 'layer split 25: two stages');
  assert(p.fit.eta > 0 && p.fit.eta < 1, 'eta is a fraction of the int8 peak');
});

test('prefillPass: t_pass = streamed bytes x (1 - cpu share) / link + resident bytes / VRAM bandwidth, summed over stages', () => {
  const p = FM.prefillPass(data, fitQ);
  near(p.t_pass, p.stages.reduce((a, s) => a + s.nonresident_gb * 0.75 / s.link_gbs + s.resident_gb / s.vram_gbs, 0), 1e-12);
  const lane = v(PP.settings.pcie_gbs_per_lane)[4] * 16 * v(PP.settings.link_eff);
  p.stages.forEach(s => near(s.link_gbs, lane, 1e-9));
  near(p.stages.reduce((a, s) => a + s.resident_gb + s.nonresident_gb, 0), v(PP.stores.fnx.expert_bytes) / 1e9, 1e-6);
  const cpu = FM.prefillPass(data, fitQ, { cpu_share: 1 });
  near(cpu.t_pass, p.stages.reduce((a, s) => a + s.resident_gb / s.vram_gbs, 0), 1e-12, 'the CPU computes every nonresident expert: no PCIe term');
  const d = clone(data); d.workload.prefill_pass.stores.fnx.cache_bytes_per_card.odie.v = 1e12;
  const all = FM.prefillPass(d, fitQ);
  assert(all.stages.every(s => s.nonresident_gb === 0), 'a cache bigger than the stage holds every expert');
  assert(all.t_pass < p.t_pass);
});

test('prefillPass: a bigger chunk helps only reads longer than the old chunk; P rises with the chunk and stays under the long-read rate', () => {
  const ps = [2048, 4096, 8192, 16384].map(C => FM.prefillPass(data, Object.assign({}, fitQ, { flags: fitQ.flags + ' --prefill ' + C })));
  ps.forEach((p, i) => { assert.strictEqual(p.chunk, [2048, 4096, 8192, 16384][i], 'the last --prefill N wins over auto'); near(p.tps(1500), ps[0].tps(1500), 1e-9); assert(p.mix_tps < p.shared_tps); });
  ps.slice(1).forEach((p, i) => { assert(p.mix_tps >= ps[i].mix_tps); assert(p.tps(16384) > ps[i].tps(16384)); });
  near(ps[0].tps(4096), 4096 / (2 * ps[0].t_pass + 4096 * ps[0].m_ms / 1e3), 1e-9);
});

test('prefillPass: no store, cache or card row is UNMODELLED with the reason', () => {
  assert.strictEqual(FM.prefillPass(data, Object.assign({}, fitQ, { model: 'v41' })).status, 'UNMODELLED');
  assert(/expert cache bytes/.test(FM.prefillPass(data, Object.assign({}, fitQ, { box: 'lassie' })).why));
  assert(/has no prefill pass model/.test(FM.prefillPass(data, Object.assign({}, fitQ, { engine: 'ik' })).why));
});

test('prefillPass: the held-out droopy point (CUDA, one RTX 6000, PCIe 3) is bracketed by its CPU-share range', () => {
  const c = FM.prefillPassChecks(data).find(x => x.id === 'droopy-fnx-rtx6000-11565');
  assert(c && c.pred.every(x => x > 0), JSON.stringify(c));
  assert(c.brackets, 'predicted ' + c.pred.map(Math.round).join('-') + ' vs measured ' + c.measured.join('-'));
});

test('the committed prefill-chunk predictions are this model\'s numbers (written before the arms ran)', () => {
  const pr = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'predictions-prefill-chunk-1026.json'), 'utf8'));
  pr.predictions.rows.forEach(r => { const p = FM.prefillPass(data, Object.assign({}, fitQ, { flags: fitQ.flags + ' --prefill ' + r.chunk })); assert(Math.abs(p.mix_tps - r.P) < 1, r.chunk + ': ' + p.mix_tps + ' vs ' + r.P); });
  pr.predictions.cross_agent_batching.rows.forEach(r => { const p = FM.prefillPass(data, fitQ, { chunk: r.chunk }); assert(Math.abs(p.shared_tps - r.P_shared) < 1, r.chunk + ': ' + p.shared_tps); });
});

test('prefill levers multiply P: the prefill-bound ceiling rises by the same factor', () => {
  const c = cfg([[16, 270], [32, 356]]), d = withReal([realRow()]);
  const P = FM.agentPlan(d, c, { n: [16], fresh: 25 }), L = v(W.engines.strata.prefill_levers);
  assert.strictEqual(P.levers.length, L.length);
  P.levers.forEach((x, i) => x.gain.forEach((g, j) => near(x.prefill_cap[j], 1000 * g / 25)));
});

test('a lever carries its own tag; a MEASURED gain within the replay noise is marked inside noise', () => {
  const c = cfg([[16, 270], [32, 356]]), d = withReal([realRow()]);
  d.workload.settings.replay_noise_pct = tag(10);
  d.workload.engines.strata.prefill_levers = { v: [{ id: 'm_small', lever: 'm', gain: [1.05, 1.2], tag: 'MEASURED', src: 'test' },
    { id: 'm_big', lever: 'm', gain: [1.15, 1.2], tag: 'MEASURED', src: 'test' }, { id: 'inf', lever: 'i', gain: [1.01, 1.01] }], tag: 'INFERRED-BY-US', src: 'list' };
  const P = FM.agentPlan(d, c, { n: [16], fresh: 25 });
  assert.deepStrictEqual(P.levers.map(x => [x.id, x.tag, x.inside_noise]), [['m_small', 'MEASURED', true], ['m_big', 'MEASURED', false], ['inf', 'INFERRED-BY-US', null]]);
  assert.strictEqual(P.levers[2].src, 'list', 'a lever without its own src cites the list');
  assert.strictEqual(P.noise_pct, 10);
  const sp = v(W.engines.strata.prefill_levers).find(x => x.id === 'shared_prefix');
  assert(sp && sp.tag === 'MEASURED' && /STRATA_SHARED_PREFIX=1/.test(sp.lever), 'the shipped shared-prefix lever is MEASURED');
  assert(v(W.settings.replay_noise_pct) > 0, 'the shipped replay noise is a setting');
});

// the window-row check: a layer split computes missed experts on the CPU pool, so ceil(batch / groups) <= the pool cap
const rec = (batch, o) => Object.assign({ id: 't', box: 'odie', model: 'fnx', engine: 'strata', ctx: 32768, draft: true, engine_build: 'image 00000000 (test)',
  conditions: { engine_flags: '--batch ' + batch + ' --batch-groups 2; layer split 25' }, serving: { max_context: 32768, batch, batch_groups: 2, kv_resident: 2048, kv: 'k8v4', batch_mtp: false } }, o || {});

test('cpu_pool_rows: 32 agents in 2 groups (16 rows) fit the cap of 16; 64 agents (32 rows) FAIL', () => {
  const cap = v(data.engines.strata.serving.defaults.cpu_pool_max_rows);
  assert.strictEqual(FM.cpuPoolCheck(data, rec(2 * cap)).status, 'PASS');
  const f = FM.cpuPoolCheck(data, rec(4 * cap));
  assert.strictEqual(f.status, 'FAIL');
  assert.strictEqual(f.pred.rows, 2 * cap);
});

test('cpu_pool_rows: the cap is a setting (serving.cpu_pool_max_rows, or the build table); no split, no check', () => {
  const cap = v(data.engines.strata.serving.defaults.cpu_pool_max_rows);
  assert.strictEqual(FM.cpuPoolCheck(data, rec(4 * cap, { serving: Object.assign(rec(4 * cap).serving, { cpu_pool_max_rows: 2 * cap, max_t: 2 * cap }) })).status, 'PASS');
  const byBuild = v(data.engines.strata.serving.cpu_pool_rows_by_build), key = Object.keys(byBuild)[0];
  if (key) assert.strictEqual(FM.cpuPoolCheck(data, rec(2 * byBuild[key], { engine_build: 'image ' + key + ' (test)', serving: Object.assign(rec(2 * byBuild[key]).serving, { max_t: byBuild[key] }) })).status, 'PASS');
  assert.strictEqual(FM.cpuPoolCheck(data, rec(4 * cap, { conditions: { engine_flags: '--batch 64 --batch-groups 2' } })), null, 'one stage: no CPU-expert stage to judge');
  assert.strictEqual(FM.cpuPoolCheck(data, rec(4 * cap, { conditions: { engine_flags: '--batch 64' }, serving: Object.assign(rec(4 * cap).serving, { cpu_experts: true }) })).status, 'FAIL', 'serving.cpu_experts declares it without a split');
});

test('cpu_pool_rows: max_t above the cap lets window-row prefill pass it: UNKNOWN', () => {
  const cap = v(data.engines.strata.serving.defaults.cpu_pool_max_rows);
  assert.strictEqual(FM.cpuPoolCheck(data, rec(2 * cap, { serving: Object.assign(rec(2 * cap).serving, { max_t: 2 * cap }) })).status, 'UNKNOWN');
});

test('agent_workload: a many-agent recipe without a real-agent row of its build is FLAGGED (UNKNOWN); a few-agent one is not judged', () => {
  const d = withReal([realRow({ build_key: 'ffffffff' })]), many = v(W.settings.many_agents_min);
  const c = FM.agentWorkloadCheck(d, rec(32, { conditions: { engine_flags: 'layer split 25', workload: { agents: many } } }));
  assert.strictEqual(c.status, 'UNKNOWN');
  assert(/^FLAG: /.test(c.detail) && /decode capacity, not agent throughput/.test(c.detail), c.detail);
  assert(/not borrowed/.test(c.detail));
  assert.strictEqual(FM.agentWorkloadCheck(d, rec(32, { conditions: { workload: { agents: many - 1 } } })), null);
});

test('agent_workload: with a real-agent row of the build it PASSes and names the binding term', () => {
  const d = withReal([realRow({ build_key: '00000000' })]);
  const c = FM.agentWorkloadCheck(d, rec(32, { conditions: { workload: { agents: 16 } } }));
  assert.strictEqual(c.status, 'PASS');
  near(c.pred.prefill_cap, 1000 / v(FM.agentProfile(d, {}).fresh));
  assert(/16 agents/.test(c.detail));
});

test('checkRecipe carries both new checks', () => {
  const d = withReal([]), ids = FM.checkRecipe(d, rec(64)).checks.map(c => c.id);
  assert(ids.includes('cpu_pool_rows') && ids.includes('agent_workload'), ids.join(' '));
});

test('agent_workload: no real row of the build still FLAGS, but carries the predicted binding term (prefill MODEL); prediction_first PASSes on it', () => {
  const many = v(W.settings.many_agents_min), r = rec(32, { conditions: { engine_flags: 'layer split 25 STRATA_PREFILL_STREAM_CPU_SHARE=0.25', workload: { agents: many } } });
  const d = withReal([]);
  d.curves.configs = [cfg([[8, 200], [16, 270], [32, 356]], { build: 'image 00000000 (test)' })];
  const aw = FM.agentWorkloadCheck(d, r);
  assert.strictEqual(aw.status, 'UNKNOWN');
  assert(/^FLAG: /.test(aw.detail) && /Before launch: /.test(aw.detail) && /MODEL, predicted/.test(aw.detail), aw.detail);
  assert.strictEqual(aw.pred.prefill_tag, 'MODEL');
  const pf = FM.checkRecipe(d, r).checks.find(c => c.id === 'prediction_first');
  assert.strictEqual(pf.status, 'PASS', pf.detail);
  assert.strictEqual(pf.pred.prefill_tag, 'MODEL');
  assert(/\(MODEL\)/.test(pf.detail), pf.detail);
});

test('curve reports say their totals are decode capacity', () => {
  const R = FM.curveReport(data, cfg([[16, 270], [32, 356]]));
  assert.strictEqual(R.decode_capacity, true);
});

// the served Flash-Next config on odie's two B70s, from the ledger (left out of a public copy)
const r8 = data.curves.configs.find(c => c.runs.some(r => /ctlw-prof-b16/.test(r)));
if (!r8 || !W.measured.length) skip('the served build at 16/32/64 agents');
else test('the served build at 16/32/64 agents: prefill-bound at every count, ceiling P / F; 64 agents FAIL the window check on the served build', () => {
  const P = FM.agentPlan(data, r8);
  assert(P.prefill, 'a real-agent row of the build gives P');
  P.rows.forEach(r => { assert.strictEqual(r.binding, 'prefill', 'n=' + r.n); near(r.ceiling, v(P.prefill) / v(P.fresh)); });
  assert(P.rows.every((r, i) => i === 0 || r.decode > P.rows[i - 1].decode), 'more agents raise decode capacity, not output');
  const r64 = P.rows.find(r => r.n === 64);
  assert.strictEqual(r64.cpu_pool.status, 'FAIL');
  const r16 = P.rows.find(r => r.n === 16), m = r16.real.find(x => /real-ctl/.test(x.run));
  assert(m && m.own.serial > m.tps && m.own.ceiling > m.own.serial, 'the measured 16-agent replay sits under the serial prediction, under the ceiling');
});

// ---------- grounding checks: replay_artifact, steady_state, prediction_first, serving priors ----------
const prof = W.profiles[v(W.settings.default_profile)], traces = Object.keys(prof.trace_ideal_fresh_per_output || {});
const K = v(W.settings.replay_artifact_k);

test('replay_artifact: a planted F over K x the trace-ideal ratio FLAGS with the excess and the causes; F at the ideal PASSes', () => {
  assert(traces.length, 'the default profile carries a trace-ideal ratio');
  const ideal = v(prof.trace_ideal_fresh_per_output[traces[0]]);
  const bad = FM.replayArtifactCheck(data, realRow({ trace: traces[0], fresh_per_output: tag(ideal * K * 1.5), out_tokens: 1000, cache_evictions: 7 }));
  assert.strictEqual(bad.status, 'FLAG', bad.detail);
  near(bad.excess, K * 1.5, 1e-9);
  assert.strictEqual(bad.ref.basis, 'trace-ideal');
  near(bad.surplus_tokens, Math.round((ideal * K * 1.5 - ideal) * 1000), 0.5);
  assert(/suspect replay artifact \(cold restarts \/ evictions \/ template drift\)/.test(bad.detail) && /7 cache evictions/.test(bad.detail), bad.detail);
  assert.deepStrictEqual(bad.causes, v(W.settings.replay_artifact_causes));
  const ok = FM.replayArtifactCheck(data, realRow({ trace: traces[0], fresh_per_output: tag(ideal) }));
  assert.strictEqual(ok.status, 'PASS', ok.detail);
  assert.deepStrictEqual(ok.causes, []);
});

test('replay_artifact: a row naming no known trace is judged against the profile\'s reference prior; K is a setting; no F is UNKNOWN', () => {
  const ref = FM.freshReference(data, realRow({ trace: null }));
  assert.strictEqual(ref.basis, 'prior');
  assert.strictEqual(ref.name, v(prof.fresh_reference_prior));
  near(ref.v, FM.priorValue(data.priors.rows.find(p => p.id === ref.name)));
  const row = realRow({ trace: 'no-such-trace', fresh_per_output: tag(ref.v * (K + 0.5)) });
  assert.strictEqual(FM.replayArtifactCheck(data, row).status, 'FLAG');
  const d = clone(data); d.workload.settings.replay_artifact_k.v = K + 1;
  assert.strictEqual(FM.replayArtifactCheck(d, row).status, 'PASS', 'a larger K lets the same row pass');
  assert.strictEqual(FM.replayArtifactCheck(data, realRow({ fresh_per_output: undefined })).status, 'UNKNOWN');
});

test('replay_artifact: a fresh breakdown in the row is printed with its shares', () => {
  const c = FM.replayArtifactCheck(data, realRow({ trace: null, fresh_breakdown: { cold: 100, growth: 100, reread: 200 } }));
  assert.strictEqual(c.status, 'FLAG');
  assert(/fresh breakdown cold 100 \(25 %\), growth 100 \(25 %\), reread 200 \(50 %\)/.test(c.detail), c.detail);
});

test('steady_state: no warmup_s FLAGS a cold-start window (F upper bound); none is a NOTE; seconds PASS; the breakdown shows', () => {
  const miss = FM.steadyStateCheck(data, realRow());
  assert.strictEqual(miss.status, 'FLAG');
  assert(/cold-start window, F upper bound/.test(miss.detail) && /no fresh breakdown/.test(miss.detail), miss.detail);
  const none = FM.steadyStateCheck(data, realRow({ warmup_s: v(W.settings.warmup_none) }));
  assert.strictEqual(none.status, 'NOTE');
  assert(/cold-start window, F upper bound/.test(none.detail));
  const warm = FM.steadyStateCheck(data, realRow({ warmup_s: 120, fresh_breakdown: { cold: 0, growth: 300, reread: 100 } }));
  assert.strictEqual(warm.status, 'PASS', warm.detail);
  assert(/120 s of warm-up/.test(warm.detail) && /growth 300 \(75 %\)/.test(warm.detail), warm.detail);
  assert.strictEqual(FM.steadyStateCheck(data, realRow({ warmup_s: 'soon' })).status, 'FLAG');
});

test('prediction_first: a many-agent recipe the model cannot bind is FLAGGED; a fitted curve of its build names the term; a declared prediction counts', () => {
  const many = v(W.settings.many_agents_min), r = rec(32, { conditions: { engine_flags: 'layer split 25', workload: { agents: many } } });
  const none = FM.checkRecipe(withReal([]), r).checks.find(c => c.id === 'prediction_first');
  assert.strictEqual(none.status, 'UNKNOWN');
  assert(/^FLAG: no predicted binding term/.test(none.detail), none.detail);
  const d = withReal([realRow({ build_key: '00000000' })]);
  d.curves.configs = [cfg([[8, 200], [16, 270], [32, 356]], { build: 'image 00000000 (test)' })];
  const pf = FM.checkRecipe(d, r).checks.find(c => c.id === 'prediction_first');
  assert.strictEqual(pf.status, 'PASS', pf.detail);
  assert(['decode', 'prefill'].includes(pf.pred.binding) && pf.pred.useful_tps > 0 && pf.pred.fresh === v(prof.fresh_per_output), JSON.stringify(pf.pred));
  assert(new RegExp(pf.pred.binding.toUpperCase() + '-BOUND, assuming F = ').test(pf.detail), pf.detail);
  const own = rec(32, { conditions: { workload: { agents: many, prediction: { useful_tps: 40, binding: 'prefill', fresh: 24, src: 'test' } } } });
  const dec = FM.checkRecipe(withReal([]), own).checks.find(c => c.id === 'prediction_first');
  assert.strictEqual(dec.status, 'PASS');
  assert.strictEqual(dec.pred.basis, 'declared');
  assert.strictEqual(FM.checkRecipe(withReal([]), rec(32, { conditions: { workload: { agents: many - 1 } } })).checks.find(c => c.id === 'prediction_first'), undefined, 'a few-agent recipe is not judged');
});

test('serving priors: each applies only where its recipe facts hold; checkRecipe lists them; an unknown predicate is an error', () => {
  const ids = r => FM.priorsFor(data, r).map(p => p.id), many = v(W.settings.many_agents_min);
  const big = ids(rec(64, { conditions: { engine_flags: 'layer split 25', workload: { agents: 64 } } }));
  data.priors.rows.filter(p => p.applies_when.agent_traffic && Object.keys(p.applies_when).length === 1).forEach(p => assert(big.includes(p.id), p.id));
  assert(data.priors.rows.some(p => p.applies_when.layer_split) && data.priors.rows.filter(p => p.applies_when.layer_split).every(p => big.includes(p.id)), 'a layer split gets the pipelining prior');
  assert(data.priors.rows.filter(p => p.applies_when.draft).every(p => big.includes(p.id)), 'a draft head at 64 agents gets the MTP priors');
  const lone = ids(rec(1, { draft: false, conditions: { engine_flags: '--batch 1', workload: { agents: 1 } } }));
  assert.deepStrictEqual(lone, [], 'one agent, no split, no draft: no serving prior applies');
  const below = data.priors.rows.find(p => p.applies_when.ctx_below);
  if (below) {
    assert(ids(rec(many, { ctx: below.applies_when.ctx_below - 1 })).includes(below.id));
    assert(!ids(rec(many, { ctx: below.applies_when.ctx_below })).includes(below.id));
  }
  assert.deepStrictEqual(FM.checkRecipe(data, rec(64)).priors.map(p => p.id), ids(rec(64)));
  const d = clone(data); d.priors.rows = [{ id: 'x', statement: 's', value: 1, unit: 'u', tag: 'INFERRED', src: 's', applies_when: { phase_of_moon: true } }];
  assert.throws(() => FM.priorsFor(d, rec(64)), /no test in model.js PRIOR_TESTS/);
});

// the ledger's own real-agent rows (left out of a public copy): each named trace has its ideal, and each is judged
if (!W.measured.length) skip('the ledger\'s real-agent rows under the replay checks');
else test('the ledger\'s real-agent rows: each is judged by both replay checks; a row over K x its trace-ideal ratio is FLAGGED', () => {
  FM.replayChecks(data).forEach(x => {
    const [a, s] = x.checks, row = W.measured.find(y => y.run === x.run && y.n === x.n);
    // a row whose server reports no prompt tokens (the snoopy GLM seat) has no F: UNKNOWN, never a PASS on a missing number
    assert.strictEqual(a.status, a.fresh === null ? 'UNKNOWN' : a.fresh / a.ref.v > K ? 'FLAG' : 'PASS', x.run);
    assert(['PASS', 'NOTE', 'FLAG'].includes(s.status), x.run);
    if (row.trace && traces.includes(row.trace)) assert.strictEqual(a.ref.basis, 'trace-ideal', x.run);
  });
});

// ---------- per-turn agent model (turn_model, agentTurns, agentReplaySim, agentTurnRescore) ----------
test('agentTurns: one agent cycles in out / D1 + fresh / P + gap; many stalled agents reduce to agentThroughput serial', () => {
  const one = FM.agentTurns({ n: 1, out: 300, fresh: 2000, gap: 2, decode_1: 60, decode_n: 60, prefill: 1000, stall: true });
  near(one.cycle_s, 300 / 60 + 2 + 2);
  near(one.turns_per_agent_min, 60 / 9);
  near(one.useful_tps, 300 / 9);
  assert.strictEqual(one.binding, 'agents');
  const many = FM.agentTurns({ n: 64, out: 300, fresh: 2000, gap: 2, decode_1: 60, decode_n: 150, prefill: 1000, stall: true });
  assert.strictEqual(many.binding, 'server');
  near(many.useful_tps, FM.agentThroughput({ decode: 150, prefill: 1000, fresh: 2000 / 300, stall: true }).serial, 1e-9);
});

// a synthetic trace: [gap_s, out, fresh] per turn
const simIn = o => Object.assign({ n: 1, trace: [{ start_s: 0, turns: [[1, 100, 1000], [1, 100, 1000]] }], tok: [1], out_ratio: 1, reread: 0, D1: 50, Dn: 50, P: 1000,
  stall: true, gap_scale: 1, stagger: 0, overhead: 0, warmup_s: 0, window_s: 400 }, o || {});
test('agentReplaySim: one agent alone matches the closed form; tool gaps and re-reads move it the right way; the warm-up is not counted', () => {
  const a = FM.agentReplaySim(simIn());
  // each turn 100 / 50 + 1000 / 1000 = 3 s, then a 1 s gap: 100 turns in 400 s
  assert.strictEqual(a.turns, 100);
  near(a.turns_per_agent_min, 15);
  near(a.useful_tps, 100 * 100 / 400);
  near(a.share.gap, 0.25, 1e-9);
  assert(FM.agentReplaySim(simIn({ gap_scale: 0 })).turns > a.turns, 'no tool gaps: more turns');
  assert(FM.agentReplaySim(simIn({ reread: 1000 })).turns < a.turns, 'more re-read: fewer turns');
  assert.strictEqual(FM.agentReplaySim(simIn({ warmup_s: 2, window_s: 398 })).turns, 99, 'the turn sent before the warm-up is not counted');
});
test('agentReplaySim: many agents saturate at the server bound (1 / w turns a second, less the window edges)', () => {
  const n = 32, s = FM.agentReplaySim(simIn({ n, trace: Array.from({ length: n }, () => ({ start_s: 0, turns: [[0, 100, 1000]] })), tok: Array(n).fill(1), Dn: 200, window_s: 3000 }));
  const w = 100 / 200 + 1000 / 1000, bound = 3000 / w;
  assert(s.turns <= bound + 1e-9 && s.turns > bound - 2 * n, s.turns + ' vs bound ' + bound);
});

const TR = FM.agentTurnRescore(data);
test('turn rescore: every entry models (after and held-out), and the numbers stored at commit still hold', () => {
  assert(TR.length >= 3, 'the prediction files carry rescore blocks');
  TR.forEach(r => ['after', 'held_out'].forEach(k => {
    const x = r[k];
    assert.strictEqual(x.status, 'MODEL', r.id + ' ' + k + ': ' + x.why);
    assert(x.stored, r.id + ' ' + k + ': after_at_commit missing');
    near(x.turns, x.stored.turns_per_agent_min, 1e-3 * x.stored.turns_per_agent_min);
    near(x.useful, x.stored.useful_tps, 1e-3 * x.stored.useful_tps);
  }));
});
test('turn rescore: a held-out score uses no row measured on the scored run except the inputs it names in-sample', () => {
  const TM = W.turn_model;
  W.turn_rescore.forEach(e => {
    const runs = [].concat(e.run), p = FM.agentTurnPredict(data, e.query, e.run, 'held_out'), m = TM.models[e.query.model], S = TM.servers[e.query.server];
    const onlyHere = t => Object.values(t).every(x => (x.runs || []).some(r => runs.includes(r)));
    assert.strictEqual(p.bound === 'upper', onlyHere(S.reread_per_turn), e.id + ': an upper bound exactly when no held-out re-read exists');
    assert.strictEqual(p.inputs.some(i => i.basis === 'recording'), onlyHere(m.out_vs_recorded), e.id + ': output lengths fall back to the recording');
    const all = FM.agentTurnPredict(data, e.query, e.run, 'all');
    assert(p.in_sample.every(x => all.in_sample.includes(x)), e.id + ': held-out in-sample inputs are a subset');
  });
});
test('turn rescore: on the 1-agent runs the per-turn model lands within 10 % on turns, where the old formula missed by more', () => {
  TR.filter(r => r.n === 1).forEach(r => assert(Math.abs(r.after.err_turns) < 0.1 && Math.abs(r.after.err_turns) < Math.abs(r.before.err_useful), r.id + ' after ' + r.after.err_turns));
});
test('turn_model: every input carries a tag, a src and an evidence class', () => {
  const TM = W.turn_model, rows = [TM.settings.gap_scale, TM.settings.stagger_scale, TM.settings.client_overhead_s, TM.tokenizers.qwen.per_agent]
    .concat(Object.values(TM.tokenizers).filter(t => t.ratio).map(t => t.ratio))
    .concat(...Object.values(TM.models).map(m => Object.values(m.out_vs_recorded)), ...Object.values(TM.servers).map(s => Object.values(s.reread_per_turn)));
  rows.forEach(x => assert(x.tag && x.src && x.evidence, JSON.stringify(x).slice(0, 80)));
  Object.values(TM.traces).forEach(t => assert(t.tag === 'MEASURED' && t.v.length && t.v.every(a => a.turns.every(r => r.length === 3))));
});

// Consumer qualification uses a deterministic trace, not a fitted accuracy target.
const planFixture = () => {
  const d = withReal([realRow({ n: 1, prefill_tps: tag(1000) })]), T = d.workload.turn_model;
  d.workload.profiles['coding-agent'].turn_trace = tag('unit', 'READ');
  T.traces.unit = tag([{ start_s: 0, turns: [[1, 100, 1000]] }]);
  T.tokenizers.qwen.per_agent = tag([1]);
  T.models.fnx.out_vs_recorded = { 1: tag(1) };
  T.servers = { unit: { engine: 'strata', reread_per_turn: { 1: tag(0) } } };
  T.settings.client_overhead_s = tag(0);
  T.settings.plan_warmup_s = tag(0, 'READ');
  T.settings.plan_window_s = tag(400, 'READ');
  T.settings.gap_scale = tag(1, 'READ');
  T.settings.stagger_scale = tag(0, 'READ');
  return { d, c: cfg([[1, 50], [16, 200]]), opts: { n: [1] } };
};
test('agentPlan primary result is the existing replay primitive with explicit units and compatible legacy diagnostics', () => {
  const { d, c, opts } = planFixture(), p = FM.agentPlan(d, c, opts), r = p.rows[0], t = r.turn;
  assert.strictEqual(p.primary, 'turns_per_agent_min');
  assert.strictEqual(t.status, 'MODEL', t.why);
  near(t.turns_per_agent_min, 15);
  near(t.useful_tps, 25);
  near(t.out_per_turn, 100);
  const primitive = FM.agentTurnPredict(d, t.query, [], 'all');
  ['turns', 'turns_per_agent_min', 'useful_tps', 'fresh_tokens'].forEach(k => near(t[k], primitive[k]));
  near(t.share.gap, 0.25);
  assert.strictEqual(t.server_basis, 'unique engine prior (assumed, not build-verified)');
  Object.keys(r.legacy_server).forEach(k => assert.deepStrictEqual(r[k], r.legacy_server[k]));
  assert(t.inputs.every(x => x.tag && x.src && Array.isArray(x.runs)));
});
test('agentPlan responds to tool gaps, server re-read, explicit fresh and seconds overrides', () => {
  const { d, c, opts } = planFixture(), base = FM.agentPlan(d, c, opts).rows[0].turn;
  assert(FM.agentPlan(d, c, { ...opts, gap_scale: 0 }).rows[0].turn.turns > base.turns);
  d.workload.turn_model.servers.unit.reread_per_turn[1] = tag(1000);
  assert(FM.agentPlan(d, c, opts).rows[0].turn.turns < base.turns);
  const fresh = FM.agentPlan(d, c, { ...opts, fresh: 0 }).rows[0].turn;
  near(fresh.fresh_tokens, 0);
  assert.strictEqual(fresh.prefix_reread, 'replaced by --fresh');
  const window = FM.agentPlan(d, c, { ...opts, warmup_s: 2, window_s: 398, fresh: 10 }).rows[0].turn;
  assert.strictEqual(window.turns, 99);
  assert.strictEqual(window.warmup_s, 2);
  assert.strictEqual(window.window_s, 398);
});
test('agentPlan uses excluded solo decode points for D1 and labels an unmeasured solo curve', () => {
  const { d, c, opts } = planFixture();
  c.points[0].used = false;
  c.points.push({ ...c.points[1], n: 32, tps: tag(250) });
  const t = FM.agentPlan(d, c, opts).rows[0].turn;
  assert.strictEqual(t.query.decode_1.v, 50);
  assert.strictEqual(t.query.decode_1.tag, 'MEASURED');
  c.points = c.points.filter(p => p.n !== 1);
  const extrap = FM.agentPlan(d, c, opts).rows[0].turn;
  assert.strictEqual(extrap.query.decode_1.tag, 'FIT');
  assert.match(extrap.query.decode_1.src, /extrapolate/);
  assert.strictEqual(FM.agentPlan(d, c, { ...opts, decode_one: 80 }).rows[0].turn.query.decode_1.v, 80);
});
test('agentPlan keeps missing, ambiguous, cross-engine, decode-only and invalid inputs honest', () => {
  const { d, c, opts } = planFixture();
  const unknown = o => { const t = FM.agentPlan(d, c, { ...opts, ...o }).rows[0].turn; assert.strictEqual(t.status, 'UNMODELLED', JSON.stringify(t)); assert(t.why); assert.strictEqual(t.useful_tps, undefined); };
  unknown({ server: 'missing' }); unknown({ trace: 'missing' }); unknown({ prefill: 0 }); unknown({ n: [0] });
  unknown({ gap_scale: -1 }); unknown({ window_s: 0 }); unknown({ warmup_s: NaN }); unknown({ profile: 'synthetic-short' });
  d.workload.turn_model.servers.other = { engine: 'strata', reread_per_turn: { 1: tag(9999) } };
  unknown({});
  assert.strictEqual(FM.agentPlan(d, c, { ...opts, server: 'unit' }).rows[0].turn.status, 'MODEL');
  d.workload.turn_model.servers.other.engine = 'another-engine';
  unknown({ server: 'other' });
  delete d.workload.turn_model.servers.unit.reread_per_turn[1];
  const t = FM.agentPlan(d, c, opts).rows[0].turn;
  assert.strictEqual(t.bound, 'upper');
  assert(t.inputs.some(x => x.tag === 'UNKNOWN' && x.basis === 'unknown'));
});
test('held-out GLM miss and larger all-input misses remain visible without changing frozen predictions', () => {
  const glm = TR.find(x => x.id.includes('glm'));
  assert(glm, 'GLM retained rescore');
  near(glm.held_out.err_turns, -0.3, 0.005);
  assert(glm.held_out.err_useful > 2.8);
  assert(TR.some(x => x.after.err_turns < -0.085), 'no blanket five-percent claim');
});
test('actual UI renderer uses agentPlan numbers, shows every rescore, unknowns and source-safe text', () => {
  const { d, c } = planFixture(), src = fs.readFileSync(path.join(ROOT, 'src/app.js'), 'utf8');
  const start = src.indexOf('  function workloadHTML('), end = src.indexOf('  // what the replays', start);
  assert(start >= 0 && end > start);
  const esc = x => String(x).replace(/[&<>"']/g, x => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[x]));
  const fmt = n => x => x === null || x === undefined || !Number.isFinite(x) ? '-' : x.toFixed(n);
  const render = new Function('data', 'FM', 'v', 'f1', 'f2', 'esc', 'chip', 'groundingHTML', src.slice(start, end) + '; return workloadHTML;');
  const html = () => render(d, { ...FM, agentTurnRescore: () => TR }, v, fmt(1), fmt(2), esc, x => esc(x.tag), () => '')({ cfg: c, headline: { n: 1 } });
  assert.match(html(), /15\.00 turns\/agent-minute at 1 agent/);
  assert.match(html(), /25\.0 useful tok\/s/);
  assert.match(html(), /POST-HOC inputs, not new prediction accuracy/);
  assert.match(html(), /-30\.0%/);
  TR.forEach(x => assert(html().includes(esc(x.id)), x.id));
  d.workload.profiles['coding-agent'].turn_trace.v = '<missing>';
  assert.match(html(), /UNMODELLED/);
  assert(html().includes('&lt;missing&gt;'));
  assert(!html().includes('<missing>'));
});
test('CLI JSON binds the same agentPlan query, and text leads with turns and preserves the full error table', () => {
  const c = data.curves.configs[0];
  if (!c) return skip('CLI config binding');
  const { execFileSync } = require('child_process');
  const run = extra => execFileSync(process.execPath, [path.join(ROOT, 'report.js'), '--workload', c.id, '--n', '1', '--warmup-s', '2', '--window-s', '60', ...extra], { cwd: ROOT, encoding: 'utf8' });
  const p = JSON.parse(run(['--json'])), own = FM.agentPlan(data, c, { n: [1], warmup_s: 2, window_s: 60 });
  assert.deepStrictEqual(p.rows[0].turn, JSON.parse(JSON.stringify(own.rows[0].turn)));
  assert.strictEqual(p.turn_rescore.length, TR.length);
  const text = run([]);
  assert.match(text, /Primary: turns\/agent-minute/);
  assert.match(text, /Legacy server-bound diagnostic WITHOUT tool gaps/);
  assert.match(text, /GLM held-out turns miss by about -30%/);
  assert.match(text, /Configured-input means each entry's declared mixture, not all measured inputs/);
  assert.match(text, /Chunk rows retain advance predicted P, not the later measured P/);
  TR.forEach(x => assert(text.includes(x.id), x.id));
});

test('resident plan: the decode fit reproduces its two anchors; agents split over instances; slots cap Dn; the frozen flint-1030 numbers still hold', () => {
  const rec = JSON.parse(fs.readFileSync(path.join(ROOT, 'recipes', 'odie-flint-q4km-1perCard.json'), 'utf8')), R = rec.resident_plan;
  const one = Object.assign(clone(R), { inputs: Object.assign(clone(R.inputs), { engine_slowdown: tag(1, 'READ'), kv_read_eff: tag(1, 'READ') }) }), T = FM.residentTerms(one);
  v(R.inputs.decode_anchors).forEach(x => near(1000 / (x.read_gb * T.a_ms_per_gb + x.layers * T.layer_ms), x.tps, 1e-6));
  const t = FM.residentTerms(R);
  assert(t.decode(1, 80000) < t.decode(1, 0), 'KV at depth slows decode');
  assert(t.decode(2, 80000) > t.decode(1, 80000) && t.decode(2, 80000) < 2 * t.decode(1, 80000), 'two slots batch, sublinearly');
  const P = FM.residentAgentPlan(data, rec, [1, 4, 8]);
  assert.deepStrictEqual(P.rows.map(r => r.per_instance), [[1], [2, 2], [4, 4]]);
  assert.strictEqual(P.rows[0].decode_card_at_slots[0], P.rows[0].decode_per_request, 'one agent decodes alone');
  near(P.rows[2].decode_card_at_slots[0], t.decode(2, 80000), 1e-9);
  const frozen = JSON.parse(fs.readFileSync(path.join(ROOT, 'data', 'predictions-flint-1030.json'), 'utf8')).predictions;
  P.rows.forEach(r => { near(r.turns_per_agent_min, frozen[r.n].turns_per_agent_min, 0.006); near(r.useful_tps_per_agent, frozen[r.n].useful_tps_per_agent, 0.06); });
  const no = clone(rec); delete no.resident_plan.inputs.slots;
  assert.throws(() => FM.residentAgentPlan(data, no, [1]), /slots/);
});

test('resident plan levers: stall off, decode_mult, out_ratio and reread_per_turn reach the turn model; the fit1030 baseline holds its replay rows', () => {
  const rec = JSON.parse(fs.readFileSync(path.join(ROOT, 'recipes', 'odie-flint-q4km-1perCard-fit1030.json'), 'utf8'));
  const run = (set, ns) => { const r = clone(rec); Object.entries(set).forEach(([k, x]) => { r.resident_plan.inputs[k] = tag(x, 'LEVER'); }); return FM.residentAgentPlan(data, r, ns).rows; };
  const base = run({}, [1, 2, 4]);
  // with each run's own measured inputs the baseline lands within the pinned error of the flint-1030 replay rows
  near(run({ reread_per_turn: 1187 }, [1])[0].turns_per_agent_min, 8.1, 0.25);
  near(base[1].turns_per_agent_min, 9.2, 0.3);
  assert(run({ out_ratio: 0.325, reread_per_turn: 600 }, [4])[0].turns_per_agent_min > 4.6, 'RM is optimistic at 2 agents a card (stall model): a fix must update this pin');
  const free = run({ stall: false }, [4])[0], d = run({ decode_mult: 1.4 }, [1])[0];
  assert(free.turns_per_agent_min > base[2].turns_per_agent_min, 'reads that never stall decode help 2 agents a card');
  near(run({ stall: false }, [1])[0].turns_per_agent_min, base[0].turns_per_agent_min, 1e-9);
  near(d.decode_per_request, 1.4 * base[0].decode_per_request, 1e-9);
  const long = run({ out_ratio: 1 }, [1])[0];
  near(long.out_per_turn / base[0].out_per_turn, 4, 0.6);
  assert(long.turns_per_agent_min < base[0].turns_per_agent_min);
  assert(run({ reread_per_turn: 0 }, [1])[0].turns_per_agent_min > base[0].turns_per_agent_min);
  assert(FM.RESIDENT_LEVERS.includes('stall'));
  // the CLI --set reaches the same plan, and refuses an empty or unknown value instead of a silent 0
  const { spawnSync } = require('child_process'), R = path.join(ROOT, 'recipes', 'odie-flint-q4km-1perCard-fit1030.json');
  const cli = set => spawnSync(process.execPath, [path.join(ROOT, 'report.js'), '--resident', R, '--n', '4', '--set', set, '--json'], { cwd: ROOT, encoding: 'utf8' });
  near(JSON.parse(cli('stall=false').stdout).rows[0].turns_per_agent_min, free.turns_per_agent_min, 1e-9);
  ['out_ratio=', 'reread_per_turn', 'stall=maybe', 'nosuch=1'].forEach(s => assert.strictEqual(cli(s).status, 2, '--set ' + s + ' must exit 2'));
});

console.log(passed + ' passed');
