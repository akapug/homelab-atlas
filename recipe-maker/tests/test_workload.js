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
    assert.strictEqual(a.status, a.fresh / a.ref.v > K ? 'FLAG' : 'PASS', x.run);
    assert(['PASS', 'NOTE', 'FLAG'].includes(s.status), x.run);
    if (row.trace && traces.includes(row.trace)) assert.strictEqual(a.ref.basis, 'trace-ideal', x.run);
  });
});

console.log(passed + ' passed');
