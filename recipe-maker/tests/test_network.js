'use strict';
// Pure synthetic laws plus actual catalog/report consumers. No device, model or service operation.
const assert = require('assert'), fs = require('fs'), path = require('path'), FM = require('../src/model.js');
const tag = (v, kind = 'ESTIMATE') => ({ v, tag: kind, src: 'synthetic network test fixture, not a hardware observation' });
const clone = x => JSON.parse(JSON.stringify(x));
function fixture(shape = 'pipeline') {
  const identity = { mode: tag('fixture-fast'), evidence_config: tag('fixture-context-build-layout'), metric: tag('decode_tokens_s') };
  const box = { cpu: { threads: tag(8) }, dram: { total_gib: tag(64), host_floor_gib: tag(4) }, gpus: [{ id: 'fixture-card', vram_gib: tag(16), other: [] }] };
  const W = { key: 'fixture-window', box: 'alpha', model: 'fixture-model', engine: 'fixture-engine', settings: { plan_agents: tag(4) },
    profile: { point: 'fixture-profile', ...clone(identity) }, frontier: { tps: tag(100), noise_pct: tag(3), ledger: { id: 'fixture-baseline', tps: 'c4' }, ...clone(identity) } };
  const transfers = [{ observation: 'beta-alpha', from: 'beta', to: 'alpha', from_memory: 'host', to_memory: 'host', bytes: tag(125000000), rtt_ms: tag(2), round_trips: tag(1) }];
  const resources = ['alpha', 'beta'].map(b => ({ box: b, baseline_host_gib: tag(0), extra_host_gib: tag(1), other_host_gib: tag(0), cpu_threads: tag(2), host_only: tag(true), gpu_gib: {} }));
  const power = ['alpha', 'beta'].map(b => ({ box: b, idle_w: tag(50), cpu_incremental_w: tag(10), gpu_incremental_w: tag(20) }));
  const values = { stage_ms: [40, 50], rows: 2, groups: 2, eta: 1, prompt_tokens: 1000, local_prefill_tps: 100, remote_prefill_tps: 200, local_decode_pause_s: 10, remote_decode_pause_s: 1,
    draft_tps: 100, draft_tokens: 2, acceptance: 0.5, verify_base_ms_at_n: 40, verify_extra_ms_per_token_at_n: 5, kv_bytes_per_slot: 125000000, parked_slots: 2,
    summary_tokens: 100, summary_tps: 20, reread_tps: 100, text_bytes_per_token: 4 };
  const recipe = { id: 'fixture-' + shape, label: 'Synthetic ' + shape, window: W.key, shape, boxes: ['alpha', 'beta'], agents: tag(4), ...clone(identity),
    backend_supported: tag(true), fidelity: tag(true, 'MEASURED'), params: Object.fromEntries(Object.entries(values).map(([k, v]) => [k, tag(v)])), transfers, resources, power, stage_boxes: ['beta', 'alpha'], remote_box: 'beta' };
  const observation = (id, from, to) => ({ id, from, to, transport: 'tcp', version: 'synthetic-v1', streams: 1, protocol: 'synthetic test', gbps: tag(10) });
  return { meta: { evidence: { by_tag: { ESTIMATE: 'inferred', MEASURED: 'seen', UNKNOWN: 'unknown' } } }, boxes: { alpha: clone(box), beta: clone(box) }, recipes: [], models: {}, engines: {}, checks: { host_pinned: { default_margin_gib: tag(2) }, cpu_reserve: { default_reserve: tag(1) } }, window: { default: W.key, configs: { [W.key]: W } },
    network: { schema: 'synthetic-network-v1', recipes: [recipe], observations: [observation('beta-alpha', 'beta', 'alpha'), observation('alpha-beta', 'alpha', 'beta')], pcie: ['alpha', 'beta'].map(b => ({ box: b, card: 'fixture-card', h2d_gbs: tag(5), d2h_gbs: tag(4) })) } };
}
const result = d => FM.winNetwork(d, d.window.default)[0];
let passed = 0;
const test = (name, f) => { f(); passed++; console.log('ok ' + name); };
function tests() {
  test('directed TCP transfer uses decimal bit rate and explicit RTT', () => {
    const d = fixture(), t = FM.networkTransfer(d, d.network.recipes[0].transfers[0]);
    assert(Math.abs(t.seconds - 0.102) < 1e-12);
    const bad = clone(d.network.recipes[0].transfers[0]); bad.from = 'alpha'; bad.to = 'beta';
    assert.strictEqual(FM.networkTransfer(d, bad).seconds, null);
    assert(FM.networkTransfer(d, bad).checks.some(c => c.id === 'direction'));
  });
  test('PCIe host staging adds both actual per-card transfer costs', () => {
    const d = fixture(), t = d.network.recipes[0].transfers[0];
    Object.assign(t, { from_memory: 'gpu', to_memory: 'gpu', from_card: 'fixture-card', to_card: 'fixture-card' });
    assert(Math.abs(FM.networkTransfer(d, t).seconds - (0.102 + 125000000 / 4e9 + 125000000 / 5e9)) < 1e-12);
    d.network.pcie[0].h2d_gbs = tag(null, 'UNKNOWN'); assert.strictEqual(FM.networkTransfer(d, t).seconds, null);
  });
  test('missing RTT, memory path and stream/version mismatches do not get optimistic prices', () => {
    for (const field of ['rtt_ms', 'from_memory']) { const d = fixture(); delete d.network.recipes[0].transfers[0][field]; assert.strictEqual(result(d).eligible, false); }
    const d = fixture(); d.network.recipes[0].transfers[0].streams = 8; assert.strictEqual(result(d).status, 'FAIL');
  });
  test('pipeline composes existing slowest-stage/group law with serialized hop service', () => {
    const d = fixture(), r = result(d);
    assert.strictEqual(r.status, 'MODELED'); assert.strictEqual(r.eligible, true);
    assert(r.metrics.stage_ms.every((x, i) => Math.abs(x - [142, 50][i]) < 1e-10));
    assert(Math.abs(r.metrics.decode_tps - FM.winLaw([142, 50], 2, 2, 1)) < 1e-10);
    d.network.recipes[0].stage_boxes.reverse(); assert.strictEqual(result(d).status, 'FAIL');
  });
  test('kernel and network rows share one existing frontier, but modes never mix', () => {
    const d = fixture(), W = d.window.configs[d.window.default], L = { rebalanced: { total: 100 }, levers: [{ lever: 'fixture-kernel', kind: 'fixture', total_rebalanced: 102 }], unmeasured: [] };
    const F = FM.winFrontier(W, L, [result(d)], d);
    assert.strictEqual(F.tps, 100); assert.strictEqual(F.rank[0].at_frontier, 102); assert.strictEqual(F.shared_rank.length, 2);
    W.frontier.mode = tag('fixture-slow');
    const blocked = FM.winFrontier(W, L, [result(d)], d);
    assert.strictEqual(blocked.shared_rank.length, 0); assert.strictEqual(blocked.rank[0].at_frontier, 102);
    assert(result(d).comparison.why.some(x => x.includes('mode differs')));
  });
  test('resource admission rejects aggregate host, concurrent CPU and household GPU overcommit', () => {
    const host = fixture(); host.network.recipes[0].resources[0].extra_host_gib = tag(60); assert.strictEqual(result(host).status, 'FAIL');
    const cpu = fixture(); cpu.network.recipes[0].resources[1].cpu_threads = tag(8); assert.strictEqual(result(cpu).status, 'FAIL');
    const gpu = fixture(), r = gpu.network.recipes[0].resources[1]; delete r.host_only; r.gpu_gib = { 'fixture-card': tag(17) }; assert.strictEqual(result(gpu).status, 'FAIL');
    const missing = fixture(); missing.network.recipes[0].resources.pop(); assert.strictEqual(result(missing).eligible, false);
  });
  test('all-box power includes lent idle once, with unknown power preserving unknown Joules', () => {
    const d = fixture(), r = result(d); assert.strictEqual(r.power.watts, 160); assert.strictEqual(r.modeled_per_watt, r.metrics.decode_tps / 160); assert.strictEqual(r.useful_work_per_joule, null);
    d.network.recipes[0].power.pop(); assert.strictEqual(result(d).power.watts, null); assert.strictEqual(result(d).modeled_per_watt, null);
    const duplicate = fixture(); duplicate.network.recipes[0].power[1].box = 'alpha'; assert.strictEqual(result(duplicate).power.watts, null);
  });
  test('prefill reports shipping and decode pause, not throughput without explicit cadence', () => {
    const d = fixture('prefill'), r = result(d); assert.strictEqual(r.metrics.local_ttft_s, 10); assert.strictEqual(r.metrics.remote_ttft_s, 5.102);
    assert.strictEqual(r.metrics.pause_saved_s, 9); assert.strictEqual(r.eligible, false);
    Object.assign(d.network.recipes[0].params, { requests_per_s: tag(0.01), baseline_includes_prefill: tag(true) });
    assert.strictEqual(result(d).eligible, true); assert(Math.abs(result(d).metrics.decode_tps - 110) < 1e-10);
  });
  test('draft prices target bonus, conditional prefix survival and verification at exact N', () => {
    const d = fixture('draft'), r = result(d);
    assert.strictEqual(r.metrics.accepted_per_agent_cycle, 1.75);
    assert(Math.abs(r.metrics.cycle_s - 0.232) < 1e-12); assert(Math.abs(r.metrics.decode_tps - 7 / 0.232) < 1e-10);
    d.network.recipes[0].params.acceptance = tag(1); assert.strictEqual(result(d).metrics.accepted_per_agent_cycle, 3);
    d.network.recipes[0].agents = tag(8); assert.strictEqual(result(d).status, 'FAIL');
  });
  test('parked capacity needs admitted bytes and two directions; never becomes tok/s', () => {
    const d = fixture('parked_kv'), r = d.network.recipes[0];
    r.transfers.unshift({ ...clone(r.transfers[0]), observation: 'alpha-beta', from: 'alpha', to: 'beta' });
    const x = result(d); assert.strictEqual(x.status, 'MODELED'); assert.strictEqual(x.metrics.extra_slots, 2); assert.strictEqual(x.eligible, false); assert.strictEqual(x.at_frontier, null);
    r.resources[1].extra_host_gib = tag(0); assert.strictEqual(result(d).status, 'FAIL');
  });
  test('compaction preserves dimensions, reread cost and measured fidelity gate', () => {
    const d = fixture('compaction'), r = d.network.recipes[0];
    r.transfers.unshift({ ...clone(r.transfers[0]), observation: 'alpha-beta', from: 'alpha', to: 'beta' });
    const x = result(d); assert.strictEqual(x.metrics.saved_generation_slot_s, 4); assert.strictEqual(x.metrics.baseline_odie_slot_s, 14); assert.strictEqual(x.metrics.reread_slot_s, 1); assert.strictEqual(x.metrics.odie_slot_s_saved, 13); assert.strictEqual(x.metrics.credited_slot_s_saved, 13); assert.strictEqual(x.eligible, false);
    r.resources[0].extra_host_gib = tag(70); assert.strictEqual(result(d).status, 'FAIL'); assert.strictEqual(result(d).metrics.credited_slot_s_saved, null);
    r.resources[0].extra_host_gib = tag(1); r.mode = tag('fixture-slow'); assert.strictEqual(result(d).metrics.credited_slot_s_saved, null);
    r.mode = tag('fixture-fast'); r.fidelity = tag(null, 'UNKNOWN'); assert.strictEqual(result(d).status, 'UNKNOWN'); assert.strictEqual(result(d).metrics.credited_slot_s_saved, null);
  });
  test('priced identity binds link, PCIe, mode, resource and power inputs at the normal gate', () => {
    const d = fixture(), x = result(d), W = d.window.configs[d.window.default];
    const r = { id: 'fixture-launch', kind: 'network', box: W.box, model: W.model, engine: W.engine, network_window: W.key, network_recipe: x.id, network_prediction_identity: clone(x.prediction_identity) };
    assert(FM.checkRecipe(d, r).checks.some(c => c.id === 'network_identity' && c.status === 'PASS'));
    d.network.observations[0].gbps = tag(9);
    const stale = FM.checkRecipe(d, r);
    assert(stale.checks.some(c => c.id === 'network_identity' && c.status === 'FAIL'));
    assert.strictEqual(stale.prediction.status, 'UNMODELLED');
    assert.strictEqual(stale.prediction.decode_tps, null);
    assert.strictEqual(FM.predictRecipe(d, r).decode_tps, null);
  });
  test('changing the recipe law invalidates gate and fold-back identity even when all numeric inputs stay unchanged', () => {
    const d = fixture(), before = result(d), W = d.window.configs[d.window.default];
    const r = { id: 'fixture-launch', kind: 'network', box: W.box, model: W.model, engine: W.engine, network_window: W.key, network_recipe: before.id, network_prediction_identity: clone(before.prediction_identity) };
    W.attempts = [{ id: 'fixture-attempt', network_recipe: before.id, outcome: 'win', classes: ['*'], network_receipt: { run: 'fixture-run', recipe: before.id, identity: clone(before.prediction_identity), measured_identity: clone(before.prediction_identity), outcome: 'ok' } }];
    assert.strictEqual(result(d).attempts[0].matches_priced_identity, true);
    d.network.recipes[0].shape = 'draft';
    const after = result(d), stale = FM.checkRecipe(d, r);
    assert.strictEqual(after.status, 'MODELED'); assert.strictEqual(after.eligible, true);
    assert.notStrictEqual(after.metrics.decode_tps, before.metrics.decode_tps);
    assert.notDeepStrictEqual(after.prediction_identity, before.prediction_identity);
    assert(stale.checks.some(c => c.id === 'network_identity' && c.status === 'FAIL'));
    assert.strictEqual(stale.prediction.status, 'UNMODELLED'); assert.strictEqual(stale.prediction.decode_tps, null);
    assert.strictEqual(after.attempts[0].matches_priced_identity, false);
    assert.strictEqual(after.attempts[0].outcome, 'win');
  });
  test('normal gate reprices the selected nondefault N and ubench source rather than borrowing the default profile', () => {
    const d = fixture(), W = d.window.configs[d.window.default]; W.ubench = { mode: tag('fixture-fast'), evidence_config: tag('fixture-context-build-layout'), metric: tag('decode_tokens_s') };
    Object.assign(d.network.recipes[0], { agents: tag(8) }); d.network.recipes[0].params.groups = tag(4);
    assert.strictEqual(FM.winNetwork(d, W.key, { agents: 8, source: 'ubench' })[0].eligible, false);
    W.frontier.ledger.tps = 'c8';
    const x = FM.winNetwork(d, W.key, { agents: 8, source: 'ubench' })[0];
    const r = { id: 'fixture-launch', kind: 'network', box: W.box, model: W.model, engine: W.engine, network_window: W.key, network_recipe: x.id, network_prediction_identity: clone(x.prediction_identity) };
    assert.strictEqual(x.eligible, true); assert(FM.checkRecipe(d, r).checks.some(c => c.id === 'network_identity' && c.status === 'PASS'));
    r.network_prediction_identity.agents = 0; assert.strictEqual(FM.checkRecipe(d, r).bad_input, true);
  });
  test('fold-back matches exact ledger identity and retains an old loss without laundering it into a current win', () => {
    const d = fixture(), before = result(d), W = d.window.configs[d.window.default];
    W.attempts = [{ id: 'fixture-attempt', network_recipe: before.id, outcome: 'loss', classes: ['*'], network_receipt: { run: 'fixture-run', recipe: before.id, identity: clone(before.prediction_identity), measured_identity: clone(before.prediction_identity), outcome: 'ok' } }];
    assert.strictEqual(result(d).attempts[0].matches_priced_identity, true);
    d.network.recipes[0].params.stage_ms = tag([39, 50]);
    assert.strictEqual(result(d).attempts[0].matches_priced_identity, false);
    assert.strictEqual(result(d).attempts[0].outcome, 'loss');
    assert.deepStrictEqual(FM.winEvidence(W, { kind: 'fixture', classes: ['fixture-kernel'] }), []);
  });
  test('CLI engine-wide kernel evidence excludes network attempts while retaining ordinary kernel attempts', () => {
    const source = fs.readFileSync(path.join(__dirname, '..', 'report.js'), 'utf8');
    const start = source.indexOf('  const tag = a => a.id'), end = source.indexOf('  console.log(`  rank  lever', start);
    assert(start >= 0 && end > start, 'actual CLI engine-wide evidence renderer must exist');
    const render = new Function('W', 'R', 'console', source.slice(start, end));
    const kernel = { id: 'fixture-kernel-attempt', classes: ['*'], outcome: 'win' };
    const network = { id: 'fixture-network-attempt', classes: ['*'], network_recipe: 'fixture-pipeline', outcome: 'loss' };
    const lines = [];
    render({ attempts: [kernel, network] }, {}, { log: x => lines.push(x) });
    assert.strictEqual(lines.length, 1); assert(lines[0].includes('fixture-kernel-attempt win'));
    assert(!lines[0].includes('fixture-network-attempt'));
    const networkOnly = [];
    render({ attempts: [network] }, {}, { log: x => networkOnly.push(x) });
    assert.deepStrictEqual(networkOnly, []);
  });
  test('unrecognized evidence and forgotten participants cannot buy frontier eligibility', () => {
    for (const evidence of ['MADE_UP', 'toString']) { const d = fixture(); d.network.recipes[0].params.eta = tag(1, evidence); assert.strictEqual(result(d).eligible, false); }
    const protocol = fixture(); delete protocol.network.observations[0].protocol; assert.strictEqual(result(protocol).eligible, false);
    const mismatch = fixture(); mismatch.network.recipes[0].transfers[0].protocol = 'another benchmark'; assert.strictEqual(result(mismatch).status, 'FAIL');
    const missing = fixture(); missing.network.recipes[0].boxes = ['alpha']; assert.strictEqual(result(missing).status, 'FAIL');
    const zero = fixture(); zero.network.recipes[0].transfers[0].round_trips = tag(0); assert.strictEqual(result(zero).status, 'FAIL');
  });
  test('browser assumptions cannot overwrite the measured one-box frontier or captured profile costs', () => {
    const d = fixture(), W = d.window.configs[d.window.default], identity = Object.fromEntries(['mode', 'evidence_config', 'metric'].map(k => [k, clone(W.profile[k])]));
    const edit = { catalog: { ...clone(d.network), settings: {} }, profile_identity: clone(identity), frontier_identity: clone(identity) };
    const next = FM.networkAssumptions(d, W.key, edit); assert.strictEqual(next.window.configs[W.key].frontier.tps.v, 100);
    edit.frontier_identity.tps = tag(10000); assert.throws(() => FM.networkAssumptions(d, W.key, edit), /baseline\/cost/);
    delete edit.frontier_identity.tps; edit.profile_identity.terms = []; assert.throws(() => FM.networkAssumptions(d, W.key, edit), /baseline\/cost/);
    assert.strictEqual(W.frontier.tps.v, 100);
  });
  test('frontier and cadence reject invented tags, and receipt binds frontier identity', () => {
    const d = fixture(), W = d.window.configs[d.window.default], before = result(d);
    W.frontier.mode.src += ' changed capture'; assert.notDeepStrictEqual(result(d).prediction_identity, before.prediction_identity);
    W.profile.mode = tag('fixture-fast', 'MADE_UP'); W.frontier.mode = clone(W.profile.mode);
    const L = { rebalanced: { total: 100 }, levers: [{ lever: 'fixture-kernel', kind: 'fixture', total_rebalanced: 102 }], unmeasured: [] };
    assert.strictEqual(FM.winFrontier(W, L, [], d).shared_rank.length, 0);
    const p = fixture('prefill'); Object.assign(p.network.recipes[0].params, { requests_per_s: tag(0.01), baseline_includes_prefill: tag(true, 'MADE_UP') });
    assert.strictEqual(result(p).eligible, false);
  });
  const file = path.join(__dirname, '..', 'data.json'), built = fs.existsSync(file) ? JSON.parse(fs.readFileSync(file, 'utf8')) : null;
  if (!built || !built.network || !built.window.default) console.log('SKIP actual private catalog consumer: source catalog absent in this export/build');
  else test('actual build catalog -> report -> shared frontier preserves the historical headline and unknown gates', () => {
    const d = built, source = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data', 'interconnects.json'), 'utf8'));
    const R = FM.winReport(d, d.window.default);
    assert.strictEqual(R.network.length, source.recipes.length); assert(R.network.every(x => !x.eligible && x.status !== 'MODELED'));
    assert.deepStrictEqual(d.network.observations.map(x => x.id), source.observations.map(x => x.id));
    source.observations.forEach((x, i) => {
      const observed = d.network.observations[i];
      for (const field of ['from', 'to', 'transport', 'version', 'protocol', 'streams']) assert.strictEqual(observed[field], x[field]);
      for (const field of ['v', 'tag', 'src']) assert.strictEqual(observed.gbps[field], x.gbps[field]);
    });
    assert.strictEqual(R.frontier.tps, FM.v(d.window.configs[d.window.default].frontier.tps)); assert.strictEqual(R.frontier.shared_rank.length, 0);
    assert(d.network.pcie.every(x => FM.v(x.h2d_gbs) === null && FM.v(x.d2h_gbs) === null));
  });
  console.log(passed + ' tests passed');
}
module.exports = { fixture, tag, clone };
if (require.main === module) tests();
