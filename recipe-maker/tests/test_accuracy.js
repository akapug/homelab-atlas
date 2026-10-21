#!/usr/bin/env node
// predictRecipe on concrete recipes, and the accuracy math behind report.js --accuracy. Run: node tests/test_accuracy.js
// (needs data.json: python3 build.py)
'use strict';
const assert = require('assert');
const fs = require('fs'), path = require('path');
const ROOT = path.join(__dirname, '..');
const FM = require(path.join(ROOT, 'src', 'model.js'));
const ACC = require(path.join(ROOT, 'src', 'accuracy.js'));
const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'data.json'), 'utf8'));
FM.applyCal(data);
const recipe = f => JSON.parse(fs.readFileSync(path.join(ROOT, 'recipes', f), 'utf8'));
let n = 0;
const t = (name, fn) => { fn(); n++; console.log('ok ' + name); };

t('predictRecipe: droopy V4.1 split with draft and dram-fill', () => {
  const p = FM.predictRecipe(data, recipe('droopy-b1b-hot2p2-t20.json'));
  assert.strictEqual(p.status, 'PREDICTED');
  assert.ok(p.decode_tps > p.decode_tps_nodraft, 'the draft speeds decode up');
  assert.ok(Math.abs(p.decode_tps_nodraft - 1000 / p.ms_per_token.total) < 1e-9, 'no-draft decode = 1 / time per token');
  assert.ok(p.decode_tps > 5 && p.decode_tps < 12, 'decode in the measured range of the B0 family: ' + p.decode_tps);
  assert.strictEqual(p.prefill_tps, data.prefill.cpu_tps['droopy|v41|ik|4096'].banks_on.v, 'no offload_batch, banks 2.24: the CPU prefill path at its banks-on rate');
  assert.strictEqual(p.prefill.offload_on, false);
  const gpu = p.tiers.find(x => x.id === 'gpu0');
  assert.ok(gpu.experts_gib <= 2.24 + 1e-9, 'GPU banks stay within banks_gib');
  assert.ok(p.vram_peak_range_gib[0] < p.vram_peak_range_gib[1]);
  const le = data.checks.numa_free.load_extra['droopy|v41|ik|banks<=2.25'].v;
  assert.strictEqual(p.node_memfree_min_gib, 10 - le, 'dram-fill headroom 10 minus the measured load extra for banks <= 2.25');
  assert.ok(p.assumptions.some(a => /alloc3/.test(a)) && p.assumptions.some(a => /headroom 10/.test(a)));
  const share = p.tiers.reduce((a, x) => a + x.read_share, 0);
  assert.ok(Math.abs(share - 1) < 1e-6, 'read shares sum to 1: ' + share);
});

t('predictRecipe: snoopy GLM per-expert mix on glm-lite, and the build gate', () => {
  const r = recipe('snoopy-glm-p4-s25.json');
  const a = FM.predictRecipe(data, Object.assign({}, r, { engine_build: 'p512-lane', banks_experts: 1600, ram_tier_gib: 107 }));
  assert.strictEqual(a.status, 'PREDICTED');
  assert.ok(Math.abs(a.lowbit_share - 0.304) < 0.01, 'coldest experts at 2.05 bpw up to the requested read share: ' + a.lowbit_share);
  const b = FM.predictRecipe(data, Object.assign({}, r, { engine_build: 'p512-lane', banks_experts: 1600, ram_tier_gib: 107, store: { split: false, types: ['exl3_3.05'] } }));
  assert.ok(a.decode_tps > b.decode_tps, 'fewer bytes per token decode faster');
  const c = FM.predictRecipe(data, Object.assign({}, r, { engine_build: 'avx2-lane' }));
  assert.strictEqual(c.status, 'UNMODELLED');
  assert.strictEqual(c.decode_tps, null);
  assert.ok(/not a calibrated build/.test(c.why));
});

t('predictRecipe: UNMODELLED engines and HDD reads give no number', () => {
  const s = FM.predictRecipe(data, recipe('snoopy-flash-next-strata-q8.json'));
  assert.strictEqual(s.status, 'UNMODELLED');
  assert.strictEqual(s.decode_tps, null);
  const h = FM.predictRecipe(data, recipe('regress-droopy-b1017-dry.json'));
  assert.strictEqual(h.status, 'UNMODELLED');
  assert.ok(/rotational/.test(h.why));
});

t('checkRecipe carries the prediction; checkRecipes overlays a run observation', () => {
  const r = recipe('droopy-b1b-hot3p0-t40.json');
  const res = FM.checkRecipe(data, r);
  assert.strictEqual(res.prediction.status, 'PREDICTED');
  const before = res.checks.find(c => c.id === 'numa_free').status;
  const d2 = Object.assign({}, data, { recipes: [{ id: 'x', recipe: r, observed: { numa: { observed_memfree_gib: 7 } } }] });
  const after = FM.checkRecipes(d2)[0].checks.find(c => c.id === 'numa_free');
  assert.strictEqual(before, 'FAIL');
  assert.strictEqual(after.status, 'PASS');
  assert.ok(/OBSERVED/.test(after.detail));
});

t('mountKind: longest probed mount prefix; "~/" resolves through the box home mount, no user name in code', () => {
  const box = { disks: [{ mount: '/', kind: 'nvme' }, { mount: '/home', kind: 'sata-ssd' }, { mount: '/scratch', kind: 'hdd' }, { mount: '/mnt/a', kind: 'hdd' }, { mount: '/mnt/a/b', kind: 'nvme' }] };
  assert.strictEqual(FM.mountKind(box, '/mnt/a/b/c.gguf').mount, '/mnt/a/b');
  assert.strictEqual(FM.mountKind(box, '/mnt/ab/c.gguf').mount, '/', 'a prefix must end at a path separator');
  assert.strictEqual(FM.mountKind(box, '~/models/x.gguf').kind, 'sata-ssd', 'no home_mount: the mount holding /home');
  assert.strictEqual(FM.mountKind(box, '~/scratch/x').kind, 'sata-ssd', '~/scratch is under home, not the /scratch mount');
  const b2 = Object.assign({}, box, { home_mount: { v: '/scratch', tag: 'MEASURED', src: 'probe' } });
  assert.strictEqual(FM.mountKind(b2, '~/x').kind, 'hdd', 'the probed home mount wins');
  assert.strictEqual(FM.mountKind(b2, '~/x').path, '~/x', 'the path is reported as given');
  assert.strictEqual(FM.mountKind({ disks: [] }, '/x').kind, 'unknown');
  assert.ok(!/\/home\/[a-z]/.test(fs.readFileSync(path.join(ROOT, 'src', 'model.js'), 'utf8')), 'no user home path in model.js');
});

const ledger = () => require('../src/ledger.js').rows(path.join(ROOT, 'runs', 'runs.jsonl'));
const runRecipe = id => ledger().find(r => r.id === id).recipe;
const chk = (r, id) => FM.checkRecipe(data, r, null, { predict: false }).checks.find(c => c.id === id);

t('vram units: MiB / 1000 recipe figures convert to GiB; the card total then matches the box table', () => {
  const g = FM.vramGiB({ post_warmup_gib: 21.56, gpu_total_gib: 23.04, units: 'nvidia-smi MiB / 1000' });
  assert.ok(g.converted && Math.abs(g.total - 22.5) < 1e-9 && Math.abs(g.post - 21.05469) < 1e-4);
  assert.deepStrictEqual(FM.vramGiB({ post_warmup_gib: 22.49, gpu_total_gib: 23.99 }), { post: 22.49, peak: null, total: 23.99, converted: false });
  assert.ok(Math.abs(data.boxes.droopy.gpus[0].vram_gib.v - 22.5) < 1e-9, 'droopy box table is GiB');
  const e2 = chk(runRecipe('droopy-v41-e2'), 'vram_peak');
  assert.notStrictEqual(e2.status, 'FAIL', 'E2 ran: its predicted post-warm-up in GiB fits the 22.50 GiB card: ' + e2.detail);
  assert.ok(/of 22\.50 GiB/.test(e2.detail));
  const m = data.checks.vram_peak.margin.ik_split;
  assert.ok(Math.abs(m.hi - (23.04 - 21.56) * 1000 / 1024) < 1e-3 && Math.abs(m.lo - 0.5 * 1000 / 1024) < 1e-3, 'margin bounds in GiB');
});

t('numa_free + DRAM cap: a per-node headroom list (node 0 carries the load extra)', () => {
  const r = Object.assign(recipe('droopy-b1b-hot2p2-t20.json'), { numa: { headroom_gib: [13, 7] } });
  const one = FM.predictRecipe(data, Object.assign({}, r, { numa: { headroom_gib: 10 } })), two = FM.predictRecipe(data, r);
  const dram = p => p.tiers.filter(x => x.id === 'dram').reduce((a, x) => a + x.experts_gib, 0);
  assert.ok(Math.abs(dram(one) - dram(two)) < 1e-6, 'headroom 13 + 7 holds the same DRAM experts as 10 + 10');
  assert.strictEqual(two.node_memfree_gib[0], 13 - data.checks.numa_free.load_extra['droopy|v41|ik|banks<=2.25'].v, 'node 0 gets its own headroom minus the load extra');
  assert.strictEqual(two.node_memfree_gib[1], 7, 'node 1: its headroom, no load extra');
  assert.ok(/N0 13 \/ N1 7/.test(chk(r, 'numa_free').detail));
});

t('numa_free: load extra by load style, node 0 only; chain loads use the measured band', () => {
  const le = data.checks.numa_free.load_extra;
  assert.strictEqual(le['droopy|v41|ik|banks<=2.25'].v, 10.55);
  assert.strictEqual(le['droopy|v41|ik|banks<=2.25'].lo, 9.41);
  assert.ok(le['droopy|v41|ik|chain'].lo < le['droopy|v41|ik|chain'].v);
  const d1g = chk(runRecipe('droopy-v41-d1g-q2d-p0'), 'numa_free');
  assert.strictEqual(d1g.status, 'UNKNOWN', 'one-chain d1g ran fine: inside the band, not a FAIL: ' + d1g.detail);
  assert.ok(/\|chain\]/.test(d1g.detail));
  assert.strictEqual(chk(runRecipe('droopy-v41-b1d'), 'numa_free').status, 'PASS', 'B1d ran clean: node 1 carries no load extra');
  assert.strictEqual(chk(runRecipe('droopy-v41-b1b-1'), 'numa_free').status, 'FAIL', 'B1b (split, 3.0 GiB banks, headroom 10) starved node 0');
  const b0 = chk(runRecipe('droopy-v41-b0c-nortr'), 'numa_free');
  assert.ok(b0.status === 'FAIL' && /banks<=2\.25/.test(b0.detail), 'B0 ran clean only with swap on; under swap off its layout is b1017: ' + b0.detail);
  const split = Object.assign({}, runRecipe('droopy-v41-b0c-nortr'), { store: { types: ['mxfp4'] } });
  assert.ok(/\|chain\]/.test(chk(split, 'numa_free').detail), 'a recipe without store.split is a chain load even with small banks');
});

t('numa_free regression: b1017 (headroom 10, 2.2 GiB banks) FAILs without SPILL_HEAL and passes with it', () => {
  const r = recipe('regress-droopy-b1017-dry.json');
  const c = chk(r, 'numa_free');
  assert.strictEqual(c.status, 'FAIL', 'b1017 measured node0 MemFree 0.59 + 1.14 GiB spilled to PMem (prefill 8-10): ' + c.detail);
  assert.ok(/fall back to PMem/.test(c.detail));
  const h = chk(Object.assign({}, r, { numa: Object.assign({}, r.numa, { spill_heal: true }) }), 'numa_free');
  assert.strictEqual(h.status, 'PASS', 'b1017heal ran prefill 46-51 at node0 MemFree 0.5: ' + h.detail);
  assert.ok(/summed/.test(h.detail) && /kernel floor 0\.5/.test(h.detail));
  const H = data.checks.numa_free.spill_heal, se = H.sum_extra['droopy|v41|ik|banks<=2.25'];
  const p = FM.predictRecipe(data, Object.assign({}, runRecipe('droopy-v41-b1017heal'), { numa: { headroom_gib: 10, spill_heal: true } }));
  assert.strictEqual(p.node_memfree_gib[0], H.kernel_floor_gib.v, 'node 0 runs at the kernel floor after the heal');
  const le = data.checks.numa_free.load_extra['droopy|v41|ik|banks<=2.25'].v, fl = H.kernel_floor_gib.v;
  assert.ok(Math.abs(p.node_memfree_gib[1] - (10 - (le - (10 - fl)))) < 1e-9, 'node 1 takes node 0 overflow: ' + p.node_memfree_gib);
  assert.ok(se.v < le + 0 && se.lo < se.v, 'the summed extra is measured below node 0 extra (node 1 ran under its plan)');
  // the floor stays per node: a heal cannot save a layout where both DRAM nodes are full (b1017pm OOM)
  const tight = chk(Object.assign({}, r, { numa: { headroom_gib: 4, spill_heal: true } }), 'numa_free');
  assert.strictEqual(tight.status, 'FAIL', tight.detail);
});

t('prefill model: calibrated on b1017off2 (offload ON at 1532 / 2210 tokens), crossover inside the measured band', () => {
  const base = Object.assign({}, runRecipe('droopy-v41-b1017heal'), { banks_gib: 0, engine_build: 'h2dstage-unpublished', offload_batch: 32 });
  const p = FM.predictRecipe(data, base);
  assert.strictEqual(p.status, 'PREDICTED', p.why);
  const pf = p.prefill, cal = data.prefill.calibration;
  assert.ok(pf.offload_on && pf.h2d === 'ring');
  assert.ok(Math.abs(pf.upload_gib - cal.upload_gib) / cal.upload_gib < 0.06, 'CPU-resident experts ~ the 220 GiB the stager moved per ubatch: ' + pf.upload_gib);
  cal.points.forEach(q => {
    const got = pf.at[q.tokens].offload_tps;
    assert.ok(Math.abs(got - q.on_tps) / q.on_tps < 0.12, 'offload at ' + q.tokens + ' tokens: ' + got.toFixed(1) + ' vs MEASURED ' + q.on_tps);
  });
  assert.ok(pf.crossover_tokens > 1532 && pf.crossover_tokens < 2600, 'MEASURED: ON loses at 1532, ties at 2210: crossover ' + pf.crossover_tokens);
  assert.ok(Math.abs(p.prefill_tps - pf.at[2048].offload_tps) < 1e-9, 'the recipe offloads at 2048 tokens');
  const old = FM.predictRecipe(data, Object.assign({}, base, { engine_build: undefined })).prefill;
  assert.ok(old === undefined || old.crossover_tokens === null || old.h2d === 'driver');
  const cpu = FM.predictRecipe(data, Object.assign({}, base, { offload_batch: 100000 }));
  assert.strictEqual(cpu.prefill_tps, data.prefill.cpu_tps['droopy|v41|ik|4096'].v, 'offload_batch above ub = the CPU path (banks off: the MEASURED cell)');
  const on = FM.checkRecipe(data, Object.assign({}, base, { files: [] }), null, { predict: false }).checks.find(c => c.id === 'vram_peak');
  assert.ok(/prefill offload upload buffers/.test(FM.checkRecipe(data, Object.assign({}, base, { files: [] }), null, { predict: false }).notes.join(' ')), 'the upload buffers count in VRAM: ' + on.detail);
});

t('vram regression: BE1-Q3 as launched 06:44Z (8.7 GiB banks) FAILs; the GLM terms match the measured buffers', () => {
  const run = ledger().find(r => r.id === 'droopy-glm-glm-be1q3-0644');
  assert.ok(run && run.measured.outcome === 'oom');
  const c = chk(run.recipe, 'vram_peak');
  assert.strictEqual(c.status, 'FAIL', 'the load ran out of VRAM at the compute buffer: ' + c.detail);
  const p = FM.predictRecipe(data, run.recipe), mib = run.measured.vram_mib;
  const real = (mib.total - mib.free_at_start + mib.model_buffer + mib.hot_experts + mib.kv + mib.compute_requested) / 1024;
  // the load died at the main compute buffer: the run-time parts (prefill pool, MTP context, checkpoints) never allocated
  const RT = data.checks.vram_peak.runtime['glm|ik'], ub = run.recipe.ub;
  const later = RT.pool_gib_per_tok.v * ub + RT.mtp_compute_gib_per_tok.v * ub + RT.draft_fixed_gib.v;
  assert.ok(Math.abs(p.vram_post_gib - later - real) < 0.1, 'load-time part ' + (p.vram_post_gib - later).toFixed(3) + ' vs the measured demand ' + real.toFixed(3) + ' GiB');
  assert.ok(p.vram_post_gib > mib.total / 1024, 'the measured demand is above the card');
  const glm = data.models.glm;
  assert.strictEqual(glm.kv_gib.tag, 'MEASURED');
  assert.ok(glm.kv_gib_ub512.v < glm.kv_gib.v, 'the compute buffer scales with -ub');
  const cpu = FM.predictRecipe(data, Object.assign({}, run.recipe, { mtp_experts: 'cpu' }));
  assert.ok(Math.abs(p.vram_post_gib - cpu.vram_post_gib - glm.draft.experts_gib.ud_q3kxl.v) < 1e-9, 'mtp_experts cpu frees the block 45 experts');
  // the recut BE1-Q3 (6.9 GiB banks, -ub 4096, MTP context at the target -ub) ran as Q3c and failed on the MTP compute buffer
  const be1 = FM.checkRecipe(data, recipe('droopy-glm-be1-q3.json'));
  assert.strictEqual(be1.checks.find(x => x.id === 'vram_peak').status, 'FAIL', 'MEASURED Q3c: MTP compute 2484 MiB cudaMalloc failed');
  const a2 = FM.checkRecipe(data, recipe('droopy-glm-be1-q3-arm2.json'));
  assert.strictEqual(a2.checks.find(x => x.id === 'vram_peak').status, 'FAIL', 'MEASURED a2b: main compute 4550 MiB cudaMalloc failed with offload on at -ub 4096');
  assert.ok(a2.checks.filter(x => x.id !== 'vram_peak' && x.id !== 'quality' && x.id !== 'conditions' && x.id !== 'cpu_reserve' && !(x.id === 'placement' && x.status === 'UNKNOWN')).every(x => x.status === 'PASS'), 'arm 2 fails on VRAM (and the MEASURED UD-Q3_K_XL quality and its T40 without a CPU reserve) only; it predates the placement declaration: ' + a2.checks.map(x => x.id + ' ' + x.status).join('; '));
});

t('quality fit: P3, P4 S25 and P4 S15 points, least squares through the base', () => {
  const g = data.checks.quality.families.glm;
  assert.deepStrictEqual(g.points.map(p => p.run), ['snoopy-glm-p3-1503', 'snoopy-glm-p4-s25-2047', 'snoopy-glm-p4-s15-2047']);
  assert.ok(!g.pending.length, 'S15 measured 2026-10-19, promoted from pending');
  const sxx = g.points.reduce((a, p) => a + p.share * p.share, 0), sxy = g.points.reduce((a, p) => a + p.share * (p.kl - g.base_kl.v), 0);
  assert.ok(Math.abs(g.slope_kl.v - sxy / sxx) < 1e-4);
  assert.ok(!g.pending.some(x => x.id === 'P4 S25'));
  const s25 = ledger().find(r => r.id === 'snoopy-glm-p4-s25-2047');
  assert.ok(s25 && s25.measured.kl === 0.03926 && s25.measured.top1 === 0.9612 && s25.measured.degraded.length);
  assert.ok(!JSON.stringify(s25).includes('/home/'), 'ledger rows name no user home');
});

t('quality fit: V4.1 has its own line on ik (reference floor + the 1017 split + the q3_K/q2_K requant); GLM unchanged', () => {
  const L = data.checks.quality.families.v41, v = FM.v;
  assert.deepStrictEqual(L.engines, ['ik']);
  assert.deepStrictEqual(L.points.map(p => p.share), [0.225, 0.65]);
  const sxx = L.points.reduce((a, p) => a + p.share * p.share, 0);
  const lsq = f => L.points.reduce((a, p) => a + p.share * f(p), 0) / sxx;   // least squares through the base at share 0
  assert.ok(Math.abs(v(L.slope_kl) - lsq(p => p.kl - v(L.base_kl))) < 1e-4, String(v(L.slope_kl)));
  assert.ok(Math.abs(v(L.slope_top1) - lsq(p => L.base_top1 - p.top1)) < 1e-4, String(v(L.slope_top1)));
  assert.ok(v(L.max_share) > 0.35 && v(L.max_share) < 0.45, 'max 2-bit share ' + v(L.max_share));
  assert.ok(/top-1 binds/.test(L.max_share.src), L.max_share.src);
  // the check reads the V4.1 line: the 1017 split and 30 % pass, the 65 % requant fails (as measured: 0.0404 / 0.9535)
  const base = { box: 'droopy', model: 'v41', engine: 'ik', rtr: false, draft: false, threads: 20, files: [], store: { split: false, types: ['q2_K'] } };
  const at = sh => FM.checkRecipe(data, Object.assign({ lowbit_share: sh }, base), null, { predict: false }).checks.find(c => c.id === 'quality');
  assert.strictEqual(at(0.225).status, 'PASS', at(0.225).detail);
  // since the layer map (2026-10-19) the V4.1 check reads the map's two fitted bands (exactly determined by the two points, so
  // the 1017 split reproduces its measured 0.0169), no longer the line's 0.0143 / 0.983
  assert.ok(/PREDICTED \(layer map, 2 calibration points\) KL vs EXL3-3\.05 0\.0169 .*top-1 0\.980/.test(at(0.225).detail), at(0.225).detail);
  assert.strictEqual(at(0.3).status, 'PASS', at(0.3).detail);
  assert.strictEqual(at(0.65).status, 'FAIL', at(0.65).detail);
  assert.ok(!/borrowed from GLM/.test(at(0.65).detail), at(0.65).detail);
});

t('s-glm-ctl anchor pairs 17.88 with the ctl-0929 stats of the same run', () => {
  const a = data.anchors.find(x => x.id === 's-glm-ctl');
  assert.strictEqual(a.meas, 17.88);
  assert.ok(/ctl-0929\.json c1 median 17\.88/.test(a.src));
  assert.ok(ledger().find(r => r.id === 'snoopy-glm-ctl-0929').tags.includes('anchor:s-glm-ctl'));
  assert.ok(!ledger().find(r => r.id === 'snoopy-glm-ctl-0922').tags.includes('anchor:s-glm-ctl'));
});

t('accuracy: median, pairs and percent error', () => {
  assert.strictEqual(ACC.median([3, 1, 2]), 2);
  assert.strictEqual(ACC.median([4, 1, 2, 3]), 2.5);
  assert.strictEqual(ACC.median([]), null);
  const run = { recipe: { draft: true }, measured: { outcome: 'ok', decode_tps: { n4: 8, n0: 5 }, node_memfree_min_gib: 0.5 } };
  const p = ACC.pairs(run, { decode_tps: 10, decode_tps_nodraft: 4, node_memfree_min_gib: 2, prefill_tps: null });
  const d = p.find(x => x.metric === 'decode_tps'), nd = p.find(x => x.metric === 'decode_tps_nodraft'), mf = p.find(x => x.metric === 'node_memfree_min_gib');
  assert.deepStrictEqual([d.phase, d.pct], ['n4', 0.25]);
  assert.deepStrictEqual([nd.phase, nd.pct], ['n0', -0.2]);
  assert.deepStrictEqual([mf.err, mf.pct], [1.5, null], 'MemFree near zero: absolute error only');
  assert.strictEqual(ACC.primaryPhase({ recipe: { draft: false }, measured: { decode_tps: { c1: 20 } } }), 'c1');
  const glm = ACC.pairs({ recipe: { draft: true }, measured: { decode_tps: { m3: 10, m0: 8 } } }, { decode_tps: 12, decode_tps_nodraft: 8.8 });
  assert.deepStrictEqual(glm.map(x => [x.metric, x.phase]), [['decode_tps', 'm3'], ['decode_tps_nodraft', 'm0']], 'GLM kits: m0 is the no-draft phase');
  const off = { recipe: { draft: true, conditions: { draft_n: '0 per request (speculative.n_max 0) although the server loads MTP n_max 3' } }, measured: { decode_tps: { V0_0: 8 } } };
  assert.strictEqual(ACC.draftN(off), 0);
  assert.deepStrictEqual(ACC.pairs(off, { decode_tps: 10, decode_tps_nodraft: 8.8 }).map(x => [x.metric, x.pred]), [['decode_tps_nodraft', 8.8]], 'draft_n 0 declared: no-draft decode, no draft credit');
  assert.strictEqual(ACC.draftN({ recipe: { conditions: { draft_n: 3 } } }), 3);
  const dr = (id, d, dist) => ({ id, box: 'b', disturbed: dist || [], pred: { decode_tps: 12, decode_tps_nodraft: 10 }, run: { recipe: { draft: true }, measured: { decode_tps: d } } });
  const R = ACC.draftRatios([dr('a', { m3: 12, m0: 10 }), dr('b', { m3: 13, m0: 10 }), dr('c', { m3: 9, m0: 10 }, ['m0']), dr('d', { m3: 9 })], r => r.box);
  assert.deepStrictEqual([R[0].n, R[0].median, R[0].pred], [2, 1.25, 1.2], 'a disturbed phase or a missing no-draft phase leaves the run out');
  assert.strictEqual(ACC.draftN({ recipe: {} }), null);
  assert.strictEqual(ACC.primaryPhase({ recipe: { draft: true }, measured: { primary_phase: 'prompt1', decode_tps: { prompt1: 3 } } }), 'prompt1');
});

t('accuracy: summary per group scores only runs that went fine', () => {
  const row = (key, pct, scored) => ({ key, scored, pairs: [{ metric: 'decode_tps', pct }] });
  const s = ACC.summarize([row('a', 0.1, true), row('a', -0.3, true), row('a', 0.2, true), row('a', 5, false), row('b', -0.05, true)], r => r.key, 'decode_tps');
  assert.deepStrictEqual(s.map(x => x.key), ['a', 'b']);
  assert.strictEqual(s[0].n, 3);
  assert.ok(Math.abs(s[0].median_abs_pct - 20) < 1e-9);
  assert.ok(Math.abs(s[0].bias_pct - 10) < 1e-9);
  assert.ok(Math.abs(s[1].bias_pct + 5) < 1e-9);
});

t('accuracy: check verdicts and per-check counts', () => {
  const ck = (...st) => st.map(([id, status]) => ({ id, status }));
  const rows = [
    { id: 'tp', bad: ['oom'], checks: ck(['vram_peak', 'FAIL'], ['storage', 'PASS']), bound_source: [] },
    { id: 'miss', bad: ['crash'], checks: ck(['vram_peak', 'PASS'], ['quality', 'UNKNOWN']), bound_source: [] },
    { id: 'fa', bad: [], checks: ck(['vram_peak', 'PASS'], ['storage', 'FAIL']), bound_source: [] },
    { id: 'ok', bad: [], checks: ck(['vram_peak', 'PASS']), bound_source: [] }];
  const V = ACC.verdicts(rows);
  assert.deepStrictEqual([V.true_positive, V.miss, V.false_alarm, V.clean_pass].map(x => x.map(e => e.id)), [['tp'], ['miss'], ['fa'], ['ok']]);
  assert.deepStrictEqual(V.miss[0].unknown, ['quality']);
  const P = Object.fromEntries(ACC.perCheck(rows).map(x => [x.id, x]));
  assert.deepStrictEqual([P.vram_peak.fail_bad, P.vram_peak.pass_bad, P.storage.fail_fine], [['tp'], ['miss'], ['fa']]);
  assert.ok(ACC.wentBad({ measured: { outcome: 'ok', degraded: ['kswapd'] } }).length === 1);
  assert.ok(ACC.wentBad({ measured: { outcome: 'ok' } }).length === 0);
});

t('accuracy trend: buckets by day or ISO week, slope, and the shrinking verdict per group and for the gate', () => {
  assert.strictEqual(ACC.bucketOf('2026-10-19T04:20Z', 'day'), '2026-10-19');
  assert.strictEqual(ACC.bucketOf('2026-10-19T04:20Z', 'week'), 'wk 2026-10-19', 'a Monday starts its own week');
  assert.strictEqual(ACC.bucketOf('2026-10-18T23:00Z', 'week'), 'wk 2026-10-12', 'a Sunday belongs to the week before');
  assert.ok(Math.abs(ACC.slope([3, 2, 1]) + 1) < 1e-12 && ACC.slope([5]) === 0);
  assert.strictEqual(ACC.gapVerdict([{ y: 30 }, { y: 10 }]).verdict, 'shrinking');
  assert.strictEqual(ACC.gapVerdict([{ y: 10 }, { y: 30 }, { y: 5 }]).verdict, 'shrinking', 'last below first and a falling fit');
  assert.strictEqual(ACC.gapVerdict([{ y: 10 }, { y: 20 }]).verdict, 'NOT shrinking');
  assert.ok(/too few/.test(ACC.gapVerdict([{ y: 1 }]).verdict));
  const ck = st => [{ id: 'vram_peak', status: st }];
  const row = (when, key, pct, bad, st) => ({ id: when + key, when, key, pred_basis: 'launch', scored: !bad, bad: bad ? ['oom'] : [], checks: ck(st), bound_source: [], pairs: [{ metric: 'decode_tps', pct }] });
  const T = ACC.trend([row('2026-10-17T01Z', 'a', 0.4, false, 'PASS'), row('2026-10-17T02Z', 'b', 0.2, true, 'PASS'), row('2026-10-18T01Z', 'a', 0.1, false, 'PASS'),
    row('2026-10-18T02Z', 'b', -0.3, true, 'FAIL')], r => r.key, 'day', ['decode_tps']);
  assert.deepStrictEqual(T.buckets.map(b => [b.bucket, b.runs, b.gate.tp, b.gate.miss]), [['2026-10-17', 2, 0, 1], ['2026-10-18', 2, 1, 0]]);
  assert.strictEqual(T.overall[0].verdict, 'shrinking', 'scored runs only: 40 % -> 10 %');
  assert.deepStrictEqual(T.groups.map(g => [g.key, g.verdict]), [['a', 'shrinking']], 'b never scored: it went bad');
  assert.deepStrictEqual([T.gate.points.map(p => p.y), T.gate.verdict], [[100, 0], 'shrinking']);
});

t('cpu_reserve: every CPU to decode FAILs on the box reserve; T39 + reserve 1 declared PASSes; undeclared is UNKNOWN; unknown engines stay UNKNOWN', () => {
  const r = runRecipe('droopy-glm-glm-be1q4t-0330'), chk = x => FM.cpuReserveCheck(data, x);
  assert.strictEqual(data.boxes.droopy.cpu.reserve.v, 1);
  assert.strictEqual(chk(r).status, 'PASS', 'q4t: T39 + cpu_reserve 1 (a string with a leading count)');
  const t40 = chk(Object.assign({}, r, { threads: 40, conditions: Object.assign({}, r.conditions, { threads: 40, cpu_reserve: 0 }) }));
  assert.strictEqual(t40.status, 'FAIL'); assert.ok(/q4t/.test(t40.detail) && /reserve 1/.test(t40.detail));
  assert.strictEqual(chk(Object.assign({}, r, { threads: 40, conditions: {} })).status, 'FAIL', 'recipe threads alone decide a FAIL');
  assert.strictEqual(chk(Object.assign({}, r, { threads: 20, conditions: {} })).status, 'UNKNOWN', 'T20 with nothing declared');
  assert.strictEqual(chk(Object.assign({}, r, { threads: 30, conditions: { threads: 30, cpu_reserve: 0 } })).status, 'FAIL', 'a declared reserve under the box setting');
  assert.strictEqual(chk(Object.assign({}, r, { conditions: { threads: { a: 40, c: 39 }, cpu_reserve: { a: 0, c: 1 } } })).status, 'FAIL', 'per-cell values: the worst cell (q4i)');
  const au = chk(Object.assign({}, r, { conditions: { threads: 'auto', cpu_reserve: 1 } }));
  assert.ok(au.status === 'UNKNOWN' && /threads/.test(au.detail) && /recipe threads 39/.test(au.detail), 'threads not declared as a number: ' + au.detail);
  const sn = chk(Object.assign({}, r, { box: 'snoopy', engine: 'strata', threads: 32, conditions: {} }));
  assert.strictEqual(sn.status, 'UNKNOWN', 'strata cpu_decode UNKNOWN: never a FAIL on a guess'); assert.ok(/cpu_decode/.test(sn.detail));
  const od = chk(Object.assign({}, r, { box: 'odie', engine: 'sycl', threads: 44, conditions: {} }));
  assert.ok(od.status === 'FAIL' && /default_reserve/.test(od.detail), 'a box without its own reserve takes the named default');
  assert.ok(FM.checkRecipe(data, r, null, { predict: false }).checks.some(c => c.id === 'cpu_reserve'), 'the gate runs it');
});

t('labels: interference marks a run bad and lists its phases; suspicions and no-effect labels do not; cause-matched verdicts', () => {
  const run = { measured: { outcome: 'ok' } };
  const lab = { causes: [{ class: 'interference', phases: ['V0_4'] }, { class: 'interference', phases: ['m1'], confirmed: false }, { class: 'interference', phases: ['m2'], effect: 'none visible' }] };
  assert.deepStrictEqual(ACC.disturbed(lab), ['V0_4']);
  assert.deepStrictEqual(ACC.wentBad(run, lab), ['interference in V0_4 (runs/labels.json)']);
  assert.deepStrictEqual(ACC.wentBad(run, { causes: [lab.causes[1]] }), [], 'a suspicion (confirmed false) is not a cause');
  assert.deepStrictEqual(ACC.causesOf(lab).map(c => c.phases[0]), ['V0_4'], 'a suspicion and a contact with no effect are not causes');
  const L = JSON.parse(fs.readFileSync(path.join(ROOT, 'runs', 'labels.json'), 'utf8')), ids = new Set(ledger().map(r => r.id));
  Object.keys(L).filter(k => k !== '_doc').forEach(k => {
    assert.ok(ids.has(k), 'label for a ledger row: ' + k);
    L[k].causes.forEach(c => { assert.ok(data.causes[c.class], k + ' class ' + c.class); assert.ok(c.src && c.evidence, k + ' names its evidence'); });
  });
  const rows = [{ id: 'm', bad: ['x'], causes: ['interference'], checks: [{ id: 'cpu_reserve', status: 'FAIL' }], bound_source: [] },
    { id: 'o', bad: ['oom'], causes: [], checks: [{ id: 'cpu_reserve', status: 'FAIL' }], bound_source: [] }];
  const V = ACC.verdicts(rows, data.causes);
  assert.deepStrictEqual(V.true_positive.map(e => [e.id, e.matched]), [['m', ['cpu_reserve']], ['o', []]], 'a FAIL for another cause is a TP, not a cause-matched one');
});

t('system: swap on FAILs only under a dram-fill headroom (harm measured under node pressure only); numa_balancing / demotion stay strict', () => {
  const Y = data.checks.system.boxes.droopy;
  assert.deepStrictEqual(Y.dram_fill_only, ['swap']);
  assert.ok(/d1g/.test(Y.dram_fill_only_src) && /b1b-1/.test(Y.dram_fill_only_src), 'the relaxation names its evidence');
  const base = runRecipe('droopy-v41-e2');
  assert.strictEqual(base.system.swap, 'on');
  assert.ok(base.numa === undefined || base.numa.headroom_gib === undefined, 'e2 has no dram-fill headroom');
  const e2 = chk(base, 'system');
  assert.ok(e2.status === 'UNKNOWN' && /swap without dram-fill: harm unmeasured/.test(e2.detail), e2.detail);
  assert.strictEqual(chk(Object.assign({}, base, { numa: { headroom_gib: 10 } }), 'system').status, 'FAIL', 'swap on + dram-fill (b1b / b1c)');
  assert.strictEqual(chk(Object.assign({}, base, { system: Object.assign({}, base.system, { numa_balancing: 2 }) }), 'system').status, 'FAIL', 'numa_balancing 2 FAILs without dram-fill');
  assert.strictEqual(chk(Object.assign({}, base, { system: { swap: 'off', numa_balancing: 0, demotion: false } }), 'system').status, 'PASS');
  // the 7 swap-on runs without dram-fill that ran fine are no longer false alarms
  ['droopy-v41-d1g-q2d-p0', 'droopy-v41-d1g-place12-p0', 'droopy-glm-d2-glm-ctl', 'droopy-v41-e2'].forEach(id => assert.notStrictEqual(chk(runRecipe(id), 'system').status, 'FAIL', id));
  ['droopy-v41-b1b-1', 'droopy-v41-b1c'].forEach(id => assert.strictEqual(chk(runRecipe(id), 'system').status, 'FAIL', id));
});

t('cause taxonomy: each labelled cause, its deciding checks, launch vs today verdicts; undecided classes listed', () => {
  ['vram_load', 'vram_prefill', 'host_pinned_dram', 'dram_node_pressure', 'tool_bug', 'vram_other_process', 'placement_short', 'long_prompt_vram'].forEach(k =>
    assert.ok(data.causes[k] && Array.isArray(data.causes[k].decided_by) && data.causes[k].src, k));
  Object.keys(data.causes).forEach(k => data.causes[k].decided_by.forEach(id => assert.ok(id === 'mem_cap' || data.checks[id], k + ' decided by a real check: ' + id)));
  const L = JSON.parse(fs.readFileSync(path.join(ROOT, 'runs', 'labels.json'), 'utf8'));
  // every crash the brief names carries a class
  ['droopy-glm-glm-be1q3b-0822', 'droopy-glm-glm-be1q3c-0917', 'droopy-glm-glm-be1q3d-1123', 'droopy-glm-glm-be1q3a2b-0823', 'droopy-glm-glm-be1q3a2e-1300', 'odie-glm-q3-b1n', 'odie-glm-q3-b1n2']
    .forEach(id => assert.ok(L[id] && L[id].causes.length && L[id].causes[0].src, id));
  const rows = [
    { id: 'a', bad: ['crash'], causes: ['vram_load'], check_basis: 'launch', checks: [{ id: 'vram_peak', status: 'PASS' }], today: [{ id: 'vram_peak', status: 'FAIL' }] },
    { id: 'b', bad: ['crash'], causes: ['tool_bug'], check_basis: 'post-hoc', checks: [], today: [] },
    { id: 'c', bad: [], causes: ['vram_load'], check_basis: 'launch', checks: [], today: [] }];
  const T = ACC.taxonomy(rows, data.causes);
  assert.deepStrictEqual(T.items.map(x => [x.id, x.cls, x.caught_then, x.caught_now]), [['a', 'vram_load', false, true], ['b', 'tool_bug', false, false]], 'a run that ran fine is not in the taxonomy');
  assert.deepStrictEqual(T.items[0].then, ['vram_peak PASS']);
  assert.deepStrictEqual(T.undecided, ['tool_bug']);
  assert.deepStrictEqual(T.classes.find(c => c.cls === 'vram_load'), { cls: 'vram_load', n: 1, decided_by: ['vram_peak'], caught_then: 0, caught_now: 1 });
});

t('ledger: every row validates against the planner and carries sources', () => {
  const runs = ledger();
  assert.ok(runs.length > 0);
  const ids = new Set();
  runs.forEach(r => {
    assert.ok(!ids.has(r.id), 'duplicate ' + r.id); ids.add(r.id);
    assert.ok(r.sources.length && r.measured.src, r.id);
    const res = FM.checkRecipe(data, r.recipe, null, { predict: false });
    assert.ok(!res.bad_input, r.id + ' recipe is bad input');
  });
});

t('quality: per-layer types give the 2-bit share of the layout the prediction places (one function), never 0', () => {
  // pfeifferj GSQ-RCO 3.0bit as harvest maps its GGUF header: q3_K layers 0-13, q2_K the rest (master showed quality PASS
  // 'nothing at 2 bits' beside a prediction with ~65 % of routed reads at 2 bits)
  const r = { box: 'droopy', model: 'v41', engine: 'ik', store: { split: false, types: ['q2_K'], layer_types: [{ layers: [0, 13], type: 'q3_K' }] },
    rtr: false, draft: false, threads: 20, heat: { plan: 'imatrix' }, files: [] };
  const res = FM.checkRecipe(data, r, null);
  const q = res.checks.find(c => c.id === 'quality');
  assert.strictEqual(res.prediction.status, 'PREDICTED');
  assert.strictEqual(res.facts.lowbit_share, res.prediction.lowbit_share, 'check and prediction read one layout');
  assert.ok(res.facts.lowbit_share > 0.6 && res.facts.lowbit_share < 0.7, 'share ' + res.facts.lowbit_share);
  assert.notStrictEqual(q.status, 'PASS', q.detail);
  // V4.1 on ik has its own measured line since 10-18 (1017 split 0.225 -> KL 0.01689 / top-1 0.9746): ~65 % 2-bit is over the bar
  const sh = res.facts.lowbit_share;
  assert.strictEqual(q.status, 'FAIL', q.detail);
  // since the layer map the check reads the map's weighted share (layers 0-13 q3_K drop out of it), not the plain share line;
  // this layout is the measured q3q2 requant, so the map gives its measured KL / top-1 (0.0404 / 0.954 -> 0.951 PREDICTED)
  const lm = FM.layerMap(data, 'v41'), w = FM.layerWeights(data, 'v41'), dm = FM.recipeDamage(data, r);
  const eff = dm.d.reduce((a, x, l) => a + x * w[l], 0) / dm.d.length, mq = FM.mapQuality(data, 'v41', eff, 0);
  assert.strictEqual(lm.status, 'calibrated');
  assert.ok(eff < sh, 'the q3_K layers 0-13 weigh more but carry no 2-bit reads: weighted ' + eff + ' < plain ' + sh);
  assert.ok(q.detail.includes('PREDICTED (layer map, 2 calibration points) KL vs EXL3-3.05 ' + mq.kl.toFixed(4)), q.detail);
  assert.ok(q.detail.includes('top-1 ' + mq.top1.toFixed(3)), q.detail);
  assert.ok(Math.abs(mq.kl - 0.0404) < 0.002, 'the q3q2 requant point: ' + mq.kl);
  assert.ok(res.notes.some(x => /^2-bit share 6\d\.\d % of routed reads from the layout types: layers 0-13 q3_K/.test(x)));
  // same layout without the stated heat: the share moves with the score heat, still equal to the prediction's
  const r2 = Object.assign({}, r, { heat: undefined });
  assert.strictEqual(FM.checkRecipe(data, r2, null).facts.lowbit_share, FM.predictRecipe(data, r2).lowbit_share);
  // ledger rows (no prediction) read the same share
  assert.strictEqual(FM.checkRecipe(data, r, null, { predict: false }).facts.lowbit_share, res.facts.lowbit_share);
});

t('gpu_split: per-card VRAM over two B70s judges the fullest card; a length mismatch FAILs inputs', () => {
  const r = recipe('odie-glm-q3-a1.json');
  const c = FM.checkRecipe(data, r);
  const vp = c.checks.find(x => x.id === 'vram_peak');
  assert.strictEqual(vp.status, 'PASS', vp.detail);
  assert.ok(c.notes.some(x => /per card \(gpu_split/.test(x)), 'the note lists each card');
  const single = FM.checkRecipe(data, Object.assign({}, r, { gpu_split: undefined }));
  assert.strictEqual(single.checks.find(x => x.id === 'vram_peak').status, 'FAIL', 'all banks on one card do not fit');
  const bad = FM.checkRecipe(data, Object.assign({}, r, { gpu_split: { banks_gib: [20], fixed_share: [1] } }));
  assert.strictEqual(bad.checks.find(x => x.id === 'inputs').status, 'FAIL');
  const sum = FM.checkRecipe(data, Object.assign({}, r, { banks_gib: 10 }));
  assert.ok(/banks sum/.test(sum.checks.find(x => x.id === 'inputs').detail));
  const p = FM.predictRecipe(data, r);
  assert.strictEqual(p.vram_per_card.length, 2);
  assert.ok(Math.abs(p.vram_post_gib - Math.max.apply(null, p.vram_per_card.map(x => x.post_gib))) < 1e-9);
});

t('gpu_layers placement: whole MoE layers on the GPU read k/E of each such layer, slower than hottest-expert banks', () => {
  const r = recipe('odie-glm-q3-a1.json');
  const p = FM.predictRecipe(data, r);
  assert.strictEqual(p.status, 'PREDICTED', p.why);
  const g = p.tiers.find(x => x.id === 'gpu0');
  assert.ok(Math.abs(g.read_share - 14 / 42) < 1e-6, '14 of 42 layers on the GPU = 1/3 of routed reads: ' + g.read_share);
  const hot = FM.predictRecipe(data, Object.assign({}, r, { placement: undefined }));
  assert.ok(hot.decode_tps > p.decode_tps, 'the hottest experts in the same banks decode faster');
  const over = FM.predictRecipe(data, Object.assign({}, r, { placement: { mode: 'gpu_layers', layers: Array.from({ length: 30 }, (_, i) => i) } }));
  assert.strictEqual(over.status, 'UNMODELLED');
  assert.ok(/exceed the GPU banks/.test(over.why));
  assert.strictEqual(FM.predictRecipe(data, Object.assign({}, r, { placement: { mode: 'gpu_layers', layers: [42] } })).status, 'UNMODELLED');
});

t('quality: one 2-bit type = every read at 2 bits; no 2-bit type = 0 without a layout; a 2-bit type with no type rule = UNKNOWN', () => {
  const base = { box: 'droopy', model: 'v41', engine: 'ik', rtr: false, draft: false, threads: 20, files: [] };
  const all2 = FM.checkRecipe(data, Object.assign({ store: { split: false, types: ['q2_K'] } }, base), null, { predict: false });
  assert.ok(Math.abs(all2.facts.lowbit_share - 1) < 1e-6, String(all2.facts.lowbit_share));
  assert.notStrictEqual(all2.checks.find(c => c.id === 'quality').status, 'PASS');
  const hi = FM.checkRecipe(data, Object.assign({ store: { split: false, types: ['q4_K'], layer_types: [{ layers: [0, 3], type: 'q3_K' }] } }, base), null, { predict: false });
  assert.strictEqual(hi.facts.lowbit_share, 0);
  assert.ok(/nothing at 2 bits/.test(hi.checks.find(c => c.id === 'quality').detail));
  const two = FM.checkRecipe(data, Object.assign({ store: { split: false, types: ['q2_K', 'q3_K'] } }, base), null, { predict: false });
  const q = two.checks.find(c => c.id === 'quality');
  assert.strictEqual(two.facts.lowbit_share, null);
  assert.strictEqual(q.status, 'UNKNOWN', q.detail);
  assert.ok(/2-bit share unknown: a store type is 2-bit but the store gives no per-expert type rule/.test(q.detail), q.detail);
  // a stated share still wins (the recipe's own figure, e.g. the GLM P4 mixes)
  const st = FM.checkRecipe(data, Object.assign({ store: { split: false, types: ['q2_K'] }, lowbit_share: 0.1 }, base), null, { predict: false });
  assert.strictEqual(st.facts.lowbit_share, 0.1);
});

t('checkRecipe: heat sets resolve, and a cited hot-expert map digest must match the data', () => {
  const r = recipe('droopy-b1b-hot2p2-t20.json');
  const m = data.models[r.model], plan = (r.heat && r.heat.plan) || m.heat.plan_default;
  const cite = FM.heatCite(m, plan);
  assert.ok(cite.length >= 1 && /^sha256:/.test(cite[0].digest), 'the plan set is backed by a map: ' + JSON.stringify(cite));
  const inputs = x => x.checks.find(c => c.id === 'inputs');
  const ok = FM.checkRecipe(data, Object.assign({}, r, { heat: Object.assign({}, r.heat, { plan, maps: { plan: cite } }) }), null, { predict: false });
  assert.ok(!/heat/.test(inputs(ok).detail), inputs(ok).detail);
  assert.ok(ok.notes.some(n => /cited by the recipe, matches/.test(n)), ok.notes.join(' | '));
  const stale = cite.map(c => Object.assign({}, c, { digest: 'sha256:' + '0'.repeat(64) }));
  const bad = FM.checkRecipe(data, Object.assign({}, r, { heat: Object.assign({}, r.heat, { plan, maps: { plan: stale } }) }), null, { predict: false });
  assert.strictEqual(inputs(bad).status, 'FAIL');
  assert.ok(/plan heat .* the recipe cites .* the data has/.test(inputs(bad).detail), inputs(bad).detail);
  const none = FM.checkRecipe(data, Object.assign({}, r, { heat: { plan: 'no-such-set' } }), null, { predict: false });
  assert.strictEqual(inputs(none).status, 'FAIL');
  assert.ok(/heat set no-such-set is not in/.test(inputs(none).detail));
  const p = FM.predictRecipe(data, r);
  assert.deepStrictEqual(p.heat_maps.plan, cite, 'the prediction names the map it planned on');
});

t('mainline recipe knobs move the prediction: threads, -ub, -sm tensor, op offload, moe cache, pinned RAM fit', () => {
  const r = recipe('odie-glm-q3-b1n.json');
  const base = FM.predictRecipe(data, r);
  assert.strictEqual(base.status, 'PREDICTED', base.why);
  // threads below the physical cores slow the CPU expert share
  const t16 = FM.predictRecipe(data, Object.assign({}, r, { threads: 16 }));
  assert.ok(t16.decode_tps < base.decode_tps && t16.ms_per_token.cpu > base.ms_per_token.cpu, 'threads 16 slower');
  // a larger ubatch: bigger compute buffer, faster prefill on long prompts, same short-prompt pace
  const u8 = FM.predictRecipe(data, Object.assign({}, r, { ub: 8192 }));
  assert.ok(u8.vram_post_gib > base.vram_post_gib, 'ub 8192 grows VRAM');
  assert.ok(u8.prefill_tps_at[8192] > 1.5 * base.prefill_tps_at[8192], 'ub 8192 prefill');
  assert.ok(Math.abs(u8.prefill_tps_at[2048] - base.prefill_tps_at[2048]) < 1e-6, 'a 2k prompt is one ubatch either way');
  // -sm tensor halves the dense read and adds reductions
  const ten = FM.predictRecipe(data, Object.assign({}, r, { split_mode: 'tensor' }));
  assert.ok(ten.ms_per_token.gpu < base.ms_per_token.gpu && ten.ms_per_token.host > base.ms_per_token.host);
  // op offload off: CPU prefill rate; --no-host: slower uploads
  assert.strictEqual(FM.predictRecipe(data, Object.assign({}, r, { op_offload: false })).prefill_tps, data.engines.sycl.recipe_rates.cpu_prefill_tps.v);
  assert.ok(FM.predictRecipe(data, Object.assign({}, r, { no_host: true })).prefill_tps < base.prefill_tps);
  // moe cache: decode MoE of host experts on the GPU (no CPU expert time), a hit share in (0, 1)
  const c = FM.predictRecipe(data, recipe('odie-glm-q3-b1.json'));
  assert.strictEqual(c.status, 'PREDICTED', c.why);
  assert.ok(c.moe_cache.hit_share > 0 && c.moe_cache.hit_share < 1 && c.ms_per_token.cpu < 1, JSON.stringify(c.moe_cache));
  // pinned experts beyond the DRAM cap: UNMODELLED + a FAIL check
  const one = FM.checkRecipe(data, recipe('odie-glm-q3-b3.json'));
  assert.strictEqual(one.checks.find(x => x.id === 'ram_fit').status, 'FAIL');
  assert.strictEqual(one.prediction.status, 'UNMODELLED');
});

t('mem_cap: the pinned host peak (buffers + driver extra) stays under DRAM total - household floor - pinned margin; b1n4 FAILs', () => {
  const ck = (r, cap) => FM.checkRecipe(data, Object.assign({}, r, { mem_cap_gib: cap })).checks.find(x => x.id === 'mem_cap').status;
  const d = data.boxes.odie.dram, top = d.total_gib.v - d.host_floor_gib.v - d.pinned_margin_gib.v;
  assert.ok(d.pinned_margin_gib.v >= 16 && top < 80, String(top));
  const p = FM.predictRecipe(data, recipe('odie-glm-q3-b1n4.json'));
  assert.ok(p.pinned_driver_extra_gib >= 11.9 && p.host_peak_gib > top, String(p.host_peak_gib));
  for (const f of ['odie-glm-q3-b1n4.json', 'odie-glm-q3-b1n.json', 'odie-glm-q3-b1n2.json']) assert.strictEqual(ck(recipe(f), 8), 'FAIL', f);
  assert.strictEqual(ck(recipe('odie-glm-q3-b1n4.json'), undefined), 'FAIL');
});

t('prefill: one model for every engine; mainline uploads the routed experts per ubatch, ik every CPU-resident expert', () => {
  const sy = FM.predictRecipe(data, recipe('odie-glm-q3-b1n4.json'));
  assert.strictEqual(sy.prefill.upload_touched, true);
  assert.ok(sy.prefill.offload_on && sy.prefill.h2d === 'engine' && sy.prefill.upload_gib <= sy.prefill.upload_gib + 1e-9);
  assert.ok(Math.abs(sy.prefill_tps - sy.prefill_tps_at[2048]) < 1e-9, 'prefill_tps is at the recipe prefill_tokens (2048 default)');
  const nh = FM.predictRecipe(data, Object.assign({}, recipe('odie-glm-q3-b1n4.json'), { no_host: true }));
  assert.strictEqual(nh.prefill.h2d, 'engine --no-host');
  const ik = FM.predictRecipe(data, Object.assign({}, runRecipe('droopy-v41-b1017heal'), { banks_gib: 0, engine_build: 'h2dstage-unpublished', offload_batch: 32 }));
  assert.strictEqual(ik.prefill.upload_touched, false);
  assert.ok(Math.abs(ik.prefill.upload_gib - ik.tiers.filter(x => !/^gpu/.test(x.id)).reduce((a, x) => a + x.experts_gib, 0)) < 1e-6, 'ik stages every CPU-resident expert');
});

t('vram runtime terms (glm|ik): prefill pool by -ub, MTP compute by mtp_ubatch, offload graph; Q3c/Q3d/a2b/a2e FAIL, Q3e PASS', () => {
  const RT = data.checks.vram_peak.runtime['glm|ik'];
  ['pool_gib_per_tok', 'mtp_compute_gib_per_tok', 'draft_fixed_gib', 'offload_compute_gib_per_tok'].forEach(k => assert.strictEqual(RT[k].tag, 'MEASURED', k));
  assert.ok(Math.abs(RT.mtp_compute_gib_per_tok.v * 128 * 1024 - 77.62) < 0.1, 'MTP compute 77.62 MiB at micro-batch 128');
  assert.ok(Math.abs(RT.mtp_compute_gib_per_tok.v * 4096 * 1024 - 2484) < 1, 'MTP compute 2484 MiB at 4096');
  assert.ok(RT.pool_gib_per_tok.v * 4096 > 574 / 1024, 'the pool at -ub 4096 exceeds the 574 MiB Q3d had free');
  const vp = r => chk(r, 'vram_peak');
  // the three measured crashes FAIL (each crashed on a VRAM allocation), the recut that runs PASSes
  [['droopy-glm-glm-be1q3c-0917', 'MTP compute buffer'], ['droopy-glm-glm-be1q3d-1123', 'prefill pool cuMemCreate'], ['droopy-glm-glm-be1q3a2b-0823', 'offload main compute 4550 MiB']].forEach(([id, why]) => {
    const run = ledger().find(r => r.id === id);
    assert.ok(run && run.measured.outcome === 'crash', id);
    assert.strictEqual(vp(run.recipe).status, 'FAIL', id + ' (' + why + '): ' + vp(run.recipe).detail);
  });
  const e = recipe('droopy-glm-be1-q3e.json');
  assert.strictEqual(e.ub, 2048);
  assert.strictEqual(e.mtp_ubatch, 128);
  assert.strictEqual(vp(e).status, 'PASS', 'Q3e runs (MEASURED 22,200 of 23,040 MiB): ' + vp(e).detail);
  const p = FM.predictRecipe(data, e);
  assert.ok(Math.abs(p.vram_post_gib - 22200 / 1024) < 0.15, 'Q3e post-warm-up ' + p.vram_post_gib.toFixed(3) + ' vs MEASURED 21.68 GiB after the first prompts');
  // the knobs move the check the way the runs did
  assert.strictEqual(vp(Object.assign({}, e, { mtp_ubatch: undefined })).status, 'FAIL', 'no LLAMA_MTP_UBATCH: the MTP context compute buffer at -ub 2048');
  assert.strictEqual(vp(Object.assign({}, e, { ub: 4096 })).status, 'FAIL', '-ub 4096 = Q3d');
  const mt = u => FM.predictRecipe(data, Object.assign({}, e, { mtp_ubatch: u })).vram_post_gib;
  assert.ok(Math.abs(mt(512) - mt(128) - RT.mtp_compute_gib_per_tok.v * 384) < 1e-9, 'MTP compute linear in mtp_ubatch');
  assert.ok(Math.abs(mt(8192) - mt(2048)) < 1e-9, 'mtp_ubatch above -ub is capped at -ub');
  const off = FM.checkRecipe(data, Object.assign({}, e, { offload_batch: 32, files: [] }), null, { predict: false });
  assert.ok(/upload buffers \+ compute graph/.test(off.notes.join(' ')), 'the offload graph is in the VRAM sum');
  // a2e = Q3e + GPU prefill offload, banks 5.0 GiB, planned on a through-0 offload graph term: MEASURED main compute 4,102 MiB at
  // -ub 2048 and a CUDA pool OOM at the first prefill. With the two-point term (a2b 4,550 at 4096, a2e 4,102 at 2048) it FAILs
  const a2e = recipe('droopy-glm-be1-q3a2e.json');
  assert.ok(a2e.offload_batch <= a2e.ub && a2e.ub === 2048 && a2e.mtp_ubatch === 128);
  const a2eRun = ledger().find(r => r.id === 'droopy-glm-glm-be1q3a2e-1300');
  assert.ok(a2eRun && a2eRun.measured.outcome === 'crash');
  assert.strictEqual(vp(a2eRun.recipe).status, 'FAIL', 'a2e as launched: ' + vp(a2eRun.recipe).detail);
  const base = u => 2484 / 4096 * u;
  [[2048, 4102.01], [4096, 4550.02]].forEach(([u, mib]) => {
    const extra = RT.offload_compute_fixed_gib.v + RT.offload_compute_gib_per_tok.v * u;
    assert.ok(Math.abs((base(u) / 1024 + extra) * 1024 - mib) < 2, 'offload-on compute buffer at -ub ' + u + ': ' + ((base(u) / 1024 + extra) * 1024).toFixed(1) + ' vs MEASURED ' + mib);
  });
  assert.strictEqual(vp(Object.assign({}, a2e, { banks_gib: 3.4 })).status, 'PASS', 'the recut: banks 3.4 GiB');
  const nd = FM.predictRecipe(data, Object.assign({}, e, { draft: false }));
  assert.ok(p.vram_post_gib - nd.vram_post_gib > RT.draft_fixed_gib.v, 'MTP off drops the draft runtime terms');
});

t('GLM CPU prefill on droopy: the -ub-keyed MEASURED rate (Q3e 73.4 at -ub 2048), no other -ub; a2e offload pays above ~1.4k tokens inside one ubatch', () => {
  const C = data.prefill.cpu_tps, e = recipe('droopy-glm-be1-q3e.json'), a = recipe('droopy-glm-be1-q3a2e.json');
  assert.strictEqual(C['droopy|glm|ik|2048'].tag, 'MEASURED');
  const p = FM.predictRecipe(data, e);
  assert.strictEqual(p.prefill.key, 'droopy|glm|ik|2048');
  assert.ok(Math.abs(p.prefill_tps - 73.4) < 1e-9 && p.prefill.cpu_tag === 'MEASURED', 'Q3e CPU prefill = its measured mean');
  const run = ledger().find(r => r.id === 'droopy-glm-glm-be1q3e-1215');
  assert.ok(Math.abs(p.prefill_tps - run.measured.prefill_tps) / run.measured.prefill_tps < 0.05, 'the ledger row agrees');
  const u4 = FM.predictRecipe(data, Object.assign({}, e, { ub: 4096 })).prefill;
  assert.strictEqual(u4.key, 'droopy|glm|ik|4096');
  assert.ok(u4.cpu_tps === null && /^UNMODELLED: no MEASURED CPU prefill cell droopy\|glm\|ik\|4096/.test(u4.cpu_src), 'no -ub 4096 reading: UNMODELLED, never borrowed: ' + u4.cpu_src);
  const q = FM.predictRecipe(data, a), pf = q.prefill;
  assert.ok(pf.offload_on && pf.h2d === 'ring');
  assert.ok(Math.abs(q.prefill_tps - pf.at[2048].offload_tps) < 1e-9 && q.prefill_tps > 100 && q.prefill_tps < 112, 'a2e at 2048 tokens: ' + q.prefill_tps);
  assert.ok(pf.crossover_tokens > 1300 && pf.crossover_tokens < 1550, 'crossover = 73.4 x upload_s: ' + pf.crossover_tokens);
  assert.ok(Math.abs(pf.at[4096].offload_tps - pf.at[2048].offload_tps) < 1e-6 && pf.at[2210].offload_tps < 73.4, 'a second ubatch pays the upload again: below the CPU rate at 2210 tokens');
});

t('prefill envelope: a CPU rate holds only inside its measured -ub / PMem / banks cell; -rtr x0.32 on a split model; else UNMODELLED in the gate line', () => {
  const C = data.prefill.cpu_tps, V = C['droopy|v41|ik|4096'], G = C['droopy|glm|ik|2048'];
  assert.deepStrictEqual(Object.keys(C).sort(), ['droopy|glm|ik|2048', 'droopy|v41|ik|4096'], 'only MEASURED cells, each with its -ub');
  [V, G].forEach(c => assert.ok(c.envelope.pmem_gib.length === 2 && c.envelope.src, 'each cell names its envelope and source'));
  assert.ok(V.tag === 'MEASURED' && V.banks_on.tag === 'FIT' && /held out/.test(V.banks_on.src));
  const pr = (id, o) => FM.predictRecipe(data, Object.assign({}, runRecipe(id), o || {}));
  // the FIT banks-on rate predicts its held-out row
  const ho = ledger().find(r => r.id === V.banks_on.holdout.id);
  assert.ok(Math.abs(pr(ho.id).prefill_tps - ho.measured.prefill_tps) / ho.measured.prefill_tps < 0.03, 'held out ' + ho.id);
  assert.strictEqual(V.banks_on.holdout.meas, ho.measured.prefill_tps);
  // banks off reads the MEASURED cell; banks outside the fitted range are UNMODELLED
  assert.strictEqual(pr('droopy-v41-b0c-nortr', { banks_gib: 0 }).prefill_tps, V.v);
  const big = pr('droopy-v41-b0c-nortr', { banks_gib: 1.5 });
  assert.ok(big.prefill_tps === null && /banks 1.5 GiB outside/.test(big.prefill_why), big.prefill_why);
  // -rtr on the split V4.1 store: x checks.rtr.prefill_mult (B0 pair: 16.0 vs 50.5); on a one-chain store: UNMODELLED
  const rtr = pr('droopy-v41-b0c-rtr'), run = ledger().find(r => r.id === 'droopy-v41-b0c-rtr');
  assert.ok(Math.abs(rtr.prefill_tps - V.banks_on.v * data.checks.rtr.prefill_mult.v) < 1e-9);
  assert.ok(Math.abs(rtr.prefill_tps - run.measured.prefill_tps) / run.measured.prefill_tps < 0.05, 'b0c-rtr ' + rtr.prefill_tps);
  const one = pr('droopy-glm-glm-be1q3e-1215', { rtr: true });
  assert.ok(one.prefill_tps === null && /one-chain/.test(one.prefill_why), one.prefill_why);
  // PMem outside the envelope (E2 at 112 GiB read 34.5, the -ub 512 rows 23.2 / 38.1): UNMODELLED, and the gate line says so
  const e2 = pr('droopy-v41-e2');
  assert.ok(e2.prefill_tps === null && /PMem experts 112.0 GiB outside the measured 62-70.3/.test(e2.prefill_why), e2.prefill_why);
  assert.ok(/prefill UNMODELLED \(PMem experts 112.0 GiB outside/.test(ACC.fmtPrediction(e2)), ACC.fmtPrediction(e2));
  assert.ok(pr('droopy-v41-d1g-q2d-p0').prefill_tps === null, '-ub 512 has no cell');
  // inside the envelope every banks-on V4.1 / GLM row that did not go bad reads within 10 %
  ['droopy-v41-b1c', 'droopy-glm-glm-be1q3f-1314', 'droopy-glm-glm-be1q3m-1720', 'droopy-glm-glm-be1q4c-1831'].forEach(id => {
    const x = ledger().find(r => r.id === id).measured.prefill_tps, y = pr(id).prefill_tps;
    assert.ok(Math.abs(y - x) / x < 0.1, id + ': ' + y + ' vs ' + x);
  });
});

t('ctx knob (glm|ik): KV per context token MEASURED at 4096 and 34816, no context-sized prefill buffer (a2f); a2f predicts prefill at 16k / 32k', () => {
  const RT = data.checks.vram_peak.runtime['glm|ik'], f = recipe('droopy-glm-be1-q3a2f.json');
  assert.strictEqual(f.ctx, 34816);
  assert.strictEqual(RT.kv_gib_per_ctx_tok.tag, 'MEASURED');
  assert.ok(Math.abs(RT.kv_gib_per_ctx_tok.v * (34816 - 4096) * 1024 - 555) < 0.01, 'main 215.56 -> 740.56 + MTP 4.00 -> 34.00 MiB from 4096 to 34816');
  assert.strictEqual(RT.compute_gib_per_ctx_tok.v, 0);
  // a2f's main compute buffer with offload on and the MTP experts on the host: 3,032.01 MiB (off buffer 1,242 + the offload extra)
  const v = x => x.v, ext = F => (v(F) + v(RT.offload_compute_gib_per_tok) * 2048) * 1024;
  assert.ok(Math.abs(1242 + ext(RT.offload_compute_fixed_gib_mtp_cpu) - 3032.01) < 0.5, 'a2f compute buffer');
  assert.ok(Math.abs(1242 + ext(RT.offload_compute_fixed_gib) - 4102.01) < 0.5, 'a2e compute buffer (MTP experts on the GPU)');
  const p = r => FM.predictRecipe(data, r).vram_post_gib;
  const dg = p(Object.assign({}, f, { mtp_experts: 'gpu' })) - p(f), off = (v(RT.offload_compute_fixed_gib) - v(RT.offload_compute_fixed_gib_mtp_cpu));
  assert.ok(off > 1.0 && dg > off + 3.0, 'mtp_experts gpu adds its experts and the larger offload fixed part: ' + dg);
  const d = p(f) - p(Object.assign({}, f, { ctx: 4096 }));
  assert.ok(Math.abs(d - (RT.kv_gib_per_ctx_tok.v + RT.compute_gib_per_ctx_tok.v) * (34816 - 4096)) < 1e-9, 'ctx adds KV + context buffers beyond 4096');
  assert.ok(Math.abs(p(Object.assign({}, f, { ctx: undefined })) - p(Object.assign({}, f, { ctx: 4096 }))) < 1e-12, 'unset ctx = 4096');
  assert.strictEqual(chk(f, 'vram_peak').status, 'PASS', chk(f, 'vram_peak').detail);
  assert.strictEqual(chk(Object.assign({}, f, { banks_gib: 6.6, ctx: 4096 }), 'vram_peak').status, 'PASS');
  const pf = FM.predictRecipe(data, f).prefill;
  [16384, 32768].forEach(t => assert.ok(pf.at[t].offload_tps > pf.at[t].cpu_tps && pf.at[t].offload_tps > 95, t + ' tokens: offload ' + pf.at[t].offload_tps + ' vs CPU ' + pf.at[t].cpu_tps));
});

t('quality: a whole GLM store measured on the gate decides; UD-Q3_K_XL FAILs (a2f / q3f), UD-Q4_K_XL is not marked by it', () => {
  const Q = data.checks.quality, m = Q.stores['glm|ud_q3kxl'];
  assert.strictEqual(m.kl.tag, 'MEASURED');
  assert.ok(m.kl.v > Q.bar_kl && m.top1 < Q.bar_top1);
  ['droopy-glm-be1-q3a2f.json', 'droopy-glm-be1-q3f.json', 'droopy-glm-be1-q3e.json'].forEach(f => {
    const c = chk(recipe(f), 'quality'); assert.strictEqual(c.status, 'FAIL', f + ': ' + c.detail); assert.ok(/MEASURED on the gate/.test(c.detail));
  });
  const q4 = chk(recipe('droopy-glm-be1.json'), 'quality');
  assert.ok(!/MEASURED on the gate/.test(q4.detail), q4.detail);
});

t('droopy gsq_iq3s CPU kernel MEASURED (K1-FNX): the GSQ-RCO IQ3_S layer mix runs at 0.274 of the xread ceiling, half of iq3_ks', () => {
  const k = data.boxes.droopy.kappa;
  assert.strictEqual(k.gsq_iq3s.tag, 'MEASURED');
  assert.ok(Math.abs(k.gsq_iq3s.v - 0.274) < 1e-9 && k.gsq_iq3s.v < k.iq3_ks.v * 0.6);
});

t('store with a type the table lacks: UNMODELLED with the type named, never a crash (read share, hot/cold, one type)', () => {
  const r = recipe('droopy-glm-be1.json');
  [{ split: true, types: ['ud_q4kxl', 'iq2_s'], read_share_by_type: { ud_q4kxl: 0.8, iq2_s: 0.2 } }, { split: true, types: ['mxfp4', 'iq2_s'], hot_type: 'mxfp4', cold_type: 'iq2_s' },
    { split: false, types: ['iq2_s'] }].forEach(store => {
    const p = FM.predictRecipe(data, Object.assign({}, r, { store }));
    assert.strictEqual(p.status, 'UNMODELLED', JSON.stringify(store));
    assert.ok(/type iq2_s is not in the type table/.test(p.why), p.why);
    const c = FM.checkRecipe(data, Object.assign({}, r, { store }), null, { predict: true });
    assert.ok(c.checks.length > 0);
  });
});

t('a recipe lowbit_share that disagrees with the store layout is named in the assumptions (it moves quality only, not speed)', () => {
  const r = recipe('droopy-b1b-hot2p2-t20.json'), p0 = FM.predictRecipe(data, r);
  const p1 = FM.predictRecipe(data, Object.assign({}, r, { lowbit_share: p0.lowbit_share + 0.15 }));
  assert.strictEqual(p1.decode_tps, p0.decode_tps, 'the speed follows the store, not lowbit_share');
  assert.ok(p1.assumptions.some(a => /not the recipe lowbit_share/.test(a)), p1.assumptions.join(' | '));
  const p2 = FM.predictRecipe(data, Object.assign({}, r, { lowbit_share: p0.lowbit_share }));
  assert.ok(!p2.assumptions.some(a => /not the recipe lowbit_share/.test(a)), 'no note when they agree');
});

t('knobs outside the decode model are named, never silent: -rtr without a repacked row, SMT use unlike the fitted runs, an unmeasured page layout', () => {
  const g = recipe('droopy-glm-be1.json'), p0 = FM.predictRecipe(data, g), has = (p, re) => p.assumptions.some(a => re.test(a));
  const rtr = FM.predictRecipe(data, Object.assign({}, g, { rtr: true }));
  assert.ok(has(rtr, /-rtr: no repacked kernel row for ud_q4kxl/), rtr.assumptions.join(' | '));
  const b = recipe('droopy-b1b-hot2p2-t20.json'), rb = FM.predictRecipe(data, Object.assign({}, b, { rtr: true }));
  assert.ok(has(rb, /no repacked kernel row for q2_K \(/), 'mxfp4 has mxfp4_r8, so only q2_K is named: ' + rb.assumptions.join(' | '));
  assert.ok(!has(p0, /-rtr:/), 'no note with -rtr off');
  const phys = data.boxes.droopy.cpu.phys.v;
  const t = FM.predictRecipe(data, Object.assign({}, g, { threads: phys }));
  assert.ok(has(t, /leave out SMT siblings/), t.assumptions.join(' | '));
  assert.strictEqual(t.decode_tps, p0.decode_tps, 'threads do not move the prediction (that is what the note says)');
  assert.ok(!has(FM.predictRecipe(data, Object.assign({}, g, { threads: phys * 2 - 1 })), /SMT siblings/), 'T39 matches the fitted runs');
  const odd = FM.predictRecipe(data, Object.assign({}, g, { numa: Object.assign({}, g.numa, { layout: 'striped' }) }));
  assert.ok(has(odd, /NUMA page layout "striped" is not in the decode model \(measured layouts on droopy: /), odd.assumptions.join(' | '));
  assert.strictEqual(odd.decode_tps, p0.decode_tps, 'an unmeasured layout does not move the prediction');
  assert.ok(!has(FM.predictRecipe(data, Object.assign({}, g, { numa: Object.assign({}, g.numa, { layout: 'interleave' }) })), /NUMA page layout/));
});

t('NUMA page layout: a measured layout scales the DRAM expert CPU time (rows faster, whole-node slower than interleave) and is a decode term', () => {
  const g = recipe('droopy-glm-be1.json'), lay = l => FM.predictRecipe(data, Object.assign({}, g, { numa: Object.assign({}, g.numa, { layout: l }) }));
  const il = lay('interleave'), rows = lay('rows'), node = lay('node'), L = data.boxes.droopy.numa_layout;
  assert.ok(L.rows.v < 1 && L.node.v > 1 && L.interleave.v === 1, JSON.stringify(L));
  assert.strictEqual(il.decode_tps, FM.predictRecipe(data, g).decode_tps, 'interleave = the fitted reference');
  assert.ok(rows.decode_tps > il.decode_tps && node.decode_tps < il.decode_tps, [rows.decode_tps, il.decode_tps, node.decode_tps].join(' '));
  assert.ok(rows.assumptions.some(a => /NUMA page layout "rows": DRAM expert CPU time x0\.737/.test(a)), rows.assumptions.join(' | '));
  assert.ok(JSON.stringify(rows.evidence).indexOf('NUMA page layout rows') >= 0, 'the layout factor is a listed decode term');
  // only the DRAM tier moves: a recipe with no DRAM experts would not change (here: the GPU + PMem parts stay, so the gain is below 1/0.737)
  assert.ok(rows.decode_tps_nodraft / il.decode_tps_nodraft < 1 / L.rows.v, 'host + GPU time do not scale');
});

t('draft speed-up: 2+ in-load ledger twins (served width vs no draft, same load) set the box x model fit; fewer keep the anchors', () => {
  const D = JSON.parse(JSON.stringify(data)), xs = D.draft_twins['droopy|glm'].map(x => x.mult);
  const c = FM.applyCal(D).boxes.droopy.draft.glm;
  assert.ok(Math.abs(c.v - xs.reduce((a, x) => a + x, 0) / xs.length) < 1e-9 && /in-load twins in the run ledger/.test(c.src), c.src);
  D.draft_twins['droopy|glm'] = D.draft_twins['droopy|glm'].slice(0, 1);
  const a = FM.applyCal(D).boxes.droopy.draft.glm;
  assert.ok(!/in-load twins/.test(a.src) && /measured arm/.test(a.src), a.src);
  D.draft_twins['droopy|glm'] = [{ id: 'x', what: 'MTP n_max 3', mult: 1.5 }, { id: 'y', what: 'MTP n_max 3', mult: 1.3 }];
  FM.applyCal(D);
  const p = FM.predictRecipe(D, recipe('droopy-glm-be1.json'));
  assert.ok(Math.abs(p.decode_tps / p.decode_tps_nodraft - 1.4) < 1e-9, 'the fit is the decode multiplier: ' + p.decode_tps / p.decode_tps_nodraft);
});

t('split GGUF host time is per model: GLM has its own row (UNKNOWN, not added, an unknown decode term), V4.1 keeps the fitted box term', () => {
  const D = JSON.parse(JSON.stringify(data)); FM.applyCal(D);
  const g = Object.assign({}, recipe('droopy-glm-be1.json'), { store: { split: true, types: ['ud_q4kxl', 'q2_K'], hot_type: 'ud_q4kxl', cold_type: 'q2_K' }, lowbit_share: 0.2 });
  const dec = p => p.evidence.find(x => x.metric === 'decode'), sp = p => dec(p).terms.find(t => /^split GGUF host ms\/token/.test(t.term));
  const u = FM.predictRecipe(D, g);
  assert.ok(sp(u) && sp(u).term === 'split GGUF host ms/token (glm)' && sp(u).evidence === 'unknown' && sp(u).value === null, JSON.stringify(sp(u)));
  assert.strictEqual(dec(u).grade, 'unknown');
  D.models.glm.split_host_ms = { v: 10, tag: 'ESTIMATE', src: 'test' };
  const e = FM.predictRecipe(D, g);
  assert.ok(Math.abs(1000 / e.decode_tps_nodraft - 1000 / u.decode_tps_nodraft - 10) < 1e-6, 'a value is added per token: ' + [e.decode_tps_nodraft, u.decode_tps_nodraft]);
  delete D.models.glm.split_host_ms;
  const b = FM.predictRecipe(D, g);
  assert.ok(sp(b).tag === 'FIT' && /d-v41-b0/.test(sp(b).src), 'without its own row a model borrows the box fit, named: ' + JSON.stringify(sp(b)));
  const v = FM.predictRecipe(D, recipe('droopy-b1b-hot2p2-t20.json'));
  assert.ok(sp(v) && sp(v).term === 'split GGUF host ms/token (v41)' && sp(v).tag === 'FIT', JSON.stringify(sp(v)));
});

t('rtr check: -rtr FAILs on both the split pair (B0) and the one-chain GLM rows layout (r2 vs r1)', () => {
  const g = recipe('droopy-glm-be1.json'), ck = x => FM.checkRecipe(data, Object.assign({}, g, x)).checks.find(c => c.id === 'rtr');
  assert.strictEqual(ck({ rtr: false }).status, 'PASS');
  const c = ck({ rtr: true });
  assert.strictEqual(c.status, 'FAIL', c.detail);
  assert.ok(/one-chain GGUF with -rtr: decode x0\.668/.test(c.detail), c.detail);
});

t('placement check: a DRAM-first recipe declares pageable host buffers and a kit that stops off target; pinned or verify off FAILs', () => {
  const g = recipe('droopy-glm-be1.json'), ck = (numa, x) => FM.checkRecipe(data, Object.assign({}, g, x || {}, { numa })).checks.find(c => c.id === 'placement');
  const base = Object.assign({}, g.numa);
  assert.ok(base.headroom_gib !== undefined, 'the fixture is a dram-fill recipe');
  assert.strictEqual(ck(base).status, 'UNKNOWN', ck(base).detail);
  const ok = ck(Object.assign({}, base, { pinned_host: false, verify: 'fail', layout: 'rows' }));
  assert.strictEqual(ok.status, 'PASS', ok.detail);
  assert.ok(!/layout interleave/.test(ok.detail));
  assert.ok(/layout interleave: an mbind interleave/.test(ck(Object.assign({}, base, { pinned_host: false, verify: 'fail' })).detail));
  assert.strictEqual(ck(Object.assign({}, base, { pinned_host: true, verify: 'fail' })).status, 'FAIL');
  assert.strictEqual(ck(Object.assign({}, base, { pinned_host: false, verify: 'off' })).status, 'FAIL');
  assert.strictEqual(ck(Object.assign({}, base, { pinned_host: false, verify: 'warn' })).status, 'UNKNOWN');
  assert.strictEqual(ck({}).status, 'PASS', 'no per-node placement, nothing to verify');
  assert.ok(data.causes.placement_short.decided_by.indexOf('placement') >= 0 && data.accuracy.replay_checks.v.indexOf('placement') >= 0);
});

t('heat scored on the map it was planned on is named as in sample (the cold-tier read shares are a floor)', () => {
  const g = recipe('droopy-glm-be1.json'), note = p => p.assumptions.find(a => /^heat:/.test(a));
  assert.ok(/the same map: in sample/.test(note(FM.predictRecipe(data, g))), note(FM.predictRecipe(data, g)));
  const out = FM.predictRecipe(data, Object.assign({}, g, { heat: { plan: g.heat.plan, score: 'coding' } }));
  assert.ok(!/in sample/.test(note(out)), note(out));
});

t('GGML_EXPERT_CHUNKING: an ik recipe on droopy that does not declare it OFF gets the planner note; OFF does not; the prediction never moves', () => {
  const g = recipe('droopy-glm-be1.json'), has = (p, re) => p.assumptions.some(a => re.test(a));
  const und = FM.predictRecipe(data, Object.assign({}, g, { expert_chunking: undefined }));
  const on = FM.predictRecipe(data, Object.assign({}, g, { expert_chunking: 'on' }));
  const off = FM.predictRecipe(data, Object.assign({}, g, { expert_chunking: 'off' }));
  assert.ok(has(und, /expert_chunking not declared: ik packages on droopy must build GGML_EXPERT_CHUNKING=OFF \(ON measured decode x0\.797/), und.assumptions.join(' | '));
  assert.ok(has(on, /^GGML_EXPERT_CHUNKING=ON: /), on.assumptions.join(' | '));
  assert.ok(!has(off, /EXPERT_CHUNKING/), off.assumptions.join(' | '));
  assert.strictEqual(on.decode_tps, off.decode_tps, 'the note does not scale the prediction');
  assert.strictEqual(data.boxes.droopy.ik_expert_chunking_on.evidence, 'seen');
});

t('conditions: a recipe without its serving conditions is UNKNOWN and names each missing one; declared ones PASS; draft_n only with a draft', () => {
  const r = recipe('droopy-glm-be1.json'), chk = x => FM.checkRecipe(data, x, null, { predict: false }).checks.find(c => c.id === 'conditions');
  const bare = chk(Object.assign({}, r, { ctx: undefined, conditions: undefined, draft: true }));
  assert.strictEqual(bare.status, 'UNKNOWN');
  Object.keys(data.conditions.fields).forEach(k => assert.ok(bare.detail.indexOf(k + ' (') >= 0, k + ': ' + bare.detail));
  const all = { ctx: 34816, kv_type: 'f16', draft_n: 3, sampling: { temperature: 0 }, reasoning_effort: 'none (raw completion)', workload: 'bench.py coding, depth 600' };
  assert.strictEqual(chk(Object.assign({}, r, { draft: true, conditions: all })).status, 'PASS');
  const nod = Object.assign({}, all); delete nod.draft_n;
  assert.strictEqual(chk(Object.assign({}, r, { draft: false, conditions: nod })).status, 'PASS', 'no draft: draft_n is not owed');
  const top = Object.assign({}, nod); delete top.ctx;
  assert.strictEqual(chk(Object.assign({}, r, { draft: false, ctx: 32768, conditions: top })).status, 'PASS', 'a top-level ctx counts');
});

// ---------- evidence: seen / inferred / unknown on every prediction term, and the external validation rows ----------
t('rollup: counts each class, grades by the weakest term, says plainly when it rests on INFERRED terms', () => {
  const S = { v: 2, tag: 'MEASURED', src: 'run x', evidence: 'seen' }, I = { v: 3, tag: 'FIT', src: 'fit to 4 points', evidence: 'inferred' };
  const r = FM.rollup('decode', 21, 'tok/s', [FM.term(data, 'a', S), FM.term(data, 'b', S), FM.term(data, 'c', I)]);
  assert.deepStrictEqual([r.seen, r.inferred, r.unknown, r.grade], [2, 1, 0, 'inferred']);
  const line = FM.fmtRollup(r);
  assert.ok(/^decode 21.00 tok\/s: 2 seen, 1 inferred, 0 unknown terms; rests on INFERRED terms: c 3.00 \(FIT: fit to 4 points\)/.test(line), line);
  assert.ok(!/every term SEEN/.test(line));
  const all = FM.rollup('vram', 20, 'GiB', [FM.term(data, 'a', S)]);
  assert.strictEqual(all.grade, 'seen'); assert.ok(/every term SEEN$/.test(FM.fmtRollup(all)));
  const u = FM.rollup('prefill', null, 'tok/s', [FM.term(data, 'rate', { v: null, tag: 'UNKNOWN', src: 'UNKNOWN: no cell' })]);
  assert.strictEqual(u.grade, 'unknown');
  const ul = FM.fmtRollup(u);
  assert.ok(/^prefill UNMODELLED: 0 seen, 0 inferred, 1 unknown terms; UNKNOWN: rate \(UNKNOWN: no cell\)$/.test(ul), 'a src that opens with its tag prints it once: ' + ul);
  assert.strictEqual(FM.evidenceOf(data, { v: 1, tag: 'GUESS', src: 'x' }), 'unknown', 'a term with no known tag (or none) counts as unknown, never as seen');
  assert.strictEqual(FM.evidenceOf(data, 0.6), 'unknown');
});

t('prediction evidence: every --check prediction term carries a tag, class and source; the counts add up; UNMODELLED prefill is an unknown term', () => {
  const files = fs.readdirSync(path.join(ROOT, 'recipes')).filter(f => f.endsWith('.json'));
  let k = 0;
  files.forEach(f => [].concat(recipe(f)).forEach(r => {
    const p = FM.predictRecipe(data, r);
    if (p.status !== 'PREDICTED') return;
    k++;
    assert.ok(p.evidence.length >= 3 && p.evidence[0].metric === 'decode', f);
    p.evidence.forEach(e => {
      assert.strictEqual(e.seen + e.inferred + e.unknown, e.terms.length, f + ' ' + e.metric);
      e.terms.forEach(x => { assert.ok(['seen', 'inferred', 'unknown'].indexOf(x.evidence) >= 0 && typeof x.tag === 'string', f + ' ' + x.term); if (x.evidence !== 'unknown') assert.ok(String(x.src || '').length > 0, f + ' ' + x.term + ' has no source'); });
      if (e.metric === 'prefill' && e.value === null) assert.ok(e.unknown >= 1, f + ': UNMODELLED prefill names its unknown term');
    });
    if (p.prefill_tps === null) assert.ok(p.evidence.some(e => e.metric === 'prefill' && e.grade === 'unknown'), f);
  }));
  assert.ok(k >= 15, 'recipes predicted: ' + k);
  // a droopy decode rests on our FIT host and the vendor GPU nameplate: inferred, and the line says so
  const d = FM.predictRecipe(data, recipe('droopy-glm-be1-q3e.json')).evidence[0];
  assert.strictEqual(d.grade, 'inferred');
  assert.ok(d.terms.some(x => /GPU nameplate/.test(x.term) && x.evidence === 'inferred'));
  assert.ok(/rests on INFERRED terms: .*host overhead/.test(FM.fmtRollup(d)));
  assert.ok(d.external && d.external.group.startsWith('nvidia_consumer') && d.external.n > 0, 'a droopy decode carries the nvidia class gap');
});

t('Optane upgrade: rate = the fitted 2-stick rate x bw_scale (knob optaneBwScale overrides), capacity from the upgrade, the evidence term names it', () => {
  const r = recipe('droopy-v41-pmem6-buy.json'), now = FM.predictRecipe(data, Object.assign({}, r, { knobs: {} }));
  const lo = FM.predictRecipe(data, Object.assign({}, r, { knobs: { addOptane: true } })), hi = FM.predictRecipe(data, r);
  const up = data.boxes.droopy.upgrades.pmem, fit = FM.v(data.boxes.droopy.pmem.bw);
  const rate = p => p.evidence[0].terms.find(x => x.term === 'PMem read rate GB/s');
  assert.strictEqual(rate(now).value, fit, 'no upgrade: the fitted box rate');
  assert.ok(Math.abs(rate(lo).value - fit * FM.v(up.bw_scale)) < 1e-9 && rate(lo).evidence === 'inferred' && /bw_scale|30 GB/.test(rate(lo).src), 'default: fitted x bw_scale');
  assert.ok(Math.abs(rate(hi).value - fit * 3) < 1e-9 && /optaneBwScale/.test(rate(hi).src), 'knob: fitted x 3');
  assert.ok(now.decode_tps < lo.decode_tps && lo.decode_tps < hi.decode_tps, 'faster PMem decodes faster: ' + [now, lo, hi].map(p => p.decode_tps.toFixed(2)));
  assert.ok(hi.ms_per_token.host === now.ms_per_token.host && hi.ms_per_token.gpu === now.ms_per_token.gpu, 'only the CPU/PMem term moves');
  const ctx = FM.context(data, 'droopy', 'v41', 'ik', Object.assign({}, data.knobs, { addOptane: true }));
  assert.strictEqual(ctx.tiers.find(t => t.id === 'pmem').total, FM.v(up.gib) * FM.GiB, 'capacity = the upgrade gib');
});

t('external rows: our physics on a community row, never fitted on it', () => {
  const base = { url: 'https://example.org/x', hw_class: 'nvidia_consumer', engine: 'llama.cpp', backend: 'CUDA', quant: 'Q4_0', measured: { decode_tps: 100 }, settings: {} };
  const L2 = { total_b: 6.74, active_b: 6.74, layers: 32, params_src: 'card' };
  const gpu = { gpu: 'X', gpu_bw_gbs: 1000, spec_src: 'spec' };
  // dense all-GPU, no backend stated: bytes = active x file bpw / 8 over nameplate x gpu_eta, plus the borrowed GPU host
  const a = FM.predictExternal(data, Object.assign({ id: 'a', model_params: L2, file_gib: 3.56, placement: 'all_gpu', hardware: gpu }, base, { backend: undefined }));
  const host = FM.v(data.models[data.external.host_borrow.v.gpu[0]].host_engine[data.external.host_borrow.v.gpu[1]]);
  const want = 1 / (3.56 * FM.GiB / (1000e9 * FM.v(data.assume.gpu_eta)) + host / 1000);
  assert.strictEqual(a.status, 'PREDICTED'); assert.strictEqual(a.placement, 'gpu');
  assert.ok(Math.abs(a.decode_tps - want) < 1e-9 && Math.abs(a.err - (want / 100 - 1)) < 1e-12, a.decode_tps + ' vs ' + want);
  assert.strictEqual(a.evidence[0].grade, 'inferred');
  assert.ok(/rests on INFERRED terms: .*GPU nameplate bandwidth GB\/s \(X\) 1000 \(MEASURED-BY-THEM: spec\)/.test(FM.fmtRollup(a.evidence[0])), FM.fmtRollup(a.evidence[0]));
  assert.strictEqual(a.evidence[1].grade, 'unknown', 'no general prefill model');
  // MoE: experts of n of L layers on the CPU read DRAM spec x our spec->read ratio x kernel x f; all experts on the CPU is slower
  const moe = { total_b: 116.83, active_b: 5.13, experts: 128, experts_active: 4, layers: 36, params_src: 'card' };
  const hyb = Object.assign({ gpu: 'Y', gpu_bw_gbs: 936, dram_bw_gbs: 108.8, spec_src: 'spec' });
  const h = n => FM.predictExternal(data, Object.assign({ id: 'h' + n, model_params: moe, quant: 'MXFP4', placement: 'n_cpu_moe:' + n, hardware: hyb }, base, { quant: 'MXFP4' }));
  const all = FM.predictExternal(data, Object.assign({ id: 'e', model_params: moe, placement: 'experts_cpu', hardware: hyb }, base, { quant: 'MXFP4' }));
  assert.strictEqual(h(28).placement, 'hybrid');
  assert.ok(h(18).decode_tps > h(28).decode_tps && h(28).decode_tps > all.decode_tps, 'fewer CPU layers decode faster');
  assert.ok(Math.abs(h(36).decode_tps - all.decode_tps) < 1e-9, 'n_cpu_moe:L = experts_cpu');
  ['DRAM spec -> read ratio', 'CPU kernel efficiency mxfp4', 'CPU box factor f'].forEach(x => assert.ok(all.terms.some(y => y.term === x && y.evidence === 'inferred'), x));
  const D = FM.externalDerived(data);
  assert.ok(D.dram_eta.v > 0.5 && D.dram_eta.v < 1 && D.cpu_f.v > 0, JSON.stringify(D));
  const c = FM.predictExternal(data, Object.assign({ id: 'c', model_params: moe, placement: 'cpu_only', hardware: { cpu: 'Z', dram_bw_gbs: 108.8, spec_src: 'spec' } }, base, { quant: 'MXFP4' }));
  assert.strictEqual(c.placement, 'cpu'); assert.ok(c.decode_tps < all.decode_tps, 'the dense part on the CPU too');
  // what it cannot map is UNMODELLED with the reason, never a number
  const un = (o, re) => { const x = FM.predictExternal(data, Object.assign({ id: 'u', model_params: moe, placement: 'experts_cpu', hardware: hyb }, base, { quant: 'MXFP4' }, o)); assert.ok(x.status === 'UNMODELLED' && x.decode_tps === null && re.test(x.why), x.why); assert.strictEqual(x.evidence[0].grade, 'unknown'); };
  un({ hardware: { gpu: 'Y', gpu_bw_gbs: 936 } }, /DRAM spec bandwidth not stated/);
  un({ settings: { draft_n: 3 } }, /draft/);
  un({ placement: 'tensor_split' }, /cannot be mapped/);
  un({ quant: 'EXL9', file_gib: null }, /quant label/);
  un({ model_params: { total_b: 10 } }, /model params/);
  assert.strictEqual(FM.quantType(data, 'UD-Q4_K_XL'), 'ud_q4kxl'); assert.strictEqual(FM.quantType(data, 'Q4_K_M'), 'q4_K'); assert.strictEqual(FM.quantType(data, 'F16'), null);
});

t('bench terms on external rows: the backend line (reach, host), the per-token MoE term, the NUMA gain; each UNMODELLED when missing', () => {
  const G = data.bench.gpu.cuda, base = { url: 'https://example.org/x', hw_class: 'nvidia_consumer', engine: 'llama.cpp', backend: 'CUDA', quant: 'Q4_0', measured: { decode_tps: 100 }, settings: {} };
  assert.ok(G && G.eta.evidence === 'seen' && G.host_ms.evidence === 'seen' && G.moe_layer_ms.evidence === 'seen', 'CUDA line measured on our card');
  assert.ok(G.host_ms.runs.length >= 2 && G.host_ms.runs.every(r => /^droopy-cuda-/.test(r)), 'cites its bench rows');
  const gpu = { gpu: 'X', gpu_bw_gbs: 1000, spec_src: 'spec' }, L2 = { total_b: 6.74, active_b: 6.74, layers: 32, params_src: 'card' };
  // dense: nameplate x the CUDA reach, plus the CUDA host (seen), not the borrowed SYCL host
  const a = FM.predictExternal(data, Object.assign({ id: 'a', model_params: L2, file_gib: 3.56, placement: 'all_gpu', hardware: gpu }, base));
  const want = 1 / (3.56 * FM.GiB / (1000e9 * FM.v(G.eta)) + FM.v(G.host_ms) / 1000);
  assert.ok(Math.abs(a.decode_tps - want) < 1e-9, a.decode_tps + ' vs ' + want);
  assert.ok(a.terms.some(x => /host overhead ms\/token \(cuda line\)/.test(x.term) && x.evidence === 'seen'));
  // MoE all on the GPU: + layers x the per-layer MoE term; the same bytes as a dense row otherwise
  const moe = { total_b: 20.91, active_b: 3.61, experts: 32, experts_active: 4, layers: 24, params_src: 'card' };
  const m = FM.predictExternal(data, Object.assign({ id: 'm', model_params: moe, file_gib: 11.27, placement: 'all_gpu', hardware: gpu }, base, { quant: 'MXFP4' }));
  const bpw = 11.27 * FM.GiB * 8 / 20.91e9, gpuS = 3.61e9 * bpw / 8 / (1000e9 * FM.v(G.eta));
  assert.ok(Math.abs(1 / m.decode_tps - (gpuS + 24 * FM.v(G.moe_layer_ms) / 1000 + FM.v(G.host_ms) / 1000)) < 1e-12);
  assert.ok(m.terms.some(x => /MoE overhead ms per GPU MoE layer \(cuda, x 24 layers\)/.test(x.term)));
  // hybrid: only the MoE layers left on the GPU pay the GPU MoE term
  const hyb = { gpu: 'Y', gpu_bw_gbs: 936, dram_bw_gbs: 108.8, spec_src: 'spec' };
  const h = FM.predictExternal(data, Object.assign({ id: 'h', model_params: moe, placement: 'n_cpu_moe:20', hardware: hyb }, base, { quant: 'MXFP4' }));
  assert.ok(h.terms.some(x => /x 4 layers/.test(x.term)), 'n_cpu_moe:20 of 24 leaves 4 MoE layers on the GPU');
  const un = (o, re) => { const x = FM.predictExternal(data, Object.assign({ id: 'u', model_params: moe, file_gib: 11.27, placement: 'all_gpu', hardware: gpu }, base, { quant: 'MXFP4' }, o)); assert.ok(x.status === 'UNMODELLED' && re.test(x.why), x.why); };
  un({ backend: 'Glide' }, /no per-token MoE term for backend Glide/);
  un({ model_params: Object.assign({}, moe, { layers: null }) }, /no layer count/);
  // a line is a driver stack: the Intel Vulkan line serves an Arc row, never an AMD Vulkan row (that falls back, and a MoE there is UNMODELLED)
  const VK = data.bench.gpu.vulkan;
  assert.deepStrictEqual(VK.vendors, ['intel']);
  const vk = cls => FM.predictExternal(data, Object.assign({ id: 'v' + cls, model_params: L2, file_gib: 3.56, placement: 'all_gpu', hardware: gpu }, base, { backend: 'Vulkan', hw_class: cls }));
  assert.ok(vk('intel_arc').terms.some(x => /vulkan line/.test(x.term)) && !vk('amd').terms.some(x => /vulkan line/.test(x.term)), 'AMD Vulkan takes the fallback');
  un({ backend: 'Vulkan', hw_class: 'amd' }, /backend Vulkan on amd cards \(measured on intel only\)/);
  assert.ok(FM.predictExternal(data, Object.assign({ id: 'r', model_params: L2, file_gib: 3.56, placement: 'all_gpu', hardware: gpu }, base, { backend: 'ROCm', hw_class: 'amd' })).terms.some(x => /rocm line/.test(x.term) && x.evidence === 'inferred'), 'a borrow names the vendor it stands in for');
  // NUMA: a 2-socket row reads one node's share x the measured gain over one node, never the summed spec
  const cpu = { cpu: '2x Z', dram_bw_gbs: 921.6, spec_src: 'spec' }, q3 = { total_b: 30.53, active_b: 3.35, experts: 128, experts_active: 8, layers: 48, params_src: 'card' };
  const c = s => FM.predictExternal(data, Object.assign({ id: 'c' + s, model_params: q3, placement: 'cpu_only', hardware: Object.assign({ sockets: s }, cpu) }, base, { quant: 'Q4_K_M', backend: 'CPU' }));
  const g = FM.numaGain(data, 'interleave', 2, 2), host = FM.v(c(1).terms.find(x => /host overhead/.test(x.term)).value) / 1000;
  assert.ok(g && g.g.evidence === 'seen' && /lassie interleave m2 c2 s2 t\d+/.test(g.key), JSON.stringify(g && g.key));
  assert.ok(Math.abs((1 / c(2).decode_tps - host) / (1 / c(1).decode_tps - host) - 2 / FM.v(g.g)) < 1e-9, 'cpu time x 2 / gain');
  assert.ok(c(2).terms.some(x => /NUMA placement gain/.test(x.term) && x.evidence === 'seen'));
  assert.ok(!c(1).terms.some(x => /NUMA/.test(x.term)), 'one socket, one node: no gain term');
  const nm = FM.predictExternal(data, Object.assign({ id: 'n', model_params: q3, placement: 'cpu_only', hardware: Object.assign({ sockets: 4 }, cpu) }, base, { quant: 'Q4_K_M', backend: 'CPU' }));
  assert.ok(nm.status === 'UNMODELLED' && /no measured NUMA arm for interleave over 4 nodes on 4 sockets/.test(nm.why), nm.why);
  // the arm with the most threads wins a placement
  const fake = { bench: { numa: { b: { 'interleave m2 c2 s2 t3': { v: 0.5 }, 'interleave m2 c2 s2 t15': { v: 0.8 }, 'interleave m2 c2 s1 t15': { v: 1.1 } } } } };
  assert.strictEqual(FM.numaGain(fake, 'interleave', 2, 2).g.v, 0.8);
  assert.strictEqual(FM.numaGain(fake, 'membind', 2, 2), null);
});

t('externalSummary: per class median |error| and bias over the predicted rows; every class is covered; data terms untouched', () => {
  const before = JSON.stringify([data.assume.gpu_eta, data.types.q4_0]);
  const S = FM.externalSummary(data), rows = data.external.rows;
  assert.ok(rows.length >= 20 && rows.length <= 60, 'rows ' + rows.length);
  assert.deepStrictEqual(S.by_class.map(g => g.key).sort(), Object.keys(data.external.classes).sort());
  S.by_class.forEach(g => {
    const errs = S.rows.filter(r => r.hw_class === g.key && r.err !== null).map(r => r.err).sort((x, y) => x - y);
    assert.strictEqual(g.n, errs.length);
    const abs = errs.map(Math.abs).sort((x, y) => x - y), m = abs.length % 2 ? abs[(abs.length - 1) / 2] : (abs[abs.length / 2 - 1] + abs[abs.length / 2]) / 2;
    assert.ok(Math.abs(g.median_abs_pct - 100 * m) < 1e-9, g.key);
  });
  assert.ok(S.unmodelled.every(u => u.why), 'every unmodelled row says why');
  assert.strictEqual(FM.externalSummary(data), S, 'cached');
  assert.strictEqual(JSON.stringify([data.assume.gpu_eta, data.types.q4_0]), before, 'validation rows never set a term');
  const L = FM.fmtExternal(S);
  assert.strictEqual(L.rows.length, rows.length); assert.ok(L.by_class.every(x => /median \|error\| \d+ %, bias [+-]\d+ %/.test(x)));
});

console.log(`${n} tests passed`);
