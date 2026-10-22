#!/usr/bin/env node
// Writes NOTES.md: the narrative, the flags, and the source of every tagged number in data.json (+ the fitted values).
'use strict';
const fs = require('fs'), path = require('path');
const ROOT = path.join(__dirname, '..');
const FM = require('./model.js');
const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'data.json'), 'utf8'));
FM.applyCal(data);
const v = FM.v;
const rows = [];
function walk(x, p) {
  if (x === null || typeof x !== 'object') return;
  if (Array.isArray(x)) { if (x.length > 60) return; x.forEach((y, i) => walk(y, p + '[' + i + ']')); return; }
  if ('v' in x && 'tag' in x && ('src' in x)) { rows.push([p, x.v, x.tag, x.src, FM.evidenceOf(data, x)]); return; }
  Object.keys(x).forEach(k => { if (k === 'layers' || k === 'siblings' || k === '_heat' || k === 'quality_pairs') return; walk(x[k], p ? p + '.' + k : k); });
}
walk(Object.assign({}, data, { siblings: undefined }), '');
const fmt = x => typeof x === 'number' ? (Math.abs(x) >= 100 ? x.toFixed(1) : Math.abs(x) >= 1 ? x.toFixed(3) : x.toPrecision(3)) : String(x);
const cell = s => String(s === undefined || s === null ? '' : s).replace(/\|/g, '/').replace(/\n/g, ' ');
const rep = fs.existsSync(path.join(__dirname, 'report-out.txt')) ? fs.readFileSync(path.join(__dirname, 'report-out.txt'), 'utf8') : '(run: node report.js --json src/report-out.json > src/report-out.txt)';
const sib = data.siblings || {};
const recs = FM.checkRecipes(data);
const caught = recs.filter(r => r.wasted && r.checks.some(c => c.status === 'FAIL'));
// evidence roll-ups over every recipe that predicts, and the external validation summary
const EV = data.meta.evidence, XS = FM.externalSummary(data), B = data.bench || { gpu: {}, numa: {} };
// an external row's gap as text, or UNMODELLED
const gap = id => { const r = XS.rows.find(x => x.id === id); return r && r.err !== null ? (r.err >= 0 ? '+' : '') + (100 * r.err).toFixed(0) + ' %' : 'UNMODELLED'; };
const med = xs => { const a = xs.slice().sort((x, y) => x - y), k = a.length; return k ? (k % 2 ? a[(k - 1) / 2] : (a[k / 2 - 1] + a[k / 2]) / 2) : null; };
let nRuns = 0; (function cnt(x) { if (Array.isArray(x)) return x.forEach(cnt); if (x && typeof x === 'object') { if (x.runs && x.evidence === 'seen') nRuns++; Object.values(x).forEach(cnt); } })(data);
const predRecs = fs.readdirSync(path.join(ROOT, 'recipes')).filter(f => f.endsWith('.json')).flatMap(f => [].concat(JSON.parse(fs.readFileSync(path.join(ROOT, 'recipes', f), 'utf8')))).map(r => FM.predictRecipe(data, r)).filter(p => p.status === 'PREDICTED');
const tsum = predRecs.flatMap(p => p.evidence).reduce((a, e) => ({ seen: a.seen + e.seen, inferred: a.inferred + e.inferred, unknown: a.unknown + e.unknown }), { seen: 0, inferred: 0, unknown: 0 });
const share = data.external.rows.filter(x => x.placement === 'all_gpu' && x.model_params.total_b === x.model_params.active_b && x.file_gib).map(x => x.measured.decode_tps * x.file_gib * FM.GiB / 1e9 / x.hardware.gpu_bw_gbs).sort((x, y) => x - y);
const ttot = tsum.seen + tsum.inferred + tsum.unknown, termShare = ['seen', 'inferred', 'unknown'].map(k => tsum[k] + ' ' + k + ' (' + (100 * tsum[k] / ttot).toFixed(0) + ' %)').join(', ') + ' prediction terms';
// dLLM (block diffusion): the fitted terms, the held-out run and leave-one-out
const DL = data.dllm, DT = FM.dllmTerms(data), DV = FM.dllmValidation(data);
const pc = x => x === null || x === undefined ? '-' : (x >= 0 ? '+' : '') + (100 * x).toFixed(1) + ' %';
const dllmMd = !DL ? '' : `
## dLLM: block-diffusion decode (\`runs/runs.jsonl\` rows tagged dllm)

Source: homelab lab notes section 3a (\`llama-dllm-gen\` on the n-engine fork, lab notes feat/dllm-block-kv), recorded by
\`tools/record_dllm.py\` (${DL.rows.length} rows: ${DL.rows.filter(r => r.ok).length} clean, the rest crashed, cut or probe-disturbed; stripped copies in \`runs/dllm-raw/\`).
A dLLM recipe has \`kind: "dllm"\`; \`--check\` prints its decode and ms-per-forward trust lines.

decode tok/s = TPF x 1000 / ms per forward, with
ms per forward = host + non-expert bytes / line + expert bytes x coverage(B) / (line x kernel gap).
- **TPF** (tokens committed per forward) and **B** (tokens submitted per forward) are measured per cell = model | schedule | block |
  workload (median over the cell's rows, SEEN). Rows of one cell that spread past ${100 * v(DL.tpf_spread_max)} % are MEASURED, CONFOUNDED: the kernel
  numerics move near-threshold commits.
- **line** = the box's measured backend line (SYCL: bench.gpu.sycl, rate + host, SEEN; CPU: the box DRAM read).
- **coverage(B)** = 1 - (1 - k/E)^(rho B) distinct experts per MoE layer, rho = ${v(DL.routing)} (${DL.routing.tag}, \`${DL.coverage_file.v}\`): a 32-row
  block forward touches ${(100 * FM.dllmCover({ moe: true, k: 8, E: 256 }, 32, v(DL.routing))).toFixed(0)} % of the experts per layer, uniform routing would touch
  ${(100 * FM.dllmCover({ moe: true, k: 8, E: 256 }, 32)).toFixed(0)} %. Measured with the fixed probe (fork 66972be94 reads the strided \`ffn_moe_topk\` view row by row;
  the old count was 4864 = 19 x 256 on every prompt, so "nearly all 256 experts" was the probe bug). Source: ${DL.routing.src}.
- **kernel gap** = the share of the line the MoE batch kernel reaches, FIT per box | backend | kernel path from the clean cache/fuse
  rows at block ${DL.fit_blocks.v.join('/')} (median per expert shape x quant; groups pool when they agree within ${100 * v(DL.eff_spread_max)} %). A forward
  wider or narrower than the fitted rows by more than x${v(DL.b_slack)} is UNMODELLED.

| kernel path | groups (shape x quant: gap, rows) | pools | B fitted |
|---|---|---|---|
${Object.keys(DT.eff).map(k => { const e = DT.eff[k]; return `| ${k} | ${Object.keys(e.groups).map(g => g + ': ' + e.groups[g].v.toFixed(3) + ' (' + e.groups[g].n + ')').join('; ')} | ${e.pools ? 'yes, ' + e.pooled.v.toFixed(3) : 'no (spread ' + (100 * e.spread).toFixed(0) + ' %)'} | ${e.b_min.toFixed(1)}-${e.b_max.toFixed(1)} |`; }).join('\n')}

The per-pair kernel reaches the same share of the line on mini (2048x512) and flash (4096x1024) experts, so it transfers across
models; the stock kernel does not (small experts pay its per-expert loop). With the measured coverage (a 32-row forward reads ~24 %
of the expert bytes, not 64 % or all) both sit at ${(100 * Math.min(...Object.values(DT.eff).map(e => Math.min(...Object.values(e.groups).map(g => g.v))))).toFixed(0)}-${(100 * Math.max(...Object.values(DT.eff).map(e => Math.max(...Object.values(e.groups).map(g => g.v))))).toFixed(0)} % of the line, which sharpens the
lab notes finding: the SYCL MoE batch kernels, not the decoder, bound diffusion decode on the B70.

**Held-out run** (\`dllm.holdout\`, out of every term):

| run | predicted tok/s | measured | decode error | TPF error | ms/forward error |
|---|---|---|---|---|---|
${DV.holdout.map(x => x.status === 'PREDICTED' ? `| ${x.id} | ${x.pred.toFixed(2)} | ${x.meas} | ${pc(x.err)} | ${pc(x.tpf_err)} (${x.tpf.toFixed(2)}) | ${pc(x.ms_err)} (${x.ms.toFixed(0)} vs ${x.ms_meas} ms) |` : `| ${x.id} | UNMODELLED | | ${cell(x.why)} | | |`).join('\n')}

**Leave-one-out** over the ${DV.loo.length} fit rows (each predicted with the terms refitted without it): decode median |error|
${DV.loo_summary.median_abs_pct === null ? '-' : DV.loo_summary.median_abs_pct.toFixed(1)} % (bias ${pc(DV.loo_summary.bias_pct / 100)}, n ${DV.loo_summary.n}, rows whose TPF cell holds another row); ms per forward median |error|
${DV.loo_ms_summary.median_abs_pct === null ? '-' : DV.loo_ms_summary.median_abs_pct.toFixed(1)} % (bias ${pc(DV.loo_ms_summary.bias_pct / 100)}, n ${DV.loo_ms_summary.n}). UNMODELLED in LOO: ${DV.unmodelled.map(x => x.id).join(', ') || 'none'} (the stock flash group has
too few rows once one is left out, and the stock groups do not pool).

| run | decode error | TPF error | ms/forward error | TPF from |
|---|---|---|---|---|
${DV.loo.filter(x => x.status === 'PREDICTED').map(x => `| ${x.id} | ${pc(x.err)} | ${pc(x.tpf_err)} | ${pc(x.ms_err)} | ${x.tpf_from} |`).join('\n')}

Not modelled (UNMODELLED, the reason printed by \`--check\`): spec decoding (two models), full-sequence forwards (\`--mode full\`, the
fork's \`llama-diffusion-cli\`, Dream-Coder 7B), experts on the CPU beside a GPU (\`--n-cpu-moe\`), block 64 (outside the fitted B).
The Ling-mini-2.0 AR control is predicted from the bench SYCL line and its per-layer MoE term alone (no dLLM fit):
${(() => { const r = DL.rows.find(x => x.id === 'odie-dllm-n1-ling-q4-ar-full'); if (!r) return 'no row'; const p = FM.predictDllm(data, { kind: 'dllm', box: r.box, model: r.model, backend: r.backend, moe_path: r.path, cards: ['b70_2'], quant: r.quant, schedule: { name: r.schedule, mode: 'ar' }, block: r.block, workload: r.workload, placement: r.placement }); return p.status === 'PREDICTED' ? p.decode_tps.toFixed(1) + ' vs ' + r.decode_tps + ' tok/s (' + pc(p.decode_tps / r.decode_tps - 1) + '): the bench line plus the per-layer MoE term is too fast for a 1.4B-active MoE at batch 1 (per-layer dispatch cost is not in the bench line), ' + p.ms_per_fwd.toFixed(1) + ' vs ' + r.ms_per_fwd + ' ms per token' : p.why; })()}.
`;
const m10 = `## Measured 10-17 (what changed in the solver)

Inputs: lab notes (08:00Z on) + lab notes (excl0, k2e-1450, p3-1503,
k2-cpubench, p512-cpubench). Measured values replace priors where they conflict; the previous build is kept at
lab notes

**Levers folded in** (each also a calibration anchor or a check input):

| box | lever | before | after | tag | source |
|---|---|---|---|---|---|
${(data.levers || []).map(l => `| ${l.box} ${l.model} | ${cell(l.lever)} | ${cell(l.before)} | ${cell(l.after)} | ${l.tag} | ${cell(l.src)} |`).join('\n')}

**Model changes.**
- droopy CPU kernel shares are the K1 per-type measurements (frac of the read roofline), not the odie AVX2 smoke; the box
  factor f refit to ${v(data.cal.boxes.droopy.f).toFixed(3)} (about 1, so K1 explains droopy's CPU rate on its own).
- Split GGUF (hot MXFP4 / cold q2_K store, ik) carries a fitted host overhead of ${data.cal.boxes.droopy.split_ms ? v(data.cal.boxes.droopy.split_ms).toFixed(1) : '?'} ms/token
  (FIT from B0; confounded with the T40 thread count, other boxes borrow it as ESTIMATE). Without it B0 was over-predicted by ~26 %.
- V4.1 KV + compute at -ub 4096 is the B0 measurement (post-warm-up 20.79 GiB minus banks, dense part, draft). New knob
  -ub auto: 4096 when the GPU holds it, else 512 (lassie's 1080 Ti now runs at 512 instead of "cannot run").
- DSpark draft: acceptance ~90 %, measured twins (E2/E2d, B0/B0d) set the droopy V4.1 draft multiplier.
- glm-lite: the CPU lane is one DRAM-bound rate (k2e: 1.99 GB/token in 34.07 ms = ~58 GB/s, roofline 59.5),
  host overhead = 1000/22.22 - 34.07 ms; zero-copy folded into the lane. excl0 / cs32k / k2e are anchors; P3 is a check row.
- glm-lite GPU cache: the live VRAM cache hit ${data.cal.engines && data.cal.engines.glmlite ? '' : '?'}37.7 % of routed reads with 1600 slots (k2e), where the planner's static hot set of the
  same size would hit ${data.cal.engines && data.cal.engines.glmlite ? (100 * 126.57 / 336 / v(data.cal.engines.glmlite.hit_real)).toFixed(1) : '?'} % on the in-sample code heat. The planner now credits the GPU with only
  ${data.cal.engines && data.cal.engines.glmlite ? (100 * v(data.cal.engines.glmlite.hit_real)).toFixed(0) : '?'} % of its planned heat (FIT) and reads the rest from DRAM. Before this, the solver's own all-3.05 layout projected
  26.6 tok/s for the k2e config that measured 22.22; now 22.2.
- mxfp4_r8 (load-time repack) is only offered with the new -rtr knob, and the rtr check prices it.

**New hard checks** (evaluated per recipe and on every solver point; FAIL / UNKNOWN / PASS, never silent):
- **storage**: on-file tensors (V4.1 engram tables, ${'~'}104 GB) must sit on NVMe/SSD, never a rotational disk. Per-box disk lists
  now carry the kind; the solver never places an on-file lookup on an HDD.
- **vram_peak**: post-warm-up VRAM + transient margin must fit (0.005 GiB tolerance: readings carry 2 decimals). Margin for
  the ik split GGUF is bounded (${data.checks.vram_peak.margin.ik_split.lo}, ${data.checks.vram_peak.margin.ik_split.hi}] GiB: B1 OOMed at 22.54 + <0.50, and B1b (3.0 GiB banks) ran its timed prompts at
  21.56 of 23.04 with no OOM (droopy notebook readings are nvidia-smi MiB / 1000; the planner converts them to GiB x 1000/1024,
  so the card is 22.50 GiB). The B0 #1 graph-growth candidate 1.15 GiB sits inside (knob "VRAM peak margin = likely").
  The planner reserves the margin on the first GPU only (multi-GPU layer split is not modeled per card).
- **numa_free**: predicted MemFree per NUMA node = free at plan time - planned experts on the node - load extra (interleaved
  non-expert share + CUDA host compute/staging buffers + page cache of on-file tensors) must stay >= ${v(data.checks.numa_free.min_free_gib)} GiB. The load extra
  lands on node 0 (other nodes: ${v(data.checks.numa_free.other_nodes)} GiB, every placed run read at or below plan there) and is keyed by load style:
  droopy V4.1 ik split = ${v(data.checks.numa_free.load_extra['droopy|v41|ik'])} GiB MEASURED (B1b: dram-fill headroom 10 -> node0 MemFree 0.5, kswapd, prefill 10-12 vs ~50),
  split with banks <= 2.25 GiB = ${v(data.checks.numa_free.load_extra['droopy|v41|ik|banks<=2.25'])} (B0 nortr ran clean), one-chain GGUF = the band
  ${data.checks.numa_free.load_extra['droopy|v41|ik|chain'].lo}-${v(data.checks.numa_free.load_extra['droopy|v41|ik|chain'])} from the d1g runs that ran fine (FAIL only when the low end breaks the bar,
  UNKNOWN inside the band); ESTIMATE elsewhere (UNKNOWN when within 3 GiB of the bar). Headroom ledger:
${(data.checks.numa_free.headrooms || []).map(x => '  ' + x.gib + ' GiB = ' + x.verdict + ' (' + x.src + ')').join(';\n')}.
  Single-node boxes (snoopy, odie) PASS on the DRAM usable cap. Knob "node headroom GiB" (default 16, the current
  setting) caps DRAM experts at sum(free - headroom) = 143 GiB on droopy (usable was 150): droopy V4.1 max 16.9 -> 16.7 tok/s.
- **rtr**: a split MXFP4/q2_K model with -rtr gets prefill x0.32 and no decode gain (B0); FAIL on split models.
- **quality**: KL vs share of routed reads at 2 bits, fitted on GLM glm-lite (base 0.00446; ${data.checks.quality.families.glm.slope_kl.src});
  bar KL 0.03 / top-1 0.97 -> max 2-bit share ${v(data.checks.quality.families.glm.max_share)}. The solver repairs any point over the gate by upgrading the hottest 2-bit
  units. V4.1 (ik) has its own line since 10-18: reference floor 0 and the 1017 split at 22.5 % 2-bit reads = KL
  ${data.checks.quality.families.v41.points[0].kl} / top-1 ${data.checks.quality.families.v41.points[0].top1} (MEASURED on lassie vs the 1017 Q8_0) -> max 2-bit share
  ${v(data.checks.quality.families.v41.max_share)} (a 1-point fit; ${data.checks.quality.families.v41.slope_kl.src}).

**What each check would have caught today:**

${caught.map(r => `- ${r.label} (${r.box}, cost ${r.wasted}): ` + r.checks.filter(c => c.status === 'FAIL').map(c => '**' + c.id + '** FAIL (' + c.detail + ')').join('; ')).join('\n')}

Hindsight caveat: the VRAM lower bound 0.50 GiB comes from B1's own OOM. With only what was known before B1 started (15:52Z:
bounds 0-2.25 GiB, B0 #1 graph-growth candidate 1.15 GiB from 12:44Z), the vram_peak check reads UNKNOWN with "OOM expected"
(22.54 + 1.15 = 23.69 > 23.04, in the notebook's MiB / 1000): it still flags B1 before launch. The HDD penalty was paid twice (B0 #1, then both B0b
arms) before the root cause was found at 13:4xZ; a storage check on the recipe would have stopped B0 #1 at launch, because the
disk kind is inventory, not a measurement. The -rtr penalty was first isolated in B0c (the HDD masked it in B0b); from now on
the rtr check prices it for every split recipe.
NUMA headroom: only B1b is an observed failure (node0 MemFree 0.5 GiB). The FAILs on B0c -rtr and B1 are predictions at the
B1b load extra (9.5 GiB) and were not the observed cause of those runs' cost (-rtr, OOM). B0c -rtr off ran clean at the same
10 GiB headroom (+6.0 GiB over its experts), so the extra depends on the config (banks, buffers, page cache) and the check is
conservative between 6.0 and 9.5; B1c at 16 GiB (predicted MemFree 6.5) narrows it.

Other runs (not wasted, or still running), where a check is not PASS (V4.1 quality UNKNOWN on every row is left out):

${recs.filter(r => !caught.includes(r)).map(r => [r, r.checks.filter(c => c.status !== 'PASS' && !(c.id === 'quality' && r.model === 'v41'))]).filter(x => x[1].length).map(([r, cs]) => `- ${r.label} (${r.box}, ${r.when}): ` + cs.map(c => c.id + ' ' + c.status + ' (' + c.detail + ')').join('; ')).join('\n')}

`;
// Flash-Next + 1017 + the pre-launch CLI
const RD = path.join(ROOT, 'recipes');
const rfiles = fs.existsSync(RD) ? fs.readdirSync(RD).filter(f => f.endsWith('.json')).sort() : [];
const rres = rfiles.map(f => { const r = JSON.parse(fs.readFileSync(path.join(RD, f), 'utf8')); return [f, r, FM.checkRecipe(data, r, null)]; });
const verdict = c => c.bad_input ? 'BAD INPUT' : c.checks.some(x => x.status === 'FAIL') ? 'REFUSE' : 'GO';
const ck = data.models.v41.checkpoint || {};
const fnxNew = `## Qwen3.8-Flash-Next, V4.1 Flash-1017, and the pre-launch check

**Flash-Next** (models fnx, fnxc; source lab notes): 125B total / 6B active,
48 layers, 512 experts top-10 + 1 shared, expert = 3 x 2560 x 640. The 51B n-gram table (28.8 GB at IQ4_NL) is an on-file lookup
(NVMe, storage check applies), the 4B MTP head (${'~'}2.2 GiB) rides with the dense part on GPU with an ESTIMATE x1.5 multiplier.
The 640-row expert down tensor only takes 32-wide block types (q2_0, q4_0, iq4_nl, mxfp4, q8_0); mixes that handle the row
limit themselves (q4k_nl, the GSQ and KT mixes) are exempt. Expert heat is SYNTHETIC (GLM's live coding concentration curve
fitted as C(x) = x^a, permuted per layer): the hot-set shares are shaped like GLM's, not read from Flash-Next routing.
KLD per type is a PRIOR log-linear fit on two community points (IQ3_KT 0.078, IQ4_KT 0.056). The quality gate is **UNKNOWN for
every Flash-Next point** until we measure KL on our own prompts. fnxc = the 1x B70 Coder variant (256 experts, ships only as
GSQ IQ3_S; the prune's quality cost is not in the KLD). Host overhead is borrowed from V4.1 per box (ESTIMATE); SYCL host 2.0 ms ESTIMATE.
odie layouts: 2x B70 SYCL layer split (knob "free odie B70 #2" turns the resident-service reserve off) and 1x B70 Coder.

**Strata** (engines.strata, Niko1221/Strata @477ac13c) is check-only: no rate model, so no solver curves. It is listed on
snoopy's 3090 Ti (the CUDA sm_86 + znver5 package odie:/mnt/bulk/build/strata-477ac13c-sm86-z5) and on odie's B70s, because
the source has a SYCL backend. There is no upstream SYCL build; the XeStrata port below runs on odie. Strata sizes its own
expert cache from free VRAM, so its VRAM check is UNKNOWN until the recipe gives vram.peak_gib (the qfn-e2e.sh run
samples it under the 12.7k prompt); then the check is peak vs card total. snoopy /scratch stays a disk of kind UNKNOWN until read.

**XeStrata on odie** (the SYCL port, image xe0.1.40.2.2, Flash-Next GSQ-RCO IQ3_S on B70 #1; MEASURED 10-18, runs tagged floor-1018 and
soak-1018): the CPU-side experts are pinned host RAM (GPUActive), and **XeStrata pins at most the free RAM inside its cgroup minus
8 GiB**. Pinned pages are not charged to the memcg, but the pin sizing reads the cgroup's free memory, so a docker --memory cap shrinks
the pinned share: a 16 GiB --resident-budget-gib pinned 18.2 GiB at --memory 56g but only 3.8 at 12g and 2.2 at 8g, and decode fell
from 69 to 20 and 17 tok/s (the 24g arm pinned 15.6 and thrashed at 51). Rule: **cap >= pinned + charged + 8 GiB**; for
--resident-experts that is 28.3 pinned + about 5 charged (anon 3.0 measured, 4.3 peak in the 79-min soak) + 8 = **42g minimum**, the
standing service's cap. The host floor to reserve is pinned + anon, about 33 GiB (MemAvailable already excludes the pinned part).
The --mmap-experts modes are slower (55-74 tok/s vs 75-80) and lean on page cache: a cold start fills the memcg file cache to the cap.

**V4.1 Flash-1017** is the V4.1 entry from now on: same layout as the checkpoint we measured (READ from its config), so every
droopy measurement transfers. Quality prior = vendor claims, tagged CLAIMED (${cell(ck.quality_prior && ck.quality_prior.src || '')}).
Expert heat is still the pre-1017 routing counts.

**Pre-launch check**: \`node report.js --check <recipe.json> [more.json ...] [--json] [--strict]\` prints PASS / FAIL / UNKNOWN per hard
check for one concrete recipe (box, model, engine, store types + per-tensor types, banks GiB, -rtr, ub, threads, draft, VRAM
reading, NUMA headroom, files with access mode -> disk tier from the box's mount table). Exit 0 = GO, 1 = a FAIL (refuse the run),
2 = bad input, 3 = UNKNOWN or an UNMODELLED speed under --strict. File paths resolve symlinks only when run on the box itself; from the hub pass
readlink -f paths. Example recipes in recipes/:

| recipe | verdict | not PASS |
|---|---|---|
${rres.map(([f, r, c]) => `| ${f} | ${verdict(c)} | ${cell(c.checks.filter(x => x.status !== 'PASS').map(x => x.id + ' ' + x.status).join(', ') || '-')} |`).join('\n')}

`;
let md = `# fit-viz: per-box model fit planner (GLM-5.3-Flash, DeepSeek V4.1 Flash)

Built ${data.meta.built} by ${data.meta.by}. Open \`index.html\` in any browser (it loads \`page/*.js\` and \`src/*.js\`; nothing is fetched). \`data.json\` is the same data.
Rebuild: \`python3 build.py\` (reads the homelab docs, the lab notes lane, the GGUF header dumps in src/, and the sibling
quant-priors.json / quant-recipes.json). Numbers for a report: \`node report.js\`. This file: \`node src/notes.js\`.

Tags: ${Object.keys(data.meta.tags).map(k => '**' + k + '** = ' + data.meta.tags[k]).join('; ')}. No number here is invented as MEASURED:
MEASURED means we ran it or read it live (probe, run log); everything else says what it is.

Sibling files embedded at build time: quant-priors.json ${sib.quant_priors && sib.quant_priors.present ? 'present (' + sib.quant_priors.mtime + ')' : 'absent'},
quant-recipes.json ${sib.quant_recipes && sib.quant_recipes.present ? 'present (' + sib.quant_recipes.mtime + ')' : 'absent'}, QUANT-PRIORS.md ${sib.quant_priors_md ? 'present' : 'absent'}, QUANT-RECIPES.md ${sib.quant_recipes_md ? 'present' : 'absent'}.

${m10}${fnxNew}## The model

**Speed.** Time per token = host overhead (FIT per box and model family) + non-expert weights read on the first usable GPU
+ for every routed expert: heat x bytes / effective rate of its tier. Heat h_e = probability the expert is routed for one
token (per layer the heats sum to k). GPU rate = nameplate bandwidth x ${v(data.assume.gpu_eta)} (ESTIMATE). CPU-side rate =
min(tier bandwidth, DRAM bandwidth x type efficiency kappa_q x box factor f x [2 if the 2x-kernel knob] x [SMT factor]); DRAM
also gets glm-lite's zero-copy rate. ik runs GPU and CPU one after the other (overlap ~1 %, MEASURED lever ledger); glm-lite
overlaps them fully. tok/s = draft multiplier / time.

**Departure from the brief (flagged).** The brief said Optane effective bandwidth = PMem bandwidth x kernel efficiency. The
model uses min(PMem bandwidth, CPU kernel rate) instead: the Qwen sweep ran 3.33-3.68 tok/s on socket-0 Optane for every
type, so the type does not change the Optane rate; the CPU kernel only binds when it is slower than the medium.

**Quality.** KLD of a layout = sum over experts of heat x KLD(type) / (k x L) + the non-expert part. KLD(type) = full-model
KLD if every routed expert used that type (PRIOR where a community row exists). Reported quality = projected top-1 agreement
with the full model = 100 - a x KLD^b, a = ${v(data.assume.quality_a)}, b = ${v(data.assume.quality_b)} (${data.assume.quality_a.src}).
KLD is not comparable across harnesses (quant-priors harness_warning), so each family uses ONE ladder: V4.1 types from the
AtomicChat DeepSeek-V4-Flash-0731 ladder as delta over lossless MXFP4 by expert bpw (log-linear interpolation; K-quants x1.15
per recipe rule 9; above 3.94 bpw the delta halves per extra bit, ESTIMATE); GLM types from the unsloth GLM-5.3-Flash ladder
(AesSedai rows scaled where unsloth has no row); glm-lite EXL3 stores from turboderp's GLM numbers mapped to the unsloth scale
through the two GGUF files both harnesses measured. V4.1 experts are natively MXFP4 (QAT), so MXFP4 has zero expert KLD and
nothing above it gains. lab notes G6 (paired decode KL 0.0032 on the 3.05 store) is kept as a separate MEASURED note: it is an
own-trajectory measure, much gentler than the wikitext scale.

**Solver.** Options are (tier, type). Experts are grouped by plan heat (per layer in GGUF mode, where one type covers a whole
layer tensor; across layers in per-expert mode, which needs the split-expert ik build; one type for the whole store with
glm-lite). A Lagrangian with a price per tier keeps every tier within capacity; sweeping the speed weight lambda
(10^-3 .. 10^4) draws the quality-vs-speed curve. Best-at-target bisects lambda, then refines per expert: promote experts into
leftover faster capacity, then upgrade types (whole layers in GGUF mode) by quality gain per byte while the target holds.
Projections are scored on held-out heat (V4.1: plan on odd coding prompts, score on even; GLM: code prompts in-sample).
Non-expert weights, KV and the draft go on the first usable GPU first; lookup tables (V4.1 engram, token_embd) go to the
slowest tier with room; host copies of GPU-resident experts (ik prefill reads them) go to Optane, else NVMe.

**Higher bits in the slow tier.** In DRAM a byte costs read time plus DRAM space; in Optane only read time, and both that
time and the quality cost scale with the expert's heat, so the Optane type choice is the same for every expert in it. When
DRAM is full, its coldest experts get squeezed below the Optane experts. On droopy (per-expert types) this shows between
about 11 and 14 tok/s for V4.1 (e.g. 12.3 tok/s: Optane experts 3.19 bpw vs the coldest fifth of DRAM experts 2.63) and
about 8-9 tok/s for GLM. Averaged over the tier, DRAM still holds more bits, because DRAM reads are ~5x faster per byte.

## Flags and limits

- Nothing on any box was written, started or loaded. Live capacity came from one round of read-only probes (~01:10Z
  2026-10-17: nvidia-smi / xpu-smi, free, numactl -H, lsblk, df); droopy got one probe. GGUF headers were read header-only.
- Host overhead is fitted per model family (V4.0 and V4.1 share one value; fitting V4.1 alone hit the 0 ms bound).
  Boxes with no anchor for a model borrow droopy's host (ESTIMATE).
- odie's only GLM point (lab notes G12, 7.5 tok/s) is CONTAMINATED (experts partly not resident): shown as a reference, not fitted.
- lassie's GTX 1080 Ti: the 10-17 probe hit an nvidia-smi "Driver/library version mismatch"; the 10-21 probe reads the
  card again (driver 580.178.04).
- ik_llama.cpp cannot drive a B70 (CUDA/CPU only): ik with a moved B70 needs a SYCL/XPU expert worker (code). glm-lite cannot
  use a B70. Mainline SYCL can drive a B70 alone; DeepSeek V4.1 support in mainline SYCL is UNKNOWN.
- Per-expert types need the split-expert ik build (lab notes). Per-expert TIER placement (hot cache + Optane
  placement) is what droopy's lab-best build already does on CUDA.
- The CPU kernel efficiency table is the odie AVX2 smoke (ESTIMATE for other ISAs); droopy uses its own K1 table (MEASURED 10-17).
- Draft multipliers: fitted where a measured twin exists; elsewhere ESTIMATE.

## Calibration (measured vs model)

| box | f | host ms (per model) | split GGUF ms | draft x | Optane GB/s | rms log err |
|---|---|---|---|---|---|---|
${Object.keys(data.cal.boxes).map(bk => { const c = data.cal.boxes[bk]; return `| ${bk} | ${v(c.f).toFixed(3)} ${c.f.tag} | ${Object.keys(c.host).map(m => m + ' ' + v(c.host[m]).toFixed(1) + ' ' + c.host[m].tag).join(', ')} | ${c.split_ms ? v(c.split_ms).toFixed(1) + ' ' + c.split_ms.tag : '-'} | ${Object.keys(c.draft).map(m => m + ' ' + v(c.draft[m]).toFixed(3)).join(', ') || '-'} | ${c.pmem_bw ? v(c.pmem_bw).toFixed(1) + ' FIT' : '-'} | ${c.rms !== undefined ? c.rms.toFixed(3) : '-'} |`; }).join('\n')}

| run | use | measured | model | residual | source |
|---|---|---|---|---|---|
${data.cal.rows.map(r => `| ${cell(r.label)} | ${r.fit} | ${r.meas.toFixed(2)} | ${r.pred.toFixed(2)} | ${(100 * r.resid).toFixed(1)} % | ${cell(r.src)} |`).join('\n')}

## Evidence: which numbers to trust (seen / inferred / unknown)

Every tagged value carries \`evidence\`: **seen** = our own measurement (a run row, probe, bench, or a read of our own file;
\`runs\` lists the ledger rows its source names), **inferred** = fitted, derived, estimated, or a vendor spec / community
claim (\`src\` names it), **unknown** = not modelled. The tag keeps the finer kind (a FIT is inferred, fitted on seen points).
Catalog: ${EV.counts.seen || 0} seen, ${EV.counts.inferred || 0} inferred, ${EV.counts.unknown || 0} unknown values; ${nRuns} seen values cite a ledger row.
Every \`--check\` prediction prints one trust line per number (decode, prefill, VRAM, node MemFree): the count per class, the
UNKNOWN terms, and each INFERRED term with its source. Over the ${predRecs.length} recipes that predict: ${termShare}.

| tag | evidence |
|---|---|
${Object.keys(EV.by_tag).map(k => `| ${k} | ${EV.by_tag[k]} |`).join('\n')}

## External validation (community measurements, \`runs/external.jsonl\`)

${data.external.rows.length} published measurements on hardware we do not own (llama.cpp scoreboards, GitHub threads, r/LocalLLaMA,
blogs), each MEASURED-BY-THEM (inferred) with its URL, date, hardware, engine build, quant, settings and the conditions it
does not declare (\`missing\`). \`node report.js --external\` runs our physics on each row's hardware spec. The rows never set a
term: a decode prediction on a box of the same class prints the class gap in its trust line.

| class (placement) | rows | median abs error | bias (ours / theirs - 1) |
|---|---|---|---|
${XS.by_class_placement.map(g => `| ${g.key} | ${g.n} | ${g.median_abs_pct.toFixed(0)} % | ${(g.bias_pct >= 0 ? '+' : '') + g.bias_pct.toFixed(0)} % |`).join('\n')}
| all predicted | ${XS.rows.filter(r => r.err !== null).length} | ${(100 * med(XS.rows.filter(r => r.err !== null).map(r => Math.abs(r.err)))).toFixed(0)} % | ${(100 * med(XS.rows.filter(r => r.err !== null).map(r => r.err))).toFixed(0)} % |

| row | hardware | engine | quant | placement | theirs tok/s | ours | gap | missing |
|---|---|---|---|---|---|---|---|---|
${XS.rows.map((r, i) => { const x = data.external.rows[i]; return `| [${x.id}](${x.url}) | ${cell(x.hardware.gpu || x.hardware.cpu || '')} | ${cell(x.engine + ' ' + (x.engine_version || '?') + ' ' + (x.backend || ''))} | ${cell(x.quant)} | ${cell(x.placement)} | ${x.measured.decode_tps} | ${r.decode_tps === null ? 'UNMODELLED' : r.decode_tps.toFixed(1)} | ${r.err === null ? cell(r.why) : (r.err >= 0 ? '+' : '') + (100 * r.err).toFixed(0) + ' %'} | ${cell(x.missing.join(', '))} |`; }).join('\n')}

## Bench terms (our boxes, \`runs/bench.jsonl\`)

Three terms the external gaps exposed are measured on our own boxes (\`tools/bench_gpu.sh\`, \`tools/bench_numa.sh\`,
\`tools/record_bench.py\`) and fitted in \`build.py\` (\`bench.gpu\`, \`bench.numa\`). The external rows only re-score them.

GPU line per backend: time per token = read bytes / (nameplate x reach) + host, least squares over dense models of three
sizes (two quant kernels on odie) on one card; MoE overhead = (measured - line at the active bytes) / MoE layers, median over
the MoE models. A line is a driver stack: it predicts rows on its own card vendor (a borrow names the vendor it stands in for).
A dense row off the line by more than ${B.line_resid_max_ms ? v(B.line_resid_max_ms) : '?'} ms, or MoE rows spread past ${B.moe_spread_max ? 100 * v(B.moe_spread_max) : '?'} %, marks the term MEASURED, CONFOUNDED: the
quant kernel moves the time, not only the bytes.

| backend | cards (vendor) | reach of nameplate | host ms/token | MoE ms per MoE layer | evidence |
|---|---|---|---|---|---|
${Object.keys(B.gpu).map(k => { const g = B.gpu[k]; return `| ${k} | ${(g.cards.join(', ') || 'borrowed from ' + g.borrowed_from) + ' (' + (g.vendors || []).join(', ') + ')'} | ${v(g.eta)} | ${v(g.host_ms)} | ${v(g.moe_layer_ms) === null ? 'UNKNOWN' : v(g.moe_layer_ms) + (g.moe_layer_ms.tag !== g.eta.tag ? ' (' + g.moe_layer_ms.tag + ')' : '')} | ${g.eta.evidence} (${g.eta.tag}) |`; }).join('\n')}

NUMA placement gain: decode over the same round's one-node local arm (lassie, 2x EPYC 7601, 8 nodes, HybridAgent-8B Q8_0,
llama.cpp CPU, weights loaded with --mmap 0 under numactl, threads = cores - 1). An external row on S sockets with N nodes
reads one node's share of its spec x the gain of the matching interleave arm, never the summed spec.

| box | placement (policy, weight nodes, thread nodes, sockets, threads) | gain | tag |
|---|---|---|---|
${Object.keys(B.numa).map(b => Object.keys(B.numa[b]).map(k => `| ${b} | ${k} | ${v(B.numa[b][k])} | ${B.numa[b][k].tag} |`).join('\n')).join('\n')}

What the gaps say about our physics (read from the table, not fitted):
- GPU reach: the ${share.length} dense all-GPU rows read ${(100 * share[0]).toFixed(0)}-${(100 * share[share.length - 1]).toFixed(0)} % of nameplate (file bytes x tok/s / spec),
  median ${(100 * med(share)).toFixed(0)} %. Our CUDA card reaches ${B.gpu.cuda ? (100 * v(B.gpu.cuda.eta)).toFixed(0) : '?'} % with a ${B.gpu.cuda ? v(B.gpu.cuda.host_ms) : '?'} ms host; the old borrowed
  2 ms SYCL host with \`assume.gpu_eta\` ${v(data.assume.gpu_eta)} read small dense models 15-38 % slow on CUDA. Rows whose backend has no line keep
  gpu_eta and the borrowed host (both inferred).
- Per-token MoE overhead: all-GPU MoE pays the backend's per-layer term x its MoE layers. A row whose backend has no MoE term
  on its vendor's cards is UNMODELLED, never predicted from bandwidth alone.
- Intel stacks on our B70 (odie, llama.cpp b11706 containers): SYCL reaches ${B.gpu.sycl ? (100 * v(B.gpu.sycl.eta)).toFixed(0) : '?'} % of nameplate; Vulkan (Mesa ANV) reached
  26 % on the image's Mesa 26.0.8 and ${B.gpu.vulkan ? (100 * v(B.gpu.vulkan.eta)).toFixed(0) : '?'} % on Mesa 26.2.5 (int dot and coopmat on in both). The Vulkan line is now the Mesa 26.2.5 stack (\`bench.line_stack\`).
  **Mesa A/B (2026-10-19, odie B70 #1 only: /dev/dri/renderD129 alone in the container, -sm none -mg 0)**: A0 image Mesa 26.0.8,
  A1 A0 + GGML_VK_DISABLE_MMVQ=1, B0 Mesa 26.2.5 (kisak PPA debs mounted read-only, VK_DRIVER_FILES + LD_LIBRARY_PATH), B1 B0 + MMVQ off;
  llama2-7B Q4_0 + the five dense bench models, -fa 0 and 1 (rows \`*-ab-<arm>-fa<fa>-*\`). Gate, every arm: test-backend-ops
  -o MUL_MAT (q4_0, q8_0, q4_K, q6_K, f16 x f32; 603 cases) 5 runs, 0 failures; llama2-7B Q4_0 perplexity on a 7.5 KB text (4 x 512)
  bit-identical across all arms and repeats (4.1658, every chunk). Perplexity runs batched matmuls, so it does not exercise the
  mat-vec (MMVQ) path; the MUL_MAT gate covers n = 1-8. One line per stack (dense rows, reach of the 608 GB/s nameplate):

| stack | MMVQ | -fa | rate GB/s | reach | host ms | on the line |
|---|---|---|---|---|---|---|
${(B.stacks.vulkan || []).map(x => `| ${x.stack.stack} | ${x.stack.mmvq} | ${x.stack.fa} | ${v(x.rate_gbs)} | ${(100 * v(x.eta)).toFixed(0)} % | ${v(x.host_ms)} | ${x.on_line ? 'yes' : ''} |`).join('\n')}

  - B0 beats A0 on every model: llama2-7B Q4_0 86-87 to 101-104 tok/s (a Linux B70 on Mesa 26.0.3 reads 80.4, seen; the
    scoreboard's 112 is Windows on Intel's driver, llama-bench.exe b8665, seen), Q8_0 2.4-2.6x, Q4_K_M 1.8x, Q4_0 1.2-1.6x.
    MoE on B0 (-fa 1): gpt-oss-20b MXFP4 99.9 and Qwen3-30B-A3B Q4_K_S 118.9 tok/s (Mesa 26.0.8: 44.8 and 78.5; SYCL 137.1 and 114.8).
  - MMVQ off (A1, B1) helps only Q4_0 on Mesa 26.0.8 (+4-17 %) and costs Q8_0 / Q4_K_M everywhere; it stays on.
  - -fa 0 vs 1 moves decode by at most ~3 %; the line takes -fa 1, as llama-server runs it.
  - The external B70 Vulkan rows ran older or unstated stacks, so against this line the Gemma4-26B-A4B Vulkan row reads
    ${gap('arc-b70-gemma4-26b-a4b-vulkan')} (their 40 tok/s) and the scoreboard llama2-7B Q4_0 ${gap('arc-b70-vulkan-llama2-7b-q4_0')}. The Q4_0 rows sit above
    the line (the quant kernels differ, so the line is MEASURED, CONFOUNDED). Rollback = drop the env and the mount (odie
    ~/b70-lane/mesa-26.2.5 is a copy of lab notes).
  Vulkan reach on Intel is a property of the stack, not the card. The B70 Gemma4 SYCL row (${gap('arc-b70-gemma4-26b-a4b-sycl')})
  is not a MoE term: our SYCL MoE rows cost ${B.gpu.sycl ? v(B.gpu.sycl.moe_layer_ms) : '?'} ms per layer, so an 8x miss is that build's Gemma4 path (unmeasured here).
- Dual-socket CPU decode: one node's share x the measured cross-socket interleave gain. On lassie two nodes on two sockets
  decode at ${B.numa.lassie && B.numa.lassie['interleave m2 c2 s2 t15'] ? v(B.numa.lassie['interleave m2 c2 s2 t15']) : '?'}x one local node, two nodes on one socket at ${B.numa.lassie && B.numa.lassie['interleave m2 c2 s1 t15'] ? v(B.numa.lassie['interleave m2 c2 s1 t15']) : '?'}x: spreading across sockets loses bandwidth instead of adding it.
  The Naples gain is applied to Genoa rows (one node per socket): a topology transfer, so those rows keep their gap in view.
- Single-socket CPU-only and the 4 consumer hybrid rows (expert layers on DDR5, the rest on a GeForce) are unchanged by these terms.
${dllmMd}
## Report output (node report.js)

\`\`\`
${rep.trim()}
\`\`\`

## Every tagged number (path, value, evidence, tag, source)

| path | value | evidence | tag | source |
|---|---|---|---|---|
${rows.map(r => `| ${cell(r[0])} | ${cell(fmt(r[1]))} | ${r[4]} | ${r[2]} | ${cell(r[3])} |`).join('\n')}

Fitted at load time (FIT): ${Object.keys(data.cal.boxes).map(bk => { const c = data.cal.boxes[bk]; return bk + ' f ' + v(c.f).toFixed(3) + ' (' + c.f.src + ')'; }).join('; ')}.

Heat sources: ${Object.keys(data.models).map(mk => data.models[mk].short + ': ' + Object.keys(data.models[mk].heat.sets).map(k => { const h = data.models[mk].heat.sets[k]; return k + (h.src ? ' = ' + h.src : ' = mean of ' + h.sum.join('+')); }).join('; ')).join(' | ')}.
Owner-layout coding set: ${data.alloc3 ? data.alloc3.src + ', ' + data.alloc3.count + ' experts' : 'not built (input 2026-10-08-alloc-3bpw.json missing)'}.
`;
fs.writeFileSync(path.join(ROOT, 'NOTES.md'), md);
console.log('NOTES.md', rows.length, 'tagged numbers,', Math.round(md.length / 1024), 'KB');
