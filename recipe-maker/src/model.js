// Fit planner model + solver. Shared by index.html (browser) and report.js (node).
// Units: bytes, seconds internally; GiB / ms / tok/s at the edges.
(function (root) {
  'use strict';
  const GiB = 1073741824;
  const v = x => (x !== null && typeof x === 'object' && 'v' in x) ? x.v : x;

  // ---------- evidence: seen | inferred | unknown (data.meta.evidence, the one table in build.py) ----------
  // A catalog value carries its class from the build; a value made here (calibrate's FIT rows) is classed from its tag by the same
  // table. No tag, or a tag the table does not know: unknown (nothing untracked passes as trusted).
  function evidenceOf(data, x) {
    if (x !== null && typeof x === 'object' && x.evidence) return x.evidence;
    const E = data.meta && data.meta.evidence, tag = x !== null && typeof x === 'object' ? x.tag : null;
    if (!E || !tag || !(tag in E.by_tag)) return 'unknown';
    if (tag === 'READ' && E.seen_reads.v.some(p => String(x.src || '').startsWith(p))) return 'seen';
    return E.by_tag[tag];
  }
  // one term of a prediction: its name, value, evidence class, tag and source (and the ledger rows a seen value cites)
  function term(data, name, x) {
    const o = x !== null && typeof x === 'object' ? x : { v: x === undefined ? null : x };
    return Object.assign({ term: name, value: o.v === undefined ? null : o.v, evidence: evidenceOf(data, o), tag: o.tag || null, src: o.src || null }, o.runs ? { runs: o.runs } : {});
  }
  // roll-up of one predicted number: counts per class, and a grade = the weakest class it rests on
  function rollup(metric, value, unit, terms) {
    const c = { seen: 0, inferred: 0, unknown: 0 };
    terms.forEach(t => c[t.evidence]++);
    const grade = value === null || value === undefined || c.unknown ? 'unknown' : c.inferred ? 'inferred' : 'seen';
    return { metric, value: value === undefined ? null : value, unit, seen: c.seen, inferred: c.inferred, unknown: c.unknown, grade, terms };
  }
  // "decode 21.0 tok/s: 3 seen, 2 inferred terms; rests on INFERRED terms: draft speed-up 1.20 (FIT: ...)"
  function fmtRollup(r, srcLen) {
    const L = srcLen === undefined ? 70 : srcLen, num = x => typeof x === 'number' ? (Math.abs(x) >= 100 ? x.toFixed(0) : Math.abs(x) >= 1 ? x.toFixed(2) : x.toPrecision(2)) : String(x);
    const cut = s => { s = String(s || 'no source'); return s.length > L ? s.slice(0, L - 3) + '...' : s; };
    // a src that already opens with its tag ("ESTIMATE: ...") is printed once
    const one = t => { const tg = t.tag || 'untagged', s = String(t.src || ''); return t.term + ' ' + (t.value === null ? '' : num(t.value) + ' ') + '(' + (s.startsWith(tg + ':') ? cut(s) : tg + ': ' + cut(s)) + ')'; };
    const head = r.metric + ' ' + (r.value === null ? 'UNMODELLED' : num(r.value) + (r.unit ? ' ' + r.unit : '')) + ': ' + r.seen + ' seen, ' + r.inferred + ' inferred, ' + r.unknown + ' unknown terms';
    const inf = r.terms.filter(t => t.evidence === 'inferred'), unk = r.terms.filter(t => t.evidence === 'unknown');
    const x = r.external;
    const ext = !x ? '' : x.n ? '; external check (' + x.group + ', ' + x.n + ' community rows): our physics is off by median ' + x.median_abs_pct.toFixed(0) + ' % (bias ' + (x.bias_pct >= 0 ? '+' : '') + x.bias_pct.toFixed(0) + ' %) there' : '; external check (' + x.group + '): no community rows';
    return head + (unk.length ? '; UNKNOWN: ' + unk.map(one).join('; ') : '') + (inf.length ? '; rests on INFERRED terms: ' + inf.map(one).join('; ') : '') + (r.grade === 'seen' ? '; every term SEEN' : '') + ext;
  }

  // ---------- heat ----------
  // Heat h_e = P(expert e is routed for one token). Per layer it sums to k.
  function heatSet(data, mk, name) {
    const m = data.models[mk];
    m._heat = m._heat || {};
    if (m._heat[name]) return m._heat[name];
    const spec = m.heat.sets[name];
    let layers;
    if (spec.sum) {
      const parts = spec.sum.map(n => heatSet(data, mk, n));
      const out = new Float64Array(m.L * m.E);
      parts.forEach(p => { for (let i = 0; i < out.length; i++) out[i] += p[i]; });
      // parts are already per-layer normalized to k; average keeps the sum at k
      for (let i = 0; i < out.length; i++) out[i] /= parts.length;
      m._heat[name] = out;
      return out;
    }
    layers = spec.layers;
    const out = new Float64Array(m.L * m.E);
    const eps = v(m.heat.floor) || 0; // pseudo-count so never-seen experts are not exactly free
    for (let l = 0; l < m.L; l++) {
      const row = layers[l];
      let s = 0;
      for (let e = 0; e < m.E; e++) s += row[e] + eps;
      for (let e = 0; e < m.E; e++) out[l * m.E + e] = m.k * (row[e] + eps) / s;
    }
    m._heat[name] = out;
    return out;
  }

  // ---------- types ----------
  function typeBytes(data, mk, q) {
    const m = data.models[mk];
    return m.P * v(data.types[q].bpw) / 8;
  }
  function typeKld(data, mk, q) {
    const t = data.types[q];
    const fam = data.models[mk].kld_family;
    const o = data.overrides && data.overrides.kld && data.overrides.kld[fam] && data.overrides.kld[fam][q];
    if (o !== undefined) return o;
    return v(t.kld[fam]);
  }

  // projected top-1 agreement with the full model (%) from KLD: 100 - a * KLD^b (community fit)
  function qualityOf(data, kld) {
    return 100 - v(data.assume.quality_a) * Math.pow(Math.max(0, kld), v(data.assume.quality_b));
  }

  // ---------- box context ----------
  // Builds the tiers (capacity for routed experts, bandwidth, reserves) for one box x model x engine x knobs.
  function context(data, bk, mk, eng, knobs) {
    const box = data.boxes[bk], m = data.models[mk], E = data.engines[eng];
    const cal = (data.cal && data.cal.boxes[bk]) || {};
    const kn = knobs || {};
    const tiers = [];
    const notes = [];
    // lane probes and parts (src/advisor.js): kn.lane = { scale: { vram_bw, dram_bw, pmem_bw, nvme_bw, cpu }, add_gib: { vram, dram, pmem } }
    const LS = (kn.lane && kn.lane.scale) || {}, LA = (kn.lane && kn.lane.add_gib) || {}, ls = k => LS[k] || 1;
    // GPUs
    const gpus = box.gpus.map(g => JSON.parse(JSON.stringify(g)));
    if (kn.moveB70 && kn.moveB70 === bk) gpus.push(JSON.parse(JSON.stringify(data.b70move.gpu)));
    if (kn.moveB70 && bk === 'odie' && kn.moveB70 !== 'none') {
      const i = gpus.findIndex(g => g.id === 'b70_1');
      if (i >= 0) gpus.splice(i, 1);
    }
    gpus.forEach((g, gi) => {
      const res = [];
      (g.other || []).forEach(o => {
        let gib = v(o.gib);
        if (o.desktop && kn.freeDesktop) gib = 0;
        if (o.household && kn.freeHousehold) gib = 0;
        res.push({ label: o.what, gib, tag: o.gib.tag, src: o.gib.src, kind: 'other' });
      });
      // a card the engine reaches through a second backend (g.drive[eng], catalog row): banks only, read at the backend's bench
      // line when that line was measured on this card (else nameplate x gpu_eta), plus the hop the token pays per layer it
      // crosses to the card (cal.hop, fitted on the ledger runs that bank experts there; 0 = UNKNOWN until one does)
      const d = g.drive && g.drive[eng], bl = d && data.bench && data.bench.gpu && data.bench.gpu[d.backend];
      const line = bl && (bl.cards || []).indexOf(bk + ' ' + g.id) >= 0 ? bl.rate_gbs : null, hop = d && cal.hop && cal.hop[g.id];
      tiers.push({
        id: 'gpu' + gi, kind: 'gpu', name: g.name, total: (v(g.vram_gib) + (gi === 0 ? LA.vram || 0 : 0)) * GiB,
        bw: v(line || g.bw) * 1e9 * ls('vram_bw'), eta: line ? 1 : v(data.assume.gpu_eta), reserves: res, gpu: g,
        engineOk: !g.engines || g.engines.indexOf(eng) >= 0,
        drive: d || null, banks: !!d && d.role === 'banks', bwTerm: line, hop: hop ? v(hop) / 1000 : 0, hopTerm: d ? hop || { v: 0, tag: 'UNKNOWN', src: 'UNKNOWN: no clean ledger run banks experts on ' + g.id + ' through ' + d.backend } : null
      });
      if (d) notes.push(g.name + ' holds ' + (d.role === 'banks' ? 'expert banks only' : 'experts') + ' (' + d.backend + ', ' + d.types.join('/') + '; ' + d.build + '), read at ' + v(line || g.bw).toFixed(0) + ' GB/s' + (line ? ' (bench line)' : ' x gpu_eta') + (hop ? ' + ' + v(hop).toFixed(3) + ' ms per layer crossed (' + hop.tag + ')' : ' (cross-backend hop UNKNOWN: not added)') + '.');
    });
    const smt = kn.smt ? v(box.cpu.smt_mult) : 1;
    const kern = kn.kernel2x ? 2 : 1;
    const f = v(cal.f !== undefined ? cal.f : box.cpu.f_default);
    // DRAM
    let dramGiB = v(box.dram.usable_gib), dramTotal = v(box.dram.total_gib);
    // a fixed per-node headroom (dram-fill placement) caps the experts DRAM can hold: sum over nodes of (free at plan time - headroom)
    const nodeAvail = (box.dram.nodes || []).map(x => x.avail_gib ? v(x.avail_gib) : null);
    // headroom is one figure for every node, or a per-node list (more on node 0, which carries the load extra)
    if (kn.numaHeadroom && nodeAvail.length && nodeAvail.every(x => x !== null)) {
      const cap = nodeAvail.reduce((a, x, i) => a + Math.max(0, x - headroomAt(kn.numaHeadroom, i)), 0);
      if (cap < dramGiB) { notes.push('dram-fill headroom ' + kn.numaHeadroom + ' GiB per node caps DRAM experts at ' + cap.toFixed(1) + ' GiB (usable ' + dramGiB.toFixed(1) + ').'); dramGiB = cap; }
    }
    if (kn.addDRAM && box.upgrades && box.upgrades.dram) {
      dramGiB += v(box.upgrades.dram.gib); dramTotal += v(box.upgrades.dram.gib);
    }
    if (LA.dram) { dramGiB += LA.dram; dramTotal += LA.dram; }
    // the CPU expert-kernel rate is fitted as a fraction of the box's DRAM read rate (f x kappa), and read as compute: a faster
    // DRAM lane (lane dram_bw) raises only the tier's read ceiling, a faster CPU (lane cpu, kernel2x) only the kernel rate
    const Bd = v(box.dram.bw) * 1e9;
    tiers.push({
      id: 'dram', kind: 'cpu', name: 'DRAM', total: dramTotal * GiB, usableOverride: dramGiB * GiB,
      bw: Bd * ls('dram_bw'), reserves: [{ label: 'OS, buffers, KV, co-tenants', gib: dramTotal - dramGiB, tag: box.dram.usable_gib.tag, src: box.dram.usable_gib.src, kind: 'os' }],
      nodes: box.dram.nodes
    });
    if (box.pmem) {
      const up = kn.addOptane && box.upgrades && box.upgrades.pmem;
      // upgrade rate = the box's current (fitted) rate x bw_scale (knob optaneBwScale overrides), else the upgrade's fixed bw
      const sc = up && (kn.optaneBwScale || (up.bw_scale && v(up.bw_scale)));
      const bwTerm = !up ? box.pmem.bw : sc ? { v: v(box.pmem.bw) * sc, tag: 'ESTIMATE', evidence: 'inferred', src: 'upgrade ' + up.what + ': ' + v(box.pmem.bw).toFixed(1) + ' GB/s now x ' + sc + (kn.optaneBwScale ? ' (knob optaneBwScale)' : ' (' + up.bw_scale.src + ')') } : up.bw;
      tiers.push({ id: 'pmem', kind: 'cpu', name: 'Optane PMem', total: ((up ? v(up.gib) : v(box.pmem.gib)) + (LA.pmem || 0)) * GiB, bw: v(bwTerm) * 1e9 * ls('pmem_bw'), bwTerm, reserves: [] });
    }
    tiers.push({ id: 'nvme', kind: 'disk', name: 'NVMe', total: v(box.nvme.gib) * GiB, bw: v(box.nvme.bw) * 1e9 * ls('nvme_bw'), reserves: [] });

    // CPU expert-kernel rate for a type (bytes of that type per second); a box's own measured table wins over the type prior
    const kappa = q => kappaOf(data, bk, q);
    // threads below the physical core count slow the compute-bound CPU expert kernels in proportion (ESTIMATE: the kernel rate
    // table assumes every physical core; SMT siblings are the smt knob)
    const phys = box.cpu.phys ? v(box.cpu.phys) : null;
    const thr = kn.threads && phys ? Math.min(kn.threads, phys) / phys : 1;
    if (thr < 1) notes.push('threads ' + kn.threads + ' of ' + phys + ' physical cores: CPU expert kernels at ' + (100 * thr).toFixed(0) + ' % (ESTIMATE: compute-bound kernels scale with cores).');
    const cpuRate = q => Bd * kappa(q) * f * kern * smt * thr * ls('cpu');
    // NUMA page layout of the DRAM experts (recipe numa.layout): the box's measured CPU expert time vs interleave, 1 when unmeasured
    const layF = kn.numaLayout && box.numa_layout && box.numa_layout[kn.numaLayout] ? v(box.numa_layout[kn.numaLayout]) : 1;
    const zc = (E.zero_copy_gbs ? v(E.zero_copy_gbs) * 1e9 : 0);
    // a card's hop per layer crossed, spread over the k routed picks of that layer (exact for whole layers on the card)
    const rawRate = (t, q) => {
      if (t.kind === 'gpu') return t.hop ? 1 / (1 / (t.bw * t.eta) + t.hop / (m.k * typeBytes(data, mk, q))) : t.bw * t.eta;
      const r = Math.min(t.bw, cpuRate(q) / (t.id === 'dram' ? layF : 1));
      return t.id === 'dram' ? r + zc : r;
    };
    // dynamic GPU cache (glm-lite): only a measured share of the planned GPU heat hits the GPU; the rest is read from DRAM
    const hc = data.cal && data.cal.engines && data.cal.engines[eng] && data.cal.engines[eng].hit_real;
    const hitR = hc ? Math.min(1, v(hc)) : 1;
    const dramT = tiers.find(t => t.id === 'dram');
    const rate = (t, q) => (t.kind === 'gpu' && hitR < 1 && dramT) ? 1 / (hitR / rawRate(t, q) + (1 - hitR) / rawRate(dramT, q)) : rawRate(t, q);
    if (hitR < 1) notes.push('GPU expert cache realizes ' + (100 * hitR).toFixed(0) + ' % of the planned GPU heat (' + hc.tag + ': ' + hc.src + '); the misses are read from DRAM.');

    // fixed allocations: dense, draft and KV on the engine's first usable GPU; lookup tables to the slowest tier with room
    const g0 = tiers.find(t => t.kind === 'gpu' && t.engineOk && !t.banks) || tiers[0];
    const densePrec = kn.densePrec || 'q8_0';
    const denseGiB = denseGiBpre(E, m, data, kn);
    const split = eng === 'ik' && !!kn.perExpertTypes;
    const mg = marginOf(data, eng, split);
    const marginGiB = E.kv_includes_margin ? 0 : marginPick(mg, kn);
    // -ub auto: 4096 (prefill +45 %, E2 vs E1) when its compute buffer fits beside the non-expert weights + margin, else 512
    let ub = kn.ub || 'auto';
    if (ub === 'auto') {
      const room = (g0.total - g0.reserves.reduce((a, r) => a + r.gib * GiB, 0)) / GiB - denseGiBpre(E, m, data, kn) - marginGiB;
      ub = (E.kv_gib !== undefined || !m.kv_gib_ub512 || room >= v(m.kv_gib) + 0.5) ? '4096' : '512';
      if (ub === '512') notes.push('-ub 4096 does not fit beside the non-expert weights on ' + g0.name + ': -ub 512 (prefill about 30 % slower, E1 24.5 vs E2 35 tok/s on droopy).');
    }
    const kvGiB = kvGiBpre(E, m, Object.assign({}, kn, { ub }), mk);
    let draftOn = !!kn.draft && !!m.draft && !E.no_draft;
    if (draftOn && (g0.total - g0.reserves.reduce((a, r) => a + r.gib * GiB, 0)) / GiB < kvGiB + marginGiB + denseGiBpre(E, m, data, kn) + draftGiBpre(m, kn)) {
      draftOn = false;
      notes.push('The draft model does not fit beside the non-expert weights on ' + g0.name + ', so it is off here.');
    }
    if (kn.draft && E.no_draft) notes.push(E.name.split(' (')[0] + ' has no draft/MTP path today.');
    const draftGiB = draftOn ? draftGiBpre(m, kn) : 0;
    const fixed = [];
    const kvS = kvSrc(E, m, mk, ub), dnS = denseSrc(E, m, kn);
    fixed.push({ tier: g0.id, label: 'KV cache + compute buffers (-ub ' + (E.kv_gib ? 'engine' : ub) + ')', gib: kvGiB, kind: 'kv', tag: kvS && kvS.tag, src: kvS && kvS.src, evidence: evidenceOf(data, kvS) });
    if (marginGiB > 0) fixed.push({ tier: g0.id, label: 'VRAM peak margin (first real prompt grows the graph; ' + (mg ? mg.lo + '-' + mg.hi + ' GiB ' + mg.tag : 'unknown') + ')', gib: marginGiB, kind: 'margin', tag: mg ? mg.tag : 'ESTIMATE', src: mg ? mg.src : '' });
    fixed.push({ tier: g0.id, label: 'non-expert weights (' + (E.dense_gib ? E.dense_label : densePrec) + ')', gib: denseGiB, kind: 'dense', tag: dnS && dnS.tag, src: dnS && dnS.src, evidence: evidenceOf(data, dnS) });
    if (draftOn) fixed.push({ tier: g0.id, label: m.draft.label + (mtpExpertsGiB(m, kn) ? ' incl. its experts ' + mtpExpertsGiB(m, kn).toFixed(2) : kn.mtpExperts === 'cpu' ? ', its experts on the host (mtp_experts cpu)' : ''), gib: draftGiB, kind: 'draft', tag: m.draft.vram_gib.tag, src: m.draft.vram_gib.src, evidence: evidenceOf(data, m.draft.vram_gib) });
    // measured run-time parts (checks.vram_peak.runtime): the prefill pool scales with -ub, the MTP context compute buffer with its
    // micro-batch (mtp_ubatch, default the target -ub), plus the speculative checkpoints of an MTP run
    const RT = runtimeOf(data, mk, eng);
    if (RT) {
      const ubN = Number(ub) > 0 ? Number(ub) : 512;
      if (RT.pool_gib_per_tok) fixed.push({ tier: g0.id, label: 'prefill memory pool (-ub ' + ubN + ')', gib: v(RT.pool_gib_per_tok) * ubN, kind: 'pool', tag: RT.pool_gib_per_tok.tag, src: RT.pool_gib_per_tok.src });
      if (draftOn && RT.mtp_compute_gib_per_tok) {
        const mu = Number(kn.mtpUb) > 0 ? Math.min(ubN, Number(kn.mtpUb)) : ubN;
        fixed.push({ tier: g0.id, label: 'MTP context compute buffer (micro-batch ' + mu + (Number(kn.mtpUb) > 0 ? ', mtp_ubatch' : ', the target -ub') + ')', gib: v(RT.mtp_compute_gib_per_tok) * mu, kind: 'draft_rt', tag: RT.mtp_compute_gib_per_tok.tag, src: RT.mtp_compute_gib_per_tok.src });
      }
      if (draftOn && RT.draft_fixed_gib) fixed.push({ tier: g0.id, label: 'speculative checkpoints + CUDA runtime (MTP on)', gib: v(RT.draft_fixed_gib), kind: 'draft_rt', tag: RT.draft_fixed_gib.tag, src: RT.draft_fixed_gib.src });
      const cx = Number(kn.ctx) > 0 ? Number(kn.ctx) : 0, cb = RT.ctx_base || 4096;
      if (cx > cb && RT.kv_gib_per_ctx_tok) fixed.push({ tier: g0.id, label: 'KV cache beyond n_ctx ' + cb + ' (ctx ' + cx + ')', gib: v(RT.kv_gib_per_ctx_tok) * (cx - cb), kind: 'ctx', tag: RT.kv_gib_per_ctx_tok.tag, src: RT.kv_gib_per_ctx_tok.src });
      if (cx > cb && RT.compute_gib_per_ctx_tok) fixed.push({ tier: g0.id, label: 'prefill graph context buffers beyond n_ctx ' + cb + ' (ctx ' + cx + ')', gib: v(RT.compute_gib_per_ctx_tok) * (cx - cb), kind: 'ctx', tag: RT.compute_gib_per_ctx_tok.tag, src: RT.compute_gib_per_ctx_tok.src });
    }
    const free = t => {
      let c = (t.usableOverride !== undefined ? t.usableOverride : t.total);
      if (t.usableOverride === undefined) t.reserves.forEach(r => c -= r.gib * GiB);
      fixed.filter(x => x.tier === t.id).forEach(x => c -= x.gib * GiB);
      return c;
    };
    (m.lookups || []).forEach(lk => {
      const order = ['nvme', 'pmem', 'dram'];
      let placed = false;
      for (const id of order) {
        const t = tiers.find(x => x.id === id);
        if (lk.on_file && storageKind(box, id) === 'hdd') continue; // storage check: never a rotational disk
        if (t && free(t) >= v(lk.gib) * GiB) { fixed.push({ tier: id, label: lk.label, gib: v(lk.gib), kind: 'lookup', tag: lk.gib.tag, cls: lk.cls, on_file: !!lk.on_file }); placed = true; break; }
      }
      if (!placed) notes.push('No tier has room for ' + lk.label + ' (' + v(lk.gib).toFixed(0) + ' GiB).');
    });
    tiers.forEach(t => { t.cap = (t.kind === 'gpu' && !t.engineOk) ? 0 : Math.max(0, free(t)); });
    tiers.forEach(t => { if (t.kind === 'gpu' && !t.engineOk) notes.push(t.name + ' sits idle: ' + E.name.split(' (')[0] + ' cannot drive it.'); });
    const bad = !g0.engineOk || free(g0) < 0;
    if (bad) notes.push('Non-expert weights + KV do not fit ' + g0.name + ': this engine cannot run here.');
    // host copy of GPU-resident experts (ik prefill reads the host copy): slowest CPU-side tier
    const copyTier = kn.hostCopy ? (tiers.find(t => t.id === 'pmem') || tiers.find(t => t.id === 'nvme')) : null;
    // -rtr: the CPU repacks MXFP4 (mxfp4_r8 replaces mxfp4); off by default (rtr check)
    const famOk = q => { const k = data.types[q].kld && data.types[q].kld[m.kld_family]; return !!k && v(k) !== null; };
    const types = (E.types || []).filter(q => data.types[q] && famOk(q) && (!m.types_only || m.types_only.indexOf(q) >= 0) && !(data.types[q].needs_code && !kn.allowCodeTypes))
      .filter(q => kn.rtr ? !(q === 'mxfp4' && (E.types || []).indexOf('mxfp4_r8') >= 0) : !data.types[q].rtr);
    const spS = eng === 'ik' && kn.perExpertTypes ? splitSrc(data, bk, mk) : null, splitMs = spS ? v(spS) || 0 : 0;
    if (spS && (v(spS) === null || v(spS) === undefined)) notes.push('Split GGUF (per-expert types): host overhead for ' + mk + ' UNKNOWN, not added (' + spS.src + ').');
    const host = (hostMs(data, bk, mk, eng) + splitMs) / 1000;
    if (splitMs) notes.push('Split GGUF (per-expert types): +' + splitMs.toFixed(1) + ' ms/token host overhead (' + spS.tag + ', ' + spS.src + ').');
    const denseTime = denseGiB * GiB / (g0.bw * g0.eta);
    const given = kn.draftMult !== undefined && kn.draftMult !== null, mult = draftOn ? (given ? kn.draftMult : draftDefault(data, bk, mk)) : 1;
    // the draft's verify weight on CPU expert time (ledgerDrafts): with the draft on, decode = mult / (T + cpuW x CPU expert time)
    const cpuW = draftOn && !given ? draftCpuW(data, bk, mk) : 0;
    return {
      g0, data, bk, mk, eng, m, box, E, kn, tiers, rate, rawRate, hitR, missTi: dramT ? tiers.indexOf(dramT) : -1, cpuRate, types, host, denseTime, mult, cpuW, fixed, copyTier, notes, f, smt, kern,
      overlap: v(E.overlap) || 0, kldDense: v(data.types[densePrec].kld_dense) || 0, draftOn, bad, split, margin: mg, marginGiB, kvGiB, ub
    };
  }

  function kappaOf(data, bk, q) {
    const o = data.overrides && data.overrides.kappa && data.overrides.kappa[q];
    if (o !== undefined) return o;
    const bx = bk && data.boxes[bk] && data.boxes[bk].kappa;
    if (bx && bx[q] !== undefined) return v(bx[q]);
    return v(data.types[q].kappa);
  }
  function storageKind(box, tierId) {
    if (tierId === 'dram' || tierId === 'pmem') return 'memory';
    if (tierId === 'nvme') return (box.nvme && box.nvme.kind) || 'unknown';
    return tierId; // a mount kind given directly (recipes)
  }
  function marginOf(data, eng, split) {
    const M = data.checks && data.checks.vram_peak && data.checks.vram_peak.margin;
    if (!M) return null;
    return M[eng === 'ik' ? (split ? 'ik_split' : 'ik_gguf') : eng] || null;
  }
  function marginPick(mg, kn) {
    if (!mg) return 0;
    if (kn && kn.vramMargin === 'likely' && mg.candidate !== null && mg.candidate !== undefined) return mg.candidate;
    return mg.hi || 0;
  }

  function denseGiBpre(E, m, data, kn) {
    const densePrec = kn.densePrec || 'q8_0';
    if (E.dense_gib) return v(E.dense_gib);
    const byType = m.dense_gib_by_type && kn.storeType && m.dense_gib_by_type[kn.storeType];   // a shipped file's own non-expert bytes
    if (byType && densePrec === 'q8_0') return v(byType);
    return v(m.dense_gib) * v(data.types[densePrec].bpw) / v(m.dense_bpw);
  }
  // the MTP block's routed experts on the GPU (GLM on ik --cpu-moe, MEASURED) unless the recipe sends them to the host
  function mtpExpertsGiB(m, kn) {
    const d = m.draft;
    if (!d || !d.experts_gib || !(d.experts_on_gpu && v(d.experts_on_gpu)) || kn.mtpExperts === 'cpu') return 0;
    const x = d.experts_gib[kn.storeType] || d.experts_gib[m.base_type];
    return x ? v(x) : 0;
  }
  // a recipe's shipped store type (one unsplit type: its own non-expert bytes) and where its MTP experts live
  const storeKnobs = r => { const st = r.store || {}; return { storeType: !st.split && (st.types || []).length === 1 ? st.types[0] : undefined, mtpExperts: r.mtp_experts || 'gpu', mtpUb: r.mtp_ubatch, ctx: r.ctx }; };
  // VRAM parts a predicted post-warm-up adds for a model|engine (checks.vram_peak.runtime): prefill pool, MTP runtime, offload graph
  const runtimeOf = (data, mk, eng) => { const R = data.checks && data.checks.vram_peak && data.checks.vram_peak.runtime; return (R && R[mk + '|' + eng]) || null; };
  const draftGiBpre = (m, kn) => v(m.draft.vram_gib) + mtpExpertsGiB(m, kn);
  function kvGiBpre(E, m, kn, mk) {
    if (E.kv_gib !== undefined) return v(E.kv_gib);
    // an engine's own measured KV + compute figures for a model (recipe_rates.kv_model) win over the model's
    const km = E.recipe_rates && E.recipe_rates.kv_model && E.recipe_rates.kv_model[mk];
    if (km) m = Object.assign({}, m, { kv_gib: km.ub4096, kv_gib_ub512: km.ub512 });
    const ub = kn ? Number(kn.ub) : NaN;
    // a numeric -ub other than 512/4096: the compute buffer grows linearly with the ubatch (ESTIMATE, between the 512 and 4096 figures)
    if (m.kv_gib_ub512 && ub > 0 && ub !== 512 && ub !== 4096) return v(m.kv_gib_ub512) + (v(m.kv_gib) - v(m.kv_gib_ub512)) * (ub - 512) / 3584;
    return v(kn && kn.ub === '512' && m.kv_gib_ub512 ? m.kv_gib_ub512 : m.kv_gib);
  }

  // the tagged value a planner term reads (hostSrc / draftSrc / kappaSrc / denseSrc / kvSrc); the plain number is v() of it
  function hostSrc(data, bk, mk, eng) {
    const E = data.engines[eng];
    if (E.host_ms) return E.host_ms;
    const he = data.models[mk].host_engine;
    if (he && he[eng]) return he[eng];
    const c = data.cal && data.cal.boxes[bk];
    if (c && c.host && c.host[mk] !== undefined) return c.host[mk];
    return data.boxes[bk].host_default[mk];
  }
  function hostMs(data, bk, mk, eng) { return v(hostSrc(data, bk, mk, eng)); }
  function draftSrc(data, bk, mk) {
    const c = data.cal && data.cal.boxes[bk];
    if (c && c.draft && c.draft[mk] !== undefined) return c.draft[mk];
    return data.models[mk].draft.mult_default;
  }
  function draftDefault(data, bk, mk) { return v(draftSrc(data, bk, mk)); }
  function draftCpuW(data, bk, mk) { const d = draftSrc(data, bk, mk); return d && d.beta ? v(d.beta) : 0; }
  // split GGUF host ms/token: the model's own row (models[mk].split_host_ms) wins over the box fit, which is fitted on one model's split anchor
  function splitSrc(data, bk, mk) {
    const own = data.models[mk] && data.models[mk].split_host_ms;
    if (own) return own;
    const c = data.cal && data.cal.boxes[bk];
    return (c && c.split_ms) || null;
  }
  function kappaSrc(data, bk, q) {
    const o = data.overrides && data.overrides.kappa && data.overrides.kappa[q];
    if (o !== undefined) return { v: o, tag: null, src: 'knob override (data.overrides.kappa)' };
    const bx = bk && data.boxes[bk] && data.boxes[bk].kappa;
    return bx && bx[q] !== undefined ? bx[q] : data.types[q].kappa;
  }
  function denseSrc(E, m, kn) {
    if (E.dense_gib) return E.dense_gib;
    const byType = m.dense_gib_by_type && kn.storeType && m.dense_gib_by_type[kn.storeType];
    return byType && (kn.densePrec || 'q8_0') === 'q8_0' ? byType : m.dense_gib;
  }
  function kvSrc(E, m, mk, ub) {
    if (E.kv_gib !== undefined) return E.kv_gib;
    const km = E.recipe_rates && E.recipe_rates.kv_model && E.recipe_rates.kv_model[mk];
    if (km) return ub === '512' ? km.ub512 : km.ub4096;
    return ub === '512' && m.kv_gib_ub512 ? m.kv_gib_ub512 : m.kv_gib;
  }

  // ---------- options and groups ----------
  function prepare(ctx, planName, scoreName) {
    const { data, mk, m } = ctx;
    const plan = heatSet(data, mk, planName), score = heatSet(data, mk, scoreName);
    const N = m.L * m.E;
    const order = Array.from({ length: N }, (_, i) => i).sort((a, b) => plan[b] - plan[a]);
    const groups = [];
    const scope = ctx.E.type_scope === 'global' ? 'global' : (ctx.kn.perExpertTypes ? 'expert' : 'layer');
    const perExpert = scope === 'expert';
    const mk2 = idx => { let H = 0, S = 0; idx.forEach(e => { H += plan[e]; S += score[e]; }); return { idx, H, S, n: idx.length }; };
    if (perExpert) {
      // one type per expert: groups of similar heat across all layers
      const G = Math.max(16, Math.round(N / 160));
      for (let i = 0; i < N; i += G) groups.push(mk2(order.slice(i, i + G)));
    } else {
      // GGUF today: one type per layer tensor, tier per expert -> heat groups inside each layer
      const per = Math.ceil(m.E / 8);
      for (let l = 0; l < m.L; l++) {
        const ids = Array.from({ length: m.E }, (_, e) => l * m.E + e).sort((a, b) => plan[b] - plan[a]);
        for (let i = 0; i < m.E; i += per) { const g = mk2(ids.slice(i, i + per)); g.layer = l; groups.push(g); }
      }
    }
    const opts = [];
    ctx.tiers.forEach((t, ti) => {
      if (t.kind === 'gpu' && !t.engineOk) return;
      ctx.types.forEach(q => {
        const tq = data.types[q];
        if (t.kind === 'gpu' && (tq.gpu === false || (t.drive && t.drive.types.indexOf(q) < 0))) return;
        const s = typeBytes(data, mk, q);
        const r = ctx.rate(t, q);
        // tw: the option's weight in the draft-on token time (CPU-side expert time carries the draft's verify weight, evaluate: Tw)
        opts.push({ ti, q, s, kld: typeKld(data, mk, q), inv: 1 / r, tw: t.kind === 'gpu' ? 1 : 1 + (ctx.cpuW || 0), copy: (t.kind === 'gpu' && ctx.copyTier) ? ctx.tiers.indexOf(ctx.copyTier) : -1 });
      });
    });
    const kL = m.k * m.L;
    const byLayer = [];
    if (scope === 'layer') groups.forEach((g, gi) => (byLayer[g.layer] = byLayer[g.layer] || []).push(gi));
    if (scope === 'global') byLayer.push(groups.map((g, gi) => gi));
    const optsByQ = {};
    opts.forEach((O, o) => (optsByQ[O.q] = optsByQ[O.q] || []).push(o));
    return Object.assign(ctx, { plan, score, order, groups, opts, kL, planName, scoreName, perExpert, scope, byLayer, optsByQ, qList: Object.keys(optsByQ) });
  }

  // choose per group given lambda and capacity prices mu (per byte)
  // base[g*nO+o] = heat-weighted quality + lambda * time for group g on option o (fixed per lambda);
  // the capacity price term n*s*mu is added in choose()
  // the quality term weighs each group's plan heat by its layers' sensitivity (layer map; uniform without one)
  function groupHW(ctx) {
    if (ctx.HW) return ctx.HW;
    const w = layerWeights(ctx.data, ctx.mk), Ex = ctx.m.E;
    return (ctx.HW = ctx.groups.map(G => w ? G.idx.reduce((a, e) => a + ctx.plan[e] * w[(e / Ex) | 0], 0) : G.H));
  }
  function baseCost(ctx, lam) {
    const { groups, opts, kL } = ctx, nO = opts.length, HW = groupHW(ctx);
    const b = new Float64Array(groups.length * nO);
    for (let g = 0; g < groups.length; g++) {
      const G = groups[g];
      for (let o = 0; o < nO; o++) { const O = opts[o]; b[g * nO + o] = HW[g] * O.kld / kL + G.H * lam * O.s * O.inv * O.tw; }
    }
    return b;
  }
  function choose(ctx, base, mu, out) {
    const { groups, opts } = ctx;
    const nT = ctx.tiers.length, nO = opts.length;
    const use = new Float64Array(nT);
    const price = new Float64Array(nO);
    for (let o = 0; o < nO; o++) { const O = opts[o]; price[o] = O.s * (mu[O.ti] + (O.copy >= 0 ? mu[O.copy] : 0)); }
    const add = (g, o) => { const O = opts[o], G = groups[g]; out[g] = o; use[O.ti] += G.n * O.s; if (O.copy >= 0) use[O.copy] += G.n * O.s; };
    if (!ctx.perExpert) {
      // one type per layer (or per model): pick the type whose best-tier costs summed over the scope's groups are lowest
      const qs = ctx.qList;
      for (const gis of ctx.byLayer) {
        let bc = Infinity, bsel = null, fsel = null;
        const sel = new Int32Array(gis.length);
        for (const q of qs) {
          const os = ctx.optsByQ[q];
          if (!os || !os.length) continue;
          let c = 0;
          for (let i = 0; i < gis.length; i++) {
            const g = gis[i], n = groups[g].n, row = g * nO;
            let b = -1, cb = Infinity;
            for (const o of os) {
              const x = base[row + o] + n * price[o];
              if (Number.isFinite(x) ? (b < 0 || !Number.isFinite(cb) || x < cb) : b < 0) { cb = x; b = o; }
            }
            c += cb; sel[i] = b;
            if (bsel !== null && !(c < bc)) break;
          }
          if (!fsel) fsel = Int32Array.from(sel);
          if (Number.isFinite(c) && c < bc) { bc = c; bsel = Int32Array.from(sel); }
        }
        const pick = bsel || fsel;
        if (!pick) throw new Error('choose: no valid quant/tier options for layer group');
        for (let i = 0; i < gis.length; i++) add(gis[i], pick[i]);
      }
      return use;
    }
    if (!nO) throw new Error('choose: no valid quant/tier options');
    for (let g = 0; g < groups.length; g++) {
      const n = groups[g].n, row = g * nO;
      let best = -1, bc = Infinity;
      for (let o = 0; o < nO; o++) {
        const c = base[row + o] + n * price[o];
        if (Number.isFinite(c) ? (best < 0 || !Number.isFinite(bc) || c < bc) : best < 0) { bc = c; best = o; }
      }
      add(g, best);
    }
    return use;
  }

  function solveLambda(ctx, lam, muWarm) {
    const nT = ctx.tiers.length;
    const mu = muWarm ? Float64Array.from(muWarm) : new Float64Array(nT);
    const out = new Int32Array(ctx.groups.length);
    const caps = ctx.tiers.map(t => t.cap);
    const base = baseCost(ctx, lam);
    let use = choose(ctx, base, mu, out);
    const ok = t => { use = choose(ctx, base, mu, out); return use[t] <= caps[t]; };
    for (let sweep = 0; sweep < 3; sweep++) {
      let changed = false;
      for (let t = 0; t < nT; t++) {
        if (use[t] <= caps[t] && mu[t] === 0) continue;
        const old = mu[t];
        // smallest mu_t with use_t <= cap_t, bracketed around the warm value
        mu[t] = 0;
        if (ok(t)) { if (old !== 0) changed = true; continue; }
        let lo, hi;
        if (old > 0) {
          mu[t] = old;
          if (ok(t)) { hi = old; lo = old / 2; mu[t] = lo; while (ok(t) && lo > 1e-30) { hi = lo; lo /= 2; mu[t] = lo; } }
          else { lo = old; hi = old * 2; mu[t] = hi; let i = 0; while (!ok(t) && i++ < 80) { lo = hi; hi *= 2; mu[t] = hi; } }
        } else {
          lo = 0; hi = 1e-12; mu[t] = hi; let i = 0; while (!ok(t) && i++ < 120) { lo = hi; hi *= 4; mu[t] = hi; }
        }
        for (let i = 0; i < 12; i++) { const mid = lo > 0 ? Math.sqrt(lo * hi) : 0.5 * hi; mu[t] = mid; if (ok(t)) hi = mid; else lo = mid; }
        mu[t] = hi;
        use = choose(ctx, base, mu, out);
        if (Math.abs(hi - old) > 1e-3 * Math.max(hi, old)) changed = true;
      }
      if (!changed) break;
    }
    // repair any residual overflow: push the coldest groups of an overfull tier to the cheapest feasible option
    for (let t = 0; t < nT; t++) {
      let guard = 0;
      while (use[t] > caps[t] + 1 && guard++ < ctx.groups.length) {
        let gi = -1;
        for (let g = ctx.groups.length - 1; g >= 0; g--) { const O = ctx.opts[out[g]]; if (O.ti === t || O.copy === t) { gi = g; break; } }
        if (gi < 0) break;
        const G = ctx.groups[gi];
        let best = -1, bc = Infinity;
        const curQ = ctx.opts[out[gi]].q;
        ctx.opts.forEach((O, o) => {
          if (O.ti === t || O.copy === t) return;
          if (!ctx.perExpert && O.q !== curQ) return;
          if (use[O.ti] + G.n * O.s > caps[O.ti]) return;
          if (O.copy >= 0 && use[O.copy] + G.n * O.s > caps[O.copy]) return;
          const c = groupHW(ctx)[gi] * O.kld / ctx.kL + G.H * lam * O.s * O.inv;
          if (c < bc) { bc = c; best = o; }
        });
        if (best < 0) break;
        const P = ctx.opts[out[gi]];
        use[P.ti] -= G.n * P.s; if (P.copy >= 0) use[P.copy] -= G.n * P.s;
        out[gi] = best;
        const O = ctx.opts[best];
        use[O.ti] += G.n * O.s; if (O.copy >= 0) use[O.copy] += G.n * O.s;
      }
    }
    const feasible = use.every((u, t) => u <= caps[t] + 1);
    return { out, mu, use, feasible };
  }

  // expand group choice to per-expert arrays
  function expand(ctx, sol) {
    const N = ctx.m.L * ctx.m.E;
    const tier = new Int8Array(N), type = new Int8Array(N);
    const qIndex = {}; ctx.types.forEach((q, i) => qIndex[q] = i);
    ctx.groups.forEach((G, g) => {
      const O = ctx.opts[sol.out[g]];
      G.idx.forEach(e => { tier[e] = O.ti; type[e] = qIndex[O.q]; });
    });
    return { tier, type };
  }

  // evaluate a per-expert layout with a heat array
  function evaluate(ctx, lay, heat, readHeat) {
    const { data, mk, tiers, types } = ctx;
    const h = heat || ctx.score;
    const nT = tiers.length;
    const N = ctx.m.L * ctx.m.E;
    const sz = types.map(q => typeBytes(data, mk, q));
    const kl = types.map(q => typeKld(data, mk, q));
    const inv = tiers.map(t => types.map(q => 1 / ctx.rate(t, q)));
    const split = ctx.hitR < 1 && ctx.missTi >= 0, r = ctx.hitR;
    const rawInv = split ? tiers.map(t => types.map(q => 1 / ctx.rawRate(t, q))) : null;
    const bytesTok = new Float64Array(nT), timeT = new Float64Array(nT), capUse = new Float64Array(nT), heatT = new Float64Array(nT), cnt = new Float64Array(nT), bitsW = new Float64Array(nT);
    const byType = tiers.map(() => types.map(() => 0));
    let kld = 0;
    const copyIdx = ctx.copyTier ? tiers.indexOf(ctx.copyTier) : -1;
    const w = layerWeights(data, mk), Ex = ctx.m.E;
    for (let e = 0; e < N; e++) {
      const t = lay.tier[e], q = lay.type[e];
      const b = (readHeat || h)[e] * sz[q];
      if (split && tiers[t].kind === 'gpu') { timeT[t] += r * b * rawInv[t][q]; timeT[ctx.missTi] += (1 - r) * b * rawInv[ctx.missTi][q]; bytesTok[t] += r * b; bytesTok[ctx.missTi] += (1 - r) * b; }
      else { bytesTok[t] += b; timeT[t] += b * inv[t][q]; }
      capUse[t] += sz[q]; heatT[t] += (readHeat || h)[e]; cnt[t]++;
      bitsW[t] += v(data.types[types[q]].bpw);
      byType[t][q] += sz[q];
      if (tiers[t].kind === 'gpu' && copyIdx >= 0) { capUse[copyIdx] += sz[q]; }
      kld += h[e] * kl[q] * (w ? w[(e / Ex) | 0] : 1);
    }
    kld = kld / ctx.kL + ctx.kldDense;
    let gpuT = ctx.denseTime, cpuT = 0;
    tiers.forEach((t, i) => { if (t.kind === 'gpu') gpuT += timeT[i]; else cpuT += timeT[i]; });
    const T = ctx.host + (ctx.overlap ? Math.max(gpuT, cpuT) + (1 - ctx.overlap) * Math.min(gpuT, cpuT) : gpuT + cpuT);
    const Tw = T + (ctx.cpuW || 0) * cpuT, tokps = ctx.mult / Tw;
    const quality = qualityOf(ctx.data, kld);
    return {
      T, Tw, tokps, mult: ctx.mult * T / Tw, kld, quality, gpuT, cpuT, host: ctx.host, dense: ctx.denseTime, bytesTok, timeT, capUse, heatT, cnt,
      avgBpw: Array.from(bitsW, (b, i) => cnt[i] ? b / cnt[i] : 0), byType, copyIdx
    };
  }

  // per-expert fill: promote into leftover faster capacity, then upgrade types while the target holds
  function fill(ctx, lay, targetTokps) {
    const { data, mk, tiers, types } = ctx;
    const N = ctx.m.L * ctx.m.E;
    const sz = types.map(q => typeBytes(data, mk, q));
    const kl = types.map(q => typeKld(data, mk, q));
    const inv = tiers.map(t => types.map(q => 1 / ctx.rate(t, q)));
    const ev = evaluate(ctx, lay);
    const capLeft = tiers.map((t, i) => t.cap - ev.capUse[i]);
    const copyIdx = ev.copyIdx;
    const order = ctx.order; // hot first by plan heat
    const lw = layerWeights(data, mk), wl = l => lw ? lw[l] : 1;
    const optOk = (t, q) => ctx.opts.some(O => O.ti === t && O.q === types[q]);
    // a tier's time in the draft-on token (CPU-side expert time carries the draft's verify weight, as in evaluate's Tw)
    const tw = t => tiers[t].kind === 'gpu' ? 1 : 1 + (ctx.cpuW || 0);
    // promote (same type) to faster tiers: never slower, only time drops
    for (let t = 0; t < tiers.length; t++) {
      for (const e of order) {
        if (capLeft[t] <= 0) break;
        const ct = lay.tier[e], q = lay.type[e];
        if (ct <= t) continue;
        if (!optOk(t, q)) continue;
        if (inv[t][q] >= inv[ct][q]) continue;
        const need = sz[q];
        const needCopy = (tiers[t].kind === 'gpu' && copyIdx >= 0 && tiers[ct].kind !== 'gpu') ? sz[q] : 0;
        if (need > capLeft[t]) continue;
        if (needCopy && capLeft[copyIdx] + (ct === copyIdx ? sz[q] : 0) < needCopy) continue;
        capLeft[t] -= need; capLeft[ct] += sz[q];
        if (needCopy) capLeft[copyIdx] -= needCopy;
        lay.tier[e] = t;
      }
    }
    // upgrade types in place, best quality gain per byte first, while score time stays within target
    if (targetTokps && !ctx.perExpert) {
      // whole-layer type upgrades (GGUF: one type per layer tensor), best quality gain per byte first
      const Tmax = ctx.mult / targetTokps;
      let cur = evaluate(ctx, lay).Tw;
      const glob = ctx.scope === 'global';
      const L = glob ? 1 : ctx.m.L, Ex = glob ? ctx.m.L * ctx.m.E : ctx.m.E;
      const cand = [];
      for (let l = 0; l < L; l++) {
        const q = lay.type[l * Ex];
        let H = 0; for (let e = 0; e < Ex; e++) H += ctx.plan[l * Ex + e] * wl(((l * Ex + e) / ctx.m.E) | 0);
        for (let q2 = 0; q2 < types.length; q2++) {
          if (kl[q2] >= kl[q] || sz[q2] <= sz[q]) continue;
          cand.push([H * (kl[q] - kl[q2]) / (Ex * (sz[q2] - sz[q])), l, q2]);
        }
      }
      cand.sort((a, b) => b[0] - a[0]);
      for (const [, l, q2] of cand) {
        const q = lay.type[l * Ex];
        if (kl[q2] >= kl[q] || sz[q2] <= sz[q]) continue;
        const dCap = tiers.map(() => 0);
        let dT = 0, ok = true;
        for (let e = l * Ex; e < (l + 1) * Ex; e++) {
          const t = lay.tier[e];
          if (!optOk(t, q2)) { ok = false; break; }
          dCap[t] += sz[q2] - sz[q];
          if (tiers[t].kind === 'gpu' && copyIdx >= 0) dCap[copyIdx] += sz[q2] - sz[q];
          dT += ctx.score[e] * (sz[q2] * inv[t][q2] - sz[q] * inv[t][q]) * tw(t);
        }
        if (!ok || cur + dT > Tmax || dCap.some((d, t) => d > capLeft[t])) continue;
        cur += dT; dCap.forEach((d, t) => capLeft[t] -= d);
        for (let e = l * Ex; e < (l + 1) * Ex; e++) lay.type[e] = q2;
      }
    }
    if (targetTokps && ctx.perExpert) {
      const Tmax = ctx.mult / targetTokps;
      let cur = evaluate(ctx, lay).Tw;
      const cand = [];
      for (const e of order) {
        const t = lay.tier[e], q = lay.type[e];
        let best = -1, bg = 0;
        for (let q2 = 0; q2 < types.length; q2++) {
          if (kl[q2] >= kl[q] || sz[q2] <= sz[q] || !optOk(t, q2)) continue;
          const g = ctx.plan[e] * wl((e / ctx.m.E) | 0) * (kl[q] - kl[q2]) / (sz[q2] - sz[q]);
          if (g > bg) { bg = g; best = q2; }
        }
        if (best >= 0) cand.push([bg, e, best]);
      }
      cand.sort((a, b) => b[0] - a[0]);
      for (const [, e, q2] of cand) {
        const t = lay.tier[e], q = lay.type[e];
        const dS = sz[q2] - sz[q];
        if (dS > capLeft[t]) continue;
        const dCopy = (tiers[t].kind === 'gpu' && copyIdx >= 0) ? dS : 0;
        if (dCopy && dCopy > capLeft[copyIdx]) continue;
        const dT = ctx.score[e] * (sz[q2] * inv[t][q2] - sz[q] * inv[t][q]) * tw(t);
        if (cur + dT > Tmax) continue;
        cur += dT; capLeft[t] -= dS; if (dCopy) capLeft[copyIdx] -= dCopy;
        lay.type[e] = q2;
      }
    }
    return lay;
  }

  // ---------- hard checks (measured 10-17) ----------
  const isLow = (data, q) => { const t = data.types[q]; return t.lowbit !== undefined ? !!t.lowbit : v(t.bpw) < 3; };
  // share of routed reads served by 2-bit experts (heat-weighted over all k x L picks); with w (per planner layer, mean 1:
  // the layer map) the sensitivity-weighted share the map's quality line reads
  function lowShare(ctx, lay, heat, w) {
    const h = heat || ctx.score, Ex = ctx.m.E; let s = 0;
    for (let e = 0; e < h.length; e++) if (isLow(ctx.data, ctx.types[lay.type[e]])) s += h[e] * (w ? w[(e / Ex) | 0] : 1);
    return s / ctx.kL;
  }

  // ---------- layer map: per-layer sensitivity (data.layer_maps; calibrated here on our measured KL points) ----------
  // planner layers of a model-id range [lo, hi] (models.<m>.layer_ids maps planner layer -> model layer id)
  const layerIds = m => m.layer_ids || Array.from({ length: m.L }, (_, l) => l);
  const inRange = (m, l, x) => { const id = layerIds(m)[l]; return id >= x[0] && id <= x[1]; };
  // per planner layer, the share of that layer's routed reads at 2 bits for a type layout (h per layer sums to k)
  function layerDamage(ctx, type, h) {
    const m = ctx.m, d = new Float64Array(m.L);
    for (let e = 0; e < m.L * m.E; e++) if (isLow(ctx.data, ctx.types[type[e]])) d[(e / m.E) | 0] += h[e];
    for (let l = 0; l < m.L; l++) d[l] /= m.k;
    return d;
  }
  // a recipe's per-layer 2-bit damage: the store layout (storeTypes) on the score heat, rescaled to the recipe's own
  // lowbit_share when it states one (the recipe's share is what the gate reads; the layout gives its shape over layers)
  function recipeDamage(data, r) {
    const st = r.store || {};
    const named = (st.types || []).concat(Object.keys(st.read_share_by_type || {}), (st.layer_types || []).map(x => x.type), st.hot_type || [], st.cold_type || []);
    const m = data.models[r.model];
    if (!named.some(q => data.types[q] && isLow(data, q))) return { d: new Float64Array(m.L), how: 'nothing at 2 bits' };
    const bad = ['plan', 'score'].map(w => (r.heat || {})[w]).filter(x => x && !m.heat.sets[x]);
    if (bad.length) return { d: null, why: 'heat set ' + bad.join(', ') + ' is not in ' + m.short };
    const rc = data.engines[r.engine] && recipeContext(data, r);
    if (!rc) return { d: null, why: r.engine + ' has no planner model to lay out the types' };
    const ty = storeTypes(data, r, rc.ctx);
    if (ty.why) return { d: null, why: ty.why };
    const d = layerDamage(rc.ctx, ty.type, rc.ctx.score), sh = d.reduce((a, x) => a + x, 0) / m.L;
    const want = r.lowbit_share !== undefined && r.lowbit_share !== null ? r.lowbit_share : sh;
    const f = sh > 0 ? want / sh : 0;
    return { d: d.map(x => Math.min(1, x * f)), how: ty.note + (Math.abs(f - 1) > 0.02 ? ' (rescaled to the recipe lowbit_share ' + want + ')' : '') };
  }
  // small dense least squares (normal equations, Gaussian elimination); null when singular
  function lsq(X, y) {
    const n = X[0].length, A = Array.from({ length: n }, (_, i) => Array.from({ length: n + 1 }, (_, j) => j < n ? X.reduce((a, r) => a + r[i] * r[j], 0) : X.reduce((a, r, k) => a + r[i] * y[k], 0)));
    for (let i = 0; i < n; i++) {
      const p = A.reduce((b, r, k) => k >= i && Math.abs(r[i]) > Math.abs(A[b][i]) ? k : b, i);
      [A[i], A[p]] = [A[p], A[i]];
      if (Math.abs(A[i][i]) < 1e-15) return null;
      for (let k = 0; k < n; k++) if (k !== i) { const c = A[k][i] / A[i][i]; for (let j = i; j <= n; j++) A[k][j] -= c * A[i][j]; }
    }
    return A.map((r, i) => r[n] / r[i]);
  }
  // The map of one model, resolved per planner layer and calibrated: claimed band values (spread evenly over the band's layers)
  // x alpha, and each fit band's own coefficient, by least squares on the quality family's measured points (KL vs the gate
  // reference - base = sum over layers of coefficient x 2-bit damage). c = KL per fully 2-bit layer (gate units); w = c / mean
  // (the solver's and the gate's layer weights); s = the band's value in the map's unit (fit bands: c / alpha). Layers in no band
  // are UNKNOWN (w = the largest known weight, so an unknown layer is never cheap). status: calibrated | UNKNOWN (with why).
  function layerMap(data, mk) {
    data._lm = data._lm || {};
    if (mk in data._lm) return data._lm[mk];
    const M = data.layer_maps && data.layer_maps.models && data.layer_maps.models[mk], m = data.models[mk];
    if (!M || !m) return (data._lm[mk] = null);
    data._lm[mk] = null;  // a recipe layout below must not read a half-built map
    const L = m.L, k = m.k, Q = data.checks.quality, fam = Q.families[m.kld_family] || {};
    const band = Array.from({ length: L }, (_, l) => M.bands.findIndex(b => inRange(m, l, b.layers)));
    const nIn = M.bands.map((_, bi) => band.filter(x => x === bi).length);
    const claim = band.map(bi => bi < 0 || M.bands[bi].fit || v(M.bands[bi].kl) === null || v(M.bands[bi].kl) === undefined ? null : v(M.bands[bi].kl) / nIn[bi]);
    const fitB = M.bands.map((b, bi) => b.fit ? bi : -1).filter(x => x >= 0), hasClaim = claim.some(x => x !== null);
    const out = { model: mk, unit: M.unit, L, band, layer_ids: layerIds(m), status: 'UNKNOWN', why: null, points: [], claims: [], dof: null,
      prov: band.map(bi => bi < 0 ? 'UNKNOWN' : M.bands[bi].tag), base_kl: fam.base_kl ? v(fam.base_kl) : null, base_top1: fam.base_top1 };
    const pts = (fam.points || []).filter(p => p.run).map(p => {
      const rec = (data.recipes || []).find(x => x.run_id === p.run), dm = rec ? recipeDamage(data, rec.recipe) : { d: null, why: 'run ' + p.run + ' not in the ledger' };
      return { run: p.run, share: p.share, kl: p.kl, top1: p.top1, d: dm.d, why: dm.why };
    });
    const bad = pts.filter(p => !p.d);
    const cols = (hasClaim ? 1 : 0) + fitB.length;
    if (out.base_kl === null || bad.length || pts.length < cols) {
      out.why = out.base_kl === null ? 'the quality family has no base KL' : bad.length ? bad.map(p => p.run + ': ' + p.why).join('; ') : pts.length + ' measured points for ' + cols + ' unknowns';
      return (data._lm[mk] = out);
    }
    const row = d => (hasClaim ? [band.reduce((a, bi, l) => a + (claim[l] !== null ? d[l] * claim[l] : 0), 0)] : []).concat(fitB.map(bi => band.reduce((a, b, l) => a + (b === bi ? d[l] : 0), 0)));
    const sol = lsq(pts.map(p => row(p.d)), pts.map(p => p.kl - out.base_kl));
    if (!sol || sol.some(x => !(x > 0))) { out.why = 'no physical fit (a coefficient <= 0 or singular): ' + (sol ? sol.map(x => x.toPrecision(3)).join(', ') : 'singular'); return (data._lm[mk] = out); }
    const alpha = hasClaim ? sol[0] : null, beta = {};
    fitB.forEach((bi, i) => beta[bi] = sol[(hasClaim ? 1 : 0) + i]);
    const c = band.map((bi, l) => claim[l] !== null ? alpha * claim[l] : bi >= 0 && beta[bi] !== undefined ? beta[bi] : null);
    const known = c.filter(x => x !== null), mean = known.reduce((a, x) => a + x, 0) / known.length, wmax = Math.max.apply(null, known) / mean;
    const w = Float64Array.from(c, x => x === null ? wmax : x / mean);
    const eff = d => d.reduce((a, x, l) => a + x * w[l], 0) / L;
    pts.forEach(p => p.eff = eff(p.d));
    const sxx = pts.reduce((a, p) => a + p.eff * p.eff, 0);
    const slope_kl = pts.reduce((a, p) => a + p.eff * (p.kl - out.base_kl), 0) / sxx, slope_top1 = pts.reduce((a, p) => a + p.eff * (out.base_top1 - p.top1), 0) / sxx;
    const mk2 = (Q.bar_kl - out.base_kl) / slope_kl, mt = (out.base_top1 - Q.bar_top1) / slope_top1;
    pts.forEach(p => { p.pred_kl = out.base_kl + slope_kl * p.eff; p.pred_top1 = out.base_top1 - slope_top1 * p.eff; delete p.d; });
    const s = band.map((bi, l) => claim[l] !== null ? claim[l] : c[l] !== null && alpha ? c[l] / alpha : null);
    Object.assign(out, { status: 'calibrated', alpha, beta: fitB.map(bi => ({ layers: M.bands[bi].layers, kl_per_layer: beta[bi], unit_value: alpha ? beta[bi] * nIn[bi] / alpha : null })),
      c, w: Array.from(w), s, dof: pts.length - cols, points: pts,
      residual_max: { kl: Math.max.apply(null, pts.map(p => Math.abs(p.pred_kl - p.kl))), top1: Math.max.apply(null, pts.map(p => Math.abs(p.pred_top1 - p.top1))) },
      line: { base_kl: out.base_kl, base_top1: out.base_top1, slope_kl, slope_top1, max_eff: Math.min(mk2, mt, 1), binds: mk2 <= mt ? 'KL' : 'top-1', points: pts } });
    data._lm[mk] = out;
    // published whole-recipe claims, predicted from the map in its own unit (the claimant's own miss share and KL scale)
    const T = M.trim;
    out.claims = (M.claims || []).map(cl => {
      const keep = trimKeeps(m, cl.trim), miss = new Float64Array(L).fill(T ? v(T.ref_miss_share) : 1);
      const r = trimKl(data, mk, keep, miss, { unit: true });
      return { what: cl.what, claimed_kl: cl.kl, pred_kl: r.kl, why: r.why, tag: cl.tag, url: cl.url };
    });
    return (data._lm[mk] = out);
  }
  // per planner layer weights of a calibrated map (mean 1), else null (uniform)
  function layerWeights(data, mk) { const lm = layerMap(data, mk); return lm && lm.status === 'calibrated' ? lm.w : null; }
  // keep per planner layer from trim bands [{layers: [lo, hi] model ids, keep}] (k = no trim)
  function trimKeeps(m, trim) {
    const keep = new Int16Array(m.L).fill(m.k);
    (trim || []).forEach(t => { for (let l = 0; l < m.L; l++) if (inRange(m, l, t.layers)) keep[l] = Math.min(keep[l], t.keep); });
    return keep;
  }
  // KL of dropping ranks keep..k-1 of the picks that miss VRAM: sum over layers of s (map unit, ref_keep trim) x (k - keep) /
  // (k - ref_keep) x miss / ref_miss x transfer (gate units). opts.unit: in the map's own unit (no miss ratio, no transfer).
  function trimKl(data, mk, keep, miss, opts) {
    const lm = layerMap(data, mk), M = data.layer_maps.models[mk], T = M && M.trim, m = data.models[mk];
    if (!T) return { kl: null, why: 'no trim unit in the layer map of ' + mk };
    if (!lm || lm.status !== 'calibrated') return { kl: null, why: 'layer map ' + (lm ? lm.why : 'missing') };
    const unit = opts && opts.unit, rm = v(T.ref_miss_share), tr = unit ? 1 : v(T.transfer);
    let kl = 0;
    const unk = [];
    for (let l = 0; l < m.L; l++) {
      if (keep[l] >= m.k) continue;
      if (lm.s[l] === null) { unk.push(lm.layer_ids[l]); continue; }
      kl += lm.s[l] * (m.k - keep[l]) / (m.k - T.ref_keep) * (unit ? 1 : miss[l] / rm) * tr;
    }
    return unk.length ? { kl: null, why: 'trimmed layers ' + unk.join(',') + ' have no map value (UNKNOWN)' } : { kl, why: null };
  }
  // does the engine drop routed picks per layer (layer_maps.models.<m>.trim.engines.<engine>: yes | needs_code)
  function trimSupport(data, r) {
    const T = ((data.layer_maps || {}).models || {})[r.model], x = T && T.trim && T.trim.engines[r.engine];
    return x || { status: 'no', src: 'no route-trim path for ' + r.model + ' on ' + r.engine + ' (layer_maps.models.' + r.model + '.trim.engines)' };
  }
  // the map's quality prediction for a layout: KL = base + slope x weighted 2-bit share + trim KL; top-1 on the same line
  function mapQuality(data, mk, eff, trim) {
    const lm = layerMap(data, mk);
    if (!lm || lm.status !== 'calibrated') return null;
    // a route trim's top-1 cost per unit KL is its own measured rate (layer_maps.models.<m>.trim.top1_per_kl) when the map
    // has one: a trim drops whole picks, which flips top-1 faster per KL than low-bit error does (lab notes); else the 2-bit line
    const L = lm.line, T = ((data.layer_maps || {}).models || {})[mk], tp = T && T.trim ? v(T.trim.top1_per_kl) : null;
    const x = eff + (tp ? 0 : (trim || 0) / L.slope_kl);
    return { kl: L.base_kl + L.slope_kl * eff + (trim || 0), top1: L.base_top1 - L.slope_top1 * x - (tp ? tp * (trim || 0) : 0), eff, trim: trim || 0 };
  }
  function gateOf(ctx) {
    const Q = ctx.data.checks && ctx.data.checks.quality;
    const f = Q && Q.families[ctx.m.kld_family];
    if (!f) return null;
    return Object.assign({ enforced: (f.engines || []).indexOf(ctx.eng) >= 0 && !!f.max_share }, f);
  }
  // enforce the quality gate: upgrade the hottest 2-bit units (expert / layer / store) to the cheapest non-2-bit type until
  // the 2-bit share of routed reads is within max_share (plan and score heat both)
  function gateRepair(ctx, lay) {
    const g = gateOf(ctx);
    if (!g || !g.enforced) return lay;
    // with a calibrated layer map the gate reads the sensitivity-weighted 2-bit share and its own line's max
    const lm = layerMap(ctx.data, ctx.mk), w = lm && lm.status === 'calibrated' ? lm.w : null, wE = e => w ? w[(e / ctx.m.E) | 0] : 1;
    const max = w ? lm.line.max_eff : v(g.max_share);
    const { data, mk, tiers, types } = ctx;
    const sz = types.map(q => typeBytes(data, mk, q));
    const hi = types.map((q, i) => i).filter(i => !isLow(data, types[i])).sort((a, b) => sz[a] - sz[b]);
    if (!hi.length) return lay;
    const optOk = (t, qi) => ctx.opts.some(O => O.ti === t && O.q === types[qi]);
    const ev = evaluate(ctx, lay);
    const capLeft = tiers.map((t, i) => t.cap - ev.capUse[i]);
    let sP = lowShare(ctx, lay, ctx.plan, w), sS = lowShare(ctx, lay, ctx.score, w);
    const share = () => Math.max(sP, sS);
    const done = e => { sP -= ctx.plan[e] * wE(e) / ctx.kL; sS -= ctx.score[e] * wE(e) / ctx.kL; return true; };
    const up = e => {
      const t = lay.tier[e], q = lay.type[e];
      for (const q2 of hi) {
        const d = sz[q2] - sz[q];
        if (optOk(t, q2) && capLeft[t] >= d) { capLeft[t] -= d; lay.type[e] = q2; return done(e); }
        for (let t2 = t + 1; t2 < tiers.length; t2++) {
          if (tiers[t2].kind === 'gpu' || !optOk(t2, q2) || capLeft[t2] < sz[q2]) continue;
          capLeft[t] += sz[q]; capLeft[t2] -= sz[q2]; lay.tier[e] = t2; lay.type[e] = q2; return done(e);
        }
      }
      return false;
    };
    const N = ctx.m.L * ctx.m.E, Ex = ctx.m.E;
    let guard = 0;
    while (share() > max + 1e-9 && guard++ < N) {
      if (ctx.scope === 'expert') {
        // hottest 2-bit experts first, in batches
        // hottest 2-bit experts first, one at a time until the share fits
        const ord = w ? ctx.order.slice().sort((x, y) => ctx.plan[y] * wE(y) - ctx.plan[x] * wE(x)) : ctx.order;
        for (const e of ord) {
          if (share() <= max + 1e-9) break;
          if (isLow(data, types[lay.type[e]])) up(e);
        }
        break;
      } else {
        // whole layer (or the whole store) at once: the 2-bit layer with the most routed reads
        const units = ctx.scope === 'global' ? [[0, N]] : Array.from({ length: ctx.m.L }, (_, l) => [l * Ex, (l + 1) * Ex]);
        let best = null, bh = -1;
        units.forEach(([a, b]) => { let h = 0; for (let e = a; e < b; e++) if (isLow(data, types[lay.type[e]])) h += (ctx.plan[e] + ctx.score[e]) * wE(e); if (h > bh) { bh = h; best = [a, b]; } });
        if (!best || bh <= 0) break;
        let moved = 0;
        for (let e = best[0]; e < best[1]; e++) if (isLow(data, types[lay.type[e]]) && up(e)) moved++;
        if (!moved) break;
      }
    }
    return lay;
  }

  // facts of a solver layout, in the same shape as a data.recipes row
  function layoutFacts(ctx, lay, ev) {
    const gi = ctx.tiers.indexOf(ctx.g0);
    const g0 = ctx.g0;
    let post = g0.reserves.reduce((a, r) => a + r.gib, 0);
    ctx.fixed.filter(x => x.tier === g0.id && x.kind !== 'margin').forEach(x => post += x.gib);
    post += ev.capUse[gi] / GiB;
    if (ctx.E.kv_includes_margin) post -= ctx.E.kv_includes_margin;
    const storage = {};
    ctx.fixed.filter(x => x.kind === 'lookup' && x.on_file).forEach(x => { storage[x.cls || x.label] = storageKind(ctx.box, x.tier); });
    let rtr = false;
    for (let e = 0; e < lay.type.length; e++) if (ctx.data.types[ctx.types[lay.type[e]]].rtr) { rtr = true; break; }
    const di = ctx.tiers.findIndex(t => t.id === 'dram');
    const numa = di >= 0 ? { dram_experts_gib: ev.capUse[di] / GiB } : null;
    const sy = ctx.data.checks.system && ctx.data.checks.system.boxes[ctx.bk];  // a plan assumes the box's P0 settings are held
    const w = layerWeights(ctx.data, ctx.mk);
    return { box: ctx.bk, model: ctx.mk, engine: ctx.eng, split: ctx.split || ctx.scope === 'expert', rtr, gpu_total: g0.total / GiB, post_gib: post, storage, lowbit_share: lowShare(ctx, lay),
      lowbit_eff: w ? lowShare(ctx, lay, null, w) : null, trim_kl: 0, numa, system: sy ? sy.required : null };
  }

  // a dram-fill headroom: one figure for every node or a per-node list
  function headroomAt(h, i) { return Array.isArray(h) ? h[Math.min(i, h.length - 1)] : h; }
  // predicted MemFree per NUMA node: free at plan time - planned experts on the node - load extra (non-expert interleave + host buffers + page cache)
  // f.numa: { headroom_gib } (dram-fill: same headroom on every node) | { nodes: [{ experts_gib, avail_gib? }] } | { dram_experts_gib } (solver layout)
  function numaCheck(data, f) {
    const N = data.checks.numa_free, box = data.boxes[f.box], label = N && N.label;
    if (!N) return { id: 'numa_free', label: 'DRAM headroom per NUMA node', status: 'UNKNOWN', detail: 'no numa_free check in the data' };
    const nodes = (box && box.dram.nodes) || [], bar = v(N.min_free_gib);
    // load extra keyed by load style: a one-chain GGUF ('<box>|<model>|<engine>|chain'); a split load by hot-bank size
    // ('...|banks<=X', the smallest X that covers f.banks_gib); then '<box>|<model>|<engine>', then the engine
    const LE = N.load_extra, base = f.box + '|' + f.model + '|' + f.engine;
    const bk = !f.split || f.banks_gib === undefined || f.banks_gib === null ? null : Object.keys(LE).filter(k => k.startsWith(base + '|banks<=')).map(k => [k, +k.split('<=')[1]]).filter(x => f.banks_gib <= x[1] + 0.005).sort((a, b) => a[1] - b[1])[0];
    const key = (!f.split && LE[base + '|chain'] && base + '|chain') || (bk && bk[0]) || (LE[base] && base) || f.engine;
    const le = LE[key];
    const nf = f.numa || {};
    if (nodes.length <= 1 && !nf.headroom_gib && !nf.nodes) return { id: 'numa_free', label, status: 'PASS', detail: 'one NUMA node: covered by the DRAM usable cap' };
    if (!le || v(le) === null) return { id: 'numa_free', label, status: 'UNKNOWN', detail: 'no load-extra figure for ' + f.engine + ' (' + ((le && le.src) || 'none') + ')' };
    const hi = v(le), lo = le.lo !== undefined ? le.lo : hi, est = le.tag !== 'MEASURED';
    // the extra lands on node 0; other DRAM nodes get other_nodes (MEASURED at or below plan on every placed run)
    const other = N.other_nodes ? v(N.other_nodes) : null;
    const ex = (x, i) => (i === 0 || other === null) ? x : other;
    const avail = nodes.map(x => x.avail_gib ? v(x.avail_gib) : null);
    let freeAt = null, how = '';
    if (nf.headroom_gib !== undefined && nf.headroom_gib !== null) {
      freeAt = x => (nodes.length ? nodes : [0]).map((_, i) => headroomAt(nf.headroom_gib, i) - ex(x, i));
      how = 'dram-fill headroom ' + (Array.isArray(nf.headroom_gib) ? nf.headroom_gib.map((h, i) => 'N' + i + ' ' + h).join(' / ') : nf.headroom_gib) + ' GiB';
    } else if (nf.nodes) {
      freeAt = x => nf.nodes.map((y, i) => { const a = y.avail_gib !== undefined ? y.avail_gib : avail[i]; return a === null || a === undefined ? null : a - y.experts_gib - ex(x, i); });
      how = 'node free at plan time - planned experts';
    } else if (nf.dram_experts_gib !== undefined && avail.length && avail.every(x => x !== null)) {
      // dram-fill fills every node to the same headroom
      const h = (avail.reduce((a, x) => a + x, 0) - nf.dram_experts_gib) / avail.length;
      freeAt = x => avail.map((_, i) => h - ex(x, i));
      how = nf.dram_experts_gib.toFixed(1) + ' GiB experts in DRAM, dram-fill at equal headroom ' + h.toFixed(1) + ' GiB';
    }
    const free = freeAt && freeAt(hi), freeLo = freeAt && freeAt(lo);
    if (!free || free.some(x => x === null)) return { id: 'numa_free', label, status: 'UNKNOWN', detail: 'no per-node figures: give numa.headroom_gib or numa.nodes[].experts_gib' + (avail.some(x => x === null) ? ' + avail_gib (box table has none)' : '') };
    const mn = a => Math.min.apply(null, a);
    const band = lo !== hi;
    how += ' - load extra ' + (band ? lo + '-' + hi : hi) + ' on node 0' + (other !== null && free.length > 1 ? ', ' + other + ' on the other nodes' : '') + ' [' + key + ']';
    if (nf.spill_heal && free.length > 1 && N.spill_heal && nf.observed_memfree_gib === undefined && !nf.observed_clean) return healCheck(N, key, le, free, freeLo, free.reduce((a, x, i) => a + x + ex(hi, i), 0), hi, how, bar, label);
    const pred = { memfree_gib: free, load_extra_gib: hi, load_extra_tag: le.tag, load_extra: le };
    let st = mn(freeLo) < bar ? 'FAIL' : mn(free) < bar ? 'UNKNOWN' : (est && mn(free) < bar + 3) ? 'UNKNOWN' : 'PASS';
    // an observation of this run beats the prediction
    if (nf.observed_memfree_gib !== undefined) { st = nf.observed_memfree_gib < bar ? 'FAIL' : 'PASS'; how = 'OBSERVED node MemFree ' + nf.observed_memfree_gib + ' GiB; predicted from ' + how; }
    else if (nf.observed_clean) { st = 'PASS'; return { id: 'numa_free', label, status: st, pred, detail: 'OBSERVED clean: ' + nf.observed_clean + '. The model predicts ' + free.map((x, i) => 'N' + i + ' ' + x.toFixed(1)).join(' / ') + ' GiB at load extra ' + hi + ' (' + key + '): the extra is config-dependent (B0 nortr +6.0, B1b +9.5), so the check is conservative here' }; }
    let d = how + ' (' + le.tag + ') -> predicted MemFree ' + free.map((x, i) => 'N' + i + ' ' + (freeLo[i] !== x ? freeLo[i].toFixed(1) + '..' : '') + x.toFixed(1)).join(' / ') + ' GiB (bar ' + bar + ')';
    if (st === 'FAIL') d += ': node 0 runs out; its runtime pages fall back to PMem (b1017 at headroom 10: MemFree 0.59, 1.14 GiB on N2, prefill 8-10 vs 46-51 healed). Set numa.spill_heal (SPILL_HEAL=1) or give node 0 more headroom';
    else if (mn(free) < bar) d += ': inside the measured load-extra band the low end clears the bar and the high end does not';
    else if (st === 'UNKNOWN') d += ': load extra is an ESTIMATE and the margin is under 3 GiB';
    const known = (N.headrooms || []).filter(x => nf.headroom_gib === x.gib);
    if (known.length) d += '. Headroom ' + known[0].gib + ': ' + known[0].verdict;
    return { id: 'numa_free', label, status: st, pred, detail: d };
  }
  // numa_free with SPILL_HEAL: runtime pages that overflow node 0 land on PMem and spill-heal moves them to the DRAM node with
  // the most MemFree, so the bar applies to DRAM free summed over the nodes; every node keeps the kernel floor (PMem is
  // ZONE_MOVABLE: unmovable kernel allocations need DRAM, b1017pm OOM). A measured summed load extra for the key beats
  // node 0's extra + other_nodes.
  function healCheck(N, key, le, free, freeLo, hSum, hi, how, bar, label) {
    const H = N.spill_heal, floor = v(H.kernel_floor_gib), se = H.sum_extra && H.sum_extra[key];
    const sum = a => a.reduce((x, y) => x + y, 0), mn = a => Math.min.apply(null, a);
    const healed = f => {   // overflow below the floor moves to the node with the most free
      const g = f.slice(); let o = 0;
      g.forEach((x, i) => { if (x < floor) { o += floor - x; g[i] = floor; } });
      const j = g.indexOf(Math.max.apply(null, g)); g[j] -= o;
      return g;
    };
    const tot = se ? v(se) : null, totHi = se ? hSum - tot : sum(free), totLo = se ? hSum - (se.lo !== undefined ? se.lo : tot) : sum(freeLo);
    const post = healed(free), postLo = healed(freeLo), est = le.tag !== 'MEASURED' || (se && se.tag !== 'MEASURED');
    const st = (totLo < bar || mn(postLo) < floor) ? 'FAIL' : (totHi < bar || mn(post) < floor) ? 'UNKNOWN' : (est && totHi < bar + 3) ? 'UNKNOWN' : 'PASS';
    let d = how + (se ? '; summed load extra ' + (se.lo !== undefined ? se.lo + '-' : '') + tot + ' (' + se.tag + ')' : '') + ' (' + le.tag + '), SPILL_HEAL on -> DRAM free summed ' +
      (totLo !== totHi ? totLo.toFixed(1) + '..' : '') + totHi.toFixed(1) + ' GiB (bar ' + bar + '), after the heal ' + post.map((x, i) => 'N' + i + ' ' + x.toFixed(1)).join(' / ') + ' GiB (kernel floor ' + floor + ')';
    if (st === 'FAIL') d += ': the DRAM nodes together cannot hold the runtime pages, or a node drops under the kernel floor (b1017pm: global OOM)';
    else if (st === 'UNKNOWN') d += ': inside the load-extra band, or an ESTIMATE with a margin under 3 GiB';
    return { id: 'numa_free', label, status: st, pred: { memfree_gib: post, dram_free_sum_gib: totHi, load_extra_gib: hi, load_extra_tag: le.tag, load_extra: le, spill_heal: true }, detail: d };
  }
  // evaluate the hard checks on facts (a recipe row or layoutFacts); status PASS / FAIL / UNKNOWN
  // a DRAM-first recipe (numa.headroom_gib) declares numa.pinned_host false (pageable expert host buffers) and numa.verify "fail"
  // (the kit checks the placement per node after the load and stops the arm off target); pinned or verify off FAILs, undeclared
  // is UNKNOWN; "warn" records without stopping (UNKNOWN)
  function placeCheck(data, f) {
    const C = data.checks.placement, nf = f.numa || {};
    if (nf.headroom_gib === undefined || nf.headroom_gib === null) return { id: 'placement', label: C.label, status: 'PASS', detail: 'no per-node expert placement' };
    const bad = [], unk = [];
    if (nf.pinned_host === true) bad.push('numa.pinned_host true: ' + C.pinned.v);
    else if (nf.pinned_host !== false) unk.push('numa.pinned_host not given (' + C.pinned.v + ')');
    if (nf.verify === 'off') bad.push('numa.verify off: ' + C.verify.v);
    else if (nf.verify !== 'fail') unk.push('numa.verify ' + (nf.verify ? '"' + nf.verify + '" records only' : 'not given') + ' (' + C.verify.v + ')');
    const il = (nf.layout || 'interleave') === 'interleave' ? '; layout interleave: ' + C.interleave.v : '';
    return { id: 'placement', label: C.label, status: bad.length ? 'FAIL' : unk.length ? 'UNKNOWN' : 'PASS',
      detail: (bad.concat(unk).join('; ') || 'pageable expert host buffers; the kit verifies per node and stops off target') + il };
  }
  function runChecks(data, f) {
    const C = data.checks, out = [];
    // (a) storage tier per tensor class
    {
      const S = C.storage, cls = Object.keys(f.storage || {});
      const bad = cls.filter(c => S.forbidden.indexOf(f.storage[c]) >= 0), unk = cls.filter(c => f.storage[c] === 'unknown');
      out.push({ id: 'storage', label: S.label, status: bad.length ? 'FAIL' : unk.length ? 'UNKNOWN' : 'PASS',
        detail: cls.length ? cls.map(c => c + ' on ' + f.storage[c]).join(', ') + (bad.length ? ': on-file tensors read every token on a rotational disk (B0b: prefill 3.51 / decode 3.05)' : '') : 'no on-file tensor class' });
    }
    // (b) VRAM peak = post-warm-up + transient margin
    {
      const mg = marginOf(data, f.engine, f.split);
      let st = 'UNKNOWN', d = '';
      if (mg && mg.self_sizing) {
        // the engine fills VRAM itself: only a measured peak under a long prompt can be judged
        const T = f.gpu_total;
        if (f.peak_gib !== undefined && f.peak_gib !== null) { st = f.peak_gib > T + 0.005 ? 'FAIL' : 'PASS'; d = 'measured peak ' + f.peak_gib.toFixed(2) + ' of ' + T.toFixed(2) + ' GiB (' + (f.peak_src || 'recipe') + ')'; }
        else d = f.engine + ' self-sizes its VRAM cache, so the margin is MEASURED-UNKNOWN; give vram.peak_gib. Settles with ' + mg.pending;
      }
      else if (!mg || f.post_gib === undefined || f.post_gib === null) d = 'no margin or post-warm-up figure';
      else {
        const lo = f.post_gib + mg.lo, hi = f.post_gib + mg.hi, T = f.gpu_total;
        const tol = 0.005;  // readings carry 2 decimals; a 0.005 GiB tie is a fit
        st = lo >= T + tol ? 'FAIL' : hi <= T + tol ? 'PASS' : 'UNKNOWN';
        d = 'post-warm-up ' + f.post_gib.toFixed(2) + ' + margin ' + mg.lo + '-' + mg.hi + ' = ' + lo.toFixed(2) + '-' + hi.toFixed(2) + ' of ' + T.toFixed(2) + ' GiB';
        if (mg.candidate !== null && mg.candidate !== undefined) {
          const c = f.post_gib + mg.candidate;
          d += '; with the B0 #1 graph growth ' + mg.candidate + ' GiB: ' + c.toFixed(2) + (c > T ? ' -> OOM expected' : ' -> fits (' + (T - c).toFixed(2) + ' GiB spare)');
        }
        if (st === 'UNKNOWN' && mg.pending) d += '. Settles with ' + mg.pending;
      }
      out.push({ id: 'vram_peak', label: C.vram_peak.label, status: st, detail: d });
    }
    // (e) DRAM headroom per NUMA node
    out.push(numaCheck(data, f));
    // (e2) the per-expert placement lands where the plan put it
    out.push(placeCheck(data, f));
    // (f) box P0 system settings the run must hold (kernel tiering, swap)
    {
      const Y = C.system, req = Y && Y.boxes[f.box];
      if (req) {
        // dram_fill_only keys: a wrong value FAILs only with a dram-fill headroom (the harm was measured under node pressure only)
        const got = f.system || {}, keys = Object.keys(req.required), fill = !!(f.numa && f.numa.headroom_gib !== undefined && f.numa.headroom_gib !== null);
        const off = keys.filter(k => got[k] !== undefined && got[k] !== req.required[k]), soft = off.filter(k => !fill && (req.dram_fill_only || []).indexOf(k) >= 0);
        const bad = off.filter(k => soft.indexOf(k) < 0), miss = keys.filter(k => got[k] === undefined);
        out.push({ id: 'system', label: Y.label, status: bad.length ? 'FAIL' : miss.length || soft.length ? 'UNKNOWN' : 'PASS',
          detail: keys.map(k => k + ' ' + (got[k] === undefined ? '?' : got[k]) + (got[k] === req.required[k] ? '' : ' (want ' + req.required[k] + ')')).join(', ') + (bad.length || miss.length ? ': ' + req.why : '') +
            (soft.length ? '; ' + soft.join(', ') + ' without dram-fill: harm unmeasured (' + req.dram_fill_only_src + ')' : '') });
      }
    }
    // (c) -rtr on a split model
    {
      const R = C.rtr;
      out.push({ id: 'rtr', label: R.label, status: !f.rtr ? 'PASS' : 'FAIL',
        detail: !f.rtr ? '-rtr off' : f.split ? 'split model with -rtr: prefill x' + v(R.prefill_mult) + ', decode x' + v(R.decode_mult) + ' (B0, MEASURED)'
          : 'one-chain GGUF with -rtr: decode x' + v(R.chain_decode_mult) + ', prefill x' + v(R.chain_prefill_mult) + ' (GLM r2 vs r1, MEASURED)' });
    }
    // (d) quality gate on the 2-bit share of routed reads
    {
      const Q = C.quality, fam = Q.families[data.models[f.model].kld_family] || Q.families[f.model];
      // each family names its own gate reference (GLM: the EXL3 3.05 store; V4.1: the 1017 Q8_0); a borrowed GLM line keeps GLM's
      const ref = 'KL vs ' + ((fam && fam.slope_kl && fam.ref) || Q.families.glm.ref);
      const sh = f.lowbit_share || 0;
      let st = 'PASS', d = '2-bit share ' + (100 * sh).toFixed(1) + ' %';
      // the family's own fitted line once it has measured points (build.py), else the GLM line as an ESTIMATE
      const glm = Q.families.glm, line = fam && fam.slope_kl ? fam : glm;
      const pred = x => ({ kl: v(line.base_kl) + v(line.slope_kl) * x, top1: line.base_top1 - v(line.slope_top1) * x });
      const measured = f.decode_kl !== undefined && f.decode_kl !== null;
      const lm0 = layerMap(data, f.model), lm = lm0 && lm0.status === 'calibrated' ? lm0 : null;
      // a whole store file measured on the gate (its own KL vs the gate reference) decides before any 2-bit-share line
      const ms = f.store_type && !f.trim && (Q.stores || {})[f.model + '|' + f.store_type];
      if (ms) {
        st = v(ms.kl) <= Q.bar_kl && ms.top1 >= Q.bar_top1 ? 'PASS' : 'FAIL';
        d = 'store ' + f.store_type + ' MEASURED on the gate: ' + ref + ' ' + v(ms.kl).toFixed(4) + ' (bar ' + Q.bar_kl + '), top-1 ' + ms.top1.toFixed(3) + ' (bar ' + Q.bar_top1 + '), n ' + ms.n + ' (' + ms.src.split(':')[0] + ')';
      } else if (f.lowbit_share === null) {
        d = '2-bit share unknown: a store type is 2-bit but ' + (f.lowbit_why || 'the layout cannot be built');
        if (measured) { st = f.decode_kl <= Q.bar_kl ? 'PASS' : 'FAIL'; d += '; measured ' + ref + ' ' + f.decode_kl.toFixed(4) + ' (bar ' + Q.bar_kl + ')'; }
        else st = 'UNKNOWN';
      } else if ((f.trim_kl === null || f.trim_why) && f.trim) {
        st = 'UNKNOWN'; d += '; route trim KL UNKNOWN: ' + (f.trim_why || 'no layer map');
      } else if (lm && (sh > 1e-6 || f.trim_kl) && f.lowbit_eff !== undefined && f.lowbit_eff !== null) {
        // the layer map: 2-bit reads weighted by their layers' sensitivity, on the map's own calibrated line, plus the trim KL
        const p = mapQuality(data, f.model, f.lowbit_eff, f.trim_kl || 0), kl = measured ? f.decode_kl : p.kl;
        d += ', layer-map weighted ' + (100 * f.lowbit_eff).toFixed(1) + ' %' + (f.trim_kl ? ', route trim KL ' + f.trim_kl.toFixed(4) : '') + ': ' + (measured ? 'measured ' + ref + ' ' : 'PREDICTED (layer map, ' + lm.points.length + ' calibration points) ' + ref + ' ') + kl.toFixed(4) + ' (bar ' + Q.bar_kl + '), ' + (measured ? 'predicted top-1 ' : 'top-1 ') + p.top1.toFixed(3) + ' (bar ' + Q.bar_top1 + ')';
        if (fam && (fam.engines || []).indexOf(f.engine) >= 0) {
          st = (kl <= Q.bar_kl && (measured || p.top1 >= Q.bar_top1)) ? 'PASS' : (!measured && kl <= Q.bar_kl * 1.15) ? 'UNKNOWN' : 'FAIL';
          if (st === 'UNKNOWN') d += '; within 15 % of the bar';
        } else { st = 'UNKNOWN'; d += '; map line fitted on ' + ((fam && fam.engines) || []).join('/') + ' only, engine ' + f.engine + ' unmeasured'; }
      } else if (sh > 1e-6) {
        const p = pred(sh);
        const kl = measured ? f.decode_kl : p.kl;
        d += ': ' + (measured ? 'measured ' + ref + ' ' : 'predicted ' + ref + ' ') + kl.toFixed(4) + ' (bar ' + Q.bar_kl + '), ' + (measured ? 'predicted top-1 ' : 'top-1 ') + p.top1.toFixed(3) + ' (bar ' + Q.bar_top1 + ')';
        if (fam && fam.unmeasured) { st = 'UNKNOWN'; d += '; ' + fam.note; }
        else if (fam && (fam.engines || []).indexOf(f.engine) >= 0) {
          st = (kl <= Q.bar_kl && (measured || p.top1 >= Q.bar_top1)) ? 'PASS' : (!measured && kl <= Q.bar_kl * 1.15) ? 'UNKNOWN' : 'FAIL';
          if (st === 'UNKNOWN') d += '; within 15 % of the bar on a ' + line.points.length + '-point fit' + ((line.pending || []).length ? ': ' + line.pending.map(x => x.id).join(' / ') + ' adds a point' : '');
        } else if (line !== glm) { st = 'UNKNOWN'; d += '; line fitted on ' + (fam.engines || []).join('/') + ' only, engine ' + f.engine + ' unmeasured'; }
        else { st = 'UNKNOWN'; d += '; slope borrowed from GLM glm-lite (ESTIMATE): ' + ((fam && fam.note) || 'no measurement for this family/engine'); }
      } else if (fam && fam.unmeasured) { st = 'UNKNOWN'; d += ': nothing at 2 bits, but ' + fam.note; }
      else d += ': nothing at 2 bits';
      out.push({ id: 'quality', label: Q.label, status: st, detail: d });
    }
    return out;
  }
  // ---------- screening: verdict-ready (data.checks.verdict_ready, data.cards, data.screen) ----------
  // a screening row (runs/screen.jsonl) against its model's card conditions: FAIL (NOT verdict-ready) when the failure-shape
  // rate passes shape_rate_max or a setting departs from the card with no recorded best-config sweep choosing that value;
  // UNKNOWN when a compared card value or run setting is unknown; PASS = a keep/delete verdict may be given
  const cardField = (card, p) => p.split('.').reduce((o, k) => (o ? o[k] : undefined), card);
  function verdictReady(data, row) {
    const C = data.checks.verdict_ready, card = (data.cards || {})[row.card], tol = v(C.eq_tol), max = v(C.shape_rate_max);
    const same = (a, b) => typeof a === 'number' && typeof b === 'number' ? Math.abs(a - b) <= tol : String(a) === String(b);
    const s = row.shapes || {}, rate = s.requests ? s.any / s.requests : null, bad = [], unk = [], dep = [];
    if (!card) unk.push('no card conditions for ' + row.card + ' (data/cards)');
    if (rate === null) unk.push('no requests recorded');
    else if (rate > max) bad.push(`failure shapes on ${s.any} of ${s.requests} requests (${(100 * rate).toFixed(0)} %, over ${(100 * max).toFixed(0)} %): ` +
      `cap hits ${s.cap_hit}, empty ${s.empty}, loops ${s.loop}, leaked tags ${s.leaked_tags}; answered only on retry ${s.retry_only}`);
    const sweeps = ((data.screen || {}).rows || []).filter(r => r.purpose === 'sweep' && r.card === row.card && r.sweep && r.sweep.best);
    Object.entries(v(C.fields)).forEach(([k, f]) => {
      if (!card) return;
      const used = (row.settings || {})[k], cv = cardField(card, f.card), want = cv && typeof cv === 'object' ? cv.v : null;
      const has = x => x !== null && x !== undefined;
      let off = null;
      if (f.rule === 'unset') { if (has(used) && !(has(want) && same(used, want))) off = `${k} ${used}: the card sets ${has(want) ? want : 'none'}`; }
      else if (!has(want)) { unk.push(`${k}: the card value is unknown (${cv && cv.src ? cv.src : 'no ' + f.card + ' on the card'})`); return; }
      else if (!has(used)) { unk.push(`${k}: the run does not record it`); return; }
      else if (f.rule === 'eq') { if (!same(used, want)) off = `${k} ${used} vs card ${want}`; }
      else if (f.rule === 'min') { if (used < want) off = `${k} ${used} < card ${want}`; }
      else if (f.rule === 'room') {
        const out = (cardField(card, f.output) || {}).v, o = has(out) ? out : row.settings.max_output, need = Math.min(want, (row.prompt_tokens_max || 0) + o);
        if (used < need) off = `${k} ${used} < ${need} (longest prompt ${row.prompt_tokens_max} + ${has(out) ? 'card' : 'run'} max output ${o}, at most the card context ${want})`;
      }
      if (!off) return;
      const sw = sweeps.find(r => has(r.sweep.best[k]) && same(r.sweep.best[k], used));
      dep.push(Object.assign({ field: k, text: off, src: (row.settings_src || {})[k] || null, card_src: cv ? cv.src : null }, sw ? { sweep: sw.id } : {}));
    });
    const open = dep.filter(d => !d.sweep), st = bad.length || open.length ? 'FAIL' : unk.length ? 'UNKNOWN' : 'PASS';
    const head = st === 'PASS' ? 'verdict-ready' : st === 'FAIL' ? 'NOT verdict-ready' : 'verdict-ready UNKNOWN';
    const parts = bad.concat(open.map(d => 'departs: ' + d.text + (d.src ? ' [' + d.src + ']' : '') + ', no recorded best-config sweep'),
      dep.filter(d => d.sweep).map(d => d.text + ' (chosen by sweep ' + d.sweep + ')'), unk);
    return { id: 'verdict_ready', label: C.label, status: st, rate, departures: dep, unknown: unk,
      verdict: row.verdict || null, refused: st !== 'PASS' && !!row.verdict,
      detail: head + (parts.length ? ': ' + parts.join('; ') : ': card conditions held, failure shapes ' + (100 * rate).toFixed(0) + ' % of requests') };
  }
  // every screening row with its check, newest first
  function screenChecks(data) {
    return ((data.screen || {}).rows || []).map(r => ({ row: r, check: verdictReady(data, r) })).sort((a, b) => String(b.row.when).localeCompare(String(a.row.when)));
  }
  // ---------- pre-launch check of ONE concrete recipe (node report.js --check recipe.json) ----------
  // recipe: { id, box, model, engine, store: {split, types, tensor_types, read_share_by_type}, lowbit_share, banks_gib, rtr, ub,
  //           threads, draft, knobs, vram: {post_warmup_gib, gpu_total_gib}, decode_kl, files: [{path, holds, access}] }
  // access: 'on_file' / 'mmap' (read while serving) or 'loaded' (read once at load, e.g. --no-mmap)
  // disk under a path: the longest mount prefix among the box's disks (probed mounts overlay the hand table). A '~/...' path
  // is the home of whoever runs the recipe on the box: the mount the probe found holding $HOME (box.home_mount), else the
  // mount holding /home (the Linux default). No user name is assumed.
  function mountKind(box, path) {
    const p = String(path), disks = box.disks || [];
    const under = (q, d) => q === d.mount || q.startsWith(d.mount === '/' ? '/' : d.mount + '/');
    const longest = q => disks.filter(d => under(q, d)).reduce((b, d) => (!b || d.mount.length > b.mount.length ? d : b), null);
    let best;
    if (/^~(\/|$)/.test(p)) {
      const hm = v(box.home_mount);
      best = (hm && disks.find(d => d.mount === hm)) || longest('/home');
    } else best = longest(p);
    return best ? { kind: best.kind, mount: best.mount, path: p } : { kind: 'unknown', mount: null, path: p };
  }
  // recipe VRAM figures in GiB. vram.units naming 'MiB / 1000' (droopy lab notes/PLAN readings: a 23,040 MiB card reads 23.04)
  // converts post_warmup_gib, peak_gib and gpu_total_gib by x 1000/1024; anything else is taken as GiB
  function vramGiB(vram) {
    const x = vram || {}, k = /MiB\s*\/\s*1000/i.test(String(x.units || '')) ? 1000 / 1024 : 1;
    const g = y => (y === undefined || y === null || y === 0) ? null : y * k;
    return { post: g(x.post_warmup_gib), peak: g(x.peak_gib), total: g(x.gpu_total_gib), converted: k !== 1 };
  }
  // gpu_split: { banks_gib: [per usable GPU], fixed_share: [per usable GPU] } - an engine that splits the model by layer over
  // several cards (llama.cpp -sm layer). Each usable GPU (engine-capable, not banks-only, box order) holds its own reserves, fixed_share of the
  // fixed parts the single-GPU plan puts on g0 (dense, KV + compute buffers, draft), and its banks. Returns one row per card;
  // the worst card (least room) is the one the VRAM check judges. null when the recipe has no gpu_split.
  function splitPosts(ctx, r) {
    const gs = r.gpu_split;
    if (!gs) return null;
    const cards = ctx.tiers.filter(t => t.kind === 'gpu' && t.engineOk && !t.banks);
    const banks = gs.banks_gib || [], share = gs.fixed_share || cards.map(() => 1 / cards.length);
    if (banks.length !== cards.length || share.length !== cards.length)
      return { bad: 'gpu_split gives ' + banks.length + ' banks / ' + share.length + ' shares for ' + cards.length + ' usable GPUs (' + cards.map(c => c.name).join(', ') + ')' };
    let fixedG0 = 0;
    ctx.fixed.filter(x => x.tier === ctx.g0.id && x.kind !== 'margin').forEach(x => fixedG0 += x.gib);
    if (ctx.E.kv_includes_margin) fixedG0 -= ctx.E.kv_includes_margin;
    const rows = cards.map((c, i) => ({ name: c.name, total_gib: c.total / GiB, banks_gib: banks[i],
      post_gib: c.reserves.reduce((a, x) => a + x.gib, 0) + share[i] * fixedG0 + banks[i] }));
    const worst = rows.reduce((b, x) => (!b || x.total_gib - x.post_gib < b.total_gib - b.post_gib ? x : b), null);
    return { rows, worst, banks_sum: banks.reduce((a, x) => a + x, 0),
      how: 'post-warm-up PREDICTED per card (gpu_split, fixed parts ' + fixedG0.toFixed(2) + ' GiB shared ' + share.join('/') + '): ' +
        rows.map(x => x.name + ' ' + x.post_gib.toFixed(2) + ' of ' + x.total_gib.toFixed(2) + ' (banks ' + x.banks_gib + ')').join('; ') + '; judged on ' + worst.name };
  }
  function checkRecipe(data, r, resolve, opts) {
    if (r && r.kind === 'network') return networkCheck(data, r);
    if (r && r.kind === 'dllm') return dllmCheck(data, r);
    const inp = [], notes = [];
    const box = data.boxes[r.box], m = data.models[r.model], E = data.engines[r.engine];
    if (!box || !m || !E) return { checks: [{ id: 'inputs', status: 'FAIL', detail: 'unknown ' + [!box && 'box ' + r.box, !m && 'model ' + r.model, !E && 'engine ' + r.engine].filter(Boolean).join(', ') + ' (boxes: ' + Object.keys(data.boxes).join(' ') + '; models: ' + Object.keys(data.models).join(' ') + '; engines: ' + Object.keys(data.engines).join(' ') + ')' }], facts: null, notes, bad_input: true };
    const kn = Object.assign({}, data.knobs, r.knobs || {}, { ub: r.ub !== undefined ? String(r.ub) : 'auto', draft: r.draft !== undefined ? !!r.draft : !!(r.knobs && r.knobs.draft), rtr: !!r.rtr }, storeKnobs(r));
    const gpus = box.gpus.concat(kn.moveB70 === r.box ? [data.b70move.gpu] : []);
    if (!gpus.some(g => (g.engines || []).indexOf(r.engine) >= 0)) inp.push(['FAIL', r.engine + ' drives no GPU on ' + r.box + ' (' + gpus.map(g => g.id + ': ' + (g.engines || []).join('/')).join(', ') + ')']);
    if (E.models && E.models.indexOf(r.model) < 0) inp.push(['FAIL', r.engine + ' runs only ' + E.models.join('/')]);
    if (E.packages && !E.packages[r.box]) inp.push(['UNKNOWN', 'no ' + r.engine + ' build for ' + r.box + ': ' + (E.backend_note || '')]);
    const st = r.store || {};
    const ts = new Set((st.types || []).concat(Object.values(st.tensor_types || {})).concat(Object.keys(st.read_share_by_type || {})));
    ts.forEach(q => {
      if (!data.types[q]) { inp.push(['UNKNOWN', 'type ' + q + ' is not in the type table']); return; }
      if ((E.types || []).indexOf(q) < 0) inp.push(['FAIL', 'type ' + q + ' is not available in engine ' + r.engine]);
      if (m.types_only && m.types_only.indexOf(q) < 0) inp.push(['FAIL', m.short + ' ships only as ' + m.types_only.join('/') + ' (no source weights to requant)']);
    });
    Object.keys(m.row_limits || {}).forEach(tn => {
      const lim = m.row_limits[tn], q = (st.tensor_types || {})[tn] || null;
      const cand = q ? [q] : (st.tensor_types ? [] : (st.types || []).filter(x => !(data.types[x] && data.types[x].mix_handles_rows)));
      cand.forEach(x => { if (lim.allowed.indexOf(x) < 0) inp.push([q ? 'FAIL' : 'UNKNOWN', tn + ' has ' + lim.rows + ' rows: ' + x + ' needs 256-row blocks; allowed ' + lim.allowed.join('/') + (q ? '' : ' (give store.tensor_types.' + tn + ' to make this exact)') + ' (' + lim.src + ')']); });
    });
    if (r.threads !== undefined) {
      if (r.threads > v(box.cpu.threads)) inp.push(['FAIL', 'threads ' + r.threads + ' > ' + v(box.cpu.threads) + ' CPUs on ' + r.box]);
      else if (r.threads > v(box.cpu.phys)) notes.push('threads ' + r.threads + ' > ' + v(box.cpu.phys) + ' physical cores: SMT siblings in use (' + box.threads_note + ')');
    }
    const sv = servingOf(data, r);
    if (sv) sv.bad.forEach(x => inp.push(['FAIL', x]));
    else if (r.serving) inp.push(['UNKNOWN', 'recipe.serving given, but engine ' + r.engine + ' has no serving model (engines.' + r.engine + '.serving): not judged']);
    // heat: the sets the recipe plans on must exist; a cited hot-expert map digest must match the data's map, so a recipe
    // planned on other counts than the data now holds refuses instead of predicting from the wrong heat
    const hs = r.heat || {};
    ['plan', 'score'].forEach(w => {
      const name = hs[w] || m.heat[w + '_default'], set = m.heat.sets[name];
      if (!set) { inp.push(['FAIL', w + ' heat set ' + name + ' is not in ' + m.short + ' (sets: ' + Object.keys(m.heat.sets).join(' ') + ')']); return; }
      const have = heatCite(m, name), want = [].concat((hs.maps || {})[w] || []);
      const miss = want.filter(c => !have.some(x => x.digest === c.digest));
      if (miss.length || (want.length && want.length !== have.length)) inp.push(['FAIL', w + ' heat ' + name + ': the recipe cites ' + want.map(c => (c.map_id || '?') + ' ' + c.digest).join(' + ') + ', the data has ' + (have.map(c => c.map_id + ' ' + c.digest).join(' + ') || 'no map')]);
      else if (have.length) notes.push(w + ' heat ' + name + ' = ' + have.map(c => c.map_id + ' slice ' + c.slice + ' ' + c.digest.slice(0, 19)).join(' + ') + (want.length ? ' (cited by the recipe, matches)' : ' (the recipe does not cite it: add heat.maps.' + w + ')'));
      else notes.push(w + ' heat ' + name + ': no hot-expert map behind it (' + (set.tag || 'untagged') + ')');
    });
    // storage per tensor class from the file list
    const storage = {};
    (r.files || []).forEach(fl => {
      const path = resolve ? resolve(fl.path) : fl.path;
      const mk = mountKind(box, path);
      const runtime = fl.access === 'on_file' || fl.access === 'mmap';
      const kind = mk.kind;
      if (runtime) {
        const cls = fl.holds || 'file';
        const rank = { hdd: 3, unknown: 2 };
        if (!storage[cls] || (rank[kind] || 0) > (rank[storage[cls]] || 0)) storage[cls] = kind;
      } else if (kind === 'hdd') notes.push(fl.holds + ' (' + mk.path + ') is loaded from a rotational disk: load time only (droopy V4.1: ~35 min per --no-mmap load at ~140 MB/s, lab notes 20:43Z 10-16)');
      notes.push('file ' + mk.path + (path !== fl.path ? ' (resolved from ' + fl.path + ')' : '') + ' -> ' + (mk.mount || 'no known mount') + ' [' + kind + '], ' + (fl.holds || '?') + ', ' + (runtime ? 'read while serving' : 'read at load'));
    });
    // VRAM: measured post-warm-up if given, else the model's prediction from the same parts the planner reserves
    const ctx = context(data, r.box, r.model, r.engine, kn);
    const g0 = ctx.g0;
    let post, how;
    const vr = vramGiB(r.vram);
    if (vr.converted) notes.push('vram figures given in nvidia-smi MiB / 1000 (vram.units): converted to GiB x 1000/1024' + (vr.total !== null ? ', card ' + r.vram.gpu_total_gib + ' -> ' + vr.total.toFixed(2) + ' GiB' : ''));
    if (vr.post !== null) {
      post = vr.post; how = 'post-warm-up given by the recipe (' + (r.vram.src || 'no source') + ')' + (vr.converted ? ': ' + r.vram.post_warmup_gib + ' MiB/1000 = ' + post.toFixed(2) + ' GiB' : '');
      if (vr.total === null) notes.push('vram.gpu_total_gib not given: using ' + (g0.total / GiB).toFixed(2) + ' GiB from the box table');
    }
    else if (r.gpu_split) {
      const sp = splitPosts(ctx, r);
      if (sp.bad) { inp.push(['FAIL', sp.bad]); post = null; how = sp.bad; }
      else {
        post = sp.worst.post_gib; how = sp.how; vr.total = sp.worst.total_gib;
        if (r.banks_gib !== undefined && r.banks_gib !== null && Math.abs(r.banks_gib - sp.banks_sum) > 0.01) inp.push(['FAIL', 'banks_gib ' + r.banks_gib + ' != the gpu_split banks sum ' + sp.banks_sum.toFixed(2)]);
      }
    }
    else {
      post = g0.reserves.reduce((a, x) => a + x.gib, 0);
      ctx.fixed.filter(x => x.tier === g0.id && x.kind !== 'margin').forEach(x => post += x.gib);
      if (E.kv_includes_margin) post -= E.kv_includes_margin;
      post += (r.banks_gib || 0) + offloadVram(data, r);
      how = 'post-warm-up PREDICTED on ' + g0.name + ': ' + g0.reserves.filter(x => x.gib > 0.01).map(x => x.label + ' ' + x.gib.toFixed(2)).concat(ctx.fixed.filter(x => x.tier === g0.id && x.kind !== 'margin').map(x => x.label + ' ' + x.gib.toFixed(2))).concat(r.banks_gib ? ['expert banks ' + r.banks_gib] : []).concat(offloadVram(data, r) ? ['prefill offload upload buffers + compute graph ' + offloadVram(data, r).toFixed(2)] : []).join(' + ') + (E.kv_includes_margin ? ' - reserve ' + E.kv_includes_margin : '');
    }
    notes.push(how + (ctx.notes.length ? '. ' + ctx.notes.join(' ') : ''));
    const low = recipeLowShare(data, r);
    if (low.how) notes.push('2-bit share ' + (100 * low.share).toFixed(1) + ' % of routed reads from the ' + low.how);
    // the layer map: the same 2-bit share weighted by its layers' sensitivity, and a route trim's KL (from the placement)
    const lw = layerWeights(data, r.model), dm = lw && low.share !== null ? recipeDamage(data, r) : null;
    const eff = dm && dm.d ? dm.d.reduce((a, x, l) => a + x * lw[l], 0) / dm.d.length : null;
    const predictFirst = r.trim ? predictRecipe(data, r) : null, tk = predictFirst && predictFirst.trim;
    const tsup = r.trim ? trimSupport(data, r) : null;
    if (tsup && tsup.status !== 'yes') inp.push([tsup.status === 'needs_code' ? 'UNKNOWN' : 'FAIL', 'route trim on ' + r.engine + ' ' + (tsup.status === 'needs_code' ? 'needs code: ' : '') + tsup.src]);
    const facts = { box: r.box, model: r.model, engine: r.engine, split: !!st.split, rtr: !!r.rtr, gpu_total: vr.total !== null ? vr.total : g0.total / GiB,
      post_gib: post, storage, lowbit_share: low.share, lowbit_why: low.why, decode_kl: r.decode_kl, lowbit_eff: eff,
      trim: !!r.trim, trim_kl: tk ? tk.kl : r.trim ? null : 0, trim_why: r.trim ? (tk ? tk.why : (predictFirst && predictFirst.why) || 'no prediction') : null, store_type: storeKnobs(r).storeType, numa: r.numa || null, system: r.system || null, banks_gib: r.banks_gib,
      peak_gib: vr.peak, peak_src: r.vram && r.vram.src };
    const worst = inp.some(x => x[0] === 'FAIL') ? 'FAIL' : inp.length ? 'UNKNOWN' : 'PASS';
    const checks = [{ id: 'inputs', label: 'recipe inputs (box, engine, types, row limits, threads)', status: worst, detail: inp.length ? inp.map(x => x[0] + ': ' + x[1]).join('; ') : 'consistent' }].concat(runChecks(data, facts));
    checks.push(conditionsCheck(data, r, kn.draft));
    checks.push(cpuReserveCheck(data, r));
    const qChk = checks.find(c => c.id === 'quality');
    checks.push(earfCheck(data, r, Object.assign({}, opts || {}, { qualityStatus: qChk && qChk.status })));
    checks.push(...seatSafetyChecks(data, r, opts));
    const kvh = kvHostCheck(data, r);
    if (kvh) checks.push(kvh);
    const hp = hostPinnedCheck(data, r);
    if (hp) checks.push(hp);
    const er = expertResidencyCheck(data, r);
    if (er) checks.push(er);
    const fb = fallbackCheck(data, r, opts && opts.measured);
    if (fb) checks.push(fb);
    const cp = cpuPoolCheck(data, r);
    if (cp) checks.push(cp);
    const aw = agentWorkloadCheck(data, r);
    if (aw) checks.push(aw);
    const pf = predictionFirstCheck(data, r, aw);
    if (pf) checks.push(pf);
    const prediction = opts && opts.predict === false ? null : predictFirst || predictRecipe(data, r);
    const pbc = phaseBoundCheck(data, r, prediction);
    if (pbc) checks.push(pbc);
    if (prediction && prediction.ram_overflow_gib) checks.push({ id: 'ram_fit', label: 'loaded (pinned) experts fit DRAM', status: 'FAIL', detail: prediction.why });
    // host memory: the SYCL pinned host buffer is driver memory (GPUActive); the container memcg does not hold it (MEASURED odie
    // c0 2026-10-18: memcg anon 0.9 GiB + page cache, GPUActive 52.8-54.7 GiB). So the pinned peak must fit DRAM total - the
    // household floor by itself, and the docker cap (mem_cap_gib) only bounds the container's anon + page cache: it must hold the
    // container's anon with 1 GiB to spare and stay under that ceiling.
    const floor = box.dram && box.dram.host_floor_gib ? v(box.dram.host_floor_gib) : null;
    if (prediction && prediction.host_peak_gib !== null && prediction.host_peak_gib !== undefined && floor !== null) {
      const margin = box.dram.pinned_margin_gib ? v(box.dram.pinned_margin_gib) : 0;
      const cap = r.mem_cap_gib, top = v(box.dram.total_gib) - floor - margin, pk = prediction.host_peak_gib, an = prediction.container_anon_gib || 0;
      const st = pk + an > top ? 'FAIL' : cap === undefined || cap === null ? 'UNKNOWN' : (cap < an + 1 || cap > top) ? 'FAIL' : 'PASS';
      checks.push({ id: 'mem_cap', label: 'pinned host peak + container anon fit under DRAM total - household floor; docker --memory holds the anon', status: st,
        detail: 'predicted pinned host peak ' + pk.toFixed(1) + ' GiB (pinned experts + host extra ' + prediction.host_extra_gib + ' FIT + driver extra ' + prediction.pinned_driver_extra_gib + ') + container anon ' + an + ' GiB; ceiling ' + top.toFixed(1) + ' GiB (' + v(box.dram.total_gib) + ' total - floor ' + floor + ' - margin ' + margin + '; a kill cannot land inside a driver allocation, so this ceiling is the containment); cap ' + (cap === undefined || cap === null ? 'not declared (mem_cap_gib)' : cap + ' GiB (holds anon + page cache, not the pinned buffer)') });
    }
    return { checks, facts, notes, bad_input: false, prediction, priors: priorsFor(data, r) };
  }
  // Behavioral admission: executed outcomes first, then EARf relative to the model perturbation floor.
  function earfStoreTypes(r) {
    const st = (r && r.store) || {};
    if (typeof st === 'string') return { key: st, types: [st] };
    const key = st.name || st.id || (!st.split && !st.layer_types && !st.read_share_by_type && !st.hot_type && (st.types || []).length === 1 ? st.types[0] : null);
    const raw = [].concat(st.types || [], st.hot_type || [], st.cold_type || [], (st.layer_types || []).map(x => x && x.type),
      Object.keys(st.read_share_by_type || {}), Object.values(st.tensor_types || {})).filter(Boolean);
    return { key, types: [...new Set(raw)] };
  }
  function earfOf(data, r, opts) {
    const m = (opts && opts.measured) || {}, C = (data && data.checks && data.checks.earf) || {};
    const num = x => { const y = v(x); return typeof y === 'number' && Number.isFinite(y) && y >= 0 && y <= 1 ? y : null; };
    if (m.earf !== undefined && m.earf !== null && num(m.earf) !== null)
      return { earf: num(m.earf), src: (m.earf && m.earf.src) || m.earf_src || 'measured.earf' };
    if (r && r.earf !== undefined && r.earf !== null && num(r.earf) !== null)
      return { earf: num(r.earf), src: (r.earf && r.earf.src) || r.earf_src || 'recipe.earf' };
    const { key, types } = earfStoreTypes(r), entries = C.entries || {};
    const cands = [key].concat(types).filter(Boolean);
    const hits = [], covered = new Set();
    for (const k of cands) {
      for (const ek of [r.model + '|' + r.engine + '|' + k, r.model + '|' + k, k]) {
        if (entries[ek] !== undefined && num(entries[ek]) !== null) {
          covered.add(k);
          hits.push({ earf: num(entries[ek]), src: (entries[ek] && entries[ek].src) || ('checks.earf.entries.' + ek) });
        }
      }
    }
    const MB = (data && data.multibox) || {}, M = (MB.models || {})[r && r.model], stores = (M && M.stores) || {};
    for (const k of cands) {
      const f = stores[k] && stores[k].fidelity && stores[k].fidelity.arg_values_earf;
      if (f && num(f) !== null) { covered.add(k); hits.push({ earf: num(f), src: f.src || ('multibox.models.' + r.model + '.stores.' + k) }); }
    }
    const T = v((MB.settings || {}).replica_type_earf) || {}, tSrc = ((MB.settings || {}).replica_type_earf || {}).src || 'multibox.settings.replica_type_earf';
    for (const q of types) {
      if (T[q] !== undefined && num(T[q]) !== null) { covered.add(q); hits.push({ earf: num(T[q]), src: tSrc + ' (' + q + ')' }); }
    }
    if (!hits.length) return { earf: null, src: null };
    return Object.assign({}, hits.reduce((a, b) => (b.earf < a.earf ? b : a)), {
      unmeasured: key && covered.has(key) ? [] : types.filter(q => !covered.has(q))
    });
  }
  function isSub4BitQuant(data, model, q) {
    const t = data && data.types && data.types[q];
    if (t && Number.isFinite(v(t.bpw))) {
      if (v(t.bpw) < 4) return true;
    } else if (/^(?:gsq_)?iq[123]|^ud_iq[123]|^ud_q[23]|^q[23]_|^exl3_[123]\.|int[23]/i.test(String(q || ''))) return true;
    const S = ((((data && data.multibox) || {}).models || {})[model] || {}).stores || {};
    const note = S[q] && S[q].fidelity && S[q].fidelity.note;
    return !!(note && /iq[123]|sub-4|not a clean 4-bit/i.test(note));
  }
  function requiresEarf(r, opts) {
    const role = (opts && opts.role) || (r && r.role) || ((((r || {}).conditions || {}).workload || {}).role);
    return !!(opts && opts.requireEarf) || ['coding', 'agent', 'agentic', 'agentic-coding', 'tool-calling', 'tool-use', 'author_under_judge', 'diff_review'].includes(role);
  }
  // Deployment budgets are distinct from a backend's advertised context capacity.
  function seatSafetyChecks(data, r, opts) {
    const c = r.conditions || {}, w = c.workload || {}, sv = r.serving || {};
    const out = [], model = (data.models || {})[r.model] || {};
    const name = [r.model, model.name, model.short].join(' ');
    const depth = w.context_tokens ?? c.context_tokens ?? r.ctx ?? sv.max_context;
    // The only depth-recall probe we hold is on the int4 file of Qwen3.8-27B; another quant of it, or a declared budget, is its own case.
    const declared = r.measured_recall_budget_tokens ?? model.measured_recall_budget_tokens ?? null;
    const q27 = declared === null && /(?:^|[^a-z0-9])qwen3\.8-27b(?![0-9]).*int4/i.test(name);
    const budget = q27 ? 12288 : declared;
    // Judged for an agent seat, or wherever a recall budget is known; a plain decode run with no budget gets no row.
    if (depth !== undefined && depth !== null && (budget !== null || requiresEarf(r, opts))) {
      const tokens = firstInt(depth), known = Number.isInteger(budget) && budget > 0 && Number.isFinite(tokens);
      const over = known && tokens > budget, degraded = over && (!q27 || tokens > 23000);
      out.push({ id: 'context_recall', status: known && !over ? 'PASS' : 'UNKNOWN',
        severity: degraded ? 'degraded' : over ? 'warn' : null,
        detail: !Number.isFinite(tokens) ? 'Declared context ' + JSON.stringify(depth) + ' is not a token count; recall not judged.' :
          !known ? 'No per-model depth-recall probe; context capacity does not establish recall.' :
          q27 ? 'Qwen3.8-27B int4: 9/9 at <=12k, 3/9 at ~23k; 12–23k unmeasured. Warn above 12288, degraded above 23000 tokens.' :
          'Explicit measured recall budget ' + budget + ' tokens; declared context ' + tokens });
    }
    const startup = w.startup_tokens ?? r.startup_tokens, startupLimit = sv.max_startup_tokens;
    if (startup !== undefined && startupLimit !== undefined)
      out.push({ id: 'startup_context', status: Number.isInteger(startup) && startup >= 0 && startup <= startupLimit ? 'PASS' : 'FAIL',
        detail: 'Configured startup budget ' + startupLimit + ' tokens; declared ' + startup });
    // a workload may sweep agent counts ([1, 4, 8]) and tag its figures ({v, tag, src}): judge the largest count
    const count = x => Array.isArray(x) ? Math.max(...x.map(v).filter(Number.isFinite)) : v(x);
    const agents = count(w.agents ?? w.concurrency ?? r.agents), slots = count(w.slots ?? sv.batch);
    const admission = w.admission || r.admission;
    if (Number.isFinite(agents) && Number.isFinite(slots) && agents > slots) {
      const door = admission && admission.policy === 'door' && Number.isInteger(admission.conversations) && admission.conversations > 0 && admission.conversations <= slots;
      out.push({ id: 'slot_admission', status: door ? 'PASS' : 'FAIL', detail: agents + ' agents / ' + slots + ' slots: require a conversation door capped at slots. R8 B=33 on 32 slots collapsed from 79.1 to 9.7 useful tok/s; an engine queue is not stable admission.' });
    }
    return out;
  }
  function earfCheck(data, r, opts) {
    const C = (data && data.checks && data.checks.earf) || {}, m = (opts && opts.measured) || {};
    const model = ((data || {}).models || {})[r.model] || {};
    const familyKey = model.kld_family || r.family || r.model;
    const family = (C.families || {})[familyKey] || {};
    const qualityFamily = (((data || {}).checks || {}).quality?.families || {})[familyKey] || {};
    const sources = [m, r, model, family, qualityFamily];
    const field = key => sources.find(x => x[key] !== undefined && x[key] !== null)?.[key];
    const valid = x => typeof x === 'number' && Number.isFinite(x) && x >= 0 && x <= 1;
    const review = field('review_screen');
    const reviewer = !!(review && review.executed === true && review.planted === true && review.clean === true &&
      valid(review.false_alarm_rate) && review.false_alarm_rate <= .10 && valid(review.recall) && review.recall >= .50);
    const { key, types } = earfStoreTypes(r);
    const sub4bit = [key, ...types].filter(Boolean).some(q => isSub4BitQuant(data, r.model, q));
    const base = { id: 'earf', label: 'bounded author admission under an executed Judge', role: 'author_under_judge',
      diff_reviewer: reviewer, capabilities: { diff_review: reviewer ? 'screened' : 'advisory_excluded' },
      earf: null, min_earf: null, sub4bit };
    const result = (status, detail, extra) => Object.assign({}, base, { status, detail }, extra);
    // Outcome probes concern executable authoring under a Judge, not self-review. A declared
    // battery_pass_rate is already a measurement; structured probes must explicitly be executed.
    const probe = field('authoring_probe');
    const rate = field('battery_pass_rate') ?? (probe && probe.executed === true ?
      (probe.pass_rate ?? (Number.isInteger(probe.passed) && Number.isInteger(probe.total) && probe.total > 0 ? probe.passed / probe.total : NaN)) : null);
    if (rate !== null && rate !== undefined) {
      const bar = field('battery_min_pass_rate') ?? .90;
      if (!valid(v(rate)) || !valid(v(bar))) return result('FAIL', 'Invalid outcome battery rate or threshold');
      return result(v(rate) >= v(bar) ? 'PASS' : 'FAIL', 'Executed authoring battery ' + v(rate) + ' (bar ' + v(bar) + '); requires an executed Judge',
        { admission_basis: 'outcome_probe', battery_pass_rate: v(rate) });
    }
    const supplied = [m, r].find(x => x.earf !== undefined && x.earf !== null);
    if (supplied && !valid(v(supplied.earf))) return result('FAIL', 'Invalid measured EARf (expected finite [0, 1])');
    const floor = v(field('perturbation_floor'));
    if (floor !== undefined && !valid(floor)) return result('FAIL', 'Invalid model perturbation floor');
    const min = floor === undefined ? null : floor - .015;
    const found = earfOf(data, r, opts);
    // Admission is a property of an agent seat. A run that declares no coding/agent role and brings no behavioural
    // evidence of its own (a decode benchmark, a hardware probe) is not a seat: it is not judged here, and its KL gate stands.
    if (!requiresEarf(r, opts) && !supplied)
      return result('PASS', 'Not an agent seat (no coding/agent role declared): behavioural admission not required',
        { earf: found.earf, min_earf: min, admission_basis: 'not_required', required: false });
    if (found.earf === null || (found.unmeasured && found.unmeasured.length))
      return result('UNKNOWN', 'EARf unmeasured' + (sub4bit ? ' on sub-4-bit quant' : '') +
        (found.unmeasured?.length ? ': incomplete mixed layout (' + found.unmeasured.join(', ') + ')' : '') + '; outcome probe needed', { min_earf: min });
    const extra = { earf: found.earf, min_earf: min, admission_basis: min === null ? 'earf_screen' : 'perturbation_margin' };
    if (min !== null) return result(found.earf >= min ? 'PASS' : 'FAIL',
      'EARf ' + found.earf.toFixed(3) + ' vs perturbation floor minus 0.015 = ' + min.toFixed(3) + ' (' + found.src + ')', extra);
    if (found.earf < .90) return result('UNKNOWN', 'Severe EARf degradation: ' + found.earf.toFixed(3) + ' < 0.90; model perturbation floor unmeasured',
      Object.assign(extra, { severity: 'degraded', warning: 'Severe EARf degradation; run an outcome probe' }));
    return result('PASS', 'Measured EARf ' + found.earf.toFixed(3) + '; no severe degradation, perturbation floor unknown (outcome probe preferred)', extra);
  }
  // the serving conditions of data.conditions: a run that leaves one undeclared still runs (UNKNOWN; --strict holds it), but its
  // numbers do not compare with a run that differs in it
  function conditionsCheck(data, r, draft) {
    const C = data.conditions, have = Object.assign(r.ctx !== undefined ? { ctx: r.ctx } : {}, r.conditions || {});
    const want = Object.keys(C.fields).filter(k => draft || C.draft_only.indexOf(k) < 0), miss = want.filter(k => have[k] === undefined);
    return { id: 'conditions', label: 'serving conditions declared (each moves tok/s, quality or output length)', status: miss.length ? 'UNKNOWN' : 'PASS',
      detail: miss.length ? 'not declared in recipe.conditions: ' + miss.map(k => k + ' (' + C.fields[k] + ')').join('; ') : want.map(k => k + ' ' + JSON.stringify(have[k])).join(', ') };
  }
  // CPU decode on spin-barrier worker threads: a co-runner that takes one decode CPU stalls every barrier (droopy T40: 2-6.6 s
  // token stalls); one CPU kept free with the system pinned to it removes them at no cost (q4t). FAIL when the threads leave
  // fewer free CPUs than the box reserve (or a declared reserve is below it); UNKNOWN until recipe.conditions declares both.
  // A per-cell value ({cell: n}) is judged at its worst cell; a string counts by its leading integer
  const firstInt = x => typeof x === 'number' ? x : typeof x === 'string' && /^\s*\d+/.test(x) ? parseInt(x, 10) : null;
  const cellVals = x => x !== null && typeof x === 'object' ? Object.values(x).map(firstInt) : [firstInt(x)];
  const allNum = xs => xs.length > 0 && xs.every(x => x !== null);
  function cpuReserveCheck(data, r) {
    const C = data.checks.cpu_reserve, box = data.boxes[r.box], E = data.engines[r.engine], c = r.conditions || {};
    const cpus = v(box.cpu.threads), own = box.cpu.reserve, res = own ? v(own) : v(C.default_reserve);
    const resWhy = 'box reserve ' + res + (own ? ' (' + r.box + ' cpu.reserve ' + own.tag + ')' : ' (checks.cpu_reserve.default_reserve ESTIMATE)');
    const tc = c.threads === undefined ? [] : cellVals(c.threads), rc = c.cpu_reserve === undefined ? [] : cellVals(c.cpu_reserve);
    const thr = allNum(tc) ? Math.max(...tc) : r.threads !== undefined ? r.threads : null, dec = allNum(rc) ? Math.min(...rc) : null;
    const out = (status, detail) => ({ id: 'cpu_reserve', label: C.label, status, detail });
    if (E.cpu_decode !== true) return out('UNKNOWN', 'whether ' + r.engine + ' decodes on CPU threads is UNKNOWN (engines.' + r.engine + '.cpu_decode)' + (thr !== null ? '; threads ' + thr + ' of ' + cpus + ' CPUs' : ''));
    if (thr !== null && thr > cpus - res) return out('FAIL', 'threads ' + thr + ' of ' + cpus + ' CPUs leave ' + (cpus - thr) + ' free, under the ' + resWhy + ': a co-runner on a decode CPU stalls every spin barrier (droopy T40: 2-6.6 s token stalls, q4l/q4i/q4j); T' + (cpus - res) + ' with the system pinned to the reserve ran 0 stalls at no cost (q4t)');
    if (dec !== null && dec < res) return out('FAIL', 'cpu_reserve ' + dec + ' under the ' + resWhy);
    const miss = [!allNum(tc) && 'threads', !allNum(rc) && 'cpu_reserve'].filter(Boolean);
    if (miss.length) return out('UNKNOWN', 'not declared (as a number) in recipe.conditions: ' + miss.join(', ') + ' (' + resWhy + ' of ' + cpus + ' CPUs' + (thr !== null ? '; recipe threads ' + thr : '') + ')');
    return out('PASS', 'threads ' + thr + ' + cpu_reserve ' + dec + ' of ' + cpus + ' CPUs (' + resWhy + ')');
  }
  // the serving fields of a recipe (recipe.serving: max_context, batch, batch_groups, slot_context, big_slots, kv_resident, kv,
  // batch_mtp, mtp_window), each the engine's own default (engines.<engine>.serving.defaults) when left out; max_context
  // falls back to the recipe ctx. null when the engine has no serving model. bad = fields the engine would not take
  function servingOf(data, r) {
    const S = (data.engines[r.engine] || {}).serving;
    if (!S) return null;
    const s = r.serving || {}, D = S.defaults, bad = [];
    const out = Object.fromEntries(Object.keys(D).map(k => [k, s[k] !== undefined && s[k] !== null ? s[k] : v(D[k])]));
    if (s.max_context === undefined || s.max_context === null) out.max_context = r.ctx !== undefined ? firstInt(r.ctx) : v(D.max_context);
    else if (r.ctx !== undefined && firstInt(r.ctx) !== s.max_context) bad.push('serving.max_context ' + s.max_context + ' != ctx ' + r.ctx);
    ['max_context', 'batch', 'batch_groups', 'slot_context', 'big_slots', 'kv_resident', 'mtp_window'].concat(['max_t', 'cpu_pool_max_rows'].filter(k => k in D)).forEach(k => { if (!Number.isInteger(out[k]) || out[k] < 0) bad.push('serving.' + k + ' ' + JSON.stringify(out[k]) + ' is not a whole number >= 0'); });
    if (out.batch_mtp === null) out.batch_mtp = !!r.draft;
    Object.keys(s).filter(k => !(k in D) && ['kv_host_reserve_gib', 'mem_available_gib', 'mem_available_src', 'host_reserve_gib', 'host_experts_gib', 'max_startup_tokens'].indexOf(k) < 0).forEach(k => bad.push('serving.' + k + ' is not a serving field (' + Object.keys(D).join(', ') + ')'));
    return { s: out, S, bad };
  }
  // KV streaming pins a host copy of each session's whole KV capacity (XeStrata --kv-resident), never reclaimed. Sessions: the
  // main one (always --max-context: every prompt is read there), the first big_slots batch slots at --max-context, the rest
  // at --slot-context; a drafter per session that drafts (the main one with a draft, every slot with batch_mtp) copies its
  // own layer while its window is shorter than its context. Pinned ~= bytes_per_cell x (max_context x (1 + K) + slot_context
  // x (batch - K)); the engine refuses to start when that would leave under STRATA_KV_HOST_RESERVE_GIB of MemAvailable
  function kvHostPlan(data, r) {
    const sv = servingOf(data, r), m = data.models[r.model], K = m && m.kv_host;
    if (!sv || !K) return { sv, why: !sv ? null : 'models.' + r.model + '.kv_host not given (layers, bytes per cell)' };
    const { s, S } = sv, page = v(S.page_cells), round = c => Math.ceil(c / page) * page;
    const bpc = K.bytes_per_cell_layer[s.kv], dfmt = v(K.draft_format)[s.kv] || s.kv, dpc = K.bytes_per_cell_layer[dfmt];
    if (!bpc || !dpc) return { sv, why: '--kv ' + s.kv + ' has no bytes per cell in models.' + r.model + '.kv_host (' + Object.keys(K.bytes_per_cell_layer).join(', ') + ')' };
    const slot = b => s.slot_context > 0 && b >= s.big_slots ? Math.min(s.slot_context, s.max_context) : s.max_context;
    // main sessions: one per row group when the engine keeps one a group (engines.<engine>.serving.main_session_per_group)
    const mains = s.batch > 0 && S.main_session_per_group && v(S.main_session_per_group) ? Math.max(1, s.batch_groups) : 1;
    const cells = Array(mains).fill(s.max_context).concat(Array.from({ length: s.batch }, (_, b) => slot(b))).map(round);
    const over = S.copy_overhead_mib ? v(S.copy_overhead_mib) * 2 ** 20 : 0;
    const ring = round(Math.max(s.kv_resident, v(S.kv_resident_min)));
    const streamed = c => s.kv_resident > 0 && ring < c;
    const drafts = cells.map((c, i) => (i === 0 ? !!r.draft : s.batch_mtp) && s.kv_resident > 0 && s.mtp_window > 0 && s.mtp_window < c);
    // each streamed session pins one host copy per full-attention layer (its drafter one per draft layer), each copy its cells
    // plus the engine's per-copy overhead
    const main = cells.reduce((a, c) => a + (streamed(c) ? c * v(bpc) + over : 0), 0) * v(K.layers) / GiB;
    const draft = cells.reduce((a, c, i) => a + (drafts[i] ? c * v(dpc) + over : 0), 0) * v(K.draft_layers) / GiB;
    const ringGiB = cells.reduce((a, c) => a + Math.min(c, ring), 0) * v(K.layers) * v(bpc) / GiB;
    const copies = cells.filter(streamed).length * v(K.layers) + drafts.filter(Boolean).length * v(K.draft_layers);
    return { sv, cells, mains, ring, streamed: cells.filter(streamed).length, copies, overhead_mib: over / 2 ** 20, main_gib: main, draft_gib: draft, pinned_gib: main + draft, vram_ring_gib: ringGiB, bpc: v(bpc), layers: v(K.layers) };
  }
  function kvHostCheck(data, r) {
    const C = data.checks.kv_host, p = kvHostPlan(data, r), box = data.boxes[r.box];
    if (!p.sv) return null;
    const out = (status, detail) => ({ id: 'kv_host', label: C.label, status, detail, pred: p.pinned_gib === undefined ? null : { pinned_gib: p.pinned_gib, main_gib: p.main_gib, draft_gib: p.draft_gib, vram_ring_gib: p.vram_ring_gib } });
    const s = p.sv.s, sr = r.serving || {};
    if (p.why) return out('UNKNOWN', p.why);
    const shape = (s.batch ? 'main + ' + s.batch + ' slots' + (s.slot_context > 0 && s.batch > s.big_slots ? ' (' + Math.min(s.big_slots, s.batch) + ' at ' + s.max_context + ', ' + (s.batch - s.big_slots) + ' at ' + Math.min(s.slot_context, s.max_context) + ')' : ' at ' + s.max_context) : 'the main session at ' + s.max_context) + ' cells, --kv ' + s.kv;
    if (!p.streamed) return out('PASS', 'no pinned host copy: ' + (s.kv_resident > 0 ? 'every session fits its VRAM ring of ' + p.ring + ' cells' : 'kv_resident 0 keeps every cell in VRAM') + ' (' + shape + ')');
    const own = sr.kv_host_reserve_gib !== undefined, res = own ? sr.kv_host_reserve_gib : v(C.reserve_gib);
    const dram = box && box.dram, decl = sr.mem_available_gib !== undefined;
    const avail = decl ? sr.mem_available_gib : dram && dram.host_floor_gib ? v(dram.total_gib) - v(dram.host_floor_gib) : null;
    const how = p.layers + ' layers x ' + p.bpc + ' B/cell = ' + (p.main_gib).toFixed(2) + ' GiB' + (p.draft_gib ? ' + drafter copies ' + p.draft_gib.toFixed(2) : '') + ' pinned for ' + shape + ', ' + p.streamed + ' sessions streamed; VRAM rings ' + p.vram_ring_gib.toFixed(2) + ' GiB (' + p.ring + ' cells each)';
    const resWhy = 'reserve ' + res + ' GiB (' + (own ? 'serving.kv_host_reserve_gib' : 'checks.kv_host.reserve_gib, the engine default') + ')';
    if (avail === null) return out('UNKNOWN', how + '; no MemAvailable figure: give serving.mem_available_gib (box ' + r.box + ' has no dram.host_floor_gib); ' + resWhy);
    const left = avail - p.pinned_gib, availWhy = decl ? 'serving.mem_available_gib ' + avail + (sr.mem_available_src ? ' (' + sr.mem_available_src + ')' : '') : 'MemAvailable ' + avail.toFixed(1) + ' GiB ESTIMATE (' + v(C.available_basis) + ': ' + v(dram.total_gib) + ' - ' + v(dram.host_floor_gib) + ')';
    return out(left < res ? 'FAIL' : 'PASS', how + '; ' + availWhy + ' leaves ' + left.toFixed(1) + ' GiB against the ' + resWhy + (left < res ? ': the engine refuses to start (' + v(C.event) + ')' : ''));
  }
  // every host-pinned buffer of a serving engine: the KV host copies (kvHostPlan) plus the per model/engine terms of
  // checks.host_pinned (expert complement, the stage-lend duplicate of a layer split unless serving.stage_lend_drop, prefill
  // staging, prompt hand-off, embeddings, the rest). The driver's GTT sits outside every cgroup and the OOM killer, so it must fit
  // under DRAM total - household floor - pinned margin by itself
  function hostPinnedPlan(data, r) {
    const C = data.checks.host_pinned, kv = kvHostPlan(data, r), T = C && C.terms[r.model + '/' + r.engine];
    // a recipe without a serving block (a ledger row of an older layout) is not judged: its host buffers are other terms
    if (!kv.sv || !r.serving) return null;
    if (kv.why || !T) return { kv, why: kv.why || 'checks.host_pinned.terms has no ' + r.model + '/' + r.engine + ' entry (the engine\'s host-pinned buffers besides the KV copies)' };
    const s = kv.sv.s, sr = r.serving || {}, banks = ((r.gpu_split || {}).banks_gib || []).length;
    const lend = banks > 1 && !s.stage_lend_drop ? v(T.stage_lend_gib) : 0;
    const terms = [
      { id: 'kv_copies', gib: kv.pinned_gib, why: kv.copies + ' KV host copies (' + kv.mains + ' main session' + (kv.mains > 1 ? 's' : '') + ' + slots, ' + kv.layers + ' layers each, drafters apart) + ' + kv.overhead_mib + ' MiB a copy' },
      { id: 'experts_offgpu', gib: sr.host_experts_gib !== undefined ? sr.host_experts_gib : v(T.experts_offgpu_gib), why: sr.host_experts_gib !== undefined ? 'serving.host_experts_gib' : T.experts_offgpu_gib.tag },
      { id: 'stage_lend', gib: lend, why: banks > 1 ? (s.stage_lend_drop ? 'dropped (serving.stage_lend_drop)' : T.stage_lend_gib.tag + '; serving.stage_lend_drop removes it') : 'no layer split' },
    ].concat(['prefill_staging_gib', 'prompt_handoff_gib', 'embeddings_gib', 'misc_gib'].filter(k => T[k]).map(k => ({ id: k.replace(/_gib$/, ''), gib: v(T[k]), why: T[k].tag })));
    return { kv, terms, pinned_gib: terms.reduce((a, x) => a + x.gib, 0) };
  }
  function hostPinnedBudget(data, r) {
    const dram = (data.boxes[r.box] || {}).dram, sr = r.serving || {}, C = data.checks.host_pinned;
    const own = sr.host_reserve_gib !== undefined, floor = dram && dram.host_floor_gib ? v(dram.host_floor_gib) : null;
    const margin = dram && dram.pinned_margin_gib ? v(dram.pinned_margin_gib) : C ? v(C.default_margin_gib) : null;
    const reserve = own ? sr.host_reserve_gib : floor !== null && margin !== null ? floor + margin : null;
    return { own, floor, margin, reserve_gib: reserve, ceiling_gib: dram && reserve !== null ? v(dram.total_gib) - reserve : null };
  }
  // Resident experts share the existing host/KV budget, not a total-RAM fit shortcut. available_gib is a
  // declared pre-load MemAvailable (free + reclaimable after services); its explicit reserve_gib is additional
  // headroom, including any allocatability discount that measurement did not exclude. The independent total
  // ceiling stays hostPinnedBudget's containment law; never split its composite floor into guessed current/future use.
  // other_gib includes staging/non-expert allocations not already charged by that pre-load measurement.
  // Missing/historical inputs stay unverified.
  function expertResidencyCheck(data, r) {
    const p = r.expert_residency;
    if (!p) return null;
    const box = data.boxes[r.box], C = data.checks.expert_residency, budget = hostPinnedBudget(data, r);
    const total = box && box.dram && v(box.dram.total_gib), kv = kvHostPlan(data, r);
    const n = x => x !== null && x !== undefined && Number.isFinite(v(x)) && v(x) >= 0;
    const required = n(p.host_experts_gib) ? v(p.host_experts_gib) : null;
    const available = n(p.available_gib) ? v(p.available_gib) : null;
    const other = n(p.other_gib) ? v(p.other_gib) : null;
    const pinned = p.kv_gib !== undefined ? n(p.kv_gib) ? v(p.kv_gib) : null : kv.pinned_gib === undefined ? null : kv.pinned_gib;
    const reserve = n(p.reserve_gib) ? v(p.reserve_gib) : null;
    const usable = [total, available, other, pinned, reserve, budget.reserve_gib].every(n) && Number.isFinite(budget.ceiling_gib)
      ? Math.min(budget.ceiling_gib, available - reserve) - other - pinned : null;
    const control = C && C.controls[r.engine], mode = p.mode || 'unknown';
    const fit = required !== null && usable !== null ? required <= usable : null;
    const invalid = ['host_experts_gib', 'available_gib', 'other_gib', 'kv_gib', 'reserve_gib'].filter(k => p[k] !== undefined && v(p[k]) !== null && !n(p[k]));
    const live = p.live_verified === undefined ? !!(C && C.default_live_verified) : p.live_verified === true;
    const sourced = p.src && ['host_experts_gib', 'available_gib', 'other_gib', 'kv_gib', 'reserve_gib'].filter(k => p[k] !== undefined)
      .every(k => p[k] && p[k].src && evidenceOf(data, p[k]) !== 'unknown');
    const status = invalid.length ? 'FAIL' : fit === null || !live || !sourced ? 'UNKNOWN' : fit ? 'PASS' : 'FAIL';
    const matching = control && (!control.model || control.model === r.model) && (!control.engine_build || control.engine_build === r.engine_build);
    const avoid = status === 'PASS' && ['mmap', 'auto'].includes(mode) && matching && control.resident.length ? control.resident : null;
    return { id: 'expert_residency', label: C ? C.label : 'resident experts fit usable RAM', status,
      pred: { host_experts_gib: required, total_gib: total, available_gib: available, other_gib: other, kv_gib: pinned, reserve_gib: reserve,
        total_reserve_gib: budget.reserve_gib, total_ceiling_gib: budget.ceiling_gib,
        usable_gib: usable, fits_declared_budget: fit, live_verified: p.live_verified === true, avoidable_streaming: !!avoid, resident_controls: avoid },
      detail: 'host experts ' + (required === null ? 'UNKNOWN' : required.toFixed(2)) + ' GiB; totalRAM ' + total +
        ' GiB is not usableRAM; usable = min(hostPinnedBudget total ceiling, pre-load MemAvailable - explicit additional reserve_gib) - KV - staging/other = ' +
        (usable === null ? 'UNKNOWN (declare the total containment budget and available_gib, other_gib, kv_gib and explicit additional reserve_gib with sources)' : usable.toFixed(2) + ' GiB') +
        '; total reserve ' + budget.reserve_gib + ' GiB, total ceiling ' + budget.ceiling_gib + ' GiB, additional available-budget reserve ' + reserve + ' GiB' +
        '; ' + (p.live_verified === true ? 'declared live budget (not probed by the planner)' : 'current live residency/RAM UNVERIFIED; inventory and retained source only') +
        (invalid.length ? '; invalid nonnegative GiB fields: ' + invalid.join(', ') : '') +
        (avoid ? '; avoidable SSD/USB expert streaming: matching engine controls ' + avoid.join(' ') : '') +
        (p.src ? '; ' + p.src : '; source UNKNOWN') };
  }
  // The recipe owns intent; measured.fallback_ledger owns receipts. Neither catalog capacity, an enabled flag,
  // a dispatch setup line nor absence of a warning proves a path ran. Rows preserve partial proof, not a green blanket.
  function fallbackCheck(data, r, measured) {
    const P = r.fallback_plan, L = measured && measured.fallback_ledger, C = data.checks.fallback;
    if (!P && !L) return null;
    if (!C) return { id: 'fallback', status: 'UNKNOWN', detail: 'fallback check catalog is unavailable' };
    const object = x => x && typeof x === 'object' && !Array.isArray(x);
    if ((P && (!object(P) || !object(P.classes) || Object.values(P.classes).some(p => !object(p)))) ||
        (L && (!object(L) || !Array.isArray(L.events) || L.events.some(e => !object(e)))))
      return { id: 'fallback', label: C.label, status: 'FAIL', detail: 'invalid fallback plan or receipt structure' };
    const rows = [], plans = P && P.classes || {}, box = data.boxes[r.box];
    const add = (id, status, detail, extra) => rows.push(Object.assign({ id, status, detail }, extra || {}));
    const source = x => x && typeof x.src === 'string' && x.src.trim() && !/^UNKNOWN\b/i.test(x.src.trim());
    const proved = x => source(x) && C.capacity_tags.includes(x.tag) && Number.isFinite(x.v) && x.v >= 0 &&
      evidenceOf(data, { tag: x.tag, src: x.src }) === 'seen';
    const family = t => typeof t === 'string' && /^gpu\d*$/.test(t) ? 'gpu' : t;
    const decision = x => x && source(x) && typeof x.reason === 'string' && x.reason.trim();
    const identity = ['box', 'model', 'engine', 'engine_build'];
    const bound = L && source(L) && identity.every(k => typeof r[k] === 'string' && r[k] && L.identity && L.identity[k] === r[k]);
    const ctx = context(data, r.box, r.model, r.engine, Object.assign({}, data.knobs, r.knobs || {}));
    const memory = new Set(ctx.tiers.filter(t => t.id !== 'pmem' || box.pmem).map(t => family(t.id)));
    // GPU presence alone is not engine capability. USB exists only when an actual box mount identifies it.
    if (!(box.gpus || []).some(g => (g.engines || []).includes(r.engine))) memory.delete('gpu');
    (box.disks || []).forEach(d => { if (C.disk_tiers[d.kind]) memory.add(C.disk_tiers[d.kind]); });
    const fills = Object.fromEntries(Object.entries(plans).filter(([id]) => C.classes[id] === 'memory')
      .map(([id, p]) => [id, { tier: p.tier, gib: p.gib === undefined ? null : v(p.gib) }]));
    const sameFills = b => b && b.fills && Object.keys(b.fills).length === Object.keys(fills).length &&
      Object.entries(fills).every(([id, x]) => b.fills[id] && x.tier === b.fills[id].tier && x.gib === b.fills[id].gib);
    if (!P || !source(P)) add('plan', 'UNKNOWN', 'intended paths lack a sourced fallback_plan');
    Object.entries(C.classes).forEach(([id, kind]) => {
      const p = plans[id], ladder = C.ladders[kind], tier = p && family(p.tier);
      if (!p || !source(p) || !p.tier || p.tier === 'unknown') { add(id + ':plan', 'UNKNOWN', 'intended ' + id + ' path is unproved'); return; }
      if (!ladder.includes(tier)) { add(id + ':plan', 'FAIL', 'invalid ' + id + ' tier ' + p.tier); return; }
      if (kind === 'memory' && !memory.has(tier)) {
        add(id + ':capability', 'UNKNOWN', p.tier + ' is not a proved ' + r.engine + ' tier on ' + r.box);
      }
      const cap = P.capabilities && P.capabilities[id];
      if (kind !== 'memory' && tier !== ladder[0] && !decision(p.decision)) {
        const faster = cap && source(cap) && (cap.tiers || []).filter(t => ladder.includes(t) && ladder.indexOf(t) < ladder.indexOf(tier));
        add(id + ':capacity', faster && faster.length ? 'FAIL' : 'UNKNOWN', faster && faster.length
          ? 'unchosen compute downgrade to ' + tier + '; supported faster paths ' + faster.join('/')
          : 'faster operation-specific capability is unproved; no inference from GPU model');
      }
      if (kind === 'memory' && ladder.indexOf(tier) > 0 && !decision(p.decision)) {
        const higher = ladder.slice(0, ladder.indexOf(tier)).filter(t => memory.has(t));
        higher.forEach(t => {
          const b = (P.headroom || {})[t], u = p.unit_gib;
          const eligible = b && b.live_verified === true && source(b) && sameFills(b) && proved(b.idle_gib) && proved(u) && u.v > 0 &&
            C.capacity_accounting.every(k => source((b.accounting || {})[k]));
          // The certificate is residual allocatable idle AFTER all fills, KV, staging, service/safety, CMA and
          // GTT discounts. Its exact fill snapshot prevents reuse after a plan change. No raw MemAvailable shortcut.
          const total = t === 'dram' ? hostPinnedBudget(data, r).ceiling_gib : null;
          const allocated = Object.values(fills).filter(x => family(x.tier) === t);
          const impossible = eligible && t === 'dram' && (!Number.isFinite(total) || allocated.some(x => !Number.isFinite(x.gib) || x.gib < 0) ||
            b.idle_gib.v + allocated.reduce((s, x) => s + x.gib, 0) > total);
          add(id + ':idle:' + t, !eligible || impossible ? 'UNKNOWN' : b.idle_gib.v >= u.v ? 'FAIL' : 'PASS',
            !eligible || impossible ? 'allocatable idle after all fills/reserves on ' + t + ' is UNPROVED'
              : b.idle_gib.v >= u.v ? 'unchosen ' + p.tier + ' placement leaves ' + b.idle_gib.v + ' GiB usable on ' + t
                : t + ' residual headroom is below the movable allocation unit');
        });
      }
      const events = bound ? (L.events || []).filter(e => e.class === id && source(e)) : [];
      const paths = events.filter(e => e.tier && e.proof === 'observed');
      const lower = events.filter(e => ['observed', 'configured'].includes(e.proof) && ladder.includes(family(e.tier)) && ladder.indexOf(family(e.tier)) > ladder.indexOf(tier));
      if (lower.length) add(id + ':run', 'FAIL', 'observed below plan ' + p.tier + ': ' + lower.map(e => e.tier + ' (' + e.src + ')').join('; '));
      else if (paths.some(e => family(e.tier) === tier)) add(id + ':run', 'PASS', 'observed planned path ' + p.tier);
      else add(id + ':run', 'UNKNOWN', (!bound ? 'no matching box/model/engine/build receipt' : 'no positive path receipt; setup/flags/quiet logs are not dispatch proof') +
        ((C.placement_controls[r.engine] || {})[id] ? '; verify ' + C.placement_controls[r.engine][id].join(', ') : ''));
      Object.entries(C.metrics).forEach(([metric, field]) => {
        if (p[field] === undefined) return;
        const limit = v(p[field]), hits = events.filter(e => e.metric === metric && Number.isFinite(e.value) && e.proof === 'observed');
        const bad = hits.filter(e => field.startsWith('min_') ? e.value < limit : e.value > limit);
        add(id + ':' + metric, !Number.isFinite(limit) ? 'FAIL' : !hits.length ? 'UNKNOWN' : bad.length ? 'FAIL' : 'PASS',
          !hits.length ? 'no measured ' + metric : metric + ' ' + hits.map(e => e.value).join('/') + ', plan ' + field + ' ' + limit);
      });
    });
    const flags = [(r.conditions || {}).engine_flags, (r.conditions || {}).engine_extra].filter(Boolean).join(' ');
    (C.controls || []).filter(c => c.engine === r.engine && (!c.model || c.model === r.model)).forEach(c => {
      const p = plans[c.class];
      if (!p || p.tier !== c.want) return;
      const match = c.pattern && new RegExp(c.pattern).exec(flags + ' ' + ((r.conditions || {}).draft_n || ''));
      const triggered = c.pattern ? !!match : !(c.requires || []).every(pattern => new RegExp(pattern).test(flags));
      if (!triggered) return;
      if (c.requires_layer_split) {
        // gpu_split is the existing per-stage bank plan; explicit null follows its single-GPU branch.
        // Retained split flags are configuration evidence too, never proof of executed dispatch.
        const split = C.layer_split_pattern && new RegExp(C.layer_split_pattern).exec(flags + ' ' + ((r.conditions || {}).isolation || ''));
        const banks = r.gpu_split && r.gpu_split.banks_gib;
        const declared = Array.isArray(banks) && banks.length ? banks.length : Object.prototype.hasOwnProperty.call(r, 'gpu_split') && r.gpu_split === null ? 1 : null;
        const stages = [declared, split ? split[1].split(',').length + 1 : null].filter(x => x !== null);
        if (!stages.length || !stages.every(x => x === stages[0])) {
          add('control:' + c.id, 'UNKNOWN', 'configured layer split is missing or contradictory; build/group flags alone do not establish this rule');
          return;
        }
        if (stages[0] === 1) return;
        const s = r.serving || {}, group = match && match[1];
        // For auto, an explicit serving count must demonstrate a divisor available on the declared stages.
        // Otherwise retain UNKNOWN; do not invent an engine resolver from box hardware or defaults.
        if (group === 'auto' && !(Number.isInteger(s.batch_groups) && s.batch_groups > 1 && s.batch_groups <= stages[0] &&
            Number.isInteger(s.batch) && s.batch > 0 && s.batch % s.batch_groups === 0)) {
          add('control:' + c.id, 'UNKNOWN', 'auto effective groups > 1 are unproved by the declared serving shape');
          return;
        }
      }
      add('control:' + c.id, c.builds.includes(r.engine_build) ? 'FAIL' : 'UNKNOWN', c.src + '; configured intent applicability ' + r.engine_build);
    });
    const sv = servingOf(data, r);
    if (plans.batch_mtp && plans.batch_mtp.tier === 'on' && sv && sv.s.batch > 0 && sv.s.batch_groups > 1 && !decision(plans.batch_mtp.decision))
      add('control:groups-mtp', 'UNKNOWN', 'grouped pipeline may disable batch-MTP on this build; require its engine receipt or an explicit chosen off plan');
    if (plans.hot_experts && sv && sv.s.batch > sv.s.big_slots && sv.s.max_context > 0) {
      const effective = sv.s.slot_context > 0 ? Math.min(sv.s.slot_context, sv.s.max_context) : sv.s.max_context;
      const limit = v(plans.hot_experts.max_regular_slot_context);
      if (Number.isFinite(limit)) add('control:slot-context', effective > limit ? 'FAIL' : 'PASS',
        'regular slots reserve ' + effective + ' cells each; planned ceiling ' + limit + '; big slots remain separately charged by kvHostPlan');
      else if (!sv.s.slot_context) add('control:slot-context', 'UNKNOWN',
        'every slot inherits MAX_CONTEXT; cache capacity requires per-slot allocation and engine slot receipts');
    }
    return { id: 'fallback', label: C.label, status: rows.some(x => x.status === 'FAIL') ? 'FAIL' : rows.some(x => x.status === 'UNKNOWN') ? 'UNKNOWN' : 'PASS',
      detail: (r.fallback_plan_basis ? r.fallback_plan_basis + '; ' : '') + rows.map(x => x.id + ' ' + x.status + ': ' + x.detail).join('; '),
      pred: { rows, identity_matched: !!bound, live_capacity_inferred: false } };
  }
  function hostPinnedCheck(data, r) {
    const C = data.checks.host_pinned, p = C ? hostPinnedPlan(data, r) : null, box = data.boxes[r.box];
    if (!p) return null;
    const out = (status, detail) => ({ id: 'host_pinned', label: C.label, status, detail, pred: p.pinned_gib === undefined ? null : { pinned_gib: p.pinned_gib, terms: Object.fromEntries(p.terms.map(x => [x.id, x.gib])) } });
    if (p.why) return out('UNKNOWN', p.why);
    const sr = r.serving || {}, dram = box && box.dram, list = p.terms.filter(x => x.gib > 0).map(x => x.id + ' ' + x.gib.toFixed(2) + ' (' + x.why + ')').join(' + ');
    const budget = hostPinnedBudget(data, r), { own, floor, margin } = budget;
    if (!own && floor === null) return out('UNKNOWN', 'pinned ' + p.pinned_gib.toFixed(2) + ' GiB = ' + list + '; box ' + r.box + ' has no dram.host_floor_gib: give serving.host_reserve_gib');
    const res = budget.reserve_gib, top = budget.ceiling_gib;
    const resWhy = own ? 'serving.host_reserve_gib ' + res : '(household floor ' + floor + ' + pinned margin ' + margin + (dram.pinned_margin_gib ? '' : ', checks.host_pinned.default_margin_gib') + ')';
    const seen = sr.mem_available_gib !== undefined ? '; declared MemAvailable ' + sr.mem_available_gib + ' GiB' + (sr.mem_available_src ? ' (' + sr.mem_available_src + ')' : '') + ' means ' + (v(dram.total_gib) - p.pinned_gib - sr.mem_available_gib).toFixed(1) + ' GiB of other host use beside the pinned set' : '';
    const fail = p.pinned_gib > top;
    return out(fail ? 'FAIL' : 'PASS', 'pinned ' + p.pinned_gib.toFixed(2) + ' GiB = ' + list + '; ceiling ' + top.toFixed(1) + ' GiB (' + v(dram.total_gib) + ' total - ' + resWhy + ')' + seen +
      (fail ? ': over by ' + (p.pinned_gib - top).toFixed(1) + ' GiB (' + v(C.event) + '); levers: serving.slot_context near the VRAM ring with serving.big_slots, serving.stage_lend_drop' : ''));
  }
  function checksFor(ctx, lay, ev) { return runChecks(ctx.data, layoutFacts(ctx, lay, ev || evaluate(ctx, lay))); }
  // rows of data.recipes: a run-ledger row ({recipe, observed}) is checked through checkRecipe, then the run's own observations
  // (node MemFree, measured KL) overlay the prediction; legacy fact rows go straight to runChecks
  function checkRecipes(data) {
    return (data.recipes || []).map(r => {
      if (!r.recipe) return Object.assign({}, r, { checks: runChecks(data, r) });
      const res = checkRecipe(data, r.recipe, null, { predict: false, measured: r.observed });
      if (res.bad_input || ['dllm', 'network'].includes(r.recipe.kind)) return Object.assign({}, r, { checks: res.checks });
      const o = r.observed || {};
      const f = Object.assign({}, res.facts, { numa: Object.assign({}, res.facts.numa || {}, o.numa || {}) });
      if (o.decode_kl !== undefined && o.decode_kl !== null) f.decode_kl = o.decode_kl;
      const inp = res.checks.filter(c => (c.id === 'inputs' && c.status !== 'PASS') || c.id === 'expert_residency' || c.id === 'fallback' || c.id === 'cpu_pool_rows' || c.id === 'earf');
      // a serving run keeps its host_pinned check with what it measured beside the prediction (end of run, or a guard stop's lower bound)
      const hp = res.checks.find(c => c.id === 'host_pinned'), end = o.pinned_gib, stop = o.pinned_at_stop_gib;
      const seen = !hp || !hp.pred ? '' : end !== undefined ? '; MEASURED pinned ' + end + ' GiB at the run end (predicted ' + hp.pred.pinned_gib.toFixed(2) + ', ' + (hp.pred.pinned_gib >= end ? '+' : '') + (100 * (hp.pred.pinned_gib / end - 1)).toFixed(1) + ' %)'
        : stop !== undefined ? '; MEASURED pinned ' + stop + ' GiB where a guard stopped the run (a lower bound; predicted ' + hp.pred.pinned_gib.toFixed(2) + ')' : '';
      const host = hp ? [Object.assign({}, hp, { detail: hp.detail + seen, measured: end !== undefined ? { pinned_gib: end } : stop !== undefined ? { pinned_at_stop_gib: stop } : null })] : [];
      return Object.assign({}, r, { checks: inp.concat(runChecks(data, f), host) });
    });
  }

  // the hot-expert map citations behind a heat set ({map_id, digest, slice}; a sum set cites each part); [] for a set with no map
  function heatCite(m, name) {
    const set = m.heat.sets[name];
    if (!set) return [];
    if (set.sum) return [].concat.apply([], set.sum.map(n => heatCite(m, n)));
    return set.map ? [set.map] : [];
  }
  // per-expert types of a recipe's store on the planner's expert grid: { type, qi, note } or { why }. predictRecipe places
  // these experts; checkRecipe's quality check reads its 2-bit share from the same layout (recipeLowShare)
  function storeTypes(data, r, ctx) {
    const m = data.models[r.model], st = r.store || {}, N = m.L * m.E, Ex = m.E;
    const tq = q => (r.rtr && q === 'mxfp4' && ctx.types.indexOf('mxfp4_r8') >= 0) ? 'mxfp4_r8' : q;
    const qi = q => ctx.types.indexOf(tq(q));
    const named = (st.types || []).concat(Object.keys(st.read_share_by_type || {}), (st.layer_types || []).map(x => x.type), st.hot_type || [], st.cold_type || []);
    const unknown = named.filter(q => !data.types[q]);
    if (unknown.length) return { why: 'type ' + [...new Set(unknown)].join(', ') + ' is not in the type table' };
    const type = new Int8Array(N);
    const need = [];
    const setAll = q => { need.push(q); type.fill(qi(q)); };
    let note;
    if (st.hot_type && st.cold_type) {
      const hs = data[st.hot_set || 'alloc3'];
      if (!hs) return { why: 'store hot set ' + (st.hot_set || 'alloc3') + ' is not in the data' };
      setAll(st.cold_type); need.push(st.hot_type);
      hs.layers.forEach((ids, l) => ids.forEach(e => type[l * Ex + e] = qi(st.hot_type)));
      note = 'types: the ' + hs.count + ' experts of ' + (st.hot_set || 'alloc3') + ' at ' + tq(st.hot_type) + ', the rest ' + st.cold_type;
    } else if (st.layer_types) {
      setAll(st.types[0]);
      const ids = m.layer_ids || Array.from({ length: m.L }, (_, l) => l);
      st.layer_types.forEach(x => { need.push(x.type); ids.forEach((id, l) => { if (id >= x.layers[0] && id <= x.layers[1]) for (let e = 0; e < Ex; e++) type[l * Ex + e] = qi(x.type); }); });
      note = 'types: ' + st.layer_types.map(x => 'layers ' + x.layers.join('-') + ' ' + x.type).join(', ') + ', the rest ' + st.types[0];
    } else if (st.read_share_by_type && Object.keys(st.read_share_by_type).length === 2) {
      const qs = Object.keys(st.read_share_by_type).sort((a, b) => v(data.types[a].bpw) - v(data.types[b].bpw));
      const lo = qs[0], target = st.read_share_by_type[lo];
      setAll(qs[1]); need.push(lo);
      let s = 0;
      for (let i = N - 1; i >= 0 && s < target; i--) { const e = ctx.order[i]; type[e] = qi(lo); s += ctx.plan[e] / ctx.kL; }
      note = 'types: coldest experts (' + ctx.planName + ') at ' + lo + ' until ' + (100 * target).toFixed(1) + ' % of routed reads, the rest ' + qs[1];
    } else if ((st.types || []).length === 1) {
      setAll(st.types[0]);
      note = 'types: every routed expert ' + tq(st.types[0]);
    } else return { why: 'the store gives no per-expert type rule (hot/cold split, layer types, a two-type read share, or one type)' };
    const miss = need.filter(q => qi(q) < 0);
    if (miss.length) return { why: 'type ' + miss.join(', ') + ' is not in the planner type set for ' + r.engine + ' on ' + r.model };
    return { type, qi, note };
  }
  // the planner context predictRecipe builds for a recipe (knobs, heat); null when the engine has no planner model
  function recipeContext(data, r) {
    const m = data.models[r.model], E = data.engines[r.engine];
    if (E.no_solver) return null;
    const nf = r.numa || {};
    const kn = Object.assign({}, data.knobs, r.knobs || {}, { ub: r.ub !== undefined ? String(r.ub) : 'auto', draft: !!r.draft, rtr: !!r.rtr,
      perExpertTypes: r.engine === 'ik' && !!(r.store || {}).split, numaHeadroom: 0, threads: r.threads }, storeKnobs(r));
    if (nf.headroom_gib !== undefined && nf.headroom_gib !== null) kn.numaHeadroom = nf.headroom_gib;
    if (nf.layout) kn.numaLayout = nf.layout;
    return { kn, ctx: prepare(context(data, r.box, r.model, r.engine, kn), (r.heat && r.heat.plan) || m.heat.plan_default, (r.heat && r.heat.score) || m.heat.score_default) };
  }
  // projected top-1 agreement of a ledger recipe's stored weights: its store type rule laid out on the score heat, read-weighted
  // KLD (layer weights as in evaluate) plus the dense part; the chart places a measured run at this height,
  // projected on the same scale as the curves (a measured top-1 is against its own reference, so it stays in the tooltip).
  // Falls back to the first named type's KLD when the store rule cannot be laid out; null for a model with no KLD family.
  function recipeQuality(data, r) {
    const st = r.store || {}, m = data.models[r.model];
    if (!m || !m.kld_family) return null;
    const rc = data.engines[r.engine] && recipeContext(data, r);
    const ty = rc ? storeTypes(data, r, rc.ctx) : { why: r.engine + ' has no planner model' };
    if (!ty.why) {
      const ctx = rc.ctx, w = layerWeights(data, r.model), kl = ctx.types.map(q => typeKld(data, r.model, q));
      let k = 0;
      for (let e = 0; e < ty.type.length; e++) k += ctx.score[e] * kl[ty.type[e]] * (w ? w[(e / m.E) | 0] : 1);
      const kld = k / ctx.kL + ctx.kldDense;
      return { quality: qualityOf(data, kld), kld, how: ty.note };
    }
    const q = (st.types || []).concat(st.hot_type || []).find(x => data.types[x] && typeKld(data, r.model, x) !== null && typeKld(data, r.model, x) !== undefined) || m.base_type;
    const kld = typeKld(data, r.model, q) || 0;
    return { quality: qualityOf(data, kld), kld, how: 'every routed expert ' + q + ' (' + ty.why + ')' };
  }
  // 2-bit share of routed reads for the quality check: the recipe's own figure if it gives one (lowbit_share, read_share_by_type),
  // else the share of the layout predictRecipe would place (storeTypes on the score heat). 0 when no store type is 2-bit;
  // { share: null, why } when a 2-bit type is named but the layout cannot be built
  function recipeLowShare(data, r) {
    const st = r.store || {};
    if (r.lowbit_share !== undefined && r.lowbit_share !== null) return { share: r.lowbit_share };
    if (st.read_share_by_type) return { share: Object.keys(st.read_share_by_type).filter(q => data.types[q] && isLow(data, q)).reduce((a, q) => a + st.read_share_by_type[q], 0) };
    const named = (st.types || []).concat((st.layer_types || []).map(x => x.type), st.hot_type || [], st.cold_type || []);
    if (!named.some(q => data.types[q] && isLow(data, q))) return { share: 0 };
    const rc = recipeContext(data, r);
    if (!rc) return { share: null, why: r.engine + ' has no planner model to lay out the types' };
    const ty = storeTypes(data, r, rc.ctx);
    if (ty.why) return { share: null, why: ty.why };
    return { share: lowShare(rc.ctx, { type: ty.type }), how: 'layout ' + ty.note + ', reads scored on ' + rc.ctx.scoreName };
  }

  // ---------- prefill: CPU path vs GPU expert offload, one model for every engine ----------
  // An offloaded ubatch uploads the CPU-resident expert bytes it needs and the GPU compute overlaps the copy, so
  // offload tok/s = P / (sum over the ubatches of upload bytes / h2d rate). ik stages every CPU-resident expert per ubatch
  // (b1017off2: ~220 GiB per ubatch); mainline's scheduler copies only the experts a ubatch routes to (engine rate
  // upload_touched), 1 - (1 - p)^n of each expert's bytes for n tokens.
  // h2d rate: the engine's upload_gbs (MEASURED end to end on odie SYCL, x nohost_mult with --no-host), else
  // data.prefill.h2d_gbs[ring|driver]. Offload: mainline unless op_offload is false, from offload_min_batch tokens; ik when
  // offload_batch <= ub and the ubatch reaches it. CPU path: data.prefill.cpu_tps[box|model|engine|ub] inside its measured envelope
  // (PMem expert GiB), x checks.rtr.prefill_mult for -rtr on a split model (the B0 pair); else the engine's cpu_prefill_tps;
  // else UNMODELLED (null, the reason in cpu_src).
  const engRate = (data, r, k) => { const R = ((data.engines || {})[r.engine] || {}).recipe_rates || {}; return R[k] ? v(R[k]) : null; };
  const offloadOn = (data, r) => engRate(data, r, 'upload_gbs') ? r.op_offload !== false
    : r.offload_batch !== undefined && r.offload_batch !== null && r.offload_batch <= (Number(r.ub) || 512);
  // ik's upload buffers sit beside the compute buffer (MEASURED); mainline's are inside its measured compute buffer
  // GPU prefill offload: the upload buffers, plus (model|engine runtime) the extra compute graph F + s x ub over the offload-off buffer
  const offloadVram = (data, r) => {
    if (engRate(data, r, 'upload_gbs') || !offloadOn(data, r) || !data.prefill) return 0;
    const RT = runtimeOf(data, r.model, r.engine), ub = Number(r.ub) > 0 ? Number(r.ub) : 512;
    const F = RT && (r.mtp_experts === 'cpu' && RT.offload_compute_fixed_gib_mtp_cpu || RT.offload_compute_fixed_gib);
    return v(data.prefill.vram_gib) + (RT && RT.offload_compute_gib_per_tok ? Math.max(0, (F ? v(F) : 0) + v(RT.offload_compute_gib_per_tok) * ub) : 0);
  };
  function prefillModel(data, r, cpuGiB, touched, pmemGiB) {
    const P = data.prefill, eu = engRate(data, r, 'upload_gbs');
    if (!P && !eu) return null;
    const ub = Number(r.ub) > 0 ? Number(r.ub) : 512, key = r.box + '|' + r.model + '|' + r.engine + '|' + ub, c = P && P.cpu_tps[key];
    const h2d = eu ? (r.no_host ? 'engine --no-host' : 'engine') : (P.ring_builds || []).indexOf(r.engine_build) >= 0 ? 'ring' : 'driver';
    const bw = eu ? eu * (r.no_host ? engRate(data, r, 'nohost_mult') : 1) : v(P.h2d_gbs[h2d]);
    // a cell with banks_on: banks 0 reads the cell, banks inside banks_on.banks_gib read its FIT rate, other banks are UNMODELLED
    const env = c && c.envelope ? c.envelope.pmem_gib : null, pm = Math.round((pmemGiB || 0) * 10) / 10, split = !!(r.store && r.store.split);
    const bk = Number(r.banks_gib) || 0, bo = c && c.banks_on && bk > 0 ? c.banks_on : null, cell = bo || c;
    const bad = !c ? null : env && (pm < env[0] || pm > env[1]) ? 'PMem experts ' + pm.toFixed(1) + ' GiB outside the measured ' + env[0] + '-' + env[1] + ' GiB of ' + key + ' (' + c.envelope.src + ')'
      : bo && (bk < bo.banks_gib[0] || bk > bo.banks_gib[1]) ? 'banks ' + bk + ' GiB outside the measured ' + bo.banks_gib.join('-') + ' GiB of ' + key
      : r.rtr && !split ? '-rtr on a one-chain model: only the split pair (B0) is measured' : null;
    const unm = 'UNMODELLED: ' + (bad || 'no MEASURED CPU prefill cell ' + key + ' (cells: ' + Object.keys((P || {}).cpu_tps || {}).join(', ') + ')');
    const cpu = c ? (bad ? null : v(cell) * (r.rtr ? v(data.checks.rtr.prefill_mult) : 1)) : engRate(data, r, 'cpu_prefill_tps');
    const part = !!(touched && engRate(data, r, 'upload_touched'));
    const bytes = n => part ? touched(n) : cpuGiB * GiB;
    const up = bytes(ub) / 1e9 / bw;
    const on = offloadOn(data, r), thr = eu ? engRate(data, r, 'offload_min_batch') || 1 : r.offload_batch;
    const off = t => { let s = 0; for (let d = 0; d < t; d += ub) s += bytes(Math.min(ub, t - d)) / 1e9 / bw; return t / s; };
    const tps = t => (on && Math.min(t, ub) >= thr) ? off(t) : cpu;
    const at = {};
    ((P && P.prompts) || []).concat([2048, 8192, 32768]).forEach(t => { at[t] = { cpu_tps: cpu, offload_tps: off(t) }; });
    return { key, cpu_tps: cpu, cpu_band: c && cpu !== null && !bo ? [c.lo, c.hi] : null, cpu_tag: c && cpu !== null ? cell.tag + (r.rtr ? ' x rtr ' + v(data.checks.rtr.prefill_mult) : '') : cpu !== null ? r.engine + ' cpu_prefill_tps' : null,
      cpu_src: c && cpu !== null ? cell.src : unm, h2d, h2d_gbs: bw, upload_gib: bytes(ub) / GiB, upload_touched: part,
      upload_s_per_ubatch: up, ub, offload_on: on, offload_batch: on ? thr : null, offload_max_tps: ub / up,
      crossover_tokens: cpu !== null && cpu * up <= ub ? cpu * up : null, tps, at };
  }

  // ---------- external rows: our physics on hardware we do not own (data.external, runs/external.jsonl) ----------
  // Each row is a community measurement (MEASURED-BY-THEM = inferred). The prediction uses only our own terms: GPU nameplate x
  // gpu_eta, DRAM spec x the spec->read ratio of our boxes, the CPU kernel table x the median fitted f, and a borrowed host
  // overhead. Rows validate the physics; nothing is fitted on them.
  const med = xs => { const a = xs.filter(x => x !== null && x !== undefined && isFinite(x)).sort((x, y) => x - y), k = a.length; return k ? (k % 2 ? a[(k - 1) / 2] : (a[k / 2 - 1] + a[k / 2]) / 2) : null; };
  function externalDerived(data) {
    if (!data.cal) applyCal(data);
    const bx = Object.keys(data.boxes).filter(k => { const d = data.boxes[k].dram; return d.bw_peak_gbs && v(d.bw_peak_gbs) && d.bw && evidenceOf(data, d.bw) === 'seen'; });
    const ratio = bx.map(k => v(data.boxes[k].dram.bw) / v(data.boxes[k].dram.bw_peak_gbs));
    const fk = Object.keys((data.cal && data.cal.boxes) || {}).filter(k => data.cal.boxes[k].f && data.cal.boxes[k].f.tag === 'FIT');
    return {
      dram_eta: { v: med(ratio), tag: 'INFERRED-BY-US', src: 'median measured DRAM read / spec over ' + bx.map((k, i) => k + ' ' + ratio[i].toFixed(2)).join(', ') + ' (' + data.external.dram_eta_from.v + ')' },
      cpu_f: { v: med(fk.map(k => v(data.cal.boxes[k].f))), tag: 'FIT', src: 'median of the fitted CPU factor f over ' + fk.map(k => k + ' ' + v(data.cal.boxes[k].f).toFixed(2)).join(', ') }
    };
  }
  function quantType(data, label) {
    const Q = data.external.quant_types.v, u = String(label || '').toUpperCase();
    const k = Object.keys(Q).filter(p => u.startsWith(p)).sort((a, b) => b.length - a.length)[0];
    return k ? Q[k] : null;
  }
  // the measured NUMA gain (bench.numa, over one node read locally) for a policy over N nodes on S sockets, threads on the
  // same nodes; the arm with the most threads wins. null when no box measured that placement
  function numaGain(data, policy, nodes, sockets) {
    const pre = policy + ' m' + nodes + ' c' + nodes + ' s' + sockets + ' t';
    const hits = [];
    Object.keys((data.bench && data.bench.numa) || {}).forEach(b => Object.keys(data.bench.numa[b]).filter(k => k.startsWith(pre)).forEach(k => hits.push({ key: b + ' ' + k, t: +k.slice(pre.length), g: data.bench.numa[b][k] })));
    return hits.sort((a, b) => b.t - a.t)[0] || null;
  }
  function predictExternal(data, row) {
    const X = data.external, hw = row.hardware || {}, mp = row.model_params || {}, st = row.settings || {}, meas = (row.measured || {}).decode_tps;
    const T = (n, x) => term(data, n, x), terms = [], their = src => ({ tag: X.tag, src: src || row.url });
    const out = { id: row.id, hw_class: row.hw_class, status: 'UNMODELLED', why: null, decode_tps: null, measured: meas, err: null, placement: null, terms };
    const prefillUnknown = rollup('prefill', null, 'tok/s', [{ term: 'prefill rate', value: null, evidence: 'unknown', tag: 'UNKNOWN', src: 'no general prefill model: prefill has a rate only inside our measured cells (data.prefill.cpu_tps)' }]);
    const un = why => Object.assign(out, { why, evidence: [rollup('decode', null, 'tok/s', terms.concat([{ term: 'decode', value: null, evidence: 'unknown', tag: 'UNKNOWN', src: why }])), prefillUnknown] });
    if (!(mp.total_b > 0) || !(mp.active_b > 0)) return un('model params (total, active) not stated');
    if (st.draft_n > 0) return un('a draft / MTP run: no draft term for this model');
    const total = mp.total_b * 1e9, active = mp.active_b * 1e9, q = quantType(data, row.quant);
    const bpw = row.file_gib ? row.file_gib * GiB * 8 / total : q ? v(data.types[q].bpw) : null;
    if (!bpw) return un('neither a file size nor a known quant label (' + row.quant + ')');
    terms.push(T('bits per weight (' + row.quant + ')', row.file_gib ? Object.assign({ v: bpw }, their('file ' + row.file_gib + ' GiB / ' + mp.total_b + 'B params, as stated (' + row.url + ')')) : data.types[q].bpw));
    terms.push(T('active params B', Object.assign({ v: mp.active_b }, their(mp.params_src))));
    // dense (attention, shared experts, embeddings) vs routed: total = D + X, active = D + X k/E
    const moe = mp.experts > 0 && mp.experts_active > 0 && total > active;
    const xTot = moe ? (total - active) / (1 - mp.experts_active / mp.experts) : 0, xAct = moe ? active - (total - xTot) : 0, dense = active - xAct, bw = bpw / 8;
    const pl = String(row.placement || ''), L = mp.layers, nOf = () => Math.min(L, Number(pl.split(':')[1]));
    // bytes read per token on the GPU and on the CPU side, per placement (first match wins)
    const PL = [
      [() => pl === 'cpu_only' || (!hw.gpu && !hw.unified), () => [0, active * bw]],
      [() => pl === 'all_gpu' || hw.unified, () => [active * bw, 0]],
      [() => pl === 'experts_cpu', () => [dense * bw, xAct * bw]],
      [() => /^n_cpu_moe:\d+$/.test(pl) && L && moe, () => [(dense + xAct * (1 - nOf() / L)) * bw, xAct * nOf() / L * bw]],
      [() => /^ngl:\d+$/.test(pl) && L, () => [active * bw * nOf() / L, active * bw * (1 - nOf() / L)]],
    ].find(([ok]) => ok());
    if (!PL) return un('placement ' + (row.placement || 'not stated') + ' cannot be mapped (needs all_gpu, cpu_only, experts_cpu, n_cpu_moe:N or ngl:N with a layer count)');
    const [gpuB, cpuB] = PL[1]();
    if (gpuB > 0 && !hw.gpu_bw_gbs) return un('GPU bandwidth not stated');
    if (cpuB > 0 && !hw.dram_bw_gbs) return un('DRAM spec bandwidth not stated');
    if (cpuB > 0 && (!q || !data.types[q].kappa)) return un('quant ' + row.quant + ' has no CPU kernel efficiency');
    // GPU side: the backend's measured line (bench.gpu: reach x nameplate, host ms), else the old gpu_eta + a borrowed host
    // a line is a driver stack: it serves rows on its own vendor's cards (a borrow names the vendor it stands in for)
    const B = data.bench || {}, be = (B.backend_of && B.backend_of.v[row.backend]) || null, L0 = be && B.gpu ? B.gpu[be] : null;
    const vendor = B.vendor_of_class ? B.vendor_of_class.v[row.hw_class] : null, G = L0 && (L0.vendors || []).includes(vendor) ? L0 : null;
    const eta = G ? G.eta : data.assume.gpu_eta;
    const gpuS = gpuB > 0 ? gpuB / (hw.gpu_bw_gbs * 1e9 * v(eta)) : 0;
    if (gpuB > 0) terms.push(T('GPU nameplate bandwidth GB/s (' + (hw.gpu || 'unified') + ')', Object.assign({ v: hw.gpu_bw_gbs }, their(hw.spec_src))), T('GPU decode reach (share of nameplate' + (G ? ', ' + be + ' line' : '') + ')', eta));
    // per-token MoE overhead (router, gather, extra kernel launches) for each MoE layer held on the GPU
    const isMoe = moe || mp.total_b > mp.active_b;
    const mL = !isMoe || !(gpuB > 0) ? 0 : !(cpuB > 0) ? L : pl === 'experts_cpu' ? 0 : /^n_cpu_moe:/.test(pl) ? L - nOf() : /^ngl:/.test(pl) ? nOf() : 0;
    if (isMoe && gpuB > 0 && !(L > 0)) return un('a MoE on the GPU with no layer count');
    if (mL > 0 && !(G && G.moe_layer_ms && v(G.moe_layer_ms) !== null)) return un('a MoE on the GPU with no per-token MoE term for backend ' + (row.backend || 'not stated') + (L0 && !G ? ' on ' + vendor + ' cards (measured on ' + L0.vendors.join(', ') + ' only)' : ''));
    const moeS = mL > 0 ? mL * v(G.moe_layer_ms) / 1000 : 0;
    if (mL > 0) terms.push(T('MoE overhead ms per GPU MoE layer (' + be + ', x ' + mL + ' layers)', G.moe_layer_ms));
    // CPU side: one node's share of the spec x our spec->read ratio x the measured placement gain (bench.numa), never the sum
    const D = externalDerived(data), k = cpuB > 0 ? kappaSrc(data, null, q) : null;
    const nodes = (hw.sockets || v(X.sockets_default)) * (hw.nodes_per_socket || v(X.nodes_per_socket_default)), sk = hw.sockets || v(X.sockets_default);
    const gain = cpuB > 0 && nodes > 1 ? numaGain(data, (st.numa_policy || v(X.numa_policy_default)), nodes, sk) : null;
    if (cpuB > 0 && nodes > 1 && !gain) return un('no measured NUMA arm for ' + (st.numa_policy || v(X.numa_policy_default)) + ' over ' + nodes + ' nodes on ' + sk + ' sockets');
    const Bd = cpuB > 0 ? hw.dram_bw_gbs / nodes * 1e9 * v(D.dram_eta) * (gain ? v(gain.g) : 1) : 0;
    const cpuS = cpuB > 0 ? cpuB / Math.min(Bd, Bd * v(k) * v(D.cpu_f)) : 0;
    if (cpuB > 0) terms.push(T('DRAM spec bandwidth GB/s' + (nodes > 1 ? ' (/ ' + nodes + ' nodes)' : ''), Object.assign({ v: hw.dram_bw_gbs }, their(hw.spec_src))), T('DRAM spec -> read ratio', D.dram_eta), T('CPU kernel efficiency ' + q, k), T('CPU box factor f', D.cpu_f));
    if (gain) terms.push(T('NUMA placement gain over one node (' + gain.key + ')', gain.g));
    if (gpuB > 0 && cpuB > 0) terms.push(T('GPU/CPU overlap', data.engines.llamacpp.overlap));
    const hb = X.host_borrow.v[cpuB > 0 ? 'cpu' : 'gpu'];
    const host = cpuB > 0 ? hostSrc(data, data.assume.host_ref_box, hb[0], hb[1]) : G ? G.host_ms : data.models[hb[0]].host_engine[hb[1]];
    terms.push(T('host overhead ms/token (' + (cpuB > 0 || !G ? 'borrowed, ' + hb.join('|') : be + ' line') + ')', host));
    const t = gpuS + moeS + cpuS + v(host) / 1000;
    if (!(t > 0 && isFinite(t))) return un('a borrowed or derived term has no value: ' + terms.filter(x => !(isFinite(x.value) && x.value !== null)).map(x => x.term).join(', '));
    return Object.assign(out, { status: 'PREDICTED', decode_tps: 1 / t, err: meas ? (1 / t) / meas - 1 : null, placement: gpuB > 0 && cpuB > 0 ? 'hybrid' : gpuB > 0 ? 'gpu' : 'cpu',
      bytes_gb: { gpu: gpuB / 1e9, cpu: cpuB / 1e9 }, evidence: [rollup('decode', 1 / t, 'tok/s', terms), prefillUnknown] });
  }
  // ---------- dense-path calibration (data/dense-cal.json): kernel rates by quant type, the cost of non-plain layers, the KV read rate ----------
  // The bench lines were fitted on plain llama-style models whose weights are K-quants, Q4_0 or Q8_0. denseCal reads three corrections
  // at run time: a quant type's kernel rate on a card (MEASURED microbenchmarks, or ms per billion weights FITTED on whole-model points),
  // ms a non-plain layer adds (FITTED), and the KV read rate at depth (FITTED on a depth pair); each card falls back to the plain line
  // where nothing is measured. Points marked holdout never set a term: they test the fitted ones (denseCalReport, report.js --dense-cal).
  function denseCardLine(data, ck) {
    const [bk, id] = ck.split('/'), box = data.boxes[bk], g = box && box.gpus.find(x => x.id === id);
    if (!g) return null;
    const nb = (data.bench.native_backend && data.bench.native_backend.v) || {}, be = nb[g.vendor], L = be && data.bench.gpu && data.bench.gpu[be];
    return L && v(L.eta) !== null ? { ck, backend: be, line: L, rate: v(L.eta) * v(g.bw) * 1e9, host: L.host_ms } : null;
  }
  const denseApplies = (e, be, ck) => (!e.backend || e.backend === '*' || e.backend === be) && (!e.cards || e.cards.includes(ck));
  const denseKindOf = l => l.kind || (l.f16_bytes_per_token ? (l.window ? 'sliding' : 'full') : 'linear');
  // the time one card spends on a map of {ggml type: bytes} at its line, each type at its kernel rate (cal null = the plain line)
  function denseTypesMs(cal, ln, types, skip) {
    let ms = 0;
    Object.keys(types || {}).forEach(t => {
      const T = t.toUpperCase(), b = types[t], f = cal ? denseTypeFactor(cal, ln, T) : null;
      if (skip && skip(T, f)) return;
      ms += f && f.per === 'weight' ? b * 8 / f.bpw / 1e9 * f.ms : b / (ln.rate * (f ? f.eff : 1)) * 1000;
    });
    return ms;
  }
  function denseTypeFactor(cal, ln, T) {
    const w = cal.terms.find(x => x.kind === 'weight_ms' && x.v !== null && x.types.includes(T) && denseApplies(x, ln.backend, ln.ck));
    if (w && cal.bpw[T]) return { per: 'weight', ms: w.v, bpw: cal.bpw[T], tag: w.tag, src: w.src, id: w.id };
    const k = cal.kernels.find(x => x.eff[T] !== undefined && denseApplies(x, ln.backend, ln.ck));
    return k ? { per: 'byte', eff: k.eff[T], tag: k.tag, src: k.src } : null;
  }
  // the fastest MEASURED kernel-bound rate on a card (per byte, below the line): the faster side of a borrowed rate
  function denseFastestKernel(cal, ln) {
    let hi = null;
    cal.kernels.filter(k => k.tag === 'MEASURED' && denseApplies(k, ln.backend, ln.ck)).forEach(k => Object.keys(k.eff).forEach(T => {
      if (cal.kernel_bound.test(T) && k.eff[T] < 1 && (!hi || k.eff[T] > hi.eff)) hi = { type: T, eff: k.eff[T] };
    }));
    return hi;
  }
  function denseLayerTerm(cal, ln, kind) {
    if (kind === 'full') return null;
    const f = cal.terms.find(x => x.kind === 'layer_ms' && x.layer === kind && x.v !== null && denseApplies(x, ln.backend, ln.ck));
    return f || cal.assumed.find(x => x.kind === 'layer_ms' && x.layer === kind && denseApplies(x, ln.backend, ln.ck)) || null;
  }
  function denseKvTerm(cal, ln) {
    return cal.terms.find(x => x.kind === 'kv_eff' && x.v !== null && denseApplies(x, ln.backend, ln.ck)) || cal.assumed.find(x => x.kind === 'kv_eff' && denseApplies(x, ln.backend, ln.ck));
  }
  // one point's decode step in ms on its card: host + every layer's weights at their kernel rates + its KV at depth + the layer-kind
  // cost; cal null = the plain line (the dense path before this calibration); without = term ids left out, skipTypes = ggml types
  // left out (a fit solves for them)
  function densePointMs(data, cal, p, without, skipTypes) {
    const ln = denseCardLine(data, p.card), skip = without || [], kvB = p.kv ? v((data.types[p.kv.type] || {}).bpw || 16) / 16 : 1;
    if (!ln) return null;
    let ms = v(ln.host);
    const kvT = cal && denseKvTerm(cal, ln), kvEff = kvT && !skip.includes(kvT.id) ? v(kvT.v) : 1;
    Object.keys(p.layers).forEach(k => {
      const L = p.layers[k];
      ms += denseTypesMs(cal, ln, L.types, cal ? (T, f) => (f && f.id && skip.includes(f.id)) || (skipTypes || []).includes(T) : null);
      if (k === 'output') return;
      if (k !== 'linear' && p.kv) ms += L.n * p.kv.f16_bytes_per_token * kvB * p.depth / (ln.rate * kvEff) * 1000;
      const lt = cal && denseLayerTerm(cal, ln, k);
      if (lt && !skip.includes(lt.id)) ms += L.n * v(lt.v);
    });
    return ms;
  }
  function denseCal(data) {
    if (data._dense_cal) return data._dense_cal;
    const D = data.dense_cal;
    if (!D) return null;
    const byId = {}, points = D.points.map(p0 => {
      const base = p0.same_as ? D.points.find(x => x.id === p0.same_as) : null, p = Object.assign({}, base ? { layers: base.layers, kv: base.kv, model: base.model, build: base.build } : {}, p0);
      p.layers = Object.fromEntries(Object.entries(p.layers).map(([k, L]) => [k, Object.assign({}, L, { types: Object.fromEntries(Object.entries(L.types).map(([t, gb]) => [t, gb * 1e9])) })]));   // GB -> bytes
      byId[p.id] = p;
      return p;
    });
    const kernels = [];
    D.kernels.forEach(k => {
      const eff = {};
      Object.keys(k.gbs || {}).forEach(t => { eff[t] = k.gbs[t] / k.ref_gbs; });
      Object.keys(k.borrow || {}).forEach(t => { const from = kernels.find(x => x.eff[k.borrow[t]] !== undefined && x.backend === k.backend); if (from) eff[t] = from.eff[k.borrow[t]]; });
      kernels.push({ backend: k.backend, cards: k.cards, eff, tag: k.tag, src: k.src + (k.borrow ? ' (' + Object.keys(k.borrow).map(t => t + ' <- ' + k.borrow[t]).join(', ') + ')' : '') });
    });
    const cal = { kernels, terms: [], assumed: D.assumed.map(x => Object.assign({ id: 'assumed:' + x.kind + ':' + (x.layer || '') + ':' + x.backend }, x)), bpw: D.bpw.v, error: D.error, kernel_bound: new RegExp(D.kernel_bound.v), points };
    const tms = p => 1000 / p.tps;
    D.terms.forEach(t0 => {
      const t = Object.assign({ v: null, tag: 'FIT' }, t0), ps = t.fit_on.map(id => byId[id]);
      cal.terms.push(t);
      // only a point marked fit sets a term: a holdout, context or missing id named in fit_on leaves the term unfitted
      const notFit = t.fit_on.filter((id, i) => !ps[i] || ps[i].use !== 'fit');
      if (notFit.length) { t.tag = 'UNKNOWN'; t.src = 'UNKNOWN: fit_on names ' + notFit.join(', ') + ', not a point marked use fit. ' + t.src; return; }
      const ln = denseCardLine(data, ps[0].card);
      if (!ln) { t.tag = 'UNKNOWN'; return; }
      let how;
      if (t.kind === 'kv_eff') {
        const [a, b] = ps, kvB = x => Object.keys(x.layers).filter(k => k !== 'linear' && k !== 'output').reduce((s, k) => s + x.layers[k].n * x.kv.f16_bytes_per_token * v((data.types[x.kv.type] || {}).bpw || 16) / 16 * x.depth, 0);
        t.v = (kvB(b) - kvB(a)) / (ln.rate * (tms(b) - tms(a)) / 1000);
        how = ((kvB(b) - kvB(a)) / 1e9).toFixed(3) + ' GB more KV at ' + b.depth + ' than at ' + a.depth + ' took ' + (tms(b) - tms(a)).toFixed(2) + ' ms more (' + a.tps + ' -> ' + b.tps + ' tok/s); at the line ' + (ln.rate / 1e9).toFixed(0) + ' GB/s that is ' + t.v.toFixed(3) + ' of the line';
      } else {
        const xs = ps.map(p => {
          const rest = densePointMs(data, cal, p, [t.id], t.kind === 'weight_ms' ? t.types : null);
          const den = t.kind === 'layer_ms' ? p.layers[t.layer].n : Object.keys(p.layers).reduce((s, k) => s + Object.keys(p.layers[k].types).filter(T => t.types.includes(T.toUpperCase()) && cal.bpw[T.toUpperCase()]).reduce((a, T) => a + p.layers[k].types[T] * 8 / cal.bpw[T.toUpperCase()] / 1e9, 0), 0);
          return { p, x: (tms(p) - rest) / den, rest, den };
        });
        t.v = med(xs.map(x => x.x));
        how = xs.map(x => x.p.id + ': measured ' + tms(x.p).toFixed(2) + ' ms a step (' + x.p.tps + ' tok/s), every other term ' + x.rest.toFixed(2) + ' ms, over ' + (t.kind === 'layer_ms' ? x.den + ' ' + t.layer + ' layers' : x.den.toFixed(2) + ' G weights of ' + t.types.join('/')) + ' = ' + x.x.toFixed(4)).join('; ');
      }
      t.src = 'FIT: ' + t.src + '. ' + how + ' (' + ps.map(p => p.src).join(' | ') + ')';
      t.runs = t.fit_on;
    });
    // predicted vs measured for every point, before (the plain line) and after; the stated error per backend = the largest holdout miss
    const rows = points.map(p => {
      const before = densePointMs(data, null, p), after = densePointMs(data, cal, p), sc = p.power ? p.power.scale : null;
      const meas = p.tps * (sc ? (sc[0] + sc[1]) / 2 : 1), ln = denseCardLine(data, p.card);
      return { id: p.id, use: p.use, card: p.card, backend: ln && ln.backend, model: p.model, build: p.build, depth: p.depth, measured: p.tps, measured_scaled: sc ? [p.tps * sc[0], p.tps * sc[1]] : null,
        before: before && 1000 / before, after: after && 1000 / after, err_before: before && (1000 / before) / meas - 1, err_after: after && (1000 / after) / meas - 1, src: p.src, power: p.power || null };
    });
    cal.rows = rows;
    cal.base_err = {};
    rows.filter(r => r.use === 'holdout' && r.err_after !== null).forEach(r => { cal.base_err[r.backend] = Math.max(cal.base_err[r.backend] || 0, Math.abs(r.err_after)); });
    Object.defineProperty(data, '_dense_cal', { value: cal, enumerable: false, writable: true, configurable: true });
    return cal;
  }
  function denseCalReport(data) {
    const c = denseCal(data);
    return c ? { terms: c.terms.map(t => ({ id: t.id, kind: t.kind, v: t.v, tag: t.tag, src: t.src })), kernels: c.kernels, assumed: c.assumed, base_err: c.base_err, rows: c.rows } : null;
  }
  // ---------- a dense model the planner has no entry for, planned from its layout alone (tools/harvest.py --hf) ----------
  // spec (built by harvest from config.json + safetensors or GGUF headers): { model, arch, ctx, kv_type, depths, kv_layers:
  // [{ f16_bytes_per_token, window }], quants: [{ label, engine, layer_bytes: [...], output_bytes, embd_bytes, src }] }.
  // Per box: each card runs llama.cpp on its vendor's backend (bench.native_backend) and reads at that backend's bench line
  // (line reach x the card nameplate, + the line's host ms), the same terms predictExternal uses for a community card. Layers
  // fill the cards first (weights + their KV at ctx, the engine margin kept free), the rest and the token embedding stay in
  // DRAM and read at min(DRAM read, DRAM read x kappa x f), the MoE planner's CPU rate. Decode at depth d reads every
  // resident weight once plus each layer's KV for min(d, window) tokens, each weight at its type's kernel rate on the card, each
  // non-plain layer at its fitted cost and the KV at the card's fitted read rate (denseCal). llama.cpp keeps checks.vram_peak.margin
  // llamacpp_fit free per card under --fit on (its default), llamacpp_ngl with -ngl pinned. opts: { knobs: { freeDesktop,
  // freeHousehold, ngl, fit: 'on' | 'off' }, kvReadEff (overrides the calibrated KV read rate), uncal (the plain line, no calibration) }
  // the VRAM llama.cpp leaves free on a card for a dense model: -ngl pinned (or --fit off) keeps its compute buffers and context,
  // --fit on (llama-server's default) also its free-memory target and its own projection (checks.vram_peak.margin llamacpp_*)
  function denseFitMargin(data, kn) {
    const M = data.checks && data.checks.vram_peak && data.checks.vram_peak.margin;
    if (!M) return null;
    return kn.ngl !== undefined || kn.fit === 'off' ? M.llamacpp_ngl : M.llamacpp_fit;
  }
  // a resident on a test card is planned as stopped unless knobs.freeTestCards is false (dense-cal.json plan.free_test_cards);
  // each card lists what it displaces
  function denseCards(data, bk, kn, eng) {
    const B = data.bench || {}, nb = (B.native_backend && B.native_backend.v) || {}, P = (data.dense_cal && data.dense_cal.plan) || {};
    const freeTest = kn.freeTestCards !== undefined ? !!kn.freeTestCards : !!(P.free_test_cards && P.free_test_cards.v);
    return data.boxes[bk].gpus.map(g => {
      const be = nb[g.vendor], L = be && B.gpu && B.gpu[be];
      const freed = o => (o.desktop && kn.freeDesktop) || (o.household && kn.freeHousehold) || (o.test_card && freeTest);
      const res = (g.other || []).reduce((a, o) => a + (freed(o) ? 0 : v(o.gib) || 0), 0);
      const displaced = (g.other || []).filter(o => freed(o) && (v(o.gib) || 0) > 0).map(o => ({ what: o.service || o.what, gib: v(o.gib), test_card: !!o.test_card,
        src: o.test_card && freeTest ? (P.free_test_cards || {}).src || 'knobs.freeTestCards' : 'a knob (freeDesktop / freeHousehold)' }));
      const fm = eng === 'llamacpp' ? denseFitMargin(data, kn) : null;
      const meng = fm ? Object.assign({}, fm, { hi: fm.v !== undefined ? fm.v : fm.hi }) : (g.engines || []).map(e => marginOf(data, e, false)).find(Boolean) || null;
      const ok = L && (L.vendors || []).includes(g.vendor) && v(L.eta) !== null && v(L.host_ms) !== null;
      return { id: g.id, name: g.name, vendor: g.vendor, backend: be || null, line: ok ? L : null, rate: ok ? v(L.eta) * v(g.bw) * 1e9 : null,
        vram: v(g.vram_gib), reserved: res, displaced, margin: meng ? (meng.hi || 0) : 0, marginTerm: meng, why: ok ? null : 'no ' + (be || '?') + ' bench line for ' + g.vendor + ' cards' };
    });
  }
  function predictDense(data, spec, bk, q, opts) {
    if (!data.cal) applyCal(data);
    const o = opts || {}, kn = o.knobs || {}, box = data.boxes[bk], T = (n, x) => term(data, n, x), terms = [];
    const kvB = (spec.kv_type && data.types[spec.kv_type] ? v(data.types[spec.kv_type].bpw) / 8 : 2) / 2;   // x f16 bytes
    const kvEff = o.kvReadEff || 1, ctx = spec.ctx, Lyr = q.layer_bytes.length;
    const kvAt = (i, d) => spec.kv_layers[i] ? spec.kv_layers[i].f16_bytes_per_token * kvB * Math.min(d, spec.kv_layers[i].window || d) : 0;
    // vLLM runs one backend (tensor parallel over cards of the first card's backend); llama.cpp splits layers over any cards
    const live = denseCards(data, bk, kn, q.engine).filter(c => c.line), cards = q.engine === 'vllm' ? live.filter(c => c.backend === (live[0] || {}).backend) : live;
    // fill from the output end (llama.cpp -ngl offloads the last layers first), card by card
    const caps = cards.map(c => Math.max(0, (c.vram - c.reserved - c.margin) * GiB)), on = new Array(Lyr).fill(-1);
    let ci = 0;
    const ngl = kn.ngl === undefined ? Lyr : Math.min(Lyr, kn.ngl);   // -ngl N: at most N layers on the cards (the knob), else as many as fit
    for (let i = Lyr - 1; i >= Lyr - ngl && ci < cards.length; i--) {
      const need = q.layer_bytes[i] + kvAt(i, ctx);
      while (ci < cards.length && caps[ci] < need) ci++;
      if (ci >= cards.length) break;
      caps[ci] -= need; on[i] = ci;
    }
    const allGpu = cards.length && on.every(x => x >= 0);
    let outOn = -1;
    if (allGpu) { const k = cards.findIndex((c, j) => caps[j] >= q.output_bytes); outOn = k; if (k >= 0) caps[k] -= q.output_bytes; }
    const cpuL = on.map((x, i) => x < 0 ? i : -1).filter(i => i >= 0);
    const cpuW = cpuL.reduce((a, i) => a + q.layer_bytes[i], 0) + (outOn < 0 ? q.output_bytes : 0);
    const cpuKvCtx = cpuL.reduce((a, i) => a + kvAt(i, ctx), 0);
    const dramNeed = (cpuW + cpuKvCtx + q.embd_bytes) / GiB, dramCap = v(box.dram.usable_gib);
    const fits = dramNeed > dramCap ? 'no' : 'yes';
    const qt = quantType(data, q.label), kap = qt ? kappaOf(data, bk, qt) : null, cal = (data.cal && data.cal.boxes[bk]) || {};
    const f = v(cal.f !== undefined ? cal.f : box.cpu.f_default), Bd = v(box.dram.bw) * 1e9;
    const cpuRate = kap ? Math.min(Bd, Bd * kap * f) : null;
    const perCard = cards.map((c, j) => ({ card: bk + ' ' + c.id, backend: c.backend, layers: on.filter(x => x === j).length, output: outOn === j,
      gib: (on.reduce((a, x, i) => a + (x === j ? q.layer_bytes[i] + kvAt(i, ctx) : 0), 0) + (outOn === j ? q.output_bytes : 0)) / GiB,
      free_gib: caps[j] / GiB, reserved_gib: c.reserved, margin_gib: c.margin })).filter(x => x.layers || x.output);
    // the services this plan stops: residents freed on a card that holds part of the model (a freed card left empty displaces nothing)
    const displaces = [].concat(...cards.map((c, j) => on.includes(j) || outOn === j ? c.displaced.map(d => Object.assign({ card: bk + ' ' + c.id }, d)) : []));
    const out = { box: bk, quant: q.label, engine: q.engine, fits, layers_gpu: Lyr - cpuL.length, layers_cpu: cpuL.length, output_on: outOn >= 0 ? 'gpu' : 'cpu',
      cards: perCard, displaces, cpu_gib: (cpuW + cpuKvCtx) / GiB, dram_need_gib: dramNeed, dram_usable_gib: dramCap, ctx, kv_type: spec.kv_type || 'f16', decode: [], terms, why: null, status: 'PREDICTED' };
    const un = why => Object.assign(out, { status: 'UNMODELLED', why, decode_tps: null });
    if (fits === 'no') return un('does not fit: ' + dramNeed.toFixed(1) + ' GiB on the CPU side > ' + dramCap.toFixed(0) + ' GiB usable DRAM');
    if (q.gpu_only && cpuL.length) out.fits = 'no';
    if (q.gpu_only && cpuL.length) return un(q.engine + ' keeps every weight in VRAM: ' + cpuL.length + ' of ' + Lyr + ' layers do not fit the cards');
    if (q.engine === 'vllm' && cards.some((c, j) => on.includes(j) && c.vendor === 'intel'))
      return un('vLLM XPU: ' + (data.engines.vllm && data.engines.vllm.unmodelled_why || 'no rate model'));
    const cpuSide = cpuW > 0;
    if (cpuSide && cpuRate === null) return un('quant ' + q.label + ' has no CPU kernel efficiency on ' + bk);
    cards.forEach((c, j) => { if (on.includes(j) || outOn === j) terms.push(T('GPU decode reach x nameplate GB/s (' + bk + ' ' + c.id + ', ' + c.backend + ' line)', { v: c.rate / 1e9, tag: c.line.eta.tag, src: 'bench.gpu.' + c.backend + ' eta ' + v(c.line.eta) + ' x ' + v(data.boxes[bk].gpus.find(g => g.id === c.id).bw) + ' GB/s nameplate; ' + c.line.eta.src })); });
    if (cpuSide) terms.push(T('DRAM read GB/s (' + bk + ')', box.dram.bw), T('CPU kernel efficiency ' + qt, kappaSrc(data, bk, qt)), T('CPU box factor f (' + bk + ')', cal.f !== undefined ? cal.f : box.cpu.f_default));
    if (perCard.length && cpuSide) terms.push(T('GPU/CPU overlap', data.engines.llamacpp.overlap));
    const hb = data.external.host_borrow.v.cpu, gl = cards.find((c, j) => on.includes(j));
    const host = gl ? gl.line.host_ms : hostSrc(data, data.assume.host_ref_box, hb[0], hb[1]);
    terms.push(T('host overhead ms/token (' + (gl ? gl.backend + ' line' : 'borrowed, ' + hb.join('|')) + ')', host));
    if (q.engine !== 'llamacpp') terms.push(T(q.engine + ' decode rate', { v: null, tag: 'ESTIMATE', src: 'ESTIMATE: the llama.cpp bench line stands in for ' + q.engine + ' (the planner has no ' + q.engine + ' rate model: ' + ((data.engines[q.engine] || {}).unmodelled_why || '') + ')' }));
    // per card: each layer's weights at their types' kernel rates, its KV at the card's read rate, its kind's cost (denseCal);
    // o.uncal = the plain line (every byte at the line rate, KV at kvEff, no layer cost)
    const DC = o.uncal ? null : denseCal(data), lns = cards.map(c => denseCardLine(data, bk + '/' + c.id)), kinds = spec.kv_layers.map(denseKindOf);
    const used = new Map(), err = { why: [] }, use = (name, x) => used.set(name, x);
    const cardKv = j => o.kvReadEff || !DC ? { v: kvEff } : denseKvTerm(DC, lns[j]) || { v: 1 };
    const layerMs = (j, i) => {
      if (!DC || !lns[j]) return 0;
      const lt = denseLayerTerm(DC, lns[j], kinds[i]);
      if (lt) use('ms a ' + kinds[i] + ' layer adds (' + bk + ' ' + cards[j].id + ')', lt);
      return lt ? v(lt.v) : 0;
    };
    const typesMs = (j, types, bytes) => {
      if (!DC || !lns[j] || !types) return bytes / cards[j].rate * 1000;
      Object.keys(types).forEach(t => {
        const f = denseTypeFactor(DC, lns[j], t.toUpperCase());
        if (f) use(f.per === 'weight' ? 'ms per G weights of ' + f.id + ' (' + bk + ' ' + cards[j].id + ')' : t.toUpperCase() + ' kernel rate / line (' + bk + ' ' + cards[j].id + ')', { v: f.per === 'weight' ? f.ms : f.eff, tag: f.tag, src: f.src });
      });
      return denseTypesMs(DC, lns[j], types);
    };
    // the fixed part of a step per card (weights + layer cost) does not change with depth
    const fixedMs = cards.map((c, j) => on.reduce((a, x, i) => a + (x === j ? typesMs(j, q.layer_types && q.layer_types[i], q.layer_bytes[i]) + layerMs(j, i) : 0), 0) +
      (outOn === j ? typesMs(j, q.output_types, q.output_bytes) : 0));
    cards.forEach((c, j) => {
      if (!on.includes(j) && outOn !== j) return;
      const kt = cardKv(j);
      if (!o.kvReadEff) terms.push(T('KV read rate / weight read rate (' + bk + ' ' + c.id + ')', { v: v(kt.v), tag: kt.tag || 'ESTIMATE', src: kt.src || 'ESTIMATE: KV read at the weight-read rate' }));
    });
    if (o.kvReadEff) terms.push(T('KV read rate / weight read rate', { v: kvEff, tag: 'READ', src: 'a setting' }));
    used.forEach((u, name) => terms.push(T(name, { v: u.v !== undefined ? u.v : null, tag: u.tag, src: u.src })));
    displaces.forEach(d => terms.push(T('resident service displaced (' + d.card + ')', { v: d.gib, tag: 'READ', src: d.what + ' stops for this plan: ' + d.src })));
    if (q.engine === 'llamacpp' && cards.length) { const m = cards[0].marginTerm; if (m) terms.push(T('VRAM llama.cpp keeps free per card (' + (kn.ngl !== undefined || kn.fit === 'off' ? '-ngl pinned' : '--fit on') + ')', { v: cards[0].margin, tag: m.tag, src: m.src })); }
    const step = d => {
      let t = v(host) / 1000;
      cards.forEach((c, j) => {
        if (!on.includes(j) && outOn !== j) return;
        const ke = v(cardKv(j).v);
        t += fixedMs[j] / 1000 + on.reduce((a, x, i) => a + (x === j ? kvAt(i, d) : 0), 0) / (c.rate * ke);
      });
      if (cpuSide) t += (cpuW + cpuL.reduce((a, i) => a + kvAt(i, d) / kvEff, 0)) / cpuRate;
      return t;
    };
    // the stated error: the largest holdout miss on the cards' backends (denseCal), widened toward slower for each term no
    // measurement covers and toward faster for a borrowed kernel rate (dense-cal.json error)
    if (DC) {
      const E = DC.error, pct = k => v(E[k]) / 100, be = [...new Set(cards.filter((c, j) => on.includes(j) || outOn === j).map(c => c.backend))];
      const base = Math.max(pct('floor_pct'), ...be.map(b => DC.base_err[b] || 0));
      let minus = base;
      const kb = new Set();
      cards.forEach((c, j) => {
        if (!lns[j] || !(on.includes(j) || outOn === j)) return;
        on.forEach((x, i) => { if (x === j && q.layer_types) Object.keys(q.layer_types[i]).forEach(t => { const T_ = t.toUpperCase(), f = DC.kernel_bound.test(T_) ? denseTypeFactor(DC, lns[j], T_) : true; if (!f || f !== true && !['MEASURED', 'FIT'].includes(f.tag)) kb.add(T_ + (f ? ' (borrowed rate)' : '') + ' on ' + bk + ' ' + c.id); }); });
      });
      if (kb.size) { minus += pct('unmeasured_kernel_pct'); err.why.push('no measured kernel rate: ' + [...kb].join(', ') + ' (-' + v(E.unmeasured_kernel_pct) + ' %)'); }
      const nh = be.filter(b => DC.base_err[b] === undefined);
      if (nh.length) { minus += pct('no_holdout_pct'); err.why.push('no holdout point on ' + nh.join(', ') + ' (-' + v(E.no_holdout_pct) + ' %)'); }
      const lk = new Set();
      cards.forEach((c, j) => { if (lns[j]) on.forEach((x, i) => { if (x === j && kinds[i] !== 'full') { const lt = denseLayerTerm(DC, lns[j], kinds[i]); if (!lt || lt.tag !== 'FIT' && !(lt.measured_on || []).includes(bk + '/' + c.id)) lk.add(kinds[i] + ' layers on ' + bk + ' ' + c.id); } }); });
      if (lk.size) { minus += pct('unmeasured_layer_pct'); err.why.push('no fitted cost: ' + [...lk].join(', ') + ' (-' + v(E.unmeasured_layer_pct) + ' %)'); }
      if (Lyr > v(E.deep_layers)) { minus += pct('deep_model_pct'); err.why.push(Lyr + ' layers, the lines saw at most ' + v(E.deep_layers) + ' (-' + v(E.deep_model_pct) + ' %)'); }
      if (cpuSide) { minus += pct('cpu_spill_pct'); err.why.push((cpuL.length ? cpuL.length + ' layers' : '') + (cpuL.length && outOn < 0 ? ' and ' : '') + (outOn < 0 ? 'the output head' : '') + ' on the CPU (-' + v(E.cpu_spill_pct) + ' %)'); }
      // a borrowed rate can be wrong both ways: the faster side re-times every borrowed byte at the fastest kernel-bound rate
      // measured on its card (dense-cal.json error.borrowed_kernel_plus)
      let tb = 0, tbHi = 0;
      const hiOf = new Set();
      cards.forEach((c, j) => {
        if (!lns[j] || !(on.includes(j) || outOn === j)) return;
        const hi = denseFastestKernel(DC, lns[j]);
        const add = types => Object.keys(types || {}).forEach(t => {
          const T_ = t.toUpperCase(), f = DC.kernel_bound.test(T_) ? denseTypeFactor(DC, lns[j], T_) : null;
          if (!f || f.per !== 'byte' || ['MEASURED', 'FIT'].includes(f.tag) || !hi || hi.eff <= f.eff) return;
          tb += types[t] / (lns[j].rate * f.eff); tbHi += types[t] / (lns[j].rate * hi.eff); hiOf.add(hi.type + ' on ' + bk + ' ' + c.id);
        });
        on.forEach((x, i) => { if (x === j && q.layer_types) add(q.layer_types[i]); });
        if (outOn === j) add(q.output_types);
      });
      let plus = base;
      if (tb > 0) {
        const s = step(ctx / 2), up = s / (s - (tb - tbHi)) - 1;
        plus += up;
        err.why.push('borrowed rate: at the fastest measured kernel-bound rate (' + [...hiOf].join(', ') + ') the borrowed bytes decode ' + (100 * up).toFixed(0) + ' % faster (+' + (100 * up).toFixed(0) + ' %)');
      }
      err.why.unshift('holdout miss ' + (be.filter(b => DC.base_err[b] !== undefined).map(b => b + ' ' + (100 * DC.base_err[b]).toFixed(0) + ' %').join(', ') || 'none') + ' (floor ' + v(E.floor_pct) + ' %)');
      out.error = { minus_pct: Math.round(100 * Math.min(minus, 0.9)), plus_pct: Math.round(100 * plus), why: err.why };
    }
    out.decode = (spec.depths || [0, Math.round(ctx / 2), ctx]).map(d => ({ depth: d, tps: 1 / step(d), kv_read_gb: spec.kv_layers.reduce((a, _, i) => a + kvAt(i, d), 0) / 1e9 }));
    out.decode_tps = 1 / step(ctx / 2);   // tokens / time of one decode that fills 0..ctx (step time is linear in depth)
    out.read_gb = { gpu: perCard.reduce((a, c) => a + c.gib, 0) * GiB / 1e9, cpu: (cpuW + cpuKvCtx) / 1e9 };
    out.evidence = rollup('decode', out.decode_tps, 'tok/s', terms);
    return out;
  }
  // ---------- a MoE model the planner has no family for (tools/harvest.py, the generic MoE path) ----------
  // extra = { models: { key: planner model entry built from GGUF or safetensors headers }, kld: { key: { type: tagged KLD } } }.
  // The entries join data.models before applyCal, so each borrows its host from models[key].host_borrow on every box
  // (borrowHostLists, after the run-ledger host fit) and plans through predictRecipe / checkRecipe like any family. A key that names a family already in the planner is refused:
  // the generic path never replaces a hand family.
  function addModels(data, extra) {
    const X = extra || {}, keys = Object.keys(X.models || {});
    keys.forEach(k => {
      if (data.models[k]) throw new Error('addModels: ' + k + ' is already a planner model');
      data.models[k] = X.models[k];
      const kl = (X.kld || {})[k] || {};
      Object.keys(data.types).forEach(q => { data.types[q].kld = data.types[q].kld || {}; data.types[q].kld[k] = kl[q] || { v: null, tag: 'UNKNOWN', src: 'UNKNOWN: no KLD analog for ' + q }; });
    });
    if (keys.length && data.cal) applyCal(data);
    return keys;
  }
  function predictDenseAll(data, spec, opts) {
    const rows = [];
    spec.quants.forEach(q => Object.keys(data.boxes).forEach(bk => rows.push(predictDense(data, spec, bk, q, opts))));
    return rows;
  }
  // per hardware class (and placement): median |error| and bias of our physics over the predicted external rows
  function externalSummary(data) {
    if (data._ext) return data._ext;
    const rows = ((data.external && data.external.rows) || []).map(r => predictExternal(data, r));
    const group = key => { const g = {}; rows.filter(r => r.err !== null && isFinite(r.err)).forEach(r => (g[key(r)] = g[key(r)] || []).push(r)); return Object.keys(g).sort().map(k => ({ key: k, n: g[k].length, median_abs_pct: 100 * med(g[k].map(r => Math.abs(r.err))), bias_pct: 100 * med(g[k].map(r => r.err)), ids: g[k].map(r => r.id) })); };
    const S = { rows, by_class: group(r => r.hw_class), by_class_placement: group(r => r.hw_class + '/' + r.placement), unmodelled: rows.filter(r => r.status !== 'PREDICTED').map(r => ({ id: r.id, why: r.why })) };
    Object.defineProperty(data, '_ext', { value: S, enumerable: false, writable: true, configurable: true });
    return S;
  }

  // text lines for report.js --external and NOTES.md: one line per row (gap, missing fields), then the per-class medians
  function fmtExternal(S) {
    const pct = x => (x >= 0 ? '+' : '') + (100 * x).toFixed(0) + ' %';
    const rows = S.rows.map(r => r.status === 'PREDICTED'
      ? r.id + ': ours ' + r.decode_tps.toFixed(1) + ' vs theirs ' + r.measured + ' tok/s (' + pct(r.err) + ', ' + r.placement + ')'
      : r.id + ': UNMODELLED (' + r.why + '), theirs ' + r.measured + ' tok/s');
    const grp = g => g.key + ': n ' + g.n + ', median |error| ' + g.median_abs_pct.toFixed(0) + ' %, bias ' + (g.bias_pct >= 0 ? '+' : '') + g.bias_pct.toFixed(0) + ' %';
    return { rows, by_class: S.by_class.map(grp), by_class_placement: S.by_class_placement.map(grp) };
  }

  // ---------- dLLM: block-diffusion decode = tokens per forward (TPF) x forwards per second ----------
  // data.dllm (build.py): settings, model bytes (GGUF headers) and one summary per ledger row tagged dllm. The terms are fitted
  // here from the rows, minus data.dllm.holdout and any ids in `drop` (leave-one-out):
  //   TPF and B (tokens submitted per forward) per cell = model|schedule|block|workload: median over the cell's rows (MEASURED)
  //   kernel gap per box|backend|path = (expert bytes touched / line) / (ms - host - non-expert bytes / line), median per
  //   (expert shape x quant) group; groups pool when they agree (FIT)
  const dllmCell = r => [r.model, r.schedule && typeof r.schedule === 'object' ? r.schedule.name : r.schedule, r.block, r.workload].join('|');
  // distinct experts per MoE layer of B rows: rho x B independent uniform top-k picks (rho = dllm.routing, measured locality; 1 = uniform)
  const dllmCover = (m, B, rho) => m.moe ? 1 - Math.pow(1 - m.k / m.E, B * (rho === undefined ? 1 : rho)) : 1;
  function dllmLine(data, box, backend) {
    const D = data.dllm, B = data.bench || {}, bx = data.boxes[box];
    if (backend === 'cpu') return bx && bx.dram && bx.dram.bw ? { rate: bx.dram.bw, host: D.cpu_host_ms, name: box + ' DRAM read' } : null;
    const L = B.gpu && B.gpu[backend];
    return L && L.cards.length && L.cards.some(c => c.startsWith(box + ' ')) ? { rate: L.rate_gbs, host: L.host_ms, name: backend + ' line (' + L.cards.join(', ') + ')' } : null;
  }
  // expert-touched and non-expert ms of one forward of B rows at the line (no kernel gap)
  function dllmParts(data, r, B) {
    const m = data.dllm.models[r.model], ln = dllmLine(data, r.box, r.backend), GB = GiB / 1e9;
    if (!m || !ln) return null;
    const rate = v(ln.rate), cov = dllmCover(m, B, v(data.dllm.routing));
    return { m, ln, cov, host: v(ln.host), ne_ms: v(m.nonexpert_gib) * GB / rate * 1000, ex_ms: v(m.experts_gib) * GB * cov / rate * 1000 };
  }
  function dllmTerms(data, drop) {
    const D = data.dllm, out = { tpf: {}, b: {}, eff: {}, used: [] };
    if (!D) return out;
    const skip = new Set((D.holdout.v || []).concat(drop || []));
    const ok = D.rows.filter(r => r.ok && !skip.has(r.id) && r.tpf > 0);
    const byCell = {};
    ok.filter(r => r.mode !== 'spec' && r.mode !== 'cli-full').forEach(r => (byCell[dllmCell(r)] = byCell[dllmCell(r)] || []).push(r));
    Object.keys(byCell).forEach(k => {
      const R = byCell[k], t = med(R.map(r => r.tpf)), sp = (Math.max(...R.map(r => r.tpf)) - Math.min(...R.map(r => r.tpf))) / t;
      const tag = sp <= v(D.tpf_spread_max) ? 'MEASURED' : 'MEASURED, CONFOUNDED';
      out.tpf[k] = { v: t, tag, src: 'MEASURED: median TPF of ' + R.map(r => r.id + ' ' + r.tpf.toFixed(2)).join(', ') + (tag === 'MEASURED' ? '' : ' (rows spread ' + (100 * sp).toFixed(0) + ' % > ' + (100 * v(D.tpf_spread_max)).toFixed(0) + ' %: the kernel numerics move near-threshold commits)'), runs: R.map(r => r.id) };
      const bs = R.filter(r => r.b > 0);
      if (bs.length) out.b[k] = { v: med(bs.map(r => r.b)), tag: 'MEASURED', src: 'MEASURED: median tokens submitted per forward of ' + bs.map(r => r.id + ' ' + r.b.toFixed(1)).join(', '), runs: bs.map(r => r.id) };
    });
    const fit = ok.filter(r => D.fit_modes.v.indexOf(r.mode) >= 0 && D.fit_blocks.v.indexOf(r.block) >= 0 && r.ms_per_fwd > 0 && r.b > 0 && (r.placement === 'all_gpu' || r.placement === 'cpu_only'));
    const byKey = {};
    fit.forEach(r => {
      const p = dllmParts(data, r, r.b);
      if (!p || !p.m.moe) return;
      const rest = r.ms_per_fwd - p.host - p.ne_ms;
      if (!(rest > 0)) return;
      const key = [r.box, r.backend, r.path || r.backend].join('|');
      const g = p.m.shape + '|' + r.quant;
      ((byKey[key] = byKey[key] || {})[g] = byKey[key][g] || []).push({ id: r.id, eff: p.ex_ms / rest, b: r.b });
      out.used.push(r.id);
    });
    Object.keys(byKey).forEach(key => {
      const G = byKey[key], groups = {};
      Object.keys(G).forEach(g => { groups[g] = { v: med(G[g].map(x => x.eff)), n: G[g].length, tag: 'FIT', src: 'FIT: (expert bytes touched / line) / (ms per forward - host - non-expert / line), median of ' + G[g].map(x => x.id + ' ' + x.eff.toFixed(3)).join(', '), runs: G[g].map(x => x.id) }; });
      const gm = Object.values(groups).map(x => x.v), pm = med(gm), sp = (Math.max(...gm) - Math.min(...gm)) / pm, all = [].concat(...Object.values(G));
      out.eff[key] = { groups, spread: sp, pools: sp <= v(D.eff_spread_max), b_min: Math.min(...all.map(x => x.b)), b_max: Math.max(...all.map(x => x.b)),
        pooled: { v: pm, tag: 'FIT', src: 'FIT: median of the group gaps ' + Object.keys(groups).map(g => g + ' ' + groups[g].v.toFixed(3) + ' (n ' + groups[g].n + ')').join(', ') + ' (spread ' + (100 * sp).toFixed(0) + ' %)', runs: all.map(x => x.id) } };
    });
    return out;
  }
  // one dLLM recipe or ledger recipe: decode tok/s = TPF x 1000 / ms per forward; UNMODELLED (why) where a term is missing
  function predictDllm(data, r, terms) {
    const D = data.dllm, T = (n, x) => term(data, n, x), A = [];
    const out = { status: 'UNMODELLED', why: null, kind: 'dllm', decode_tps: null, decode_tps_nodraft: null, prefill_tps: null, prefill_why: 'no dLLM prefill model', ms_per_fwd: null, tpf: null, b: null, cover: null, assumptions: A };
    const sch = r.schedule || {}, mode = sch.mode, terms_ = [];
    const noPrefill = () => rollup('prefill', null, 'tok/s', [{ term: 'prefill', value: null, evidence: 'unknown', tag: 'UNKNOWN', src: out.prefill_why }]);
    const un = why => Object.assign(out, { why, evidence: [rollup('decode', null, 'tok/s', terms_.concat([{ term: 'decode', value: null, evidence: 'unknown', tag: 'UNKNOWN', src: why }]))] });
    if (!D) return un('no dLLM data in this build');
    const m = D.models[r.model];
    if (!m) return un('model ' + r.model + ' has no GGUF header intake in data.dllm.models');
    if (mode === 'spec') return un('spec decoding runs two models (LLaDA drafts, Ling verifies): no two-model forward term');
    if (mode === 'cli-full' || mode === 'full') return un('the forward resubmits the whole sequence (no block KV reuse): B is the sequence, outside every fitted forward');
    if (r.placement !== 'all_gpu' && r.placement !== 'cpu_only') return un('placement ' + r.placement + ': no batch term for CPU-held experts beside a GPU');
    const p0 = dllmParts(data, r, 1);
    if (!p0) return un('no measured line for backend ' + r.backend + ' on ' + r.box + ' (' + JSON.stringify(D.line_of.v) + ')');
    const TT = terms || dllmTerms(data), GB = GiB / 1e9;
    terms_.push(T('line rate GB/s (' + p0.ln.name + ')', p0.ln.rate), T('host ms per forward', p0.ln.host), T('non-expert GiB', m.nonexpert_gib));
    if (sch.name === 'ar' || sch.name === 'ar-full' || mode === 'ar') {
      // the AR control: one token per forward, the bench GPU line + its per-token MoE term per MoE layer
      const L = data.bench.gpu[r.backend], ml = L && L.moe_layer_ms;
      if (r.backend === 'cpu' || !ml || v(ml) === null) return un('AR on ' + r.backend + ': no per-token MoE term');
      const ms = p0.host + p0.ne_ms + v(m.experts_gib) * GB * m.k / m.E / v(p0.ln.rate) * 1000 + m.L * v(ml);
      terms_.push(T('expert GiB (x k/E = ' + m.k + '/' + m.E + ' per token)', m.experts_gib), T('MoE ms per MoE layer (x ' + m.L + ')', ml), T('tokens per forward (AR)', { v: 1, tag: 'READ', src: 'GGUF header: an autoregressive model commits one token per forward' }));
      return Object.assign(out, { status: 'PREDICTED', decode_tps: 1000 / ms, decode_tps_nodraft: 1000 / ms, ms_per_fwd: ms, tpf: 1, b: 1, evidence: [rollup('decode', 1000 / ms, 'tok/s', terms_), rollup('ms per forward', ms, 'ms', terms_.filter(t => !/tokens per forward/.test(t.term))), noPrefill()] });
    }
    const cell = dllmCell(r), tp = TT.tpf[cell], bb = TT.b[cell];
    if (!tp) return un('no measured TPF for ' + cell + ' (cells of ' + r.model + ': ' + (Object.keys(TT.tpf).filter(k => k.startsWith(r.model + '|')).join('; ') || 'none') + ')');
    if (!bb) return un('no measured tokens per forward for ' + cell);
    const B = bb.v, key = [r.box, r.backend, r.moe_path || r.backend].join('|'), K = TT.eff[key];
    terms_.push(T('TPF (' + cell + ')', tp), T('tokens submitted per forward', bb));
    if (!m.moe) return un('a dense diffusion model: no batch-GEMM term (the kernel gap is fitted on MoE rows)');
    if (!K) return un('no kernel gap for ' + key + ' (fitted: ' + Object.keys(TT.eff).join(', ') + ')');
    const s = v(D.b_slack);
    if (B < K.b_min / s || B > K.b_max * s) return un(B.toFixed(1) + ' tokens per forward is outside the fitted ' + K.b_min.toFixed(1) + '-' + K.b_max.toFixed(1) + ' (x/÷ dllm.b_slack ' + s + ') for ' + key);
    const g = m.shape + '|' + (r.quant || ''), own = K.groups[g];
    const eff = own && own.n >= v(D.eff_min_rows) ? own : K.pools ? K.pooled : null;
    if (!eff) return un('the ' + key + ' kernel gap differs by expert shape (spread ' + (100 * K.spread).toFixed(0) + ' % > dllm.eff_spread_max) and ' + g + ' has ' + (own ? own.n + ' row' : 'no rows') + ' (dllm.eff_min_rows ' + v(D.eff_min_rows) + ')');
    const p = dllmParts(data, r, B), ms = p.host + p.ne_ms + p.ex_ms / v(eff);
    A.push('expert coverage of ' + B.toFixed(1) + ' rows: ' + (100 * p.cov).toFixed(0) + ' % per MoE layer (routing locality rho ' + v(D.routing) + ', ' + D.routing.tag + ')', 'kernel gap ' + v(eff).toFixed(3) + (eff === own ? ' (own group ' + g + ')' : ' (pooled over ' + Object.keys(K.groups).join(', ') + ')'));
    terms_.push(T('expert GiB', m.experts_gib), T('expert coverage per MoE layer (' + (100 * p.cov).toFixed(0) + ' %)', Object.assign({}, D.routing, { v: p.cov })), T('MoE kernel gap (share of the line, ' + key + ')', eff));
    if ((r.cards || []).length > 1) terms_.push(T('cards in turn (layer split, ' + r.cards.length + ' cards)', D.cards_serial));
    const cls = data.external && data.external.box_class.v[r.box], XS = cls && r.backend !== 'cpu' ? externalSummary(data) : null;
    const xg = XS ? (XS.by_class_placement.find(x => x.key === cls + '/gpu') || XS.by_class.find(x => x.key === cls)) : null;
    const dec = rollup('decode', v(tp) * 1000 / ms, 'tok/s', terms_);
    if (xg) dec.external = { group: xg.key, n: xg.n, median_abs_pct: xg.median_abs_pct, bias_pct: xg.bias_pct };
    return Object.assign(out, { status: 'PREDICTED', decode_tps: v(tp) * 1000 / ms, decode_tps_nodraft: v(tp) * 1000 / ms, ms_per_fwd: ms, tpf: v(tp), b: B, cover: p.cov,
      evidence: [dec, rollup('ms per forward', ms, 'ms', terms_.filter(t => !/^TPF/.test(t.term))), noPrefill()] });
  }
  // held-out and leave-one-out errors of the dLLM term: each held-out run is predicted with the terms fitted without it (always);
  // each fit row is predicted with the terms refitted without it. err = predicted / measured - 1, for decode, TPF and ms per forward
  function dllmValidation(data) {
    const D = data.dllm;
    if (!D) return { holdout: [], loo: [] };
    // a row summary as a recipe; the ms-per-forward error needs only the refitted kernel gap, so a row whose TPF cell holds no
    // other row still scores its forward cost (TPF and B then come from the full fit and the decode error is left out)
    const rec = r => ({ kind: 'dllm', box: r.box, model: r.model, backend: r.backend, moe_path: r.path, cards: new Array(r.cards).fill('card'), quant: r.quant,
      schedule: { name: r.schedule, mode: r.mode }, block: r.block, workload: r.workload, placement: r.placement });
    const full = dllmTerms(data), e = (a, b) => b ? a / b - 1 : null;
    const one = (r, drop) => {
      const T = dllmTerms(data, drop), cell = dllmCell(rec(r)), own = !T.tpf[cell];
      const p = predictDllm(data, rec(r), own ? Object.assign({}, T, { tpf: full.tpf, b: full.b }) : T);
      if (p.status !== 'PREDICTED' || !r.decode_tps) return { id: r.id, status: p.status, why: p.why };
      return { id: r.id, status: p.status, tpf_from: own ? 'its own row (no other row in the cell)' : 'the other rows of the cell', pred: own ? null : p.decode_tps, meas: r.decode_tps,
        err: own ? null : e(p.decode_tps, r.decode_tps), tpf_err: own ? null : e(p.tpf, r.tpf), ms_err: e(p.ms_per_fwd, r.ms_per_fwd), tpf: p.tpf, ms: p.ms_per_fwd, ms_meas: r.ms_per_fwd };
    };
    const ho = new Set(D.holdout.v), used = dllmTerms(data).used;
    const S = xs => { const a = xs.filter(x => x.err !== null && x.err !== undefined); return { n: a.length, median_abs_pct: a.length ? 100 * med(a.map(x => Math.abs(x.err))) : null, bias_pct: a.length ? 100 * med(a.map(x => x.err)) : null }; };
    const holdout = D.rows.filter(r => ho.has(r.id)).map(r => one(r, []));
    const loo = D.rows.filter(r => used.indexOf(r.id) >= 0).map(r => one(r, [r.id]));
    const P = loo.filter(x => x.status === 'PREDICTED');
    return { holdout, loo, loo_summary: S(P), loo_ms_summary: S(P.map(x => ({ err: x.ms_err }))), unmodelled: loo.filter(x => x.status !== 'PREDICTED') };
  }
  // the gate for a dLLM recipe: inputs, the model on its cards, the conditions; the prediction carries the trust lines
  function dllmCheck(data, r) {
    const D = data.dllm, box = data.boxes[r.box], m = D && D.models[r.model], inp = [];
    if (!box || !m) return { checks: [{ id: 'inputs', status: 'FAIL', detail: 'unknown ' + [!box && 'box ' + r.box, !m && 'dLLM model ' + r.model].filter(Boolean).join(', ') + ' (dLLM models: ' + Object.keys((D && D.models) || {}).join(' ') + ')' }], facts: null, notes: [], bad_input: true };
    if (r.backend !== 'cpu' && !dllmLine(data, r.box, r.backend)) inp.push(['UNKNOWN', 'no measured ' + r.backend + ' line on ' + r.box]);
    if (r.backend !== 'cpu' && m.moe && !r.moe_path) inp.push(['UNKNOWN', 'moe_path not given (stock / per-pair / grouped): the kernel gap is per path']);
    const cards = (r.cards || []).map(c => box.gpus.find(g => g.id === c));
    if (cards.some(c => !c)) inp.push(['FAIL', 'card ' + r.cards.join('+') + ' not on ' + r.box + ' (' + box.gpus.map(g => g.id).join(', ') + ')']);
    const checks = [{ id: 'inputs', label: 'dLLM recipe inputs (box, model, line, kernel path, cards)', status: inp.some(x => x[0] === 'FAIL') ? 'FAIL' : inp.length ? 'UNKNOWN' : 'PASS', detail: inp.length ? inp.map(x => x[0] + ': ' + x[1]).join('; ') : 'consistent' }];
    if (cards.length && cards.every(Boolean)) {
      const per = v(m.total_gib) / cards.length, cap = Math.min(...cards.map(c => v(c.vram_gib)));
      checks.push({ id: 'vram_fit', label: 'model weights split over the cards fit each card', status: per < cap ? 'PASS' : 'FAIL', detail: 'weights ' + v(m.total_gib).toFixed(2) + ' GiB / ' + cards.length + ' card(s) = ' + per.toFixed(2) + ' of ' + cap.toFixed(2) + ' GiB per card (KV and compute buffers are not counted: ctx ' + r.ctx + ')' });
    }
    checks.push(conditionsCheck(data, r, false));
    return { checks, facts: { box: r.box, model: r.model, kind: 'dllm' }, notes: [], bad_input: false, prediction: predictDllm(data, r) };
  }

  // ---------- batched decode: N concurrent streams of one model (data.batch, build.py) ----------
  // One step serves N rows. Its bytes: non-expert weights once, the distinct experts the step's positions touch (read once),
  // and per row the DeltaNet state (read + written) and the attention reads at the row's depth. Cache misses and KV streamed
  // from pinned RAM cross the miss link. kernel = fixed ms + ms per position (generic). step = max(bandwidth, kernel) with the
  // misses inside the bandwidth term (overlap) or added after it (serial). batchFit() fits the free constants from data.batch.rows
  const batchDistinct = (E, k, P) => E * (1 - Math.pow(1 - k / E, P));
  // the bytes of one expert and of the non-expert weights for a config (the measured store, or the config's own bpw)
  function batchWeights(B, c) {
    if (!c.expert_bpw) return { ex: v(B.experts_gib) * GiB / (B.L * B.E), ne: v(B.nonexpert_gib) * GiB, all: (v(B.experts_gib) + v(B.nonexpert_gib)) * GiB };
    const ex = B.P * v(c.expert_bpw) / 8, ne = v(B.nonexpert_bf16_gib) * GiB * v(c.nonexpert_bpw) / v(B.bf16_bpw);
    return { ex, ne, all: ex * B.L * B.E + ne };
  }
  // VRAM per row (GiB) when its KV is allocated for `ctx` tokens, `stream` keeping only the newest `window` in VRAM
  const batchRowGiB = (B, ctx, stream, window) => (v(B.state_mib_per_row) * 2 ** 20 + (stream ? Math.min(ctx, window) : ctx) * v(B.kv_bytes_per_token)) / GiB;
  // VRAM the rows may take without touching the weights: null for the cache layout (rows take it from the expert cache)
  function batchFreeGiB(B, c) {
    if (c.layout === 'split') return v(B.vram_free_split_gib);
    if (c.layout === 'tp') return c.cards * (v(B.card_vram_gib) - v(c.overhead_gib)) - batchWeights(B, c).all / GiB;
    return null;
  }
  // one step. s = {c: config, N, depth, ctx, window, K: {fixed, row}, F: {miss1, churn, mpg, tau}, mode, m: miss-share override}
  function batchStep(data, s) {
    const B = data.batch, c = s.c, N = s.N, L = B.L, W = batchWeights(B, c);
    const mtp = c.mtp && N <= v(B.mtp_rows_max), P = mtp ? v(B.mtp_verify_positions) : N;
    // auto: KV stays in VRAM while N rows fit the free VRAM; past that each row keeps the newest tokens the free VRAM holds
    // (VRAM first, the rest in pinned RAM), never fewer than none
    const free = batchFreeGiB(B, c), fullGiB = batchRowGiB(B, s.ctx, false, null);
    const fill = free === null ? null : Math.max(0, Math.floor((free * GiB / N - v(B.state_mib_per_row) * 2 ** 20) / v(B.kv_bytes_per_token)));
    const window = c.kv === 'auto' ? fill : s.window === undefined ? v(B.kv_resident_tokens) : s.window;
    const stream = window !== null && (c.kv === 'stream' || (c.kv === 'auto' && N * fullGiB > free));
    const rowGiB = batchRowGiB(B, s.ctx, stream, window);
    const m = c.layout !== 'cache' ? 0 : s.m !== undefined ? s.m : Math.min(1, s.F.miss1 + s.F.churn * (N - 1) + s.F.mpg * (N - 1) * rowGiB);
    const att = Math.min(s.depth, v(B.attn_budget_tokens)) * v(B.kv_bytes_per_token) + s.depth * v(B.index_bytes_per_token);
    const far = stream && s.depth > window ? (s.depth - window) / s.depth : 0, state = v(B.state_mib_per_row) * 2 ** 20;
    const touched = L * batchDistinct(B.E, B.k, P);
    const vram = W.ne + touched * W.ex * (1 - m) + N * (2 * state + att * (1 - far));
    const link = touched * W.ex * m + N * att * far;
    const lanes = c.layout === 'tp' ? c.cards : 1;
    const bw_ms = vram / (lanes * v(B.card_bw_gbs) * 1e9) * 1000, link_ms = link / (lanes * v(B.miss_gbs) * 1e9) * 1000;
    const sync = c.layout === 'split' ? v(B.split_hop_ms) : c.layout === 'tp' ? L * v(c.sync_ms_per_layer) : 0;
    const kernel_ms = s.K.fixed + s.K.row * P + sync;
    const serial = s.mode === 'serial', side = serial ? bw_ms : bw_ms + link_ms;
    const ms = Math.max(kernel_ms, side) + (serial ? link_ms : 0);
    const binding = kernel_ms >= side ? 'kernel' + (serial && link_ms > 0 ? ' + misses' : '') : (!serial && link_ms > bw_ms ? 'PCIe (misses / streamed KV)' : 'VRAM bandwidth' + (serial && link_ms > 0 ? ' + misses' : ''));
    const tokens = mtp ? s.F.tau : N, total = tokens * 1000 / ms;
    const ram = (c.layout === 'cache' ? v(B.experts_gib) - v(B.cache_gib_one) : 0) + (stream ? N * Math.max(0, s.ctx - window) * v(B.kv_bytes_per_token) / GiB : 0);
    return { N, positions: P, ms, bw_ms, link_ms, kernel_ms, binding, miss: m, total, per_stream: total / N, stream, row_gib: rowGiB, rows_gib: N * rowGiB,
      free_gib: free, fits: free === null ? N * rowGiB < v(B.cache_gib_one) : N * rowGiB <= free, ram_gib: ram, weights_gib: W.all / GiB };
  }
  const batchCfgOf = (data, r) => Object.assign({}, data.batch.configs.one, { kv: r.kv_resident ? 'stream' : 'resident' });
  // fit the free constants on data.batch.rows for one miss mode (and an optional four-stream hit rate)
  function batchFit(data, mode, hit4) {
    const B = data.batch, h1 = v(B.hit_one), h4 = hit4 === undefined ? v(B.hit_four) : hit4, n4 = v(B.hit_four_streams);
    const F = { miss1: 1 - h1, churn: ((1 - h4) - (1 - h1)) / (n4 - 1), mpg: 0, tau: 1 }, K = { fixed: 0, row: 0 };
    const at = (r, N, extra) => Object.assign({ c: batchCfgOf(data, r), N, depth: v(B.fit_depth_tokens), ctx: r.ctx, window: r.kv_resident, K, F, mode }, extra || {});
    const pts = (use, pick) => [].concat(...B.rows.filter(r => r.use === use).map(r => Object.keys(r.tps).map(Number).filter(pick).map(N => ({ r, N, ms: N * 1000 / r.tps[N] }))));
    const kp = pts('kernel', N => N > v(B.mtp_rows_max)), vp = pts('vram', N => N > 1), tp = pts('kernel', N => N <= v(B.mtp_rows_max));
    for (let round = 0; round < B.fit_rounds.v; round++) {
      // kernel: the step time the kernel must explain (all of it with overlap; what the misses leave with serial), least squares
      // with both constants >= 0
      const y = kp.map(p => p.ms - (mode === 'serial' ? batchStep(data, at(p.r, p.N, { K: { fixed: 0, row: 0 } })).link_ms : 0)), x = kp.map(p => p.N);
      const mx = x.reduce((a, b) => a + b, 0) / x.length, my = y.reduce((a, b) => a + b, 0) / y.length;
      const slope = x.reduce((a, xi, i) => a + (xi - mx) * (y[i] - my), 0) / x.reduce((a, xi) => a + (xi - mx) ** 2, 0);
      Object.assign(K, slope < 0 ? { fixed: my, row: 0 } : my - slope * mx >= 0 ? { fixed: my - slope * mx, row: slope } : { fixed: 0, row: x.reduce((a, xi, i) => a + xi * y[i], 0) / x.reduce((a, xi) => a + xi * xi, 0) });
      // miss share per GiB the rows take: each VRAM row's miss share by bisection, then the share left after the churn term
      const g = vp.map(p => {
        let lo = 0, hi = 1;
        for (let i = 0; i < B.fit_bisect_steps.v; i++) { const mid = (lo + hi) / 2; if (batchStep(data, at(p.r, p.N, { m: mid })).ms < p.ms) lo = mid; else hi = mid; }
        const sl = batchStep(data, at(p.r, p.N)).row_gib;
        return Math.max(0, (lo - F.miss1 - F.churn * (p.N - 1)) / ((p.N - 1) * sl));
      });
      F.mpg = g.length ? med(g) : 0;
      // MTP tokens per one-stream step: the measured rate x the modelled step
      F.tau = tp.length ? med(tp.map(p => p.r.tps[p.N] * batchStep(data, at(p.r, p.N)).ms / 1000)) : 1;
    }
    const pred = (r, N) => batchStep(data, at(r, N)).total;
    const residuals = [].concat(...B.rows.map(r => Object.keys(r.tps).map(Number).map(N => ({ id: r.id, use: r.use, N, meas: r.tps[N], pred: pred(r, N), err: pred(r, N) / r.tps[N] - 1, binding: batchStep(data, at(r, N)).binding }))));
    // a check is predicted on its own config (x.cfg; the fit rows are all one-card rows)
    (B.checks || []).forEach(x => {
      const r = { ctx: x.ctx, kv_resident: x.kv_resident }, c = x.cfg && B.configs[x.cfg] ? B.configs[x.cfg] : batchCfgOf(data, r), st = batchStep(data, at(r, x.N, { c }));
      residuals.push({ id: x.id, use: 'check', cfg: x.cfg || 'one', N: x.N, meas: v(x.tps), pred: st.total, err: st.total / v(x.tps) - 1, binding: st.binding });
    });
    const tw = B.twin, tws = c => batchStep(data, { c: B.configs[c], N: 1, depth: v(B.twin_depth_tokens), ctx: v(B.twin_depth_tokens), K, F, mode }).ms;
    const twin = { meas: tw.tps.split / tw.tps.one, pred: tws('one') / tws('split') };
    const fitted = residuals.filter(x => x.use !== 'check');
    return { mode, hit4: h4, K, F, residuals, twin, median_abs_pct: 100 * med(fitted.map(x => Math.abs(x.err))), max_abs_pct: 100 * Math.max(...fitted.map(x => Math.abs(x.err))) };
  }
  // the planning table: every config x its kernel variants x context x stream count, on one fit (default: the first miss mode).
  // A variant's ms per row is what the per-row overhead drops to: never above the fitted one
  function batchTable(data, fit) {
    const B = data.batch, f = fit || batchFit(data, B.miss_modes.v[0]), out = [];
    Object.keys(B.configs).forEach(ck => {
      const c = B.configs[ck];
      c.variants.forEach(vk => {
        const V = B.variants[vk], K = { fixed: V.no_fixed ? 0 : f.K.fixed, row: V.row_ms ? Math.min(f.K.row, v(V.row_ms)) : f.K.row };
        B.contexts.v.forEach(ctx => {
          const steps = B.streams.v.map(N => batchStep(data, { c, N, depth: ctx, ctx, K, F: f.F, mode: f.mode }));
          const hit = steps.find(x => x.total >= B.target_tps.v);
          out.push({ config: ck, label: c.label, variant: vk, variant_label: V.label, ctx, steps, first_n_at_target: hit ? hit.N : null });
        });
      });
    });
    return out;
  }

  // ---------- batch window: a bottom-up cost model of one batched-decode window per pipeline stage (data.window, build.py) ----------
  // Where batchFit() fits a step to measured tok/s, this builds it from kernels, so it can price a kernel before it is written.
  // A window serves `rows` rows of one row group on one stage (a contiguous layer range on one card). Its time is the sum of the
  // card's microbenchmarked kernel classes (each runs in the layers of its kind, at the per-layer average of the measured layers)
  // plus the unmeasured extras of the first and last stage. A class's in-graph ms = gamma x max(0, isolated ms - its launches x
  // omega) x (the card's bandwidth / a what-if card's), omega being the per-launch overhead of an isolated measurement (flush,
  // sync). omega and the extras are FIT on measured stage times; eta (FIT) scales the pipeline law. G row groups take turns
  // through S stages: tok/s = eta x rows x 1000 x min(G / sum of stage ms, 1 / slowest stage ms). No class or kind is named here:
  // they all come from the config's spec and ubench log.
  const winCfg = (data, key) => data.window.configs[key || data.window.default];
  function winInterp(grid, ys, r) {
    if (r <= grid[0]) return ys[0];
    const j = grid.findIndex(g => g >= r), i = j < 0 ? grid.length - 1 : j;
    return ys[i - 1] + (ys[i] - ys[i - 1]) * (r - grid[i - 1]) / (grid[i] - grid[i - 1]);
  }
  // the calibration before any fit: the priors, nothing taken off, no extras
  const winPrior = W => ({ omega_us: 0, gamma: v(W.settings.gamma), phi: v(W.ubench.routing_phi), extras: Object.fromEntries(W.extras.map(e => [e.id, 0])), eta: 1 });
  // a class's in-graph ms over all its measured layers at `rows`; o.scale = {class name: factor} (a lever), o.bw = a card's GB/s
  function winClassMs(W, c, rows, K, o) {
    const U = W.ubench, R = U.routing, x = o || {};
    const iso = winInterp(U.grid, c.ms, rows) + (R && c.name === R.class ? K.phi * winInterp(U.grid, R.alt_delta_ms, rows) : 0);
    const bw = x.bw ? v(W.card_bw_gbs) / x.bw : 1, f = x.scale && x.scale[c.name] !== undefined ? x.scale[c.name] : 1;
    const lo = c.layers * c.launches * v(W.settings.graph_launch_us) / 1000;
    return Math.max(lo, K.gamma * Math.max(lo, iso - c.layers * c.launches * K.omega_us / 1000) * bw * f);
  }
  // a class's bandwidth floor over its measured layers (ms): its time at the floor's rows over its x floor (the ubench's own
  // full-precision ratio), else its bytes over the floor bandwidth
  const winFloorMs = (W, c) => {
    const U = W.ubench, t = c.ms[U.grid.indexOf(U.floor_rows)];
    return c.xfloor > 0 && t > 0 ? t / c.xfloor : c.mb_floor_lyr * c.layers / v(U.floor_gbs);
  };
  const winKinds = W => Array.from(new Set(W.layers.map(x => x.kind).concat(Object.keys(W.ubench.kinds))));
  // ms per layer of each kind at `rows`: a class's in-graph time spread evenly over its measured layers. A kind the ubench did
  // not measure gets only the classes that run in every layer (winReport lists it as unmeasured)
  function winLayerMs(W, rows, K, o) {
    const ks = winKinds(W);
    return Object.fromEntries(ks.map(k => [k, W.ubench.classes.filter(c => c.kind === null || c.kind === k).reduce((a, c) => a + winClassMs(W, c, rows, K, o) / c.layers, 0)]));
  }
  const winSplit = (W, o) => (o && o.split) || W.stages.map(s => s.layers);
  // a point (or the profile) measured on another layout carries its own split, which overrides the config's stages for it
  const winPointSplit = (W, p) => (p && p.split) || winSplit(W);
  const winExtraOn = (e, i, nS) => (e.stage === 'first' && i === 0) || (e.stage === 'last' && i === nS - 1);
  const winExtraMs = (W, K, i, nS, rows) => W.extras.reduce((a, e) => a + (winExtraOn(e, i, nS) ? K.extras[e.id] * (e.per === 'row' ? rows : 1) : 0), 0);
  // an anchor's offsets (K.offset: measured - modelled stage ms a row at a measured point; the gm round moved stage times in
  // proportion to the rows: V4 - G2 is -0.2/-0.4 ms at one row, -4.6/-7.0 at 16) ride with the first and the last stage
  const winOffsetMs = (K, i, nS, rows) => rows * (!K.offset ? 0 : nS === 1 ? K.offset.reduce((a, x) => a + x, 0) : i === 0 ? K.offset[0] : i === nS - 1 ? K.offset[K.offset.length - 1] : 0);
  // the cost of each layer and of each stage position's own terms at `rows`: from the profile when K carries one, else the ubench
  // classes plus the fitted extras and an anchor's offsets
  function winCosts(W, rows, K, o) {
    if (K.prof) return K.prof.costs(rows, o || {});
    const per = winLayerMs(W, rows, K, o);
    return { layer: W.layers.map(x => per[x.kind]), stage: (i, nS) => winExtraMs(W, K, i, nS, rows) + winOffsetMs(K, i, nS, rows) };
  }
  function winStageMs(W, split, rows, K, o) {
    const c = winCosts(W, rows, K, o);
    return split.map(([a, b], i) => c.layer.slice(a, b + 1).reduce((s, x) => s + x, 0) + c.stage(i, split.length));
  }
  // anchor a fit on a measured point (a later config whose kernels the ubench did not measure): the plan and the levers then
  // start from that point's measured stage times, each lever moving them by what it saves in the model
  function winAnchor(data, key, fit, id) {
    const W = winCfg(data, key), p = W.points.find(x => x.id === id);
    if (!p) return null;
    const split = winPointSplit(W, p);
    const G = winGroups(W, p.agents, split.length), rows = p.agents / G, model = winStageMs(W, split, rows, Object.assign({}, fit.K, { offset: null }));
    return Object.assign({}, fit, { anchor: p.id, K: Object.assign({}, fit.K, { offset: v(p.stage_ms).map((m, i) => (m - model[i]) / rows) }) });
  }
  function winLaw(stage, rows, groups, eta) {
    if (!Number.isFinite(eta) || eta <= 0) return null;
    const tot = stage.reduce((a, b) => a + b, 0), mx = Math.max(...stage);
    return eta * rows * 1000 * Math.min(groups / tot, 1 / mx);
  }
  // row groups: the option, else the config's ratio of groups to stages times the stages, and enough groups that no window
  // passes the engine's row cap; never more than the agents
  const winGroups = (W, N, nS, o) => Math.min(N, (o && o.groups) || Math.max(1, Math.round(nS * v(W.groups) / W.stages.length), Math.ceil(N / v(W.max_window_rows))));
  function winStep(W, N, K, o) {
    const split = winSplit(W, o), G = winGroups(W, N, split.length, o), rows = N / G, U = W.ubench;
    const stage = winStageMs(W, split, rows, K, o), total = winLaw(stage, rows, G, K.eta);
    return { N, groups: G, rows, split, stage_ms: stage, bottleneck: stage.indexOf(Math.max(...stage)), total, per_agent: total === null ? null : total / N,
      beyond_grid: rows > U.grid[U.grid.length - 1], beyond_engine: rows > v(W.max_window_rows) || N > v(W.max_agents) };
  }
  // the best contiguous split of the layers over nS stages at the step's rows (least slowest stage), then the step on it
  function winRebalance(W, N, K, o, nS) {
    const S = nS || winSplit(W, o).length, G = winGroups(W, N, S, o), rows = N / G, C = winCosts(W, rows, K, o);
    const c = C.layer, L = c.length, P = c.reduce((a, x) => a.concat(a[a.length - 1] + x), [0]);
    const cost = (i, a, b) => P[b] - P[a] + C.stage(i, S);
    // best[i][j]: the least slowest stage putting layers [0, j) on stages 0..i; cut[i][j]: where stage i starts
    const best = [], cut = [];
    Array.from({ length: S }, (_, i) => i).forEach(i => {
      best[i] = []; cut[i] = [];
      Array.from({ length: L + 1 }, (_, j) => j).forEach(j => {
        const opts = i === 0 ? [{ k: 0, m: j > 0 ? cost(0, 0, j) : Infinity }] : Array.from({ length: Math.max(0, j - i) }, (_, t) => t + i).map(k => ({ k, m: Math.max(best[i - 1][k], cost(i, k, j)) }));
        const pick = opts.reduce((a, x) => (x.m < a.m ? x : a), { k: -1, m: Infinity });
        best[i][j] = pick.m; cut[i][j] = pick.k;
      });
    });
    const split = [];
    Array.from({ length: S }, (_, t) => S - 1 - t).reduce((end, i) => { split.unshift([cut[i][end], end - 1]); return cut[i][end]; }, L);
    return winStep(W, N, K, Object.assign({}, o, { split, groups: G }));
  }
  // least squares with x >= 0 (drop the most negative term, re-solve) on rows {a, y, w}
  function winNnls(eqs, n) {
    const solve = free => {
      const A = free.map(i => free.map(j => eqs.reduce((s, e) => s + e.w * e.w * e.a[i] * e.a[j], 0) + (i === j ? 1e-12 : 0)));
      const b = free.map(i => eqs.reduce((s, e) => s + e.w * e.w * e.a[i] * e.y, 0));
      const M = A.map((r, i) => r.concat(b[i]));
      M.forEach((_, p) => {
        const piv = M.slice(p).reduce((best, r, q) => (Math.abs(r[p]) > Math.abs(M[best][p]) ? q + p : best), p);
        [M[p], M[piv]] = [M[piv], M[p]];
        M.forEach((r, q) => { if (q !== p && M[p][p] !== 0) { const f = r[p] / M[p][p]; r.forEach((_, k) => { r[k] -= f * M[p][k]; }); } });
      });
      return M.map((r, i) => (r[i] !== 0 ? r[r.length - 1] / r[i] : 0));
    };
    const go = free => {
      const s = solve(free), neg = s.reduce((a, x, i) => (x < (a < 0 ? s[a] : 0) ? i : a), -1);
      if (neg < 0) return Array.from({ length: n }, (_, i) => (free.includes(i) ? s[free.indexOf(i)] : 0));
      return go(free.filter((_, i) => i !== neg));
    };
    return go(Array.from({ length: n }, (_, i) => i));
  }
  // calibrate: omega by a grid + golden-section search, the extras by non-negative least squares on the 'fit' points' stage
  // times (relative error), eta = the median of valid training tok/s over the law on those same training stage times.
  // Checks/law points remain observations only. Without throughput training there is no efficiency estimate.
  // fo: {phi: a routing share, gamma_free: fit gamma too, dropping the extras marked kernel_scale_alt (the variant where the
  // in-graph per-row cost the isolated kernels miss lives in the kernels, so a lever saves more)}
  function winFit(data, key, fo) {
    const W = winCfg(data, key), S = W.settings, prior = winPrior(W), sp0 = p => winPointSplit(W, p), x0 = fo || {};
    if (x0.phi !== undefined) prior.phi = x0.phi;
    const ex = W.extras.filter(e => !(x0.gamma_free && e.kernel_scale_alt));
    const fit = W.points.filter(p => p.use === 'fit'), G = p => winGroups(W, p.agents, sp0(p).length), rowsOf = p => p.agents / G(p);
    const at = (omega, gamma) => {
      const K0 = Object.assign({}, prior, { omega_us: omega, gamma });
      const eqs = [].concat(...fit.map(p => { const r = rowsOf(p), m = winStageMs(W, sp0(p), r, K0); return v(p.stage_ms).map((y, i) => ({ y: y - m[i], w: 1 / y, a: ex.map(e => (winExtraOn(e, i, sp0(p).length) ? (e.per === 'row' ? r : 1) : 0)) })); }));
      const x = winNnls(eqs, ex.length), sse = eqs.reduce((s, e) => s + (e.w * (e.a.reduce((t, a, j) => t + a * x[j], 0) - e.y)) ** 2, 0);
      return { omega, gamma, x, sse };
    };
    const phiR = (Math.sqrt(5) - 1) / 2, lo0 = (a, b) => (b.sse < a.sse ? b : a);
    const golden = (f, lo, hi) => { const [a, b] = Array.from({ length: 60 }).reduce(([l, h]) => { const c = h - phiR * (h - l), d = l + phiR * (h - l); return f(c).sse < f(d).sse ? [l, d] : [c, h]; }, [lo, hi]); return f((a + b) / 2); };
    // omega: a grid, then golden-section around the best grid point
    const omegaAt = gamma => {
      const step = v(S.omega_step_us), g0 = Array.from({ length: Math.floor(v(S.omega_max_us) / step) + 1 }, (_, i) => at(i * step, gamma)).reduce(lo0);
      return lo0(g0, golden(o => at(o, gamma), Math.max(0, g0.omega - step), g0.omega + step));
    };
    const gr = S.gamma_range ? v(S.gamma_range) : [v(S.gamma), v(S.gamma)];
    const best = x0.gamma_free ? golden(omegaAt, gr[0], gr[1]) : omegaAt(prior.gamma);
    const K = Object.assign({}, prior, { omega_us: best.omega, gamma: best.gamma, extras: Object.fromEntries(W.extras.map(e => [e.id, ex.includes(e) ? best.x[ex.indexOf(e)] : 0])) });
    const etaPoints = fit.filter(p => Number.isFinite(v(p.tps)) && v(p.tps) > 0 && v(p.stage_ms).every(x => Number.isFinite(x) && x > 0));
    const law = etaPoints.map(p => v(p.tps) / winLaw(v(p.stage_ms), rowsOf(p), G(p), 1)).filter(x => Number.isFinite(x) && x > 0);
    K.eta = law.length ? med(law) : null;
    const residuals = W.points.map(p => {
      const r = rowsOf(p), sp = winStageMs(W, sp0(p), r, K), meas = v(p.stage_ms), t = p.tps ? v(p.tps) : null;
      const chain = winLaw(sp, r, G(p), K.eta), lawp = winLaw(meas, r, G(p), K.eta);
      return { id: p.id, use: p.use, change: p.change || null, agents: p.agents, rows: r, split: sp0(p), stage_meas: meas, stage_pred: sp, stage_err: sp.map((x, i) => x / meas[i] - 1),
        tps_meas: t, tps_pred: p.use === 'law' ? null : chain, tps_err: t > 0 && chain !== null && p.use !== 'law' ? chain / t - 1 : null, law_pred: lawp, law_err: t > 0 && lawp !== null ? lawp / t - 1 : null };
    });
    const pick = (use, f) => [].concat(...residuals.filter(x => use.includes(x.use)).map(f)).filter(x => x !== null).map(Math.abs);
    const mx = xs => (xs.length ? 100 * Math.max(...xs) : null);
    return { key: W.key, K, status: K.eta === null ? 'UNKNOWN' : 'FIT', why: K.eta === null ? 'no finite positive training throughput for pipeline eta' : null,
      eta_training: etaPoints.map(p => p.id), gamma_free: !!x0.gamma_free, residuals, free: ex.length + 1 + (x0.gamma_free ? 1 : 0), equations: fit.reduce((a, p) => a + sp0(p).length, 0),
      stage_fit_max_abs_pct: mx(pick(['fit'], x => x.stage_err)), stage_check_max_abs_pct: mx(pick(['check'], x => x.stage_err)),
      tps_fit_max_abs_pct: mx(pick(['fit'], x => [x.tps_err])), tps_check_max_abs_pct: mx(pick(['check'], x => [x.tps_err])), law_max_abs_pct: mx(pick(['fit', 'check', 'law'], x => [x.law_err])) };
  }
  // the lever table: each class taken to k x its bandwidth floor (at the floor's rows; never below graph_launch_us a launch), the
  // same factor at every row count; predicted at the plan's agents on the config's split and on the rebalanced split. The
  // layer-split rebalance is a lever of its own. o: {agents, xfloor, bw, fit}
  function winLevers(data, key, opts) {
    const W = winCfg(data, key), o = opts || {}, f = o.fit || winFit(data, key), K = f.K, U = W.ubench;
    const N = o.agents || v(W.settings.plan_agents), k = o.xfloor || v(W.settings.lever_xfloor), base = { bw: o.bw };
    const now = winStep(W, N, K, base), reb = winRebalance(W, N, K, base);
    if (K.eta === null) return { status: 'UNKNOWN', why: f.why, now, rebalanced: reb, levers: [], unmeasured: [] };
    const one = (c, kk) => {
      const cur = winClassMs(W, c, U.floor_rows, K), floor = winFloorMs(W, c), lo = v(W.settings.graph_launch_us) * c.layers * c.launches / 1000;
      const scale = cur > 0 ? Math.min(1, Math.max(kk * floor, lo) / cur) : 1, oo = Object.assign({}, base, { scale: { [c.name]: scale } });
      const st = winStep(W, N, K, oo), rb = winRebalance(W, N, K, oo);
      return { scale, xfloor_graph: floor > 0 ? cur / floor : null, floor_ms: floor, total: st.total, stage_ms: st.stage_ms, total_rebalanced: rb.total, split_rebalanced: rb.split };
    };
    const levers = U.classes.map(c => {
      const x = one(c, k), split = winSplit(W), per = winClassMs(W, c, now.rows, K, base) / c.layers;
      const inStage = split.map(([a, b]) => W.layers.slice(a, b + 1).filter(l => c.kind === null || l.kind === c.kind).length * per);
      return Object.assign({ lever: c.name, kind: c.kind, classes: [c.name], layers_measured: c.layers, xfloor_iso: c.xfloor, ms_in_stage: inStage, gain: x.total - now.total, gain_rebalanced: x.total_rebalanced - reb.total,
        grid: v(W.settings.lever_xfloor_grid).map(g => ({ xfloor: g, total: one(c, g).total, total_rebalanced: one(c, g).total_rebalanced })) }, x);
    });
    // the unmeasured extras have no floor: each row says what a share of it (extra_lever_share) would pay, so it gets measured
    const share = v(W.settings.extra_lever_share), onStage = e => winSplit(W).map((_, i) => winExtraOn(e, i, winSplit(W).length));
    const extras = W.extras.filter(e => K.extras[e.id] > 0).map(e => {
      const oo = Object.assign({}, base), K2 = Object.assign({}, K, { extras: Object.assign({}, K.extras, { [e.id]: K.extras[e.id] * (1 - share) }) });
      const st = winStep(W, N, K2, oo), rb = winRebalance(W, N, K2, oo);
      return { lever: 'unmeasured: ' + e.label, kind: 'extra', unmeasured: true, ms_in_stage: onStage(e).map(on => (on ? K.extras[e.id] * (e.per === 'row' ? now.rows : 1) : 0)), cut_share: share,
        total: st.total, total_rebalanced: rb.total, stage_ms: st.stage_ms, split_rebalanced: rb.split, gain: st.total - now.total, gain_rebalanced: rb.total - reb.total };
    });
    // a measured kernel variant (config `variants`): its class at the new time over the old
    const variants = (W.variants || []).map(x => {
      const oo = Object.assign({}, base, { scale: { [x.class]: v(x.new_us) / v(x.old_us) } }), st = winStep(W, N, K, oo), rb = winRebalance(W, N, K, oo);
      return { lever: x.label + ' (' + v(x.new_us) + ' vs ' + v(x.old_us) + ' us)', kind: 'variant', measured: true, class: x.class, classes: [x.class], total: st.total, stage_ms: st.stage_ms, saved_ms: now.stage_ms.map((m, i) => m - st.stage_ms[i]), total_rebalanced: rb.total, split_rebalanced: rb.split, gain: st.total - now.total, gain_rebalanced: rb.total - reb.total };
    });
    const split = { lever: 'layer split rebalance', kind: 'layout', total: reb.total, total_rebalanced: reb.total, stage_ms: reb.stage_ms, split_rebalanced: reb.split, gain: reb.total - now.total, gain_rebalanced: 0 };
    return { source: 'ubench', agents: N, xfloor: k, now, rebalanced: reb, levers: levers.concat(variants, split).sort((a, b) => b.total - a.total || b.total_rebalanced - a.total_rebalanced), unmeasured: extras };
  }
  // ---------- the profile source: a measured per-term attribution of one point's window per stage (config `profile`) ----------
  // Each term is ms a window on each stage, of a layer kind (spread evenly over that stage's layers of the kind) or of a stage
  // position (the last stage's head). The ubench stays the fallback, and gives each term its row curve (its mapped classes'
  // its bandwidth floor and a what-if card's scaling, and gives each term its row curve: rows_basis 'stage' = the calibrated
  // ubench model's stage time at r rows over that at the profile's rows (Ks: the fit, anchored when the report anchors; the
  // curve the n=2 checks score), 'classes' = its mapped classes' in-graph time ratio (noisy below 4 rows: isolated small
  // kernels), 'row' = r / the profile's rows, 'fixed' = 1. What the point's measured stage run holds beyond the terms is the
  // gap, a term of its own spread over the stage's layers.
  // costs(r, o): o.scale = {term id: factor}, o.cut = {class name: ms a layer saved at the profile's rows}, o.gap = a factor on
  // the gap, o.bw = a card's GB/s
  function winProfCompile(W, Ku, Ks) {
    const P = W.profile, pt = W.points.find(p => p.id === P.point), split = P.split || winPointSplit(W, pt), nP = split.length;
    const R0 = pt.agents / winGroups(W, pt.agents, nP), terms = P.terms, sum = xs => xs.reduce((a, b) => a + b, 0);
    const stageOf = W.layers.map(x => split.findIndex(([a, b]) => x.layer >= a && x.layer <= b));
    const count = (k, i) => W.layers.filter((x, l) => (k === null || x.kind === k) && stageOf[l] === i).length;
    const cls = n => W.ubench.classes.find(c => c.name === n);
    const curve = (t, r) => sum(t.classes.map(([n, f]) => f * winClassMs(W, cls(n), r, Ku)));
    const base = Object.fromEntries(terms.map(t => [t.id, v(t.rows_basis) === 'classes' ? curve(t, R0) : 1])), st0 = winStageMs(W, split, R0, Ks || Ku);
    const basis = { classes: (t, r) => (base[t.id] > 0 ? curve(t, r) / base[t.id] : 1), row: (t, r) => r / R0, fixed: () => 1 };
    const rowf = (t, r, sr) => (v(t.rows_basis) === 'stage' ? sr[t.stage ? (t.stage === 'first' ? 0 : nP - 1) : 0] : basis[v(t.rows_basis)](t, r));
    const termSum = Array.from({ length: nP }, (_, i) => sum(terms.map(t => v(t.ms)[i])));
    const gap = v(pt.stage_ms).map((m, i) => Math.max(0, m - termSum[i])), gapLayer = gap.map((g, i) => g / count(null, i));
    const gapf = (r, sr, i) => (v(P.gap.rows_basis) === 'stage' ? sr[i] : basis[v(P.gap.rows_basis)]({}, r));
    const at = (t, nS, i) => (t.stage === 'first' ? i === 0 : i === nS - 1);
    const pos = t => (t.stage === 'first' ? 0 : nP - 1);
    const kindTerms = Object.fromEntries(Array.from(new Set(W.layers.map(x => x.kind))).map(k => [k, terms.filter(t => t.kind === k)]));
    const stageTerms = terms.filter(t => t.stage);
    // a term's floor over the profile's stages: its classes' bandwidth floor a layer (at the ubench's floor rows) times its layers
    const floor = t => {
      const L = t.kind ? sum(Array.from({ length: nP }, (_, i) => count(t.kind, i))) : 0;
      const f = sum(t.classes.map(([n, x]) => x * winFloorMs(W, cls(n)) / cls(n).layers)), lo = sum(t.classes.map(([n, x]) => x * cls(n).launches)) * v(W.settings.graph_launch_us) / 1000;
      return { cur: sum(v(t.ms)), floor: f * L, lo: lo * L };
    };
    const costs = (r, o) => {
      const sr = winStageMs(W, split, r, Ks || Ku).map((x, i) => x / st0[i]), rf = Object.fromEntries(terms.map(t => [t.id, rowf(t, r, sr)]));
      const rk = (t, i) => (t.kind && v(t.rows_basis) === 'stage' ? sr[i] : rf[t.id]);
      const cutf = Object.fromEntries(Object.entries(o.cut || {}).map(([n, ms]) => [n, ms * winClassMs(W, cls(n), r, Ku) / winClassMs(W, cls(n), R0, Ku)]));
      const val = (t, per, i) => Math.max(0, per * rk(t, i) * (o.bw && t.bw ? v(W.card_bw_gbs) / o.bw : 1) * (o.scale && o.scale[t.id] !== undefined ? o.scale[t.id] : 1)
        - sum(t.classes.map(([n, f]) => f * (cutf[n] || 0))));
      const g = (o.gap !== undefined ? o.gap : 1);
      const parts = W.layers.map((x, l) => kindTerms[x.kind].map(t => [t.id, val(t, v(t.ms)[stageOf[l]] / count(x.kind, stageOf[l]), stageOf[l])]).concat([['gap', gapLayer[stageOf[l]] * gapf(r, sr, stageOf[l]) * g]]));
      const sparts = (i, nS) => stageTerms.filter(t => at(t, nS, i)).map(t => [t.id, val(t, v(t.ms)[pos(t)], pos(t))]);
      return { layer: parts.map(ps => sum(ps.map(p => p[1]))), stage: (i, nS) => sum(sparts(i, nS).map(p => p[1])),
        parts: (a, b, i, nS) => parts.slice(a, b + 1).concat([sparts(i, nS)]).reduce((m, ps) => { ps.forEach(([id, ms]) => { m[id] = (m[id] || 0) + ms; }); return m; }, {}) };
    };
    return { R0, split, terms, gap, term_sum: termSum, floor, costs, ms_in_stage: ts => Array.from({ length: nP }, (_, i) => sum(ts.map(t => v(t.ms)[i]))) };
  }
  // the profile fit: the ubench fit's eta, the profile in place of its classes, extras and anchor; scored on the profile's own
  // point (exact by construction) and its checks (other agent counts move each term along its classes' row curve)
  function winProfFit(data, key, fitU, fitS) {
    const W = winCfg(data, key);
    if (!W.profile) return null;
    const C = winProfCompile(W, fitU.K, (fitS || fitU).K), K = Object.assign({}, fitU.K, { offset: null, prof: C });
    const ids = [W.profile.point].concat(W.profile.checks || []);
    const checks = W.points.filter(p => ids.includes(p.id)).map(p => {
      const split = winPointSplit(W, p), G = winGroups(W, p.agents, split.length), r = p.agents / G, sp = winStageMs(W, split, r, K), meas = v(p.stage_ms), t = p.tps ? v(p.tps) : null, tp = winLaw(sp, r, G, K.eta);
      return { id: p.id, use: p.id === W.profile.point ? 'profile' : 'check', agents: p.agents, rows: r, split, change: p.change || null, stage_meas: meas, stage_pred: sp, stage_err: sp.map((x, i) => x / meas[i] - 1),
        tps_meas: t, tps_pred: tp, tps_err: t > 0 && tp !== null ? tp / t - 1 : null };
    });
    return { key: W.key, source: 'profile', K, rows: C.R0, point: W.profile.point, split: C.split, gap_ms: C.gap, term_sum_ms: C.term_sum, totals_ms: v(W.profile.totals_ms), checks };
  }
  // the lever table on the profile: each term group (config `lever`) at k x its classes' floor, the no-floor groups and the gap at
  // extra_lever_share off, each measured kernel variant (config `variants`: a class's new time over its old), the split rebalance
  function winProfLevers(data, key, opts) {
    const W = winCfg(data, key), o = opts || {}, K = o.fit.K, C = K.prof, S = W.settings, share = v(S.extra_lever_share);
    const N = o.agents || v(S.plan_agents), k = o.xfloor || v(S.lever_xfloor), base = { bw: o.bw };
    const now = winStep(W, N, K, base), reb = winRebalance(W, N, K, base);
    if (K.eta === null) return { status: 'UNKNOWN', why: 'no training throughput for pipeline eta', now, rebalanced: reb, levers: [], unmeasured: [] };
    const run = oo => { const x = Object.assign({}, base, oo); return { st: winStep(W, N, K, x), rb: winRebalance(W, N, K, x) }; };
    const row = (lever, kind, oo, extra) => {
      const { st, rb } = run(oo);
      return Object.assign({ lever, kind, total: st.total, stage_ms: st.stage_ms, total_rebalanced: rb.total, split_rebalanced: rb.split, gain: st.total - now.total, gain_rebalanced: rb.total - reb.total }, extra);
    };
    const groups = Array.from(new Set(C.terms.map(t => t.lever))).map(g => {
      const ts = C.terms.filter(t => t.lever === g), fl = ts.map(C.floor), kind = Array.from(new Set(ts.map(t => t.kind || t.stage + ' stage'))).join('+');
      return { g, ts, fl, kind, ms: C.ms_in_stage(ts), floored: fl.every(f => f.floor > 0), classes: Array.from(new Set([].concat(...ts.map(t => t.classes.map(c => c[0]))))) };
    });
    const scaleAt = (x, kk) => Object.fromEntries(x.ts.map((t, j) => [t.id, x.fl[j].cur > 0 ? Math.min(1, Math.max(kk * x.fl[j].floor, x.fl[j].lo) / x.fl[j].cur) : 1]));
    const sum = xs => xs.reduce((a, b) => a + b, 0);
    const floored = groups.filter(x => x.floored).map(x => row(x.g, x.kind, { scale: scaleAt(x, k) }, {
      classes: x.classes, ms_in_stage: x.ms, floor_ms: sum(x.fl.map(f => f.floor)), xfloor_now: sum(x.fl.map(f => f.cur)) / sum(x.fl.map(f => f.floor)),
      grid: v(S.lever_xfloor_grid).map(gg => { const r = run({ scale: scaleAt(x, gg) }); return { xfloor: gg, total: r.st.total, total_rebalanced: r.rb.total }; }) }));
    const variants = (W.variants || []).map(x => {
      const save = (v(x.old_us) - v(x.new_us)) / 1000 * v(x.launches);
      const r = row(x.label + ' (' + v(x.new_us) + ' vs ' + v(x.old_us) + ' us)', 'variant', { cut: { [x.class]: save } }, { measured: true, class: x.class, classes: [x.class], save_ms_a_layer: save });
      return Object.assign(r, { saved_ms: now.stage_ms.map((m, i) => m - r.stage_ms[i]) });
    });
    const split = { lever: 'layer split rebalance', kind: 'layout', total: reb.total, total_rebalanced: reb.total, stage_ms: reb.stage_ms, split_rebalanced: reb.split, gain: reb.total - now.total, gain_rebalanced: 0 };
    const nofloor = groups.filter(x => !x.floored).map(x => row(x.g, x.kind, { scale: Object.fromEntries(x.ts.map(t => [t.id, 1 - share])) }, { classes: x.classes, ms_in_stage: x.ms, cut_share: share, unmeasured: true }))
      .concat([row('gap: ' + W.profile.gap.label, 'gap', { gap: 1 - share }, { ms_in_stage: C.gap, cut_share: share, unmeasured: true })]);
    return { source: 'profile', agents: N, xfloor: k, now, rebalanced: reb, levers: floored.concat(variants, split).sort((a, b) => b.total - a.total || b.total_rebalanced - a.total_rebalanced),
      unmeasured: nofloor.sort((a, b) => b.total - a.total) };
  }
  // Network recipes are window-model extensions. A link observation is an immutable directed protocol identity, not a
  // best-of rate. Transfers below serialize device->host, wire and host->device; overlap needs its own measured recipe.
  const networkKnown = (data, x) => !!(x && typeof x.tag === 'string' && typeof x.src === 'string' && x.src.trim() && data.meta && data.meta.evidence && data.meta.evidence.by_tag && Object.prototype.hasOwnProperty.call(data.meta.evidence.by_tag, x.tag) && data.meta.evidence.by_tag[x.tag] !== 'unknown');
  function networkTransfer(data, t) {
    const C = data.network, checks = [], inputs = [];
    const fail = (id, status, detail) => checks.push({ id, status, detail });
    const take = (id, x, min) => {
      inputs.push({ id, value: x });
      const z = v(x);
      if (!networkKnown(data, x) || z === null || z === undefined) { fail(id, 'UNKNOWN', 'sourced ' + id + ' required'); return null; }
      if (!Number.isFinite(z) || z < min) { fail(id, 'FAIL', id + ' must be finite and >= ' + min); return null; }
      return z;
    };
    const matches = C ? C.observations.filter(x => x.id === t.observation) : [];
    if (matches.length > 1) return { seconds: null, checks: [{ id: 'observation', status: 'FAIL', detail: 'duplicate observation identity' }], inputs };
    const obs = matches[0];
    if (!obs) return { seconds: null, checks: [{ id: 'observation', status: 'UNKNOWN', detail: 'select an exact directed observation' }], inputs };
    if (!obs.from || !obs.to || obs.from === obs.to || t.from !== obs.from || t.to !== obs.to) fail('direction', 'FAIL', 'observation ' + obs.id + ' is ' + obs.from + ' -> ' + obs.to + ', never its reverse');
    if (!['transport', 'version', 'protocol'].every(k => typeof obs[k] === 'string' && obs[k].trim()) || !Number.isInteger(obs.streams) || obs.streams < 1) fail('observation_protocol', 'UNKNOWN', 'directed transport/version/protocol and positive stream count required');
    if (['transport', 'version', 'streams', 'protocol'].some(k => t[k] !== undefined && t[k] !== obs[k])) fail('protocol', 'FAIL', 'transport/version/stream count differs from observation ' + obs.id);
    const bytes = take('bytes', t.bytes, 0), gbps = take('wire_gbps', obs.gbps, Number.MIN_VALUE);
    const rtt = take('rtt_ms', t.rtt_ms || obs.rtt_ms, 0), trips = take('round_trips', t.round_trips, 1);
    if (trips !== null && !Number.isInteger(trips)) fail('round_trips', 'FAIL', 'round trips must be a positive whole number');
    const device = (end, field) => {
      if (t[end + '_memory'] === 'host') return 0;
      if (t[end + '_memory'] !== 'gpu') { fail(end + '_memory', 'UNKNOWN', 'declare host or gpu; missing PCIe is not a zero-copy path'); return null; }
      const p = C.pcie.find(x => x.box === t[end] && x.card === t[end + '_card']);
      if (!p) { fail(end + '_card', 'FAIL', 'unknown per-card PCIe identity'); return null; }
      const rate = take(end + '_' + field, p[field], Number.MIN_VALUE);
      return rate === null || bytes === null ? null : bytes / (rate * 1e9);
    };
    const out = device('from', 'd2h_gbs'), into = device('to', 'h2d_gbs');
    const seconds = checks.length || [bytes, gbps, rtt, trips, out, into].some(x => x === null) ? null : out + bytes * 8 / (gbps * 1e9) + into + rtt * trips / 1000;
    return { seconds, wire_seconds: bytes !== null && gbps !== null ? bytes * 8 / (gbps * 1e9) : null, observation: obs, inputs, checks };
  }
  // Identity is checked on both sides of the profile/frontier bridge. Two profiles of the same code may have different
  // hc-read modes; a missing label is not permission to borrow the faster mode or call streamed chunks accepted tokens.
  function winComparison(W, r, data, agents) {
    const p = W.profile || {}, f = W.frontier || {}, why = [];
    const marker = (f.ledger || {}).tps, captured = typeof marker === 'string' && /^c[1-9]\d*$/.test(marker) ? Number(marker.slice(1)) : null;
    const requested = agents === undefined ? r ? v(r.agents) : v(W.settings.plan_agents) : agents;
    if (!Number.isInteger(requested) || requested < 1 || captured === null) why.push('agents UNKNOWN on requested/frontier identity');
    else if (requested !== captured) why.push('agents differs between requested recipe and frontier capture');
    for (const k of ['mode', 'evidence_config', 'metric']) {
      const a = v(p[k]), b = v(f[k]), c = v(r ? r[k] : p[k]);
      if (!a || !b || !c || [p[k], f[k], r ? r[k] : p[k]].some(x => !x || !x.tag || !x.src || x.tag === 'UNKNOWN' || !networkKnown(data || {}, x))) why.push(k + ' UNKNOWN');
      else if (a !== b || a !== c) why.push(k + ' differs between profile, frontier and candidate');
    }
    return { eligible: !why.length, why, mode: v(p.mode) || null, evidence_config: v(p.evidence_config) || null, metric: v(p.metric) || null };
  }
  const networkCanonical = x => Array.isArray(x) ? x.map(networkCanonical) : x && typeof x === 'object' ? Object.fromEntries(Object.keys(x).sort().map(k => [k, networkCanonical(x[k])])) : x;
  function networkSame(a, b) {
    if (typeof a === 'number' || typeof b === 'number') {
      return typeof a === 'number' && typeof b === 'number' && Number.isFinite(a) && Number.isFinite(b) && a === b;
    }
    if (a === null || b === null || typeof a !== 'object' || typeof b !== 'object') return a === b;
    if (Array.isArray(a) || Array.isArray(b)) {
      return Array.isArray(a) && Array.isArray(b) && a.length === b.length && a.every((x, i) => networkSame(x, b[i]));
    }
    const ka = Object.keys(a).filter(k => a[k] !== undefined).sort();
    const kb = Object.keys(b).filter(k => b[k] !== undefined).sort();
    return ka.length === kb.length && ka.every((k, i) => k === kb[i] && networkSame(a[k], b[k]));
  }
  // The browser may edit assumptions, never the measured baseline or the profile's captured costs.
  function networkAssumptions(data, key, x) {
    const W = data.window && data.window.configs[key], fields = ['mode', 'evidence_config', 'metric'];
    if (!W) throw new Error('select an existing window');
    if (!x || !x.catalog || !Array.isArray(x.catalog.recipes) || !Array.isArray(x.catalog.observations) || !Array.isArray(x.catalog.pcie) || !x.catalog.settings) throw new Error('catalog recipes, observations, pcie and settings required');
    if ([x.profile_identity, x.frontier_identity].some(y => !y || typeof y !== 'object' || Array.isArray(y) || Object.keys(y).length !== fields.length || Object.keys(y).some(k => !fields.includes(k)))) throw new Error('comparison identity accepts only mode, evidence_config and metric; measured baseline/cost fields cannot be edited');
    return { ...data, network: x.catalog, window: { ...data.window, configs: { ...data.window.configs, [key]: { ...W, profile: { ...W.profile, ...x.profile_identity }, frontier: { ...W.frontier, ...x.frontier_identity } } } } };
  }
  function networkCheck(data, r) {
    const W = data.window && data.window.configs[r.network_window];
    if (!W) return { checks: [{ id: 'network_window', status: 'FAIL', detail: 'unknown window config' }], facts: null, notes: [], bad_input: true, prediction: null };
    const basis = r.network_prediction_identity;
    if (!basis || !Number.isInteger(basis.agents) || basis.agents < 1 || !['profile', 'ubench'].includes(basis.profile_source)) return { checks: [{ id: 'network_basis', status: 'FAIL', detail: 'priced positive integer agents and profile/ubench source required' }], facts: null, notes: [], bad_input: true, prediction: null };
    const candidate = winNetwork(data, r.network_window, { agents: basis.agents, source: basis.profile_source }).find(x => x.id === r.network_recipe);
    if (!candidate || r.box !== W.box || r.model !== W.model || r.engine !== W.engine) return { checks: [{ id: 'network_recipe', status: 'FAIL', detail: 'network recipe/window box, model and engine must match' }], facts: null, notes: [], bad_input: true, prediction: null };
    const equal = !!(r.network_prediction_identity && networkSame(r.network_prediction_identity, candidate.prediction_identity));
    const checks = candidate.checks.concat([{ id: 'network_identity', status: equal ? 'PASS' : 'FAIL', detail: equal ? 'exact priced identity retained' : 'reprice after any input, source, mode or transfer change' }]);
    checks.push({ id: 'network_comparison', status: candidate.comparison.eligible ? 'PASS' : 'UNKNOWN', detail: candidate.comparison.why.join('; ') || 'same mode, metric and evidence configuration' });
    return { checks, facts: { box: r.box, model: r.model, engine: r.engine, kind: 'network' }, notes: ['Modeled network recipe; this gate does not run or qualify any remote engine.'], bad_input: false,
      prediction: { status: candidate.status === 'MODELED' && equal ? 'PREDICTED' : 'UNMODELLED', network: candidate, decode_tps: equal && candidate.eligible ? candidate.metrics.decode_tps : null } };
  }
  function networkPower(rows, boxes, data) {
    const checks = [], terms = [];
    const ids = rows.map(x => x.box);
    if (ids.length !== boxes.length || new Set(ids).size !== ids.length || boxes.some(b => !ids.includes(b))) checks.push('one power budget for every participating box is required');
    rows.forEach(x => ['idle_w', 'cpu_incremental_w', 'gpu_incremental_w'].forEach(k => {
      const z = x[k];
      terms.push({ box: x.box, component: k, value: z });
      if (!networkKnown(data, z) || !Number.isFinite(v(z)) || v(z) < 0) checks.push(x.box + '.' + k + ' UNKNOWN or invalid');
    }));
    const watts = checks.length ? null : terms.reduce((s, x) => s + v(x.value), 0);
    return { watts: watts > 0 ? watts : null, checks, terms, boundary: 'all-box idle system + incremental CPU + incremental GPU; not a wall-plug measurement' };
  }
  // Resource admission precedes pricing. Existing pinned/KV planners own baseline allocations. Explicit additional host,
  // per-card total GPU allocations and concurrent CPU budgets prevent a remote capacity estimate from inventing free RAM.
  function networkResources(data, r) {
    const checks = [], plans = [], ids = (r.resources || []).map(x => x.box);
    const add = (id, status, detail) => checks.push({ id, status, detail });
    if (ids.length !== r.boxes.length || new Set(ids).size !== ids.length || r.boxes.some(b => !ids.includes(b))) add('resources', 'UNKNOWN', 'one resource budget per participating box required');
    const number = (id, x) => {
      if (!networkKnown(data, x) || v(x) === null) { add(id, 'UNKNOWN', 'sourced nonnegative allocation required'); return null; }
      if (!Number.isFinite(v(x)) || v(x) < 0) { add(id, 'FAIL', 'allocation must be a finite nonnegative number'); return null; }
      return v(x);
    };
    (r.resources || []).forEach(x => {
      const box = data.boxes[x.box];
      if (!box) { add('box', 'FAIL', 'unknown box ' + x.box); return; }
      const row = (data.recipes || []).find(y => y.id === x.recipe_id), recipe = row && row.recipe;
      if (x.recipe_id && (!recipe || recipe.box !== x.box)) add(x.box + '.recipe', 'FAIL', 'baseline run must name a recipe on this box');
      const pinned = recipe ? hostPinnedPlan(data, recipe) : null;
      const base = pinned && pinned.pinned_gib !== undefined ? pinned.pinned_gib : number(x.box + '.baseline_host_gib', x.baseline_host_gib);
      if (recipe) {
        const check = hostPinnedCheck(data, recipe);
        if (check && check.status !== 'PASS') checks.push(check);
      }
      const extra = number(x.box + '.extra_host_gib', x.extra_host_gib), other = number(x.box + '.other_host_gib', x.other_host_gib);
      const originalReserve = hostPinnedBudget(data, recipe || { box: x.box });
      const declaredReserve = x.host_reserve_gib === undefined ? null : number(x.box + '.host_reserve_gib', x.host_reserve_gib);
      if (declaredReserve !== null && originalReserve.reserve_gib !== null && declaredReserve < originalReserve.reserve_gib) add(x.box + '.reserve', 'FAIL', 'declared reserve is below the existing household/pinned reserve');
      const reserve = declaredReserve !== null ? hostPinnedBudget(data, { box: x.box, serving: { host_reserve_gib: declaredReserve } }) : originalReserve;
      const allocated = [base, extra, other].some(z => z === null) ? null : base + extra + other;
      if (allocated !== null && reserve.ceiling_gib !== null && allocated > reserve.ceiling_gib) add(x.box + '.host', 'FAIL', 'host allocation ' + allocated + ' exceeds existing pinned/household ceiling ' + reserve.ceiling_gib + ' GiB');
      if (reserve.ceiling_gib === null) add(x.box + '.host', 'UNKNOWN', 'host ceiling UNKNOWN');
      const threads = number(x.box + '.cpu_threads', x.cpu_threads);
      const cpuReserve = v(box.cpu.reserve || data.checks.cpu_reserve.default_reserve);
      if (threads !== null && threads > v(box.cpu.threads) - cpuReserve) add(x.box + '.cpu', 'FAIL', 'concurrent threads invade existing CPU reserve');
      if (!x.gpu_gib || !Object.keys(x.gpu_gib).length) add(x.box + '.gpu', 'UNKNOWN', 'declare total GiB for each participating GPU, or host_only=true');
      if (v(x.host_only) === true && networkKnown(data, x.host_only)) {
        const ix = checks.findIndex(c => c.id === x.box + '.gpu');
        if (ix >= 0) checks.splice(ix, 1);
      }
      Object.entries(x.gpu_gib || {}).forEach(([id, z]) => {
        const g = box.gpus.find(y => y.id === id), need = number(x.box + '.' + id, z);
        if (!g) { add(x.box + '.gpu', 'FAIL', 'unknown GPU ' + id); return; }
        const busy = (g.other || []).reduce((s, y) => s + v(y.gib), 0), free = v(g.vram_gib) - busy;
        if (need !== null && need > free) add(x.box + '.' + id, 'FAIL', 'total allocation exceeds VRAM after existing household/desktop use');
      });
      plans.push({ box: x.box, baseline_recipe: x.recipe_id || null, pinned, allocated_host_gib: allocated, ceiling_gib: reserve.ceiling_gib });
    });
    return { checks, plans };
  }
  function winNetwork(data, key, opts) {
    const C = data.network, W = winCfg(data, key), o = opts || {}, N = o.agents || v(W.settings.plan_agents), base = W.frontier && v(W.frontier.tps);
    if (!C) return [];
    return C.recipes.filter(r => r.window === W.key || r.window === key).map(r => {
      const checks = [], inputs = [], P = r.params || {}, resources = networkResources(data, r), compare = winComparison(o.source === 'ubench' ? { ...W, profile: W.ubench } : W, r, data, N);
      checks.push(...resources.checks);
      const add = (id, status, detail) => checks.push({ id, status, detail });
      const get = (k, min, max, whole) => {
        const x = P[k], z = v(x); inputs.push({ id: k, value: x || null });
        if (!networkKnown(data, x) || z === null || z === undefined) { add(k, 'UNKNOWN', 'set a sourced ' + k + ' at N=' + N); return null; }
        if (!Number.isFinite(z) || z < min || z > (max === undefined ? Infinity : max) || (whole && !Number.isInteger(z))) { add(k, 'FAIL', 'invalid ' + k); return null; }
        return z;
      };
      const bool = (k, measured) => {
        const x = r[k]; inputs.push({ id: k, value: x || null });
        if (!networkKnown(data, x) || typeof v(x) !== 'boolean' || (measured && x.tag !== 'MEASURED')) { add(k, 'UNKNOWN', (measured ? 'measured ' : 'explicit ') + k + ' required'); return; }
        if (!v(x)) add(k, 'FAIL', k + ' failed');
      };
      bool('backend_supported', false);
      if (new Set(r.boxes).size !== r.boxes.length || r.boxes.length < 2 || !r.boxes.includes(W.box)) add('participants', 'FAIL', 'unique participants including the baseline box required');
      (r.transfers || []).forEach(t => {
        if (!r.boxes.includes(t.from) || !r.boxes.includes(t.to)) add('transfer_participants', 'FAIL', 'transfer uses a box outside the admitted power/resource budget');
        for (const end of ['from', 'to']) if (t[end + '_memory'] === 'gpu') {
          const resource = (r.resources || []).find(x => x.box === t[end]);
          if (!resource || !resource.gpu_gib || !resource.gpu_gib[t[end + '_card']] || v(resource.host_only) === true) add('transfer_gpu', 'UNKNOWN', 'GPU endpoint needs an admitted allocation for the exact box/card');
        }
      });
      if (!networkKnown(data, r.agents)) add('agents_evidence', 'UNKNOWN', 'sourced recipe agent count required');
      if (v(r.agents) !== N) add('agents', 'FAIL', 'recipe rates/verification are at N=' + v(r.agents) + ', requested N=' + N);
      const transfers = (r.transfers || []).map(t => networkTransfer(data, t));
      transfers.forEach((t, i) => t.checks.forEach(c => add('transfer' + i + '.' + c.id, c.status, c.detail)));
      if (!transfers.length) add('transfers', 'UNKNOWN', 'explicit byte/direction/transport/RTT/PCIe transfer recipe required');
      const shipping = transfers.length && transfers.every(t => t.seconds !== null) ? transfers.reduce((s, t) => s + t.seconds, 0) : null;
      const metrics = {}, known = xs => xs.every(x => x !== null && x !== undefined);
      if (r.shape === 'pipeline') {
        const stage = v(P.stage_ms), rows = get('rows', Number.MIN_VALUE), groups = get('groups', 1, undefined, true), eta = get('eta', Number.MIN_VALUE, 1);
        if (!networkKnown(data, P.stage_ms) || !Array.isArray(stage) || stage.length < 2 || stage.some(x => !Number.isFinite(x) || x <= 0)) add('stage_ms', 'UNKNOWN', 'sourced full remote/local stage milliseconds per window required');
        else if (known([rows, groups, eta, shipping])) {
          if (rows * groups !== N || transfers.length !== stage.length - 1 || !r.stage_boxes || r.stage_boxes.length !== stage.length) add('pipeline_shape', 'FAIL', 'N=rows*groups and one directed hop per stage boundary required');
          else if (r.stage_boxes.some(b => !r.boxes.includes(b)) || (r.transfers || []).some((t, i) => t.from !== r.stage_boxes[i] || t.to !== r.stage_boxes[i + 1])) add('pipeline_direction', 'FAIL', 'hop endpoints differ from stage owners');
          else { metrics.stage_ms = stage.map((ms, i) => ms + (transfers[i] ? transfers[i].seconds * 1000 : 0)); metrics.decode_tps = winLaw(metrics.stage_ms, rows, groups, eta); }
        }
      } else if (r.shape === 'prefill') {
        const prompt = get('prompt_tokens', 1, undefined, true), local = get('local_prefill_tps', Number.MIN_VALUE), remote = get('remote_prefill_tps', Number.MIN_VALUE);
        const lp = get('local_decode_pause_s', 0), rp = get('remote_decode_pause_s', 0);
        const source = (r.resources || []).map(x => (data.recipes || []).find(y => y.id === x.recipe_id)).find(x => x && x.box === W.box);
        if (source && source.recipe) {
          metrics.kv_plan = kvHostPlan(data, source.recipe);
          if (networkKnown(data, P.upload_gib) && networkKnown(data, P.pmem_gib)) metrics.prefill_basis = prefillModel(data, source.recipe, v(P.upload_gib), null, v(P.pmem_gib)).at;
        }
        const kv = metrics.kv_plan, payload = kv && kv.bpc && kv.layers && prompt !== null ? prompt * kv.bpc * kv.layers : get('kv_bytes_per_slot', Number.MIN_VALUE);
        metrics.kv_shipping_bytes = payload;
        if (payload !== null && (r.transfers.length !== 1 || r.transfers[0].from !== r.remote_box || r.transfers[0].to !== W.box || v(r.transfers[0].bytes) < payload)) add('prefill_kv', 'FAIL', 'ship the full prompt KV from the remote prefill box to the decode box');
        if (known([prompt, local, remote, lp, rp, shipping, payload])) {
          metrics.local_ttft_s = prompt / local; metrics.remote_ttft_s = prompt / remote + shipping;
          metrics.pause_saved_s = lp - rp; metrics.decode_tokens_saved_per_request = base * (lp - rp);
          if (v(P.requests_per_s) !== null && v(P.requests_per_s) !== undefined || v(P.baseline_includes_prefill) === true) {
            const rate = get('requests_per_s', 0);
            if (v(P.baseline_includes_prefill) !== true || !networkKnown(data, P.baseline_includes_prefill)) add('baseline_includes_prefill', 'UNKNOWN', 'frontier must include the same request cadence and local pause');
            else if (rate !== null && rate * lp < 1 && rate * rp < 1) metrics.decode_tps = base * (1 - rate * rp) / (1 - rate * lp);
            else add('request_cadence', 'FAIL', 'prefill pauses saturate the decode timeline');
          }
        }
      } else if (r.shape === 'draft') {
        const dt = get('draft_tps', Number.MIN_VALUE), k = get('draft_tokens', 1, undefined, true), a = get('acceptance', 0, 1), vb = get('verify_base_ms_at_n', Number.MIN_VALUE), ve = get('verify_extra_ms_per_token_at_n', 0);
        if (known([dt, k, a, vb, ve, shipping])) {
          // Conditional per-position survival a, not an average accepted/drafted fraction. Stable geometric sum at a=1.
          const accepted = a === 1 ? k : a * (1 - Math.pow(a, k)) / (1 - a);
          metrics.accepted_per_agent_cycle = 1 + accepted;
          metrics.cycle_s = N * k / dt + (vb + k * ve) / 1000 + shipping;
          metrics.decode_tps = N * metrics.accepted_per_agent_cycle / metrics.cycle_s;
        }
      } else if (r.shape === 'parked_kv') {
        const bytes = get('kv_bytes_per_slot', Number.MIN_VALUE), slots = get('parked_slots', 1, undefined, true);
        if (known([bytes, slots, shipping])) {
          metrics.extra_slots = slots; metrics.parked_gib = bytes * slots / GiB; metrics.park_and_resume_s = shipping;
          metrics.park_s = transfers[0] ? transfers[0].seconds : null; metrics.resume_s = transfers[1] ? transfers[1].seconds : null;
          const budget = (r.resources || []).find(x => x.box === r.remote_box);
          if (!budget || r.remote_box === W.box || !Number.isFinite(v(budget.extra_host_gib))) add('parked_capacity', 'UNKNOWN', 'select a remote host allocation for the parked slots');
          else if (v(budget.extra_host_gib) < metrics.parked_gib) add('parked_capacity', 'FAIL', 'parked slots exceed the admitted incremental remote host allocation');
          if (r.transfers.length !== 2 || r.transfers[0].from !== W.box || r.transfers[0].to !== r.remote_box || r.transfers[1].from !== r.remote_box || r.transfers[1].to !== W.box || r.transfers.some(t => v(t.bytes) < bytes)) add('parked_transfer', 'FAIL', 'park and resume need one full-slot transfer in each exact direction');
        }
        // Capacity is not active decode throughput. There is intentionally no capacity->tok/s conversion here.
      } else if (r.shape === 'compaction') {
        const prompt = get('prompt_tokens', 1, undefined, true), summary = get('summary_tokens', 1, undefined, true), remote = get('remote_prefill_tps', Number.MIN_VALUE), sum = get('summary_tps', Number.MIN_VALUE), reread = get('reread_tps', Number.MIN_VALUE), local = get('local_prefill_tps', Number.MIN_VALUE), bytes = get('text_bytes_per_token', Number.MIN_VALUE);
        bool('fidelity', true);
        if (known([prompt, summary, remote, sum, reread, local, bytes, shipping]) && base > 0) {
          metrics.text_bytes = (prompt + summary) * bytes;
          if (r.transfers.length !== 2 || r.transfers[0].from !== W.box || r.transfers[0].to !== r.remote_box || r.transfers[1].from !== r.remote_box || r.transfers[1].to !== W.box || v(r.transfers[0].bytes) < prompt * bytes || v(r.transfers[1].bytes) < summary * bytes) add('compaction_text', 'FAIL', 'ship the prompt and summary text in their exact directions; small is not zero network latency');
          if (summary >= prompt) add('compaction_size', 'FAIL', 'summary does not compact the prompt');
          metrics.remote_elapsed_s = prompt / remote + summary / sum + shipping;
          metrics.saved_generation_slot_s = summary / (base / N);
          metrics.baseline_odie_slot_s = prompt / local + metrics.saved_generation_slot_s;
          metrics.reread_slot_s = summary / reread;
          metrics.odie_slot_s_saved = metrics.baseline_odie_slot_s - metrics.reread_slot_s;
          // Native-unit credit is admitted below, after every resource, payload and comparison check.
        }
      } else add('shape', 'FAIL', 'unknown network recipe law ' + r.shape);
      const power = networkPower(r.power || [], r.boxes, data), failed = checks.some(c => c.status === 'FAIL'), incomplete = checks.some(c => c.status === 'UNKNOWN');
      const status = failed ? 'FAIL' : incomplete ? 'UNKNOWN' : 'MODELED';
      if (r.shape === 'compaction') metrics.credited_slot_s_saved = status === 'MODELED' && compare.eligible && v(r.fidelity) === true && r.fidelity.tag === 'MEASURED' ? metrics.odie_slot_s_saved ?? null : null;
      const eligible = status === 'MODELED' && compare.eligible && base > 0 && Number.isFinite(metrics.decode_tps) && metrics.decode_tps > 0;
      const identity = { recipe: r.id, shape: r.shape, window: W.key, agents: N, mode: r.mode, evidence_config: r.evidence_config, metric: r.metric, catalog_schema: C.schema,
        profile_source: o.source || 'profile', profile_point: (o.source === 'ubench' ? W.ubench || {} : W.profile || {}).point,
        profile_identity: Object.fromEntries(['mode', 'evidence_config', 'metric'].map(k => [k, (o.source === 'ubench' ? W.ubench || {} : W.profile || {})[k] || null])),
        frontier_identity: Object.fromEntries(['mode', 'evidence_config', 'metric'].map(k => [k, (W.frontier || {})[k] || null])), frontier: (W.frontier || {}).ledger, baseline_tps: base,
        backend_supported: r.backend_supported, fidelity: r.fidelity, boxes: r.boxes, stage_boxes: r.stage_boxes || null, remote_box: r.remote_box || null,
        transfers: (r.transfers || []).map((t, i) => ({ recipe: t, observation: transfers[i].observation || null, inputs: transfers[i].inputs })), inputs: r.params, resources: r.resources, resource_plans: resources.plans, power: r.power };
      const attempts = (W.attempts || []).filter(a => a.network_recipe === r.id).map(a => {
        const receipt = a.network_receipt, matches = !!(receipt && receipt.recipe === r.id && networkSame(receipt.identity, identity) && networkSame(receipt.measured_identity, identity));
        return Object.assign({}, a, { matches_priced_identity: matches, measurement_status: matches ? 'same priced identity' : 'historical or unqualified identity; no current recipe credit' });
      });
      return { id: r.id, lever: r.label, kind: 'network', shape: r.shape, boxes: r.boxes, status, checks, resources, inputs, transfers, metrics, power, comparison: compare, eligible,
        at_frontier: eligible ? metrics.decode_tps : null, gain_pct: eligible && base > 0 ? 100 * (metrics.decode_tps / base - 1) : null,
        modeled_per_watt: eligible && power.watts ? metrics.decode_tps / power.watts : null,
        useful_work_per_joule: eligible && power.watts && v(r.fidelity) === true && r.fidelity.tag === 'MEASURED' && v(r.metric) === 'accepted_tokens_s' ? metrics.decode_tps / power.watts : null,
        evidence: attempts.map(a => a.id), attempts, test: r.test, falsifier: r.falsifier, src: r.src, prediction_identity: networkCanonical(identity) };
    });
  }

  // ---------- evidence, proposals and the frontier (config attempts, proposals, frontier, gate; RSI-top harvest H1, H4/H5, H8) ----------
  // An attempt touches a lever when it acts on one of the lever's classes ('*' = every kernel), or on the layout for the split
  // rebalance. Each touching attempt comes with the attempt it repairs and the one that repaired it, so a loss is never read
  // without its repair (the evidence subgraph), and an engine-wide attempt shows on every kernel lever.
  function winEvidence(W, lever) {
    const A = W.attempts || [], fixedBy = id => A.filter(a => a.repairs === id).map(a => a.id);
    const hit = a => (!a.network_recipe && (lever.kind === 'layout' ? !!a.layout : !a.layout && (lever.classes || []).some(c => a.classes.includes('*') || a.classes.includes(c))));
    return A.filter(a => a.outcome !== 'control' && hit(a)).map(a => ({ id: a.id, label: a.label, mechanism: a.mechanism, action: a.action, outcome: a.outcome, scope: a.scope,
      parent: a.parent || null, repairs: a.repairs || null, repaired_by: fixedBy(a.id), engine_wide: a.classes.includes('*'), tps: a.tps ? Object.fromEntries(Object.entries(a.tps).map(([k, x]) => [k, v(x)])) : null,
      sanity: a.sanity ? v(a.sanity) : null, result: a.result || null, source_scope: a.source_scope || null, power_w: a.power_w || null,
      quality_gate: a.quality_gate || null, crossover: a.crossover || null, diagnostic_profile: a.diagnostic_profile || null }));
  }
  const winTried = ev => (ev.length ? ev.reduce((m, a) => Object.assign(m, { [a.outcome]: (m[a.outcome] || 0) + 1 }), {}) : null);
  // whole-engine value against the measured frontier: the frontier is a tuned split, so a lever's modelled gain on its own
  // rebalanced split (its total_rebalanced over today's rebalanced total) applies to the frontier's measured tok/s; a gain
  // under the frontier's run-to-run noise is flagged, and nothing is accepted before the gate
  function winFrontier(W, L, network, data) {
    const F = W.frontier;
    if (!F || !F.tps) return null;
    const f = v(F.tps), noise = v(F.noise_pct), base = L.rebalanced.total;
    const rank = L.levers.concat(L.unmeasured.map(x => Object.assign({ no_floor: true }, x))).filter(x => x.kind !== 'layout').map(x => {
      const at = f * x.total_rebalanced / base, ev = winEvidence(W, x);
      return { lever: x.lever, kind: x.kind, no_floor: !!x.no_floor, at_frontier: at, gain: at - f, gain_pct: 100 * (at / f - 1), inside_noise: 100 * (at / f - 1) < noise, tried: winTried(ev), evidence: ev.map(a => a.id) };
    }).sort((a, b) => b.at_frontier - a.at_frontier);
    const comparison = winComparison(L.source === 'ubench' ? { ...W, profile: W.ubench } : W, null, data, L.agents), kernels = rank.map(x => Object.assign({}, x, { id: x.lever, origin: 'kernel', status: comparison.eligible ? 'MODELED' : 'UNKNOWN', eligible: comparison.eligible, comparison }));
    const candidates = kernels.concat((network || []).map(x => Object.assign({ origin: 'network' }, x))).sort((a, b) => Number(b.eligible) - Number(a.eligible) || (b.at_frontier || 0) - (a.at_frontier || 0));
    return { label: F.label, tps: f, noise_pct: noise, split: F.split, src: F.src, model_rebalanced: base, rank, comparison,
      candidates, shared_rank: candidates.filter(x => x.eligible), network: network || [], gate: W.gate || null };
  }
  // proposals: a lane's lever before it is built. Dedup is conservative: the same mechanism key and the same action (tokens in
  // any order) as an attempt or an earlier proposal is a duplicate; the same key with another action is related, never merged.
  // The expected effect comes from the model: a measured variant's lever, else the best lever over the proposal's classes
  const winActionKey = a => a.split(/\s+/).filter(Boolean).map(x => x.toUpperCase()).sort().join(' ');
  function winProposals(W, L) {
    const A = W.attempts || [], seen = [];
    return (W.proposals || []).map(p => {
      const same = x => x.mechanism === p.mechanism.key || (x.mechanism && x.mechanism.key === p.mechanism.key);
      const act = x => winActionKey(x.action) === winActionKey(p.action);
      const prior = A.filter(same).map(a => ({ id: a.id, outcome: a.outcome, duplicate: act(a) })).concat(seen.filter(same).map(q => ({ id: q.id, outcome: 'proposed', duplicate: act(q) })));
      seen.push(p);
      const vl = p.variant && (W.variants || []).find(x => x.id === p.variant), cand = L.levers.concat(L.unmeasured).filter(x => (vl ? x.kind === 'variant' && x.class === vl.class : x.kind !== 'variant' && (x.classes || []).some(c => p.classes.includes(c))));
      const best = cand.sort((a, b) => b.total - a.total)[0] || null, ev = [].concat(...cand.map(x => winEvidence(W, x))).filter((a, i, xs) => xs.findIndex(b => b.id === a.id) === i);
      return Object.assign({}, p, { dedup: prior.some(x => x.duplicate) ? 'duplicate' : prior.length ? 'related' : 'new', prior, expected: best ? { lever: best.lever, total: best.total, gain: best.gain, total_rebalanced: best.total_rebalanced } : null,
        evidence_on_targets: ev.map(a => ({ id: a.id, outcome: a.outcome })) });
    });
  }
  // a blank proposal for a lever, the model's fields filled in: what a lane copies into the config's proposals
  function winProposalTemplate(W, L, name) {
    const x = L.levers.concat(L.unmeasured).find(l => l.lever === name);
    if (!x) return null;
    const ev = winEvidence(W, x);
    return { id: '', lane: '', mechanism: { key: '', text: '' }, action: '', classes: x.classes || [], evidence: ev.length ? 'reuse' : 'exploration',
      reuse: ev.map(a => a.id + ' (' + a.outcome + ')'), exploration: '', test: '', falsifier: '',
      model: { lever: x.lever, kind: x.kind, ms_in_stage: x.ms_in_stage || null, xfloor_now: x.xfloor_now !== undefined ? x.xfloor_now : null, total: x.total, gain: x.gain, total_rebalanced: x.total_rebalanced },
      gate: W.gate ? W.gate.steps : [] };
  }
  // everything report.js --window prints: the fit (and its refits at the ends of the routing range), the agents grid, the bottleneck
  // stage's largest classes, the levers, and the what-ifs (each card's bandwidth with the config's stages; 1..n stages)
  function winReport(data, key, opts) {
    const W = winCfg(data, key), o = opts || {}, fit0 = winFit(data, key), N = o.agents || v(W.settings.plan_agents);
    if (fit0.K.eta === null) return { key: W.key, label: W.label, status: 'UNKNOWN', why: fit0.why, fit: fit0, plan: null };
    // the plan, levers and what-ifs start from the anchor point when one is named (o.anchor, else settings.plan_anchor)
    const aid = o.anchor !== undefined ? o.anchor : W.settings.plan_anchor ? W.settings.plan_anchor.v : null, fit = (aid && winAnchor(data, key, fit0, aid)) || fit0;
    // a measured profile (config `profile`) is the source when present, the ubench classes the fallback (o.source = 'ubench')
    const ca = W.profile && W.profile.curve_anchor ? v(W.profile.curve_anchor) : null;
    const prof = winProfFit(data, key, fit0, (ca && winAnchor(data, key, fit0, ca)) || fit), source = prof && o.source !== 'ubench' ? 'profile' : 'ubench', K = source === 'profile' ? prof.K : fit.K;
    const levOf = x => (source === 'profile' ? winProfLevers(data, key, Object.assign({ fit: prof }, x)) : winLevers(data, key, Object.assign({ fit }, x)));
    const agents = v(W.settings.agents_grid).filter(n => n >= v(W.min_batched_agents)).map(n => winStep(W, n, K));
    const plan = winStep(W, N, K), b = plan.bottleneck, [a0, b0] = plan.split[b];
    const label = id => (id === 'gap' ? 'gap: ' + W.profile.gap.label : id);
    const top = (source === 'profile' ? Object.entries(K.prof.costs(plan.rows, {}).parts(a0, b0, b, plan.split.length)).map(([id, ms]) => ({ class: label(id), ms }))
      : W.ubench.classes.map(c => ({ class: c.name, ms: winClassMs(W, c, plan.rows, K) / c.layers * W.layers.slice(a0, b0 + 1).filter(l => c.kind === null || l.kind === c.kind).length }))
        .concat(W.extras.filter(e => winExtraOn(e, b, plan.split.length)).map(e => ({ class: 'extra: ' + e.label, ms: K.extras[e.id] * (e.per === 'row' ? plan.rows : 1) })))).sort((x, y) => y.ms - x.ms);
    const unmeasured = winKinds(W).filter(k => !(k in W.ubench.kinds));
    const phis = W.ubench.routing_phi_range.v.map(p => { const f = winFit(data, key, { phi: p }); return { phi: p, K: f.K, plan: winStep(W, N, f.K), stage_fit_max_abs_pct: f.stage_fit_max_abs_pct, tps_check_max_abs_pct: f.tps_check_max_abs_pct }; });
    const cards = W.whatif_cards.concat(o.bw ? [{ box: null, id: 'what-if', name: 'a card at ' + o.bw + ' GB/s', bw: { v: o.bw } }] : []).map(c => ({ box: c.box, id: c.id, name: c.name, bw: v(c.bw), step: winStep(W, N, K, { bw: v(c.bw) }), rebalanced: winRebalance(W, N, K, { bw: v(c.bw) }), levers: levOf({ agents: N, bw: v(c.bw) }).levers.slice(0, 3).map(x => x.lever) }));
    const stages = Array.from(new Set(v(W.settings.whatif_stages).concat(o.stages ? [o.stages] : []))).sort((x, y) => x - y).map(s => winRebalance(W, N, K, {}, s));
    // the other reading of the unexplained per-row cost: in the kernels (gamma fitted), so every kernel lever pays more
    const gfit0 = winFit(data, key, { gamma_free: true }), gfit = (aid && winAnchor(data, key, gfit0, aid)) || gfit0, glev = winLevers(data, key, { fit: gfit, agents: N, xfloor: o.xfloor });
    const gamma_variant = { fit: gfit0, levers: glev.levers.map(x => ({ lever: x.lever, total: x.total, total_rebalanced: x.total_rebalanced })) };
    // with the profile as the source, the ubench reading of the same levers stays beside it (the fallback, and what changed)
    const ubench_levers = source === 'profile' ? winLevers(data, key, { fit, agents: N, xfloor: o.xfloor }).levers.map(x => ({ lever: x.lever, total: x.total, total_rebalanced: x.total_rebalanced })) : null;
    const profile = prof ? Object.assign({}, prof, { K: undefined }) : null, L = levOf({ agents: N, xfloor: o.xfloor }), network = winNetwork(data, key, { agents: N, source });
    return { key: W.key, label: W.label, source, fit: fit0, anchor: fit.anchor ? { id: fit.anchor, offset_ms: fit.K.offset } : null, profile, phis, gamma_variant, plan, top, agents, unmeasured_kinds: unmeasured,
      levers: L, evidence: Object.fromEntries(L.levers.concat(L.unmeasured).map(x => [x.lever, winEvidence(W, x)])), frontier: winFrontier(W, L, network, data), network, proposals: winProposals(W, L),
      template: o.propose ? winProposalTemplate(W, L, o.propose) : null, ubench_levers, whatif: { cards, stages } };
  }


  // Explicit tier-read fractions price the existing kernels, not storage or quality. The within-tier type mix still comes
  // from the heat map: aggregate routing counts do not identify a new per-expert layout or KL distribution.
  function recipeReadHeat(ctx, tier, reads) {
    if (reads === undefined) return {};
    if (!reads || !reads.shares || !reads.src || !reads.tag) return { why: 'decode_reads needs tagged, sourced all-read tier fractions' };
    if (ctx.hitR < 1) return { why: 'decode_reads with a dynamic GPU cache is not modelled' };
    const shares = reads.shares, ids = ctx.tiers.map(t => t.id), sums = new Float64Array(ids.length);
    if (Object.keys(shares).some(k => !ids.includes(k) || typeof shares[k] !== 'number' || !Number.isFinite(shares[k]) || shares[k] < 0 || shares[k] > 1))
      return { why: 'decode_reads has an unknown tier or invalid fraction' };
    if (Math.abs(Object.values(shares).reduce((a, x) => a + x, 0) - 1) > Number.EPSILON * tier.length)
      return { why: 'decode_reads fractions must sum to one over ALL routed reads' };
    for (let e = 0; e < tier.length; e++) sums[tier[e]] += ctx.score[e];
    if (ids.some((id, i) => (sums[i] && shares[id] === undefined) || (!sums[i] && shares[id] > 0)))
      return { why: 'decode_reads omits a populated tier or assigns reads to an empty tier' };
    return { heat: Float64Array.from(tier, (t, e) => sums[t] ? ctx.score[e] * shares[ids[t]] * ctx.kL / sums[t] : 0) };
  }

  const atPath = (x, path) => path.split('.').reduce((o, k) => o && typeof o === 'object' ? o[k] : undefined, x);
  const ordered = x => Array.isArray(x) ? x.map(ordered) : x && typeof x === 'object'
    ? Object.fromEntries(Object.keys(x).sort().map(k => [k, ordered(x[k])])) : x;
  const sameValue = (a, b) => JSON.stringify(ordered(a)) === JSON.stringify(ordered(b));
  const hasValues = x => x && typeof x === 'object' ? Object.values(x).some(hasValues) : x !== undefined;
  const draftGroup = (spec, r) => JSON.stringify(spec.draft_group.map(k => ordered(k === 'mtp_experts' ? storeKnobs(r).mtpExperts : atPath(r, k))));

  function recipeDecodeFit(data, r) {
    if (r.decode_profile === undefined) return {};
    const specs = data.decode_fits || {}, fits = data.cal && data.cal.recipe_decode || {};
    if (typeof r.decode_profile !== 'string' || !Object.prototype.hasOwnProperty.call(specs, r.decode_profile) ||
        !Object.prototype.hasOwnProperty.call(fits, r.decode_profile)) return { why: 'decode profile is unknown or not calibrated' };
    const spec = specs[r.decode_profile], fit = fits[r.decode_profile];
    if (!spec || !fit) return { why: 'decode profile is unknown or not calibrated' };
    if (typeof r.draft !== 'boolean' || hasValues(data.overrides) || spec.same.some(k => !sameValue(atPath(r, k), atPath(fit.reference, k))) ||
        (r.knobs && Object.keys(r.knobs).length) || !sameValue(data.knobs, fit.knobs) || !sameValue(data.overrides, fit.overrides))
      return { why: 'decode profile is outside its measured recipe/workload/knob envelope' };
    const group = draftGroup(spec, r), flag = atPath(r, spec.lever), flat = atPath(r, spec.flat_lever);
    if (!fit.observed.some(x => x.group === group && x.flag === flag) || !fit.flat.includes(flat))
      return { why: 'decode profile bank/draft/placement/lever combination has not been measured' };
    return { host: { v: fit.host + (flag === spec.baseline ? 0 : fit.penalty), tag: 'FIT', src: spec.src + '; ' + spec.train.join(', ') },
      draft: { v: fit.draft[group], tag: 'FIT', src: 'in-load n4/n4b versus n0 twins for the measured combined bank/draft/placement configuration; ' + fit.group_runs[group].join(', ') } };
  }

  function fitRecipeDecode(data) {
    return Object.fromEntries(Object.entries(data.decode_fits || {}).filter(([, spec]) => spec.knobs && typeof spec.knobs === 'object' && !Array.isArray(spec.knobs) && Object.keys(spec.knobs).length && !hasValues(data.overrides) && sameValue(data.knobs, spec.knobs)).map(([key, spec]) => {
      const rows = spec.training.map(x => {
        const p = predictRecipe(data, Object.assign({}, x.recipe, { decode_profile: undefined }));
        const d = x.decode_tps, on = [d.n4, d.n4b].filter(x => x !== undefined);
        if (p.status !== 'PREDICTED' || !Number.isFinite(d.n0) || d.n0 <= 0 || !on.length || on.some(x => !Number.isFinite(x) || x <= 0))
          throw new Error('invalid decode fit training point: ' + x.id);
        return { id: x.id, recipe: x.recipe, residual: 1000 / d.n0 - p.ms_per_token.total + p.ms_per_token.host,
          flag: atPath(x.recipe, spec.lever), group: draftGroup(spec, x.recipe), mult: on.reduce((a, x) => a + x, 0) / on.length / d.n0 };
      });
      const pair = spec.pair.map(id => rows.find(x => x.id === id));
      if (pair.some(x => !x) || pair[0].flag !== spec.baseline || pair[1].flag === spec.baseline)
        throw new Error('invalid decode fit residual pair: ' + key);
      const penalty = pair[1].residual - pair[0].residual;
      const host = rows.reduce((a, x) => a + x.residual - (x.flag === spec.baseline ? 0 : penalty), 0) / rows.length;
      if (!Number.isFinite(host) || host < 0) throw new Error('invalid decode fit residual: ' + key);
      const groups = [...new Set(rows.map(x => x.group))];
      return [key, { host, penalty, reference: JSON.parse(JSON.stringify(rows[0].recipe)), knobs: JSON.parse(JSON.stringify(data.knobs)), overrides: data.overrides === undefined ? undefined : JSON.parse(JSON.stringify(data.overrides)),
        draft: Object.fromEntries(groups.map(g => { const xs = rows.filter(x => x.group === g); return [g, xs.reduce((a, x) => a + x.mult, 0) / xs.length]; })),
        group_runs: Object.fromEntries(groups.map(g => [g, rows.filter(x => x.group === g).map(x => x.id)])),
        observed: rows.map(x => ({ group: x.group, flag: x.flag })), flat: [...new Set(rows.map(x => atPath(x.recipe, spec.flat_lever)))] }];
    }));
  }

  // ---------- prediction of ONE concrete recipe ----------
  // Maps the recipe onto the planner: per-expert types from the store, per-expert tiers from banks + placement, then evaluate().
  // Returns decode tok/s (with and without the draft), VRAM post-warm-up and peak range, predicted node MemFree, and every
  // assumption made. status 'UNMODELLED' (numbers null, why given) when the planner has no model for the recipe: never a guess.
  function predictRecipe(data, r) {
    if (r && r.kind === 'network') return networkCheck(data, r).prediction;
    if (r && r.kind === 'dllm') return predictDllm(data, r);
    const A = [];
    const out = { status: 'UNMODELLED', why: null, decode_tps: null, decode_tps_nodraft: null, prefill_tps: null,
      prefill_why: 'the planner has no prefill model (decode only)', vram_post_gib: null, vram_peak_gib: null, vram_peak_range_gib: null,
      node_memfree_gib: null, node_memfree_min_gib: null, tiers: null, assumptions: A };
    const un = why => Object.assign(out, { why });
    const box = data.boxes[r.box], m = data.models[r.model], E = data.engines[r.engine];
    if (!box || !m || !E) return un('unknown box, model or engine');
    if (r.cpu_expert_evidence) {
      out.cpu_expert_evidence = Object.assign({}, r.cpu_expert_evidence, { reference_only: true,
        residency_matches: !!r.expert_residency && r.expert_residency.mode === r.cpu_expert_evidence.applies_to_mode });
      A.push(r.cpu_expert_evidence.calibration_policy);
    }
    if (E.no_solver) return un(E.unmodelled_why || (r.engine + ' has no planner model (it sizes and places its own caches)'));
    if (E.calibrated_builds && E.calibrated_builds.indexOf(r.engine_build) < 0)
      return un(r.engine + ' build ' + (r.engine_build || '(not given)') + ' is not a calibrated build (' + E.calibrated_builds.join(', ') + '): its host and lane rates are unmeasured');
    // a recorded recipe whose label alone states its per-card placement takes it from data.placements (the ledger stays as recorded)
    const st = r.store || {}, pl = r.placement || (data.placements && data.placements[r.id]) || {};
    if (pl.mode === 'none' || pl.mode === 'interleave') return un('no per-expert placement (' + pl.mode + (pl.note ? ': ' + pl.note : '') + '): the planner places whole experts, the kernel spread these pages');
    const hdd = (r.files || []).filter(fl => (fl.access === 'on_file' || fl.access === 'mmap') && mountKind(box, fl.path).kind === 'hdd');
    if (hdd.length) return un('tensors read while serving sit on a rotational disk (' + hdd.map(x => x.holds).join(', ') + '): the planner has no seek-bound read model');
    const split = r.engine === 'ik' && !!st.split;
    const nf = r.numa || {};
    if (nf.headroom_gib !== undefined && nf.headroom_gib !== null) A.push('dram-fill at headroom ' + nf.headroom_gib + ' GiB per node: DRAM experts = sum(node free at plan time - headroom)');
    const df = recipeDecodeFit(data, r);
    if (df.why) return un(df.why);
    const { ctx } = recipeContext(data, r);
    A.push('heat: hot order by ' + ctx.planName + ', reads scored on ' + ctx.scoreName + (ctx.planName === ctx.scoreName
      ? ' (the same map: in sample, so the CPU and PMem read shares are a floor; live routes read the cold tiers more, e.g. the GLM q4x route dump: PMem read share 2.9 % measured vs 1.2 % scored in sample)' : ''));
    out.heat_maps = { plan: heatCite(m, ctx.planName), score: heatCite(m, ctx.scoreName) };
    if (ctx.bad) return un('non-expert weights + KV do not fit ' + ctx.g0.name);
    if (r.draft && !ctx.draftOn) A.push('draft requested but off in the planner: ' + ctx.notes.filter(x => /draft|MTP/.test(x)).join(' '));
    const N = m.L * m.E;
    const ty = storeTypes(data, r, ctx);
    if (ty.why) return un(ty.why);
    A.push(ty.note);
    const { type, qi } = ty;
    // per-expert tier: GPU banks (hottest first), PMem holds the coldest (coldest placement), DRAM to its cap, then PMem, then NVMe
    const sz = ctx.types.map(q => typeBytes(data, r.model, q));
    const tid = id => ctx.tiers.findIndex(t => t.id === id);
    const gi = ctx.tiers.indexOf(ctx.g0), di = tid('dram'), pi = tid('pmem'), ni = tid('nvme');
    const left = ctx.tiers.map(t => t.cap);
    const tier = new Int8Array(N).fill(-1);
    if (r.banks_experts !== undefined && r.banks_experts !== null) A.push('GPU: the ' + r.banks_experts + ' hottest experts (' + ctx.planName + ')' + (ctx.hitR < 1 ? ', dynamic cache realizing ' + (100 * ctx.hitR).toFixed(0) + ' % of their heat' : ''));
    else if (r.gpu_split && !(splitPosts(ctx, r) || {}).bad) { const sp = splitPosts(ctx, r); left[gi] = sp.banks_sum * GiB; A.push('GPU: hottest experts up to ' + sp.banks_sum.toFixed(2) + ' GiB banks over ' + sp.rows.length + ' cards (gpu_split ' + sp.rows.map(x => x.banks_gib).join(' + ') + '; layers run one card after the other, so one GPU tier)'); }
    else if (r.banks_gib !== undefined && r.banks_gib !== null) { left[gi] = Math.min(left[gi], r.banks_gib * GiB); A.push('GPU: hottest ' + (split ? 'hot-set ' : '') + 'experts up to ' + r.banks_gib + ' GiB banks'); }
    else A.push('GPU: no banks given, hottest experts fill the planner GPU capacity ' + (left[gi] / GiB).toFixed(2) + ' GiB');
    let dramCap = left[di];
    if (nf.nodes && nf.nodes.every(x => x.experts_gib !== undefined)) { dramCap = nf.nodes.reduce((a, x) => a + x.experts_gib, 0) * GiB; A.push('DRAM: ' + nf.nodes.map(x => x.experts_gib).join(' + ') + ' GiB experts per the node plan'); }
    if (r.ram_tier_gib !== undefined && r.ram_tier_gib !== null) { dramCap = r.ram_tier_gib * GiB; A.push('DRAM: expert RAM tier ' + r.ram_tier_gib + ' GiB'); }
    // a lane probe or part that adds DRAM (knob lane.add_gib.dram) also adds to a DRAM tier the recipe states (the box usable
    // figure already carries it through context)
    const addDram = (ctx.kn.lane && ctx.kn.lane.add_gib && ctx.kn.lane.add_gib.dram) || 0;
    if (addDram && ((nf.nodes && nf.nodes.every(x => x.experts_gib !== undefined)) || (r.ram_tier_gib !== undefined && r.ram_tier_gib !== null))) { dramCap += addDram * GiB; A.push('DRAM: +' + addDram.toFixed(1) + ' GiB (lane add_gib.dram)'); }
    // a stated expert RAM tier (ram_tier_gib, e.g. a window that stops the box's services) replaces the usable estimate, up to DRAM total
    left[di] = (r.ram_tier_gib !== undefined && r.ram_tier_gib !== null) ? Math.min(dramCap, ctx.tiers[di].total) : Math.min(left[di], dramCap);
    if (pl.mode === 'coldest') {
      if (pi < 0) return un('placement coldest-to-PMem on a box without PMem');
      let pm = 0;
      for (let i = N - 1; i >= 0; i--) { const e = ctx.order[i], b = sz[type[e]]; if (pm + b > pl.pmem_gib * GiB) break; tier[e] = pi; pm += b; }
      left[pi] -= pm; left[di] = Infinity;
      A.push('PMem: the coldest ' + pl.pmem_gib + ' GiB of experts (planned placement' + (pl.note ? ', ' + pl.note : '') + '), the rest in DRAM');
    }
    // gpu_layers: llama.cpp -ot places whole MoE layers' expert tensors on the GPU (pl.layers = 0-based MoE layer indices),
    // not the hottest experts; every expert of those layers is on the GPU, no other expert is
    // --moe-cache-mib: the cache is part of the GPU banks (gpu_split banks_gib = static experts + cache), so static experts get the rest
    const cacheGiB = r.moe_cache_gib || 0;
    if (cacheGiB) {
      if (!E.recipe_rates || !E.recipe_rates.cache_link_gbs) return un('moe_cache_gib: engine ' + r.engine + ' has no cache rates (recipe_rates.cache_link_gbs)');
      left[gi] -= cacheGiB * GiB;
      if (left[gi] < -0.01 * GiB) return un('moe_cache_gib ' + cacheGiB + ' exceeds the GPU banks');
    }
    // pl.cards = {gpu id: layers} puts layers on the engine's other cards (a banks-only card through its second backend)
    const byLayers = pl.mode === 'gpu_layers';
    if (byLayers) {
      const sets = Object.entries(Object.assign(pl.layers ? { [ctx.g0.gpu.id]: pl.layers } : {}, pl.cards || {}));
      if (!sets.length) return un('placement gpu_layers needs layers (or cards) in 0..' + (m.L - 1) + ' (MoE layer indices)');
      for (const [id, layers] of sets) {
        const ti = ctx.tiers.findIndex(t => t.kind === 'gpu' && t.engineOk && t.gpu.id === id), t = ctx.tiers[ti];
        if (ti < 0) return un('placement gpu_layers: ' + r.engine + ' drives no card ' + id + ' on ' + r.box);
        const bad = layers.filter(l => !(l >= 0 && l < m.L));
        if (bad.length || !layers.length) return un('placement gpu_layers needs layers in 0..' + (m.L - 1) + ' (MoE layer indices)' + (bad.length ? ', got ' + bad.join(' ') : ''));
        for (const l of layers) for (let x = 0; x < m.E; x++) {
          const e = l * m.E + x;
          if (t.drive && t.drive.types.indexOf(ctx.types[type[e]]) < 0) return un('placement gpu_layers: ' + ctx.types[type[e]] + ' is not a type seen on ' + id + ' (' + t.drive.backend + ': ' + t.drive.types.join('/') + ')');
          tier[e] = ti; left[ti] -= sz[type[e]];
        }
        if (left[ti] < -0.01 * GiB) return un('placement gpu_layers: ' + layers.length + ' layers of experts exceed the ' + (ti === gi ? 'GPU banks' : id + ' capacity') + ' by ' + (-left[ti] / GiB).toFixed(2) + ' GiB');
        A.push('GPU ' + id + ': every expert of MoE layers ' + layers.join(',') + ' (gpu_layers placement, not the hottest experts)');
      }
      if (pl.src) A.push('placement ' + (pl.tag || 'untagged') + ': ' + pl.src);
    }
    let ng = 0;
    for (const e of ctx.order) {
      if (tier[e] >= 0) continue;
      const b = sz[type[e]];
      const gpuOk = byLayers ? false : (r.banks_experts !== undefined && r.banks_experts !== null) ? ng < r.banks_experts : left[gi] >= b;
      if (gpuOk && (!split || !st.hot_type || type[e] === qi(st.hot_type))) { tier[e] = gi; left[gi] -= b; ng++; continue; }
      const t = [di, pi, ni].find(x => x >= 0 && left[x] >= b);
      if (t === undefined) return un('the experts do not fit the tiers (GPU banks + DRAM cap + PMem + NVMe)');
      tier[e] = t; left[t] -= b;
    }
    // every file loaded into RAM (pinned / -lm none): experts the DRAM cap cannot hold have no NVMe tier to fall to
    const files = r.files || [];
    if (ni >= 0 && files.length && files.every(fl => fl.access === 'loaded')) {
      let spill = 0;
      for (let e = 0; e < N; e++) if (tier[e] === ni) spill += sz[type[e]];
      if (spill > 0) out.ram_overflow_gib = spill / GiB;
      if (spill > 0) return un('every file is loaded into RAM (access loaded), but ' + (spill / GiB).toFixed(1) + ' GiB of experts exceed the DRAM cap (' + (dramCap / GiB).toFixed(1) + ' GiB; set ram_tier_gib if the box frees more)');
    }
    // route trim (recipe.trim [{layers: [lo, hi] model ids, keep}]): picks that miss VRAM (experts off the GPU, and a dynamic
    // GPU cache's misses) drop ranks keep..k-1 on those layers, so their reads fall to keep / k (INFERRED: a miss is independent
    // of its rank); GPU hits stay whole. The trim's KL is the layer map's (trimKl) on the same miss share.
    let hEval, trimMissT = 0;
    if (r.trim) {
      const ts = trimSupport(data, r);
      if (ts.status === 'no') return un(ts.src);
      const keep = trimKeeps(m, r.trim), hs = ctx.score, Ex = m.E, miss = new Float64Array(m.L);
      let cacheCut = 0;
      hEval = Float64Array.from(hs);
      for (let e = 0; e < N; e++) {
        const l = (e / Ex) | 0, gpu = ctx.tiers[tier[e]].kind === 'gpu';
        miss[l] += (gpu ? (1 - ctx.hitR) * hs[e] : hs[e]) / m.k;
        if (!gpu && keep[l] < m.k) hEval[e] = hs[e] * keep[l] / m.k;
        if (gpu && keep[l] < m.k && ctx.hitR < 1 && ctx.missTi >= 0) {
          const q = ctx.types[type[e]], cut = (1 - ctx.hitR) * hs[e] * (1 - keep[l] / m.k);
          trimMissT += cut * sz[type[e]] / ctx.rawRate(ctx.tiers[ctx.missTi], q); cacheCut += cut;
        }
      }
      const tk = trimKl(data, r.model, keep, miss), drop = (hs.reduce((a, x, e) => a + x - hEval[e], 0) + cacheCut) / ctx.kL;
      out.trim = { kl: tk.kl, why: tk.why, reads_dropped: drop, miss_by_layer: Array.from(miss, x => +x.toFixed(4)), engine: ts };
      A.push('route trim ' + r.trim.map(t => 'keep ' + t.keep + ' on ' + t.layers.join('-')).join(', ') + ' (picks off the GPU only): ' + (100 * drop).toFixed(1) + ' % of routed reads dropped; trim KL ' + (tk.kl === null ? 'UNKNOWN (' + tk.why + ')' : tk.kl.toFixed(4) + ' PREDICTED from the layer map'));
    }
    if (r.decode_reads !== undefined && (r.trim || cacheGiB)) return un('decode_reads with route trim or a MoE cache is not modelled');
    const routing = recipeReadHeat(ctx, tier, r.decode_reads);
    if (routing.why) return un(routing.why);
    if (routing.heat) A.push('decode timing uses sourced all-read tier fractions; within-tier type mix remains an estimate from ' + ctx.scoreName + '; storage, quality and prefill heat are unchanged');
    const ev = evaluate(ctx, { tier, type }, hEval, routing.heat);
    // layers the token crosses to each card reached through a second backend (its picks / k): the weight of its hop term
    ctx.tiers.forEach((t, i) => { if (t.drive && ev.heatT[i] > 0) (out.cross_layers = out.cross_layers || {})[t.gpu.id] = ev.heatT[i] / m.k; });
    const RR = E.recipe_rates || {}, rr = k => RR[k] ? v(RR[k]) : null;
    const h = ctx.score, gbw = ctx.g0.bw * ctx.g0.eta;
    // a trimmed cache miss is not read: evaluate() read every GPU-tier pick's miss share from DRAM, so take the dropped part out
    let gpuT = ev.gpuT, cpuT = ev.cpuT - trimMissT, hostT = df.host ? df.host.v / 1000 : ctx.host;
    if (df.host) { out.decode_fit = { profile: r.decode_profile, host: df.host, draft: df.draft }; A.push('scoped residual replaces BOTH old host and split terms; placement/pinning remains confounded, not an identified host-compute penalty'); }
    if (cacheGiB) {
      // every host-expert MUL_MAT_ID of a decode batch (<= 32 tokens) runs on the GPU: hits read the cache, misses are uploaded
      // first (PR #29887, READ); the cache holds the hottest host experts it can, realizing cache_realize of their heat (ESTIMATE)
      const rho = rr('cache_realize'), link = rr('cache_link_gbs') * 1e9;
      let c = 0, hit = 0, miss = 0;
      for (const e of ctx.order) {
        if (tier[e] !== di) continue;
        const b = sz[type[e]], rb = h[e] * b;
        if (c + b <= cacheGiB * GiB) { c += b; hit += rho * rb; miss += (1 - rho) * rb; } else miss += rb;
      }
      cpuT -= ev.timeT[di];
      gpuT += (hit + miss) / gbw + miss / link;
      out.moe_cache = { gib: c / GiB, hit_share: hit / (hit + miss), miss_gb_per_token: miss / 1e9, upload_ms: 1000 * miss / link };
      A.push('moe cache ' + cacheGiB + ' GiB: hottest host experts, hit share ' + (100 * out.moe_cache.hit_share).toFixed(0) + ' % (realize ' + rho + ' ESTIMATE), misses ' + (miss / 1e9).toFixed(2) + ' GB/token uploaded at ' + rr('cache_link_gbs') + ' GB/s (ESTIMATE); no CPU expert kernel in decode');
    }
    if (r.split_mode === 'tensor') {
      // -sm tensor: every card reads its share of the dense weights at once; two reductions per layer cross PCIe (ESTIMATE)
      const nc = ctx.tiers.filter(t => t.kind === 'gpu' && t.engineOk && !t.banks).length;
      gpuT -= ctx.denseTime * (1 - 1 / nc);
      hostT += 2 * rr('layers_total') * rr('allreduce_ms') / 1000;
      A.push('-sm tensor over ' + nc + ' cards: dense read / ' + nc + ', + ' + (2 * rr('layers_total') * rr('allreduce_ms')).toFixed(1) + ' ms/token reductions (ESTIMATE)');
    }
    const T = hostT + (ctx.overlap ? Math.max(gpuT, cpuT) + (1 - ctx.overlap) * Math.min(gpuT, cpuT) : gpuT + cpuT);
    out.status = 'PREDICTED';
    // the draft speed-up at this recipe's CPU share of the token (ledgerDrafts): x mult / (1 + cpuW x CPU expert time / T)
    const dm = df.draft && ctx.draftOn ? df.draft.v : ctx.mult * T / (T + ctx.cpuW * cpuT);
    out.decode_tps = dm / T;
    out.decode_tps_nodraft = 1 / T;
    if (ctx.draftOn) A.push('draft speed-up x' + dm.toFixed(3) + (df.draft ? ' (measured combined configuration twins)' : ctx.cpuW ? ' (box x model fit x' + ctx.mult.toFixed(3) + ' at no CPU expert time, / (1 + ' + ctx.cpuW.toFixed(3) + ' x CPU share ' + (cpuT / T).toFixed(3) + '))' : ' (box x model fit)'));
    // an external draft file served to a model whose planner draft is its own MTP head: another draft, its speed-up unmodelled
    const xd = ctx.draftOn && m.draft.kind === 'mtp' ? (r.files || []).filter(fl => /draft/i.test(fl.holds || '')) : [];
    if (xd.length) {
      out.decode_tps = null;
      out.draft_why = 'the recipe serves an external draft (' + xd.map(x => x.holds).join(', ') + '); the planner draft of ' + m.short + ' is its own MTP head, so the speed-up of this draft is unmodelled (the no-draft prediction stands)';
      A.push('draft: ' + out.draft_why);
    }
    out.ms_per_token = { host: 1000 * hostT, gpu: 1000 * gpuT, cpu: 1000 * cpuT, total: 1000 * T };
    // the token's time per hardware lane (before GPU/CPU overlap; src/advisor.js): GPU reads on the VRAM lane; each CPU-side
    // tier's expert bytes on its read ceiling (<tier>_bw) or on the CPU kernel rate (cpu), whichever is slower, scaled to cpuT
    {
      const he = routing.heat || hEval || h, lane = { vram_bw: 1000 * gpuT, dram_bw: 0, pmem_bw: 0, nvme_bw: 0, cpu: 0 };
      const at = (t, q, b) => { const tt = ctx.tiers[t]; lane[ctx.cpuRate(ctx.types[q]) < tt.bw ? 'cpu' : tt.id + '_bw'] += b / ctx.rawRate(tt, ctx.types[q]); };
      for (let e = 0; e < N; e++) {
        const b = he[e] * sz[type[e]], t = tier[e];
        if (ctx.tiers[t].kind !== 'gpu') at(t, type[e], b);
        else if (ctx.hitR < 1 && ctx.missTi >= 0) at(ctx.missTi, type[e], (1 - ctx.hitR) * b);
      }
      const raw = lane.dram_bw + lane.pmem_bw + lane.nvme_bw + lane.cpu, sc = raw > 0 ? 1000 * cpuT / raw : 0;
      ['dram_bw', 'pmem_bw', 'nvme_bw', 'cpu'].forEach(k => lane[k] *= sc);
      out.ms_per_lane = Object.assign({ host: 1000 * hostT }, lane);
    }
    const held = new Float64Array(ctx.tiers.length);
    for (let e = 0; e < N; e++) held[tier[e]] += sz[type[e]];
    if (rr('host_extra_gib') !== null && files.length && files.every(fl => fl.access === 'loaded')) {
      out.host_extra_gib = rr('host_extra_gib');
      out.pinned_driver_extra_gib = rr('pinned_driver_extra_gib') || 0;
      out.container_anon_gib = rr('container_anon_gib') || 0;
      out.host_peak_gib = held[di] / GiB + out.host_extra_gib + out.pinned_driver_extra_gib;
    }
    out.tiers = ctx.tiers.map((t, i) => ({ id: t.id, experts_gib: held[i] / GiB, read_share: ev.heatT[i] / ctx.kL })).filter(x => x.experts_gib > 0);
    out.lowbit_share = lowShare(ctx, { tier, type });
    // the store rule sets the layout; a recipe lowbit_share that disagrees with it moves only the quality check, never the speed
    if (r.lowbit_share !== undefined && r.lowbit_share !== null && Math.abs(r.lowbit_share - out.lowbit_share) > 0.02)
      A.push('speed is predicted on the store layout\'s 2-bit share ' + (100 * out.lowbit_share).toFixed(1) + ' % (' + ctx.scoreName + '), not the recipe lowbit_share ' + (100 * r.lowbit_share).toFixed(1) + ' %, which feeds only the quality check: to move the speed, change the store (hot set or read_share_by_type)');
    // knobs the decode model does not see are named, so a lever that predicts no change is not read as measured no change
    const stTypes = [...new Set((st.types || []).concat(Object.keys(st.read_share_by_type || {}), st.hot_type || [], st.cold_type || []))];
    const noRepack = stTypes.filter(q => !data.types[q + '_r8'] && !data.types[q + '_r4']);
    if (r.rtr && noRepack.length) A.push('-rtr: no repacked kernel row for ' + noRepack.join(', ') + ' (the type table has repacked rows only as <type>_r4/_r8), so decode is predicted at the unrepacked rate');
    const phys = v(box.cpu.phys), used = v(box.cpu.used);
    if (r.threads !== undefined && (r.threads > phys) !== (used > phys))
      A.push('threads ' + r.threads + (r.threads > phys ? ' use' : ' leave out') + ' SMT siblings, unlike the ' + used + '-thread runs the box f is fitted on: thread count is not in the decode model, so the prediction does not move with it');
    const lay = r.numa && typeof r.numa === 'object' ? r.numa.layout : undefined;
    const layT = lay !== undefined && box.numa_layout ? box.numa_layout[lay] : undefined;
    if (layT && lay !== 'interleave') A.push('NUMA page layout "' + lay + '": DRAM expert CPU time x' + v(layT).toFixed(3) + ' vs the interleaved pages the box f is fitted on (' + layT.tag + ')');
    else if (lay !== undefined && lay !== 'interleave')
      A.push('NUMA page layout "' + lay + '" is not in the decode model' + (box.numa_layout ? ' (measured layouts on ' + r.box + ': ' + Object.keys(box.numa_layout).join(', ') + ')' : '') + ': the box f is fitted on interleaved expert pages, so the prediction does not move with the layout');
    // ik build flag GGML_EXPERT_CHUNKING: measured on the box (ik_expert_chunking_on), the decode model assumes OFF
    const chunk = box.ik_expert_chunking_on;
    if (chunk && r.engine === 'ik' && r.expert_chunking !== 'off')
      A.push((r.expert_chunking === 'on' ? 'GGML_EXPERT_CHUNKING=ON' : 'expert_chunking not declared') + ': ik packages on ' + r.box + ' must build GGML_EXPERT_CHUNKING=OFF (ON measured decode x' + v(chunk).toFixed(3) + ', ' + chunk.tag + '); the prediction assumes OFF and does not scale for ON');
    const host =ctx.tiers.map(t => t.kind !== 'gpu');
    const touched = n => { let u = 0; for (let e = 0; e < N; e++) if (host[tier[e]]) u += sz[type[e]] * (1 - Math.pow(1 - Math.min(1, h[e]), n)); return u; };
    const pf = prefillModel(data, r, out.tiers.filter(x => !/^gpu/.test(x.id)).reduce((a, x) => a + x.experts_gib, 0), touched, out.tiers.filter(x => x.id === 'pmem').reduce((a, x) => a + x.experts_gib, 0));
    if (pf) {
      const tok = r.prefill_tokens || 2048;
      out.prefill_tps = pf.tps(tok);
      // nothing to upload (every expert GPU-resident): prefill is GPU compute-bound, which the planner does not model
      if (!isFinite(out.prefill_tps)) { out.prefill_tps = null; pf.cpu_src = 'UNMODELLED: every routed expert is GPU-resident, so prefill is GPU compute-bound (no GPU prefill model)'; }
      const fin = t => { const x = pf.tps(t); return x !== null && isFinite(x) ? x : null; };
      out.prefill_tps_at = { 2048: fin(2048), 8192: fin(8192), 16384: fin(16384), 32768: fin(32768) };
      out.prefill_why = out.prefill_tps === null ? pf.cpu_src : (pf.offload_on && Math.min(tok, pf.ub) >= pf.offload_batch ? 'GPU offload at ' + tok + ' tokens: ' + pf.upload_gib.toFixed(1) + ' GiB ' + (pf.upload_touched ? 'routed CPU-resident experts' : 'CPU-resident experts') + ' per ubatch (-ub ' + pf.ub + ') at ' + pf.h2d_gbs.toFixed(2) + ' GB/s (' + pf.h2d + ') = ' + pf.upload_s_per_ubatch.toFixed(1) + ' s' : 'CPU path (' + (pf.offload_on ? 'below the offload batch' : 'offload off') + '), ' + pf.cpu_tag);
      delete pf.tps;
      out.prefill = Object.assign(pf, { tokens: tok });
      A.push('prefill at ' + tok + ' tokens: ' + out.prefill_why + (pf.crossover_tokens ? '; offload crossover ' + Math.round(pf.crossover_tokens) + ' tokens' : '; offload never beats the CPU path within one ubatch'));
    }
    // VRAM: reserves + fixed parts + banks (as checkRecipe predicts it), peak = post + the engine's transient margin
    let post = ctx.g0.reserves.reduce((a, x) => a + x.gib, 0);
    ctx.fixed.filter(x => x.tier === ctx.g0.id && x.kind !== 'margin').forEach(x => post += x.gib);
    if (E.kv_includes_margin) post -= E.kv_includes_margin;
    post += (r.banks_gib !== undefined && r.banks_gib !== null) ? r.banks_gib : ev.capUse[gi] / GiB;
    post += offloadVram(data, r);
    const spv = splitPosts(ctx, r);
    if (spv && !spv.bad) { post = spv.worst.post_gib; out.vram_per_card = spv.rows; }
    out.vram_post_gib = post;
    const mg = marginOf(data, r.engine, split);
    if (mg && !mg.self_sizing && mg.hi !== null && mg.hi !== undefined) {
      out.vram_peak_range_gib = [post + mg.lo, post + mg.hi];
      out.vram_peak_gib = post + (mg.candidate !== null && mg.candidate !== undefined ? mg.candidate : mg.hi);
      A.push('VRAM peak = post-warm-up + margin ' + (mg.candidate !== null && mg.candidate !== undefined ? mg.candidate + ' (candidate; bounds ' + mg.lo + '-' + mg.hi + ')' : mg.hi + ' (upper bound)'));
    }
    // node MemFree: the numa check's own prediction, without the run's observations
    const nfp = Object.assign({}, nf); delete nfp.observed_memfree_gib; delete nfp.observed_clean;
    if (!nf.headroom_gib && !nf.nodes) nfp.dram_experts_gib = ev.capUse[di] / GiB;
    const nc = numaCheck(data, { box: r.box, model: r.model, engine: r.engine, split: !!st.split, banks_gib: r.banks_gib, numa: nfp });
    if (nc.pred) { out.node_memfree_gib = nc.pred.memfree_gib; out.node_memfree_min_gib = Math.min.apply(null, nc.pred.memfree_gib); A.push('node MemFree with load extra ' + nc.pred.load_extra_gib + ' GiB (' + nc.pred.load_extra_tag + ')'); out.load_extra = nc.pred.load_extra; }
    out.evidence = predictionEvidence(data, r, ctx, out, { tier, type }, mg);
    return out;
  }

  // every term the predicted numbers rest on, per metric, with its evidence class (data.meta.evidence) and source. decode = the
  // no-draft terms + the draft speed-up; prefill = the CPU cell or the upload path (UNMODELLED = one unknown term); VRAM = the
  // GPU reserves and fixed parts (the recipe's own banks are an input, not a term) + the peak margin
  function predictionEvidence(data, r, ctx, out, lay, mg) {
    const box = ctx.box, m = ctx.m, E = ctx.E, cal = (data.cal && data.cal.boxes[r.box]) || {}, N = m.L * m.E, T = (n, x) => term(data, n, x);
    const used = new Set(), cpuTypes = new Set(), allTypes = new Set();
    for (let e = 0; e < N; e++) {
      const t = ctx.tiers[lay.tier[e]], q = ctx.types[lay.type[e]];
      allTypes.add(q);
      if (ctx.score[e] > 0) { used.add(t.id); if (t.kind !== 'gpu') cpuTypes.add(q); }
    }
    const gpuUsed = [...used].some(id => /^gpu/.test(id)), cpuUsed = cpuTypes.size > 0;
    const D = [out.decode_fit ? T('scoped decode residual ms/token (placement/pinning confounded)', out.decode_fit.host) : T('host overhead ms/token', hostSrc(data, r.box, r.model, r.engine))];
    const spT = !out.decode_fit && ctx.split ? splitSrc(data, r.box, r.model) : null;
    if (r.decode_reads) {
      Object.entries(r.decode_reads.shares).forEach(([id, share]) => D.push(T('all-read fraction on ' + id, { v: share, tag: r.decode_reads.tag, src: r.decode_reads.src })));
      D.push(T('within-tier expert type mix', { v: 1, tag: 'ESTIMATE', src: 'aggregate tier fractions preserve the score map type mix inside each tier; not measured per-expert decode routes' }));
    }
    if (spT) D.push(T('split GGUF host ms/token (' + r.model + ')', spT));
    D.push(T('GPU nameplate bandwidth GB/s (' + ctx.g0.name + ')', ctx.g0.gpu && ctx.g0.gpu.bw), T('GPU decode efficiency (share of nameplate)', data.assume.gpu_eta));
    ctx.tiers.filter(t => t.drive && used.has(t.id)).forEach(t => D.push(t.bwTerm ? T('GPU read rate GB/s (' + t.gpu.id + ', ' + t.drive.backend + ' bench line)', t.bwTerm) : T('GPU nameplate bandwidth GB/s (' + t.gpu.id + ')', t.gpu.bw),
      T('cross-backend hop ms per layer (' + t.gpu.id + ')', t.hopTerm)));
    const pl = r.placement || (data.placements && data.placements[r.id]);
    if (pl && pl.tag) D.push(T('per-card expert placement', { v: null, tag: pl.tag, src: pl.src }));
    D.push(T('non-expert GiB', denseSrc(E, m, ctx.kn)));
    [...allTypes].forEach(q => D.push(T('bits per weight ' + q, data.types[q].bpw)));
    D.push(T('routing heat, plan (' + ctx.planName + ')', m.heat.sets[ctx.planName]));
    if (ctx.scoreName !== ctx.planName) D.push(T('routing heat, score (' + ctx.scoreName + ')', m.heat.sets[ctx.scoreName]));
    if (cpuUsed) {
      D.push(T('DRAM read bandwidth GB/s', box.dram.bw), T('CPU box factor f', cal.f || box.cpu.f_default));
      [...cpuTypes].forEach(q => D.push(T('CPU kernel efficiency ' + q, kappaSrc(data, r.box, q))));
      const phys = box.cpu.phys ? v(box.cpu.phys) : null;
      if (ctx.kn.threads && phys && ctx.kn.threads < phys) D.push(T('thread scaling ' + ctx.kn.threads + '/' + phys, { v: ctx.kn.threads / phys, tag: 'ESTIMATE', src: 'ESTIMATE: compute-bound CPU expert kernels scale with physical cores' }));
      if (ctx.kn.smt) D.push(T('SMT factor', box.cpu.smt_mult));
      if (ctx.kn.numaLayout && box.numa_layout && box.numa_layout[ctx.kn.numaLayout]) D.push(T('NUMA page layout ' + ctx.kn.numaLayout + ' (CPU time vs interleave)', box.numa_layout[ctx.kn.numaLayout]));
      if (ctx.kn.kernel2x) D.push(T('2x kernel knob', { v: 2, tag: 'ESTIMATE', src: 'ESTIMATE: the kernel2x knob doubles the CPU kernel rate (what-if)' }));
    }
    if (used.has('pmem')) D.push(T('PMem read rate GB/s', ctx.tiers.find(t => t.id === 'pmem').bwTerm));
    if (used.has('nvme')) D.push(T('NVMe read rate GB/s', box.nvme.bw));
    if (gpuUsed && cpuUsed) D.push(T('GPU/CPU overlap', E.overlap));
    if (E.zero_copy_gbs && used.has('dram')) D.push(T('zero-copy rate GB/s', E.zero_copy_gbs));
    if (ctx.hitR < 1) D.push(T('GPU cache realization', data.cal.engines[r.engine].hit_real));
    const RR = E.recipe_rates || {};
    if (out.moe_cache) D.push(T('MoE cache realization', RR.cache_realize), T('MoE cache upload GB/s', RR.cache_link_gbs));
    if (r.split_mode === 'tensor') D.push(T('-sm tensor reduction ms', RR.allreduce_ms), T('layers', RR.layers_total));
    const nodraft = D.slice();
    if (ctx.draftOn) D.push(T(ctx.cpuW && !out.decode_fit ? 'draft speed-up at no CPU expert time' : 'draft speed-up', out.decode_fit ? out.decode_fit.draft : ctx.kn.draftMult !== undefined && ctx.kn.draftMult !== null ? { v: ctx.kn.draftMult, tag: 'ESTIMATE', src: 'recipe knob draftMult (given, not measured)' } : draftSrc(data, r.box, r.model)));
    if (ctx.draftOn && !out.decode_fit && ctx.cpuW) D.push(T('draft verify weight on CPU expert time', draftSrc(data, r.box, r.model).beta));
    // prefill
    const pf = out.prefill, P = [];
    if (out.prefill_tps === null || out.prefill_tps === undefined) P.push({ term: 'prefill rate', value: null, evidence: 'unknown', tag: 'UNKNOWN', src: out.prefill_why });
    else if (/^GPU offload/.test(out.prefill_why || '')) {
      P.push(T('host-to-device upload GB/s (' + pf.h2d + ')', RR.upload_gbs || (data.prefill && data.prefill.h2d_gbs[pf.h2d])));
      if (pf.upload_touched) P.push(T('upload only the touched experts', RR.upload_touched));
      [...cpuTypes].forEach(q => P.push(T('bits per weight ' + q, data.types[q].bpw)));
      P.push(T('routing heat, score (' + ctx.scoreName + ')', m.heat.sets[ctx.scoreName]));
    } else {
      const c = data.prefill && data.prefill.cpu_tps[pf.key];
      const cell = c ? (c.banks_on && Number(r.banks_gib) > 0 ? c.banks_on : c) : RR.cpu_prefill_tps;
      P.push(T('CPU prefill rate cell ' + pf.key, cell));
      if (r.rtr && c) P.push(T('-rtr prefill multiplier', data.checks.rtr.prefill_mult));
    }
    // VRAM post-warm-up and peak
    const V = ctx.g0.reserves.map(x => T('reserve: ' + x.label, { v: x.gib, tag: x.tag, src: x.src }));
    ctx.fixed.filter(x => x.tier === ctx.g0.id && x.kind !== 'margin').forEach(x => V.push(T(x.label, { v: x.gib, tag: x.tag, src: x.src, evidence: x.evidence })));
    if (pf && pf.offload_on && data.prefill && !RR.upload_gbs) V.push(T('offload upload buffers', data.prefill.vram_gib));
    const VP = V.slice();
    if (mg) VP.push(T('VRAM peak margin', { v: mg.candidate !== null && mg.candidate !== undefined ? mg.candidate : mg.hi, tag: mg.tag, src: mg.src, evidence: mg.evidence }));
    // the external validation gap of this box's hardware class (and placement) lowers confidence; it never sets a term
    const cls = data.external && data.external.box_class.v[r.box], XS = cls ? externalSummary(data) : null;
    const place = cpuUsed && gpuUsed ? 'hybrid' : gpuUsed ? 'gpu' : 'cpu';
    const xg = XS ? (XS.by_class_placement.find(g => g.key === cls + '/' + place) || XS.by_class.find(g => g.key === cls)) : null;
    const withExt = ru => Object.assign(ru, { external: xg ? { group: xg.key, n: xg.n, median_abs_pct: xg.median_abs_pct, bias_pct: xg.bias_pct } : (cls ? { group: cls, n: 0 } : null) });
    const R = [withExt(rollup('decode', out.decode_tps, 'tok/s', D)), withExt(rollup('decode no draft', out.decode_tps_nodraft, 'tok/s', nodraft)),
      rollup('prefill', out.prefill_tps === undefined ? null : out.prefill_tps, 'tok/s', P), rollup('VRAM post-warm-up', out.vram_post_gib, 'GiB', V)];
    if (out.vram_peak_gib !== null && out.vram_peak_gib !== undefined) R.push(rollup('VRAM peak', out.vram_peak_gib, 'GiB', VP));
    if (out.node_memfree_min_gib !== null && out.node_memfree_min_gib !== undefined) R.push(rollup('min node MemFree', out.node_memfree_min_gib, 'GiB', [T('node load extra GiB', out.load_extra)]));
    return R;
  }

  const LAMS = (() => { const a = []; for (let i = 0; i <= 56; i++) a.push(Math.pow(10, -3 + i * 0.125)); return a; })();

  function frontier(ctx) {
    const pts = [];
    if (ctx.bad) return pts;
    let mu = null;
    for (const lam of LAMS) {
      const sol = solveLambda(ctx, lam, mu);
      mu = sol.mu;
      const lay = gateRepair(ctx, expand(ctx, sol));
      const ev = evaluate(ctx, lay);
      pts.push({ lam, tokps: ev.tokps, quality: ev.quality, kld: ev.kld, feasible: sol.feasible, mu: Array.from(sol.mu) });
    }
    return pts;
  }

  // best quality at tokps >= target (score heat), with per-expert fill
  function bestAt(ctx, target, pts) {
    pts = pts || frontier(ctx);
    if (!pts.length) return null;
    let hiI = -1;
    for (let i = 0; i < pts.length; i++) if (pts[i].feasible && pts[i].tokps >= target) { hiI = i; break; }
    if (hiI < 0) return null;
    let lo = hiI > 0 ? pts[hiI - 1].lam : pts[0].lam / 10, hi = pts[hiI].lam;
    let mu = pts[hiI].mu, bestSol = solveLambda(ctx, hi, mu);
    for (let i = 0; i < 14; i++) {
      const mid = Math.sqrt(lo * hi);
      const sol = solveLambda(ctx, mid, mu);
      const ev = evaluate(ctx, gateRepair(ctx, expand(ctx, sol)));
      if (sol.feasible && ev.tokps >= target) { hi = mid; bestSol = sol; } else lo = mid;
    }
    const lay = fill(ctx, gateRepair(ctx, expand(ctx, bestSol)), target);
    const ev = evaluate(ctx, lay);
    return { lam: hi, lay, ev, checks: checksFor(ctx, lay, ev) };
  }

  function layoutAt(ctx, lam, target) {
    const sol = solveLambda(ctx, lam, null);
    const lay = fill(ctx, gateRepair(ctx, expand(ctx, sol)), target || null);
    const ev = evaluate(ctx, lay);
    return { lam, lay, ev, checks: checksFor(ctx, lay, ev) };
  }

  // ---------- anchors (calibration) ----------
  // Returns per-tier bytes/token for an anchor layout (explicit spec), evaluated with the score heat.
  function anchorBytes(data, a) {
    const m = data.models[a.model];
    if (a.explicit) return a.explicit; // {dense_gb, gpu_gb, dram_gb, pmem_gb, nvme_gb, type}
    if (a._bytes) return a._bytes;
    const score = heatSet(data, a.model, a.score || m.heat.anchor_score);
    const sel = heatSet(data, a.model, a.sel || m.heat.anchor_sel);
    const N = m.L * m.E;
    const s = typeBytes(data, a.model, a.type || m.base_type);
    const kScale = a.k ? a.k / m.k : 1;
    const order = Array.from({ length: N }, (_, i) => i).sort((x, y) => sel[y] - sel[x]);
    if (a.split) {
      // split GGUF: a hot set at hot_type, everything else cold_type; GPU = the hottest gpu_n of the hot set; PMem = the coldest cold experts
      const sp = a.split, hot = new Uint8Array(N);
      data[sp.hot_set].layers.forEach((ids, l) => ids.forEach(e => hot[l * m.E + e] = 1));
      const sh = typeBytes(data, a.model, sp.hot_type), sc = typeBytes(data, a.model, sp.cold_type);
      const where = new Uint8Array(N); // 0 dram, 1 gpu, 2 pmem
      let g = 0; for (const e of order) { if (g >= sp.gpu_n) break; if (hot[e]) { where[e] = 1; g++; } }
      let pm = 0; for (let i = N - 1; i >= 0; i--) { const e = order[i]; if (hot[e]) continue; if ((pm + sc) > sp.pmem_cold_gib * GiB) break; where[e] = 2; pm += sc; }
      const acc = {};
      for (let e = 0; e < N; e++) {
        const q = hot[e] ? sp.hot_type : sp.cold_type, t = ['dram', 'gpu', 'pmem'][where[e]];
        const k = t + '|' + q; acc[k] = (acc[k] || 0) + score[e] * (hot[e] ? sh : sc) * kScale / 1e9;
      }
      const parts = Object.keys(acc).map(k => { const [t, q] = k.split('|'); return [t, q, acc[k]]; });
      Object.defineProperty(a, '_bytes', { value: { dense_gb: v(m.dense_gib) * GiB / 1e9, gpu_gb: 0, dram_gb: 0, pmem_gb: 0, nvme_gb: 0, type: sp.hot_type, parts }, enumerable: false, writable: true, configurable: true });
      return a._bytes;
    }
    const onGpu = new Uint8Array(N), onPm = new Uint8Array(N);
    if (a.hot_n) for (let i = 0; i < a.hot_n; i++) onGpu[order[i]] = 1;
    if (a.pmem && a.pmem.mode === 'coldest') {
      const n = Math.floor(a.pmem.gib * GiB / s);
      for (let i = 0; i < n; i++) onPm[order[N - 1 - i]] = 1;
    }
    let gpu = 0, dram = 0, pmem = 0;
    for (let e = 0; e < N; e++) {
      const b = score[e] * s * kScale;
      if (onGpu[e]) gpu += b;
      else if (onPm[e]) pmem += b;
      else if (a.pmem && a.pmem.mode === 'uniform') { pmem += b * a.pmem.share; dram += b * (1 - a.pmem.share); }
      else dram += b;
    }
    Object.defineProperty(a, '_bytes', { value: { dense_gb: v(m.dense_gib) * GiB / 1e9, gpu_gb: gpu / 1e9, dram_gb: dram / 1e9, pmem_gb: pmem / 1e9, nvme_gb: 0, type: a.type || m.base_type }, enumerable: false, writable: true, configurable: true });
    return a._bytes;
  }

  // [tier, type, GB per token] for an anchor: explicit parts (mixed types) plus the single-type fields
  function anchorParts(B) {
    const out = (B.parts || []).slice();
    [['gpu', B.gpu_gb], ['dram', B.dram_gb], ['pmem', B.pmem_gb], ['nvme', B.nvme_gb]].forEach(([t, gb]) => { if (gb) out.push([t, B.type, gb]); });
    return out;
  }
  // sink (optional): gets .cpu, the CPU-side expert seconds of the token (the draft fit reads the anchor's CPU share)
  function anchorTime(data, a, f, host, sink, pmemBw) {
    const box = data.boxes[a.box];
    const g = box.gpus[0];
    const eta = v(data.assume.gpu_eta);
    const B = anchorBytes(data, a);
    const Bd = v(box.dram.bw) * 1e9;
    const gbw = v(g.bw) * 1e9 * eta;
    let T = host / 1000 + B.dense_gb * 1e9 / gbw;
    const pmBw = pmemBw !== undefined && pmemBw !== null ? pmemBw : (box.pmem && (box.pmem.bw_measured || box.pmem.bw));
    anchorParts(B).forEach(([t, q, gb]) => {
      const cpu = Bd * kappaOf(data, a.box, q) * f;
      const r = t === 'gpu' ? gbw : t === 'dram' ? Math.min(Bd, cpu) : t === 'pmem' ? (box.pmem ? Math.min(v(pmBw) * 1e9, cpu) : 1) : v(box.nvme.bw) * 1e9; // applyCal swaps box.pmem.bw to the fitted value
      T += gb * 1e9 / r;
      if (sink && t !== 'gpu') sink.cpu = (sink.cpu || 0) + gb * 1e9 / r;
    });
    return T;
  }

  // Fit per box: CPU efficiency factor f (shared by models), host ms per model, and (boxes with PMem) the PMem rate.
  // Least squares on log(tok/s) over the box's no-draft speed anchors.
  function golden(fn, a, b, it) {
    const g = 0.6180339887; let c = b - g * (b - a), d = a + g * (b - a), fc = fn(c), fd = fn(d);
    for (let i = 0; i < (it || 40); i++) {
      if (fc < fd) { b = d; d = c; fd = fc; c = b - g * (b - a); fc = fn(c); } else { a = c; c = d; fc = fd; d = a + g * (b - a); fd = fn(d); }
    }
    return fc < fd ? [c, fc] : [d, fd];
  }
  const calibrationCache = new WeakMap();
  function calibrate(data) {
    // A content key catches in-place knob/catalog edits; never cache merely by
    // object identity. Return a fresh result because applyCal enriches it.
    const key = JSON.stringify(['anchors', 'boxes', 'models', 'types', 'engines', 'assume', 'overrides', 'knobs', 'draft_twins'].map(k => data[k]));
    const cached = calibrationCache.get(data);
    if (cached && cached.key === key) return JSON.parse(cached.result);
    const result = calibrateUncached(data);
    calibrationCache.set(data, {key, result: JSON.stringify(result)});
    return result;
  }
  function calibrateUncached(data) {
    const cal = { boxes: {}, types: {}, rows: [] };
    const anchors = data.anchors;
    Object.keys(data.boxes).forEach(bk => {
      const box = data.boxes[bk];
      const A = anchors.filter(a => a.box === bk && a.fit === 'speed' && !a.draft);
      const hg = mk => data.models[mk].host_group || mk;
      const models = [...new Set(A.map(a => hg(a.model)))];
      const res = { f: { v: v(box.cpu.f_default), tag: box.cpu.f_default.tag, src: box.cpu.f_default.src }, host: {}, draft: {} };
      if (A.length) {
        const fixF = box.cpu.f_fixed;
        const fitPm = box.pmem && box.pmem.fit_range && A.some(a => a.pmem);
        const pmOrig = box.pmem ? (box.pmem.bw_measured || box.pmem.bw) : null;
        let best = null;
        const fGrid = fixF ? [v(box.cpu.f_default)] : (() => { const a = []; for (let x = 0.05; x <= 1.6; x += 0.005) a.push(x); return a; })();
        const pGrid = fitPm ? (() => { const a = []; for (let x = box.pmem.fit_range[0]; x <= box.pmem.fit_range[1] + 1e-9; x += 0.2) a.push(x); return a; })() : [null];
        for (const pb of pGrid) {
          const pmVal = pb !== null ? { v: pb } : pmOrig;
          for (const f of fGrid) {
            let err = 0; const hosts = {};
            for (const mk of models) {
              const Am = A.filter(a => hg(a.model) === mk);
              const hFixed = box.host_fixed && box.host_fixed[mk] !== undefined ? v(box.host_fixed[mk]) : null;
              const e2 = h => { let s2 = 0; for (const a of Am) { const d = Math.log(1 / anchorTime(data, a, f, h, null, pmVal) / a.meas); s2 += d * d * (a.w || 1); } return s2; };
              const [bh, be] = hFixed !== null ? [hFixed, e2(hFixed)] : golden(e2, 0, 120, 36);
              hosts[mk] = bh; err += be;
            }
            if (!best || err < best.err) best = { f, hosts, err, pb };
          }
        }
        const nPar = models.length + (fixF ? 0 : 1) + (fitPm ? 1 : 0);
        res.f = { v: best.f, tag: fixF ? box.cpu.f_default.tag : 'FIT', src: fixF ? box.cpu.f_default.src : ('fit to ' + A.length + ' measured no-draft points on ' + bk + ' (' + models.join(', ') + '), ' + nPar + ' parameters, rms log error ' + Math.sqrt(best.err / A.length).toFixed(3)) };
        Object.keys(data.models).forEach(mk => {
          const g = hg(mk);
          if (best.hosts[g] === undefined) return;
          res.host[mk] = { v: best.hosts[g], tag: (box.host_fixed && box.host_fixed[g] !== undefined) ? box.host_fixed[g].tag : 'FIT', src: 'host overhead ms/token beyond the GPU dense read, fitted on ' + bk + ' (shared by ' + Object.keys(data.models).filter(x => hg(x) === g).join(', ') + ')' };
        });
        if (fitPm) res.pmem_bw = { v: best.pb, tag: 'FIT', src: 'PMem expert-read rate fitted on ' + bk + ' within the measured ' + box.pmem.fit_range.join('-') + ' GB/s band (lab notes: ~10 both sockets, 13.4 socket-local)' };
        res.rms = Math.sqrt(best.err / A.length);
      }
      cal.boxes[bk] = res;
    });
    // hosts for models with no anchor on a box: scale droopy's host by the box's V4.0 ratio, else borrow
    Object.keys(data.boxes).forEach(bk => {
      const res = cal.boxes[bk];
      ['v41', 'glm', 'v40'].forEach(mk => {
        if (res.host[mk] !== undefined) return;
        const ref = cal.boxes[data.assume.host_ref_box] && cal.boxes[data.assume.host_ref_box].host[mk];
        const fb = data.boxes[bk].host_default && data.boxes[bk].host_default[mk];
        if (ref) {
          const r40 = res.host.v40 && cal.boxes[data.assume.host_ref_box].host.v40 ? v(res.host.v40) / v(cal.boxes[data.assume.host_ref_box].host.v40) : 1;
          res.host[mk] = { v: v(ref) * r40, tag: 'ESTIMATE', src: 'no ' + mk + ' anchor on ' + bk + ': ' + data.assume.host_ref_box + ' fitted host x ' + (res.host.v40 ? 'this box/' + data.assume.host_ref_box + ' V4.0 host ratio ' + r40.toFixed(2) : '1') };
        } else if (fb) res.host[mk] = fb;
      });
    });
    // models with no run anywhere borrow another model's host on each box (ESTIMATE). A list borrow (the generic MoE
    // path) is resolved after the run-ledger host fit instead (borrowHostLists, applyCal)
    Object.keys(data.boxes).forEach(bk => {
      const res = cal.boxes[bk];
      Object.keys(data.models).forEach(mk => {
        const b = data.models[mk].host_borrow;
        if (!b || Array.isArray(b) || res.host[mk] !== undefined) return;
        if (res.host[b] === undefined) return;
        res.host[mk] = { v: v(res.host[b]), tag: 'ESTIMATE', src: 'no ' + data.models[mk].short + ' run on any box: borrowed this box\'s ' + b + ' host overhead (' + res.host[b].tag + ')' };
      });
    });
    // types whose CPU efficiency is derived from a measured lane rate on one box (glm-lite EXL3)
    Object.keys(data.types).forEach(q => {
      const t = data.types[q];
      if (!t.kappa_from) return;
      const kf = t.kappa_from, box = data.boxes[kf.box];
      const base = kf.rate_gbs / (v(box.dram.bw) * v(cal.boxes[kf.box].f));
      cal.types[q] = { v: base * (kf.mult || 1), tag: kf.tag || 'FIT', src: kf.src };
    });
    // residual rows (all anchors incl. draft ones and references)
    anchors.forEach(a => {
      const res = cal.boxes[a.box];
      if (a.engine && data.engines[a.engine] && data.engines[a.engine].host_ms) {
        // engine with its own measured decomposition (glm-lite); history rows carry their own measured rates (eng_over)
        const E = data.engines[a.engine], box = data.boxes[a.box], B = a.explicit, eo = a.eng_over || {};
        const Bd = v(box.dram.bw) * 1e9, gbw = v(box.gpus[0].bw) * 1e9 * v(data.assume.gpu_eta);
        const zc = (eo.zero_copy_gbs !== undefined ? eo.zero_copy_gbs : v(E.zero_copy_gbs || 0)) * 1e9;
        let gpuT = B.dense_gb * 1e9 / gbw, cpuT = 0;
        anchorParts(B).forEach(([t, q, gb]) => {
          if (t === 'gpu') gpuT += gb * 1e9 / gbw;
          else if (t === 'dram') {
            const kap = cal.types[q] !== undefined ? v(cal.types[q]) : kappaOf(data, a.box, q);
            cpuT += gb * 1e9 / ((eo.lane_gbs ? eo.lane_gbs * 1e9 * (data.types[q].kappa_from && data.types[q].kappa_from.mult || 1) : Math.min(Bd, Bd * kap * v(res.f))) + zc);
          }
          else cpuT += gb * 1e9 / (v(box.nvme.bw) * 1e9);
        });
        const p = 1 / ((eo.host_ms !== undefined ? eo.host_ms : v(E.host_ms)) / 1000 + Math.max(gpuT, cpuT));
        cal.rows.push(Object.assign({}, a, { pred: p, pred_nodraft: p, resid: p / a.meas - 1, cpu_s: cpuT }));
        return;
      }
      const f = v(res.f), h = res.host[a.model] ? v(res.host[a.model]) : 0;
      const pmVal = res.pmem_bw || (data.boxes[a.box].pmem && (data.boxes[a.box].pmem.bw_measured || data.boxes[a.box].pmem.bw));
      const sink = { cpu: 0 }, base = 1 / anchorTime(data, a, f, h, sink, pmVal);
      cal.rows.push(Object.assign({}, a, { pred_nodraft: base, pred: base, resid: base / a.meas - 1, cpu_s: sink.cpu }));
    });
    // split-model overhead: a split GGUF builds a chain per partition per layer; the measured split run minus the fitted model = extra ms/token
    anchors.filter(a => a.fit === 'split').forEach(a => {
      const r = cal.rows.find(x => x.id === a.id);
      const extra = 1000 / a.meas - 1000 / r.pred_nodraft;
      cal.boxes[a.box].split_ms = { v: extra, tag: 'FIT', src: 'extra host ms/token of the split GGUF: ' + a.id + ' measured ' + a.meas + ' tok/s vs ' + r.pred_nodraft.toFixed(2) + ' modelled without it. ONE point, confounded with T40 (B0 was the first T40 run); B1b arm 2 (T20) separates threads' };
    });
    Object.keys(data.boxes).forEach(bk => {
      const ref = cal.boxes[data.assume.host_ref_box].split_ms;
      if (cal.boxes[bk].split_ms || !ref) return;
      cal.boxes[bk].split_ms = { v: v(ref), tag: 'ESTIMATE', src: 'borrowed from ' + data.assume.host_ref_box + ' (no split run on ' + bk + ')' };
    });
    cal.rows.forEach(r => {
      const a = anchors.find(x => x.id === r.id);
      if (!a || !a.split || !cal.boxes[r.box].split_ms) return;
      r.pred_nodraft = 1 / (1 / r.pred_nodraft + v(cal.boxes[r.box].split_ms) / 1000); r.pred = r.pred_nodraft; r.resid = r.pred / r.meas - 1;
    });
    // draft multipliers: measured / predicted-without-draft, averaged per box x model
    const dm = {};
    cal.rows.filter(r => r.draft && r.fit === 'draft').forEach(r => {
      const key = r.box + '|' + r.model;
      const tw = r.twin ? anchors.find(a => a.id === r.twin) : null;
      r.mult = tw ? r.meas / tw.meas : r.meas / r.pred_nodraft;
      r.mult_src = tw ? ('measured / measured twin ' + tw.id) : 'measured / model without draft';
      (dm[key] = dm[key] || []).push(r.mult);
    });
    // measured twins (same load, draft off vs on) win over ratios to the model
    const tw = {};
    cal.rows.filter(r => r.draft && r.fit === 'draft' && r.twin).forEach(r => (tw[r.box + '|' + r.model] = tw[r.box + '|' + r.model] || []).push(r.mult));
    Object.keys(tw).forEach(k => { dm[k] = tw[k]; });
    // in-load twins from the run ledger (the served draft width vs no draft, same load and prompts) win over anchors when there are 2+
    const LT = data.draft_twins || {}, ltSrc = {};
    // one twin is enough where no anchor measured the draft at all (it replaces the model's mult_default ESTIMATE)
    Object.keys(LT).filter(k => (LT[k].length >= 2 || !dm[k]) && cal.boxes[k.split('|')[0]]).forEach(k => {
      dm[k] = LT[k].map(x => x.mult);
      ltSrc[k] = 'draft speed-up: mean of ' + LT[k].length + ' in-load twins in the run ledger (' + LT[k][0].what + ' vs no draft, same load and prompts; a twin whose phase a degraded note names is left out): ' + LT[k].map(x => x.id + ' x' + x.mult.toFixed(3)).join(', ');
    });
    Object.keys(dm).forEach(key => {
      const [bk, mk] = key.split('|');
      const xs = dm[key];
      cal.boxes[bk].draft[mk] = { v: xs.reduce((s, x) => s + x, 0) / xs.length, tag: 'FIT', runs: ltSrc[key] ? LT[key].map(x => x.id) : undefined, src: ltSrc[key] || 'draft speed-up from ' + xs.length + ' measured arm(s) on ' + bk + ' (vs a measured no-draft twin where one exists, else vs the model)' };
    });
    cal.rows.forEach(r => {
      if (r.draft && r.pred_nodraft) {
        const m = cal.boxes[r.box].draft[r.model];
        if (m) { r.pred = r.pred_nodraft * v(m); r.resid = r.pred / r.meas - 1; }
      }
    });
    // dynamic GPU cache: measured GPU hit share vs the hit the planner's static hot set would get with the same slots
    cal.engines = {};
    Object.keys(data.engines).forEach(eng => {
      const ha = data.engines[eng].hit_anchor;
      if (!ha) return;
      const m = data.models[ha.model];
      const plan = heatSet(data, ha.model, m.heat.plan_default), score = heatSet(data, ha.model, m.heat.score_default);
      const order = Array.from(plan, (x, i) => i).sort((a, b) => plan[b] - plan[a]);
      let hit = 0; order.slice(0, ha.slots).forEach(e => hit += score[e]);
      const oracle = hit / (m.k * m.L), meas = ha.gpu_hits / ha.reads;
      cal.engines[eng] = { hit_real: { v: meas / oracle, tag: 'FIT', src: 'measured GPU hit ' + (100 * meas).toFixed(1) + ' % with ' + ha.slots + ' slots vs ' + (100 * oracle).toFixed(1) + ' % for the planner static hot set of the same size on the ' + m.heat.score_default + ' heat; ' + ha.src } };
    });
    return cal;
  }

  // a ledger fit whose runs ran under a dated box condition (data.boxes[box].conditions; each data.recipes row carries the ids in
  // box_conditions) names it: box_conditions [{box, id, short, from, to, runs}] and one clause per condition in its src
  function withConds(data, fit) {
    const by = {};
    fit.runs.forEach(id => {
      const row = (data.recipes || []).find(x => x.run_id === id);
      ((row && row.box_conditions) || []).forEach(c => (by[row.box + '|' + c] = by[row.box + '|' + c] || []).push(id));
    });
    const cs = Object.keys(by).map(k => {
      const [box, id] = k.split('|'), c = data.boxes[box].conditions.find(x => x.id === id);
      return { box, id, short: c.short, from: c.from, to: c.to, runs: by[k] };
    });
    return cs.length ? Object.assign(fit, { box_conditions: cs, src: fit.src + cs.map(c => '; ' + c.runs.join(', ') + ' ran under ' + c.box + ' ' + c.short + ' (' + c.from + ' to ' + c.to + ')').join('') }) : fit;
  }

  // Host ms/token from the run ledger where no anchor fits it: a box x model whose host is only borrowed or estimated takes the
  // mean residual of its clean ledger runs (measured no-draft ms/token minus the modelled GPU + CPU expert time of the run's own
  // recipe). A run counts only when it went fine, carries no confounded caveat, is not a decode-fit holdout, carries no gate waiver (a waiver records a part
  // of the recipe the planner does not describe, e.g. experts on a GPU it does not model) and is PREDICTED today; a fitted,
  // fixed or decode-profile host never moves. Changes nothing for a box x model without such a run.
  function ledgerHosts(data) {
    const cal = data.cal, by = {};
    (data.recipes || []).forEach(row => {
      const m = row.measured, r = row.recipe || {}, res = cal.boxes[row.box];
      if (!m || m.bad || (m.caveat || []).length || m.holdout || !(m.off > 0) || r.gate_waiver || ['dllm', 'network'].includes(r.kind) || r.decode_profile !== undefined || !res || !data.models[row.model]) return;
      const h = res.host[row.model], fx = data.boxes[row.box].host_fixed;
      if ((h && h.tag !== 'ESTIMATE') || (fx && fx[data.models[row.model].host_group || row.model] !== undefined)) return;
      const p = predictRecipe(data, r);
      if (p.status !== 'PREDICTED' || !p.ms_per_token || p.cross_layers) return;
      (by[row.box + '|' + row.model] = by[row.box + '|' + row.model] || []).push({ id: row.run_id, host: 1000 / m.off - (p.ms_per_token.total - p.ms_per_token.host), off: m.off, phase: m.off_phase });
    });
    Object.keys(by).forEach(k => {
      const [bk, mk] = k.split('|'), xs = by[k], hv = xs.reduce((a, x) => a + x.host, 0) / xs.length;
      if (!(hv >= 0)) return;
      cal.boxes[bk].host[mk] = withConds(data, { v: hv, tag: 'FIT', runs: xs.map(x => x.id), src: 'host overhead ms/token fitted on ' + xs.length + ' clean run(s) in the run ledger on ' + bk + ' (no anchor): measured no-draft decode minus the modelled GPU and CPU expert time of each run\'s own recipe: ' +
        xs.map(x => x.id + ' ' + x.phase + ' ' + x.off.toFixed(2) + ' tok/s -> ' + x.host.toFixed(1) + ' ms').join('; ') });
    });
  }

  // the hop a token pays per layer it crosses to a card reached through a second backend (cal.boxes[box].hop[gpu id], ms),
  // fitted after the host: on clean ledger runs that read experts from exactly one such card, measured no-draft decode minus
  // every modelled term (the host included), over the layers crossed. A gate waiver does not keep a run out: the waivers on
  // these runs name the card the planner did not model before this term
  function ledgerHops(data) {
    const by = {};
    (data.recipes || []).forEach(row => {
      const m = row.measured, r = row.recipe || {}, res = data.cal.boxes[row.box];
      if (!m || m.bad || (m.caveat || []).length || m.holdout || !(m.off > 0) || ['dllm', 'network'].includes(r.kind) || r.decode_profile !== undefined || !res || !data.models[row.model]) return;
      const p = predictRecipe(data, r), ids = Object.keys((p && p.cross_layers) || {});
      if (p.status !== 'PREDICTED' || ids.length !== 1) return;
      const k = row.box + '|' + ids[0], x = p.cross_layers[ids[0]];
      (by[k] = by[k] || []).push({ id: row.run_id, hop: (1000 / m.off - p.ms_per_token.total) / x, layers: x, off: m.off, phase: m.off_phase });
    });
    Object.keys(by).forEach(k => {
      const [bk, gid] = k.split('|'), xs = by[k], hv = xs.reduce((a, x) => a + x.hop, 0) / xs.length;
      if (!(hv >= 0)) return;
      const res = data.cal.boxes[bk];
      res.hop = Object.assign({}, res.hop, { [gid]: withConds(data, { v: hv, tag: 'FIT', runs: xs.map(x => x.id), src: 'cross-backend hop ms per layer crossed to ' + gid + ', fitted on ' + xs.length + ' clean run(s) in the run ledger on ' + bk + ': measured no-draft decode minus every modelled term (host, dense, bank reads at the card line), over the layers crossed: ' +
        xs.map(x => x.id + ' ' + x.phase + ' ' + x.off.toFixed(2) + ' tok/s, ' + x.layers.toFixed(1) + ' layers -> ' + x.hop.toFixed(3) + ' ms').join('; ') }) });
    });
  }

  // the draft speed-up by where the experts sit: a draft step verifies several tokens in one forward, and the CPU expert
  // time of that forward grows with the distinct experts the tokens route to (droopy-glm-glm-be1q4y-0909, conditions.draft_n)
  // more than a GPU's read does. So with the draft on, decode = tau / (T + beta x CPU expert time): tau per box x model (the
  // speed-up with every expert on a GPU) and beta per draft kind (the twins' what: an MTP head verifies inside the target's
  // forward, a separate DSpark draft model puts its own verify work beside it), both fitted (least squares on the log
  // speed-up) on the in-load ledger twins, each at the CPU share of the token that its own run's recipe gets from the planner
  // today (after the host, hop and decode-profile fits). A kind whose twins do not identify beta (no box x model of the kind
  // has twins at different CPU shares, or the fit runs to the top of the search) takes the beta pooled over every kind.
  // Measured 10-21 (27 twins, error of the predicted speed-up with each twin left out of the fit): pooled beta 0.300 gave rms
  // 0.069 held out (0.055 in sample); per kind (DSpark 0.076 on 19 droopy V4.1 twins, MTP 0.493 on GLM + Ornith) rms 0.068
  // held out (0.053 in sample); per box x model with fallback rms 0.066-0.067, but only through one GLM twin whose held-out
  // GLM-only fit sits at 2.8 next to the 4.0 search top (GLM alone is not identified). A box x model whose twins are not all
  // PREDICTED keeps the plain mean; anchor-fitted ones keep theirs.
  function ledgerDrafts(data) {
    const cal = data.cal, LT = data.draft_twins || {}, by = {};
    Object.keys(LT).forEach(k => {
      const [bk, mk] = k.split('|'), d = cal.boxes[bk] && cal.boxes[bk].draft[mk];
      if (!d || !d.runs) return;
      const xs = LT[k].map(x => {
        const row = (data.recipes || []).find(y => y.run_id === x.id), p = row && predictRecipe(data, row.recipe);
        return p && p.status === 'PREDICTED' && p.ms_per_token ? { id: x.id, mult: x.mult, s: p.ms_per_token.cpu / p.ms_per_token.total } : null;
      });
      if (xs.every(x => x)) by[k] = xs;
    });
    const keys = Object.keys(by), kind = k => LT[k][0].what, top = v(data.assume.draft_cpu_beta_max);
    // beta is seen only through twins of one box x model at different CPU shares
    const spread = ks => ks.some(k => by[k].some(x => x.s !== by[k][0].s));
    if (!spread(keys)) return;
    const lt = (b, x) => Math.log(x.mult * (1 + b * x.s)), tau = (b, xs) => xs.reduce((a, x) => a + lt(b, x), 0) / xs.length;
    const fit = (ks, what) => {
      const err = b => ks.reduce((a, k) => { const t = tau(b, by[k]); return a + by[k].reduce((e, x) => e + (lt(b, x) - t) ** 2, 0); }, 0);
      const [beta, e2] = golden(err, 0, top), all = [].concat(...ks.map(k => by[k]));
      return { beta, ceiling: beta > top * (1 - 1e-6), fit: withConds(data, { v: beta, tag: 'FIT', kind: what, runs: all.map(x => x.id), src: 'draft verify weight on CPU expert time' + (what ? ' (' + what + ')' : ', pooled over every draft kind (taken by a kind whose own twins do not identify it)') + ': decode with the draft = tau / (T + beta x CPU expert time), beta fitted with a tau per box x model on ' + all.length + ' in-load ledger twins (' + ks.join(', ') + '; CPU share of each run\'s recipe as predicted today), rms log error ' + Math.sqrt(e2 / all.length).toFixed(3) }) };
    };
    const P = fit(keys).fit, kinds = {};
    [...new Set(keys.map(kind))].forEach(w => {
      const ks = keys.filter(k => kind(k) === w), f = spread(ks) && fit(ks, w);
      if (f && !f.ceiling) kinds[w] = f.fit;
    });
    cal.draft_cpu = Object.assign({ pooled: P }, kinds);
    keys.forEach(k => {
      const [bk, mk] = k.split('|'), d = cal.boxes[bk].draft[mk], B = kinds[kind(k)] || P, beta = B.v, t = Math.exp(tau(beta, by[k]));
      cal.boxes[bk].draft[mk] = withConds(data, Object.assign({}, d, { v: t, beta: B, src: 'draft speed-up with no CPU expert time (x' + t.toFixed(3) + ' / (1 + ' + beta.toFixed(3) + ' x CPU share), the ' + (kinds[kind(k)] ? kind(k) : 'pooled') + ' weight), fitted on ' + by[k].length + ' in-load twins in the run ledger (' + kind(k) + ' vs no draft, same load and prompts): ' +
        by[k].map(x => x.id + ' x' + x.mult.toFixed(3) + ' at CPU share ' + x.s.toFixed(3) + ' (fit x' + (t / (1 + beta * x.s)).toFixed(3) + ')').join(', ') }));
    });
    cal.rows.forEach(r => {
      const d = r.draft && r.pred_nodraft && cal.boxes[r.box].draft[r.model];
      if (d && d.beta) { r.pred = r.pred_nodraft * v(d) / (1 + v(d.beta) * (r.cpu_s || 0) * r.pred_nodraft); r.resid = r.pred / r.meas - 1; }
    });
  }

  // A list borrow (the generic MoE path, tools/harvest.py: a model with no run anywhere) takes on each box the median of
  // the listed models' hosts that were fitted there, by an anchor or on the run ledger (not ESTIMATE), else the median
  // of every listed host the box has. It runs after ledgerHosts: most families' hosts on most boxes are ledger fits
  // (lassie Ornith-35B 6.9 ms against a 15 ms estimate before them)
  function borrowHostLists(data) {
    Object.keys(data.boxes).forEach(bk => {
      const res = data.cal.boxes[bk];
      Object.keys(data.models).forEach(mk => {
        const b = data.models[mk].host_borrow;
        if (!Array.isArray(b) || res.host[mk] !== undefined) return;
        const have = b.filter(k => res.host[k] !== undefined && v(res.host[k]) !== null);
        const fit = have.filter(k => res.host[k].tag !== 'ESTIMATE'), use = fit.length ? fit : have;
        if (!use.length) return;
        const xs = use.map(k => v(res.host[k])).sort((x, y) => x - y), h = xs.length;
        res.host[mk] = { v: h % 2 ? xs[(h - 1) / 2] : (xs[h / 2 - 1] + xs[h / 2]) / 2, tag: 'ESTIMATE', src: 'no ' + data.models[mk].short + ' run on any box: the median of this box\'s ' +
          (fit.length ? 'fitted' : 'estimated') + ' host overheads of ' + use.map(k => k + ' ' + v(res.host[k]).toFixed(1) + ' ms').join(', ') };
      });
    });
  }

  function applyCal(data) {
    data.cal = calibrate(data);
    Object.keys(data.cal.boxes).forEach(bk => {
      const box = data.boxes[bk], pb = data.cal.boxes[bk].pmem_bw;
      if (box.pmem && pb) {
        const meas = box.pmem.bw_measured || box.pmem.bw;
        box.pmem.bw_measured = meas;
        box.pmem.bw = Object.assign({}, pb, { src: pb.src + '; measured: ' + meas.src });
      }
    });
    Object.keys(data.cal.types || {}).forEach(q => {
      data.types[q].kappa = data.cal.types[q];
    });
    ledgerHosts(data);
    borrowHostLists(data);
    ledgerHops(data);
    data.cal.recipe_decode = fitRecipeDecode(data);
    ledgerDrafts(data);
    return data.cal;
  }

  // ---------- owner's droopy layout (V4.1) ----------
  function ownerLayout(ctx, upgradeLeftover) {
    const { data, mk, m, tiers, types } = ctx;
    const N = m.L * m.E;
    const qi = q => types.indexOf(q);
    const sz = q => typeBytes(data, mk, q);
    const tier = new Int8Array(N), type = new Int8Array(N);
    const tId = id => tiers.findIndex(t => t.id === id);
    const coding = new Uint8Array(N);
    data.alloc3.layers.forEach((ids, l) => ids.forEach(e => coding[l * m.E + e] = 1));
    const order = ctx.order;
    const left = tiers.map(t => t.cap);
    const g0 = tiers.indexOf(ctx.g0), dr = tId('dram'), pm = tId('pmem') >= 0 ? tId('pmem') : tId('nvme');
    const copy = ctx.copyTier ? tiers.indexOf(ctx.copyTier) : -1;
    const q4 = qi('mxfp4') >= 0 ? 'mxfp4' : 'q4_K', q3 = 'q3_K';
    order.forEach(e => {
      if (coding[e] && left[g0] >= sz(q4) && (copy < 0 || left[copy] >= sz(q4))) { tier[e] = g0; type[e] = qi(q4); left[g0] -= sz(q4); if (copy >= 0) left[copy] -= sz(q4); return; }
      if (coding[e] && left[dr] >= sz(q3)) { tier[e] = dr; type[e] = qi(q3); left[dr] -= sz(q3); return; }
      tier[e] = pm; type[e] = qi(q4); left[pm] -= sz(q4);
    });
    // optionally spend leftover DRAM on 4-bit for the hottest DRAM experts
    if (upgradeLeftover) for (const e of order) {
      if (tier[e] === dr && types[type[e]] === q3) {
        const d = sz(q4) - sz(q3);
        if (left[dr] >= d) { type[e] = qi(q4); left[dr] -= d; }
      }
    }
    return { tier, type };
  }

  // Fixed-workload routing evidence, not the selected plan/score heat or a speed/quality term.
  function routingSkew(data, mk) {
    return (data.routing_skew || []).find(s => s.model === mk) || null;
  }

  // ---------- export ----------
  function exportMap(ctx, lay, ev, extra) {
    const { m, tiers, types } = ctx;
    const layers = [];
    for (let l = 0; l < m.L; l++) {
      const row = { layer: m.layer_ids ? m.layer_ids[l] : l, tier: [], type: [] };
      for (let e = 0; e < m.E; e++) { row.tier.push(tiers[lay.tier[l * m.E + e]].id); row.type.push(types[lay.type[l * m.E + e]]); }
      layers.push(row);
    }
    return Object.assign({
      format: 'fit-viz expert map v1',
      model: ctx.mk, model_name: m.name, box: ctx.bk, engine: ctx.eng,
      tensor_names: m.tensor_names, experts_per_layer: m.E,
      note: 'tier/type per routed expert, index = expert id within the layer. gate/up/down of one expert share the type unless the type is a split recipe (q2kq3k = gate+up q2_K, down q3_K).',
      tiers: tiers.map(t => ({ id: t.id, name: t.name, cap_gib: +(t.cap / GiB).toFixed(2) })),
      projected: { tok_s: +ev.tokps.toFixed(2), quality_pct: +ev.quality.toFixed(2), kld: +ev.kld.toFixed(5), plan_heat: ctx.planName, score_heat: ctx.scoreName },
      heat_maps: { plan: heatCite(m, ctx.planName), score: heatCite(m, ctx.scoreName) },
      routing_skew: routingSkew(ctx.data, ctx.mk),
      knobs: ctx.kn, layers
    }, extra || {});
  }

  // ---------- agents vs speed: one curve per config (data.curves, build.py) ----------
  // With G = min(n, groups) groups in use, S pipeline stages and a stage window t(r) = a + b x r ms for r = n / G agents a
  // group, one cycle on the slowest stage takes max(G, S) windows and yields n tokens: total = eff x 1000 x n / (max(G, S) x
  // t(n / G)). From the measured (n, total) pairs alone: n / total = A x max(G, S) + B x max(G, S) x n / G, A = a / (1000 eff),
  // B = b / (1000 eff), fitted by least squares (two constants, no intercept). Uneven groups give the same sum (the window is
  // linear in r), so n need not divide by G.
  const curveK = (c, n) => { const G = Math.min(n, v(c.groups)); return { G, k: Math.max(G, v(c.stages)) }; };
  function curveFit(data, c) {
    const S = data.curves.settings, pts = c.points.filter(p => p.used), ns = [...new Set(pts.map(p => p.n))];
    if (ns.length < v(S.min_points)) return { ok: false, why: ns.length + ' distinct agent count' + (ns.length === 1 ? '' : 's') + ' measured (' + v(S.min_points) + ' needed for the two constants)', pts };
    const m = [0, 0, 0, 0, 0];
    pts.forEach(p => { const { G, k } = curveK(c, p.n), x1 = k, x2 = k * p.n / G, y = p.n / v(p.tps); m[0] += x1 * x1; m[1] += x1 * x2; m[2] += x2 * x2; m[3] += x1 * y; m[4] += x2 * y; });
    const det = m[0] * m[2] - m[1] * m[1], A = (m[3] * m[2] - m[4] * m[1]) / det, B = (m[4] * m[0] - m[3] * m[1]) / det;
    if (!(A > 0 && B > 0)) return { ok: false, why: 'the points do not follow the window law (fitted A ' + A.toPrecision(3) + ', B ' + B.toPrecision(3) + '; both must be > 0: total falls as agents rise)', pts };
    const at = n => { const { G, k } = curveK(c, n); return n / (A * k + B * k * n / G); };
    const errs = pts.map(p => ({ n: p.n, run: p.run, meas: v(p.tps), pred: at(p.n), err: at(p.n) / v(p.tps) - 1 }));
    // the stage window itself where the run printed it: eff per point = measured total / (1000 n / (max(G, S) x t)), and the
    // line t = a + b x r through the windows when two or more agent-per-group counts carry one
    const wp = pts.filter(p => p.win_ms).map(p => { const { G, k } = curveK(c, p.n); return { n: p.n, r: p.n / G, t: v(p.win_ms), eff: v(p.tps) / (1000 * p.n / (k * v(p.win_ms))) }; });
    const rs = [...new Set(wp.map(x => x.r))];
    const line = rs.length >= 2 ? lsq(wp.map(x => [1, x.r]), wp.map(x => x.t)) : null;
    const win = wp.length ? { pts: wp, eff: wp.reduce((s, x) => s + x.eff, 0) / wp.length, a: line ? line[0] : null, b: line ? line[1] : null } : null;
    return { ok: true, A, B, at, errs, err_max: Math.max(...errs.map(e => Math.abs(e.err))), win, pts, n_max: Math.max(...ns) };
  }
  // the host-pin check for n slots: the config's serving fields (READ from its flags) with --batch n, one GPU bank a stage;
  // a config with --batch N or one server per agent count (slots_each_point) has a serving shape
  function curveHost(data, c, n) {
    if (!c.slots && !c.slots_each_point) return { status: 'UNKNOWN', detail: 'the run flags carry no --batch: no serving shape to check' };
    const r = { id: c.id + '@' + n, box: c.box, model: c.model, engine: c.engine, ctx: c.ctx, draft: c.draft, serving: Object.assign({}, c.serving, { batch: n }),
      gpu_split: v(c.stages) > 1 ? { banks_gib: Array(v(c.stages)).fill(0) } : undefined };
    return hostPinnedCheck(data, r) || { status: 'UNKNOWN', detail: 'engine ' + c.engine + ' has no serving model' };
  }
  // watts at n: a measured point there (mean), else straight-line between the nearest measured counts around it (ESTIMATE)
  function curveWatts(c, n) {
    const w = c.points.filter(p => p.watts), at = w.filter(p => p.n === n);
    if (at.length) return { w: at.reduce((s, p) => s + v(p.watts), 0) / at.length, tag: 'MEASURED' };
    const lo = w.filter(p => p.n < n).sort((x, y) => y.n - x.n)[0], hi = w.filter(p => p.n > n).sort((x, y) => x.n - y.n)[0];
    return lo && hi ? { w: v(lo.watts) + (v(hi.watts) - v(lo.watts)) * (n - lo.n) / (hi.n - lo.n), tag: 'ESTIMATE' } : null;
  }
  // the curve from 1 to agents_max with the readouts: the largest n with each agent >= min_each_tps (headline) and >=
  // second_each_tps, the total there, the asymptotic total, the host-memory ceiling (largest slot count the host-pin check
  // passes) and tok/s per watt where watts are logged. opts: { min, second, max } override the settings
  function curveReport(data, c, opts) {
    const S = data.curves.settings, o = opts || {}, f = curveFit(data, c);
    const max = o.max || v(S.agents_max), min = o.min !== undefined ? o.min : v(S.min_each_tps), second = o.second !== undefined ? o.second : v(S.second_each_tps);
    const power = c.points.filter(p => p.watts).map(p => ({ n: p.n, run: p.run, watts: v(p.watts), tps: v(p.tps), tps_per_w: v(p.tps) / v(p.watts) }));
    // the largest slot count whose host-pinned set fits (null when the check cannot judge); a layer split that keeps the
    // stage-lend duplicate also shows the ceiling with serving.stage_lend_drop
    const ceiling = cc => { let ceil = 0, over = null; for (let n = 1; n <= max; n++) { const h = curveHost(data, cc, n); if (h.status !== 'PASS') { over = Object.assign({ n }, h); break; } ceil = n; } return { ceiling: over && over.status === 'UNKNOWN' ? null : ceil, all_fit: !over, first_over: over }; };
    const host = ceiling(c), lend = v(c.stages) > 1 && c.slots && !c.serving.stage_lend_drop ? ceiling(Object.assign({}, c, { serving: Object.assign({}, c.serving, { stage_lend_drop: true }) })) : null;
    if (lend && lend.ceiling !== null && lend.ceiling !== host.ceiling) host.with_lend_drop = lend.ceiling;
    if (!f.ok) return { cfg: c, contention: c.contention || [], ok: false, why: f.why, fit: f, host, power, min, second, max };
    const rows = Array.from({ length: max }, (_, i) => { const n = i + 1, t = f.at(n); return { n, total: t, each: t / n, measured: c.points.filter(p => p.n === n).map(p => ({ tps: v(p.tps), used: p.used, run: p.run })), fits_host: host.ceiling === null ? null : n <= host.ceiling, beyond_slots: c.slots ? n > v(c.slots) : null, extrapolated: n > f.n_max }; });
    const pick = (X, cap) => { const r = rows.filter(x => x.each >= X && (!cap || x.fits_host !== false)).pop(); if (!r) return { each_min: X, n: 0, total: 0 }; const w = curveWatts(c, r.n); return { each_min: X, n: r.n, total: r.total, each: r.each, at_max: r.n === max, past_host: host.ceiling !== null && r.n > host.ceiling, extrapolated: r.extrapolated, watts: w, tps_per_w: w ? r.total / w.w : null }; };
    const G = v(c.groups), K = Math.max(G, v(c.stages));
    return { cfg: c, contention: c.contention || [], ok: true, decode_capacity: true, fit: f, fit_warn: 100 * f.err_max > v(S.fit_warn_pct), rows, headline: pick(min), headline_in_memory: host.ceiling !== null && pick(min).past_host ? pick(min, true) : null, second: pick(second), asymptote: G / (K * f.B), host, power, min, second_min: second, max };
  }
  function curveReports(data, opts) { return data.curves.configs.map(c => curveReport(data, c, opts)); }

  // ---------- agent workload: useful output from decode capacity and prefill (data.workload, build.py) ----------
  // A coding agent's turn reads its fresh prompt tokens, then decodes its answer. With F fresh prompt tokens per output token,
  // a prefill rate P and a decode capacity D (total tok/s of the agents decoding together), useful output is at most
  // min(D, P / F); an engine whose reads stop every decoding slot (workload.engines.<engine>.reads_stall_decode) runs the two
  // in turn: 1 / (F / P + 1 / D). The smaller of D and P / F binds. A synthetic short-prompt run measures D only. This is the
  // server-bound limit of agentTurns below: it leaves out the tool gaps, so it reads high where the agents, not the server, bind
  // (one agent: 33 % high on the R8 run of 2026-10-26)
  function agentThroughput(t) {
    const { decode, prefill, fresh, stall } = t;
    const cap = prefill && fresh > 0 ? prefill / fresh : prefill ? Infinity : null;
    if (cap === null) return { prefill_cap: null, ceiling: decode, serial: null, expect: null, binding: null, busy: null };
    if (decode === null || decode === undefined) return { prefill_cap: cap, ceiling: cap, serial: null, expect: null, binding: null, busy: null };
    const read = fresh / prefill, serial = 1 / (read + 1 / decode), ceiling = Math.min(decode, cap);
    return { prefill_cap: cap, ceiling, serial, expect: stall ? serial : ceiling, binding: cap < decode ? 'prefill' : 'decode',
      busy: stall ? read / (read + 1 / decode) : Math.min(1, ceiling * read) };
  }
  const buildKey = (data, b) => { const m = new RegExp(v(data.workload.settings.build_key_re)).exec(String(b || '')); return m ? m[1] : b || null; };
  // the real-agent rows of a box/model/engine: same = the same build key, other = other builds (never borrowed as evidence)
  function agentEvidence(data, q) {
    const rows = ((data.workload || {}).measured || []).filter(x => x.box === q.box && x.model === q.model && x.engine === q.engine), k = buildKey(data, q.build);
    return { key: k, same: rows.filter(x => x.build_key === k), other: rows.filter(x => x.build_key !== k) };
  }
  // ---------- per-turn agent model (data.workload.turn_model) ----------
  // Turns per agent-minute is the primary output; useful tok/s = turns x each model's output a turn, which differs by model
  // (GLM wrote 76 tokens a turn where Flash-Next wrote 182 on the same prompts). Closed form on means: one agent alone cycles
  // in out / D1 + fresh / P + gap; the server spends out / Dn + fresh / P a turn (reads stall decode; else the larger of the
  // two); turns/s = min(n / cycle, 1 / server). Many agents and stall: useful = agentThroughput's serial term
  function agentTurns(t) {
    const { n, out, fresh, gap, decode_1, decode_n, prefill, stall } = t;
    const dec = out / decode_1, pre = fresh / prefill, cycle = dec + pre + gap;
    const server = stall ? out / decode_n + pre : Math.max(out / decode_n, pre), agents = n / cycle, rate = Math.min(agents, 1 / server);
    return { turns_per_s: rate, turns_per_agent_min: 60 * rate / n, useful_tps: rate * out, binding: agents < 1 / server ? 'agents' : 'server', cycle_s: cycle, server_s: server,
      share: { decode: dec / cycle, prefill: pre / cycle, gap: gap / cycle } };
  }
  // the row of a {agent count: tagged value} table at n, else at the nearest measured count (ties to the smaller); hold = the run
  // ids whose rows are skipped (a held-out score)
  const fromRun = (x, runs) => (x.runs || []).some(r => runs.includes(r));
  function byAgents(tbl, n, hold) {
    const ks = Object.keys(tbl || {}).map(Number).filter(k => !hold || !fromRun(tbl[k], hold)).sort((a, b) => Math.abs(a - n) - Math.abs(b - n) || a - b);
    return ks.length ? { row: tbl[ks[0]], at: ks[0] } : null;
  }
  // the inputs of one replay prediction: q = {trace, n, model, server, decode_1, decode_n, prefill, stall, warmup_s, window_s,
  // [gap_scale], [stagger_scale]}, the D / P / stall inputs tagged (an input may carry held_out: the value to use when its own
  // value came from the scored run). mode 'all' uses every input; 'held_out' skips the rows measured on the scored run and
  // falls back: output lengths to the recording's own, a relative tokenizer ratio to its base, a server's re-read to UNKNOWN (0
  // assumed, so the prediction is an upper bound). Every input used is listed with its tag; in_sample names those that came from
  // the scored run
  function turnInputs(data, q, run, mode) {
    const TM = (data.workload || {}).turn_model, runs = [].concat(run), hold = mode === 'held_out' ? runs : null, used = [], inS = [];
    const un = why => ({ status: 'UNMODELLED', why });
    if (!TM) return un('no data.workload.turn_model');
    const tr = TM.traces[q.trace], M = TM.models[q.model], S = TM.servers[q.server];
    if (!tr) return un('trace ' + q.trace + ' has no per-turn shape (data/trace-turns.json; tools/trace_turns.py)');
    if (!M) return un('model ' + q.model + ' has no turn_model.models row (tokenizer, output per turn)');
    if (!S) return un('server ' + q.server + ' has no turn_model.servers row (prefix re-read per turn)');
    const note = (name, x, basis) => { used.push({ name, v: v(x), tag: x.tag, src: x.src, runs: x.runs || [], basis: basis || null }); if (fromRun(x, runs)) inS.push(name); return v(x); };
    const tagged = (name, x) => { if (!x) throw new Error('turn query ' + run + ': input ' + name + ' missing'); return note(name, mode === 'held_out' && x.held_out && fromRun(x, runs) ? x.held_out : x); };
    const tok = TM.tokenizers[M.tokenizer], base = tok && (tok.relative_to ? TM.tokenizers[tok.relative_to] : tok);
    if (!tok || !base || !Array.isArray(v(base.per_agent)) || v(base.per_agent).length < v(tr).length)
      return un('model tokenizer has no complete per-agent ratio for the selected trace');
    const per = note('tokenizer ratio per recorded agent (' + (tok.relative_to || M.tokenizer) + ')', base.per_agent);
    const rel = !tok.relative_to ? 1 : hold && fromRun(tok.ratio, hold)
      ? note(M.tokenizer + ' tokenizer over ' + tok.relative_to, { v: 1, tag: 'INFERRED', src: 'INFERRED: held out (the ratio was measured on the scored run): assumed the same as ' + tok.relative_to }, 'fallback')
      : note(M.tokenizer + ' tokenizer over ' + tok.relative_to, tok.ratio);
    const o = byAgents(M.out_vs_recorded, q.n, hold);
    // q.out_ratio / q.reread_per_turn (tagged): a scenario's own value replaces the model's and the server's table rows
    const outR = q.out_ratio ? note('output a turn / recorded (the query\'s own)', q.out_ratio) : o ? note('output a turn / recorded (' + q.model + ' at ' + o.at + ' agents)', o.row, 'replay')
      : note('output a turn / recorded', { v: 1, tag: 'READ', src: 'the recording\'s own output lengths: no ' + (hold ? 'held-out ' : '') + 'measured replay of ' + q.model }, 'recording');
    const r = byAgents(S.reread_per_turn, q.n, hold);
    const reread = q.fresh_per_output !== undefined ? 0 : q.reread_per_turn ? note('re-read tokens a turn (the query\'s own)', q.reread_per_turn) : r ? note('re-read tokens a turn (' + q.server + ' at ' + r.at + ' agents)', r.row)
      : note('re-read tokens a turn', { v: 0, tag: 'UNKNOWN', src: 'UNKNOWN: no ' + (hold ? 'held-out ' : '') + 'measured prefix reuse of ' + q.server + '; 0 assumed, so turns and output are upper bounds' }, 'unknown');
    const set = (name, x, dflt) => x !== undefined ? note(name, typeof x === 'object' ? x : { v: x, tag: 'READ', src: 'the replay\'s own argument' }) : note(name, dflt);
    return { status: 'OK', run, mode, n: q.n, trace: v(tr), tok: per.map(x => x * rel), out_ratio: outR, reread, bound: r || q.fresh_per_output !== undefined ? null : 'upper',
      D1: tagged('decode one stream (D1)', q.decode_1), Dn: tagged('decode capacity at ' + q.n + ' agents (Dn)', q.decode_n), P: tagged('prefill (P)', q.prefill),
      stall: !!tagged('reads stall decode', q.stall), gap_scale: set('gap scale', q.gap_scale, TM.settings.gap_scale), stagger: set('stagger scale', q.stagger_scale, TM.settings.stagger_scale),
      overhead: note('client overhead a turn (s)', TM.settings.client_overhead_s),
      fresh_override: q.fresh_per_output === undefined ? null : tagged('fresh prompt tokens per output token (replaces trace growth and server re-read)', q.fresh_per_output),
      warmup_s: q.warmup_s, window_s: q.window_s, used, in_sample: inS };
  }
  // A replay of the trace by n agents, as agent-replay.py runs it: agent i loops recorded agent i mod A, starting at its recorded
  // offset x stagger, each turn followed by its recorded gap x gap_scale + the client overhead. A turn alone takes s = out / D1 +
  // fresh / P and costs the server w = out / Dn + fresh / P (stall; else the larger); while the running turns ask more than one
  // server-second a second (sum of w / s), all of them slow by the same factor. Counted like the replay: turns sent after the
  // warm-up and done by the window's end; time shares are the solo decode / prefill / gap seconds of those turns
  function agentReplaySim(I) {
    const A = I.trace, T1 = I.warmup_s + I.window_s, ag = [];
    for (let i = 0; i < I.n; i++) { const a = A[i % A.length]; ag.push({ a, tok: I.tok[i % A.length], k: 0, ready: a.start_s * I.stagger, run: null }); }
    let t = 0, turns = 0, out = 0, fresh = 0, dec = 0, pre = 0, gap = 0, guard = 0, decInt = 0;
    const start = g => { const [gp, ro, rf] = g.a.turns[g.k], o = ro * I.out_ratio, f = I.fresh_override === null || I.fresh_override === undefined ? rf * g.tok + I.reread : o * I.fresh_override, a = o / I.D1, b = f / I.P;
      g.run = { sent: t, o, f, a, b, s: Math.max(a + b, 1e-9), w: I.stall ? o / I.Dn + b : Math.max(o / I.Dn, b), p: 0, gap: gp * I.gap_scale + I.overhead }; };
    while (t < T1 - 1e-9) {
      if (++guard > 1e7) throw new Error('agentReplaySim: no progress at t ' + t);
      ag.forEach(g => { if (!g.run && g.ready <= t + 1e-9) start(g); });
      const act = ag.filter(g => g.run);
      // I.decode_at (aggregate decode tok/s at k requests decoding, k capped at I.slots): the server cost of a running turn
      // follows the number running now, so a queue batches (agentLandscape); without it every turn costs out / Dn
      if (I.decode_at) { const Dk = I.decode_at(Math.max(1, Math.min(act.length, I.slots || Infinity))); act.forEach(g => { const x = g.run, d = x.o / Dk; x.w = I.stall ? d + x.b : Math.max(d, x.b); }); }
      const load = act.reduce((x, g) => x + g.run.w / g.run.s, 0), f = load > 1 ? 1 / load : 1;
      let dt = T1 - t;
      act.forEach(g => { dt = Math.min(dt, (1 - g.run.p) * g.run.s / f); });
      ag.forEach(g => { if (!g.run) dt = Math.min(dt, g.ready - t); });
      if (t >= I.warmup_s) decInt += dt * act.reduce((x, g) => x + g.run.a / g.run.s, 0);
      t += dt;
      act.forEach(g => {
        g.run.p += dt * f / g.run.s;
        if (g.run.p < 1 - 1e-9) return;
        const x = g.run;
        if (x.sent >= I.warmup_s) { turns++; out += x.o; fresh += x.f; dec += x.a; pre += x.b; gap += x.gap; }
        g.ready = t + x.gap; g.run = null; g.k = (g.k + 1) % g.a.turns.length;
      });
    }
    const all = dec + pre + gap;
    return { turns, out_tokens: out, fresh_tokens: fresh, turns_per_agent_min: turns / (I.n * I.window_s / 60), useful_tps: out / I.window_s,
      out_per_turn: turns ? out / turns : null, fresh_per_output: out ? fresh / out : null, decoding_at_once: decInt / I.window_s, share: all ? { decode: dec / all, prefill: pre / all, gap: gap / all } : null };
  }
  // ---------- the served engine, turn by turn (agentEngineSim) ----------
  // agentReplaySim prices the server as one fluid capacity: a read costs the other agents only once the server saturates, and
  // every conversation keeps its prefix. A served engine with S slots (XeStrata --batch, llama-server -np) does neither:
  // (1) slot eviction: a request takes the slot that holds its conversation, else an empty slot, else the least recently used
  //     idle slot, whose conversation is evicted; a request whose conversation was evicted re-reads its whole prompt less the
  //     prefix every slot shares (shared_prefix); with S slots busy, requests queue (FIFO). With B conversations over S slots a
  //     returning agent keeps its prefix only while fewer than S other conversations were served since its last turn, so the
  //     evicted share follows from S, B and every agent's cycle (tool gap + read + decode): it is simulated, never fitted;
  // (2) reads block decode: each group of slots (slot_groups G, XeStrata --batch-groups) reads one prompt at a time at P (FIFO),
  //     so G groups read at once (MEASURED: R8 prefill busy 111-117 % of the window at B 33 / 40); while a read longer than
  //     read_batch_max tokens runs, no slot decodes (the read drains the pipe and every decoding slot waits, generate.cpp:
  //     workload.engines.<engine>.reads_stall_decode); I.read_blocks 'group' blocks only the reading group's slots (an arm).
  //     Overlapping reads block once, so the stall share runs under the read-busy share (MEASURED R8 stall / busy 0.79-0.85 at
  //     B 8..40, where a per-group block would give ~1 / G). A shorter read rides inside the decode steps;
  //     a cold first turn (a new conversation) reads its prompt less the shared prefix once any read has put it in the cache
  //     (the R8 door run: every cold first turn had the 4,924 shared tokens cached);
  // (3) decode: the k requests decoding share decode_at(k) tok/s (aggregate), k <= S;
  // (4) admission (declared, never assumed): 'engine' = the engine's own (1); 'door' = a front door that lets at most
  //     `conversations` conversations hold a lease; one without a lease waits FIFO before it sends, and a holder idle between
  //     turns for idle_release_s while another waits gives its lease up (it queues again at its next turn).
  // Counted like the replay: turns sent (past the door) after the warm-up and done by the window's end.
  function agentEngineSim(I) {
    const A = I.trace, T1 = I.warmup_s + I.window_s, W0 = I.warmup_s, S = I.slots, adm = I.admission || {}, door = adm.policy === 'door';
    if (!(S >= 1) || !Number.isFinite(S)) throw new Error('agentEngineSim: needs a finite slot count');
    if (door && !(adm.conversations >= 1)) throw new Error('agentEngineSim: a door needs conversations >= 1');
    const N = door ? adm.conversations : Infinity, X = door && Number.isFinite(adm.idle_release_s) ? adm.idle_release_s : Infinity;
    const shared = I.shared_prefix || 0, small = I.read_batch_max || 0, ovr = I.fresh_override === null || I.fresh_override === undefined ? null : I.fresh_override;
    const ag = [];
    for (let i = 0; i < I.n; i++) { const a = A[i % A.length]; ag.push({ i, a, tok: I.tok[i % A.length], k: 0, pass: 0, depth: 0, st: 'gap', ready: a.start_s * I.stagger, idle0: null, lease: false, turn: null }); }
    // slot_groups G > 1: XeStrata --batch-groups G, its frontend's pick_slot (serve/server.py of the served image): an empty slot
    // before one holding another conversation; then a free slot below a busy one of its group (a hole), first in slot order
    // (the groups interleaved: 0, gs, 1, gs + 1, ...); then the group with the fewest busy slots, empty first, least recently
    // used. G 1 (or none): an empty slot, else the least recently used idle one
    const G = I.slot_groups > 1 ? I.slot_groups : 1, gs = Math.ceil(S / G);
    const slots = Array.from({ length: S }, (_, b) => ({ b, grp: Math.floor(b / gs), conv: null, busy: false, last: -Infinity }));
    const order = []; for (let k = 0; k < gs; k++) for (let q = 0; q < G; q++) if (q * gs + k < S) order.push(q * gs + k);
    const pos = new Array(S); order.forEach((b, i) => { pos[b] = i; });
    const pick = () => {
      let free = slots.filter(x => !x.busy);
      const empty = free.filter(x => x.conv === null);
      if (G === 1) return empty.length ? empty.sort((a, b) => pos[a.b] - pos[b.b])[0] : free.sort((a, b) => a.last - b.last)[0];
      if (empty.length) free = empty;
      const holes = free.filter(x => slots.some(y => y.busy && y.grp === x.grp && y.b > x.b));
      if (holes.length) return holes.sort((a, b) => pos[a.b] - pos[b.b])[0];
      const busyIn = q => slots.filter(y => y.busy && y.grp === q).length;
      return free.sort((a, b) => busyIn(a.grp) - busyIn(b.grp) || (a.conv !== null) - (b.conv !== null) || a.last - b.last || pos[a.b] - pos[b.b])[0];
    };
    const doorQ = [], engQ = [], reads = Array.from({ length: G }, () => []);
    let prefixWarm = false;
    const acc = { turns: 0, out: 0, fresh: 0, misses: 0, warm: 0, wait: 0, decInt: 0, stall: 0, readBusy: 0, per: new Array(I.n).fill(0), maxDoorQ: 0, maxEngQ: 0, revoked: 0, dec: 0, pre: 0, gap: 0 };
    let t = 0, guard = 0, leases = 0;
    const convOf = g => g.i + ':' + g.pass;
    const grant = () => {
      for (;;) {
        if (!doorQ.length) return;
        if (leases >= N) {
          const idle = ag.filter(h => h.lease && h.st === 'gap' && t - h.idle0 >= X - 1e-9).sort((a, b) => a.idle0 - b.idle0)[0];
          if (!idle) return;
          idle.lease = false; leases--; acc.revoked++;
        }
        const g = doorQ.shift(); g.lease = true; leases++; g.st = 'eng'; g.turn.sent = t; engQ.push(g);
      }
    };
    const want = g => {
      g.turn = { want: t, sent: null };
      if (!door || g.lease) { g.st = 'eng'; g.turn.sent = t; engQ.push(g); return; }
      g.st = 'door'; doorQ.push(g); acc.maxDoorQ = Math.max(acc.maxDoorQ, doorQ.length);
    };
    const admit = () => {
      while (engQ.length && slots.filter(s => s.busy).length < S) {
        const g = engQ.shift(), c = convOf(g), [, ro, rf] = g.a.turns[g.k], growth = rf * g.tok, o = ro * I.out_ratio;
        let s = slots.find(x => x.conv === c), miss = false;
        if (!s) { s = pick(); miss = g.k > 0; }
        s.conv = c; s.busy = true; g.slot = s;
        const cold = g.k === 0 && prefixWarm && shared > 0;
        const f = ovr !== null ? o * ovr : miss ? Math.max(growth, g.depth + growth - shared) : (cold ? Math.max(0, growth - shared) : growth) + I.reread;
        Object.assign(g.turn, { f, o, left: f, miss, cold, depth_after: g.depth + growth });
        g.st = 'pre'; reads[s.grp].push(g);
      }
      acc.maxEngQ = Math.max(acc.maxEngQ, engQ.length);
    };
    const finish = g => {
      const x = g.turn; g.slot.busy = false; g.slot.last = t;
      if (x.sent >= W0 - 1e-9 && t <= T1 + 1e-9) {
        acc.turns++; acc.out += x.o; acc.fresh += x.f; acc.per[g.i]++; acc.wait += x.sent - x.want;
        if (g.k > 0) { if (x.miss) acc.misses++; else acc.warm++; }
      }
      const gp = g.a.turns[g.k][0] * I.gap_scale + I.overhead;
      g.depth = x.depth_after; g.k++; if (g.k >= g.a.turns.length) { g.k = 0; g.pass++; g.depth = 0; }
      g.st = 'gap'; g.ready = t + gp; g.idle0 = t; g.turn = null;
    };
    const Dk = k => I.decode_at ? I.decode_at(Math.max(1, Math.min(k, S))) : Math.min(k * I.D1, I.Dn);
    while (t < T1 - 1e-9) {
      if (++guard > 5e6) throw new Error('agentEngineSim: no progress at t ' + t);
      ag.forEach(g => { if (g.st === 'gap' && g.ready <= t + 1e-9) want(g); });
      grant(); admit();
      const heads = reads.map(q => q[0] || null), longG = heads.map(h => !!h && I.stall && h.turn.f > small), anyLong = longG.some(Boolean);
      const blockedG = I.read_blocks === 'group' ? longG : longG.map(() => anyLong);
      const decs = ag.filter(g => g.st === 'dec'), live = decs.filter(g => !blockedG[g.slot.grp]), rate = live.length ? Dk(live.length) / live.length : 0;
      const nHeads = heads.filter(Boolean).length;
      let dt = T1 - t;
      ag.forEach(g => { if (g.st === 'gap') dt = Math.min(dt, g.ready - t); });
      if (doorQ.length && leases >= N && Number.isFinite(X)) ag.forEach(g => { if (g.lease && g.st === 'gap') { const r = g.idle0 + X - t; if (r > 1e-9) dt = Math.min(dt, r); } });
      heads.forEach(h => { if (h) dt = Math.min(dt, h.turn.left / I.P); });
      if (rate > 0) live.forEach(g => { dt = Math.min(dt, g.turn.left / rate); });
      dt = Math.max(dt, 0);
      const ov = Math.max(0, Math.min(t + dt, T1) - Math.max(t, W0));
      if (ov > 0) {
        acc.decInt += ov * decs.length; if (decs.length) acc.stall += ov * (decs.length - live.length) / decs.length; acc.readBusy += ov * nHeads;
        ag.forEach(g => { if (g.st === 'dec') acc.dec += ov; else if (g.st === 'pre' || g.st === 'eng') acc.pre += ov; else acc.gap += ov; });
      }
      t += dt;
      heads.forEach((h, q) => { if (!h) return; h.turn.left -= I.P * dt; if (h.turn.left <= 1e-6) { reads[q].shift(); prefixWarm = true; h.st = 'dec'; h.turn.left = h.turn.o; if (h.turn.o <= 0) finish(h); } });
      if (rate > 0) live.forEach(g => { g.turn.left -= rate * dt; if (g.turn.left <= 1e-6) finish(g); });
    }
    const all = acc.dec + acc.pre + acc.gap, ret = acc.misses + acc.warm;
    return { turns: acc.turns, out_tokens: acc.out, fresh_tokens: acc.fresh, turns_per_agent_min: acc.turns / (I.n * I.window_s / 60), useful_tps: acc.out / I.window_s,
      out_per_turn: acc.turns ? acc.out / acc.turns : null, fresh_per_output: acc.out ? acc.fresh / acc.out : null, decoding_at_once: acc.decInt / I.window_s,
      share: all ? { decode: acc.dec / all, prefill: acc.pre / all, gap: acc.gap / all } : null,
      engine: { evicted_share: ret ? acc.misses / ret : null, returning_turns: ret, stall_share: acc.stall / I.window_s, read_busy_share: acc.readBusy / I.window_s,
        door_wait_mean_s: acc.turns ? acc.wait / acc.turns : null, starved_agents: acc.per.filter(x => x === 0).length, turns_by_agent: acc.per,
        max_door_queue: acc.maxDoorQ, max_engine_queue: acc.maxEngQ, leases_revoked: acc.revoked } };
  }
  // one replay prediction: the inputs (turnInputs) and the simulated replay (agentReplaySim)
  function agentTurnPredict(data, q, run, mode) {
    const positive = ['decode_1', 'decode_n', 'prefill'].every(k => Number.isFinite(v(q[k])) && v(q[k]) > 0);
    const nonnegative = ['gap_scale', 'stagger_scale', 'fresh_per_output'].every(k => q[k] === undefined || Number.isFinite(v(q[k])) && v(q[k]) >= 0);
    if (!Number.isInteger(q.n) || q.n < 1 || !positive || !nonnegative || !Number.isFinite(q.warmup_s) || q.warmup_s < 0 || !Number.isFinite(q.window_s) || q.window_s <= 0)
      return { status: 'UNMODELLED', why: 'turn model needs positive D1, Dn, P and window seconds, an integer agent count, and nonnegative warm-up, gap, stagger and fresh settings' };
    const I = turnInputs(data, q, run, mode || 'all');
    if (I.status !== 'OK') return I;
    if (q.decode_at) { I.decode_at = q.decode_at; I.slots = q.slots || null; }
    // a declared admission policy (agentLandscape opts.admission) runs the served engine turn by turn (agentEngineSim)
    if (q.admission) {
      if (!(q.slots >= 1)) return { status: 'UNMODELLED', why: 'an admission policy needs the engine\'s slot count' };
      Object.assign(I, { slots: q.slots, admission: v(q.admission), shared_prefix: q.shared_prefix ? v(q.shared_prefix) : 0, read_batch_max: q.read_batch_max ? v(q.read_batch_max) : 0, slot_groups: q.slot_groups ? v(q.slot_groups) : 1, read_blocks: q.read_blocks ? v(q.read_blocks) : 'all' });
    }
    const sim = q.admission ? agentEngineSim(I) : agentReplaySim(I);
    return Object.assign({ status: 'MODEL', mode: I.mode, bound: I.bound, in_sample: I.in_sample, inputs: I.used }, sim);
  }
  // the prediction files' re-scores (data.workload.turn_rescore, build.py from each file's rescore_turn_model block): the file's
  // own before numbers, today's model with every input (after) and with the scored run held out (held_out), against the
  // measured numbers; err = (pred - meas) / meas
  function agentTurnRescore(data) {
    const err = (p, m) => p === null || p === undefined || !m ? null : p / m - 1;
    return ((data.workload || {}).turn_rescore || []).map(e => {
      const meas = { turns: e.measured.turns_per_agent_min ? v(e.measured.turns_per_agent_min) : null, useful: e.measured.useful_tps ? v(e.measured.useful_tps) : null };
      const bef = { turns: e.before.turns_per_agent_min ? v(e.before.turns_per_agent_min) : null, useful: e.before.useful_tps ? v(e.before.useful_tps) : null };
      const row = (p, s) => p.status !== 'MODEL' ? { status: p.status, why: p.why } : { status: p.status, turns: p.turns_per_agent_min, useful: p.useful_tps, bound: p.bound, in_sample: p.in_sample,
        share: p.share, out_per_turn: p.out_per_turn, err_turns: err(p.turns_per_agent_min, meas.turns), err_useful: err(p.useful_tps, meas.useful), stored: s || null };
      const st = e.after_at_commit || {};
      return { id: e.id, file: e.file, run: e.run, label: e.label || e.id, n: e.query.n, measured: meas, before: Object.assign(bef, { err_turns: err(bef.turns, meas.turns), err_useful: err(bef.useful, meas.useful), src: e.before.src || null }),
        after: row(agentTurnPredict(data, e.query, e.run, 'all'), st.all), held_out: row(agentTurnPredict(data, e.query, e.run, 'held_out'), st.held_out) };
    });
  }
  // ---------- prefill from first principles (data.workload.prefill_pass) ----------
  // A read of T fresh tokens runs in chunks of C; each chunk is one pass over the experts of every stage (one card each): the
  // bytes a card does not hold cross its PCIe link (less the CPU's share), the bytes it holds are read once from VRAM; then
  // m per token. t(T) = ceil(T / C) x t_pass + T x m. m = sum over stages of layer share x 2 x active / (eta x card int8 ops),
  // eta the one FIT factor per engine (its fit rows' server prompt time). Attention / DeltaNet depth terms are left out.
  const lastNum = (re, s) => { const g = new RegExp(re, 'g'); let h, x = null; while ((h = g.exec(s))) x = h[1]; return x === null || x === undefined ? null : Number(x); };
  function passTerms(data, q, o) {
    const PP = (data.workload || {}).prefill_pass, E = PP && PP.engines[q.engine], S = PP && PP.stores[q.model], box = data.boxes[q.box], m = data.models[q.model] || {};
    const cache = S && S.cache_bytes_per_card[q.box], flags = String(q.flags || ''), st = PP && PP.settings, oo = o || {};
    const miss = !PP ? 'no data.workload.prefill_pass' : !E ? 'engine ' + q.engine + ' has no prefill pass model' : !S ? 'model ' + q.model + ' has no expert store bytes (prefill_pass.stores)'
      : !cache ? 'no expert cache bytes of ' + q.model + ' on ' + q.box + ' (prefill_pass.stores.' + q.model + '.cache_bytes_per_card)' : !box ? 'box ' + q.box + ' unknown' : null;
    if (miss) return { status: 'UNMODELLED', why: miss };
    const L = Number(v(m.L)), cut = lastNum(v(st.split_re), flags), layers = cut > 0 && cut < L ? [cut, L - cut] : [L];
    const share = oo.cpu_share !== undefined ? oo.cpu_share : lastNum(v(st.cpu_share_re), flags) !== null ? lastNum(v(st.cpu_share_re), flags) : v(E.cpu_share_default);
    const C = oo.chunk || lastNum(v(st.chunk_re), flags) || v(E.chunk_default), lane = v(st.pcie_gbs_per_lane), eff = v(st.link_eff);
    const stages = layers.map((n, i) => {
      const g = box.gpus[i], geo = g && g.pcie ? v(g.pcie) : null, card = g ? Object.keys(PP.cards).find(k => new RegExp(PP.cards[k].match).test(v(g.model) + ' ' + g.name)) : null;
      const bytes = v(S.expert_bytes) * n / L, res = Math.min(bytes, v(cache));
      return { layers: n, gpu: g ? g.id : null, card, resident_gb: res / 1e9, nonresident_gb: (bytes - res) / 1e9, vram_gbs: g ? v(g.bw) : null,
        link_gbs: geo && lane[geo.gen] ? lane[geo.gen] * geo.width * eff : null, int8_tops: card ? v(PP.cards[card].int8_tops) : null };
    });
    const bad = stages.find(s => !s.gpu || !s.link_gbs || !s.vram_gbs || !s.int8_tops);
    if (bad) return { status: 'UNMODELLED', why: 'stage ' + (stages.indexOf(bad) + 1) + ' of ' + q.box + ': no card, PCIe geometry, VRAM bandwidth or int8 ops row (prefill_pass.cards)' };
    const t_pass = stages.reduce((a, s) => a + s.nonresident_gb * (1 - share) / s.link_gbs + s.resident_gb / s.vram_gbs, 0);
    return { status: 'OK', PP, E, chunk: C, cpu_share: share, stages, t_pass, active: v(S.active_params), L };
  }
  // chunks per read over the measured read-size mix at chunk C
  const chunksPerRead = (PP, C) => { const x = v(PP.read_mix); return x.reduce((a, t) => a + Math.ceil(t / C), 0) / x.length; };
  // eta of an engine from its fit rows: m_ref = (prompt_s / reads - chunks per read x t_pass) / (fresh / reads) on the fit box
  function prefillFit(data, engine) {
    const PP = data.workload.prefill_pass, F = PP.engines[engine].fit, ref = passTerms(data, { box: F.box, model: F.model, engine, flags: v(F.flags) });
    if (ref.status !== 'OK') return { status: 'UNMODELLED', why: 'fit recipe: ' + ref.why };
    const reads = v(F.reads), mean = v(F.fresh_tokens) / reads, m = (v(F.prompt_s) / reads - chunksPerRead(PP, ref.chunk) * ref.t_pass) / mean;
    const ops = ref.stages.reduce((a, s) => a + s.layers / ref.L / s.int8_tops, 0);
    return { status: m > 0 ? 'FIT' : 'UNMODELLED', why: m > 0 ? null : 'the fit rows leave no per-token time after the pass term', m_ref: m, mean_read: mean, t_pass_ref: ref.t_pass, eta: 2 * ref.active * ops / (m * 1e12),
      src: 'FIT on ' + v(F.rows).join(', ') + ': ' + reads + ' reads, ' + v(F.fresh_tokens) + ' fresh tokens, ' + v(F.prompt_s) + ' s of server prompt time, t_pass ' + ref.t_pass.toFixed(3) + ' s at chunk ' + ref.chunk };
  }
  // predicted prefill of a box/model/engine/flags: per read tps(T); P over the agent read mix (separate reads, each pays its
  // passes); shared_tps = the long-read rate a cross-agent batched read would approach (one pass shared by every read in it)
  function prefillPass(data, q, o) {
    const b = passTerms(data, q, o);
    if (b.status !== 'OK') return b;
    const f = prefillFit(data, q.engine);
    if (f.status !== 'FIT') return { status: 'UNMODELLED', why: f.why };
    const m = b.stages.reduce((a, s) => a + s.layers / b.L * 2 * b.active / (f.eta * s.int8_tops * 1e12), 0), C = b.chunk;
    const tps = T => T / (Math.ceil(T / C) * b.t_pass + T * m), cpr = chunksPerRead(b.PP, C);
    return { status: 'MODEL', tag: 'MODEL', chunk: C, cpu_share: b.cpu_share, stages: b.stages, t_pass: b.t_pass, m_ms: m * 1e3, eta: f.eta, fit: f, tps,
      mix_tps: f.mean_read / (cpr * b.t_pass + f.mean_read * m), shared_tps: 1 / (b.t_pass / C + m), chunks_per_read: cpr, mean_read: f.mean_read,
      src: 'MODEL prefillPass: t_pass ' + b.t_pass.toFixed(3) + ' s a ' + C + '-token chunk (' + b.stages.map(s => s.layers + ' layers on ' + s.gpu + ': ' + s.nonresident_gb.toFixed(1) + ' GB streamed x ' + (1 - b.cpu_share).toFixed(2) + ' at ' + s.link_gbs.toFixed(1) + ' GB/s + ' + s.resident_gb.toFixed(1) + ' GB from VRAM').join('; ') +
        '), m ' + (m * 1e3).toFixed(3) + ' ms a token (eta ' + f.eta.toFixed(4) + ', ' + f.src + '), over the measured read mix (mean ' + Math.round(f.mean_read) + ' tokens, ' + cpr.toFixed(2) + ' chunks a read)' };
  }
  // the predicted P as a tagged term, or null
  const prefillPredicted = (data, q) => { const p = prefillPass(data, q); return p.status === 'MODEL' ? { v: p.mix_tps, tag: 'MODEL', evidence: 'inferred', src: p.src, model: p } : null; };
  // the held-out checks of data.workload.prefill_pass.checks: predicted tok/s at each end of the row's CPU-share range
  function prefillPassChecks(data) {
    return (((data.workload || {}).prefill_pass || {}).checks || []).map(x => {
      const ps = x.cpu_share.map(s => prefillPass(data, { box: x.box, model: x.model, engine: x.engine }, { cpu_share: s }));
      const pred = ps.map(p => p.status === 'MODEL' ? p.tps(x.tokens) : null), meas = v(x.measured);
      return { id: x.id, tokens: x.tokens, cpu_share: x.cpu_share, pred, measured: meas, why: ps.map(p => p.why).filter(Boolean)[0] || null,
        brackets: pred.every(y => y !== null) && Math.min(...pred) <= Math.max(...meas) && Math.max(...pred) >= Math.min(...meas) };
    });
  }
  // the workload profile and its F (opts.fresh overrides), the engine's stall rule
  function agentProfile(data, opts) {
    const W = data.workload, o = opts || {}, name = o.profile || v(W.settings.default_profile), p = W.profiles[name];
    if (!p) throw new Error('workload profile ' + name + ' is not in data.workload.profiles (' + Object.keys(W.profiles).join(', ') + ')');
    return { name, p, fresh: o.fresh !== undefined ? { v: o.fresh, tag: 'READ', src: 'a setting (report.js --fresh)' } : p.fresh_per_output };
  }
  // Adapt a curve to the existing replay model. A unique engine match selects a server prior, not measured cache behavior
  // of this build. No match, an ambiguous match, or an engine mismatch stays UNMODELLED.
  function planTurns(data, c, o, pr, n, D, Dtag, P, st, fit) {
    const TM = (data.workload || {}).turn_model, un = why => ({ status: 'UNMODELLED', why });
    if (!TM) return un('no data.workload.turn_model');
    if (pr.p.decode_capacity_only) return un('synthetic-short measures decode capacity only, not agent turns');
    const trace = o.trace || v(pr.p.turn_trace), candidates = Object.keys(TM.servers).filter(k => TM.servers[k].engine === c.engine);
    const server = o.server || c.turn_server || (candidates.length === 1 ? candidates[0] : null);
    if (!trace) return un('profile has no per-turn trace. Set --trace NAME');
    if (!server || !TM.servers[server]) return un('no unambiguous server prefix re-read prior. Set --server NAME');
    if (TM.servers[server].engine !== c.engine) return un('server prefix re-read prior ' + server + ' belongs to another engine');
    if (!st) return un('engine has no reads_stall_decode rule');
    const solo = c.points.filter(p => p.n === 1).slice(-1)[0];
    const D1 = o.decode_one !== undefined ? { v: o.decode_one, tag: 'READ', src: 'a setting (--decode-one, decode tok/s)' }
      : solo ? solo.tps : fit.ok ? { v: fit.at(1), tag: 'FIT', src: 'solo decode from the synthetic curve at n=1. This can extrapolate below measured counts' } : null;
    const q = { trace, server, model: c.model, n, decode_1: D1,
      decode_n: n === 1 ? D1 : { v: D, tag: Dtag, src: 'synthetic decode capacity at n=' + n + ' of curve ' + c.id }, prefill: P, stall: st,
      warmup_s: o.warmup_s !== undefined ? o.warmup_s : v(TM.settings.plan_warmup_s),
      window_s: o.window_s !== undefined ? o.window_s : v(TM.settings.plan_window_s) };
    ['gap_scale', 'stagger_scale'].forEach(k => { if (o[k] !== undefined) q[k] = { v: o[k], tag: 'READ', src: 'a plan setting (--' + k.replace('_', '-') + ')' }; });
    if (o.fresh !== undefined) q.fresh_per_output = pr.fresh;
    const p = agentTurnPredict(data, q, [], 'all');
    return Object.assign(p, { trace, server, warmup_s: q.warmup_s, window_s: q.window_s, query: q,
      server_basis: o.server || c.turn_server ? 'setting' : 'unique engine prior (assumed, not build-verified)',
      answer_length: 'measured or inferred model replay ratio, else recorded lengths. Not predicted',
      prefix_reread: o.fresh !== undefined ? 'replaced by --fresh' : 'server prior, not predicted',
      calibration: 'POST-HOC inputs. This plan is not a held-out accuracy test' });
  }
  // Per curve: primary output is simulated turns per agent-minute. The old fluid fields remain compatible aliases of
  // legacy_server, a server-bound diagnostic that omits tool gaps. Real rows below still check that legacy diagnostic.
  function agentPlan(data, c, opts) {
    const W = data.workload, o = opts || {}, pr = agentProfile(data, o), F = v(pr.fresh), ev = agentEvidence(data, c);
    const st = W.engines[c.engine] && W.engines[c.engine].reads_stall_decode, stall = st ? !!v(st) : false;
    const last = agentReference(ev);
    // no real-agent row of the build: P PREDICTED from the box and recipe (prefillPass, tag MODEL)
    const P = o.prefill !== undefined ? { v: o.prefill, tag: 'READ', src: 'a setting (report.js --prefill)' } : last ? last.prefill_tps : prefillPredicted(data, { box: c.box, model: c.model, engine: c.engine, flags: c.flags });
    const f = curveFit(data, c), ns = o.n || v(W.settings.table_n), lev = W.engines[c.engine] && W.engines[c.engine].prefill_levers;
    const rows = ns.map(n => {
      const pt = c.points.filter(p => p.n === n && p.used), D = f.ok ? f.at(n) : pt.length ? v(pt[pt.length - 1].tps) : null;
      const t = agentThroughput({ decode: D, prefill: P ? v(P) : null, fresh: F, stall });
      const cpu = cpuPoolCheck(data, { box: c.box, model: c.model, engine: c.engine, ctx: c.ctx, draft: c.draft, serving: Object.assign({}, c.serving, { batch: n }),
        engine_build: c.build, gpu_split: v(c.stages) > 1 ? { banks_gib: Array(v(c.stages)).fill(0) } : undefined, conditions: { engine_flags: c.flags } });
      // each real-agent row at n judged with its own P and F (the arm's own reads), so the miss is the model's, not the workload's
      const real = ev.same.filter(x => x.n === n).map(x => { const w = agentThroughput({ decode: D, prefill: x.prefill_tps ? v(x.prefill_tps) : null, fresh: x.fresh_per_output ? v(x.fresh_per_output) : F, stall }), m = v(x.agent_tps);
        return { run: x.run, label: x.label, tps: m, prefill: x.prefill_tps ? v(x.prefill_tps) : null, fresh: x.fresh_per_output ? v(x.fresh_per_output) : null, own: w, err_expect: w.expect ? w.expect / m - 1 : null, err_ceiling: w.ceiling ? w.ceiling / m - 1 : null, busy: x.prefill_busy_pct ? v(x.prefill_busy_pct) / 100 : null }; });
      const Dtag = f.ok ? 'FIT' : pt.length ? 'MEASURED' : 'UNKNOWN', turn = planTurns(data, c, o, pr, n, D, Dtag, P, st, f);
      return Object.assign({ n, decode: D, decode_tag: Dtag, extrapolated: f.ok ? n > f.n_max : !pt.length, cpu_pool: cpu, real, turn, legacy_server: t }, t);
    });
    // each lever carries its own tag when it has one (else the list's); a MEASURED gain whose smallest value is within the
    // replay's run-to-run noise (settings.replay_noise_pct) is marked inside_noise
    const noise = W.settings.replay_noise_pct ? v(W.settings.replay_noise_pct) / 100 : null;
    const levers = P && lev ? v(lev).map(x => { const tag = x.tag || lev.tag;
      return { id: x.id, lever: x.lever, gain: x.gain, tag, src: x.src || lev.src, prefill_cap: x.gain.map(g => v(P) * g / F),
        inside_noise: tag === 'MEASURED' && noise !== null ? Math.min(...x.gain) - 1 < noise : null }; }) : [];
    return { cfg: c, profile: pr.name, fresh: pr.fresh, primary: 'turns_per_agent_min', prefill: P, prefill_run: o.prefill !== undefined || !last ? null : last.run, prefill_basis: o.prefill !== undefined ? 'setting' : last ? 'measured' : P ? 'model' : null, stall, evidence: ev, decode_capacity_only: true, fit_ok: f.ok, why: f.ok ? null : f.why, rows, levers, noise_pct: noise === null ? null : 100 * noise };
  }
  // the prefill reference of a build: its real-agent row with the highest agent tok/s (the arm a server of that build would run)
  const agentReference = ev => ev.same.filter(x => x.prefill_tps).reduce((a, x) => !a || v(x.agent_tps) > v(a.agent_tps) ? x : a, null);
  function agentPlans(data, opts) { return data.curves.configs.map(c => agentPlan(data, c, opts)); }
  // ---------- every weight resident in VRAM, one server per card (recipe.resident_plan) ----------
  // For a model with no synthetic curve and no real-agent row: decode and prefill come from the card's bandwidth and a
  // per-layer launch cost fitted on measured decode anchors of the same engine (t = bytes x a + layers x c, a in ms/GB);
  // n agents split over the instances by conversation affinity, each instance replays its share (agentTurnPredict) with at
  // most `slots` turns decoding together. Every input is a tagged value of the recipe; the fit is least squares on the anchors
  function residentTerms(R) {
    const I = R.inputs, val = k => { if (!I[k]) throw new Error('resident_plan input ' + k + ' missing'); return v(I[k]); };
    const A = val('decode_anchors'), fit = lsq(A.map(x => [x.read_gb, x.layers]), A.map(x => 1000 / x.tps));
    const a = fit[0], c = fit[1], s = val('engine_slowdown'), L = val('layers'), E = val('experts'), k = val('experts_used');
    const kvd = d => val('kv_bytes_per_token') * d / 1e9 * a / val('kv_read_eff');
    const step = (b, d) => s * ((val('dense_read_gb') + val('experts_all_gb') * (1 - Math.pow(1 - k / E, b))) * a + L * c * (b > 1 ? val('batch_layer_mult') : 1)) + b * kvd(d);
    const decode = (b, d) => b * 1000 / step(b, d);
    const tpass = val('file_gb') * a, C = val('chunk');
    const prefill = (T, d) => T * 1000 / (Math.ceil(T / C) * tpass + T * (s * val('linear_ms_per_token') + val('attn_ms_per_token_per_ctx') * d));
    return { a_ms_per_gb: a, layer_ms: c, bw_eff_gbs: 1000 / a, decode, prefill, step, tpass_ms: tpass };
  }
  // optional lever inputs of a resident_plan (each tagged; absent = the stock server): stall (false = a prompt read never
  // delays the other slots' decode: disaggregated or non-blocking prefill), decode_mult (x decode, e.g. a draft model's
  // accepted tokens a verify step over the verify cost), out_ratio and reread_per_turn (replace the turn model's model and
  // server rows), gap_scale (x the recorded tool gaps)
  const RESIDENT_LEVERS = ['stall', 'decode_mult', 'out_ratio', 'reread_per_turn', 'gap_scale'];
  function residentAgentPlan(data, recipe, ns) {
    const R = recipe.resident_plan, I = R.inputs, T = residentTerms(R);
    ['depth_tokens', 'slots', 'instances', 'read_tokens', 'warmup_s', 'window_s'].forEach(k => { if (!I[k]) throw new Error('resident_plan input ' + k + ' missing'); });
    const d = v(I.depth_tokens), slots = v(I.slots), cards = v(I.instances);
    const P = { v: T.prefill(v(I.read_tokens), d), tag: 'MODEL', src: 'residentTerms prefill at ' + v(I.read_tokens) + ' fresh tokens a read, depth ' + d };
    const rows = ns.map(n => {
      const per = Array.from({ length: cards }, (_, i) => Math.floor(n / cards) + (i < n % cards ? 1 : 0)).filter(x => x > 0);
      const sims = per.map(nc => {
        const b = Math.min(nc, slots), dm = I.decode_mult ? v(I.decode_mult) : 1;
        const q = { trace: R.trace, server: R.server, model: recipe.model, n: nc, warmup_s: v(I.warmup_s), window_s: v(I.window_s), stall: I.stall || { v: true, tag: 'READ', src: R.stall_src },
          decode_1: { v: dm * T.decode(1, d), tag: 'MODEL', src: 'residentTerms decode, 1 stream at depth ' + d + (dm !== 1 ? ' x decode_mult ' + dm : '') },
          decode_n: { v: dm * T.decode(b, d), tag: 'MODEL', src: 'residentTerms decode, ' + b + ' streams (slots ' + slots + ') at depth ' + d + (dm !== 1 ? ' x decode_mult ' + dm : '') }, prefill: P };
        ['out_ratio', 'reread_per_turn', 'gap_scale'].forEach(k => { if (I[k]) q[k] = I[k]; });
        const p = agentTurnPredict(data, q, [], 'all');
        if (p.status !== 'MODEL') return { nc, p };
        const fresh = p.fresh_tokens / Math.max(p.turns, 1), out = p.out_per_turn, gap = p.share ? p.share.gap / (p.share.decode + p.share.prefill) * (out / q.decode_1.v + fresh / P.v) : 0;
        const cf = agentTurns({ n: nc, out, fresh, gap, decode_1: q.decode_1.v, decode_n: q.decode_n.v, prefill: P.v, stall: !!v(q.stall) });
        const binding = cf.binding === 'agents' ? 'agents' : out / q.decode_n.v >= fresh / P.v ? 'decode' : 'prefill';
        return { nc, p, D1: q.decode_1.v, Dn: q.decode_n.v, binding, server_busy: Math.min(1, cf.turns_per_s * cf.server_s), closed: cf };
      });
      const bad = sims.find(x => x.p.status !== 'MODEL');
      if (bad) return { n, status: 'UNMODELLED', why: bad.p.why };
      const turns = sims.reduce((x, y) => x + y.p.turns, 0), outT = sims.reduce((x, y) => x + y.p.out_tokens, 0), win = v(I.window_s);
      return { n, status: 'MODEL', per_instance: per, turns_per_agent_min: turns / (n * win / 60), useful_tps_total: outT / win, useful_tps_per_agent: outT / win / n,
        decode_per_request: sims[0].D1, decode_card_at_slots: sims.map(x => x.Dn), prefill_tps: P.v, out_per_turn: turns ? outT / turns : null,
        fresh_per_output: sims[0].p.fresh_per_output, share: sims[0].p.share, binding: sims.map(x => x.binding), server_busy: sims.map(x => x.server_busy), bound: sims[0].p.bound, inputs: sims[0].p.inputs };
    });
    return { terms: { bw_eff_gbs: T.bw_eff_gbs, layer_ms: T.layer_ms, tpass_ms: T.tpass_ms }, prefill: P, rows };
  }
  // a many-agent recipe (agents >= many_agents_min) must rest on a real-agent row of its own build: without one its numbers are
  // decode capacity from a synthetic short-prompt load, flagged (UNKNOWN; --strict holds it). With one: the binding term at its
  // agent count from the newest such row's P and the profile's F (D from the same build's curve when one fits)
  function agentWorkloadCheck(data, r) {
    const W = data.workload, sv = servingOf(data, r);
    if (!W || !sv) return null;
    const wl = (r.conditions || {}).workload || {}, n = firstInt(wl.agents) || sv.s.batch;
    if (!(n >= v(W.settings.many_agents_min))) return null;
    const out = (status, detail, pred) => ({ id: 'agent_workload', label: data.checks.agent_workload.label, status, detail, pred: pred || null });
    const ev = agentEvidence(data, { box: r.box, model: r.model, engine: r.engine, build: r.engine_build }), pr = agentProfile(data, {}), F = v(pr.fresh);
    const last = agentReference(ev), cfg = data.curves.configs.filter(c => c.box === r.box && c.model === r.model && c.engine === r.engine && buildKey(data, c.build) === ev.key).map(c => ({ c, f: curveFit(data, c) })).filter(x => x.f.ok).slice(-1)[0];
    // P: the build's real-agent row (MEASURED), else predicted from the box and recipe (prefillPass, MODEL)
    const P = last ? last.prefill_tps : prefillPredicted(data, { box: r.box, model: r.model, engine: r.engine, flags: (r.conditions || {}).engine_flags });
    const st = W.engines[r.engine] && W.engines[r.engine].reads_stall_decode, D = cfg ? cfg.f.at(n) : null;
    const t = agentThroughput({ decode: D, prefill: P ? v(P) : null, fresh: F, stall: st ? !!v(st) : false });
    const seen = ev.same.filter(x => x.n === n).map(x => x.n + ' agents ' + v(x.agent_tps) + ' tok/s (' + x.run + ')');
    const t1 = x => x === null || x === undefined ? '?' : x === Infinity ? 'inf' : x.toFixed(1);
    const pred = Object.assign({ n, fresh: F, fresh_tag: pr.fresh.tag, profile: pr.name, prefill_tps: P ? v(P) : null, prefill_tag: P ? P.tag : null }, t);
    const body = (t.binding ? t.binding.toUpperCase() + '-BOUND at ' + n + ' agents: ' : 'binding term UNKNOWN at ' + n + ' agents: ') + 'prefill ' + (last ? v(P) + ' tok/s (' + last.run + ')' : P ? t1(v(P)) + ' tok/s (MODEL, predicted: ' + P.src + ')' : '?') + ' / ' + F + ' fresh per output = ' + t1(t.prefill_cap) + ' tok/s; decode capacity ' + (D !== null ? t1(D) + ' tok/s (FIT, ' + cfg.c.id + ')' : 'UNKNOWN (no fitted curve of this build)') +
      '; ceiling ' + t1(t.ceiling) + (t.serial ? ', reads stall decode so ' + t1(t.serial) + ' expected' : '');
    if (!ev.same.length) return out('UNKNOWN', 'FLAG: ' + n + ' agents judged only on synthetic short-prompt decode: that is decode capacity, not agent throughput. No real-agent row (measured.workload_kind ' + v(W.settings.real_kind) + ') for build ' + (ev.key || '(none)') + ' on ' + r.box + ' ' + r.model + ' ' + r.engine +
      (ev.other.length ? '; other builds have ' + ev.other.length + ' (' + [...new Set(ev.other.map(x => x.build_key))].join(', ') + '), not borrowed' : '') + '. Replay recorded agent traffic on this build; the ' + pr.name + ' profile reads ' + F + ' fresh prompt tokens per output token (' + pr.fresh.tag + ')' +
      (P ? '. Before launch: ' + body : ''), P ? pred : null);
    return out('PASS', body + (seen.length ? '; MEASURED ' + seen.join(', ') : ''), pred);
  }
  // where a stage computes experts on the CPU pool (a layer split on an engine whose split stages do: serving.split_cpu_experts,
  // or serving.cpu_experts), every batch window row goes through that pool: ceil(batch / batch_groups) rows must fit its cap
  // (serving.cpu_pool_max_rows). A window-row prefill may fill spare rows up to max_t, so max_t above the cap is UNKNOWN
  function cpuPoolCheck(data, r) {
    const sv = servingOf(data, r);
    if (!sv || !('cpu_pool_max_rows' in sv.S.defaults)) return null;
    const s = sv.s, S = sv.S, re = new RegExp(v(S.layer_split_re));
    const split = ((r.gpu_split || {}).banks_gib || []).length > 1 || re.test(String((r.conditions || {}).engine_flags || ''));
    const cpu = s.cpu_experts !== null && s.cpu_experts !== undefined ? !!s.cpu_experts : split && !!(S.split_cpu_experts && v(S.split_cpu_experts));
    if (!cpu || !s.batch) return null;
    const byBuild = S.cpu_pool_rows_by_build ? v(S.cpu_pool_rows_by_build)[buildKey(data, r.engine_build)] : undefined;
    const own = (r.serving || {}).cpu_pool_max_rows !== undefined, cap = own || byBuild === undefined ? s.cpu_pool_max_rows : byBuild;
    const rows = Math.ceil(s.batch / Math.max(1, s.batch_groups)), why = s.cpu_experts !== null && s.cpu_experts !== undefined ? 'serving.cpu_experts' : 'a layer split computes its missed experts on the CPU pool (engines.' + r.engine + '.serving.split_cpu_experts)';
    const out = (status, detail) => ({ id: 'cpu_pool_rows', label: data.checks.cpu_pool_rows.label, status, detail, pred: { rows, cap, max_t: s.max_t } });
    const head = s.batch + ' agents in ' + s.batch_groups + ' group' + (s.batch_groups > 1 ? 's' : '') + ' = ' + rows + ' rows a window; CPU pool cap ' + cap + ' rows (' + (own ? 'serving.cpu_pool_max_rows' : byBuild !== undefined ? 'engines.' + r.engine + '.serving.cpu_pool_rows_by_build for build ' + buildKey(data, r.engine_build) : 'the engine default, serving.cpu_pool_max_rows') + '); ' + why;
    if (rows > cap) return out('FAIL', head + ': the window cannot load (' + S.split_cpu_experts.src + '); fewer agents a group (more batch_groups), or a build with a larger pool cap');
    if (s.max_t > cap) return out('UNKNOWN', head + '; max_t ' + s.max_t + ' lets window-row prefill fill a window past the cap: not measured');
    return out('PASS', head);
  }

  // ---------- grounding: serving priors and the replay checks (data.priors from data/priors-serving.json, data.workload) ----------
  // a many-agent recipe carries its prediction before launch: useful tok/s, the binding term (decode or prefill) and the F it
  // assumed. The model's own (agent_workload) when it names a binding term, else one the recipe declares in
  // conditions.workload.prediction {useful_tps, binding, fresh, src}; neither is FLAGGED (UNKNOWN; --strict holds it)
  function predictionFirstCheck(data, r, aw) {
    if (!aw) return null;
    const p = aw.pred, own = ((r.conditions || {}).workload || {}).prediction, t1 = x => x === Infinity ? 'inf' : x.toFixed(1);
    const out = (status, detail, pred) => ({ id: 'prediction_first', label: data.checks.prediction_first.label, status, detail, pred });
    if (p && p.binding && Number.isFinite(p.expect))
      return out('PASS', p.n + ' agents: useful ' + t1(p.expect) + ' tok/s predicted, ' + p.binding.toUpperCase() + '-BOUND, assuming F = ' + p.fresh + ' fresh prompt tokens per output (profile ' + p.profile + ', ' + p.fresh_tag + ') and prefill ' + t1(p.prefill_tps) + ' tok/s (' + p.prefill_tag + ')',
        { n: p.n, useful_tps: p.expect, binding: p.binding, fresh: p.fresh, fresh_tag: p.fresh_tag, profile: p.profile, prefill_tps: p.prefill_tps, prefill_tag: p.prefill_tag, basis: 'model' });
    if (own && ['decode', 'prefill'].includes(own.binding) && Number.isFinite(own.useful_tps) && Number.isFinite(own.fresh) && own.src)
      return out('PASS', 'declared: useful ' + t1(own.useful_tps) + ' tok/s, ' + own.binding.toUpperCase() + '-BOUND, assuming F = ' + own.fresh + ' (' + own.src + ')',
        { n: p ? p.n : null, useful_tps: own.useful_tps, binding: own.binding, fresh: own.fresh, fresh_tag: 'DECLARED', profile: null, basis: 'declared' });
    const why = !p ? 'the model has neither a real-agent prefill rate of this build nor a prefill pass model of its box and store (agent_workload)' :!p.binding ? 'decode capacity is UNKNOWN (no fitted curve of this build), so neither term can be named' : 'no expected useful tok/s';
    return out('UNKNOWN', 'FLAG: no predicted binding term: ' + why + '. Before launch a many-agent recipe names its predicted useful tok/s, its binding term (decode or prefill) and the F it assumed: replay agent traffic on this build and fit its curve, or declare conditions.workload.prediction {useful_tps, binding, fresh, src}', null);
  }
  // the recipe facts a serving prior's applies_when reads (data.priors.predicates); every key of a prior must hold
  function priorFacts(data, r) {
    const S0 = (data.workload || {}).settings || {}, sv = servingOf(data, r), wl = (r.conditions || {}).workload || {};
    const agents = firstInt(wl.agents) || (sv ? sv.s.batch : null) || 1, re = sv && sv.S.layer_split_re ? new RegExp(v(sv.S.layer_split_re)) : null;
    const m = data.models[r.model] || {}, E = m.E !== undefined ? Number(v(m.E)) : 0;
    return { agents, agent_traffic: agents >= (S0.many_agents_min ? v(S0.many_agents_min) : Infinity) || (!!S0.real_kind && wl.kind === v(S0.real_kind)), moe: E > 1,
      layer_split: ((r.gpu_split || {}).banks_gib || []).length > 1 || (re !== null && re.test(String((r.conditions || {}).engine_flags || ''))), draft: !!r.draft, ctx: firstInt(r.ctx) };
  }
  const PRIOR_TESTS = { agent_traffic: (f, x) => f.agent_traffic === x, agents_min: (f, x) => f.agents >= x, moe: (f, x) => f.moe === x,
    layer_split: (f, x) => f.layer_split === x, draft: (f, x) => f.draft === x, ctx_below: (f, x) => f.ctx !== null && f.ctx < x };
  function priorsFor(data, r) {
    if (!data.priors) return [];
    const f = priorFacts(data, r);
    return data.priors.rows.filter(p => Object.entries(p.applies_when).every(([k, x]) => {
      if (!PRIOR_TESTS[k]) throw new Error('prior ' + p.id + ': applies_when.' + k + ' has no test in model.js PRIOR_TESTS (' + Object.keys(PRIOR_TESTS).join(', ') + ')');
      return PRIOR_TESTS[k](f, x);
    }));
  }
  // a prior's value as one number (a [low, high] range counts as its high end; named numbers have none)
  const priorValue = p => Array.isArray(p.value) ? Math.max(...p.value) : typeof p.value === 'number' ? p.value : null;
  const fmtPrior = p => (Array.isArray(p.value) ? p.value.join('-') : p.value !== null && typeof p.value === 'object' ? Object.entries(p.value).map(([k, x]) => k + ' ' + x).join(', ') : String(p.value)) + ' ' + p.unit;
  // the fresh-per-output reference of a real-agent row (data.workload.measured): the trace-ideal ratio of the trace it replayed
  // (its profile's trace_ideal_fresh_per_output), else the profile's fresh_reference_prior (data.priors)
  function freshReference(data, x) {
    const W = data.workload, pr = W.profiles[x.profile || v(W.settings.default_profile)] || {};
    const ti = x.trace && pr.trace_ideal_fresh_per_output ? pr.trace_ideal_fresh_per_output[x.trace] : null;
    if (ti) return { basis: 'trace-ideal', name: x.trace, v: v(ti), tag: ti.tag, src: ti.src };
    const id = pr.fresh_reference_prior ? v(pr.fresh_reference_prior) : null, p = id && data.priors ? data.priors.rows.find(y => y.id === id) : null;
    return p ? { basis: 'prior', name: id, v: priorValue(p), tag: p.tag, src: p.src } : null;
  }
  // the numeric parts of a replay's fresh breakdown with their shares ([] when the row has none)
  const breakdownParts = b => { const xs = b && typeof b === 'object' ? Object.entries(b).filter(([, n]) => typeof n === 'number' && n >= 0) : [], tot = xs.reduce((a, [, n]) => a + n, 0);
    return xs.map(([k, n]) => ({ k, n, share: tot > 0 ? n / tot : null })); };
  const fmtBreakdown = ps => ps.map(p => p.k + ' ' + p.n + (p.share === null ? '' : ' (' + (100 * p.share).toFixed(0) + ' %)')).join(', ');
  // a real-agent row whose F exceeds K (settings.replay_artifact_k) x its reference read what a working cache keeps: FLAG it a
  // suspect replay artifact with the excess, the fresh tokens above the reference and the suspected causes
  function replayArtifactCheck(data, x) {
    const W = data.workload, K = v(W.settings.replay_artifact_k), ref = freshReference(data, x), F = x.fresh_per_output ? v(x.fresh_per_output) : null;
    const out = (status, detail, o) => Object.assign({ id: 'replay_artifact', label: data.checks.replay_artifact.label, status, detail, run: x.run, n: x.n, k: K, ref, fresh: F, excess: null, surplus_tokens: null, causes: [] }, o || {});
    if (F === null || !ref || !(ref.v > 0)) return out('UNKNOWN', F === null ? 'the row has no fresh_per_output' : 'no reference: trace ' + (x.trace || '(not named)') + ' has no trace-ideal ratio and the profile names no reference prior');
    const excess = F / ref.v, surplus = x.out_tokens ? Math.round((F - ref.v) * x.out_tokens) : null, bd = breakdownParts(x.fresh_breakdown);
    const head = 'F ' + F + ' fresh prompt tokens per output = ' + excess.toFixed(1) + 'x the ' + (ref.basis === 'trace-ideal' ? 'trace-ideal ratio of trace ' : 'reference prior ') + ref.name + ' (' + ref.v + ', ' + ref.tag + ')';
    if (excess <= K) return out('PASS', head + ', within K = ' + K, { excess, surplus_tokens: surplus });
    const causes = v(W.settings.replay_artifact_causes);
    return out('FLAG', head + ' > K = ' + K + ': suspect replay artifact (cold restarts / evictions / template drift)' + (surplus !== null ? '; ' + surplus + ' fresh tokens above the reference in the window' : '') +
      (x.cache_evictions ? '; ' + x.cache_evictions + ' cache evictions in the window' : '') + (bd.length ? '; fresh breakdown ' + fmtBreakdown(bd) : '; no fresh breakdown to split the causes'), { excess, surplus_tokens: surplus, causes });
  }
  // a real-agent row states its warm-up: seconds before the timed window (PASS), settings.warmup_none or 0 (NOTE: a cold-start
  // window, so its F is an upper bound), nothing (FLAG); the fresh breakdown is shown when the replay gave one
  function steadyStateCheck(data, x) {
    const none = v(data.workload.settings.warmup_none), w = x.warmup_s, bd = breakdownParts(x.fresh_breakdown);
    const out = (status, detail) => ({ id: 'steady_state', label: data.checks.steady_state.label, status, detail, run: x.run, n: x.n, warmup_s: w === undefined ? null : w, breakdown: bd });
    const cold = 'cold-start window, F upper bound (every agent\'s first prompt is read cold inside the timed window)', brk = bd.length ? '; fresh breakdown ' + fmtBreakdown(bd) : '; no fresh breakdown (cold / growth / re-read)';
    if (w === null || w === undefined) return out('FLAG', 'warmup_s not stated: ' + cold + brk);
    if (w === none || w === 0) return out('NOTE', 'warmup_s ' + w + ': ' + cold + brk);
    if (typeof w !== 'number' || !(w > 0)) return out('FLAG', 'warmup_s ' + JSON.stringify(w) + ' is neither seconds nor ' + none + brk);
    return out('PASS', w + ' s of warm-up before the timed window' + brk);
  }
  function replayChecks(data) { return ((data.workload || {}).measured || []).map(x => ({ run: x.run, n: x.n, label: x.label, checks: [replayArtifactCheck(data, x), steadyStateCheck(data, x)] })); }

  // ---------- roofline: per-phase bounds from compute ceilings, links and bytes (data.roofline, boxes[].gpus[].ceil) ----------
  // A phase's time is built from named terms: compute (FLOPs / (ceiling x eta) per precision class, the precision the engine's
  // kernels run at), weight streaming (bytes a card does not hold / link rate), resident weight reads (bytes / VRAM rate),
  // attention (FLOPs that grow with depth) and KV bytes. The engine's overlap rule combines them: 'max' (per chunk, the
  // stream hides under the kernels or the kernels under the stream) or 'sum' (in series). The BOUND is the time at eta 1 (the
  // hardware ceiling); the PREDICTION uses the engine's fitted eta (data.roofline.fits). Every phase names the term that binds.
  const RL_TERA = 1e12;
  // the firmware unlock state a query judges: the query's own, else stock before the setting's since date, else the setting
  function unlockState(g, q) {
    const u = g && g.firmware_unlock;
    if (q && q.unlock) return q.unlock;
    if (!u) return 'stock';
    if (q && q.at && u.since) {
      const qa = Date.parse(String(q.at)), us = Date.parse(String(u.since));
      if (Number.isFinite(qa) && Number.isFinite(us) ? qa < us : String(q.at) < String(u.since)) return 'stock';
    }
    return v(u) || 'stock';
  }
  // one ceiling of a card or CPU (ceil = {class, values, unlock}): the unlock state's value, else the stock value, else the
  // first key of data.roofline.precision_fallback the card has. { key, ops (per second), tag, src, state, fallback } or null
  function ceilingOf(data, ceil, prec, state) {
    if (!ceil || !prec) return null;
    const fb = v((data.roofline || {}).precision_fallback) || {}, keys = [prec].concat(fb[prec] || []);
    const at = k => {
      const byState = state && state !== 'stock' && ceil.unlock && ceil.unlock.by_state && ceil.unlock.by_state[state];
      const x = (byState && byState[k]) || ceil.values[k];
      return x && v(x) !== null && v(x) !== undefined ? { key: k, ops: v(x) * RL_TERA, value: v(x), unit: x.unit, tag: x.tag, src: x.src, evidence: x.evidence, state: byState && byState[k] ? state : 'stock', fallback: k !== prec } : null;
    };
    const own = at(prec);
    if (own) return own;
    // a card without the asked precision runs the FASTEST path it has in the fallback chain (llama.cpp takes FP32 on Pascal, whose
    // FP16 is 1/64 rate: fast_fp16_available), never the first listed when a later one is faster; ties keep the listed order
    return keys.slice(1).map(at).filter(Boolean).reduce((a, c) => (!a || c.ops > a.ops ? c : a), null);
  }
  // ---------- shape curves: a short GEMM does not reach the square ceiling (data/roofline.json cards[].shape_curves) ----------
  // 512 experts at top-10 leave ~24 rows an expert on a 1,202-token read. shapeRate reads the card's measured kernel rate for a
  // GEMM class (experts | trunk), kernel path and precision at the given rows: a power law between neighbouring measured points
  // (log rows, log rate), MEASURED within 5 % of a point on the measured dims (and quant), ESTIMATE between, beyond, on another
  // model's dims or another quant. null when the card has no curve for that class, path and precision
  function shapeRate(data, g, cls, path, prec, rows, ctx) {
    const C = g && g.ceil && g.ceil.shape_curves && g.ceil.shape_curves[cls], c = C && C[path] && C[path][prec];
    if (!c || !c.points || !c.points.length || !(rows > 0)) return null;
    const pts = c.points.slice().sort((a, b) => a.rows - b.rows), lr = Math.log(rows), x = ctx || {};
    const near = pts.find(p => Math.abs(Math.log(rows / p.rows)) <= 0.05);
    let val, where;
    if (near) { val = v(near); where = 'at ' + near.rows; }
    else if (pts.length === 1) { val = v(pts[0]); where = 'held flat from ' + pts[0].rows; }
    else {
      const i = pts.findIndex(p => p.rows > rows), a = i === 0 ? pts[0] : i < 0 ? pts[pts.length - 2] : pts[i - 1], b = i === 0 ? pts[1] : i < 0 ? pts[pts.length - 1] : pts[i];
      val = Math.exp(Math.log(v(a)) + (Math.log(v(b)) - Math.log(v(a))) * (lr - Math.log(a.rows)) / (Math.log(b.rows) - Math.log(a.rows)));
      where = i === 0 ? 'below ' + a.rows : i < 0 ? 'above ' + b.rows : 'between ' + a.rows + ' and ' + b.rows;
    }
    const norm = s => String(s || '').toLowerCase().replace(/[^a-z0-9]/g, '');
    const otherDims = c.model && x.model && c.model !== x.model, otherQuant = c.quant && x.store && !norm(x.store).includes(norm(c.quant));
    const tag = near && !otherDims && !otherQuant ? 'MEASURED' : 'ESTIMATE';
    const why = [where + ' rows', otherDims && 'measured on ' + c.model + ' dims', otherQuant && 'measured on ' + c.quant].filter(Boolean).join('; ');
    return { ops: val * RL_TERA, value: val, unit: c.unit, bits: c.bits_per_weight, tag, where: why, rows, path, prec, cls,
      src: (tag === 'MEASURED' ? near.src : 'ESTIMATE: ' + cls + ' ' + path + ' ' + prec + ' ' + why + ' (points ' + pts.map(p => p.rows + ' rows ' + v(p)).join(', ') + ' ' + c.unit + ')') };
  }
  // seconds (at eta 1) of a GEMM class on one card: flops at the square ceiling sq, or, with a shape curve at its rows, the
  // measured kernel time (compose kernel) or that time less the read of its own weights at the bench layout (compose excess,
  // the default, settings.shape_compose: the engine's own terms already price the weights at the store's bytes), never faster
  // than the square ceiling. weights = the weights the GEMM reads (count); ctx = { model, store, compose }
  function shapeTime(data, g, sq, cls, path, prec, flops, rows, weights, ctx) {
    const tSq = flops > 0 ? flops / sq.ops : 0, r = flops > 0 && path ? shapeRate(data, g, cls, path, prec, rows, ctx) : null;
    if (!r) return { s: tSq, rows, shape: null, cls, path, prec, square: sq.value };
    const mode = (ctx && ctx.compose) || v(((data.roofline || {}).settings || {}).shape_compose) || 'excess';
    const tK = flops / Math.min(r.ops, sq.ops), vbw = v(g.ceil.values.vram_gbs);
    const tW = mode === 'excess' && r.bits && weights > 0 && vbw ? weights * r.bits / 8 / (vbw * 1e9) : 0;
    const s = Math.max(tSq, tK - tW);
    return { s, rows, shape: r, mode, kernel_s: tK, layout_read_s: tW, eff_ops: flops / s, cls, path, prec, square: sq.value };
  }
  // a one-line account of a priced GEMM class: rows, the curve value read and its tag, or the square ceiling flagged unmeasured
  const shapeLine = x => x.cls + ' ' + x.rows.toFixed(1) + ' rows' + (x.cls === 'experts' ? ' an expert' : '') + ': ' + (x.shape
    ? x.shape.value.toFixed(1) + ' ' + x.shape.unit + ' ' + x.shape.tag + ' (' + x.path + ' ' + x.prec + ', ' + x.shape.where + ')' + (x.mode === 'excess' ? ', less its ' + x.shape.bits + '-bit weight read = ' + (x.eff_ops / RL_TERA).toFixed(1) + ' effective' : '')
    : 'no shape curve for ' + x.path + ' ' + x.prec + ' (unmeasured): square ceiling ' + x.square + ', ESTIMATE');
  // the heat shape's exponent (src/heat_shape.py): the share of routed reads served by the hottest fraction x of experts = x^a
  const RL_SHAPE = [[2000 / 12096, 0.391], [3000 / 12096, 0.501], [4000 / 12096, 0.597]];
  const RL_SHAPE_A = RL_SHAPE.reduce((a, p) => a + Math.log(p[1]) / Math.log(p[0]), 0) / RL_SHAPE.length;
  // a model's compute shape and a store's bytes: roofline.models[m] first, else derived from data.models[m] (tagged ESTIMATE)
  function rlShape(data, model, store) {
    const m = data.models[model], R = ((data.roofline || {}).models || {})[model] || {};
    if (!m) return { why: 'model ' + model + ' unknown' };
    const L = Number(v(m.L)), E = Number(v(m.E)), k = Number(v(m.k)), P = Number(v(m.P));
    const S = (R.stores || {})[store] || {};
    const dense = m.dense_gib && v(m.dense_gib) && m.dense_bpw && v(m.dense_bpw) ? v(m.dense_gib) * GiB * 8 / v(m.dense_bpw) : null;
    const est = (x, src) => ({ v: x, tag: 'ESTIMATE', src: 'ESTIMATE: ' + src, evidence: 'inferred' });
    const trunk = R.trunk_params || (dense !== null ? est(dense, 'models.' + model + ' dense_gib x 8 / dense_bpw (embedding rows counted as compute: an upper bound)') : null);
    const exB = S.expert_bytes || (m.experts_gib && v(m.experts_gib) ? est(v(m.experts_gib) * GiB, 'models.' + model + ' experts_gib') : null);
    const dRead = S.dense_read_bytes || (m.dense_gib && v(m.dense_gib) ? est(v(m.dense_gib) * GiB, 'models.' + model + ' dense_gib (every non-expert byte read a token)') : null);
    const miss = [!(L && E && k && P) && 'L, E, k or P', !trunk && 'trunk params', !exB && 'expert store bytes'].filter(Boolean);
    if (miss.length) return { why: 'model ' + model + ': no ' + miss.join(', ') };
    const routed = { v: k * P * L, tag: 'READ', src: 'models.' + model + ' k ' + k + ' x P ' + P + ' x L ' + L + ' (routed params a token)', evidence: 'inferred' };
    return { L, E, k, P, routed, trunk, head: R.head_params || { v: 0, tag: 'ESTIMATE', src: 'ESTIMATE: no LM-head row for ' + model + ' (counted in trunk)', evidence: 'inferred' },
      attn_layers: R.attn_layers || null, q_dim: R.q_dim || null, keys_cap: R.attn_keys_cap || null, expert_bytes: exB, dense_read: dRead,
      draft_mult: R.draft_mult || (m.draft && m.draft.mult_default) || null, kv_per_token: R.kv_bytes_per_token || (m.kv_bytes_per_token && v(m.kv_bytes_per_token) ? m.kv_bytes_per_token : null), state_bytes: rlStateBytes(R) };
  }
  // a stream's persistent recurrent state: derived from the engine's state geometry when the model states one (recurrence
  // layers x value heads x key dim x value dim + convolution layers x channels x tail, x bytes an element), else the stated value
  function rlStateBytes(R) {
    const G = R.state_geometry;
    if (!G) return R.state_bytes_per_stream || null;
    const g = v(G), rec = g.gdn_layers * g.value_heads * g.key_dim * g.value_dim * g.bytes, conv = g.gdn_layers * (g.conv_channels || 0) * (g.conv_tail || 0) * g.bytes;
    return { v: rec + conv, tag: G.tag, evidence: G.evidence, recurrence: rec, convolution: conv,
      src: G.tag + ': recurrence ' + g.gdn_layers + ' x ' + g.value_heads + ' x ' + g.key_dim + ' x ' + g.value_dim + ' x ' + g.bytes + ' = ' + rec + ' B + convolution ' + g.gdn_layers + ' x ' + g.conv_channels + ' x ' + g.conv_tail + ' x ' + g.bytes + ' = ' + conv + ' B (' + G.src + ')' };
  }
  // the stages of a query: card + layer share + resident / not-resident expert bytes. An engine with pass 'prefill_pass'
  // (strata) takes the stages of passTerms (layer split, CPU share, cache bytes); otherwise one stage on q.gpu (index, default
  // 0) holding q.resident_bytes (default: none when the engine streams every expert, else all of them)
  function rlStages(data, q, eng, shape) {
    const box = data.boxes[q.box];
    if (eng.prefill.pass === 'prefill_pass') {
      const b = passTerms(data, { box: q.box, model: q.model, engine: q.engine, flags: q.flags || '' });
      if (b.status !== 'OK') return { why: b.why };
      return { chunk: b.chunk, cpu_share: b.cpu_share, stages: b.stages.map((s, i) => ({ gi: i, g: box.gpus[i], frac: s.layers / b.L, resident: s.resident_gb * 1e9, host: s.nonresident_gb * 1e9 })) };
    }
    const gi = q.gpu || 0, g = box.gpus[gi], all = q.store_bytes || v(shape.expert_bytes);
    const res = q.resident_bytes !== undefined ? q.resident_bytes : eng.prefill.stream === 'all' ? 0 : all;
    return { chunk: null, cpu_share: 0, stages: [{ gi, g, frac: 1, resident: Math.min(res, all), host: all - Math.min(res, all), host_tiers: q.host_tiers || null }] };
  }
  function rlLink(data, g, eng) {
    const geo = g && g.pcie ? v(g.pcie) : null, R = data.roofline, lane = v(R.link.pcie_gbs_per_lane);
    const eff = eng.prefill.link_eff !== null && eng.prefill.link_eff !== undefined ? eng.prefill.link_eff : v(R.link.dma_eff);
    if (!geo || !lane[geo.max_gen || geo.gen]) return null;
    const gen = geo.max_gen || geo.gen, width = geo.max_width || geo.width;
    return { gbs: lane[gen] * width * eff, src: 'PCIe gen' + gen + ' x' + width + ' (' + (g.pcie.src || 'box probe') + ') x ' + eff + (eng.prefill.link_eff ? ' (engine link share)' : ' (roofline.link.dma_eff ESTIMATE)') };
  }
  // where a query's GEMM classes run (data.roofline.settings.placements): { name, trunk, attention, experts } or { why } for a
  // placement the table does not hold
  function rlPlacement(data, q) {
    const name = q.placement || 'gpu', P = v(((data.roofline || {}).settings || {}).placements) || { gpu: { trunk: 'gpu', attention: 'gpu', experts: 'gpu' } };
    return P[name] ? Object.assign({ name }, P[name]) : { why: 'placement ' + name + ' is not in data.roofline.settings.placements (' + Object.keys(P).join(', ') + ')' };
  }
  // the CPU side of a CPU placement: the box CPU ceiling for each class the engine runs there, scaled by the share of physical
  // cores the query's threads use (SMT siblings add no FMA ports), and the engine's cpu path for the store's expert quants.
  // { why } refuses: a CPU placement is never bounded on a GPU, and never without a CPU ceiling
  function rlCpu(data, q, box, eng, shape, pl) {
    const P = eng.prefill, cpu = box.cpu || {}, need = { trunk: P.cpu_trunk, attention: P.cpu_attention, experts: P.cpu_experts };
    const on = Object.keys(need).filter(k => pl[k] === 'cpu');
    const noMeta = on.filter(k => !need[k]);
    if (noMeta.length) return { why: 'refused: placement ' + pl.name + ' puts ' + noMeta.join(', ') + ' on the CPU but engine ' + q.engine + ' names no CPU kernel precision for it (data.roofline.engines.' + q.engine + '.prefill.cpu_' + noMeta[0] + ')' };
    if (!cpu.ceil) return { why: 'refused: placement ' + pl.name + ' puts ' + on.join(', ') + ' on the CPU but ' + q.box + ' has no CPU compute ceilings (data/roofline.json cpus has no entry matching its CPU)' };
    const ceil = {};
    for (const k of on) { const c = ceilingOf(data, cpu.ceil, need[k], 'stock'); if (!c) return { why: 'refused: placement ' + pl.name + ' puts ' + k + ' on the CPU but ' + q.box + ' (' + cpu.ceil.class + ') has no ' + need[k] + ' ceiling' }; ceil[k] = c; }
    const phys = Number(v(cpu.phys)) || null, thr = Number(q.threads) || null;
    const share = phys && thr ? Math.min(1, thr / phys) : 1;
    const quants = [].concat(q.expert_quant || ((((data.roofline.models || {})[q.model] || {}).stores || {})[q.store] || {}).expert_quant || []);
    const paths = [...new Set(quants.map(t => (P.cpu_paths || {})[t] || 'unknown (' + t + ')'))];
    return { ceil, share, phys, threads: thr, paths, quants, dram_gbs: box.dram && box.dram.bw ? v(box.dram.bw) : null,
      share_src: phys && thr ? thr + ' threads on ' + phys + ' physical cores' : 'threads or physical cores not stated: every core' };
  }
  // the inputs every phase needs: { box, g, eng, shape, st, state, why }
  function rlInputs(data, q) {
    const R = data.roofline, box = data.boxes[q.box], eng = R && R.engines[q.engine];
    const why = !R ? 'no data.roofline' : !box ? 'box ' + q.box + ' unknown' : !eng ? 'engine ' + q.engine + ' has no roofline kernel metadata (data.roofline.engines)' : null;
    if (why) return { why };
    const shape = rlShape(data, q.model, q.store);
    if (shape.why) return { why: shape.why };
    const pl = rlPlacement(data, q);
    if (pl.why) return { why: 'refused: ' + pl.why };
    if (pl.name !== 'gpu') {
      const cpu = rlCpu(data, q, box, eng, shape, pl);
      if (cpu.why) return { why: cpu.why };
      if (!cpu.dram_gbs) return { why: 'refused: placement ' + pl.name + ' reads every weight from DRAM but ' + q.box + ' has no measured DRAM read rate (boxes.' + q.box + '.dram.bw)' };
      return { box, eng, shape, pl, cpu, st: null };
    }
    const st = rlStages(data, q, eng, shape);
    if (st.why) return { why: st.why };
    const noCeil = st.stages.find(s => !s.g || !s.g.ceil);
    if (noCeil) return { why: 'stage card ' + (noCeil.g ? noCeil.g.id : '#' + noCeil.gi) + ' on ' + q.box + ' has no compute ceilings (data/roofline.json cards)' };
    return { box, eng, shape, st, pl };
  }
  // the engine's etas: { gpu, cpu } (null where no fit cell exists; a phase then runs that side at its ceiling, eta 1). eta may be
  // given as a number (every compute term, the old single factor) or { gpu, cpu }; a missing cpu eta follows the gpu one
  function rlEtaPair(eta) {
    if (eta && typeof eta === 'object') { const g = eta.gpu || 1; return { gpu: g, cpu: eta.cpu || g }; }
    return { gpu: eta || 1, cpu: eta || 1 };
  }
  function rlEta(data, engine) {
    const f = rooflineFit(data, engine), c = rooflineFit(data, engine, 'cpu');
    return { gpu: f.status === 'FIT' ? f.eta : null, cpu: c.status === 'FIT' ? c.eta : null, fit: f, cpu_fit: c, any: f.status === 'FIT' || c.status === 'FIT' };
  }
  // the eta a query's prediction runs at: a CPU placement takes the engine's CPU fit (GPU side at its fit or 1); a GPU placement
  // keeps the one GPU factor (its CPU expert term too), never the CPU-only fit: that fit is 3.6x slower than the CPU experts of
  // the GPU-dense held-out cell (droopy-v41-gpu-dense-cpu-experts), so it does not transfer to a hybrid placement
  // the composition of a prefill time: groups (a chunk on one stage) of compute and stream seconds per term; rule 'max' = each
  // group takes the larger of its compute and stream sums (overlap), 'sum' = every second adds. A prediction record stores it,
  // so a run is attributed against the terms as predicted, never against the data as it reads today (rlComposedTime)
  function rlGroup(compute, stream) {
    const keep = o => Object.fromEntries(Object.entries(o).filter(([, s]) => s > 0));
    return { compute: keep(compute), stream: keep(stream) };
  }
  function rlComposedTime(comp, id, k) {
    const sum = o => Object.entries(o).reduce((a, [t, s]) => a + (t === id ? s * k : s), 0);
    return comp.groups.reduce((a, g) => { const c = sum(g.compute), s = sum(g.stream); return a + (comp.rule === 'max' ? Math.max(c, s) : c + s); }, 0);
  }
  function rlEtaFor(data, q) {
    const et = rlEta(data, q.engine);
    return (q.placement || 'gpu') !== 'gpu' ? { gpu: et.gpu || 1, cpu: et.cpu || 1 } : (et.gpu || 1);
  }
  // prefill of a CPU placement (cpu_only): per chunk of n tokens, the trunk, attention and expert GEMMs at the box CPU ceilings x
  // the core share x the CPU eta, and every weight the chunk touches read once from DRAM (the routed experts by the distinct
  // share of the chunk); the engine's overlap rule combines them. Unmodelled terms are named: the cpu path the eta was not fitted
  // on, the indexer scan, the barrier waits and the PMem share of the weights' pages (all inside the fitted eta)
  function prefillRoofCpu(data, q, I, et) {
    const { eng, shape, cpu } = I, P = eng.prefill, T = Number(q.tokens), C = Number(q.chunk) || P.chunk_default, depth = Number(q.depth) || 0, sc = id => ((q.scale || {})[id] || 1);
    const sizes = []; for (let d = 0; d < T; d += C) sizes.push(Math.min(C, T - d));
    const ec = et.cpu, rate = k => cpu.ceil[k].ops * cpu.share * ec, used = {};
    Object.keys(cpu.ceil).forEach(k => { used['cpu ' + k + ' ' + cpu.ceil[k].key] = Object.assign({}, cpu.ceil[k], { core_share: cpu.share, core_share_src: cpu.share_src }); });
    const unmodelled = [];
    if (!shape.attn_layers || !shape.q_dim) unmodelled.push('attention (no attn_layers / q_dim for ' + q.model + ')');
    const fitPath = ((data.roofline.cpu_fits || {})[q.engine] || {}).path || null;
    if (!cpu.paths.length) unmodelled.push('the cpu kernel path (no expert quant stated for store ' + (q.store || 'none') + ')');
    else if (fitPath && (cpu.paths.length > 1 || cpu.paths[0] !== fitPath)) unmodelled.push('cpu path ' + cpu.paths.join(' + ') + ' where the CPU eta is fitted on ' + fitPath + ': the per-ubatch weight convert of the other path is not a term');
    if (v(shape.keys_cap)) unmodelled.push('the key selection scan over every earlier key (O(N) a query; attention counts only the ' + v(shape.keys_cap) + ' keys it selects)');
    unmodelled.push('thread-sync waits between graph nodes and the NUMA / PMem placement of the weight pages (no derived term: inside the CPU eta)');
    const exB = v(shape.expert_bytes), dB = shape.dense_read ? v(shape.dense_read) : 0, dram = cpu.dram_gbs * 1e9;
    const part = (n, p0) => {
      const ct = 2 * v(shape.trunk) * n / rate('trunk'), ce = 2 * v(shape.routed) * n / rate('experts');
      let ca = 0;
      if (shape.attn_layers && shape.q_dim) { const ctx = depth + p0 + n / 2, keys = v(shape.keys_cap) ? Math.min(ctx, v(shape.keys_cap)) : ctx; ca = v(shape.attn_layers) * 4 * v(shape.q_dim) * keys * n / rate('attention'); }
      const touched = n > 0 ? batchDistinct(shape.E, shape.k, n) / shape.E : 0, rd = (touched * exB + dB) / dram;
      return { ct: ct * sc('compute.cpu_trunk'), ca: ca * sc('compute.cpu_attention'), ce: ce * sc('compute.cpu_experts'), rd: rd * sc('read.cpu_weights') };
    };
    let time = 0, p0 = 0, cb = 0, sb = 0;
    const acc = { ct: 0, ca: 0, ce: 0, rd: 0 }, detail = [], groups = [];
    sizes.forEach(n => {
      const x = part(n, p0), c = x.ct + x.ca + x.ce, t = P.overlap === 'max' ? Math.max(c, x.rd) : c + x.rd;
      Object.keys(acc).forEach(k => { acc[k] += x[k]; });
      groups.push(rlGroup({ 'compute.cpu_trunk': x.ct, 'compute.cpu_attention': x.ca, 'compute.cpu_experts': x.ce }, { 'read.cpu_weights': x.rd }));
      detail.push({ tokens: n, s: t, binds: c >= x.rd ? 'compute' : 'stream' }); time += t; cb += c; sb += x.rd; p0 += n;
    });
    const terms = [['compute.cpu_trunk', 'CPU ' + P.cpu_trunk, acc.ct], ['compute.cpu_attention', 'CPU ' + P.cpu_attention, acc.ca], ['compute.cpu_experts', 'CPU ' + P.cpu_experts, acc.ce], ['read.cpu_weights', 'DRAM', acc.rd]]
      .filter(t => t[2] > 0).map(([id, resource, s]) => ({ id, resource, s, share: time ? s / time : null }));
    const side = cb >= sb ? 'compute' : 'stream', top = terms.filter(t => side === 'compute' ? /^compute/.test(t.id) : !/^compute/.test(t.id)).sort((a, b) => b.s - a.s)[0];
    return { status: 'OK', phase: 'prefill', placement: I.pl.name, tokens: T, depth, chunk: C, chunks: sizes.length, overlap: P.overlap, eta: ec, eta_cpu: ec, time_s: time, tps: T / time,
      binding: { side, term: top ? top.id : null, resource: top ? top.resource : null, compute_bound_s: cb, stream_bound_s: sb },
      terms, chunk_detail: detail, composition: { rule: P.overlap === 'max' ? 'max' : 'sum', groups }, unmodelled, ceilings: used, shapes: [], cpu_paths: cpu.paths, link: [], unlock: [], engine_src: eng.prefill.cpu_src || P.src };
  }
  // prefill of q.tokens fresh tokens (q.depth earlier tokens already cached) in chunks of q.chunk; eta 1 = the bound
  function prefillRoof(data, q, eta) {
    const I = rlInputs(data, q);
    if (I.why) return { status: 'UNMODELLED', why: I.why };
    const et = rlEtaPair(eta);
    if (I.pl.name !== 'gpu') return prefillRoofCpu(data, q, I, et);
    const { box, eng, shape, st } = I, P = eng.prefill, T = Number(q.tokens), e = et.gpu, sc = id => ((q.scale || {})[id] || 1);
    const C = Number(q.chunk) || st.chunk || P.chunk_default, depth = Number(q.depth) || 0;
    const sizes = []; for (let d = 0; d < T; d += C) sizes.push(Math.min(C, T - d));
    // a measured read mix (strata's fit rows): the mean read pays the mix's chunks per read
    const nChunks = q.read_mix ? chunksPerRead(data.workload.prefill_pass, C) : sizes.length;
    const terms = {}, add = (id, resource, s, x) => { terms[id] = terms[id] || Object.assign({ id, resource, s: 0 }, x || {}); terms[id].s += s; };
    const unmodelled = [];
    if (!P.attention) unmodelled.push('attention and DeltaNet terms that grow with depth (engine metadata attention null)');
    else if (!shape.attn_layers || !shape.q_dim) unmodelled.push('attention (no attn_layers / q_dim for ' + q.model + ')');
    const used = {};
    const ceilFor = (s, prec) => { const c = ceilingOf(data, s.g.ceil, prec, unlockState(s.g, q)); if (c) used[s.g.id + ' ' + prec] = c; return c; };
    const cpuC = P.cpu_experts ? ceilingOf(data, box.cpu.ceil, P.cpu_experts, 'stock') : null;
    const cpuPhys = Number(v((box.cpu || {}).phys)) || null, cpuShare = cpuPhys && Number(q.threads) ? Math.min(1, Number(q.threads) / cpuPhys) : 1;
    const link = st.stages.map(s => rlLink(data, s.g, eng));
    const bad = st.stages.find((s, i) => !ceilFor(s, P.trunk) || !ceilFor(s, P.experts) || (P.attention && !ceilFor(s, P.attention)) || (s.host > 0 && !link[i] && P.stream !== 'offload-off'));
    if (bad) return { status: 'UNMODELLED', why: 'card ' + bad.g.id + ': no ' + [P.trunk, P.experts, P.attention].filter(Boolean).join(' / ') + ' ceiling or PCIe link' };
    // per chunk of n tokens starting at position p0: compute seconds per class and stream / read seconds, per stage
    const host = st.stages.map(s => s.host * (1 - (st.cpu_share || 0)));
    const hostOnCpu = P.stream === 'offload' && q.host_path === 'cpu';
    if (hostOnCpu && st.stages.some(s => s.host > 0) && !cpuC) return { status: 'UNMODELLED', why: 'refused: host experts computed on the CPU (host_path cpu) but ' + (P.cpu_experts ? q.box + ' has no ' + P.cpu_experts + ' CPU ceiling (data/roofline.json cpus)' : 'engine ' + q.engine + ' names no cpu_experts precision') };
    if (hostOnCpu && cpuC) used['cpu ' + P.cpu_experts] = Object.assign({}, cpuC, { core_share: cpuShare });
    // the GEMMs are priced by their rows (shapeTime): the trunk by the tokens of a chunk, the experts by the rows an expert sees,
    // chunk tokens x k / the distinct experts they touch (batchDistinct); n tokens run in chunks of at most C
    const ctx = { model: q.model, store: q.store, compose: q.shape_compose }, shapes = {};
    const chunkParts = (n, p0) => st.stages.map((s, i) => {
      const per = Math.min(n, C), nCh = per > 0 ? n / per : 0, dist = per > 0 ? batchDistinct(shape.E, shape.k, per) : 0;
      const tr = shapeTime(data, s.g, ceilFor(s, P.trunk), 'trunk', P.trunk_path || 'onednn', P.trunk, s.frac * 2 * v(shape.trunk) * n, per, nCh * s.frac * v(shape.trunk), ctx);
      const ct = tr.s / e;
      const hostShare = s.resident + s.host > 0 ? s.host / (s.resident + s.host) : 0;
      const gpuEx = hostOnCpu ? 1 - hostShare : 1;
      const ex = shapeTime(data, s.g, ceilFor(s, P.experts), 'experts', P.experts_path || 'onednn_grouped', P.experts, s.frac * 2 * v(shape.routed) * n * gpuEx,
        per > 0 ? per * shape.k / dist : 0, nCh * dist * s.frac * shape.L * shape.P * gpuEx, ctx);
      const ce = ex.s / e;
      if (n > 0) [tr, ex].forEach(x => { shapes[s.g.id + ' ' + x.cls] = shapes[s.g.id + ' ' + x.cls] || x; });
      const cc = hostOnCpu && cpuC ? s.frac * 2 * v(shape.routed) * n * hostShare / (cpuC.ops * cpuShare * et.cpu) : 0;
      let ca = 0;
      if (P.attention && shape.attn_layers && shape.q_dim) {
        const ctx = depth + p0 + n / 2, keys = v(shape.keys_cap) ? Math.min(ctx, v(shape.keys_cap)) : ctx;
        ca = s.frac * v(shape.attn_layers) * 4 * v(shape.q_dim) * keys * n / (ceilFor(s, P.attention).ops * e);
      }
      const sHost = hostOnCpu ? 0 : host[i] / 1e9 / link[i].gbs;
      const rHost = hostOnCpu ? s.host / 1e9 / v(box.dram.bw) : 0;
      const vr = s.resident / 1e9 / v(s.g.ceil.values.vram_gbs);
      return { ct: ct * sc('compute.trunk'), ce: ce * sc('compute.experts'), cc: cc * sc('compute.cpu_experts'), ca: ca * sc('compute.attention'), sHost: sHost * sc('stream.experts'), rHost: rHost * sc('read.cpu_experts'), vr: vr * sc('read.resident') };
    });
    let time = 0, p0 = 0, computeBound = 0, streamBound = 0, rule = 'sum';
    const detail = [], groups = [];
    const book = (x, w) => {
      add('compute.trunk', 'tensor ' + P.trunk, x.ct * w); add('compute.experts', 'tensor ' + P.experts, x.ce * w);
      if (x.cc) add('compute.cpu_experts', 'CPU ' + P.cpu_experts, x.cc * w);
      if (x.ca) add('compute.attention', 'tensor ' + P.attention, x.ca * w);
      if (x.sHost) add('stream.experts', 'PCIe', x.sHost * w); if (x.rHost) add('read.cpu_experts', 'DRAM', x.rHost * w);
      if (x.vr) add('read.resident', 'VRAM', x.vr * w);
    };
    if (P.overlap === 'max' && !q.read_mix) {
      sizes.forEach(n => {
        const parts = chunkParts(n, p0);
        let t = 0, cb = 0, sb = 0;
        parts.forEach(x => { const c = x.ct + x.ce + x.cc + x.ca, s = x.sHost + x.rHost + x.vr; t += Math.max(c, s); if (c >= s) cb += c; else sb += s; book(x, 1);
          groups.push(rlGroup({ 'compute.trunk': x.ct, 'compute.experts': x.ce, 'compute.cpu_experts': x.cc, 'compute.attention': x.ca }, { 'stream.experts': x.sHost, 'read.cpu_experts': x.rHost, 'read.resident': x.vr })); });
        detail.push({ tokens: n, s: t, binds: cb >= sb ? 'compute' : 'stream' });
        time += t; computeBound += cb; streamBound += sb; p0 += n;
      });
      rule = 'max';
    } else {
      // in series: every chunk's pass, then each token's compute (strata: t = chunks x t_pass + T x m)
      const pass = chunkParts(0, 0).reduce((a, x) => a + x.sHost + x.rHost + x.vr, 0);
      let comp = 0;
      if (q.read_mix) { const x = chunkParts(T, 0); x.forEach(y => { book(Object.assign({}, y, { sHost: 0, rHost: 0, vr: 0 }), 1); comp += y.ct + y.ce + y.cc + y.ca; }); }
      else sizes.forEach(n => { chunkParts(n, p0).forEach(y => { book(Object.assign({}, y, { sHost: 0, rHost: 0, vr: 0 }), 1); comp += y.ct + y.ce + y.cc + y.ca; }); p0 += n; });
      chunkParts(0, 0).forEach(y => book({ ct: 0, ce: 0, cc: 0, ca: 0, sHost: y.sHost * nChunks, rHost: y.rHost * nChunks, vr: y.vr * nChunks }, 1));
      time = comp + nChunks * pass; computeBound = comp; streamBound = nChunks * pass;
    }
    const list = Object.values(terms).map(t => Object.assign(t, { share: time ? t.s / time : null }));
    // in series every booked second adds to the time: one group holds the term totals
    if (rule === 'sum') groups.push(rlGroup(Object.fromEntries(list.filter(t => /^compute/.test(t.id)).map(t => [t.id, t.s])), Object.fromEntries(list.filter(t => !/^compute/.test(t.id)).map(t => [t.id, t.s]))));
    const side = computeBound >= streamBound ? 'compute' : 'stream';
    const top = list.filter(t => side === 'compute' ? /^compute/.test(t.id) : !/^compute/.test(t.id)).sort((a, b) => b.s - a.s)[0];
    return { status: 'OK', phase: 'prefill', placement: 'gpu', tokens: T, depth, chunk: C, chunks: nChunks, overlap: q.read_mix ? 'sum' : P.overlap, eta: e, eta_cpu: et.cpu, time_s: time, tps: T / time,
      binding: { side, term: top ? top.id : null, resource: top ? top.resource : null, compute_bound_s: computeBound, stream_bound_s: streamBound },
      terms: list, chunk_detail: detail, composition: { rule, groups }, unmodelled, ceilings: used, shapes: Object.keys(shapes).map(k => Object.assign({ card: k.split(' ')[0], line: shapeLine(shapes[k]) }, shapes[k])),
      link: link.map(l => l && l.gbs), unlock: st.stages.map(s => unlockState(s.g, q)), engine_src: P.src };
  }
  // decode of q.batch streams at depth q.ctx: per step, the bytes each tier serves and the FLOPs; terms add (layer order).
  // Weights are read once a step for every stream (amortised over the batch); the routed experts read are the union of the
  // streams' top-k (distinct = E (1 - (1 - k/E)^B) a layer, so streaming misses erode the batching gain); KV and recurrent
  // state grow with B and ctx. capacity: the streams whose KV + state fit the VRAM left beside the resident weights; past it the
  // overflow is read over the link every step (offload / eviction starts)
  function decodeRoof(data, q) {
    const I = rlInputs(data, q);
    if (I.why) return { status: 'UNMODELLED', why: I.why };
    if (I.pl.name !== 'gpu') return decodeRoofCpu(data, q, I);
    const { box, eng, shape, st } = I, D = eng.decode || {}, N = Number(q.batch) || 1, ctx = Number(q.ctx) || 0;
    const per = v(shape.expert_bytes) / (shape.L * shape.E), distinct = shape.E * (1 - Math.pow(1 - shape.k / shape.E, N));
    const terms = [], add = (id, resource, s) => terms.push({ id, resource, s });
    const unmodelled = [];
    // q.kv_format (fp16 | int8 | q4_0 | k8v4) reads the model's kv_host bytes per cell and layer (the engine's KV layout)
    const KH = data.models[q.model].kv_host, kvFmt = q.kv_format && KH && KH.bytes_per_cell_layer[q.kv_format] ? v(KH.bytes_per_cell_layer[q.kv_format]) * v(KH.layers) : null;
    const kvTok = kvFmt || (shape.kv_per_token ? v(shape.kv_per_token) : null), state = shape.state_bytes ? v(shape.state_bytes) : 0;
    let compT = 0, compE = 0, vram = 0, miss = 0, kv = 0, spill = 0, cap = Infinity, weightB = 0, kvB = 0;
    // the GEMMs of a step see tokens = B x the draft multiplier rows; the experts see tokens x k / the distinct experts they
    // touch rows each (1-4 at B 1-128 on 512 experts top-10: far below the shortest measured shape, where the bench kernel is
    // weight-read-bound, so the excess compose prices only what the read terms above do not)
    const dm = q.draft && shape.draft_mult ? v(shape.draft_mult) : 1, tokens = N * dm, distT = batchDistinct(shape.E, shape.k, tokens);
    const sctx = { model: q.model, store: q.store, compose: q.shape_compose }, shapes = [];
    st.stages.forEach((s, i) => {
      const layers = s.frac * shape.L, bytes = distinct * layers * per, all = s.resident + s.host;
      const x = q.resident_frac !== undefined ? q.resident_frac : all > 0 ? s.resident / all : 1;
      const hit = x >= 1 ? 1 : x <= 0 ? 0 : Math.pow(x, RL_SHAPE_A);
      const vbw = v(s.g.ceil.values.vram_gbs) * 1e9, dense = s.frac * (shape.dense_read ? v(shape.dense_read) : 0);
      vram += (dense + hit * bytes) / vbw; weightB += dense + bytes;
      const missB = (1 - hit) * bytes, L = rlLink(data, s.g, eng);
      if (missB > 0) {
        const rate = D.miss_path === 'cpu' ? v(box.dram.bw) : L ? L.gbs : null;
        if (rate) miss += missB / 1e9 / rate; else unmodelled.push('expert misses on ' + s.g.id + ': no rate');
      }
      // per-stream KV (attention layers x ctx) + recurrent state, read every step (state also written back)
      const readCtx = v(shape.keys_cap) ? Math.min(ctx, v(shape.keys_cap)) : ctx;  // a sparse-attention model reads only its selected keys
      const perStream = s.frac * ((kvTok && ctx ? kvTok * readCtx : 0) + 2 * state);
      kvB += N * perStream;
      const other = q.other_gib !== undefined ? q.other_gib : (s.g.other || []).filter(o => !o.household && !o.test_card).reduce((a, o) => a + (v(o.gib) || 0), 0);
      const free = v(s.g.vram_gib) * GiB - x * all - dense - other * GiB - (Number(q.headroom_gib) || 1) * GiB;
      const holds = s.frac * ((kvTok && ctx ? kvTok * ctx : 0) + state);
      if (holds > 0) cap = Math.min(cap, Math.max(0, Math.floor(free / holds)));
      const over = holds > 0 ? Math.max(0, N * holds - Math.max(0, free)) : 0, inV = N * perStream * (over > 0 ? 1 - over / (N * holds) : 1);
      kv += inV / vbw;
      if (over > 0) spill += (N * perStream - inV) / 1e9 / (L ? L.gbs : Infinity);
      const cT = ceilingOf(data, s.g.ceil, eng.prefill.trunk, unlockState(s.g, q)), cE = ceilingOf(data, s.g.ceil, eng.prefill.experts || eng.prefill.trunk, unlockState(s.g, q));
      if (cT) { const t = shapeTime(data, s.g, cT, 'trunk', eng.prefill.trunk_path || 'onednn', eng.prefill.trunk, tokens * 2 * (v(shape.trunk) + v(shape.head)) * s.frac, tokens, s.frac * (v(shape.trunk) + v(shape.head)), sctx); compT += t.s; shapes.push(Object.assign({ card: s.g.id, line: shapeLine(t) }, t)); }
      if (cE) { const t = shapeTime(data, s.g, cE, 'experts', eng.prefill.experts_path || 'onednn_grouped', eng.prefill.experts || eng.prefill.trunk, tokens * 2 * v(shape.routed) * s.frac, tokens * shape.k / distT, distT * s.frac * shape.L * shape.P, sctx); compE += t.s; shapes.push(Object.assign({ card: s.g.id, line: shapeLine(t) }, t)); }
    });
    const comp = compT + compE;
    if (!kvTok) unmodelled.push('KV reads (no kv_bytes_per_token for ' + q.model + ')');
    if (!state) unmodelled.push('recurrent state (no state_bytes_per_stream for ' + q.model + ')');
    add('read.resident', 'VRAM', vram); if (miss) add('read.misses', D.miss_path === 'cpu' ? 'DRAM (CPU computes misses)' : 'PCIe', miss);
    if (kv) add('read.kv_state', 'VRAM', kv); if (spill) add('read.kv_spill', 'PCIe (KV past the VRAM capacity)', spill);
    add('compute.trunk', 'tensor ' + eng.prefill.trunk, compT); add('compute.experts', 'tensor ' + (eng.prefill.experts || eng.prefill.trunk), compE);
    const step = terms.reduce((a, t) => a + t.s, 0);
    terms.forEach(t => { t.share = t.s / step; });
    const top = terms.slice().sort((a, b) => b.s - a.s)[0];
    return { status: 'OK', phase: 'decode', batch: N, ctx, step_s: step, tps_per_stream: dm / step, tps_total: N * dm / step, draft_mult: dm,
      binding: { term: top.id, resource: top.resource }, terms, unmodelled, distinct_experts_per_layer: distinct, weight_bytes_per_step: weightB, kv_state_bytes_per_step: kvB,
      capacity_streams: cap === Infinity ? null : cap, compute_s: comp, bandwidth_s: step - comp, tokens_per_step: tokens, rows_per_expert: tokens * shape.k / distT, shapes };
  }
  // decode of a CPU placement: every step reads the dense weights, the distinct experts of the batch and each stream's KV and state
  // from DRAM, and runs the GEMMs at the box CPU ceilings x the core share; terms add (layer order). No VRAM capacity applies
  function decodeRoofCpu(data, q, I) {
    const { shape, cpu, eng } = I, N = Number(q.batch) || 1, ctx = Number(q.ctx) || 0, P = eng.prefill;
    const dm = q.draft && shape.draft_mult ? v(shape.draft_mult) : 1, tokens = N * dm, distT = batchDistinct(shape.E, shape.k, tokens);
    const per = v(shape.expert_bytes) / (shape.L * shape.E), distinct = shape.E * (1 - Math.pow(1 - shape.k / shape.E, N));
    const dram = cpu.dram_gbs * 1e9, dense = shape.dense_read ? v(shape.dense_read) : 0, bytes = distinct * shape.L * per;
    const kvTok = shape.kv_per_token ? v(shape.kv_per_token) : null, state = shape.state_bytes ? v(shape.state_bytes) : 0;
    const readCtx = v(shape.keys_cap) ? Math.min(ctx, v(shape.keys_cap)) : ctx, kvB = N * ((kvTok && ctx ? kvTok * readCtx : 0) + 2 * state);
    const rate = k => cpu.ceil[k].ops * cpu.share;
    const compT = tokens * 2 * (v(shape.trunk) + v(shape.head)) / rate('trunk'), compE = tokens * 2 * v(shape.routed) / rate('experts');
    const unmodelled = [];
    if (!kvTok) unmodelled.push('KV reads (no kv_bytes_per_token for ' + q.model + ')');
    if (!state) unmodelled.push('recurrent state (no state_bytes_per_stream for ' + q.model + ')');
    const terms = [{ id: 'read.cpu_weights', resource: 'DRAM', s: (dense + bytes) / dram }];
    if (kvB) terms.push({ id: 'read.kv_state', resource: 'DRAM', s: kvB / dram });
    terms.push({ id: 'compute.cpu_trunk', resource: 'CPU ' + P.cpu_trunk, s: compT }, { id: 'compute.cpu_experts', resource: 'CPU ' + P.cpu_experts, s: compE });
    const step = terms.reduce((a, t) => a + t.s, 0), comp = compT + compE;
    terms.forEach(t => { t.share = t.s / step; });
    const top = terms.slice().sort((a, b) => b.s - a.s)[0];
    return { status: 'OK', phase: 'decode', placement: I.pl.name, batch: N, ctx, step_s: step, tps_per_stream: dm / step, tps_total: N * dm / step, draft_mult: dm,
      binding: { term: top.id, resource: top.resource }, terms, unmodelled, distinct_experts_per_layer: distinct, weight_bytes_per_step: dense + bytes, kv_state_bytes_per_step: kvB,
      capacity_streams: null, compute_s: comp, bandwidth_s: step - comp, tokens_per_step: tokens, rows_per_expert: tokens * shape.k / distT, shapes: [] };
  }
  // decode efficiency of an engine: measured / bound on its fit cells (every decode term is a byte or FLOP time, so one factor)
  function decodeFit(data, engine) {
    const F = ((data.roofline || {}).decode_fits || {})[engine];
    if (!F) return { status: 'UNMODELLED', why: 'no decode fit cell for ' + engine + ' (data.roofline.decode_fits)' };
    const cells = F.cells.map(id => data.roofline.checks.find(c => c.id === id));
    if (cells.some(c => !c)) return { status: 'UNMODELLED', why: 'decode fit cell missing: ' + F.cells.join(', ') };
    const r = cells.map(c => { const d = decodeRoof(data, c.query); return d.status === 'OK' ? c.measured.decode_tps / d.tps_per_stream : NaN; });
    if (r.some(x => !isFinite(x))) return { status: 'UNMODELLED', why: 'a decode fit cell has no bound' };
    return { status: 'FIT', eta: r.reduce((a, x) => a + x, 0) / r.length, cells: F.cells, tag: 'FIT', src: F.src };
  }
  // the batch tradeoff landscape: decode at B = 1, 2, 4 ... 128 streams on the same weights (per-stream and aggregate tok/s, the
  // binding term, distinct experts, weight bytes amortised, KV + state bytes, the capacity ceiling, compute vs bandwidth), and the
  // agent mix: F fresh prompt tokens per output token read in series with decode (useful = 1 / (step / (B x draft) + F / P))
  function batchLandscape(data, q, opts) {
    const o = opts || {}, Bs = o.batches || [1, 2, 4, 8, 16, 32, 64, 128];
    const W = data.workload, prof = W && W.profiles[v(W.settings.default_profile)];
    const F = o.fresh !== undefined ? o.fresh : prof ? v(prof.fresh_per_output) : null;
    const df = decodeFit(data, q.engine), pf = rlEta(data, q.engine);
    const pq = Object.assign({}, q, o.prefill_query || {}, { tokens: (o.prefill_query || {}).tokens || q.read_tokens || 2048 });
    const pre = prefillRoof(data, pq, rlEtaFor(data, pq));
    const P = pre.status === 'OK' ? pre.tps : null;
    let crossover = null;
    const rows = Bs.map(B => {
      const d = decodeRoof(data, Object.assign({}, q, { batch: B }));
      if (d.status !== 'OK') return { B, status: d.status, why: d.why };
      const step = df.status === 'FIT' ? d.step_s / df.eta : d.step_s, per = d.draft_mult / step, agg = B * per;
      if (crossover === null && d.compute_s >= d.bandwidth_s) crossover = B;
      const decS = 1 / agg, preS = F !== null && P ? F / P : null, useful = preS !== null ? 1 / (decS + preS) : null;
      return { B, status: 'OK', per_stream_tps: per, aggregate_tps: agg, binding: d.binding.term + ' (' + d.binding.resource + ')', distinct_experts_per_layer: d.distinct_experts_per_layer,
        weight_gb_per_output_token: d.weight_bytes_per_step / 1e9 / (B * d.draft_mult), kv_state_gb_per_step: d.kv_state_bytes_per_step / 1e9, fits_vram: d.capacity_streams === null || B <= d.capacity_streams,
        compute_share: d.compute_s / d.step_s, useful_tps: useful, useful_per_agent: useful !== null ? useful / B : null,
        rows_per_expert: d.rows_per_expert, experts_shape: (d.shapes.find(x => x.cls === 'experts') || {}).line || null,
        mix_binding: preS === null ? null : preS > decS ? 'prefill (F / P = ' + (preS * 1e3).toFixed(1) + ' ms an output token)' : 'decode (' + d.binding.term + ')' };
    });
    // where the binding term changes along B, and the B that maximises useful tok/s (inside the VRAM capacity, and over every row)
    const ok = rows.filter(r => r.status === 'OK'), changes = [];
    for (let i = 1; i < ok.length; i++) if (ok[i].binding !== ok[i - 1].binding) changes.push({ B: ok[i].B, from: ok[i - 1].binding, to: ok[i].binding });
    const best = xs => xs.filter(r => r.useful_tps !== null).sort((a, b) => b.useful_tps - a.useful_tps)[0] || null;
    const bi = best(ok.filter(r => r.fits_vram)), ba = best(ok);
    const d1 = decodeRoof(data, Object.assign({}, q, { batch: 1 }));
    return { query: q, F, F_src: o.fresh !== undefined ? 'a setting' : prof ? prof.fresh_per_output.src : null, prefill_tps: P, prefill_eta: pf.gpu, prefill_eta_cpu: (q.placement || 'gpu') !== 'gpu' ? pf.cpu : null, prefill_why: pre.why || null,
      decode_eta: df.status === 'FIT' ? df.eta : null, decode_why: df.why || null, capacity_streams: d1.status === 'OK' ? d1.capacity_streams : null, compute_overtakes_bandwidth_at: crossover,
      binding_changes: changes, best_useful: bi ? { B: bi.B, useful_tps: bi.useful_tps } : null, best_useful_any: ba ? { B: ba.B, useful_tps: ba.useful_tps, fits_vram: ba.fits_vram } : null,
      prefill_shapes: pre.status === 'OK' ? pre.shapes.map(x => x.card + ' ' + x.line) : [], rows };
  }
  // ---------- multi-box pipeline: one model split by layers over cards on several boxes (data/multibox.json) ----------
  // The engine path is llama.cpp RPC (lab notes README): the client box holds the last stage and the LM head,
  // every other box runs an rpc-server, a decode step runs the stages one after another (no cross-box overlap), and each
  // remote stage costs one blocking round trip a step carrying the hidden stream (B x hidden_stream x 4 bytes) each way.
  // pipelinePlan places the layers (spec.split: 'memory', llama.cpp's default by free memory, or 'bandwidth', layers
  // proportional to each card's measured decode rate) and refuses a stage whose weights + the KV and state of the plan's
  // streams pass its card, or a remote stage whose client link has no measured round trip (data/interconnects.json hop_rtt).
  // spec = { id, model, store, client, cards: [{box, gpu}] in stage order (the client's cards last), ctx, streams, kv_type,
  // split, transport, keys_read, displace: [service names freed for the plan], mode }
  const mbV = x => (x && typeof x === 'object' && 'v' in x ? x.v : x);
  function mbHop(data, a, b, transport, bytes) {
    const H = ((data.network || {}).hop_rtt || []).filter(h => h.ends.includes(a) && h.ends.includes(b) && (!transport || h.transport === transport));
    if (!H.length) return null;
    const h = H[0], pts = mbV(h.rtt_us).slice().sort((x, y) => x[0] - y[0]);
    // hop model (settings.hop_model): 'stream' = the small-payload round trip + the payload both ways at the link's measured
    // single-connection TCP rate (network observations of the link, fewest streams); 'pingpong' = the measured python
    // ping-pong curve by payload (an upper bound: python copies), extrapolated linearly past its last point
    const hm = mbV(((data.multibox || {}).settings || {}).hop_model) || 'stream';
    const obs = ((data.network || {}).observations || []).filter(x => x.link === h.link && /^tcp/.test(x.transport) && x.gbps && mbV(x.gbps)).sort((x, y) => x.streams - y.streams)[0];
    let us;
    if (h.gbps) us = pts[0][1] + Math.max(0, bytes - pts[0][0]) * 8 / (mbV(h.gbps) * 1e3);
    else if (hm === 'stream' && obs) us = pts[0][1] + 2 * bytes * 8 / (mbV(obs.gbps) * 1e3);
    else if (pts.length === 1) us = pts[0][1];
    else { const i = Math.max(1, pts.findIndex(p => p[0] >= bytes)), j = i < 0 || i >= pts.length ? pts.length - 1 : i, p0 = pts[j - 1], p1 = pts[j];
      us = p0[1] + (p1[1] - p0[1]) * (bytes - p0[0]) / (p1[0] - p0[0]); }
    return { s: Math.max(us, pts[0][1]) / 1e6, id: h.id, model: hm === 'stream' && obs ? 'stream ' + obs.id : 'pingpong', tag: h.rtt_us.tag, extrapolated: bytes > pts[pts.length - 1][0] };
  }
  function mbLine(data, box, g) {
    const MB = data.multibox, be = mbV(MB.backend_of_card)[g.ceil ? g.ceil.class : ''], L = ((data.bench || {}).gpu || {})[be];
    if (!be || !L) return { why: 'card ' + box + ' ' + g.id + ': no llama.cpp backend line (multibox backend_of_card, bench.gpu)' };
    // a line measured on a card of the same class (odie b70_1 for b70_2) is the card's own line
    const cls = g.ceil ? g.ceil.class : null, own = (L.cards || []).some(c => { const [bk, id] = c.split(' '), h = ((data.boxes[bk] || {}).gpus || []).find(y => y.id === id); return c === box + ' ' + g.id || (h && h.ceil && h.ceil.class === cls); }), nameplate = mbV(g.bw) || (g.ceil && mbV(g.ceil.values.vram_gbs));
    const rate = own && mbV(L.rate_gbs) ? mbV(L.rate_gbs) : mbV(L.eta) * nameplate;
    return { backend: be, rate_gbs: rate, host_ms: mbV(L.host_ms) || 0, moe_layer_ms: mbV(L.moe_layer_ms), own,
      tag: own ? 'MEASURED' : 'ESTIMATE', src: own ? L.rate_gbs.src : 'ESTIMATE: the ' + be + ' line\'s eta ' + mbV(L.eta) + ' (measured on ' + (L.cards || []).join(', ') + ') x this card\'s nameplate ' + nameplate + ' GB/s' };
  }
  function pipelinePlan(data, spec) {
    const MB = data.multibox, un = why => ({ status: 'REFUSED', why });
    if (!MB) return un('no data.multibox');
    const M = MB.models[spec.model], S = M && M.stores[spec.store], set = MB.settings;
    if (!S) return un('model ' + spec.model + ' store ' + spec.store + ' is not in data/multibox.json');
    const L = mbV(M.L), E = mbV(M.E), k = mbV(M.k), ctx = Number(spec.ctx) || 0, n = Number(spec.streams) || 1;
    const kvB = mbV(set.kv_type_bytes)[spec.kv_type || mbV(set.default_kv_type)];
    const kvTokLayer = mbV(M.attn_layers) * (mbV(M.kv_values_per_token_layer) + mbV(M.indexer_values_per_token_layer)) * kvB / L;
    const streamLayer = kvTokLayer * ctx + mbV(M.state_bytes_per_stream) / L;
    const layerB = (mbV(S.experts_bytes) + mbV(S.trunk_bytes)) / L, head = mbV(S.head_bytes);
    const reserve = mbV(set.card_reserve_gib) * GiB, freed = new Set(spec.displace || []);
    const cards = [];
    for (const c of spec.cards) {
      const box = data.boxes[c.box], g = box && box.gpus.find(x => x.id === c.gpu);
      if (!g) return un('no card ' + c.gpu + ' on ' + c.box);
      const line = mbLine(data, c.box, g);
      if (line.why) return un(line.why);
      const others = (g.other || []).filter(x => !freed.has(x.service || x.what)), otherB = others.reduce((a, x) => a + (mbV(x.gib) || 0), 0) * GiB;
      cards.push({ box: c.box, gpu: c.gpu, g, line, free: mbV(g.vram_gib) * GiB - reserve - otherB, kept: others.map(x => x.service || x.what) });
    }
    const last = cards.length - 1;
    if (cards[last].box !== spec.client) return un('the last stage must sit on the client ' + spec.client + ' (the LM head goes to the last device; a remote head ships n_vocab x B x 4 logits a step: multibox README a.3)');
    // layer split: by free memory (llama.cpp -ts default) or by measured decode rate; largest remainders, every card >= 1 layer
    // spec.layers: the layer count of each stage as a run set it (-ts), summing to L
    let lay;
    if (spec.layers) {
      if (spec.layers.length !== cards.length || spec.layers.reduce((a, x) => a + x, 0) !== L) return un('spec.layers ' + JSON.stringify(spec.layers) + ' must give each of the ' + cards.length + ' cards a layer count, summing to ' + L);
      lay = spec.layers.slice();
    } else {
      const w = cards.map((c, i) => spec.split === 'bandwidth' ? c.line.rate_gbs : Math.max(0, c.free - (i === last ? head : 0)));
      const tot = w.reduce((a, x) => a + x, 0), raw = w.map(x => L * x / tot);
      lay = raw.map(Math.floor);
      raw.map((x, i) => [x - Math.floor(x), i]).sort((a, b) => b[0] - a[0]).slice(0, L - lay.reduce((a, x) => a + x, 0)).forEach(([, i]) => { lay[i]++; });
    }
    const stages = cards.map((c, i) => {
      const weights = lay[i] * layerB + (i === last ? head : 0), room = c.free - weights;
      return Object.assign({}, c, { layers: lay[i], frac: lay[i] / L, weights_gb: weights / 1e9, free_gb: c.free / 1e9, room_gb: room / 1e9,
        capacity_streams: lay[i] && streamLayer > 0 ? Math.max(0, Math.floor(room / (lay[i] * streamLayer))) : Infinity, head: i === last });
    });
    // spec.no_memory_check: a plan that describes a run that happened (card_curves anchors): its memory is a fact, not a question
    const over = spec.no_memory_check ? [] : stages.filter(s => s.room_gb < 0);
    if (over.length) return Object.assign(un('stage memory passes its card: ' + over.map(s => s.box + ' ' + s.gpu + ' ' + s.layers + ' layers ' + s.weights_gb.toFixed(1) + ' GB weights > ' + s.free_gb.toFixed(1) + ' GB free').join('; ')), { stages });
    const cap = Math.min(...stages.map(s => s.capacity_streams));
    if (cap < n && !spec.no_memory_check) return Object.assign(un('the KV + state of ' + n + ' streams at ctx ' + ctx + ' pass a card: the plan holds ' + cap + ' (' + (streamLayer * L / 1e9).toFixed(2) + ' GB a stream)'), { stages, capacity_streams: cap });
    // hops: each remote stage, one blocking round trip on the client link; a device change inside the client box goes over PCIe
    const hops = [];
    for (let i = 0; i < stages.length; i++) {
      const s = stages[i];
      if (s.box !== spec.client) {
        const tr = (spec.transport_by_box || {})[s.box] || spec.transport || 'tcp', probe = mbHop(data, spec.client, s.box, tr, 64);
        if (!probe) return un('no measured round trip between ' + spec.client + ' and ' + s.box + ' over ' + tr + ' in data/interconnects.json hop_rtt: a hop the data does not hold is not priced');
        hops.push({ stage: i, kind: 'rpc', a: spec.client, b: s.box, transport: tr });
      } else if (i > 0 && stages[i - 1].box === spec.client && stages[i - 1].gpu !== s.gpu) hops.push({ stage: i, kind: 'pcie', a: s.box, b: s.box, g: s.g });
    }
    return { status: 'OK', id: spec.id, spec, model: M, store: S, L, E, k, ctx, kvB, stream_gb: streamLayer * L / 1e9, layer_gb: layerB / 1e9, stages, capacity_streams: cap, hops,
      vram_gb: stages.reduce((a, s) => a + s.free_gb, 0), weights_gb: stages.reduce((a, s) => a + s.weights_gb, 0) };
  }
  // ---------- per-card batched decode step by kernel class (data/multibox.json card_curves) ----------
  // A stage's step at b tokens: the card's host intercept x its layer share + the model graph's overhead x its layers + its
  // dense weights W (trunk share + head) + its routed experts + every stream's KV and state, each class priced by the kernel the
  // backend runs at that b (card_curves.kernels, TRACED):
  //   dense, b <= mv_max (mat-vec): max(W / R, b x W x col): the weights are read once, every column pays its ALU work, so a
  //     GEMV turns compute-bound once b x col passes 1 / R;
  //   dense, b > mv_max: SYCL dequantizes to f16 and runs a GEMM: (W + 4 x params) / R (writes and reads 2 bytes a parameter;
  //     DERIVED, no constant), at least its GEMM at the fp16 ceiling x the prefill eta; CUDA MMQ: max(W / R, W x mmq); any
  //     other: max(W / R, W x <kernel>);
  //   MoE, b <= moe_mv_max: b x k mat-vec columns a layer, each over its expert: pairs x expert bytes x max(1 / R, col) (at b 1
  //     the line); MoE, b > moe_mv_max: the distinct experts E(1 - (1 - k/E)^b) through the batched kernel, max(bytes / R,
  //     bytes x <kernel>) (SYCL grouped: grp; CUDA: mmq);
  //   KV and state: b x the stage's layer share x (KV at the read keys + the state read and written) / R; the per-MoE-layer
  //     cost of the line (moe_layer_ms) on a MoE model only.
  // R = the card's own llama.cpp line (bench.gpu) or a solved rate constant. Every constant (col, mmq, grp, rate, overhead) is
  // solved from one named calibration point (cardCurveFit); a class whose constant is missing prices at its read (the old
  // bandwidth line) and marks the stage UNCALIBRATED
  const CC_KERNEL_CONST = { grouped: 'grp', mmq: 'mmq' };
  const ccKey = s => s.line.backend + '/' + (s.g.ceil ? s.g.ceil.class : s.gpu);
  function cardStage(data, plan, s, b, keys, C) {
    const M = plan.model, S = plan.store, L = plan.L, E = plan.E, k = plan.k, CCs = (data.multibox || {}).card_curves || null, set = data.multibox.settings;
    const key = ccKey(s), cls = s.g.ceil ? s.g.ceil.class : s.gpu, K = CCs ? (mbV(CCs.kernels) || {})[s.line.backend] || null : null;
    const need = [], used = [], derived = [];
    const get = t => { const x = C[key + '.' + t]; if (x === undefined || x === null) { need.push(key + '.' + t); return null; } used.push(key + '.' + t); return x; };
    const rateC = C[key + '.rate'];
    if (rateC !== undefined && rateC !== null) used.push(key + '.rate');
    const R = (rateC !== undefined && rateC !== null ? rateC : s.line.rate_gbs) * 1e9, perGB = (bytes, c) => bytes / 1e9 * c / 1e3;
    const trunkL = mbV(S.trunk_bytes) / L, expL = mbV(S.experts_bytes) / L, tpL = mbV(M.trunk_params || M.active_params_per_token) / L;
    const W = s.layers * trunkL + (s.head ? mbV(S.head_bytes) : 0), P = s.layers * tpL + (s.head ? mbV(M.head_params) || 0 : 0);
    let dense, dk;
    if (!K) { dense = W / R; dk = 'read (no kernel table for ' + s.line.backend + ')'; if (b > 1) need.push(key + ' kernels'); }
    else if (b <= 1) { dense = W / R; dk = 'mat-vec at 1 (the line)'; }
    else if (b <= K.mv_max) { const c = get('col'); dense = Math.max(W / R, c === null ? 0 : b * perGB(W, c)); dk = 'mat-vec x ' + +b.toFixed(2) + ' columns'; }
    else if (K.dense_batched === 'dequant_f16') {
      const ce = ceilingOf(data, s.g.ceil, 'fp16_tensor', 'stock'), eta = (mbV(set.prefill_eta) || {})[s.line.backend] || 0.25;
      dense = Math.max((W + 4 * P) / R, ce ? 2 * P * b / (ce.ops * eta) : 0); dk = 'dequantize to f16 + GEMM (DERIVED: ' + (4 * P / 1e9).toFixed(2) + ' GB of f16 traffic)'; derived.push('dense ' + dk);
    } else { const t = CC_KERNEL_CONST[K.dense_batched] || K.dense_batched, c = get(t); dense = Math.max(W / R, c === null ? 0 : perGB(W, c)); dk = K.dense_batched; }
    let moe = 0, mk = null, moeBytes = 0;
    if (E > 0 && k > 0 && expL > 0) {
      const eb = expL / E, mvm = K ? ((K.moe_mv_max_by_class || {})[cls] || K.moe_mv_max) : Infinity;
      if (b <= mvm) {
        const pairs = b * k, c = b > 1 ? get('col') : 0; moeBytes = s.layers * pairs * eb;
        moe = moeBytes * Math.max(1 / R, c === null ? 0 : c / 1e12); mk = 'mat-vec-id, ' + +pairs.toFixed(1) + ' columns a layer';
      } else {
        const dist = E * (1 - Math.pow(1 - k / E, b)), t = CC_KERNEL_CONST[K.moe_batched] || K.moe_batched, c = get(t); moeBytes = s.layers * dist * eb;
        moe = Math.max(moeBytes / R, c === null ? 0 : perGB(moeBytes, c)); mk = K.moe_batched + ' over ' + dist.toFixed(1) + ' distinct experts a layer';
      }
    }
    const kvTokLayer = mbV(M.attn_layers) * (mbV(M.kv_values_per_token_layer) + mbV(M.indexer_values_per_token_layer)) * plan.kvB / L, state = mbV(M.state_bytes_per_stream);
    const streamB = b * s.frac * (kvTokLayer * L * keys + 2 * state), stream = streamB / R;
    const ov = C['overhead.' + plan.spec.model];
    if (ov !== undefined && ov !== null) used.push('overhead.' + plan.spec.model);
    const fixed = (s.line.host_ms * s.frac + (E > 0 ? (s.line.moe_layer_ms || 0) * s.layers : 0) + (ov || 0) * s.layers) / 1e3;
    const bytes = W + moeBytes + streamB, read = bytes / R, t = fixed + dense + moe + stream;
    const kern = dense + moe + stream - read;
    return { box: s.box, gpu: s.gpu, layers: s.layers, bytes_gb: bytes / 1e9, read_s: read, compute_s: Math.max(0, kern), fixed_s: fixed, dense_s: dense, moe_s: moe, stream_s: stream, s: t,
      binds: kern > 0.01 * read ? 'kernel' : 'read', kernels: { dense: dk, moe: mk }, curve: need.length ? 'UNCALIBRATED' : b <= 1 ? 'LINE' : derived.length && !used.some(u => !/overhead|rate/.test(u)) ? 'DERIVED' : 'CALIBRATED',
      needs: need, constants: used, derived, rate_gbs: R / 1e9 };
  }
  // the solved constants (cardCurveFit), cached on the data object (not enumerable, so a JSON clone refits)
  function cardConsts(data) {
    const CCs = (data.multibox || {}).card_curves;
    if (!CCs) return {};
    if (!data.__cardFit || data.__cardFit.of !== CCs) Object.defineProperty(data, '__cardFit', { value: { of: CCs, fit: cardCurveFit(data) }, enumerable: false, configurable: true, writable: true });
    return data.__cardFit.fit.consts;
  }
  // decode step of B streams at read depth ctxRead (default the plan's ctx): per stage cardStage, mode 'serial' (llama.cpp):
  // step = sum of stages + hops; 'ring' (K = stages microbatches in flight, prima.cpp's piped ring, not upstream): cycle = K x the
  // slowest stage at B / K. o.consts replaces the solved constants (cardCurveFit solves them in turn). The result says whether
  // every stage had a calibrated or derived curve at B (calibration: LINE / DERIVED / CALIBRATED / UNCALIBRATED)
  function pipelineRoof(data, plan, B, o) {
    const x = o || {}, set = data.multibox.settings, M = plan.model;
    const ctxRead = x.ctx !== undefined ? x.ctx : plan.ctx, mode = x.mode || plan.spec.mode || 'serial';
    const keys = (x.keys_read || plan.spec.keys_read || mbV(set.keys_read)) === 'cap' && data.roofline.models[plan.spec.model] ? Math.min(ctxRead, mbV(data.roofline.models[plan.spec.model].attn_keys_cap) || ctxRead) : ctxRead;
    const act = mbV(M.hidden_stream) * mbV(set.act_bytes_per_value), C = x.consts || cardConsts(data);
    const stageAt = (s, b) => cardStage(data, plan, s, b, keys, C);
    const hopAt = b => plan.hops.map(h => { if (h.kind === 'pcie') { const l = rlLink(data, h.g, { prefill: { link_eff: null } }); return { stage: h.stage, kind: 'pcie', s: l ? b * act / (l.gbs * 1e9) : 0 }; }
      const r = mbHop(data, h.a, h.b, h.transport, b * act); return { stage: h.stage, kind: 'rpc ' + h.a + '-' + h.b, s: r.s * mbV(set.rpc_round_trips_per_remote_stage), extrapolated: r.extrapolated }; });
    const st = plan.stages.map(s => stageAt(s, B)), hp = hopAt(B), sumS = st.reduce((a, t) => a + t.s, 0), sumH = hp.reduce((a, t) => a + t.s, 0);
    let step = sumS + sumH, per = 1 / step, agg = B / step, ring = null;
    if (mode === 'ring') {
      const K = Math.min(B, plan.stages.length), bm = B / K, sm = plan.stages.map(s => stageAt(s, bm)), hm = hopAt(bm), lat = sm.reduce((a, t) => a + t.s, 0) + hm.reduce((a, t) => a + t.s, 0);
      const cycle = Math.max(K * Math.max(...sm.map(t => t.s)), lat);
      ring = { K, microbatch: bm, cycle_s: cycle, slowest: sm.reduce((a, t) => (t.s > a.s ? t : a)) };
      step = cycle; per = 1 / cycle; agg = B / cycle;
    }
    const slow = st.reduce((a, t) => (t.s > a.s ? t : a)), unc = st.filter(t => t.curve === 'UNCALIBRATED');
    const rank = { LINE: 0, DERIVED: 1, CALIBRATED: 2, UNCALIBRATED: 3 }, cal = st.reduce((a, t) => (rank[t.curve] > rank[a] ? t.curve : a), 'LINE');
    return { B, mode, ctx_read: ctxRead, keys_read: keys, step_s: step, per_stream_tps: per, aggregate_tps: agg, stages: st, hops: hp, hop_s: sumH, hop_share: sumH / (sumS + sumH), slowest: slow.box + ' ' + slow.gpu + ' (' + slow.binds + ')',
      fits: B <= plan.capacity_streams, ring, read_gb: st.reduce((a, t) => a + t.bytes_gb, 0), calibration: cal,
      uncalibrated: unc.map(t => t.box + ' ' + t.gpu + ': no ' + t.needs.join(', ') + ' (dense ' + t.kernels.dense + (t.kernels.moe ? '; MoE ' + t.kernels.moe : '') + ')') };
  }
  // ---------- solving the card-curve constants from their named calibration points, and the hold-out rescore ----------
  // Fit only points marked 'calibrates', jointly within physical bounds. Residuals
  // and convergence are reported; holdouts remain independent of the fit.
  const CC_RANGE = { col: [0, 50], mmq: [0, 200], grp: [0, 200], rate: [5, 5000], overhead: [0, 20] };
  function cardCurveFit(data) {
    const CCs = (data.multibox || {}).card_curves, consts = {}, solved = [];
    if (!CCs) return { consts, solved };
    const rows = [];
    for (const a of CCs.anchors || []) for (const p of a.points) {
      if (p.use !== 'calibrates') continue;
      const plan = pipelinePlan(data, a.plan), term = /^overhead/.test(p.constant) ? 'overhead' : String(p.constant).split('.').pop();
      if (plan.status !== 'OK' || !CC_RANGE[term] || !Number.isFinite(p.tps) || !(p.tps > 0) || !Number.isInteger(p.B) || p.B <= 0) {
        solved.push({ anchor: a.id, B: p.B, constant: p.constant, status: 'REFUSED', why: plan.why || 'invalid calibration point' }); continue;
      }
      rows.push({a, p, plan, term});
    }
    // Deterministic initialization, then joint bounded least squares on calibration
    // points only. Each sweep revisits all constants; a clamped early anchor cannot
    // silently decide all later constants by JSON list order. Holdouts never fit.
    rows.sort((a, b) => a.p.constant.localeCompare(b.p.constant) || a.a.id.localeCompare(b.a.id) || a.p.B - b.p.B);
    const params = [...new Set(rows.map(x => x.p.constant))];
    const range = k => CC_RANGE[rows.find(x => x.p.constant === k).term];
    params.forEach(k => { consts[k] = range(k)[0]; });
    const loss = () => rows.reduce((sum, r) => {
      const predicted = pipelineRoof(data, r.plan, r.p.B, {ctx: r.a.ctx_read, consts}).step_s;
      return sum + (predicted > 0 && Number.isFinite(predicted) ? Math.log(predicted / (r.p.B / r.p.tps)) ** 2 : Infinity);
    }, 0);
    let previous = loss(), converged = false, sweeps = 0;
    for (; sweeps < 20; sweeps++) {
      for (const k of params) {
        const [lo, hi] = range(k), old = consts[k];
        const objective = x => { consts[k] = x; return loss(); };
        const [x, err] = golden(objective, lo, hi, 36);
        const options = [[x, err], [lo, objective(lo)], [hi, objective(hi)], [old, objective(old)]];
        options.sort((a, b) => a[1] - b[1]);
        consts[k] = options[0][0];
      }
      const current = loss();
      if (Math.abs(previous - current) < 1e-12) { converged = true; break; }
      previous = current;
    }
    for (const {a, p, plan, term} of rows) {
      const x = consts[p.constant], [lo, hi] = range(p.constant);
      const predicted = pipelineRoof(data, plan, p.B, {ctx: a.ctx_read, consts}).aggregate_tps;
      solved.push({anchor: a.id, arm: a.arm, B: p.B, measured_tps: p.tps, constant: p.constant, value: x,
        status: !Number.isFinite(predicted) ? 'REFUSED' : x === lo ? 'AT_FLOOR' : x === hi ? 'AT_CEILING' : converged ? 'SOLVED' : 'ITERATION_LIMIT',
        residual: predicted / p.tps - 1, unit: (CCs.units || {})[term] || null});
    }
    return { consts, solved, converged, sweeps: sweeps + 1, loss: loss() };
  }
  // every anchor point predicted with the solved constants: err = predicted / measured - 1; use 'calibrates' rows reproduce their
  // point (residual ~0 by construction), 'holdout' rows are the test, 'line' rows check the B 1 line on a model it was fitted on.
  // The hold-out summary leaves out the UNCALIBRATED rows (a class with no constant: the number is the old bandwidth line)
  function cardCurveRescore(data) {
    const CCs = (data.multibox || {}).card_curves;
    if (!CCs) return { status: 'UNMODELLED', why: 'no data.multibox.card_curves' };
    cardConsts(data);
    const fit = data.__cardFit.fit, rows = [];
    for (const a of CCs.anchors) {
      const plan = pipelinePlan(data, a.plan);
      for (const p of a.points) {
        if (plan.status !== 'OK') { rows.push({ anchor: a.id, arm: a.arm, B: p.B, measured: p.tps, use: p.use, status: 'REFUSED', why: plan.why }); continue; }
        const d = pipelineRoof(data, plan, p.B, { ctx: a.ctx_read }), line = pipelineRoof(data, plan, p.B, { ctx: a.ctx_read, consts: {} });
        rows.push({ anchor: a.id, arm: a.arm, tool: a.tool, B: p.B, measured: p.tps, predicted: d.aggregate_tps, err: d.aggregate_tps / p.tps - 1, line_tps: line.aggregate_tps, line_err: line.aggregate_tps / p.tps - 1, use: p.use, constant: p.constant || null, note: p.note || null,
          calibration: d.calibration, uncalibrated: d.uncalibrated, slowest: d.slowest, step_ms: 1e3 * d.step_s, hop_share: d.hop_share,
          stages: d.stages.map(t => ({ card: t.box + ' ' + t.gpu, layers: t.layers, ms: 1e3 * t.s, dense: t.kernels.dense, moe: t.kernels.moe, curve: t.curve })) });
      }
    }
    const med = xs => { const a = xs.slice().sort((x, y) => x - y), n = a.length; return n ? (n % 2 ? a[(n - 1) / 2] : (a[n / 2 - 1] + a[n / 2]) / 2) : null; };
    const H = rows.filter(r => r.use === 'holdout' && r.status !== 'REFUSED' && r.calibration !== 'UNCALIBRATED');
    const sum = xs => ({ n: xs.length, median_abs_pct: xs.length ? 100 * med(xs.map(r => Math.abs(r.err))) : null, bias_pct: xs.length ? 100 * med(xs.map(r => r.err)) : null, within_25: xs.filter(r => Math.abs(r.err) <= 0.25).length });
    return { status: 'OK', constants: fit.solved, rows, holdout: sum(H), holdout_b1: sum(H.filter(r => r.B === 1)), holdout_batched: sum(H.filter(r => r.B > 1)),
      uncalibrated: rows.filter(r => r.calibration === 'UNCALIBRATED').map(r => r.anchor + ' B ' + r.B) };
  }
  // prefill of T fresh tokens at depth 0 in ubatches: per ubatch every stage in turn (the RPC copy blocks: multibox README a.2
  // correction), each at max(compute at its tensor ceiling x eta, its weight read) + its intercept, plus the hops at the ubatch
  // payload. Attention over earlier keys is not priced (named)
  function pipelinePrefill(data, plan, T) {
    const set = data.multibox.settings, M = plan.model, S = plan.store, ub = mbV(set.ubatch), eta = mbV(set.prefill_eta), act = mbV(M.hidden_stream) * mbV(set.act_bytes_per_value);
    const n = Math.ceil(T / ub), per = T / n;
    const dist = plan.E * (1 - Math.pow(1 - plan.k / plan.E, per)) / plan.E;
    const st = plan.stages.map(s => {
      const ce = ceilingOf(data, s.g.ceil, 'int8_tensor', 'stock'), comp = ce ? per * 2 * mbV(M.active_params_per_token) * s.frac / (ce.ops * (eta[s.line.backend] || 0.25)) : 0;
      const read = s.layers * (mbV(S.trunk_bytes) / plan.L + dist * mbV(S.experts_bytes) / plan.L) / (s.line.rate_gbs * 1e9), fixed = (s.line.host_ms * s.frac + (s.line.moe_layer_ms || 0) * s.layers) / 1e3;
      return { box: s.box, gpu: s.gpu, s: Math.max(comp, read) + fixed, compute_s: comp, read_s: read };
    });
    const hops = plan.hops.map(h => h.kind === 'pcie' ? 0 : mbHop(data, h.a, h.b, h.transport, per * act).s);
    const t = n * (st.reduce((a, x) => a + x.s, 0) + hops.reduce((a, x) => a + x, 0));
    return { tokens: T, ubatches: n, time_s: t, tps: T / t, stages: st, hop_s: n * hops.reduce((a, x) => a + x, 0), unmodelled: ['attention over the earlier keys (depth)', 'the eta is an ESTIMATE (settings.prefill_eta)'] };
  }
  // the batch landscape of a plan: decode at every B (serial and ring), the hop share, capacity, and the agent duty cycle
  // (agentLandscape with the plan's rates: its aggregate decode at the running count, its prefill, its stream capacity as the
  // admitted slots), so useful tok/s carries tool gaps and F(B, cache)
  function pipelineLandscape(data, spec, opts) {
    const o = opts || {}, plan = pipelinePlan(data, spec);
    if (plan.status !== 'OK') return plan;
    const Bs = o.batches || mbV(data.multibox.settings.batches), pre = pipelinePrefill(data, plan, o.prefill_tokens || 2048);
    const rows = Bs.map(B => { const d = pipelineRoof(data, plan, B), r = pipelineRoof(data, plan, B, { mode: 'ring' }); return Object.assign(d, { ring_aggregate_tps: r.aggregate_tps }); });
    const grid = []; for (let n = 1; n <= Math.max(...Bs); n *= 2) grid.push(n); Bs.forEach(b => { if (!grid.includes(b)) grid.push(b); }); grid.sort((a, b) => a - b);
    const curve = grid.map(n => [n, pipelineRoof(data, plan, n).aggregate_tps]);
    const cap = Number.isFinite(plan.capacity_streams) ? plan.capacity_streams : null;
    const ag = o.agents === false ? null : agentLandscape(data, { model: spec.model, engine: 'llamacpp' }, { batches: Bs, server: o.server || mbV(data.multibox.settings.agent_server), model: spec.model, stall: o.stall !== undefined ? o.stall : data.multibox.settings.agent_stall,
      slots: cap ? { v: cap, tag: 'MODEL', src: 'pipelinePlan capacity_streams at ctx ' + plan.ctx } : null, cache_conversations: o.cache_conversations || (cap ? { v: cap, tag: 'INFERRED', src: 'INFERRED: llama-server keeps a conversation only in its slot, so the cache holds as many as the slots' } : null),
      rates: { prefill_tps: { v: pre.tps, tag: 'MODEL', src: 'pipelinePrefill' }, decode_curve: { v: curve, tag: 'MODEL', src: 'pipelineRoof serial aggregate decode by B' }, capacity_streams: cap }, admission: o.admission || null });
    const okA = ag && ag.status === 'OK' ? ag.rows.filter(r => r.status === 'OK') : [];
    const best = okA.slice().sort((a, b) => b.useful_tps - a.useful_tps)[0] || null;
    const uncB = rows.filter(r => r.calibration === 'UNCALIBRATED').map(r => r.B);
    // the replica gate on the serial aggregate inside the plan's stream capacity (o.replica false: skip; an object: its opts)
    const replica = o.replica === false ? null : replicaGate(data, spec, rows.filter(r => r.fits).map(r => [r.B, r.aggregate_tps, r.calibration]), Object.assign({ split_tag: 'MODEL' }, o.replica || {}));
    return { status: 'OK', plan, prefill: pre, rows, agents: ag, best_useful: best ? { B: best.B, useful_tps: best.useful_tps, b_eff: best.b_eff, per_agent: best.useful_per_agent } : null,
      uncalibrated_from_B: uncB.length ? Math.min(...uncB) : null, replica };
  }
  // ---------- the replica gate: a split must beat its own cards run apart ----------
  // A llama.cpp RPC split runs its stages one after another, so by itself it buys room (a model no single card holds), not
  // throughput: S2 measured two B70s pipelined at 82.9 tok/s at B 8 against 84.1 for one B70 alone. replicaBaseline prices
  // the other use of the same cards: each card runs its own replica of the best single-card recipe at the quality floor or
  // above, serving a agents, and the aggregates add. replicaGate compares the split serving N cards x a agents with it:
  //   GAINS             the split's aggregate passes the replicas' by more than settings.replica_gain_margin;
  //   CAPACITY-ONLY     it does not: the cards serve as much or more running apart, the split buys only room;
  //   CAPACITY-REQUIRED no card holds a recipe at the floor alone: the split is the only way to run that quality.
  // Each side serves its agents at the best concurrency it admits up to its agent count (a server may queue), so its
  // aggregate at n agents is the best measured or modelled aggregate at b <= n inside its stream capacity. The verdict is
  // read at the top of settings.replica_agents_per_card, where both sides peak; every row is printed.
  // Single-card recipes: (1) model: a one-card plan (pipelinePlan, that card alone, VRAM only, priced by the same card
  // curves as the split, at the split's ctx, KV type and displaced services) for every multibox model and store at the
  // floor; (2) MEASURED: a card_curves anchor run on one card of the same class; (3) MEASURED: a data.curves agents-vs-speed
  // config with one stage on this box when every card on the box is of this class, DRAM-assisted when the model's smallest
  // store does not fit the card. settings.replica_dram_per_box caps the DRAM-assisted replicas a box counts (they share its
  // DRAM); the result also carries the verdict with the cap lifted. A card with no recipe at the floor adds 0 and is named
  // (unpriced): no VRAM fit and no measured DRAM run, so a GAINS verdict there is over a partial baseline. A measured
  // replica ran at its own prompt depth (agent-length bench prompts), while a candidate split is priced at the plan's ctx:
  // that favours the replica; the measured splits (replicaMeasured) compare like with like.
  // The floor is a coding rung (multibox models.*.coding_rung, the owner's coding-floor ladder, kept in our lab notes); null takes
  // the split model's own rung. The rung is per model, not per store: when the split's tool-argument fidelity (arg_values
  // EARf) is measured and a replica in the verdict row ran a store measured below it, fidelity_matched gives the verdict
  // again over replicas at the split's fidelity or better. The compared quantity is aggregate decode tok/s on both sides (the measured replica
  // points are decode aggregates); the split's agent duty-cycle useful tok/s is printed beside it, not compared.
  const mbRung = (data, m) => { const M = (((data.multibox || {}).models) || {})[m]; return M && M.coding_rung ? mbV(M.coding_rung) : null; };
  // tool-argument fidelity (arg_values EARf, the fidelity panels against one Q8_0 reference) of a multibox store, else of a
  // measured run's store type (settings.replica_type_earf); null when unmeasured
  const mbEarf = (data, model, store, types) => {
    const MB = data.multibox || {}, S = store && MB.models && MB.models[model] && MB.models[model].stores[store], f = S && S.fidelity && S.fidelity.arg_values_earf ? mbV(S.fidelity.arg_values_earf) : null;
    if (f !== null && f !== undefined) return f;
    const T = mbV((MB.settings || {}).replica_type_earf) || {}, ks = (types || []).filter(t => T[t] !== undefined);
    return ks.length ? Math.min(...ks.map(t => T[t])) : null;
  };
  const bestUpTo = (pts, n) => { let b = null; for (const p of pts) if (p[0] <= n && (!b || p[1] > b[1])) b = p; return b; };
  function replicaRecipes(data, card, floor, ref) {
    const MB = data.multibox, box = data.boxes[card.box], g = box && box.gpus.find(x => x.id === card.gpu), out = [];
    if (!g) return out;
    const cls = g.ceil ? g.ceil.class : null, Bs = mbV(MB.settings.batches), r0 = ref || {};
    for (const m of Object.keys(MB.models)) {
      const rung = mbRung(data, m);
      if (rung === null || rung < floor) continue;
      for (const st of Object.keys(MB.models[m].stores)) {
        const plan = pipelinePlan(data, { id: 'replica ' + card.box + ' ' + card.gpu, model: m, store: st, client: card.box, cards: [{ box: card.box, gpu: card.gpu }],
          ctx: r0.ctx || 0, streams: 1, kv_type: r0.kv_type, split: 'memory', transport: 'tcp', displace: r0.displace || [] });
        if (plan.status !== 'OK') continue;
        const cap = Number.isFinite(plan.capacity_streams) ? plan.capacity_streams : Infinity;
        out.push({ kind: 'model', id: m + ' ' + st + ' alone', model: m, store: st, rung, earf: mbEarf(data, m, st), memory: 'vram', tag: 'MODEL', capacity: cap,
          pts: Bs.filter(b => b <= cap).map(b => { const d = pipelineRoof(data, plan, b); return [b, d.aggregate_tps, d.calibration]; }) });
      }
    }
    for (const a of (((MB.card_curves || {}).anchors) || [])) {
      if (a.plan.cards.length !== 1) continue;
      const c = a.plan.cards[0], h = (((data.boxes[c.box] || {}).gpus) || []).find(x => x.id === c.gpu), rung = mbRung(data, a.plan.model);
      if (!(c.box === card.box && c.gpu === card.gpu) && !(h && h.ceil && cls && h.ceil.class === cls)) continue;
      if (rung === null || rung < floor) continue;
      out.push({ kind: 'anchor', id: a.id, model: a.plan.model, store: a.plan.store, rung, earf: mbEarf(data, a.plan.model, a.plan.store), memory: 'vram', tag: 'MEASURED', capacity: Infinity, pts: a.points.map(p => [p.B, p.tps, 'MEASURED']) });
    }
    const oneClass = cls && box.gpus.every(x => x.ceil && x.ceil.class === cls), free = (mbV(g.vram_gib) - mbV(MB.settings.card_reserve_gib)) * GiB;
    for (const C of (((data.curves || {}).configs) || [])) {
      if (!oneClass || C.box !== card.box || mbV(C.stages) !== 1) continue;
      const rung = mbRung(data, C.model);
      if (rung === null || rung < floor) continue;
      const pts = C.points.filter(p => p.used && p.tps && mbV(p.tps) > 0).map(p => [p.n, mbV(p.tps), 'MEASURED']);
      if (!pts.length) continue;
      const S = MB.models[C.model].stores, small = Math.min(...Object.keys(S).map(k => mbV(S[k].experts_bytes) + mbV(S[k].trunk_bytes) + mbV(S[k].head_bytes)));
      out.push({ kind: 'curve', id: C.id, label: C.label, model: C.model, store: (C.store_types || []).join('+') || null, rung, earf: mbEarf(data, C.model, null, C.store_types), memory: small <= free ? 'vram' : 'vram+dram', tag: 'MEASURED', capacity: Infinity, ctx: C.ctx, pts });
    }
    return out;
  }
  // the replica arm on a set of cards at the floor: per a (agents a card), each card's best recipe at a, capped DRAM replicas
  function replicaBaseline(data, cards, floor, opts) {
    const o = opts || {}, set = data.multibox.settings, grid = o.agents_per_card || mbV(set.replica_agents_per_card);
    const cap = o.dram_per_box !== undefined ? o.dram_per_box : mbV(set.replica_dram_per_box);
    // o.earf_min: only recipes with a measured tool-argument fidelity at or above it (the split's own)
    const recs = cards.map(c => ({ box: c.box, gpu: c.gpu, recipes: replicaRecipes(data, c, floor, o.ref).filter(r => o.earf_min === undefined || o.earf_min === null || (r.earf !== null && r.earf >= o.earf_min)) }));
    const at = (c, a, mem) => c.recipes.filter(r => !mem || r.memory === mem).map(r => { const p = bestUpTo(r.pts, a); return p ? { recipe: r.id, kind: r.kind, model: r.model, store: r.store, memory: r.memory, tag: r.tag, b: p[0], tps: p[1], curve: p[2] } : null; })
      .filter(Boolean).sort((x, y) => y.tps - x.tps || (y.tag === 'MEASURED') - (x.tag === 'MEASURED'))[0] || null;
    const rows = grid.map(a => {
      const pick = recs.map(c => ({ box: c.box, gpu: c.gpu, any: at(c, a), vram: at(c, a, 'vram'), capped: false }));
      const n = {}, lift = p => p.any.tps - (p.vram ? p.vram.tps : 0);
      pick.filter(p => p.any && p.any.memory !== 'vram').sort((x, y) => lift(y) - lift(x)).forEach(p => { n[p.box] = (n[p.box] || 0) + 1; if (cap !== null && cap !== undefined && n[p.box] > cap) p.capped = true; });
      const used = pick.map(p => Object.assign({ box: p.box, gpu: p.gpu, capped: p.capped }, (p.capped ? p.vram : p.any) || { recipe: null, tps: 0 }));
      used.forEach(u => { const r = u.recipe && recs.find(c => c.box === u.box && c.gpu === u.gpu).recipes.find(x => x.id === u.recipe); u.earf = r ? r.earf : null; });
      return { a, agents: a * cards.length, tps: used.reduce((s, u) => s + u.tps, 0), cards: used };
    });
    return { floor, rows, unpriced: recs.filter(c => !c.recipes.length).map(c => c.box + ' ' + c.gpu), priced: recs.filter(c => c.recipes.length).map(c => c.box + ' ' + c.gpu) };
  }
  // splitPts: [[B, aggregate tok/s, curve]] of the split inside its capacity; splitTag 'MEASURED' or 'MODEL'
  function replicaGate(data, spec, splitPts, opts) {
    const o = opts || {}, set = data.multibox.settings, N = spec.cards.length, own = mbRung(data, spec.model);
    const f = o.floor !== undefined ? o.floor : mbV(set.replica_floor), floor = f === null || f === undefined ? own : f;
    if (floor === null) return { verdict: 'UNRANKED', why: 'model ' + spec.model + ' has no coding_rung (data/multibox.json models) and settings.replica_floor is null' };
    // the verdict row: the top of the agents grid, or past it when the split's best point needs more agents than N x a_top
    // (a split measured only at 48 agents is judged at 48, its replicas at 48 / N a card)
    const grid = (o.agents_per_card || mbV(set.replica_agents_per_card)).slice(), peak = splitPts.reduce((b, p) => (!b || p[1] > b[1] ? p : b), null);
    if (peak && Math.ceil(peak[0] / N) > grid[grid.length - 1]) grid.push(Math.ceil(peak[0] / N));
    const earf = o.earf !== undefined ? o.earf : mbEarf(data, spec.model, spec.store, spec.store_types);
    const judge = (dram, earfMin) => {
      const base = replicaBaseline(data, spec.cards, floor, { ref: spec, agents_per_card: grid, dram_per_box: dram, earf_min: earfMin });
      const rows = base.rows.map(r => { const s = bestUpTo(splitPts, N * r.a);
        return { a: r.a, agents: r.agents, split_tps: s ? s[1] : null, split_B: s ? s[0] : null, split_curve: s ? s[2] : null, replica_tps: r.tps, gain: s && r.tps > 0 ? s[1] / r.tps - 1 : null, cards: r.cards }; });
      const top = rows[rows.length - 1], measured = o.split_tag === 'MEASURED' && top.cards.every(c => !c.recipe || c.tag === 'MEASURED');
      const margin = mbV(set.replica_gain_margin)[measured ? 'measured' : 'model'];
      let verdict;
      if (!splitPts.length) verdict = 'NO-SPLIT-POINT';
      else if (!base.priced.length) verdict = 'CAPACITY-REQUIRED';
      else verdict = top.split_tps > top.replica_tps * (1 + margin) ? 'GAINS' : 'CAPACITY-ONLY';
      return { verdict, floor, own_rung: own, margin, margin_basis: measured ? 'measured' : 'model', split_tps: top.split_tps, split_B: top.split_B, replica_tps: top.replica_tps,
        gain: top.gain, agents: top.agents, a: top.a, rows, unpriced: base.unpriced, partial: verdict === 'GAINS' && base.unpriced.length > 0,
        uncalibrated: top.split_curve === 'UNCALIBRATED' || top.cards.some(c => c.curve === 'UNCALIBRATED'), dram_capped: top.cards.filter(c => c.capped).map(c => c.box + ' ' + c.gpu) };
    };
    const out = judge(o.dram_per_box);
    // replicas sharing a box's DRAM: the verdict again with one DRAM replica a box (the alternate), or with the cap lifted
    const shared = Object.entries(out.rows[out.rows.length - 1].cards.filter(c => c.memory && c.memory !== 'vram').reduce((m, c) => { m[c.box] = (m[c.box] || 0) + 1; return m; }, {})).filter(([, n]) => n > 1).map(([b]) => b);
    out.dram_shared = shared;
    const alt = out.dram_capped.length ? judge(null) : shared.length ? judge(1) : null;
    if (alt) out.alternate = { dram_per_box: out.dram_capped.length ? null : 1, verdict: alt.verdict, replica_tps: alt.replica_tps, gain: alt.gain };
    // the rung is per model, not per store: when the split's tool-argument fidelity is measured and a replica in the
    // verdict row ran a store measured below it (or never measured), the verdict again over replicas at the split's
    // fidelity or better
    out.earf = earf;
    const top = out.rows[out.rows.length - 1].cards.filter(c => c.recipe);
    if (earf !== null && top.some(c => c.earf === null || c.earf < earf)) {
      const fm = judge(o.dram_per_box, earf);
      out.fidelity_matched = { earf_min: earf, verdict: fm.verdict, replica_tps: fm.replica_tps, gain: fm.gain, below: top.filter(c => c.earf === null || c.earf < earf).map(c => c.box + ' ' + c.gpu + ' ' + c.recipe + ' (EARf ' + (c.earf === null ? 'unmeasured' : c.earf) + ')') };
    }
    return out;
  }
  // a measured split ran with its cards free, so its replicas get the same cards free: every other service on them displaced
  const mbOthers = (data, cards) => [].concat(...cards.map(c => ((((data.boxes[c.box] || {}).gpus || []).find(x => x.id === c.gpu) || {}).other || []).map(x => x.service || x.what)));
  // every measured split (card_curves anchors with more than one card, data.curves configs with as many stages as their
  // box has cards) against its replicas: both sides measured where the replicas are; a modelled replica of an anchor is
  // priced at the anchor's measured depth (ctx_read) and KV type, as the split ran
  function replicaMeasured(data) {
    const MB = data.multibox || {}, out = [];
    for (const a of (((MB.card_curves || {}).anchors) || [])) {
      if (a.plan.cards.length < 2) continue;
      out.push(Object.assign({ id: a.id, label: a.arm, source: 'card_curves anchor', model: a.plan.model, store: a.plan.store, cards: a.plan.cards.map(c => c.box + ' ' + c.gpu) },
        replicaGate(data, Object.assign({}, a.plan, { ctx: a.ctx_read, displace: mbOthers(data, a.plan.cards) }), a.points.map(p => [p.B, p.tps, 'MEASURED']), { split_tag: 'MEASURED' })));
    }
    for (const C of (((data.curves || {}).configs) || [])) {
      const box = data.boxes[C.box], S = mbV(C.stages);
      if (!box || !(S > 1) || S !== box.gpus.length || !MB.models || !MB.models[C.model]) continue;
      const pts = C.points.filter(p => p.used && p.tps && mbV(p.tps) > 0).map(p => [p.n, mbV(p.tps), 'MEASURED']);
      if (!pts.length) continue;
      const spec = { model: C.model, store_types: C.store_types, cards: box.gpus.map(g => ({ box: C.box, gpu: g.id })), ctx: 0, displace: mbOthers(data, box.gpus.map(g => ({ box: C.box, gpu: g.id }))) };
      out.push(Object.assign({ id: C.id, label: C.label, source: 'data.curves config', model: C.model, store: null, cards: spec.cards.map(c => c.box + ' ' + c.gpu) }, replicaGate(data, spec, pts, { split_tag: 'MEASURED' })));
    }
    return out;
  }
  // rank plans by the gate, never by raw tok/s: GAINS by gain, then CAPACITY-REQUIRED by split aggregate, then
  // CAPACITY-ONLY by gain (the least lost first), then everything refused or unranked
  const REPLICA_ORDER = { GAINS: 0, 'CAPACITY-REQUIRED': 1, 'CAPACITY-ONLY': 2 };
  function replicaRank(xs, gateOf) {
    const g = x => gateOf(x) || {}, k = x => (g(x).verdict in REPLICA_ORDER ? REPLICA_ORDER[g(x).verdict] : 9);
    return xs.slice().sort((x, y) => k(x) - k(y) || (g(x).verdict === 'CAPACITY-REQUIRED' ? (g(y).split_tps || 0) - (g(x).split_tps || 0) : (g(y).gain === null || g(y).gain === undefined ? -Infinity : g(y).gain) - (g(x).gain === null || g(x).gain === undefined ? -Infinity : g(x).gain)));
  }
  // the serving mode's per-step term: a recipe served in a batch mode (XeStrata --batch N; opts.serving_mode {v: 'batch' |
  // 'single', tag, src} names it, else the query's flags are read) pays workload.engines.<engine>.batch_mode_step.stores.<store>
  // on every forward step, relative to the single-session decode the curve's one-row point stands for. A store with no row
  // has no measured term (R8's served batch-mode one-agent decode matches its curve), and a curve measured in the batch mode
  // itself (decode_curve.serving_mode 'batch') already carries it: neither is charged
  function batchModeStep(data, q, o) {
    const E = (((data.workload || {}).engines) || {})[q.engine] || {}, T = E.batch_mode_step ? v(E.batch_mode_step) : null;
    const flagBatch = /(^|[\s;])--batch\s+\d+/.test(q.flags || '');
    const mode = o.serving_mode ? v(o.serving_mode) : flagBatch ? 'batch' : 'single', modeSrc = o.serving_mode ? o.serving_mode.src : flagBatch ? 'READ: --batch in the query flags' : 'READ: no --batch in the query flags';
    const row = T && T.stores ? T.stores[q.store] || null : null;
    if (mode !== 'batch') return { applied: false, mode, mode_src: modeSrc, why: 'served single-session' };
    if (o.decode_curve && o.decode_curve.serving_mode === 'batch') return { applied: false, mode, mode_src: modeSrc, why: 'the decode curve was measured in the batch mode' };
    if (!row) return { applied: false, mode, mode_src: modeSrc, why: 'no measured batch-mode step term for ' + q.engine + ' store ' + q.store };
    return { applied: true, mode, mode_src: modeSrc, term: row };
  }
  // ---------- the agent duty cycle on the batch landscape ----------
  // B real agents do not all decode at once: each loops its recorded turns (trace) and waits its tool gap after each, and
  // reads its fresh prompt tokens before it decodes. agentLandscape replays the trace (agentReplaySim, the existing turn model)
  // at every B with the landscape's own rates: decode at B_eff, the mean number of requests decoding at once (a fixed point:
  // the decode rate at B_eff sets the turn times that set B_eff), capped at the server's admitted requests (slots: past
  // them requests queue, they do not decode), the landscape's prefill rate, and a fresh-token rate F(B, cache) = the trace's
  // prompt growth + the server's re-read a turn: the measured re-read of the server's one-agent row diluted over B agents (a
  // trace-driven re-read belongs to one recorded agent) plus, once B passes the conversations the cache holds (C), a
  // whole-prompt re-read for the share 1 - C / B of turns whose conversation was evicted. duty = B_eff / B.
  // opts: { trace, server, model, slots {v,tag,src}, cache_conversations {v,tag,src}, warmup_s, window_s, batches, serving_mode {v,tag,src} }
  const agentDepthMean = (A, tok) => {
    let n = 0, sum = 0;
    A.forEach((a, i) => { let d = 0; a.turns.forEach(t => { d += t[2]; sum += d * tok[i % tok.length]; n++; }); });
    return n ? sum / n : 0;
  };
  function agentLandscape(data, q, opts) {
    let o = opts || {};
    const TM = (data.workload || {}).turn_model, W = data.workload;
    const un = why => ({ status: 'UNMODELLED', why });
    if (!TM) return un('no data.workload.turn_model');
    const trace = o.trace || v(W.profiles[v(W.settings.default_profile)].turn_trace), server = o.server, model = o.model || q.model;
    if (!TM.traces[trace]) return un('trace ' + trace + ' has no per-turn shape');
    if (!server || !TM.servers[server]) return un('no turn_model.servers row ' + server + ' (the server\'s prefix re-read)');
    if (!TM.models[model]) return un('model ' + model + ' has no turn_model.models row');
    // opts.rates { prefill_tps, decode_curve, capacity_streams } (a multi-box plan, pipelineLandscape): the rates come from the
    // caller, not from a one-box roofline query
    const R = o.rates || null, Bs = o.batches || [1, 2, 4, 8, 16, 32, 64, 128];
    const L = R ? { prefill_tps: v(R.prefill_tps), rows: Bs.map(B => ({ B, status: 'OK', useful_tps: null, per_stream_tps: null })) } : batchLandscape(data, q, Object.assign({}, o, { batches: Bs }));
    if (!L.prefill_tps) return un('no prefill rate: ' + (L.prefill_why || 'unmodelled'));
    if (R) o = Object.assign({}, o, { decode_curve: R.decode_curve });
    const df = R ? { status: 'NONE' } : decodeFit(data, q.engine), eta = df.status === 'FIT' ? df.eta : 1;
    // opts.decode_curve {v: [[n, aggregate tok/s], ...], tag, src}: a measured synthetic decode curve of the served build replaces the
    // roofline decode (log-log between neighbouring points, held flat past the last): the roofline's one decode eta is fitted at B 1
    const curve = o.decode_curve ? v(o.decode_curve).slice().sort((a, b) => a[0] - b[0]) : null;
    const curveAt = n => { const c = curve, i = c.findIndex(p => p[0] >= n); if (i === 0) return c[0][1] * n / c[0][0]; if (i < 0) return c[c.length - 1][1];
      const [a, b] = [c[i - 1], c[i]]; return Math.exp(Math.log(a[1]) + (Math.log(b[1]) - Math.log(a[1])) * (Math.log(n) - Math.log(a[0])) / (Math.log(b[0]) - Math.log(a[0]))); };
    // opts.decode_step_increment {v: lambda, tag, src}: the batched part of the curve's step scaled, the one-row step kept: a step of
    // n rows = s1 + lambda x (n / curve(n) - s1), s1 = 1 / curve(1) (the served build's per-row cost against the curve's build,
    // whose one-agent decode matched the served one: 68.8 vs 70.6 MEASURED). Solved from a named calibration point
    // (agentPointRescore), never from the point it is scored on
    const lam = o.decode_step_increment ? v(o.decode_step_increment) : 1;
    const inc = agg => n => { if (lam === 1 || n <= 1) return agg(n); const s1 = 1 / agg(1); return n / (s1 + lam * (n / agg(n) - s1)); };
    const curveAgg = inc(curveAt);
    const decRaw = n => { if (R) { const agg = curveAgg(n); return { per: agg / n, agg, fits: !v(R.capacity_streams) || n <= v(R.capacity_streams) }; } const d = decodeRoof(data, Object.assign({}, q, { batch: n })); if (d.status !== 'OK') return null; const step = d.step_s / eta, fits = d.capacity_streams === null || n <= d.capacity_streams;
      if (curve) { const agg = curveAgg(n); return { per: agg / n, agg, fits }; } return { per: d.draft_mult / step, agg: n * d.draft_mult / step, fits }; };
    // the serving mode's step term (batchModeStep): served in a batch mode, every forward step pays the mode's fixed extra
    // (m - 1) x the one-row step, so a lone stream steps m times slower and a step of n rows takes n / agg(n) + (m - 1) / agg(1)
    const BM = batchModeStep(data, q, o), bmx = BM.applied ? v(BM.term) : 1;
    const decAt = n => { const d = decRaw(n); if (!d || bmx === 1) return d; const d1 = decRaw(1); if (!d1) return null; const agg = n / (n / d.agg + (bmx - 1) / d1.agg); return { per: agg / n, agg, fits: d.fits }; };
    const one = decAt(1);
    if (!one) return un('no decode bound at B 1');
    const st = o.stall !== undefined ? o.stall : (W.engines[q.engine] || {}).reads_stall_decode, stall = st ? !!v(st) : false;
    const slots = o.slots ? v(o.slots) : Infinity, C = o.cache_conversations ? v(o.cache_conversations) : null;
    const S = TM.servers[server], r1 = byAgents(S.reread_per_turn, 1), A = v(TM.traces[trace]);
    const tokR = TM.tokenizers[TM.models[model].tokenizer], base = tokR && (tokR.relative_to ? TM.tokenizers[tokR.relative_to] : tokR);
    const tok = base ? v(base.per_agent).map(x => x * (tokR.relative_to ? v(tokR.ratio) : 1)) : A.map(() => 1);
    const depth = agentDepthMean(A, tok);
    const tag = (x, t, src) => ({ v: x, tag: t, src });
    // opts.admission {v: {policy: 'engine' | 'door', conversations, idle_release_s}, tag, src}: how requests past the slots are
    // admitted. Declared, the served engine runs turn by turn (agentEngineSim: slot eviction, reads blocking decode, the door);
    // a point past the slots with none declared is REFUSED: the fluid model below would price B > slots as a queue that
    // batches, which the R8 sweep falsified (B 33: 33 conversations on 32 slots re-read 39.5 tokens per output token)
    const adm = o.admission || null, E = W.engines[q.engine] || {};
    const readMax = o.read_batch_max || E.read_batch_max || null, sharedP = o.shared_prefix || null;
    const rows = L.rows.map(r => {
      if (r.status !== 'OK') return { B: r.B, status: r.status, why: r.why };
      const B = r.B, cap = Math.min(B, slots);
      if (B > slots && !adm) return { B, status: 'REFUSED', why: B + ' agents > ' + slots + ' slots and no admission policy declared (opts.admission: engine = the engine\'s own slot LRU, door = a front door capped at N conversations)' };
      const evict = adm || C === null ? 0 : Math.max(0, 1 - C / B), reread = (r1 ? v(r1.row) / B : 0) + evict * depth;
      const qq = { trace, server, model, n: B, prefill: tag(L.prefill_tps, 'MODEL', 'the landscape prefill rate'), stall: tag(stall, 'READ', 'workload.engines.' + q.engine + '.reads_stall_decode'),
        decode_1: tag(one.per, 'MODEL', 'roofline decode at B 1 / decode eta'), warmup_s: o.warmup_s !== undefined ? o.warmup_s : v(TM.settings.plan_warmup_s), window_s: o.window_s !== undefined ? o.window_s : v(TM.settings.plan_window_s),
        reread_per_turn: tag(reread, 'INFERRED', 'INFERRED: the ' + server + ' one-agent re-read ' + (r1 ? v(r1.row) : 0) + ' tokens a turn / B ' + B + (evict ? ' + evicted share ' + evict.toFixed(3) + ' (1 - C ' + C + ' / B) x mean prompt depth ' + depth.toFixed(0) : '')) };
      const memo = {}, aggAt = k => { if (!memo[k]) { const d = decAt(k); memo[k] = d ? d.agg : one.agg; } return memo[k]; };
      qq.decode_n = tag(decAt(cap) ? decAt(cap).agg : one.agg, 'MODEL', 'roofline aggregate decode at the running count (decode_at), capped at ' + cap);
      Object.assign(qq, { decode_at: aggAt, slots: slots === Infinity ? null : slots });
      if (adm) Object.assign(qq, { admission: adm, read_batch_max: readMax, shared_prefix: sharedP, slot_groups: o.slot_groups || null, read_blocks: o.read_blocks || null });
      const sim = agentTurnPredict(data, qq, [], 'all');
      if (sim.status !== 'MODEL') return { B, status: sim.status, why: sim.why };
      const beff = Math.min(cap, sim.decoding_at_once), d = decAt(Math.max(1, beff));
      return { B, status: 'OK', useful_tps: sim.useful_tps, useful_per_agent: sim.useful_tps / B, b_eff: beff, duty: beff / B, queued: B > slots,
        per_stream_at_beff: d.per, aggregate_at_beff: d.agg, fits_vram_at_beff: d.fits, F: sim.fresh_per_output, reread_per_turn: reread, evicted_share: sim.engine ? sim.engine.evicted_share : evict,
        out_per_turn: sim.out_per_turn, turns_per_agent_min: sim.turns_per_agent_min, share: sim.share, landscape_useful: r.useful_tps, landscape_per_stream_at_B: r.per_stream_tps,
        engine: sim.engine || null };
    });
    const ok = rows.filter(r => r.status === 'OK'), best = ok.slice().sort((a, b) => b.useful_tps - a.useful_tps)[0] || null;
    return { status: 'OK', query: q, trace, server, model, decode_basis: (curve ? 'curve: ' + o.decode_curve.src : 'roofline decode at the fitted decode eta') + (o.decode_step_increment ? '; batched step increment x ' + lam.toFixed(3) + ' (' + o.decode_step_increment.src + ')' : '') + (BM.applied ? '; ' + BM.mode + ' mode step x ' + bmx.toFixed(3) + ' (' + BM.term.src + ')' : ''), batch_mode: BM, prefill_tps: L.prefill_tps, decode_eta: df.status === 'FIT' ? df.eta : null, slots: o.slots || null, cache_conversations: o.cache_conversations || null,
      mean_prompt_depth: depth, one_agent_reread: r1 ? r1.row : null, stall, admission: adm, read_batch_max: readMax, shared_prefix: sharedP, slot_groups: o.slot_groups || null, best: best ? { B: best.B, useful_tps: best.useful_tps, b_eff: best.b_eff } : null, rows };
  }
  // the R8 agent points (card_curves.agent_points) rescored with today's served-engine sim and the frozen records' own inputs
  // (the record named by opts_from: query, trace, server, slots, the MEASURED synthetic decode curve, the prefill query), plus
  // the point's declared overrides. Nothing is refitted: every point is a hold-out. err = predicted / measured - 1
  // A measured agent point names its counting rule: measured.turn_counted_if 'ended_by_window_end' (a turn counts only if it
  // ended by the window's end). A point without it is UNVERIFIED-COUNT, and so is a lambda solved on one: the R8 sweep and
  // r8-admission counted turns that ended up to 149 s past the window (homelab conv-door-live, 2026-11-11), 85.3 for 66.8
  const AGENT_COUNT_RULE = 'ended_by_window_end';
  const agentPointCount = x => ((x && x.measured) || {}).turn_counted_if === AGENT_COUNT_RULE ? 'VERIFIED' : 'UNVERIFIED-COUNT';
  function agentPointRescore(data) {
    const AP = (((data.multibox || {}).card_curves) || {}).agent_points, P = (data.roofline || {}).predictions || [];
    if (!AP) return { status: 'UNMODELLED', why: 'no card_curves.agent_points' };
    const run = (x, ov) => {
      const rec = P.find(r => r.id === x.opts_from);
      if (!rec) return { L: { why: 'no prediction record ' + x.opts_from }, r: null };
      const L = agentLandscape(data, rec.query, Object.assign({}, rec.agents, x.override || {}, ov || {}, { batches: [x.B] }));
      return { L, r: L.status === 'OK' ? L.rows[0] : null };
    };
    // the step-increment arm: one factor on the batched part of the frozen curve's step (decode_step_increment), solved (bisection)
    // so the named calibration point's useful rate is its measured one; that point is marked calibrates, every other a hold-out
    const cal = AP.calibration ? AP.points.find(x => x.B === AP.calibration.B && x.kind === AP.calibration.kind) : null;
    let kappa = null;
    if (cal) {
      const u = k => { const z = run(cal, { decode_step_increment: { v: k, tag: 'FIT', src: 't' } }).r; return z && z.status === 'OK' ? z.useful_tps : NaN; };
      let lo = 0.5, hi = 10;
      // the sim counts whole turns, so useful(lambda) is a staircase: the bisection closes on a step edge, and the target may
      // sit inside the step's jump (no-door B 8: 61.03 just below lambda 2.7663, 56.82 just above; the recounted 57.54 is
      // between). Of the two sides, the one nearer the target is lambda; its residual is reported (cal_residual), never hidden
      const T = cal.measured.useful_tps;
      if (u(lo) >= T && u(hi) <= T) { for (let i = 0; i < 40; i++) { const mid = (lo + hi) / 2; if (u(mid) > T) lo = mid; else hi = mid; } kappa = Math.abs(u(lo) - T) < Math.abs(u(hi) - T) ? lo : hi; }
    }
    const calCount = cal ? agentPointCount(cal) : null;
    const kTag = kappa === null ? null : { v: kappa, tag: 'FIT', count: calCount, cal_residual: run(cal, { decode_step_increment: { v: kappa, tag: 'FIT', src: 't' } }).r.useful_tps / cal.measured.useful_tps - 1, src: (calCount === 'VERIFIED' ? '' : 'UNVERIFIED-COUNT (the calibration point names no turn_counted_if ' + AGENT_COUNT_RULE + '): ') + 'FIT on ONE named calibration point: the R8 ' + cal.kind + ' B ' + cal.B + ' useful ' + cal.measured.useful_tps + ' tok/s (' + AP.calibration.why + ')' };
    const rows = AP.points.map(x => {
      const { L, r } = run(x), k = kTag ? run(x, { decode_step_increment: kTag }).r : null;
      const label = x.label || x.kind + ' B ' + x.B, count = agentPointCount(x);
      if (!r || r.status !== 'OK') return { B: x.B, kind: x.kind, label, count, status: 'UNMODELLED', why: L.why || (r && r.why) };
      const m = x.measured, e = r.engine || {}, ev = typeof x.evict_branch_useful === 'number' ? x.evict_branch_useful : null;
      const use = x === cal ? 'calibrates' : x.calibrates && x.calibrates.length ? 'calibrates ' + x.calibrates.join(', ') : 'holdout';
      return { B: x.B, kind: x.kind, label, set: x.set || 'r8', count, status: 'OK', measured: m, calibrates: x.calibrates || [], batch_mode: L.batch_mode, predicted: { useful_tps: r.useful_tps, turns_per_agent_min: r.turns_per_agent_min, b_eff: r.b_eff, F: r.F, stall_pct: e.stall_share !== undefined ? 100 * e.stall_share : null, read_busy_pct: e.read_busy_share !== undefined ? 100 * e.read_busy_share : null, evicted_share: r.evicted_share },
        err: r.useful_tps / m.useful_tps - 1, evict_branch_useful: ev, evict_branch_err: ev === null ? null : ev / m.useful_tps - 1, decode_basis: L.decode_basis, src: x.src,
        scaled: k && k.status === 'OK' ? { useful_tps: k.useful_tps, err: k.useful_tps / m.useful_tps - 1, turns_per_agent_min: k.turns_per_agent_min, turns_err: m.turns_per_agent_min ? k.turns_per_agent_min / m.turns_per_agent_min - 1 : null, b_eff: k.b_eff, F: k.F, per_stream_at_beff: k.per_stream_at_beff, stall_pct: k.engine ? 100 * k.engine.stall_share : null, evicted_share: k.evicted_share, use } : null };
    });
    const ok = rows.filter(r => r.status === 'OK'), med = xs => { const a = xs.slice().sort((p, q) => p - q), n = a.length; return n ? (n % 2 ? a[(n - 1) / 2] : (a[n / 2 - 1] + a[n / 2]) / 2) : null; };
    const ho = ok.filter(r => r.scaled && r.scaled.use === 'holdout'), ev = ok.filter(r => r.evict_branch_err !== null);
    // per set (points.set; r8 when unnamed): the step-increment arm's hold-out median and the calibration points of each set
    const sets = {};
    ok.forEach(r => { const S = sets[r.set] = sets[r.set] || { n: 0, holdout_n: 0, holdout_median_abs_pct: null, calibrates: [] }; S.n++; if (r.scaled && r.scaled.use !== 'holdout') S.calibrates.push(r.label); });
    Object.keys(sets).forEach(k => { const h = ho.filter(r => r.set === k); sets[k].holdout_n = h.length; sets[k].holdout_median_abs_pct = h.length ? 100 * med(h.map(r => Math.abs(r.scaled.err))) : null; });
    return { status: 'OK', rows, sets, median_abs_pct: ok.length ? 100 * med(ok.map(r => Math.abs(r.err))) : null, evict_branch_median_abs_pct: ev.length ? 100 * med(ev.map(r => Math.abs(r.evict_branch_err))) : null, evict_branch_n: ev.length,
      decode_step_increment: kTag, scaled_holdout_median_abs_pct: ho.length ? 100 * med(ho.map(r => Math.abs(r.scaled.err))) : null, scaled_holdout_n: ho.length,
      count_rule: AGENT_COUNT_RULE, unverified_count: rows.filter(r => r.count !== 'VERIFIED').map(r => r.label) };
  }
  // the engine's eta: the factor that makes the predicted prefill time of its fit cells equal their measured time (bisection)
  const rlMeasuredS = c => c.measured.ttft_s !== undefined ? c.measured.ttft_s : c.query.tokens / c.measured.prefill_tps;
  // the engine's eta: one factor on every tensor term, so the predicted prefill time of its fit cells equals their measured
  // time (bisection). Nothing else is refitted: a later run that misses is scored term by term (rooflineAttribute), not absorbed
  // device 'cpu' fits the CPU eta of the engine (data.roofline.cpu_fits: one factor on every CPU compute term, cells of one cpu
  // path) with the engine's GPU eta held at its own fit (or 1); the default fits the GPU eta (data.roofline.fits)
  function rooflineFit(data, engine, device) {
    const cpuSide = device === 'cpu', key = cpuSide ? 'cpu_fits' : 'fits', F = ((data.roofline || {})[key] || {})[engine];
    if (!F) return { status: 'UNMODELLED', why: 'no roofline ' + (cpuSide ? 'CPU ' : '') + 'fit cells for ' + engine + ' (data.roofline.' + key + ')' };
    const cells = F.cells.map(id => data.roofline.checks.find(c => c.id === id));
    if (cells.some(c => !c)) return { status: 'UNMODELLED', why: 'fit cell missing: ' + F.cells.join(', ') };
    const want = cells.reduce((a, c) => a + rlMeasuredS(c), 0);
    const g = cpuSide ? rooflineFit(data, engine) : null, gEta = g && g.status === 'FIT' ? g.eta : 1;
    const at = e => cells.reduce((a, c) => { const p = prefillRoof(data, c.query, cpuSide ? { gpu: gEta, cpu: e } : e); return a + (p.status === 'OK' ? p.time_s : NaN); }, 0);
    if (!isFinite(at(1))) return { status: 'UNMODELLED', why: 'a fit cell has no bound: ' + cells.map(c => prefillRoof(data, c.query, 1).why).filter(Boolean)[0] };
    if (at(1) > want) return { status: 'UNMODELLED', why: 'the fit cells ran faster than the bound at eta 1 (' + at(1).toFixed(2) + ' s > ' + want.toFixed(2) + ' s measured)' };
    let lo = 1e-5, hi = 1;
    for (let i = 0; i < 80; i++) { const mid = Math.sqrt(lo * hi); if (at(mid) > want) lo = mid; else hi = mid; }
    return { status: 'FIT', eta: hi, device: cpuSide ? 'cpu' : 'gpu', path: F.path || null, cells: F.cells, measured_s: want, tag: 'FIT', src: F.src };
  }
  // prefill + decode (batch 1 and batch N) of one query: bound (eta 1) and prediction (fitted eta), each naming its binding term
  function phaseBound(data, q, opts) {
    const o = opts || {}, et = rlEta(data, q.engine), f = et.fit, nb = o.batch_n || q.batch_n || 16;
    const pb = q.tokens ? prefillRoof(data, q, 1) : { status: 'UNMODELLED', why: 'no prompt tokens given' };
    // the prediction applies the eta of the side that runs: a CPU placement needs the CPU fit, a GPU one the GPU fit
    const onCpu = (q.placement || 'gpu') !== 'gpu', has = onCpu ? et.cpu !== null : et.gpu !== null;
    const pp = q.tokens && has ? prefillRoof(data, q, rlEtaFor(data, q)) : null;
    const d1 = decodeRoof(data, Object.assign({}, q, { batch: 1 })), dn = decodeRoof(data, Object.assign({}, q, { batch: nb }));
    return { query: q, fit: f, cpu_fit: et.cpu_fit, eta: { gpu: et.gpu, cpu: et.cpu }, prefill: { bound: pb, predicted: pp }, decode: { b1: d1, bN: dn, n: nb } };
  }
  // the worked checks of data.roofline.checks: predicted vs measured and the binding term at the bound and at the fitted eta
  function rooflineChecks(data) {
    return ((data.roofline || {}).checks || []).map(c => {
      if (c.phase === 'decode') {
        const d = decodeRoof(data, c.query), m = c.measured.decode_tps;
        return { id: c.id, role: c.role, phase: 'decode', status: d.status, why: d.why || null, measured: m, bound_tps: d.status === 'OK' ? d.tps_per_stream : null,
          bound_over_measured: d.status === 'OK' ? d.tps_per_stream / m : null, binding: d.binding || null, roof: d, tag: c.tag, src: c.src };
      }
      // the fit that prices the check: the CPU fit for a CPU placement, or for GPU-dense CPU-expert cells when the GPU side has none
      // (its GPU terms then run at eta 1)
      const et = rlEta(data, c.query.engine), cpuSide = (c.query.placement || 'gpu') !== 'gpu' || (c.query.host_path === 'cpu' && et.gpu === null);
      const f = cpuSide ? et.cpu_fit : et.fit, b = prefillRoof(data, c.query, 1), p = f.status === 'FIT' ? prefillRoof(data, c.query, et) : null;
      const ms = rlMeasuredS(c), mt = c.query.tokens / ms, rec = rlRecordOf(data, c.id);
      const evidence = c.role === 'fit' ? 'fit cell (not a test)' : rec && rlAfterRecord(c.measured_utc, rec.written_utc).ok ? 'pre-registered ' + rec.id : 'post hoc: no prediction was written before this run, so it is not evidence' + (c.confounds ? '; confounds: ' + c.confounds.join('; ') : '');
      return { id: c.id, role: c.role, phase: 'prefill', status: b.status, why: b.why || null, tokens: c.query.tokens, unlock: b.unlock ? b.unlock[0] : null, measured_s: ms, measured_tps: mt,
        bound_tps: b.status === 'OK' ? b.tps : null, bound_binding: b.binding || null, pred_tps: p && p.status === 'OK' ? p.tps : null, pred_s: p && p.status === 'OK' ? p.time_s : null,
        err: p && p.status === 'OK' ? p.tps / mt - 1 : null, binding: p && p.binding ? p.binding : null, eta: f.status === 'FIT' ? f.eta : null, eta_side: cpuSide ? 'cpu' : 'gpu', evidence,
        attribution: c.role !== 'fit' && f.status === 'FIT' && p && p.status === 'OK' ? rooflineAttribute(data, c.query, ms, et) : null, fit_why: f.why || null, roof: p || b, tag: c.tag, src: c.src };
    });
  }
  // ---------- roofline: explain, rearrange, and score a run term by term ----------
  // Post-run scoring never refits: for each term of the stated prediction it finds the factor k that term alone would need for
  // the prediction to equal the measurement, and judges k against physics (a compute term may not beat its ceiling, a link or
  // memory term may not beat its spec) and against the fit cells (k far from 1 on every term = an unmodelled term: a confound or
  // a missing piece of the dataflow, named, not absorbed)
  // the PCIe link the experts stream over: its rate in the model and the slot's spec (null when the query streams over none)
  function rlLinkSpec(data, q) {
    const I = rlInputs(data, q);
    if (I.why) return null;
    const g0 = I.st ? I.st.stages[0].g : null, link = g0 ? rlLink(data, g0, I.eng) : null, geo = g0 ? v(g0.pcie || {}) || {} : {};
    const spec = geo.gen ? v(data.roofline.link.pcie_gbs_per_lane)[geo.max_gen || geo.gen] * (geo.max_width || geo.width) : null;
    return link && spec ? { gbs: link.gbs, spec_gbs: spec } : null;
  }
  // the attribution itself, on any time function at(term, k): the prediction's terms, the etas and link the k is judged against
  function rlAttribute(timeS, terms, at, measuredS, et, lk, basis) {
    const rows = terms.filter(t => t.s > 0).map(t => {
      const want = measuredS, lo0 = at(t.id, 1e-3), hi0 = at(t.id, 1e3);
      if (!(want >= lo0 && want <= hi0)) return { term: t.id, k: null, verdict: 'cannot explain the miss alone (the other side of the overlap binds)' };
      let lo = 1e-3, hi = 1e3;
      for (let i = 0; i < 70; i++) { const mid = Math.sqrt(lo * hi); if (at(t.id, mid) < want) lo = mid; else hi = mid; }
      const k = Math.sqrt(lo * hi);
      let phys = 'possible';
      const te = /^compute\.cpu_/.test(t.id) ? et.cpu : et.gpu;
      if (/^compute/.test(t.id) && te / k > 1) phys = 'IMPOSSIBLE: needs ' + (100 * te / k).toFixed(0) + ' % of the ' + (/^compute\.cpu_/.test(t.id) ? 'CPU' : 'tensor') + ' ceiling';
      if (t.id === 'stream.experts' && lk && lk.gbs / k > lk.spec_gbs) phys = 'IMPOSSIBLE: needs ' + (lk.gbs / k).toFixed(1) + ' GB/s over a ' + lk.spec_gbs.toFixed(1) + ' GB/s PCIe link';
      return { term: t.id, k, verdict: phys + (Math.abs(Math.log(k)) > Math.log(1.5) ? '; ' + (k > 1 ? 'slower' : 'faster') + ' than the fit cells by x' + (k > 1 ? k : 1 / k).toFixed(2) : '; within x1.5 of the fit') };
    });
    const ok = rows.filter(r => r.k !== null && !/IMPOSSIBLE/.test(r.verdict) && Math.abs(Math.log(r.k)) <= Math.log(1.5));
    return { status: 'OK', basis, predicted_s: timeS, measured_s: measuredS, miss_s: measuredS - timeS, ratio: measuredS / timeS, terms: rows,
      verdict: Math.abs(measuredS / timeS - 1) <= 0.1 ? 'prediction holds within 10 %' : ok.length ? 'one modelled term explains it within x1.5: ' + ok.map(r => r.term + ' x' + r.k.toFixed(2)).join(', ')
        : 'no single modelled term explains it within x1.5 of the fit cells: the run took ' + Math.abs(measuredS - timeS).toFixed(2) + ' s ' + (measuredS > timeS ? 'more' : 'less') + ' than predicted, an unmodelled term (a confound or a missing piece of the dataflow); the fit is NOT re-run on this cell' };
  }
  // attribution against the CURRENT model: the query re-run on the data as built now (a check cell, or a record's
  // attribution_current beside its frozen one)
  function rooflineAttribute(data, q, measuredS, eta, phase) {
    const P = phase === 'decode' ? null : prefillRoof(data, q, eta);
    if (!P || P.status !== 'OK') return { status: 'UNMODELLED', why: P ? P.why : 'decode attribution: use the decode terms (step seconds) directly' };
    const at = (id, k) => prefillRoof(data, Object.assign({}, q, { scale: Object.assign({}, q.scale || {}, { [id]: k }) }), eta).time_s;
    return rlAttribute(P.time_s, P.terms, at, measuredS, rlEtaPair(eta), rlLinkSpec(data, q), 'current model (the data as built now, not the record)');
  }
  // attribution against the record as predicted: the per-chunk terms it stored (predicted.composition; a record written before
  // compositions were stored may carry composition_backfill, made at its own rm_commit), its etas and its link. No data is read,
  // so nothing changed in the data after the record can move it
  function rooflineAttributeFrozen(rec, measuredS) {
    const pr = rec.predicted || {}, bf = rec.composition_backfill || null, comp = pr.composition || (bf && bf.composition) || null;
    if (!comp || !Array.isArray(comp.groups) || !Array.isArray(pr.terms) || !(pr.time_s > 0))
      return { status: 'UNFROZEN', why: 'the record stores no per-chunk composition of its prediction, so it cannot be attributed as predicted (only against the current model); a new prediction record stores one' };
    const t0 = rlComposedTime(comp, null, 1);
    if (!(Math.abs(t0 / pr.time_s - 1) < 1e-9)) return { status: 'UNFROZEN', why: 'the record composition sums to ' + t0 + ' s, not its predicted ' + pr.time_s + ' s' };
    const basis = 'frozen: the record\'s own per-chunk terms (' + comp.rule + ')' + (pr.composition ? '' : ', backfilled at RM ' + String(bf.from_rm_commit || '?').slice(0, 7));
    return rlAttribute(pr.time_s, pr.terms, (id, k) => rlComposedTime(comp, id, k), measuredS, rlEtaPair({ gpu: pr.eta || 1, cpu: pr.eta_cpu || null }), pr.link || (bf && bf.link) || null, basis);
  }
  // candidate rearrangements of a query (each a variant query + what it changes and which term it targets); needs_code marks a
  // change the engine cannot make today
  function rlCandidates(data, q, base) {
    const I = rlInputs(data, q);
    if (I.why) return [];
    const out = [], P = I.eng.prefill, g = I.st ? I.st.stages[0].g : null, onGpu = I.pl.name === 'gpu', C = base.prefill && base.prefill.status === 'OK' ? base.prefill.chunk : P.chunk_default;
    const add = (id, change, targets, dq, needs, cond) => out.push({ id, change, targets, query: Object.assign({}, q, dq), needs_code: needs || null, condition: cond || null });
    if (C) { add('chunk_x2', 'prefill chunk ' + C + ' -> ' + 2 * C, onGpu ? 'stream.experts' : 'read.cpu_weights', { chunk: 2 * C }, null, onGpu ? 'the bigger chunk\'s activations must fit VRAM (the engine plan halves a chunk that does not)' : 'the bigger ubatch\'s activations must fit RAM');
      add('chunk_half', 'prefill chunk ' + C + ' -> ' + Math.max(256, C / 2), 'compute (smaller chunk, more passes)', { chunk: Math.max(256, C / 2) }); }
    if (g && g.ceil && g.ceil.unlock) { const now = unlockState(g, q), other = g.ceil.unlock.states.find(x => x !== now); add('unlock_' + other, 'card firmware ' + now + ' -> ' + other, 'compute.trunk', { unlock: other }); }
    if (P.stream === 'all' && q.resident_frac !== undefined) {
      const frac = q.resident_frac;
      add('keep_banks_in_prefill', 'keep the decode expert banks (' + (100 * frac).toFixed(0) + ' % of experts) resident through prefill: stream only the rest', 'stream.experts', { resident_bytes: frac * v(I.shape.expert_bytes) }, 'tqLLM releases every bank for a prefill (fn_engine_prefill.cpp borrow): needs a prefill that skips resident experts');
    }
    if (P.pass === 'prefill_pass') [0, 0.25, 0.5].forEach(sh => { if (Math.abs(sh - (base.prefill && base.prefill.status === 'OK' ? I.st.cpu_share : 0)) > 1e-9) add('cpu_share_' + sh, 'STRATA_PREFILL_STREAM_CPU_SHARE -> ' + sh, 'stream.experts', { flags: String(q.flags || '').replace(/STRATA_PREFILL_STREAM_CPU_SHARE=[0-9.]+/, '') + '; STRATA_PREFILL_STREAM_CPU_SHARE=' + sh }); });
    if (q.resident_frac !== undefined && q.resident_frac < 1) add('more_resident', 'expert slots ' + (100 * q.resident_frac).toFixed(0) + ' % -> ' + (100 * Math.min(1, q.resident_frac * 1.5)).toFixed(0) + ' % of experts (x1.5 banks)', 'read.misses', { resident_frac: Math.min(1, q.resident_frac * 1.5) }, null, 'the extra banks must fit VRAM beside the KV of every stream (the landscape capacity)');
    if (onGpu && P.stream === 'offload') ['cpu', 'stream'].filter(h => h !== q.host_path).forEach(h => add('host_' + h, 'host experts ' + (q.host_path || 'stream') + ' -> ' + h, h === 'cpu' ? 'stream.experts' : 'compute.cpu_experts', { host_path: h }));
    return out;
  }
  // score one query for the search: prefill at the read size (fitted eta), decode at B agents (fitted decode efficiency), and the
  // agent mix useful = 1 / (1 / decode aggregate + F / P)
  function rlScore(data, q, o) {
    const pf = rlEta(data, q.engine), df = decodeFit(data, q.engine), B = o.agents || 1;
    const pq = Object.assign({}, q, o.prefill_query || {}, { tokens: (o.prefill_query || {}).tokens || q.read_tokens || q.tokens || 2048 });
    const p = prefillRoof(data, pq, rlEtaFor(data, pq)), d = decodeRoof(data, Object.assign({}, q, { batch: B }));
    if (p.status !== 'OK' || d.status !== 'OK') return { status: 'UNMODELLED', why: p.why || d.why };
    const step = df.status === 'FIT' ? d.step_s / df.eta : d.step_s, agg = B * d.draft_mult / step, F = o.fresh;
    const useful = F !== null && F !== undefined ? 1 / (1 / agg + F / p.tps) : null;
    return { status: 'OK', prefill: p, decode: d, prefill_tps: p.tps, decode_agg: agg, useful_tps: useful, eta_prefill: pf.gpu, eta_prefill_cpu: (q.placement || 'gpu') !== 'gpu' ? pf.cpu : null, eta_decode: df.status === 'FIT' ? df.eta : null };
  }
  const rlArith = x => x.prefill.terms.slice().sort((a, b) => b.s - a.s).map(t => t.id + ' ' + t.s.toFixed(3) + ' s').join(' + ') + ' (' + x.prefill.overlap + ') = ' + x.prefill.time_s.toFixed(3) + ' s for ' + x.prefill.tokens + ' tokens; decode step ' +
    x.decode.terms.slice().sort((a, b) => b.s - a.s).map(t => t.id + ' ' + (t.s * 1e3).toFixed(2) + ' ms').join(' + ') + (x.eta_decode ? ' / ' + x.eta_decode.toFixed(3) : '');
  // the search: every candidate scored like the base, ranked by predicted useful tok/s (prefill tok/s when no workload mix)
  function rooflineRearrange(data, q, opts) {
    const o = Object.assign({}, opts || {}), W = data.workload, prof = W && W.profiles[v(W.settings.default_profile)];
    if (o.fresh === undefined) o.fresh = prof ? v(prof.fresh_per_output) : null;
    const base = rlScore(data, q, o);
    if (base.status !== 'OK') return { status: 'UNMODELLED', why: base.why };
    const key = x => x.useful_tps !== null ? x.useful_tps : x.prefill_tps;
    const rows = rlCandidates(data, q, { prefill: base.prefill }).map(c => {
      const x = rlScore(data, c.query, o);
      return x.status !== 'OK' ? Object.assign({}, c, { status: x.status, why: x.why }) : Object.assign({}, c, { status: 'OK', prefill_tps: x.prefill_tps, decode_agg: x.decode_agg, useful_tps: x.useful_tps,
        gain: key(x) / key(base) - 1, prefill_binding: x.prefill.binding.term, decode_binding: x.decode.binding.term, arithmetic: rlArith(x) });
    });
    // batch: twice the agents on the same placement (aggregate useful; each agent's share falls), held to the VRAM capacity
    const B = o.agents || 1, d1 = decodeRoof(data, Object.assign({}, q, { batch: 1 })), cap = d1.status === 'OK' ? d1.capacity_streams : null;
    const x2 = rlScore(data, q, Object.assign({}, o, { agents: 2 * B }));
    if (x2.status === 'OK') rows.push({ id: 'agents_x2', change: 'agents ' + B + ' -> ' + 2 * B + ' on this placement', targets: 'read.resident (weights amortised over more streams)', status: cap !== null && cap < 2 * B ? 'PAST_CAPACITY' : 'OK',
      why: cap !== null && cap < 2 * B ? 'VRAM holds ' + cap + ' streams at ctx ' + q.ctx : undefined, needs_code: null, condition: 'the server runs ' + 2 * B + ' streams at once (tqserve --np, vLLM max-num-seqs, strata agents); each agent gets 1 / ' + 2 * B + ' of the useful rate', prefill_tps: x2.prefill_tps, decode_agg: x2.decode_agg, useful_tps: x2.useful_tps,
      useful_per_agent: x2.useful_tps !== null ? x2.useful_tps / (2 * B) : null, gain: key(x2) / key(base) - 1, prefill_binding: x2.prefill.binding.term, decode_binding: x2.decode.binding.term, arithmetic: rlArith(x2) });
    rows.sort((a, b) => (b.status === 'OK' ? b.gain : -9) - (a.status === 'OK' ? a.gain : -9));
    return { status: 'OK', agents: o.agents || 1, fresh: o.fresh, base: { prefill_tps: base.prefill_tps, decode_agg: base.decode_agg, useful_tps: base.useful_tps, prefill_binding: base.prefill.binding.term, decode_binding: base.decode.binding.term, arithmetic: rlArith(base) }, rows };
  }
  // explain: the binding term, the next-binding term, and the rearrangement that relieves the binding one with its predicted gain
  function rooflineExplain(data, q, opts) {
    const R = rooflineRearrange(data, q, opts);
    if (R.status !== 'OK') return R;
    const s = rlScore(data, q, Object.assign({ fresh: R.fresh }, opts || {})), order = ts => ts.slice().sort((a, b) => b.s - a.s);
    const pt = order(s.prefill.terms), dt = order(s.decode.terms);
    const pre = { binding: s.prefill.binding.term, next: (pt.find(t => s.prefill.overlap === 'max' ? (/^compute/.test(t.id) !== (s.prefill.binding.side === 'compute')) : t.id !== s.prefill.binding.term) || {}).id || null };
    const dec = { binding: dt[0].id, next: dt[1] ? dt[1].id : null };
    const mixPrefill = R.fresh ? R.fresh / R.base.prefill_tps > 1 / R.base.decode_agg : false;
    const target = mixPrefill ? pre.binding : dec.binding;
    const hits = r => r.targets === target || (/^compute/.test(target) && /^compute/.test(r.targets));
    const direct = R.rows.find(r => r.status === 'OK' && r.gain > 0.005 && hits(r)), any = R.rows.find(r => r.status === 'OK' && r.gain > 0.005);
    const relief = direct ? Object.assign({ relieves_binding: true }, direct) : any ? Object.assign({ relieves_binding: false }, any) : null;
    return { status: 'OK', agents: R.agents, mix_binds: mixPrefill ? 'prefill (F / P > 1 / decode)' : 'decode', target, prefill: pre, decode: dec, relief, search: R };
  }
  // ---------- prediction records (data/predictions-roofline-*.json): a run states its prediction and the term it tests first ----------
  // rooflinePredict makes the record (report.js --roofline --predict adds the time and RM commit and appends it); it refuses a
  // spec without a hypothesis, the term it tests, and a falsification band. rooflineScore scores a measurement against its record:
  // in or out of band, then term attribution (rooflineAttribute); a measurement older than its record is refused (post hoc)
  // the instant of an ISO-8601 time with an explicit zone (Z or +hh:mm), in ms, else null. A time with no zone, a date alone
  // or any other text has no instant, and string order is not time order across offsets
  const RL_ZONED = /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2})(?:\.\d+)?)?(Z|[+-](\d{2}):(\d{2}))$/;
  function rlInstant(s) {
    const m = typeof s === 'string' ? RL_ZONED.exec(s) : null;
    if (!m) return null;
    const [Y, M, D, h, mi, se] = [1, 2, 3, 4, 5, 6].map(i => Number(m[i] || 0)), oh = m[8] !== undefined ? Number(m[8]) : 0, om = m[9] !== undefined ? Number(m[9]) : 0;
    if (M < 1 || M > 12 || h > 23 || mi > 59 || se > 59 || oh > 23 || om > 59) return null;
    const days = new Date(Date.UTC(Y, M, 0)).getUTCDate();
    if (D < 1 || D > days) return null;
    const t = Date.parse(s);
    return Number.isFinite(t) ? t : null;
  }
  // a measurement is evidence for a record only when both times are zoned instants and the measurement is strictly later
  function rlAfterRecord(measuredUtc, writtenUtc) {
    const m = rlInstant(measuredUtc), w = rlInstant(writtenUtc);
    if (m === null) return { ok: false, why: 'measured ' + (measuredUtc ? JSON.stringify(measuredUtc) + ' is not an ISO-8601 time with a zone (Z or +hh:mm)' : 'at an unstated time') };
    if (w === null) return { ok: false, why: 'the record time ' + JSON.stringify(writtenUtc || null) + ' is not an ISO-8601 time with a zone' };
    if (m <= w) return { ok: false, why: 'measured ' + measuredUtc + ' is not after the record (' + writtenUtc + '): post hoc, not evidence' };
    return { ok: true };
  }
  function rooflinePredict(data, spec) {
    const miss = ['id', spec.phase === 'pipeline' ? 'plan' : 'query', 'hypothesis', 'tests_term', 'band_pct'].filter(k => spec[k] === undefined || spec[k] === null || spec[k] === '');
    if (miss.length) return { status: 'REFUSED', why: 'a prediction record needs ' + miss.join(', ') + ': a run with no stated prediction and tested term is not evidence' };
    const phase = spec.phase || 'prefill', f = phase === 'pipeline' ? null : rooflineFit(data, spec.query.engine);
    if (phase === 'pipeline') {
      // a multi-box plan (pipelinePlan spec in spec.plan): aggregate decode at B streams at read depth spec.ctx_read (the primary
      // number), per-stream at B 1, prefill of spec.prefill_tokens, the stream capacity, and the agent-cycle useful tok/s at
      // spec.agents_B, every stage's time and the hop share frozen beside it
      const terms = ['stage_read', 'hop', 'compute', 'capacity'];
      if (!terms.includes(spec.tests_term)) return { status: 'REFUSED', why: 'tests_term ' + spec.tests_term + ' is not a pipeline term (' + terms.join(', ') + ')' };
      const plan = pipelinePlan(data, spec.plan);
      if (plan.status !== 'OK') return { status: 'REFUSED', why: 'no plan: ' + plan.why };
      const B = Number(spec.B) || 1, at = (b, c) => pipelineRoof(data, plan, b, { ctx: c }), d = at(B, spec.ctx_read), d1 = at(1, spec.ctx_read), pre = pipelinePrefill(data, plan, spec.prefill_tokens || 2048);
      // the guard: a claim at B > 1 needs every stage's batch curve measured or derived (card_curves); a missing constant prices
      // its class at the bandwidth line, which the multi-box runs falsified (2.1-3.4x optimistic at B 8)
      if (B > 1 && d.calibration === 'UNCALIBRATED') return { status: 'REFUSED', why: 'UNCALIBRATED at B ' + B + ': ' + d.uncalibrated.join('; ') + '. A phase_bound claim at B > 1 needs a measured or derived per-card batch curve (data/multibox.json card_curves: a calibration point for the constant)' };
      const LA = spec.agents_B ? pipelineLandscape(data, spec.plan, { batches: [].concat(spec.agents_B) }) : null;
      return { status: 'OK', record: { id: spec.id, phase, B, ctx_read: d.ctx_read, plan_spec: spec.plan, hypothesis: spec.hypothesis, tests_term: spec.tests_term, band_pct: spec.band_pct, conditions: spec.conditions || null,
        predicted: { aggregate_tps: d.aggregate_tps, per_stream_tps: d.per_stream_tps, per_stream_b1_tps: d1.per_stream_tps, step_ms: 1e3 * d.step_s, hop_share: d.hop_share, slowest: d.slowest, ring_aggregate_tps: pipelineRoof(data, plan, B, { ctx: spec.ctx_read, mode: 'ring' }).aggregate_tps,
          stages: plan.stages.map((s, i) => ({ box: s.box, gpu: s.gpu, layers: s.layers, weights_gb: s.weights_gb, ms: 1e3 * d.stages[i].s, binds: d.stages[i].binds, rate_gbs: s.line.rate_gbs, rate_tag: s.line.tag })),
          prefill_tps: pre.tps, prefill_tokens: pre.tokens, capacity_streams: plan.capacity_streams, plan_ctx: plan.ctx, weights_gb: plan.weights_gb, calibration: d.calibration,
          agents: LA && LA.agents && LA.agents.status === 'OK' ? LA.agents.rows.map(r => r.status === 'OK' ? { B: r.B, useful_tps: r.useful_tps, b_eff: r.b_eff, F: r.F } : { B: r.B, status: r.status, why: r.why }) : null } } };
    }
    if (phase === 'agents') {
      // the agent duty cycle at B agents (agentLandscape): useful tok/s, B_eff, duty and F, with the opts (trace, server, slots,
      // cache, decode curve) frozen into the record; spec.variants {name: opts overrides} freezes sensitivity arms beside it
      const terms = ['duty_cycle', 'fresh_reread', 'decode_at_beff', 'slot_eviction', 'read_blocks_decode', 'admission_door'];
      if (!terms.includes(spec.tests_term)) return { status: 'REFUSED', why: 'tests_term ' + spec.tests_term + ' is not an agent-cycle term (' + terms.join(', ') + ')' };
      if (!Number.isInteger(spec.B) || spec.B < 1) return { status: 'REFUSED', why: 'an agents record needs an integer B' };
      const at = ov => { const L = agentLandscape(data, spec.query, Object.assign({}, spec.agents || {}, ov || {}, { batches: [spec.B] })), r = L.status === 'OK' ? L.rows[0] : null;
        return r && r.status === 'OK' ? { useful_tps: r.useful_tps, useful_per_agent: r.useful_per_agent, b_eff: r.b_eff, duty: r.duty, F: r.F, per_stream_at_beff: r.per_stream_at_beff, aggregate_at_beff: r.aggregate_at_beff,
          reread_per_turn: r.reread_per_turn, evicted_share: r.evicted_share, turns_per_agent_min: r.turns_per_agent_min, share: r.share, landscape_useful: r.landscape_useful, decode_basis: L.decode_basis,
          ...(r.engine ? { engine: r.engine } : {}) } : { why: L.why || (r && r.why) || 'no row' }; };
      const p = at(null);
      if (p.why) return { status: 'REFUSED', why: 'no agent-cycle prediction: ' + p.why };
      const variants = Object.fromEntries(Object.entries(spec.variants || {}).map(([k, ov]) => [k, at(ov)]));
      return { status: 'OK', record: { id: spec.id, phase, B: spec.B, query: spec.query, agents: spec.agents || null, hypothesis: spec.hypothesis, tests_term: spec.tests_term, band_pct: spec.band_pct, conditions: spec.conditions || null,
        predicted: p, variants } };
    }
    if (phase === 'decode') {
      const bq = Number(spec.query.batch) || 1;
      if (bq > 1 && !batchCurveOf(data, spec.query)) return { status: 'REFUSED', why: 'UNCALIBRATED at batch ' + bq + ': ' + spec.query.box + ' ' + spec.query.model + ' ' + spec.query.engine + ' has no measured batch curve (data.curves); a phase_bound claim at B > 1 needs one' };
      const d = decodeRoof(data, spec.query), df = decodeFit(data, spec.query.engine);
      if (d.status !== 'OK') return { status: 'REFUSED', why: 'no decode bound: ' + d.why };
      const t = df.status === 'FIT' ? d.tps_per_stream * df.eta : d.tps_per_stream;
      if (!d.terms.some(x => x.id === spec.tests_term)) return { status: 'REFUSED', why: 'tests_term ' + spec.tests_term + ' is not a decode term (' + d.terms.map(x => x.id).join(', ') + ')' };
      return { status: 'OK', record: { id: spec.id, phase, query: spec.query, hypothesis: spec.hypothesis, tests_term: spec.tests_term, band_pct: spec.band_pct, conditions: spec.conditions || null,
        predicted: { decode_tps: t, bound_tps: d.tps_per_stream, binding: d.binding.term, decode_eta: df.status === 'FIT' ? df.eta : null, terms: d.terms.map(x => ({ id: x.id, s: x.s })) } } };
    }
    const et = rlEta(data, spec.query.engine), p = prefillRoof(data, spec.query, rlEtaFor(data, spec.query)), b = prefillRoof(data, spec.query, 1);
    if (p.status !== 'OK') return { status: 'REFUSED', why: 'no prefill bound: ' + p.why };
    if (!p.terms.some(x => x.id === spec.tests_term)) return { status: 'REFUSED', why: 'tests_term ' + spec.tests_term + ' is not a prefill term (' + p.terms.map(x => x.id).join(', ') + ')' };
    return { status: 'OK', record: { id: spec.id, phase, query: spec.query, hypothesis: spec.hypothesis, tests_term: spec.tests_term, band_pct: spec.band_pct, conditions: spec.conditions || null,
      predicted: { prefill_tps: p.tps, time_s: p.time_s, bound_tps: b.tps, binding: p.binding.term, side: p.binding.side, eta: f.status === 'FIT' ? f.eta : null, eta_cpu: (spec.query.placement || 'gpu') !== 'gpu' ? et.cpu : null, terms: p.terms.map(x => ({ id: x.id, s: x.s })),
        composition: p.composition, link: rlLinkSpec(data, spec.query) } } };
  }
  function rooflineScore(data, rec, measured, measuredUtc) {
    if (!rec) return { status: 'REFUSED', why: 'no prediction record: score only a run whose prediction was written first' };
    const after = rlAfterRecord(measuredUtc, rec.written_utc);
    if (!after.ok) return { status: 'REFUSED', why: after.why };
    if (!(Number(measured) > 0)) return { status: 'REFUSED', why: 'measured ' + JSON.stringify(measured === undefined ? null : measured) + ' is not a positive rate' };
    const pr = rec.predicted, isDec = rec.phase === 'decode' || rec.phase === 'agents' || rec.phase === 'pipeline', want = rec.phase === 'agents' ? pr.useful_tps : rec.phase === 'pipeline' ? pr.aggregate_tps : isDec ? pr.decode_tps : pr.prefill_tps, err = measured / want - 1;
    const inBand = Math.abs(err) * 100 <= rec.band_pct;
    // attribution is frozen (the record's own terms); the current model's reading is a separate, labelled block
    // (prefill records only: a decode, agents or pipeline record has no prefill terms, and a pipeline record no query)
    const tok = !isDec && rec.query ? Number(rec.query.tokens) : 0;
    const att = tok > 0 ? rooflineAttributeFrozen(rec, tok / measured) : null;
    const cur = tok > 0 ? rooflineAttribute(data, rec.query, tok / measured, { gpu: pr.eta || 1, cpu: pr.eta_cpu || null }) : null;
    return { status: inBand ? 'HOLDS' : 'FALSIFIED', id: rec.id, tests_term: rec.tests_term, predicted: want, measured, err, band_pct: rec.band_pct, attribution: att, attribution_current: cur,
      tested_term_needed: att && att.terms ? (att.terms.find(t => t.term === rec.tests_term) || null) : null };
  }
  // the measured batch curves (data.curves configs) of a box / model / engine, else null
  function batchCurveOf(data, q) {
    const cs = (((data.curves || {}).configs) || []).filter(c => c.box === q.box && c.model === q.model && c.engine === q.engine);
    return cs.length ? cs : null;
  }
  // the record of a check (pre-registered when written before the check's measurement), else null
  const rlRecordOf = (data, id) => ((data.roofline || {}).predictions || []).find(r => r.check_id === id || r.id === id) || null;
  // the recipe's query for its phase bounds: box/model/engine/store from the recipe, its prompt tokens and ubatch, and the expert
  // placement of its prediction (GPU tiers resident, the rest host-side: streamed over PCIe when its offload path runs, else
  // computed on the CPU from DRAM)
  function recipeRoofQuery(data, r, prediction) {
    const q = { box: r.box, model: r.model, engine: r.engine, store: r.roofline_store || (r.store && r.store.type) || null, tokens: r.prefill_tokens || 2048,
      chunk: Number(r.ub) > 0 ? Number(r.ub) : undefined, flags: (r.conditions && r.conditions.engine_flags) || '', unlock: r.unlock, at: r.date || r.at };
    // where the GEMMs run (recipe roofline_placement, a name in data.roofline.settings.placements; none = the GPU placement), the
    // threads (the CPU core share) and the expert quants (the engine's cpu path). The recipe's own placement is the planner's
    // per-card expert placement (data.placements shape: gpu_layers, cards), never read here
    if (r.roofline_placement !== undefined && r.roofline_placement !== null) q.placement = String(r.roofline_placement);
    if (Number(r.threads) > 0) q.threads = Number(r.threads);
    if (r.store && Array.isArray(r.store.types) && r.store.types.length) q.expert_quant = r.store.types.slice();
    if ((q.placement || 'gpu') === 'gpu' && prediction && prediction.tiers && prediction.tiers.length && !((data.roofline.engines[r.engine] || {}).prefill || {}).pass) {
      const gpu = prediction.tiers.filter(t => /^gpu/.test(t.id)), hostT = prediction.tiers.filter(t => !/^gpu/.test(t.id));
      q.resident_bytes = gpu.reduce((a, t) => a + t.experts_gib * GiB, 0);
      const all = prediction.tiers.reduce((a, t) => a + t.experts_gib * GiB, 0);
      q.store_bytes = all;
      q.resident_frac = all > 0 ? q.resident_bytes / all : 1;
      q.host_path = prediction.prefill && prediction.prefill.offload_on && Math.min(q.tokens, prediction.prefill.ub) >= prediction.prefill.offload_batch ? 'stream' : 'cpu';
      if (!hostT.length) q.host_path = 'none';
    }
    return q;
  }
  // checks.phase_bound: a recipe the planner cannot bound per phase (prefill and decode, the binding term named) FAILs its fit
  // check: an unbounded recipe is a question for arithmetic, not for a run
  function phaseBoundCheck(data, r, prediction) {
    if (!data.roofline) return null;
    const q = recipeRoofQuery(data, r, prediction), pb = phaseBound(data, q);
    const wl = (r.conditions && r.conditions.workload) || {}, nAg = Number(wl.agents) || 1;
    // the guard: a recipe for many agents whose box / model / engine has no measured batch curve prints UNCALIBRATED and makes no
    // claim past one stream (the B 1 bounds still stand). A sweep (agents: [1, 4, 8]) asks for its largest count
    const nMax = Array.isArray(wl.agents) ? Math.max(1, ...wl.agents.map(Number).filter(Number.isFinite)) : nAg;
    const unc = nMax > 1 && !batchCurveOf(data, q);
    const ex = !unc && pb.prefill.bound.status === 'OK' && pb.decode.b1.status === 'OK' ? rooflineExplain(data, q, { agents: nAg }) : null;
    const p = pb.prefill.bound, d = pb.decode.b1;
    const st = p.status === 'OK' && d.status === 'OK' ? 'PASS' : 'FAIL';
    const fmt = x => x >= 100 ? x.toFixed(0) : x.toFixed(1);
    const onCpu = (q.placement || 'gpu') !== 'gpu', fe = onCpu ? pb.eta.cpu : pb.eta.gpu;
    const refused = [p, d].map(x => x.status !== 'OK' && /^refused/.test(x.why || '')).some(Boolean);
    const detail = st === 'PASS'
      ? (onCpu ? 'placement ' + q.placement + ': ' : '') + 'prefill ' + q.tokens + ' tokens: bound ' + fmt(p.tps) + ' tok/s, ' + p.binding.side + ' binds (' + p.binding.term + ', ' + p.binding.resource + ')' + (pb.prefill.predicted && pb.prefill.predicted.status === 'OK' ? '; at the fitted ' + (onCpu ? 'CPU ' : '') + 'eta ' + fe.toFixed(3) + ': ' + fmt(pb.prefill.predicted.tps) + ' tok/s, ' + pb.prefill.predicted.binding.term : '; no fitted ' + (onCpu ? 'CPU ' : '') + 'eta for ' + r.engine + ' (bound only)') +
        '; decode batch 1: bound ' + fmt(d.tps_per_stream) + ' tok/s, ' + d.binding.term + ' (' + d.binding.resource + ')' + (p.unmodelled.length ? '; not modelled: ' + p.unmodelled.join('; ') : '')
      : (refused ? 'REFUSED, no phase bound: ' : 'no phase bound: ') + [p.status !== 'OK' && 'prefill ' + p.why, d.status !== 'OK' && 'decode ' + d.why].filter(Boolean).join('; ');
    const exd = ex && ex.status === 'OK' ? '; explain (' + ex.agents + ' agent(s), the mix binds ' + ex.mix_binds + '): prefill ' + ex.prefill.binding + ' then ' + ex.prefill.next + ', decode ' + ex.decode.binding + ' then ' + ex.decode.next +
      (ex.relief ? '; ' + (ex.relief.relieves_binding ? 'relieve ' + ex.target + ' with ' : 'nothing in the search relieves ' + ex.target + '; best other gain: ') + ex.relief.change + ' (' + (ex.relief.gain >= 0 ? '+' : '') + (100 * ex.relief.gain).toFixed(1) + ' % predicted' + (ex.relief.needs_code ? ', NEEDS CODE: ' + ex.relief.needs_code : '') + ')' : '; no rearrangement in the search gains') : '';
    const ud = unc ? '; UNCALIBRATED at ' + nMax + ' agents: no measured batch curve for ' + r.box + ' ' + r.model + ' ' + r.engine + ' (data.curves), so no phase_bound claim at B > 1' : '';
    return { id: 'phase_bound', label: 'prefill and decode bounds per phase, binding term named (data.roofline)', status: st, detail: detail + exd + ud, bound: pb, explain: ex, batch_curve: nMax > 1 ? (unc ? 'UNCALIBRATED' : 'MEASURED') : null };
  }

  const api = { rlEta, rlPlacement, shapeRate, shapeTime, rooflinePredict, rooflineScore, rooflineAttribute, rooflineAttributeFrozen, rlInstant, rlAfterRecord, rooflineRearrange, rooflineExplain, decodeFit, batchLandscape, agentLandscape, batchModeStep, pipelinePlan, pipelineRoof, pipelinePrefill, pipelineLandscape, replicaRecipes, replicaBaseline, replicaGate, replicaMeasured, replicaRank, cardStage, cardConsts, cardCurveFit, cardCurveRescore, agentPointRescore, batchCurveOf, ceilingOf, unlockState, prefillRoof, decodeRoof, rooflineFit, phaseBound, rooflineChecks, recipeRoofQuery, phaseBoundCheck, RL_SHAPE_A, RESIDENT_LEVERS, residentTerms, residentAgentPlan, agentTurns, agentReplaySim, agentEngineSim, agentTurnPredict, agentTurnRescore, prefillPass, prefillFit, prefillPassChecks, chunksPerRead, predictionFirstCheck, priorFacts, priorsFor, priorValue, fmtPrior, freshReference, replayArtifactCheck, steadyStateCheck, replayChecks, agentThroughput, agentReference, agentEvidence, agentProfile, agentPlan, agentPlans, agentWorkloadCheck, cpuPoolCheck, buildKey, fallbackCheck, expertResidencyCheck, networkAssumptions, networkCanonical, networkSame, networkCheck, networkTransfer, networkPower, networkResources, winComparison, winNetwork, hostPinnedBudget, curveFit, curveHost, curveWatts, curveReport, curveReports, winInterp, winClassMs, winFloorMs, winLayerMs, winStageMs, winPointSplit, winLaw, winStep, winRebalance, winFit, winAnchor, winLevers, winProfCompile, winProfFit, winProfLevers, winEvidence, winFrontier, winProposals, winProposalTemplate, winReport, batchDistinct, batchWeights, batchRowGiB, batchStep, batchFit, batchTable, verdictReady, screenChecks, layerMap, layerWeights, layerDamage, recipeDamage, trimKl, trimKeeps, trimSupport, mapQuality, lsq, dllmTerms, dllmValidation, predictDllm, dllmCheck, dllmCover, dllmLine, evidenceOf, term, rollup, fmtRollup, predictionEvidence, predictExternal, predictDense, predictDenseAll, addModels, denseCards, denseCal, denseCalReport, densePointMs, numaGain, externalSummary, externalDerived, fmtExternal, quantType, hostSrc, draftSrc, splitSrc, kappaSrc, checkRecipe, earfOf, earfCheck, requiresEarf, seatSafetyChecks, mbEarf, cpuReserveCheck, servingOf, kvHostPlan, kvHostCheck, hostPinnedPlan, hostPinnedCheck, placeCheck, predictRecipe, prefillModel, heatCite, storeTypes, recipeLowShare, recipeQuality, mountKind, vramGiB, kappaOf, marginOf, lowShare, gateOf, gateRepair, layoutFacts, runChecks, checksFor, checkRecipes, GiB, v, qualityOf, heatSet, typeBytes, typeKld, context, prepare, solveLambda, expand, evaluate, fill, frontier, bestAt, layoutAt, calibrate, applyCal, anchorBytes, anchorTime, ownerLayout, routingSkew, exportMap, LAMS };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.FitModel = api;
})(this);
