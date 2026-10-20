// fit-viz UI. Plain DOM + SVG, no libraries. Uses window.FIT_DATA and window.FitModel.
(function () {
  'use strict';
  const data = window.FIT_DATA, FM = window.FitModel, v = FM.v, GiB = FM.GiB;
  const LS = 'fit-viz:v1:';
  const app = document.getElementById('app'), tip = document.getElementById('tip');
  const esc = s => String(s === undefined || s === null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const f1 = x => (x === null || x === undefined || !isFinite(x)) ? '-' : x.toFixed(1);
  const f2 = x => (x === null || x === undefined || !isFinite(x)) ? '-' : x.toFixed(2);
  // one chip per tagged number: its evidence class (seen / inferred / unknown) in the text, the tag and source on hover
  const chip = (n, extra) => {
    if (!n || typeof n !== 'object' || !n.tag) return '';
    const ev = n.evidence || FM.evidenceOf(data, n);
    return `<span class="chip ev-${esc(ev)} ${esc(n.tag)}" title="${esc(ev + ' (' + (n.tag === 'PRIOR' ? 'PRIOR, community' : n.tag) + '): ' + (n.src || 'no source') + (n.runs ? ' [rows: ' + n.runs.join(', ') + ']' : '') + (extra ? ' ' + extra : ''))}">${esc(ev)}</span>`;
  };
  const gradeChip = (g, title) => `<span class="chip ev-${esc(g)}" title="${esc(title || '')}">${esc(g)}</span>`;
  const tagChip = (tag, src) => chip({ tag, src });
  const load = (k, d) => { try { const x = localStorage.getItem(LS + k); return x === null ? d : JSON.parse(x); } catch (e) { return d; } };
  const save = (k, x) => { try { localStorage.setItem(LS + k, JSON.stringify(x)); } catch (e) { /* private mode */ } };

  // ---------- state ----------
  const defaults = JSON.parse(JSON.stringify(data.knobs));
  const st = {
    box: load('box', 'droopy'),
    knobs: Object.assign({}, defaults, load('knobs', {})),
    theme: load('theme', 'auto'),
    heat: load('heat', {}),
    sel: null,
    table: false
  };
  data.overrides = load('overrides', { kappa: {}, kld: {} });
  if (!data.boxes[st.box]) st.box = 'droopy';
  const applyTheme = () => { if (st.theme === 'auto') document.documentElement.removeAttribute('data-theme'); else document.documentElement.setAttribute('data-theme', st.theme); };
  applyTheme();

  const MODELS = Object.keys(data.models).filter(m => data.models[m].show);
  const css = n => getComputedStyle(document.getElementById('root')).getPropertyValue(n).trim();
  const modelColor = (mk, eng) => eng === 'glmlite' ? 'var(--s3)' : ({ v41: 'var(--s1)', fnx: 'var(--s4)', fnxc: 'var(--s5)' })[mk] || 'var(--s2)';
  const engShort = e => ({ ik: 'ik hybrid', glmlite: 'glm-lite', sycl: 'SYCL' })[e] || e;

  function enginesFor(bk, mk, kn) {
    const box = data.boxes[bk];
    const gpus = box.gpus.slice();
    if (kn.moveB70 === bk) gpus.push(data.b70move.gpu);
    const set = [];
    gpus.forEach(g => (g.engines || []).forEach(e => { if (set.indexOf(e) < 0) set.push(e); }));
    const fam = data.models[mk].kld_family;
    return set.filter(e => !data.engines[e].no_solver && data.engines[e].types.some(q => data.types[q].kld[fam] && v(data.types[q].kld[fam]) !== null));
  }
  function heatFor(mk) {
    const h = st.heat[mk] || {};
    const m = data.models[mk];
    return { plan: h.plan && m.heat.sets[h.plan] ? h.plan : m.heat.plan_default, score: h.score && m.heat.sets[h.score] ? h.score : m.heat.score_default };
  }

  // ---------- compute ----------
  let R = null;
  function compute() {
    const t0 = performance.now();
    FM.applyCal(data);
    const curves = [];
    MODELS.forEach(mk => {
      enginesFor(st.box, mk, st.knobs).forEach((eng, ei) => {
        const hs = heatFor(mk);
        const ctx = FM.prepare(FM.context(data, st.box, mk, eng, st.knobs), hs.plan, hs.score);
        const pts = FM.frontier(ctx);
        const b20 = FM.bestAt(ctx, 20, pts), b25 = FM.bestAt(ctx, 25, pts);
        curves.push({ mk, eng, ei, ctx, pts, b20, b25, color: modelColor(mk, eng), dash: ei > 0 && eng !== 'glmlite' });
      });
    });
    let owner = null;
    if (st.box === 'droopy' && data.alloc3) {
      const c = curves.find(c => c.mk === 'v41' && c.eng === 'ik');
      if (c) {
        const kn = Object.assign({}, st.knobs, { perExpertTypes: true });
        const ctx = FM.prepare(FM.context(data, 'droopy', 'v41', 'ik', kn), c.ctx.planName, c.ctx.scoreName);
        const lay = FM.ownerLayout(ctx, false);
        owner = { ctx, lay, ev: FM.evaluate(ctx, lay) };
      }
    }
    R = { curves, owner, ms: performance.now() - t0 };
    // keep the selection if it still exists, else pick the best V4.1 point at 20 (or the first curve's fastest)
    let pick = null;
    if (st.sel) {
      const c = curves.find(c => c.mk === st.sel.mk && c.eng === st.sel.eng);
      if (st.sel.kind === 'owner' && owner) pick = { kind: 'owner', mk: 'v41', eng: 'ik' };
      else if (c) pick = st.sel;
    }
    if (!pick) {
      const order = curves.slice().sort((a, b) => (b.b20 ? 1 : 0) - (a.b20 ? 1 : 0) || (a.mk === 'v41' ? -1 : 1));
      const c = order[0];
      if (c) pick = { mk: c.mk, eng: c.eng, kind: c.b20 ? 't20' : 'fast' };
    }
    st.sel = pick;
    loadSelection();
  }
  function curveOf(sel) { return R.curves.find(c => c.mk === sel.mk && c.eng === sel.eng); }
  function loadSelection() {
    const s = st.sel;
    R.selected = null;
    if (!s) return;
    if (s.kind === 'owner') { R.selected = Object.assign({ label: 'Owner layout (droopy): hottest coding experts MXFP4 in VRAM, other coding experts q3_K in DRAM, the rest MXFP4 in Optane', needsCode: true }, R.owner); return; }
    const c = curveOf(s);
    if (!c) return;
    let r = null, label = '';
    if (s.kind === 't20') { r = c.b20; label = 'best quality at 20 tok/s'; }
    else if (s.kind === 't25') { r = c.b25; label = 'best quality at 25 tok/s'; }
    else if (s.kind === 'pt') { r = FM.bestAt(c.ctx, s.tokps * 0.995, c.pts); label = 'best quality at ' + f1(s.tokps) + ' tok/s'; }
    if (!r) {
      const fe = c.pts.filter(p => p.feasible);
      if (fe.length) { const p = fe.reduce((a, b) => b.tokps > a.tokps ? b : a); r = FM.bestAt(c.ctx, p.tokps * 0.995, c.pts); label = 'fastest layout (' + f1(p.tokps) + ' tok/s)'; }
    }
    if (!r) return;
    R.selected = { ctx: c.ctx, lay: r.lay, ev: r.ev, label: data.models[c.mk].short + ' / ' + engShort(c.eng) + ': ' + label, curve: c };
  }

  // ---------- frontier chart ----------
  function envelope(pts) {
    const fe = pts.filter(p => p.feasible).slice().sort((a, b) => b.tokps - a.tokps);
    const out = []; let best = -Infinity;
    fe.forEach(p => { if (p.quality > best + 1e-9) { out.push(p); best = p.quality; } });
    return out.sort((a, b) => a.tokps - b.tokps);
  }
  function anchorsHere() {
    return data.cal.rows.filter(r => r.box === st.box && MODELS.indexOf(r.model) >= 0).map(r => {
      const q = r.explicit ? r.explicit.type : data.models[r.model].base_type;
      const kq = FM.typeKld(data, r.model, q);
      return Object.assign({}, r, { quality: FM.qualityOf(data, (kq || 0)) });
    });
  }
  function frontierSVG() {
    const AW = app.clientWidth || 720;
    const W = Math.round(Math.max(320, AW >= 980 ? AW * 0.55 - 40 : AW - 30)), H = Math.round(Math.min(400, Math.max(300, W * 0.55))), L = 46, Rm = 14, T = 14, B = 40;
    const allP = [];
    R.curves.forEach(c => envelope(c.pts).forEach(p => allP.push(p)));
    const anc = anchorsHere();
    const xmax = Math.max(40, ...allP.map(p => p.tokps * 1.05), ...anc.map(a => Math.max(a.meas, a.pred) * 1.05), R.owner ? R.owner.ev.tokps * 1.05 : 0);
    const qs = allP.map(p => p.quality).concat(anc.map(a => a.quality));
    const ymin = Math.max(0, Math.floor((Math.min(75, ...qs) - 2) / 5) * 5);
    const X = x => L + (W - L - Rm) * Math.min(x, xmax) / xmax, Y = y => T + (H - T - B) * (100 - y) / (100 - ymin);
    let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="width:${W}px;max-width:100%" role="img" aria-label="Projected quality versus decode speed for each model on ${esc(st.box)}">`;
    for (let x = 0; x <= xmax; x += 5) s += `<line x1="${X(x)}" x2="${X(x)}" y1="${T}" y2="${H - B}" stroke="var(--grid)"/><text x="${X(x)}" y="${H - B + 14}" text-anchor="middle">${x}</text>`;
    for (let y = ymin; y <= 100; y += 5) s += `<line x1="${L}" x2="${W - Rm}" y1="${Y(y)}" y2="${Y(y)}" stroke="var(--grid)"/><text x="${L - 6}" y="${Y(y) + 4}" text-anchor="end">${y}</text>`;
    s += `<text x="${(L + W - Rm) / 2}" y="${H - 6}" text-anchor="middle">projected decode speed, tok/s (one user)</text>`;
    s += `<text transform="translate(12 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">top-1 agreement with the full model, %</text>`;
    [20, 25].forEach(x => { s += `<line x1="${X(x)}" x2="${X(x)}" y1="${T}" y2="${H - B}" stroke="var(--text-secondary)" stroke-dasharray="4 3"/><text x="${X(x) + 3}" y="${T + 10}" class="ink">${x}</text>`; });
    R.curves.forEach((c, ci) => {
      const env = envelope(c.pts);
      if (env.length) {
        s += `<polyline fill="none" stroke="${c.color}" stroke-width="2" ${c.dash ? 'stroke-dasharray="6 4"' : ''} points="${env.map(p => X(p.tokps) + ',' + Y(p.quality)).join(' ')}"/>`;
        env.forEach((p, pi) => { s += `<circle cx="${X(p.tokps)}" cy="${Y(p.quality)}" r="9" fill="transparent" data-k="pt" data-c="${ci}" data-x="${p.tokps}" data-q="${p.quality}" data-kld="${p.kld}" style="cursor:pointer"/><circle cx="${X(p.tokps)}" cy="${Y(p.quality)}" r="2.5" fill="${c.color}" pointer-events="none"/>`; });
        const last = env[env.length - 1];
        s += `<text x="${Math.min(X(last.tokps) + 6, W - 120)}" y="${Y(last.quality) - 6}" class="ink">${esc(data.models[c.mk].short + ' ' + engShort(c.eng))}</text>`;
      }
      [['b20', 20], ['b25', 25]].forEach(([k, t]) => {
        const b = c[k];
        if (!b) return;
        const sel = st.sel && st.sel.mk === c.mk && st.sel.eng === c.eng && st.sel.kind === 't' + t;
        s += `<circle cx="${X(b.ev.tokps)}" cy="${Y(b.ev.quality)}" r="${sel ? 8 : 6}" fill="${c.color}" stroke="var(--surface-1)" stroke-width="2" data-k="t${t}" data-c="${ci}" style="cursor:pointer"/>`;
        s += `<text x="${X(b.ev.tokps) + 9}" y="${Y(b.ev.quality) + 14}" class="ink">${f1(b.ev.quality)} %</text>`;
      });
    });
    anc.forEach((a, ai) => {
      const col = a.model === 'v41' ? 'var(--s1)' : (a.engine === 'glmlite' ? 'var(--s3)' : 'var(--s2)');
      s += `<line x1="${X(a.meas)}" x2="${X(a.pred)}" y1="${Y(a.quality)}" y2="${Y(a.quality)}" stroke="${col}" stroke-width="1.5"/>`;
      const x = X(a.meas), y = Y(a.quality);
      s += `<path d="M${x} ${y - 7}L${x + 7} ${y}L${x} ${y + 7}L${x - 7} ${y}Z" fill="var(--surface-1)" stroke="${col}" stroke-width="2" data-k="anc" data-a="${ai}"/>`;
      s += `<circle cx="${X(a.pred)}" cy="${y}" r="2.5" fill="${col}" pointer-events="none"/>`;
    });
    if (R.owner) {
      const x = X(R.owner.ev.tokps), y = Y(R.owner.ev.quality);
      s += `<rect x="${x - 6}" y="${y - 6}" width="12" height="12" fill="var(--s7)" stroke="var(--surface-1)" stroke-width="2" data-k="owner" style="cursor:pointer"/><text x="${x + 9}" y="${y - 8}" class="ink">owner layout</text>`;
    }
    s += '</svg>';
    return { svg: s, anc };
  }

  // horizontal stacked bar as HTML (text keeps its size on any screen)
  // segs: [{w: fraction, col, text, attrs, hatch: 'r'|'n'|undefined, dark: bool}]
  function hbar(segs, h) {
    return `<div class="hbar" style="height:${h || 24}px">` + segs.filter(g => g.w > 0).map(g => {
      const bg = g.hatch === 'r' ? 'repeating-linear-gradient(45deg,var(--crit) 0 2px,var(--surface-2) 2px 6px)' : g.hatch === 'n' ? 'repeating-linear-gradient(135deg,var(--neutral) 0 2px,var(--surface-2) 2px 6px)' : g.col;
      return `<span style="flex:0 0 ${(100 * g.w).toFixed(3)}%;background:${bg}" ${g.attrs || ''}><i style="color:${g.hatch === 'r' ? 'var(--text-primary)' : g.ink || '#0b0b0b'}${g.hatch ? ';background:var(--surface-1);padding:1px 3px;border-radius:3px' : ''}">${esc(g.text || '')}</i></span>`;
    }).join('') + '</div>';
  }

  // ---------- waterfall ----------
  const typeColor = (q) => {
    const b = v(data.types[q].bpw);
    const steps = ['--r250', '--r350', '--r450', '--r550', '--r650', '--r700'];
    const i = b < 2.4 ? 0 : b < 3.0 ? 1 : b < 3.6 ? 2 : b < 4.4 ? 3 : b < 6 ? 4 : 5;
    return `var(${steps[i]})`;
  };
  const FIXCOL = { other: 'var(--neutral)', os: 'var(--neutral)', dense: 'var(--s2)', kv: 'var(--s3)', draft: 'var(--s4)', lookup: 'var(--s5)', margin: 'var(--crit)' };
  function idleReason(ctx, t, i, ev) {
    if (t.kind === 'gpu' && !t.engineOk) return ctx.E.name.split(' (')[0] + ' cannot drive this card';
    const maxQ = ev.kld <= ctx.kldDense + 1e-6;
    if (maxQ) return 'every expert already sits at its best type: nothing left to buy here';
    if (t.kind === 'gpu') return 'the fixed part fills this card; what remains is too small or the speed target is met without it';
    if (t.id === 'nvme' || t.id === 'pmem') return 'not needed: the faster tiers hold every expert this layout keeps';
    return 'left free: no move or type upgrade gains quality inside the speed target';
  }
  function tierRows(sel) {
    const { ctx, ev } = sel;
    const types = ctx.types;
    const copyBytes = ev.copyIdx >= 0 ? ctx.tiers.reduce((a, t, i) => a + (t.kind === 'gpu' ? ev.byType[i].reduce((x, y) => x + y, 0) : 0), 0) : 0;
    return ctx.tiers.map((t, i) => {
      const segs = [];
      t.reserves.forEach(r => { if (r.gib > 0.01) segs.push({ label: r.label, gib: r.gib, col: FIXCOL[r.kind] || 'var(--neutral)', tag: r.tag, src: r.src }); });
      if (t.usableOverride !== undefined) { /* reserve already listed */ }
      ctx.fixed.filter(x => x.tier === t.id).forEach(x => segs.push({ label: x.label, gib: x.gib, col: FIXCOL[x.kind], tag: x.tag }));
      types.forEach((q, qi) => { const b = ev.byType[i][qi]; if (b > 0) segs.push({ label: 'experts at ' + q + ' (' + v(data.types[q].bpw).toFixed(2) + ' bpw)', gib: b / GiB, col: typeColor(q), q }); });
      if (i === ev.copyIdx && copyBytes > 0) segs.push({ label: 'host copies of the GPU experts (prefill reads them)', gib: copyBytes / GiB, col: 'url(#hatch-n)', hatch: true });
      const used = segs.reduce((a, x) => a + x.gib, 0);
      const total = t.total / GiB;
      const idle = Math.max(0, total - used);
      if (idle > 0.05) segs.push({ label: 'idle: ' + idleReason(ctx, t, i, ev), gib: idle, col: 'url(#hatch-r)', idle: true });
      return { t, i, segs, total };
    });
  }
  function waterfallHTML(sel) {
    if (!sel) return '<p class="muted">No layout selected.</p>';
    const { ctx, ev } = sel;
    const rows = tierRows(sel);
    const box = ctx.box;
    let h = '';
    rows.forEach(r => {
      const { t, i } = r;
      const s = hbar(r.segs.map((g, gi) => ({ w: g.gib / r.total, col: g.col, hatch: g.idle ? 'r' : g.hatch ? 'n' : undefined, ink: g.q && v(data.types[g.q].bpw) >= 3.6 ? '#fff' : '#0b0b0b',
        text: g.idle ? 'idle ' + f1(g.gib) + ' GiB' : g.q ? g.q + ' ' + f1(g.gib) + ' GiB' : g.label.split(' (')[0] + ' ' + f1(g.gib), attrs: `data-k="seg" data-tier="${i}" data-seg="${gi}"` })));
      const bytes = ev.bytesTok[i], tms = ev.timeT[i] * 1000;
      const eff = tms > 0 ? bytes / (tms / 1000) / 1e9 : 0;
      let src = null;
      if (t.kind === 'gpu') src = (t.gpu && t.gpu.bw) || null;
      else if (t.id === 'dram') src = box.dram.bw;
      else if (t.id === 'pmem') src = t.bwTerm;
      else if (t.id === 'nvme') src = box.nvme.bw;
      const capSrc = t.kind === 'gpu' ? (t.gpu && t.gpu.vram_gib) : t.id === 'dram' ? box.dram.total_gib : t.id === 'pmem' ? box.pmem && box.pmem.gib : box.nvme.gib;
      const kernel = t.kind === 'cpu' ? ` CPU kernel ceiling ${f1(ctx.cpuRate(data.types.q4_K ? 'q4_K' : ctx.types[0]) / 1e9)} GB/s (q4_K, f ${f2(ctx.f)}${ctx.kern > 1 ? ', 2x kernel' : ''}${ctx.smt !== 1 ? ', SMT x' + f2(ctx.smt) : ''})` : t.kind === 'gpu' ? ` (x ${f2(t.eta)} reach${chip(data.assume.gpu_eta)})` : '';
      h += `<div class="tierbox"><div class="tierhead"><b>${esc(t.name)}</b><span class="small">${f1(r.total)} GiB${chip(capSrc)} · ${f1(t.bw / 1e9)} GB/s${chip(src)}${kernel}</span></div>${s}`;
      h += `<div class="small muted">${ev.cnt[i] ? Math.round(ev.cnt[i]).toLocaleString() + ' experts, ' + f1(ev.heatT[i] / ctx.kL * 100) + ' % of routed picks, ' : ''}reads ${f2(bytes / 1e9)} GB per token in ${f1(tms)} ms${eff ? ' (' + f1(eff) + ' GB/s effective)' : ''}</div>`;
      if (t.id === 'dram' && t.nodes && t.nodes.length > 1) {
        const usedFrac = r.segs.filter(g => !g.idle).reduce((a, g) => a + g.gib, 0) / r.total;
        h += '<div class="small muted">Per NUMA node (assumes the experts interleave evenly):</div>';
        t.nodes.forEach(nd => { h += `<div class="row small"><span style="min-width:130px">${esc(nd.name)} ${f1(nd.gib)} GiB</span><div style="flex:1;max-width:420px">${hbar([{ w: usedFrac, col: 'var(--r450)', text: '' }, { w: 1 - usedFrac, hatch: 'r', text: '' }], 10)}</div></div>`; });
      }
      h += '</div>';
    });
    // threads
    const thr = v(box.cpu.threads), used = st.knobs.smt ? thr : v(box.cpu.used);
    h += `<div class="tierbox"><div class="tierhead"><b>CPU threads</b><span class="small">${esc(box.cpu.desc)} · ${thr} threads${chip(box.cpu.threads)}</span></div>`;
    h += hbar([{ w: used / thr, col: 'var(--r450)', ink: '#fff', text: used + ' running the expert kernel', attrs: 'data-k="thr" data-u="1"' }, { w: (thr - used) / thr, hatch: 'r', text: thr > used ? (thr - used) + ' idle' : '', attrs: 'data-k="thr" data-u="0"' }]);
    h += `<div class="small muted">${used < thr ? 'idle: ' + (thr - used) + ' threads. ' : ''}${esc(box.threads_note || '')}. SMT measured effect x${f2(v(box.cpu.smt_mult))}${chip(box.cpu.smt_mult)}</div></div>`;
    return h;
  }
  function wfLegend(sel) {
    if (!sel) return '';
    const items = [['var(--neutral)', 'other users / OS'], ['var(--s2)', 'non-expert weights'], ['var(--s3)', 'KV cache + compute'], ['var(--s4)', 'draft / MTP'], ['var(--s5)', 'lookup tables']];
    const ts = sel.ctx.types.filter((q, qi) => sel.ev.byType.some(r => r[qi] > 0));
    let h = '<div class="legend">' + items.map(([c, l]) => `<span><span class="sw" style="background:${c}"></span>${l}</span>`).join('');
    h += ts.map(q => `<span><span class="sw" style="background:${typeColor(q)}"></span>experts ${esc(q)}</span>`).join('');
    h += `<span><span class="sw" style="background:repeating-linear-gradient(135deg,var(--neutral) 0 2px,transparent 2px 5px)"></span>host copies</span>`;
    h += `<span><span class="sw" style="background:repeating-linear-gradient(45deg,var(--crit) 0 2px,transparent 2px 5px)"></span>idle (reason on hover)</span></div>`;
    return h;
  }

  // ---------- time split + bits by heat ----------
  function timeSplitHTML(sel) {
    const { ctx, ev } = sel;
    const parts = [['host and fixed overhead', ev.host * 1000, 'var(--neutral)'], ['non-expert weights on the GPU', ev.dense * 1000, 'var(--s2)']];
    ctx.tiers.forEach((t, i) => { if (ev.timeT[i] > 0) parts.push(['experts in ' + t.name, ev.timeT[i] * 1000, t.kind === 'gpu' ? 'var(--r650)' : t.id === 'dram' ? 'var(--r450)' : 'var(--r250)']); });
    const tot = parts.reduce((a, p) => a + p[1], 0);
    const s = hbar(parts.map((p, pi) => ({ w: p[1] / tot, col: p[2], ink: p[2].indexOf('r650') >= 0 ? '#fff' : '#0b0b0b', text: f1(p[1]) + ' ms', attrs: `data-k="time" data-p="${pi}"` })));
    const ov = ctx.overlap ? ` GPU and CPU work overlap (factor ${f2(ctx.overlap)}), so the token takes ${f1(ev.T * 1000)} ms, not the sum.` : ` GPU and CPU work run one after the other.`;
    return { svg: s, parts, text: `${parts.map(p => esc(p[0]) + ' ' + f1(p[1]) + ' ms').join(' · ')}.${ov} Draft/MTP multiplier x${f2(ctx.mult)}.` };
  }
  function bitsByHeatSVG(sel) {
    const { ctx, lay } = sel;
    const N = ctx.order.length, nb = 48, per = Math.ceil(N / nb);
    const series = {};
    for (let b = 0; b < nb; b++) {
      const ids = ctx.order.slice(b * per, (b + 1) * per);
      const acc = {};
      ids.forEach(e => { const t = lay.tier[e]; (acc[t] = acc[t] || []).push(v(data.types[ctx.types[lay.type[e]]].bpw)); });
      Object.keys(acc).forEach(t => { (series[t] = series[t] || []).push({ b, bpw: acc[t].reduce((a, x) => a + x, 0) / acc[t].length, n: acc[t].length }); });
    }
    const W = Math.round(Math.max(320, (app.clientWidth || 720) - 40)), H = 200, L = 40, T = 10, B = 34, Rm = 10;
    const bmax = Math.max(4.5, ...Object.values(series).flat().map(p => p.bpw));
    const X = b => L + (W - L - Rm) * (b + 0.5) / nb, Y = y => T + (H - T - B) * (1 - (y - 1.5) / (bmax - 1.5));
    const tierCol = t => { const k = ctx.tiers[t]; return k.kind === 'gpu' ? 'var(--s1)' : k.id === 'dram' ? 'var(--s2)' : k.id === 'pmem' ? 'var(--s3)' : 'var(--neutral)'; };
    let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="width:${W}px;max-width:100%" aria-label="bits per weight by expert heat rank">`;
    for (let y = 2; y <= bmax; y += 1) s += `<line x1="${L}" x2="${W - Rm}" y1="${Y(y)}" y2="${Y(y)}" stroke="var(--grid)"/><text x="${L - 5}" y="${Y(y) + 4}" text-anchor="end">${y}</text>`;
    s += `<text x="${L}" y="${H - 8}">hottest experts</text><text x="${W - Rm}" y="${H - 8}" text-anchor="end">coldest (rarely or never routed)</text><text x="${(L + W) / 2}" y="${H - 8}" text-anchor="middle">heat rank (plan heat), ${nb} bins</text>`;
    Object.keys(series).forEach(t => series[t].forEach(p => {
      const r = 1.5 + 3.5 * Math.sqrt(p.n / per);
      s += `<circle cx="${X(p.b) + (+t - ctx.tiers.length / 2) * 3}" cy="${Y(p.bpw)}" r="${r}" fill="${tierCol(t)}" stroke="var(--surface-1)" stroke-width="1.5" data-k="bh" data-t="${t}" data-b="${p.b}" data-bpw="${p.bpw}" data-n="${p.n}"/>`;
    }));
    s += '</svg>';
    const leg = '<div class="legend">' + Object.keys(series).map(t => `<span><span class="sw" style="background:${tierCol(t)}"></span>${esc(ctx.tiers[t].name)}</span>`).join('') + '<span class="muted">dot size = share of the bin</span></div>';
    // callout: average bpw per tier and the PMem-vs-DRAM comparison
    const ev = sel.ev;
    const avg = ctx.tiers.map((t, i) => ({ t, i, bpw: ev.avgBpw[i], n: ev.cnt[i], heat: ev.cnt[i] ? ev.heatT[i] / ev.cnt[i] : 0 })).filter(x => x.n > 0);
    const dr = avg.find(x => x.t.id === 'dram'), pm = avg.find(x => x.t.id === 'pmem' || x.t.id === 'nvme');
    let call = 'Average bits per weight by tier: ' + avg.map(x => `${esc(x.t.name)} ${f2(x.bpw)} (mean heat ${(x.heat * 1000).toFixed(2)} per 1000 tokens)`).join(' · ') + '. ';
    // the coldest fifth of the DRAM experts vs the (even colder) slow-tier experts
    let coldDr = null;
    if (dr && pm) {
      const ids = ctx.order.filter(e => lay.tier[e] === dr.i);
      const tail = ids.slice(Math.floor(ids.length * 0.8));
      if (tail.length) coldDr = tail.reduce((a, e) => a + v(data.types[ctx.types[lay.type[e]]].bpw), 0) / tail.length;
    }
    if (dr && pm && coldDr !== null && pm.bpw > coldDr + 0.05) call += `<b>The rarely used experts in ${esc(pm.t.name)} sit at HIGHER bits (${f2(pm.bpw)}) than the coldest fifth of the DRAM experts (${f2(coldDr)}), although they are used even less.</b> In DRAM a byte costs read time plus DRAM space, and DRAM is full, so its coldest experts get squeezed. In ${esc(pm.t.name)} space is free; there both the read time and the quality cost of an expert scale with how often it is used, so its type is set only by the ${esc(pm.t.name)} read rate against quality per byte, the same for every expert in it. The hottest DRAM experts still get more bits (DRAM reads about ${f1(ctx.rate(ctx.tiers[dr.i], 'q4_K') / ctx.rate(ctx.tiers[pm.i], 'q4_K'))}x faster per byte).`;
    else if (dr && pm) call += `In this layout the slow-tier experts (${f2(pm.bpw)} bpw) are not above even the coldest DRAM experts (${coldDr === null ? '-' : f2(coldDr)}). That happens when DRAM still has room (no space price to squeeze its cold experts), when every type already sits at the top (V4.1 MXFP4 is lossless, so nothing above it gains), or when the engine forces one type per layer.`;
    if (!ctx.perExpert) call += ' <span class="muted">Types are one per layer here (GGUF today); turn on "per-expert types" to let a cold expert in one layer differ from a hot one.</span>';
    return { svg: s, leg, call };
  }

  // ---------- main render ----------
  function render() {
    const sel = R.selected;
    const box = data.boxes[st.box];
    const kn = st.knobs;
    let h = '';
    h += `<h1>Model fit planner</h1><div class="sub">The smartest GLM-5.3-Flash or DeepSeek V4.1 Flash each box can run at 20+ or 25+ tok/s. Built ${esc(data.meta.built)} from measured runs and community priors. Every number carries a tag; hover it for the source.</div>`;
    const EV = data.meta.evidence;
    h += `<div class="legend" style="margin-top:6px">${Object.keys(EV.classes).map(k => `<span>${gradeChip(k, EV.classes[k])} ${esc(EV.classes[k])} (${EV.counts[k] || 0} values)</span>`).join('')}</div>`;
    // controls
    h += '<div class="card"><div class="row">';
    h += `<label>Box <select id="c-box">${Object.keys(data.boxes).map(b => `<option value="${b}" ${b === st.box ? 'selected' : ''}>${esc(data.boxes[b].name)}</option>`).join('')}</select></label>`;
    MODELS.forEach(mk => {
      const m = data.models[mk], hs = heatFor(mk);
      const opts = n => Object.keys(m.heat.sets).map(k => `<option value="${k}" ${k === n ? 'selected' : ''}>${esc(m.heat.sets[k].label)}</option>`).join('');
      h += `<label class="small">${esc(m.short)} plan heat <select data-heat="${mk}" data-w="plan">${opts(hs.plan)}</select></label><label class="small">scored on <select data-heat="${mk}" data-w="score">${opts(hs.score)}</select></label>`;
    });
    h += `<span class="small">Theme ${['auto', 'light', 'dark'].map(t => `<button data-theme-b="${t}" class="${st.theme === t ? 'on' : ''}">${t}</button>`).join(' ')}</span>`;
    h += '</div><div class="small muted" style="margin-top:4px">' + esc(box.role) + '. Plan heat = the routing counts the layout is built from; scored on = the counts used to project its speed and quality (a held-out half avoids grading the plan on its own data).</div></div>';
    // what-if
    const upD = box.upgrades && box.upgrades.dram, upP = box.upgrades && box.upgrades.pmem;
    h += '<div class="card"><h2>What if</h2><div class="row">';
    const cb = (k, label, title, dis) => `<label title="${esc(title || '')}"><input type="checkbox" data-knob="${k}" ${kn[k] ? 'checked' : ''} ${dis ? 'disabled' : ''}> ${label}</label>`;
    h += cb('perExpertTypes', 'per-expert types', 'Today a GGUF stores one type per layer tensor. Per-expert types need the split-expert ik build (lab notes).');
    h += cb('freeDesktop', 'free the snoopy desktop VRAM', 'the compositor leaves the 3090 Ti (display off, or the desktop moved to the iGPU)', st.box !== 'snoopy');
    h += cb('freeHousehold', 'free odie B70 #2 (resident service off)', 'Both B70s for research: the resident model on B70 #2 stops for the window (book the window first)', st.box !== 'odie');
    h += `<label title="dram-fill placement: the same free headroom on every NUMA node. 10 GiB with 3.0 GiB banks = MEASURED-INSUFFICIENT on droopy V4.1 (B1b: node0 MemFree 0.5 GiB, prefill 10-12); 16 = MEASURED-INSUFFICIENT on node0 and slower (B1c); 10 = current setting with swap off and B0's 2.2 GiB banks (B0 nortr ran clean). Empty = the box's usable DRAM figure">node headroom GiB <input type="number" step="1" min="1" max="64" data-knob-n="numaHeadroom" value="${kn.numaHeadroom === null || kn.numaHeadroom === undefined ? '' : kn.numaHeadroom}" placeholder="usable" ${(box.dram.nodes || []).some(x => x.avail_gib) ? '' : 'disabled'}></label>`;
    h += `<label>move odie's B70 #1 into <select data-knob-s="moveB70">${['none', 'snoopy', 'droopy'].map(x => `<option ${kn.moveB70 === x ? 'selected' : ''}>${x}</option>`).join('')}</select></label>`;
    h += cb('addDRAM', 'add DRAM' + (upD ? ' (' + esc(upD.what) + ')' : ''), upD ? upD.gib.src : 'no DRAM upgrade listed for this box', !upD);
    h += cb('addOptane', 'add Optane' + (upP ? ' (' + esc(upP.what) + ')' : ''), upP ? upP.gib.src : 'only droopy takes Optane', !upP);
    h += cb('kernel2x', '2x faster CPU expert kernel', 'code: a kernel that reaches the memory ceiling. K1 (droopy, 10-17) measured the per-type kernel shares now in the type table; on snoopy the AVX-512 GLM lane already sits at the DRAM roofline (0.167 ms/expert K=3, 59.5 GB/s), so 2x there is not available from kernel work alone');
    h += cb('smt', 'SMT threads on', 'measured on snoopy only (x0.88 on V4.0)');
    h += cb('draft', 'draft / MTP on', '');
    h += `<label title="empty = the fitted or default multiplier for this box">draft speed-up <input type="number" step="0.05" min="1" max="3" data-knob-n="draftMult" value="${kn.draftMult === null || kn.draftMult === undefined ? '' : kn.draftMult}" placeholder="auto"></label>`;
    h += cb('hostCopy', 'keep host copies of GPU experts', 'ik prefill reads a host copy of every GPU-resident expert');
    h += `<label>non-expert weights at <select data-knob-s="densePrec">${['q8_0', 'q6_K', 'q4_K'].map(x => `<option ${kn.densePrec === x ? 'selected' : ''}>${x}</option>`).join('')}</select></label>`;
    h += cb('rtr', 'CPU repack at load (-rtr)', 'Allows the load-time repacked MXFP4 (mxfp4_r8). Measured on droopy B0: no decode gain, prefill 3x slower on a split model. ' + data.types.mxfp4_r8.note);
    h += `<label title="-ub 4096 grows the compute buffer (V4.1 B0: 5.4 GiB KV + compute); auto drops to 512 when the GPU cannot hold it">batch -ub <select data-knob-s="ub">${['auto', '4096', '512'].map(x => `<option ${String(kn.ub) === x ? 'selected' : ''}>${x}</option>`).join('')}</select></label>`;
    h += `<label title="VRAM transient margin above the post-warm-up reading. safe = the measured upper bound (droopy ik split: ${data.checks.vram_peak.margin.ik_split.hi} GiB; B1b ran prompts at 21.56 of 23.04 in nvidia-smi MiB / 1000 without OOM); likely = the B0 #1 graph-growth candidate">VRAM peak margin <select data-knob-s="vramMargin">${['safe', 'likely'].map(x => `<option ${kn.vramMargin === x ? 'selected' : ''}>${x}</option>`).join('')}</select></label>`;
    h += `<button id="c-reset">reset knobs</button><span class="busy" id="busy">computed in ${Math.round(R.ms)} ms</span></div>`;
    if (kn.moveB70 !== 'none') h += `<div class="warnline">${esc(data.b70move.gpu.note)}</div>`;
    h += '</div>';
    // frontier + selection summary
    const fr = frontierSVG();
    h += '<div class="grid2"><div class="card"><h2>Quality vs speed on ' + esc(box.name) + '</h2>';
    h += '<div class="legend">' + R.curves.map(c => `<span><span class="sw" style="background:${c.color}"></span>${esc(data.models[c.mk].short + ' ' + engShort(c.eng))}${c.dash ? ' (dashed)' : ''}</span>`).join('') + '<span>◇ measured, · model (line = residual)</span>' + (R.owner ? '<span><span class="sw" style="background:var(--s7)"></span>owner layout</span>' : '') + '</div>';
    h += fr.svg;
    h += `<div class="small muted">Each curve sweeps the speed target from slow to fast and keeps the best-quality layout. Big dots = best at 20 and at 25 tok/s. Click any point to load its layout below. <button id="c-table">${st.table ? 'hide' : 'show'} table</button></div>`;
    if (st.table) h += frontierTable(fr.anc);
    h += '</div><div class="card">' + selectionHTML(sel) + '</div></div>';
    // waterfall
    h += '<div class="card"><h2>Where every byte sits</h2>' + (sel ? `<div class="sub">${esc(sel.label)}</div>` : '') + wfLegend(sel) + waterfallHTML(sel) + '</div>';
    if (sel) {
      const bh = bitsByHeatSVG(sel);
      h += `<div class="card"><h2>Bits per weight by how often an expert is used</h2>${bh.leg}${bh.svg}<div class="callout small">${bh.call}</div></div>`;
    }
    h += trustHTML() + checksCardHTML() + calHTML() + tablesHTML() + ideaHTML() + sourcesHTML();
    app.innerHTML = h;
    bind(fr.anc);
  }

  function selectionHTML(sel) {
    if (!sel) return '<h2>Selected layout</h2><p class="muted">Nothing reaches a feasible layout on this box with these knobs.</p>';
    const { ctx, ev } = sel;
    const ts = timeSplitHTML(sel);
    let bits = 0; for (let e = 0; e < sel.lay.type.length; e++) bits += v(data.types[ctx.types[sel.lay.type[e]]].bpw);
    const avg = bits / sel.lay.type.length;
    let h = `<h2>Selected layout</h2><div class="sub">${esc(sel.label)}</div><div class="kpi" style="margin:8px 0">`;
    h += `<div><b>${f1(ev.tokps)}</b>tok/s projected</div><div><b>${f1(ev.quality)} %</b>top-1 agreement${chip(data.assume.quality_a)}</div><div><b>${ev.kld.toFixed(3)}</b>KLD (routing-weighted)</div><div><b>${f2(avg)}</b>expert bpw (mean)</div></div>`;
    if (avg < v(data.assume.cliff_bpw)) h += `<div class="warnline">Mean expert bits ${f2(avg)} is below ${v(data.assume.cliff_bpw)}: ${esc(data.assume.cliff_bpw.src)}</div>`;
    h += ts.svg + `<div class="small muted">${ts.text}</div>`;
    const needs = [];
    if (ctx.perExpert || sel.needsCode) needs.push('per-expert types need the split-expert ik build');
    ctx.tiers.forEach((t, i) => { if (t.gpu && t.gpu.needs_code && t.gpu.needs_code[ctx.eng] && sel.ev.cnt[i] > 0) needs.push(t.gpu.needs_code[ctx.eng]); });
    if (ctx.eng === 'sycl' && ctx.mk === 'v41') needs.push('V4.1 support in mainline llama.cpp SYCL is UNKNOWN');
    if (st.box === 'lassie') needs.push('lassie 1080 Ti: nvidia-smi reports a driver/library mismatch (probe 2026-10-17); fix the driver first');
    if (needs.length) h += `<div class="warnline">Needs: ${needs.map(esc).join('; ')}.</div>`;
    if (ctx.notes.length) h += `<div class="small muted">${ctx.notes.map(esc).join(' ')}</div>`;
    h += checksHTML(FM.checksFor(ctx, sel.lay, ev));
    h += `<div class="row" style="margin-top:8px"><button id="c-export">download expert map (JSON)</button><span class="small muted">tier + type for every routed expert, for a loader or a --custom-q recipe</span></div>`;
    return h;
  }
  const CHKCOL = { PASS: 'var(--s2)', FAIL: 'var(--crit)', UNKNOWN: 'var(--neutral)' };
  function checksHTML(cs) {
    return '<div class="small" style="margin-top:6px"><b>Hard checks</b> ' + cs.map(c => `<span title="${esc(c.detail || '')}" style="display:inline-block;margin:2px 6px 2px 0;padding:1px 6px;border-radius:3px;border:1px solid ${CHKCOL[c.status]}">${esc(c.id)} ${esc(c.status)}</span>`).join('') +
      cs.filter(c => c.status !== 'PASS').map(c => `<div class="${c.status === 'FAIL' ? 'warnline' : 'muted'}">${esc(c.id)}: ${esc(c.detail)}</div>`).join('') + '</div>';
  }
  // trust: every recipe's predicted numbers with their evidence roll-up (the --check trust lines), the external class gaps
  // and the bench-fitted terms. Recipe predictions are cached per set of type-table edits
  let trustCache = null;
  function trustRows() {
    const key = JSON.stringify(data.overrides);
    if (trustCache && trustCache.key === key) return trustCache.rows;
    const rows = (data.recipes || []).filter(r => r.recipe).map(r => {
      const res = FM.checkRecipe(data, r.recipe, null);
      return { r, ev: (res.prediction && res.prediction.evidence) || [], why: res.prediction && res.prediction.why };
    });
    trustCache = { key, rows };
    return rows;
  }
  function rollupHTML(e) {
    const num = x => x === null || x === undefined ? 'UNMODELLED' : Math.abs(x) >= 100 ? x.toFixed(0) : x.toFixed(2);
    const terms = e.terms.map(t => `<tr><td>${esc(t.term)}</td><td class="n">${t.value === null ? '-' : esc(typeof t.value === 'number' ? num(t.value) : t.value)}</td><td>${chip(t)}</td><td class="small muted">${esc(t.src || '')}</td></tr>`).join('');
    const x = e.external, ext = x && x.n ? ` · external ${esc(x.group)}: off by median ${x.median_abs_pct.toFixed(0)} % (bias ${(x.bias_pct >= 0 ? '+' : '') + x.bias_pct.toFixed(0)} %, ${x.n} rows)` : '';
    return `<details><summary>${esc(e.metric)} <b>${num(e.value)}</b> ${esc(e.unit || '')} ${gradeChip(e.grade, 'the weakest class this number rests on')} <span class="small muted">${e.seen} seen, ${e.inferred} inferred, ${e.unknown} unknown terms${ext}</span></summary>` +
      `<div class="small" style="margin:4px 0">${esc(FM.fmtRollup(e, 120))}</div><div class="scroll"><table><tr><th>term</th><th class="n">value</th><th>evidence</th><th>source</th></tr>${terms}</table></div></details>`;
  }
  function trustHTML() {
    const EV = data.meta.evidence, S = FM.externalSummary(data), B = data.bench || {};
    let h = '<div class="card"><h2>Which numbers to trust</h2><div class="small muted">Every number has one evidence field: <b>seen</b> = our own measurement (a run, bench or probe on our boxes, or a read of our own file), <b>inferred</b> = fitted, estimated, or from a spec or community post, <b>unknown</b> = not modelled. Hover a chip for its tag and source. Each prediction below takes the weakest class of the terms it rests on; open it for every term.</div>';
    h += `<div class="small" style="margin:6px 0">Catalog: ${Object.keys(EV.classes).map(k => `${EV.counts[k] || 0} ${gradeChip(k, EV.classes[k])}`).join(' · ')}</div>`;
    h += '<h3>Recipe predictions (trust lines)</h3>';
    trustRows().forEach(({ r, ev, why }) => {
      h += `<div style="margin:6px 0"><div><b>${esc(r.box)}</b> ${esc(r.label || r.recipe.id)}</div>` + (ev.length ? ev.map(rollupHTML).join('') : `<div class="small muted">UNMODELLED: ${esc(why || 'no prediction')}</div>`) + '</div>';
    });
    h += '<h3>Bench-fitted terms (runs/bench.jsonl)</h3><div class="scroll"><table><tr><th>backend</th><th>cards</th><th class="n">reach of nameplate</th><th class="n">host ms/token</th><th class="n">MoE ms per layer</th></tr>';
    Object.keys(B.gpu || {}).forEach(k => { const g = B.gpu[k]; h += `<tr><td>${esc(k)}</td><td>${esc(g.cards.join(', ') || 'borrowed from ' + g.borrowed_from)}</td><td class="n">${f2(v(g.eta))}${chip(g.eta)}</td><td class="n">${f2(v(g.host_ms))}${chip(g.host_ms)}</td><td class="n">${v(g.moe_layer_ms) === null ? '-' : v(g.moe_layer_ms).toFixed(3)}${chip(g.moe_layer_ms)}</td></tr>`; });
    h += '</table></div><div class="scroll"><table><tr><th>box</th><th>NUMA placement (policy, weight nodes, thread nodes, sockets, threads)</th><th class="n">decode vs one local node</th></tr>';
    Object.keys(B.numa || {}).forEach(b => Object.keys(B.numa[b]).forEach(k => { h += `<tr><td>${esc(b)}</td><td>${esc(k)}</td><td class="n">${f2(v(B.numa[b][k]))}x${chip(B.numa[b][k])}</td></tr>`; }));
    h += '</table></div><h3>External check (community rows: never a fitted term)</h3><div class="scroll"><table><tr><th>class / placement</th><th class="n">rows</th><th class="n">median |error|</th><th class="n">bias</th></tr>';
    S.by_class_placement.forEach(g => { h += `<tr><td>${esc(g.key)}</td><td class="n">${g.n}</td><td class="n">${g.median_abs_pct.toFixed(0)} %</td><td class="n">${(g.bias_pct >= 0 ? '+' : '') + g.bias_pct.toFixed(0)} %</td></tr>`; });
    h += `</table></div><div class="small muted">${S.unmodelled.length} rows UNMODELLED: ${esc(S.unmodelled.map(u => u.id + ' (' + u.why + ')').join('; '))}</div>`;
    return h + '</div>';
  }
  function checksCardHTML() {
    let h = '<div class="card"><h2>Hard checks against the 10-17 runs</h2><div class="small muted">Each recipe is checked for: storage tier of on-file tensors (never a rotational disk), VRAM peak (post-warm-up + transient margin, not the warm-up reading), the -rtr prefill penalty on split models, DRAM headroom per NUMA node (node MemFree after experts + non-expert interleave + host buffers + page cache must stay >= 3 GiB), the box P0 settings (numa_balancing, demotion, swap), and the 2-bit read-share quality gate. Recorded windows show what a run cost.</div>';
    h += '<div class="scroll"><table><tr><th>when</th><th>run</th><th>outcome</th><th>storage</th><th>VRAM peak</th><th>NUMA free</th><th>P0</th><th>-rtr</th><th>quality</th><th>cost</th></tr>';
    FM.checkRecipes(data).forEach(r => {
      const by = Object.fromEntries(r.checks.map(c => [c.id, c]));
      const cell = id => { const c = by[id]; return c ? `<td title="${esc(c.detail || '')}" style="color:${CHKCOL[c.status]};font-weight:${c.status === 'FAIL' ? 700 : 400}">${esc(c.status)}</td>` : '<td>-</td>'; };
      h += `<tr><td>${esc(r.when)}</td><td>${esc(r.box + ' ' + r.label)}</td><td class="small">${esc(r.outcome)}</td>${cell('storage')}${cell('vram_peak')}${cell('numa_free')}${cell('system')}${cell('rtr')}${cell('quality')}<td class="small muted">${esc(r.wasted || '')}</td></tr>`;
    });
    h += '</table></div><h3>Measured levers (10-17)</h3><div class="scroll"><table><tr><th>box</th><th>lever</th><th>before</th><th>after</th><th>source</th></tr>';
    (data.levers || []).forEach(l => { h += `<tr><td>${esc(l.box + ' ' + l.model)}</td><td>${esc(l.lever)}</td><td>${esc(l.before)}</td><td>${esc(l.after)}${tagChip(l.tag, l.src)}</td><td class="small muted">${esc(l.src)}</td></tr>`; });
    return h + '</table></div></div>';
  }
  function frontierTable(anc) {
    let h = '<div class="scroll"><table><tr><th>curve</th><th class="n">tok/s</th><th class="n">top-1 %</th><th class="n">KLD</th><th>kind</th></tr>';
    R.curves.forEach(c => {
      const nm = esc(data.models[c.mk].short + ' ' + engShort(c.eng));
      envelope(c.pts).forEach(p => { h += `<tr><td>${nm}</td><td class="n">${f2(p.tokps)}</td><td class="n">${f2(p.quality)}</td><td class="n">${p.kld.toFixed(4)}</td><td>sweep</td></tr>`; });
      if (c.b20) h += `<tr><td>${nm}</td><td class="n">${f2(c.b20.ev.tokps)}</td><td class="n">${f2(c.b20.ev.quality)}</td><td class="n">${c.b20.ev.kld.toFixed(4)}</td><td>best at 20</td></tr>`;
      if (c.b25) h += `<tr><td>${nm}</td><td class="n">${f2(c.b25.ev.tokps)}</td><td class="n">${f2(c.b25.ev.quality)}</td><td class="n">${c.b25.ev.kld.toFixed(4)}</td><td>best at 25</td></tr>`;
    });
    anc.forEach(a => { h += `<tr><td>${esc(a.label)}</td><td class="n">${f2(a.meas)} measured / ${f2(a.pred)} model</td><td class="n">${f2(a.quality)}</td><td></td><td>anchor ${esc(a.fit)}</td></tr>`; });
    return h + '</table></div>';
  }
  function calHTML() {
    let h = '<div class="card"><h2>Calibration: measured runs vs the model</h2><div class="small muted">Fit per box: a CPU efficiency factor f (shared by both models), a host overhead per model family, and on droopy the Optane read rate inside its measured band. Draft rows use the measured speed-up against their no-draft twin. "check" and "ref" rows are not fitted.</div>';
    h += '<div class="scroll"><table><tr><th>box</th><th>CPU factor f</th><th>host ms/token</th><th>draft speed-up</th><th>Optane rate</th><th class="n">rms log error</th></tr>';
    Object.keys(data.cal.boxes).forEach(bk => {
      const c = data.cal.boxes[bk];
      h += `<tr><td>${esc(bk)}</td><td>${f2(v(c.f))}${chip(c.f)}</td><td>${Object.keys(c.host).filter(m => MODELS.indexOf(m) >= 0 || m === 'v40').map(m => esc(m) + ' ' + f1(v(c.host[m])) + chip(c.host[m])).join(' ')}${c.split_ms ? ' · split GGUF +' + f1(v(c.split_ms)) + chip(c.split_ms) : ''}</td><td>${Object.keys(c.draft).map(m => esc(m) + ' x' + f2(v(c.draft[m])) + chip(c.draft[m])).join(' ') || '-'}</td><td>${c.pmem_bw ? f1(v(c.pmem_bw)) + ' GB/s' + chip(c.pmem_bw) : '-'}</td><td class="n">${c.rms !== undefined ? c.rms.toFixed(3) : '-'}</td></tr>`;
    });
    h += '</table></div><div class="scroll"><table><tr><th>run</th><th>use</th><th class="n">measured tok/s</th><th class="n">model</th><th class="n">residual</th><th>source</th></tr>';
    data.cal.rows.forEach(r => { h += `<tr><td>${esc(r.label)}</td><td>${esc(r.fit)}</td><td class="n">${f2(r.meas)}${tagChip(r.tag, r.src)}</td><td class="n">${f2(r.pred)}</td><td class="n">${(100 * r.resid).toFixed(1)} %</td><td class="small muted">${esc(r.src)}${r.mult_src ? ' · speed-up ' + f2(r.mult) + ' = ' + esc(r.mult_src) : ''}</td></tr>`; });
    return h + '</table></div></div>';
  }
  function tablesHTML() {
    const tk = Object.keys(data.types);
    let h = '<div class="card"><h2>Type table (editable)</h2><div class="small muted">CPU efficiency = share of the memory ceiling the CPU expert kernel reaches with this type (odie AVX2 smoke). KLD = full-model KL divergence if every routed expert used this type (community priors where a row exists). Edits recalibrate and recompute; they stay in this browser.</div>';
    h += '<div class="scroll"><table><tr><th>type</th><th class="n">bpw</th><th class="n">CPU efficiency</th><th class="n">KLD V4.1</th><th class="n">KLD GLM</th><th class="n">KLD Flash-Next</th><th>engines</th><th>note</th></tr>';
    tk.forEach(q => {
      const t = data.types[q];
      const kap = data.overrides.kappa[q] !== undefined ? data.overrides.kappa[q] : v(t.kappa);
      const kv = fam => { const o = data.overrides.kld[fam] && data.overrides.kld[fam][q]; return o !== undefined ? o : v(t.kld[fam]); };
      const cell = (fam) => t.kld[fam] && v(t.kld[fam]) !== null ? `<input type="number" step="0.001" min="0" data-kld="${fam}" data-q="${q}" value="${kv(fam)}">${chip(t.kld[fam])}` : '-';
      h += `<tr><td>${esc(q)}</td><td class="n">${v(t.bpw).toFixed(3)}${chip(t.bpw)}</td><td class="n">${kap === null || kap === undefined ? '-' : `<input type="number" step="0.01" min="0.05" max="2" data-kappa="${q}" value="${(+kap).toFixed(3)}">`}${chip(t.kappa)}</td><td class="n">${cell('v41')}</td><td class="n">${cell('glm')}</td><td class="n">${cell('fnx')}</td><td>${esc((t.engines || []).join(', '))}</td><td class="small muted">${esc(t.note || '')}</td></tr>`;
    });
    h += `</table></div><button id="c-reset-ov">reset edits</button> <span class="small muted">Quality curve: top-1 % = 100 - ${v(data.assume.quality_a).toFixed(1)} x KLD^${v(data.assume.quality_b).toFixed(3)}${chip(data.assume.quality_a)}</span></div>`;
    return h;
  }
  function ideaHTML() {
    return `<div class="card"><h2>My idea</h2><div class="small muted">Write the idea; the download carries it with every knob, the box and the selected layout's numbers. It stays in this browser too.</div><textarea id="c-idea" placeholder="e.g. put the 1080 Ti in droopy beside the RTX 6000 and give it the next 200 hottest experts">${esc(load('idea', ''))}</textarea><button id="c-idea-dl">download idea (JSON)</button></div>`;
  }
  function sourcesHTML() {
    const sb = data.siblings || {};
    let h = '<div class="card"><h2>Sources and community recipes</h2>';
    const rec = sb.quant_recipes && sb.quant_recipes.data;
    if (rec && rec.rules) {
      h += '<h3>Quant recipe rules (quant-recipes.json, ' + esc(sb.quant_recipes.mtime) + ')</h3><ol class="small">';
      rec.rules.forEach(r => { h += `<li><b>${esc(r.rule)}</b> <span class="muted">(${esc(r.strength)})</span></li>`; });
      h += '</ol>';
    }
    const pr = sb.quant_priors && sb.quant_priors.data;
    if (pr) {
      if (pr.harness_warning) h += `<div class="warnline small">${esc(pr.harness_warning)}</div>`;
      if (pr.inferred) {
        h += '<h3>Inferred expectations for mixed layouts (quant-priors.json)</h3><ul class="small">';
        const fmt = o => Object.keys(o).map(k => k.replace(/_/g, ' ') + ' ' + (o[k] && typeof o[k] === 'object' && !Array.isArray(o[k]) ? '(' + fmt(o[k]) + ')' : Array.isArray(o[k]) ? o[k].join('-') : o[k])).join(', ');
        pr.inferred.forEach(x => { h += `<li><b>${esc(x.target)}</b>: ${esc(fmt(x.expect || {}))}</li>`; });
        h += '</ul>';
      }
      if (pr.skip) { h += '<h3>Skip list</h3><ul class="small">' + pr.skip.map(x => `<li><b>${esc(x.what)}</b>: ${esc(x.why)}</li>`).join('') + '</ul>'; }
    }
    if (sb.quant_recipes_md) h += `<details><summary>QUANT-RECIPES.md</summary><pre>${esc(sb.quant_recipes_md)}</pre></details>`;
    if (sb.quant_priors_md) h += `<details><summary>QUANT-PRIORS.md</summary><pre>${esc(sb.quant_priors_md)}</pre></details>`;
    h += `<details><summary>How the model works (assumptions)</summary><div class="small">
<p>Time per token = host overhead + non-expert weights read on the GPU + for every expert: how often it is routed x its bytes / the tier's effective rate. GPU rate = nameplate bandwidth x ${f2(v(data.assume.gpu_eta))}${chip(data.assume.gpu_eta)}. CPU-side rate = the smaller of the tier bandwidth and the CPU kernel rate (DRAM bandwidth x type efficiency x the box factor f). Optane uses the same min(): the Qwen sweep ran 3.33-3.68 tok/s on Optane for every type, so the type does not matter there (a deliberate departure from multiplying Optane bandwidth by the type efficiency). Speed = draft multiplier / time.</p>
<p>Quality = routing-weighted KLD (each expert's heat x its type's KLD, over the k picks per layer) plus the non-expert part, mapped to top-1 agreement with a curve fitted to community (KLD, top-1) pairs. KLD numbers from different harnesses differ by up to 10x; V4.1 types use one ladder (AtomicChat V4 Flash, delta over lossless MXFP4) and GLM types the unsloth ladder, with turboderp EXL3 numbers mapped through two GGUF files measured in both.</p>
<p>Solver: experts are grouped by plan heat; a Lagrangian price per tier keeps every tier inside its capacity, and a sweep over the speed-vs-quality weight draws the curve. The best point at a target is refined per expert: promote into leftover faster capacity, then upgrade types (whole layers in GGUF mode) while the target holds. Projections are scored on the held-out heat.</p>
</div></details>`;
    return h + '</div>';
  }

  // ---------- events ----------
  function showTip(e, html) { tip.innerHTML = html; tip.style.display = 'block'; const x = Math.min(e.clientX + 14, window.innerWidth - 370), y = Math.min(e.clientY + 14, window.innerHeight - tip.offsetHeight - 8); tip.style.left = x + 'px'; tip.style.top = y + 'px'; }
  function hideTip() { tip.style.display = 'none'; }
  let timer = null;
  function schedule() { const b = document.getElementById('busy'); if (b) b.textContent = 'computing...'; clearTimeout(timer); timer = setTimeout(() => { compute(); render(); }, 30); }
  function bind(anc) {
    const on = (id, ev, fn) => { const x = document.getElementById(id); if (x) x.addEventListener(ev, fn); };
    on('c-box', 'change', e => { st.box = e.target.value; save('box', st.box); st.sel = null; schedule(); });
    app.querySelectorAll('[data-heat]').forEach(x => x.addEventListener('change', e => { const mk = x.dataset.heat; st.heat[mk] = st.heat[mk] || {}; st.heat[mk][x.dataset.w] = e.target.value; save('heat', st.heat); schedule(); }));
    app.querySelectorAll('[data-theme-b]').forEach(x => x.addEventListener('click', () => { st.theme = x.dataset.themeB; save('theme', st.theme); applyTheme(); render(); }));
    app.querySelectorAll('[data-knob]').forEach(x => x.addEventListener('change', () => { st.knobs[x.dataset.knob] = x.checked; save('knobs', st.knobs); schedule(); }));
    app.querySelectorAll('[data-knob-s]').forEach(x => x.addEventListener('change', () => { st.knobs[x.dataset.knobS] = x.value; save('knobs', st.knobs); schedule(); }));
    app.querySelectorAll('[data-knob-n]').forEach(x => x.addEventListener('change', () => { const n = parseFloat(x.value); st.knobs[x.dataset.knobN] = isFinite(n) && n > 0 ? n : null; save('knobs', st.knobs); schedule(); }));
    on('c-reset', 'click', () => { st.knobs = JSON.parse(JSON.stringify(defaults)); save('knobs', st.knobs); schedule(); });
    on('c-table', 'click', () => { st.table = !st.table; render(); });
    app.querySelectorAll('[data-kappa]').forEach(x => x.addEventListener('change', () => { const n = parseFloat(x.value); if (isFinite(n) && n > 0) data.overrides.kappa[x.dataset.kappa] = n; else delete data.overrides.kappa[x.dataset.kappa]; save('overrides', data.overrides); schedule(); }));
    app.querySelectorAll('[data-kld]').forEach(x => x.addEventListener('change', () => { const n = parseFloat(x.value), fam = x.dataset.kld; data.overrides.kld[fam] = data.overrides.kld[fam] || {}; if (isFinite(n) && n >= 0) data.overrides.kld[fam][x.dataset.q] = n; else delete data.overrides.kld[fam][x.dataset.q]; save('overrides', data.overrides); schedule(); }));
    on('c-reset-ov', 'click', () => { data.overrides = { kappa: {}, kld: {} }; save('overrides', data.overrides); schedule(); });
    on('c-idea', 'input', e => save('idea', e.target.value));
    on('c-idea-dl', 'click', () => {
      const s = R.selected;
      download('fit-viz-idea-' + stamp() + '.json', { format: 'fit-viz idea v1', created: new Date().toISOString(), idea: document.getElementById('c-idea').value, box: st.box, knobs: st.knobs, heat: st.heat, overrides: data.overrides,
        selected: s ? { label: s.label, tok_s: +s.ev.tokps.toFixed(2), quality_pct: +s.ev.quality.toFixed(2), kld: +s.ev.kld.toFixed(5) } : null,
        best: R.curves.map(c => ({ model: c.mk, engine: c.eng, at20: c.b20 ? { tok_s: +c.b20.ev.tokps.toFixed(2), quality_pct: +c.b20.ev.quality.toFixed(2) } : null, at25: c.b25 ? { tok_s: +c.b25.ev.tokps.toFixed(2), quality_pct: +c.b25.ev.quality.toFixed(2) } : null })), data_built: data.meta.built });
    });
    on('c-export', 'click', () => { const s = R.selected; if (!s) return; download('expert-map-' + s.ctx.mk + '-' + s.ctx.bk + '-' + stamp() + '.json', FM.exportMap(s.ctx, s.lay, s.ev, { created: new Date().toISOString(), label: s.label, data_built: data.meta.built })); });
    // chart interactions
    app.querySelectorAll('[data-k]').forEach(n => {
      const k = n.dataset.k;
      n.addEventListener('mousemove', e => {
        if (k === 'pt') { const c = R.curves[+n.dataset.c]; showTip(e, `<b>${esc(data.models[c.mk].short + ' ' + engShort(c.eng))}</b><br>${f2(+n.dataset.x)} tok/s · top-1 ${f2(+n.dataset.q)} % · KLD ${(+n.dataset.kld).toFixed(4)}<br><span class="muted">click to load this layout</span>`); }
        else if (k === 't20' || k === 't25') { const c = R.curves[+n.dataset.c], b = c['b' + k.slice(1)]; showTip(e, `<b>Best at ${k.slice(1)} tok/s: ${esc(data.models[c.mk].short + ' ' + engShort(c.eng))}</b><br>${f2(b.ev.tokps)} tok/s · top-1 ${f2(b.ev.quality)} % · KLD ${b.ev.kld.toFixed(4)}`); }
        else if (k === 'anc') { const a = anc[+n.dataset.a]; showTip(e, `<b>${esc(a.label)}</b><br>measured ${f2(a.meas)} tok/s (MEASURED) · model ${f2(a.pred)} · residual ${(100 * a.resid).toFixed(1)} %<br><span class="muted">${esc(a.src)}</span>`); }
        else if (k === 'owner') { const o = R.owner.ev; showTip(e, `<b>Owner layout (droopy)</b><br>hottest coding experts MXFP4 in VRAM, other coding experts q3_K in DRAM, the rest MXFP4 in Optane<br>${f2(o.tokps)} tok/s · top-1 ${f2(o.quality)} %<br><span class="muted">needs per-expert types (split-expert build); click to load</span>`); }
        else if (k === 'seg') { const rows = tierRows(R.selected), g = rows[+n.dataset.tier].segs[+n.dataset.seg]; showTip(e, `<b>${esc(g.label)}</b><br>${f2(g.gib)} GiB${g.tag ? ' ' + tagChip(g.tag, g.src || '') : ''}${g.src ? '<br><span class="muted">' + esc(g.src) + '</span>' : ''}`); }
        else if (k === 'time') { const p = timeSplitHTML(R.selected).parts[+n.dataset.p]; showTip(e, `<b>${esc(p[0])}</b><br>${f1(p[1])} ms per token`); }
        else if (k === 'bh') { const t = R.selected.ctx.tiers[+n.dataset.t]; showTip(e, `<b>${esc(t.name)}</b><br>heat bin ${+n.dataset.b + 1}: ${n.dataset.n} experts at ${(+n.dataset.bpw).toFixed(2)} bpw on average`); }
        else if (k === 'thr') { showTip(e, n.dataset.u === '1' ? 'threads running the expert kernel' : 'idle threads: ' + esc(data.boxes[st.box].threads_note || '')); }
      });
      n.addEventListener('mouseleave', hideTip);
      if (k === 'pt' || k === 't20' || k === 't25' || k === 'owner') n.addEventListener('click', () => {
        hideTip();
        if (k === 'owner') st.sel = { kind: 'owner', mk: 'v41', eng: 'ik' };
        else { const c = R.curves[+n.dataset.c]; st.sel = k === 'pt' ? { mk: c.mk, eng: c.eng, kind: 'pt', tokps: +n.dataset.x } : { mk: c.mk, eng: c.eng, kind: k }; }
        loadSelection(); render();
      });
    });
  }
  const stamp = () => new Date().toISOString().replace(/[-:]/g, '').slice(0, 13);
  function download(name, obj) {
    const blob = new Blob([JSON.stringify(obj, null, 1)], { type: 'application/json' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  }

  let rz = null, lastW = 0;
  window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { if (Math.abs(app.clientWidth - lastW) > 40) { lastW = app.clientWidth; render(); } }, 200); });
  try { compute(); render(); lastW = app.clientWidth; } catch (e) { app.innerHTML = '<pre>' + esc(e.stack || e) + '</pre>'; throw e; }
  window.FIT_APP = { st, get R() { return R; }, compute, render };
})();
