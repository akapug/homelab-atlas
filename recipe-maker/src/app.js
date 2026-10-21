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
  // deep link: ?box=droopy&layout=v41-ik-t20 (kinds t20, t25, fast, pt<tok/s>; or owner) wins over the saved box; #section scrolls
  const selCode = s => !s ? '' : s.kind === 'owner' ? 'owner' : [s.mk, s.eng, s.kind === 'pt' ? 'pt' + f2(s.tokps) : s.kind].join('-');
  const parseSel = c => {
    if (c === 'owner') return { kind: 'owner', mk: 'v41', eng: 'ik' };
    const m = /^([a-z0-9]+)-([a-z0-9]+)-(t20|t25|fast|pt([0-9.]+))$/.exec(c || '');
    return m && data.models[m[1]] ? { mk: m[1], eng: m[2], kind: m[4] ? 'pt' : m[3], tokps: m[4] ? +m[4] : undefined } : null;
  };
  const qs = new URLSearchParams(location.search);
  if (data.boxes[qs.get('box')]) st.box = qs.get('box');
  st.sel = parseSel(qs.get('layout'));
  if (!data.boxes[st.box]) st.box = 'droopy';
  const syncURL = () => {
    const q = new URLSearchParams(location.search);
    q.set('box', st.box);
    if (st.sel) q.set('layout', selCode(st.sel)); else q.delete('layout');
    const u = '?' + q.toString().replace(/%2C/g, ',') + location.hash;
    if (u !== location.search + location.hash) { try { history.replaceState(null, '', u); } catch (e) { /* file:// in some browsers refuses a new query */ } }
  };
  const applyTheme = () => { if (st.theme === 'auto') document.documentElement.removeAttribute('data-theme'); else document.documentElement.setAttribute('data-theme', st.theme); };
  applyTheme();

  const MODELS = Object.keys(data.models).filter(m => data.models[m].show);
  const css = n => getComputedStyle(document.getElementById('root')).getPropertyValue(n).trim();
  // colour and marker shape follow the model (never its rank), so a curve is told apart by its label, its shape and its dash
  // as well as its hue: past three series no categorical palette separates every pair for colour-blind readers
  const modelColor = (mk, eng) => eng === 'glmlite' ? 'var(--s3)' : ({ v41: 'var(--s1)', fnx: 'var(--s4)', fnxc: 'var(--s5)', orn122: 'var(--s6)' })[mk] || 'var(--s2)';
  const modelShape = mk => ({ v41: 'circle', glm: 'square', fnx: 'tri', fnxc: 'trid', orn122: 'plus' })[mk] || 'circle';
  function shapePath(sh, x, y, r) {
    const P = pts => 'M' + pts.map(p => (x + p[0] * r).toFixed(1) + ' ' + (y + p[1] * r).toFixed(1)).join('L') + 'Z';
    if (sh === 'square') return P([[-.88, -.88], [.88, -.88], [.88, .88], [-.88, .88]]);
    if (sh === 'tri') return P([[0, -1.2], [1.1, .8], [-1.1, .8]]);
    if (sh === 'trid') return P([[0, 1.2], [1.1, -.8], [-1.1, -.8]]);
    if (sh === 'diamond') return P([[0, -1], [1, 0], [0, 1], [-1, 0]]);
    if (sh === 'plus') return P([[-.38, -1.05], [.38, -1.05], [.38, -.38], [1.05, -.38], [1.05, .38], [.38, .38], [.38, 1.05], [-.38, 1.05], [-.38, .38], [-1.05, .38], [-1.05, -.38], [-.38, -.38]]);
    if (sh === 'star') return P([0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map(i => { const a = -Math.PI / 2 + i * Math.PI / 5, k = i % 2 ? .5 : 1.25; return [k * Math.cos(a), k * Math.sin(a)]; }));
    return `M${(x - r).toFixed(1)} ${y.toFixed(1)}a${r} ${r} 0 1 0 ${2 * r} 0a${r} ${r} 0 1 0 ${-2 * r} 0Z`;
  }
  // a legend key: a short line in the curve's dash with its marker shape on it, so the key shows every channel the chart uses
  const keySVG = (col, sh, dash) => `<svg class="key" width="26" height="14" viewBox="0 0 26 14" aria-hidden="true">${dash !== undefined ? `<line x1="1" x2="25" y1="7" y2="7" stroke="${col}" stroke-width="2" ${dash ? 'stroke-dasharray="5 3"' : ''}/>` : ''}<path d="${shapePath(sh, 13, 7, 4.5)}" fill="${col}" stroke="var(--surface-1)" stroke-width="1"/></svg>`;
  // text width in px for the chart font (labels are placed by measured size, never guessed)
  let mctx = null;
  const textW = (t, px) => { mctx = mctx || document.createElement('canvas').getContext('2d'); mctx.font = (px || 11) + 'px ' + getComputedStyle(document.getElementById('root')).fontFamily; return mctx.measureText(t).width; };
  // ink for text on a fill: black or white, whichever has more contrast; null when neither reaches WCAG AA 4.5:1 (the label then gets a surface pill)
  function inkOn(col) {
    const m = /^var\((--[\w-]+)\)$/.exec(col || '');
    const hex = (m ? css(m[1]) : col || '').replace('#', '');
    if (!/^[0-9a-f]{6}$/i.test(hex)) return null;
    const lin = [0, 2, 4].map(i => { const c = parseInt(hex.slice(i, i + 2), 16) / 255; return c <= 0.03928 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); });
    const L = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2], kb = (L + 0.05) / 0.0546, kw = 1.05 / (L + 0.05);
    return Math.max(kb, kw) < 4.5 ? null : kb >= kw ? '#0b0b0b' : '#fff';
  }
  // Greedy collision-free direct labels. labels: [{text, x, y, r, pri, must}] anchored on a mark at (x, y) of radius r.
  // Tries 8 positions round the mark, then vertical nudges with a leader line; scores each spot by how many curve samples
  // (pts) it covers. A label that cannot fit without overlapping a placed label or a mark is dropped unless 'must'; its text
  // stays on hover and in the table. Nothing is placed outside bx, so no text is clipped at the chart edge.
  const LABEL_GAP = 2;   // px between two chart labels
  function placeLabels(labels, marks, pts, bx) {
    const placed = [];
    const hit = (a, b) => a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
    const inside = r => r.x >= bx.x0 && r.x + r.w <= bx.x1 && r.y >= bx.y0 && r.y + r.h <= bx.y1;
    const cover = r => pts.reduce((n, p) => n + (p[0] > r.x && p[0] < r.x + r.w && p[1] > r.y && p[1] < r.y + r.h ? 1 : 0), 0);
    labels.slice().sort((a, b) => a.pri - b.pri).forEach(lb => {
      const w = textW(lb.text) + 4, h = 14, g = (lb.r || 3) + 3;
      const near = [[g, -h / 2], [g, -h], [g, 0], [-g - w, -h / 2], [-g - w, -h], [-g - w, 0], [-w / 2, -g - h], [-w / 2, g]];
      const cands = near.map(([dx, dy], i) => ({ x: lb.x + dx, y: lb.y + dy, w, h, cost: i * 0.2 }));
      for (let k = 1; k <= 7; k++) [-1, 1].forEach(sg => [g + 10, -g - 10 - w].forEach(dx => cands.push({ x: lb.x + dx, y: lb.y - h / 2 + sg * k * (h + 1), w, h, cost: 2 + k, lead: true })));
      const pad = r => ({ x: r.x - LABEL_GAP, y: r.y - LABEL_GAP, w: r.w + 2 * LABEL_GAP, h: r.h + 2 * LABEL_GAP });
      const free = r => inside(r) && !placed.some(p => hit(pad(r), p));
      const clear = cands.filter(r => free(r) && !marks.some(m => m.own !== lb.own && hit(r, m)));
      // a label sitting closer to another series' mark than to its own reads as that series' label: penalise it
      const dist = (r, x, y) => Math.hypot(Math.max(r.x - x, 0, x - r.x - r.w), Math.max(r.y - y, 0, y - r.y - r.h));
      const amb = r => { const d = dist(r, lb.x, lb.y); return marks.filter(m => m.ser !== lb.ser && dist(r, m.x + m.w / 2, m.y + m.h / 2) < d).length ? 4 : 0; };
      let best = clear.map(r => Object.assign(r, { score: r.cost + 3 * cover(r) + amb(r) })).sort((a, b) => a.score - b.score)[0];
      if (!best && lb.must) best = cands.filter(free)[0] || Object.assign({}, cands[0], { x: Math.max(bx.x0, Math.min(bx.x1 - w, cands[0].x)), y: Math.max(bx.y0, Math.min(bx.y1 - h, cands[0].y)) });
      if (!best) { lb.dropped = true; return; }
      placed.push(best);
      Object.assign(lb, { lx: best.x + 2, ly: best.y + h - 4, box: best, lead: best.lead || Math.abs(best.y + h / 2 - lb.y) > h });
    });
    return labels;
  }
  const labelSVG = (lb, cls) => lb.dropped ? '' : (lb.lead ? `<line x1="${lb.x.toFixed(1)}" y1="${lb.y.toFixed(1)}" x2="${(lb.box.x > lb.x ? lb.box.x : lb.box.x + lb.box.w).toFixed(1)}" y2="${(lb.box.y + lb.box.h / 2).toFixed(1)}" stroke="var(--text-secondary)" stroke-width="1"/>` : '') +
    `<text x="${lb.lx.toFixed(1)}" y="${lb.ly.toFixed(1)}" class="lbl ${cls || 'ink'}">${esc(lb.text)}</text>`;
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
  const DEFAULT_MIN_RUNS = 2;   // measured runs on a box before its model is the page's first layout there
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
    // keep the selection if it still exists, else the box's showcase model (data.boxes[box].showcase: the model it runs best,
    // with its source), else a model we measured on this box (DEFAULT_MIN_RUNS runs or more), then one that reaches 20 tok/s
    let pick = null;
    if (st.sel) {
      const c = curves.find(c => c.mk === st.sel.mk && c.eng === st.sel.eng);
      if (st.sel.kind === 'owner') pick = owner ? { kind: 'owner', mk: 'v41', eng: 'ik' } : null;
      else if (c) pick = st.sel;
    }
    if (!pick) {
      const runs = {};
      data.cal.rows.filter(r => r.box === st.box).forEach(r => { runs[r.model] = (runs[r.model] || 0) + 1; });
      const show = (data.boxes[st.box].showcase || {}).model;
      const rank = c => (c.mk === show ? 8 : 0) + ((runs[c.mk] || 0) >= DEFAULT_MIN_RUNS ? 4 : 0) + (c.b20 ? 2 : 0) + (c.mk === 'v41' ? 1 : 0);
      const order = curves.slice().sort((a, b) => rank(b) - rank(a));
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
    if (s.kind === 'owner') { R.selected = Object.assign({ label: 'Our per-expert plan (droopy): hottest coding experts MXFP4 in VRAM, other coding experts q3_K in DRAM, the rest MXFP4 in Optane', needsCode: true }, R.owner); return; }
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
  // measured runs on this box, read from the ledger (data.recipes rows that carry a measured decode): each one sits at its
  // measured speed and at the projected quality of its stored types, with a line to what the model predicts for that run's
  // own settings. A draft run shows its faster arm (MTP or draft on, or off when the draft lost); a run that went bad is
  // listed in the table only.
  function measuredHere() {
    return (data.recipes || []).filter(r => r.box === st.box && r.measured && MODELS.indexOf(r.model) >= 0).map(r => {
      const m = r.measured, rec = r.recipe, q = FM.recipeQuality(data, rec);
      if (!q) return null;
      const arm = m.on > 0 && !(m.off > m.on) ? 'on' : m.off > 0 ? 'off' : 'other';
      const meas = m[arm], phase = m[arm + '_phase'];
      const p = FM.predictRecipe(data, rec);
      const draftArm = arm === 'on' || (arm === 'other' && rec.draft);
      const pv = p.status !== 'PREDICTED' ? null : draftArm ? p.decode_tps : p.decode_tps_nodraft;
      const why = p.status !== 'PREDICTED' ? p.why : !(pv > 0) ? p.draft_why : null;
      return { id: r.run_id, when: r.when, label: rec.label || r.label, model: r.model, engine: r.engine, meas, phase, arm, draftArm,
        pred: pv > 0 ? pv : null, why, gap: pv > 0 ? meas / pv - 1 : null, waiver: rec.gate_waiver || null,
        quality: q.quality, qhow: q.how, top1: m.top1, bad: m.bad, guess: !!m.guess, holdout: m.holdout };
    }).filter(Boolean);
  }
  const armText = a => a.arm === 'on' ? 'MTP / draft on' : a.arm === 'off' ? 'no draft' : a.draftArm ? 'draft on' : 'no draft';
  function frontierSVG() {
    const AW = app.clientWidth || 720;
    const W = Math.round(Math.max(280, AW >= 980 ? (AW - 10) * 1.25 / 2.25 - 26 : AW - 26)), H = Math.round(Math.min(440, Math.max(340, W * 0.62))), L = 46, Rm = 12, T = 10, B = 40;
    const allP = [];
    R.curves.forEach(c => envelope(c.pts).forEach(p => allP.push(p)));
    const anc = measuredHere().filter(a => !a.bad && !a.guess);
    const xmax = Math.max(40, ...allP.map(p => p.tokps * 1.05), ...anc.map(a => Math.max(a.meas, a.pred || 0) * 1.05), R.owner ? R.owner.ev.tokps * 1.05 : 0);
    const qs = allP.map(p => p.quality).concat(anc.map(a => a.quality));
    const ymin = Math.max(0, Math.floor((Math.min(75, ...qs) - 2) / 5) * 5);
    const X = x => L + (W - L - Rm) * Math.min(x, xmax) / xmax, Y = y => T + (H - T - B) * (100 - y) / (100 - ymin);
    const xstep = xmax > 80 ? 10 : 5;
    let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="width:${W}px;max-width:100%" aria-hidden="true" focusable="false">`;
    // x tick labels: the 20 and 25 targets first, then the round ticks that do not collide with a label already set
    const ticks = [], xt = [20, 25];
    for (let x = 0; x <= xmax; x += xstep) { s += `<line x1="${X(x)}" x2="${X(x)}" y1="${T}" y2="${H - B}" stroke="var(--grid)"/>`; if (xt.indexOf(x) < 0) xt.push(x); }
    xt.forEach(x => { const w = textW(String(x)) + 6, l = X(x) - w / 2; if (l >= 0 && l + w <= W && !ticks.some(t => l < t.l + t.w && t.l < l + w)) ticks.push({ x, l, w }); });
    ticks.forEach(t => { const ref = t.x === 20 || t.x === 25; s += `<text x="${X(t.x)}" y="${H - B + 14}" text-anchor="middle"${ref ? ' class="ink" font-weight="600"' : ''}>${t.x}</text>`; });
    for (let y = ymin; y <= 100; y += 5) s += `<line x1="${L}" x2="${W - Rm}" y1="${Y(y)}" y2="${Y(y)}" stroke="var(--grid)"/><text x="${L - 6}" y="${Y(y) + 4}" text-anchor="end">${y}</text>`;
    s += `<text x="${(L + W - Rm) / 2}" y="${H - 6}" text-anchor="middle">projected decode speed, tok/s (one user)</text>`;
    s += `<text transform="translate(13 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${W < 420 ? 'top-1 agreement, %' : 'top-1 agreement with the full model, %'}</text>`;
    [20, 25].forEach(x => { s += `<line x1="${X(x)}" x2="${X(x)}" y1="${T}" y2="${H - B}" stroke="var(--text-secondary)" stroke-dasharray="4 3"/>`; });
    // marks first (as obstacles for the labels), labels last so they sit on top with a surface halo
    const marks = [], samp = [], labels = [], nav = [];
    const mark = (x, y, r, own, ser) => marks.push({ x: x - r, y: y - r, w: 2 * r, h: 2 * r, own, ser: ser === undefined ? own : ser });
    let body = '';
    R.curves.forEach((c, ci) => {
      const env = envelope(c.pts), sh = modelShape(c.mk), name = data.models[c.mk].short + ' ' + engShort(c.eng);
      nav.push({ ci, name, pts: env.map(p => ({ x: X(p.tokps), y: Y(p.quality), tokps: p.tokps, quality: p.quality, kld: p.kld })) });
      if (env.length) {
        const xy = env.map(p => [X(p.tokps), Y(p.quality)]);
        body += `<polyline fill="none" stroke="${c.color}" stroke-width="2" ${c.dash ? 'stroke-dasharray="6 4"' : ''} points="${xy.map(p => p.join(',')).join(' ')}"/>`;
        xy.forEach((p, i) => { if (i) { const q = xy[i - 1], n = Math.ceil(Math.hypot(p[0] - q[0], p[1] - q[1]) / 4); for (let k = 0; k < n; k++) samp.push([q[0] + (p[0] - q[0]) * k / n, q[1] + (p[1] - q[1]) * k / n]); } samp.push(p); });
        env.forEach((p, pi) => { body += `<circle cx="${X(p.tokps)}" cy="${Y(p.quality)}" r="9" fill="transparent" data-k="pt" data-c="${ci}" data-x="${p.tokps}" data-q="${p.quality}" data-kld="${p.kld}" style="cursor:pointer"/><circle cx="${X(p.tokps)}" cy="${Y(p.quality)}" r="2.5" fill="${c.color}" pointer-events="none"/>`; mark(X(p.tokps), Y(p.quality), 2.5, 'c' + ci, ci); });
        const last = env[env.length - 1];
        labels.push({ text: name, x: X(last.tokps), y: Y(last.quality), r: 4, pri: 1, must: true, own: 'c' + ci, ser: ci });
      }
      [['b20', 20], ['b25', 25]].forEach(([k, t]) => {
        const b = c[k];
        if (!b) return;
        const x = X(b.ev.tokps), y = Y(b.ev.quality);
        body += `<path d="${shapePath(sh, x, y, 6.5)}" fill="${c.color}" stroke="var(--surface-1)" stroke-width="2" data-k="t${t}" data-c="${ci}" style="cursor:pointer"/>`;
        mark(x, y, 7, 'b' + ci + t, ci);
        if (t === 25 && c.b20 && Math.abs(c.b20.ev.tokps - b.ev.tokps) < 1e-6 && Math.abs(c.b20.ev.quality - b.ev.quality) < 1e-6) return;
        labels.push({ text: f1(b.ev.quality) + ' %', x, y, r: 7, pri: 3, own: 'b' + ci + t, ser: ci });
      });
    });
    // the fastest clean run per model carries its gap to the model as a label
    const top = {};
    anc.forEach(a => { if (!top[a.model] || a.meas > top[a.model].meas) top[a.model] = a; });
    anc.forEach((a, ai) => {
      const col = modelColor(a.model, a.engine), x = X(a.meas), y = Y(a.quality);
      if (a.pred) {
        body += `<line x1="${x}" x2="${X(a.pred)}" y1="${y}" y2="${y}" stroke="${col}" stroke-width="1.5"${a.waiver ? ' stroke-dasharray="3 2"' : ''}/>`;
        body += `<circle cx="${X(a.pred)}" cy="${y}" r="2.5" fill="${col}" pointer-events="none"/>`;
      }
      body += `<path d="${shapePath('diamond', x, y, 6)}" fill="var(--surface-1)" stroke="${col}" stroke-width="2" data-k="anc" data-a="${ai}"/>`;
      mark(x, y, 6, 'a' + ai);
      const pc = a.pred ? Math.round(100 * a.gap) : 0;
      if (top[a.model] === a) labels.push({ text: 'run ' + f1(a.meas) + (a.pred ? ', ' + (pc > 0 ? '+' : pc < 0 ? '−' : '±') + Math.abs(pc) + ' % vs model' : ', not modelled'), x, y, r: 6, pri: 2, own: 'a' + ai, ser: 'a' + a.model });
    });
    if (R.owner) {
      const x = X(R.owner.ev.tokps), y = Y(R.owner.ev.quality);
      body += `<path d="${shapePath('star', x, y, 7)}" fill="var(--text-primary)" stroke="var(--surface-1)" stroke-width="1.5" data-k="owner" style="cursor:pointer"/>`;
      mark(x, y, 7, 'own');
      labels.push({ text: 'our per-expert plan', x, y, r: 7, pri: 2, must: true, own: 'own', ser: 'own' });
      nav.push({ owner: true, name: 'our per-expert plan', pts: [{ x, y, tokps: R.owner.ev.tokps, quality: R.owner.ev.quality, kld: R.owner.ev.kld }] });
    }
    // the loaded layout gets a ring, so the chart says which point the panels below describe
    const sel = R.selected;
    if (sel && sel.ev) body += `<circle cx="${X(sel.ev.tokps)}" cy="${Y(sel.ev.quality)}" r="11" fill="none" stroke="var(--text-primary)" stroke-width="1.5" pointer-events="none"/>`;
    [20, 25].forEach(x => { for (let y = T; y <= H - B; y += 4) samp.push([X(x), y]); });   // a label over a reference line reads as crossed out
    placeLabels(labels, marks, samp, { x0: L + 2, x1: W - 2, y0: T, y1: H - B - 2 });
    s += body + labels.map(lb => labelSVG(lb, lb.pri === 3 ? 'val' : 'ink')).join('');
    s += `<circle id="fcur" r="12" fill="none" stroke="var(--text-primary)" stroke-width="2.5" stroke-dasharray="3 2" visibility="hidden" pointer-events="none"/>`;
    s += '</svg>';
    R.nav = nav;
    return { svg: s, anc, labels };
  }
  // one sentence per curve, read by screen readers with the chart (and a quick human summary)
  // what the chart cannot show: runs this copy leaves out (data.meta.left_out, written by the public export) and measured
  // runs the planner does not model (no line)
  function chartNotes() {
    const lo = data.meta.left_out || {}, out = [];
    Object.keys(lo).forEach(mk => { const n = lo[mk].runs[st.box]; if (n) out.push(`${n} ${esc(data.models[mk] ? data.models[mk].short : mk)} run${n > 1 ? 's' : ''} on this box ${n > 1 ? 'are' : 'is'} left out of this copy: ${esc(lo[mk].why)}.`); });
    const un = measuredHere().filter(a => !a.bad && !a.guess && !a.pred).length;
    if (un) out.push(`${un} measured run${un > 1 ? 's' : ''} here ${un > 1 ? 'are' : 'is'} not modelled (no line; hover for why).`);
    return out.length ? ' ' + out.join(' ') : '';
  }
  function frontierSummary() {
    const part = c => {
      const nm = data.models[c.mk].short + ' ' + engShort(c.eng);
      const env = envelope(c.pts);
      if (!env.length) return nm + ': no feasible layout';
      const fast = env[env.length - 1];
      return nm + ': ' + (c.b20 ? 'best at 20 tok/s ' + f1(c.b20.ev.quality) + ' %' : 'does not reach 20 tok/s') + (c.b25 ? ', at 25 tok/s ' + f1(c.b25.ev.quality) + ' %' : '') + ', fastest ' + f1(fast.tokps) + ' tok/s at ' + f1(fast.quality) + ' %';
    };
    return R.curves.map(part).join('. ') + '.';
  }

  // horizontal stacked bar as HTML (text keeps its size on any screen)
  // segs: [{w: fraction, col, text, short, tip, attrs, hatch: 'r'|'n'|undefined}]. A segment shows its text, else its short
  // text, else nothing (fitBarLabels decides after layout); the full text is always its aria-label and its hover/focus tip.
  // The bar is one tab stop: arrow keys move between segments.
  function hbar(segs, h, name) {
    const vis = segs.filter(g => g.w > 0);
    return `<div class="hbar" role="list"${name ? ` aria-label="${esc(name)}"` : ''} style="height:${h || 24}px">` + vis.map((g, gi) => {
      const bg = g.hatch === 'r' ? 'repeating-linear-gradient(45deg,var(--crit) 0 2px,var(--surface-2) 2px 6px)' : g.hatch === 'n' ? 'repeating-linear-gradient(135deg,var(--neutral) 0 2px,var(--surface-2) 2px 6px)' : g.col;
      const ink = g.hatch ? null : inkOn(g.col);
      const label = g.tip || g.text || '';
      return `<span role="listitem" tabindex="${gi ? -1 : 0}" aria-label="${esc(label)}" style="flex:0 0 ${(100 * g.w).toFixed(3)}%;background:${bg}" ${g.attrs || ''}>${g.text ? `<i class="${ink ? '' : 'pill'}" style="${ink ? 'color:' + ink : ''}" data-full="${esc(g.text)}" data-short="${esc(g.short || '')}" aria-hidden="true">${esc(g.text)}</i>` : ''}</span>`;
    }).join('') + '</div>';
  }
  // after layout: each bar label shows its full text if it fits, else its short form, else nothing (hover or focus shows it)
  function fitBarLabels() {
    app.querySelectorAll('.hbar i[data-full]').forEach(i => {
      const room = i.parentElement.clientWidth - 8;
      i.textContent = i.dataset.full;
      if (i.offsetWidth > room) i.textContent = i.dataset.short;
      if (i.offsetWidth > room) i.textContent = '';
    });
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
      const s = hbar(r.segs.map((g, gi) => ({ w: g.gib / r.total, col: g.col, hatch: g.idle ? 'r' : g.hatch ? 'n' : undefined,
        text: g.idle ? 'idle ' + f1(g.gib) + ' GiB' : g.q ? g.q + ' ' + f1(g.gib) + ' GiB' : g.label.split(' (')[0] + ' ' + f1(g.gib),
        short: g.idle ? 'idle' : g.q ? g.q : f1(g.gib), tip: g.label + ': ' + f1(g.gib) + ' GiB', attrs: `data-k="seg" data-tier="${i}" data-seg="${gi}"` })), 24, t.name + ', ' + f1(r.total) + ' GiB');
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
        t.nodes.forEach(nd => { h += `<div class="row small"><span style="min-width:130px">${esc(nd.name)} ${f1(nd.gib)} GiB</span><div style="flex:1;max-width:420px">${hbar([{ w: usedFrac, col: 'var(--r450)', tip: 'used ' + f1(usedFrac * 100) + ' %' }, { w: 1 - usedFrac, hatch: 'r', tip: 'free ' + f1((1 - usedFrac) * 100) + ' %' }], 10, nd.name)}</div></div>`; });
      }
      h += '</div>';
    });
    // threads
    const thr = v(box.cpu.threads), used = st.knobs.smt ? thr : v(box.cpu.used);
    h += `<div class="tierbox"><div class="tierhead"><b>CPU threads</b><span class="small">${esc(box.cpu.desc)} · ${thr} threads${chip(box.cpu.threads)}</span></div>`;
    h += hbar([{ w: used / thr, col: 'var(--r450)', text: used + ' running the expert kernel', short: String(used), tip: used + ' threads running the expert kernel', attrs: 'data-k="thr" data-u="1"' }, { w: (thr - used) / thr, hatch: 'r', text: thr > used ? (thr - used) + ' idle' : '', short: String(thr - used), tip: (thr - used) + ' idle threads', attrs: 'data-k="thr" data-u="0"' }], 24, 'CPU threads');
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
    const s = hbar(parts.map((p, pi) => ({ w: p[1] / tot, col: p[2], text: f1(p[1]) + ' ms', short: f1(p[1]), tip: p[0] + ': ' + f1(p[1]) + ' ms', attrs: `data-k="time" data-p="${pi}"` })), 24, 'time per token');
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
    const W = Math.round(Math.max(300, (app.clientWidth || 720) - 40)), L = 40, T = 10, Rm = 10;
    const ends = ['hottest experts', 'coldest (rarely or never routed)'], mid = 'heat rank (plan heat), ' + nb + ' bins';
    const oneRow = textW(ends[0]) + textW(ends[1]) + textW(mid) + 40 < W - L - Rm, B = oneRow ? 34 : 48, H = 200 + B - 34;
    const bmax = Math.ceil(Math.max(4.5, ...Object.values(series).flat().map(p => p.bpw)));
    const X = b => L + (W - L - Rm) * (b + 0.5) / nb, Y = y => T + (H - T - B) * (1 - (y - 1.5) / (bmax - 1.5));
    const tierCol = t => { const k = ctx.tiers[t]; return k.kind === 'gpu' ? 'var(--s1)' : k.id === 'dram' ? 'var(--s2)' : k.id === 'pmem' ? 'var(--s3)' : 'var(--neutral)'; };
    const tierShape = t => { const k = ctx.tiers[t]; return k.kind === 'gpu' ? 'circle' : k.id === 'dram' ? 'square' : k.id === 'pmem' ? 'tri' : 'diamond'; };
    let s = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="width:${W}px;max-width:100%" role="img" aria-label="Chart: mean bits per weight in each heat bin, one mark per memory tier" aria-describedby="bits-call">`;
    for (let y = 2; y <= bmax; y += 1) s += `<line x1="${L}" x2="${W - Rm}" y1="${Y(y)}" y2="${Y(y)}" stroke="var(--grid)"/><text x="${L - 5}" y="${Y(y) + 4}" text-anchor="end">${y}</text>`;
    s += `<text transform="translate(13 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">bits per weight</text>`;
    s += `<text x="${L}" y="${H - B + 22}">${ends[0]}</text><text x="${W - Rm}" y="${H - B + 22}" text-anchor="end">${ends[1]}</text><text x="${(L + W) / 2}" y="${oneRow ? H - B + 22 : H - 6}" text-anchor="middle">${mid}</text>`;
    Object.keys(series).forEach(t => series[t].forEach(p => {
      const r = 2 + 3.5 * Math.sqrt(p.n / per);
      s += `<path d="${shapePath(tierShape(t), X(p.b) + (+t - ctx.tiers.length / 2) * 3, Y(p.bpw), r)}" fill="${tierCol(t)}" stroke="var(--surface-1)" stroke-width="1.5" data-k="bh" data-t="${t}" data-b="${p.b}" data-bpw="${p.bpw}" data-n="${p.n}"/>`;
    }));
    s += '</svg>';
    const leg = '<div class="legend">' + Object.keys(series).map(t => `<span>${keySVG(tierCol(t), tierShape(t))}${esc(ctx.tiers[t].name)}</span>`).join('') + '<span class="muted">mark size = share of the bin</span></div>';
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
  // the page in tabs: [id = its first section, tab name, the one-line summary first, every section on it]; a link to any section opens its tab
  const PAGES = [
    ['box', 'Plan', 'Pick a box, then a large marker on the chart. The layout card sums that layout up.', ['box', 'whatif', 'chart', 'layout']],
    ['bytes', 'Where it lives', 'The loaded layout, byte by byte: one bar per memory tier, then how many bits each expert gets by how often it is used.', ['bytes', 'bits']],
    ['trust', 'Trust', 'How far each number goes: its evidence class, the hard checks each recorded run met or failed, and how close the model came to every measured run. Each card opens one level deeper.', ['trust', 'checks', 'calibration']],
    ['types', 'Edit', 'Change the quant type numbers the model uses, and write down an idea to download with the current view.', ['types', 'idea']],
    ['sources', 'Sources', 'The community recipe rules and priors behind the defaults, and how the model works.', ['sources']]];
  const pageOf = id => (PAGES.find(p => p[0] === id || p[3].indexOf(id) >= 0) || PAGES[0])[0];
  st.page = pageOf(location.hash.slice(1));
  // a card as a landmark section with its h2 as the name and an id the nav and deep links use
  const sec = (id, html) => html.replace(/^<div class="card"><h2>/, `<section class="card" id="${id}" aria-labelledby="${id}-h"><h2 id="${id}-h">`).replace(/<\/div>$/, '</section>');
  // the header folds into one details below this width, so a phone reaches the tabs without scrolling past the intro
  const NARROW_PX = 600;
  function introHTML(head, more) {
    return `<details class="intro" id="intro" ${load('intro', false) ? 'open' : ''}><summary>${head}</summary>${more || ''}<ol>
<li>Choose a box. The chart shows every model that fits on it: how closely each layout matches the full model (up) against how fast it decodes (right). Up and to the right is better.</li>
<li>Each curve keeps only the best layouts. The large markers are the best layout at 20 and at 25 tokens per second; hollow diamonds are runs we measured, with a line to what the model predicts for each run's own settings.</li>
<li>Click a point to load its layout, or tab to the chart and use the arrow keys and Enter. <a href="#layout">Selected layout</a> sums it up; the <a href="#bytes">Where it lives</a> tab shows where each part of the model sits: one bar per memory tier, from the GPU down to disk.</li>
<li>The small tags say how far to trust a number: <b>seen</b> (we measured it), <b>inferred</b> (fitted, or from a spec or a community post) or <b>unknown</b>. Hover a tag for its source.</li>
<li>The address bar keeps the box and the layout, so a copied link opens the same view.</li></ol></details>`;
  }
  function render() {
    const sel = R.selected;
    const box = data.boxes[st.box];
    const kn = st.knobs;
    let h = '';
    const sub = `<p class="sub">Plan where each part of a big mixture-of-experts model lives on a box, and see the smartest layout it can run at 20+ or 25+ tok/s. Built ${esc(data.meta.built)} from measured runs and community priors.</p>`;
    const lead = `<p class="lead">The boxes here are our own four machines (boxes A to D in the Atlas). <b>To start:</b> pick a box below, then click a large marker on the chart; the layout card sums that layout up, and the tabs go one level deeper each: where every byte lives, how far to trust each number.${data.meta.home ? ` <b>For your own box:</b> clone the folder and run the probe, as in <a href="${esc(data.meta.home)}#with-your-own-boxes">With your own boxes</a>.` : ''}</p>`;
    const EV = data.meta.evidence;
    const leg = `<div class="legend" style="margin-top:6px">${Object.keys(EV.classes).map(k => `<span>${gradeChip(k, EV.classes[k])} ${esc(EV.classes[k])} (${EV.counts[k] || 0} values)</span>`).join('')}</div>`;
    h += '<header class="top"><h1>Recipe Maker</h1>' + ((app.clientWidth || 720) < NARROW_PX ? introHTML('What this is and how to read it', sub + lead + leg) : sub + lead + leg + introHTML('How to read this page')) + '</header>';
    h += `<nav class="toc" aria-label="Pages"><ul>${PAGES.map(([id, t]) => `<li><a href="#${id}" ${id === st.page ? 'aria-current="page"' : ''}>${t}</a></li>`).join('')}</ul></nav><main id="main">`;
    const page = id => `<div class="page" id="pg-${id}" ${id === st.page ? '' : 'hidden'}><p class="pagesum">${esc(PAGES.find(p => p[0] === id)[2])}</p>`;
    // controls
    h += page('box') + `<section class="card" id="box" aria-labelledby="box-h"><h2 id="box-h" class="sr-only">Box and routing data</h2><div class="row">`;
    h += `<label class="boxpick">Box <select id="c-box">${Object.keys(data.boxes).map(b => `<option value="${b}" ${b === st.box ? 'selected' : ''}>${esc(data.boxes[b].name)}${data.boxes[b].atlas ? ' = box ' + esc(data.boxes[b].atlas) : ''}</option>`).join('')}</select></label>`;
    h += `<span class="small" role="group" aria-label="Theme">Theme ${['auto', 'light', 'dark'].map(t => `<button id="t-${t}" data-theme-b="${t}" aria-pressed="${st.theme === t}" class="${st.theme === t ? 'on' : ''}">${t}</button>`).join(' ')}</span>`;
    h += `</div><p class="small muted" style="margin:6px 0 0">${esc(box.role)}.</p>`;
    h += `<details id="heatd" ${load('heatd', false) ? 'open' : ''}><summary>Routing data behind each model</summary><div class="small muted">Plan heat = the routing counts a layout is built from; scored on = the counts used to project its speed and quality (a held-out half avoids grading the plan on its own data).</div><div class="heatgrid">`;
    MODELS.forEach(mk => {
      const m = data.models[mk], hs = heatFor(mk);
      const opts = n => Object.keys(m.heat.sets).map(k => `<option value="${k}" ${k === n ? 'selected' : ''}>${esc(m.heat.sets[k].label)}</option>`).join('');
      h += `<b>${esc(m.short)}</b><label class="small">plan heat <select id="h-${mk}-plan" data-heat="${mk}" data-w="plan">${opts(hs.plan)}</select></label><label class="small">scored on <select id="h-${mk}-score" data-heat="${mk}" data-w="score">${opts(hs.score)}</select></label>`;
    });
    h += '</div></details></section>';
    // what-if
    const upD = box.upgrades && box.upgrades.dram, upP = box.upgrades && box.upgrades.pmem;
    const changed = Object.keys(defaults).filter(k => JSON.stringify(kn[k]) !== JSON.stringify(defaults[k])).length;
    h += `<section class="card" id="whatif" aria-labelledby="whatif-h"><details id="whatifd" ${load('whatifd', false) || changed ? 'open' : ''}><summary><h2 id="whatif-h">What if</h2> <span class="small muted">${changed ? changed + ' setting' + (changed > 1 ? 's' : '') + ' changed from the defaults' : 'add DRAM or Optane, move a GPU, change engine settings; the chart follows'}</span></summary><div class="row">`;
    const cb = (k, label, title, dis) => `<label title="${esc(title || '')}"><input type="checkbox" id="k-${k}" data-knob="${k}" ${kn[k] ? 'checked' : ''} ${dis ? 'disabled' : ''}> ${label}</label>`;
    h += cb('perExpertTypes', 'per-expert types', 'Today a GGUF stores one type per layer tensor. Per-expert types need the split-expert ik build (lab notes).');
    h += cb('freeDesktop', 'free the snoopy desktop VRAM', 'the compositor leaves the 3090 Ti (display off, or the desktop moved to the iGPU)', st.box !== 'snoopy');
    h += cb('freeHousehold', 'free odie B70 #2 (resident service off)', 'Both B70s for research: the resident model on B70 #2 stops for the window (book the window first)', st.box !== 'odie');
    h += `<label title="dram-fill placement: the same free headroom on every NUMA node. 10 GiB with 3.0 GiB banks = MEASURED-INSUFFICIENT on droopy V4.1 (B1b: node0 MemFree 0.5 GiB, prefill 10-12); 16 = MEASURED-INSUFFICIENT on node0 and slower (B1c); 10 = current setting with swap off and B0's 2.2 GiB banks (B0 nortr ran clean). Empty = the box's usable DRAM figure">node headroom GiB <input type="number" id="k-numaHeadroom" step="1" min="1" max="64" data-knob-n="numaHeadroom" value="${kn.numaHeadroom === null || kn.numaHeadroom === undefined ? '' : kn.numaHeadroom}" placeholder="usable" ${(box.dram.nodes || []).some(x => x.avail_gib) ? '' : 'disabled'}></label>`;
    h += `<label>move odie's B70 #1 into <select id="k-moveB70" data-knob-s="moveB70">${['none', 'snoopy', 'droopy'].map(x => `<option ${kn.moveB70 === x ? 'selected' : ''}>${x}</option>`).join('')}</select></label>`;
    h += cb('addDRAM', 'add DRAM' + (upD ? ' (' + esc(upD.what) + ')' : ''), upD ? upD.gib.src : 'no DRAM upgrade listed for this box', !upD);
    h += cb('addOptane', 'add Optane' + (upP ? ' (' + esc(upP.what) + ')' : ''), upP ? upP.gib.src : 'only droopy takes Optane', !upP);
    h += cb('kernel2x', '2x faster CPU expert kernel', 'code: a kernel that reaches the memory ceiling. K1 (droopy, 10-17) measured the per-type kernel shares now in the type table; on snoopy the AVX-512 GLM lane already sits at the DRAM roofline (0.167 ms/expert K=3, 59.5 GB/s), so 2x there is not available from kernel work alone');
    h += cb('smt', 'SMT threads on', 'measured on snoopy only (x0.88 on V4.0)');
    h += cb('draft', 'draft / MTP on', '');
    h += `<label title="empty = the fitted or default multiplier for this box">draft speed-up <input type="number" id="k-draftMult" step="0.05" min="1" max="3" data-knob-n="draftMult" value="${kn.draftMult === null || kn.draftMult === undefined ? '' : kn.draftMult}" placeholder="auto"></label>`;
    h += cb('hostCopy', 'keep host copies of GPU experts', 'ik prefill reads a host copy of every GPU-resident expert');
    h += `<label>non-expert weights at <select id="k-densePrec" data-knob-s="densePrec">${['q8_0', 'q6_K', 'q4_K'].map(x => `<option ${kn.densePrec === x ? 'selected' : ''}>${x}</option>`).join('')}</select></label>`;
    h += cb('rtr', 'CPU repack at load (-rtr)', 'Allows the load-time repacked MXFP4 (mxfp4_r8). Measured on droopy B0: no decode gain, prefill 3x slower on a split model. ' + data.types.mxfp4_r8.note);
    h += `<label title="-ub 4096 grows the compute buffer (V4.1 B0: 5.4 GiB KV + compute); auto drops to 512 when the GPU cannot hold it">batch -ub <select id="k-ub" data-knob-s="ub">${['auto', '4096', '512'].map(x => `<option ${String(kn.ub) === x ? 'selected' : ''}>${x}</option>`).join('')}</select></label>`;
    h += `<label title="VRAM transient margin above the post-warm-up reading. safe = the measured upper bound (droopy ik split: ${data.checks.vram_peak.margin.ik_split.hi} GiB; B1b ran prompts at 21.56 of 23.04 in nvidia-smi MiB / 1000 without OOM); likely = the B0 #1 graph-growth candidate">VRAM peak margin <select id="k-vramMargin" data-knob-s="vramMargin">${['safe', 'likely'].map(x => `<option ${kn.vramMargin === x ? 'selected' : ''}>${x}</option>`).join('')}</select></label>`;
    h += `<button id="c-reset">reset knobs</button><span class="busy" id="busy" role="status">computed in ${Math.round(R.ms)} ms</span></div>`;
    if (kn.moveB70 !== 'none') h += `<div class="warnline">${esc(data.b70move.gpu.note)}</div>`;
    h += '</details></section>';
    // frontier + selection summary
    const fr = frontierSVG();
    h += '<div class="grid2"><section class="card" id="chart" aria-labelledby="chart-h"><h2 id="chart-h">Quality vs speed on ' + esc(box.name) + '</h2>';
    h += '<div class="legend">' + R.curves.map(c => `<span>${keySVG(c.color, modelShape(c.mk), c.dash)}${esc(data.models[c.mk].short + ' ' + engShort(c.eng))}</span>`).join('') +
      `<span>${keySVG('var(--text-secondary)', 'diamond')}measured run → model</span>` + (R.owner ? `<span>${keySVG('var(--text-primary)', 'star')}our per-expert plan (needs a split-expert build)</span>` : '') + '</div>';
    h += `<div class="plot" id="plot" tabindex="0" role="group" aria-roledescription="interactive chart" aria-label="Quality against speed, one curve per model" aria-describedby="chart-sum chart-keys">${fr.svg}</div>`;
    h += `<p class="sr-only" id="chart-sum">${esc(frontierSummary())}</p>`;
    h += `<p class="small muted" id="chart-keys">Each curve sweeps the speed target from slow to fast and keeps the best-quality layout. Large markers = best at 20 and at 25 tok/s; the ring marks the loaded layout. Diamonds are measured runs from the ledger, each with a line to what the model predicts for that run's own settings (dashed: part of the run is outside the model); the fastest per model is labelled with its gap. The curves use the page's default settings, so a run with other threads, page layout or draft can sit off its curve.${chartNotes()} Click a point to load it, or focus the chart: arrow keys move along a curve (up and down switch curves), Enter loads, Escape leaves.</p>`;
    h += `<details id="chart-table" ${st.table ? 'open' : ''}><summary>Chart as a table</summary>${frontierTable()}</details>`;
    h += '</section><section class="card" id="layout" aria-labelledby="layout-h">' + selectionHTML(sel) + '</section></div></div>';
    // waterfall
    h += page('bytes') + '<section class="card" id="bytes" aria-labelledby="bytes-h"><h2 id="bytes-h">Where every byte sits</h2>' + (sel ? `<div class="sub">${esc(sel.label)}</div>` : '') + wfLegend(sel) + `<p class="small muted">One bar per memory tier, to scale within the tier. Hover a segment, or tab to a bar and use the arrow keys, for its full name and size.</p>` + waterfallHTML(sel) + '</section>';
    if (sel) {
      const bh = bitsByHeatSVG(sel);
      h += `<section class="card" id="bits" aria-labelledby="bits-h"><h2 id="bits-h">Bits per weight by how often an expert is used</h2>${bh.leg}${bh.svg}<div class="callout small" id="bits-call">${bh.call}</div></section>`;
    }
    h += '</div>' + page('trust') + sec('trust', trustHTML()) + sec('checks', checksCardHTML()) + sec('calibration', calHTML()) + '</div>' + page('types') + sec('types', tablesHTML()) + sec('idea', ideaHTML()) + '</div>' + page('sources') + sec('sources', sourcesHTML()) + '</div></main>';
    const fid = document.activeElement && document.activeElement.id;
    app.innerHTML = h;
    bind(fr.anc);
    fitBarLabels();
    syncURL();
    const f = fid && document.getElementById(fid);
    if (f) { f.focus({ preventScroll: true }); if (fid === 'plot') showCursor(); }
  }
  // a #section link (tab, deep link, in-page link) shows that section's tab without a re-render; the tab's own link starts it
  // just under the sticky tabs, a link to a section further down scrolls to that section
  function showPage(id) {
    st.page = pageOf(id);
    PAGES.forEach(([p]) => { const x = document.getElementById('pg-' + p); if (x) x.hidden = p !== st.page; });
    app.querySelectorAll('nav.toc a').forEach(a => { if (a.getAttribute('href') === '#' + st.page) a.setAttribute('aria-current', 'page'); else a.removeAttribute('aria-current'); });
    fitBarLabels();
    const nav = app.querySelector('nav.toc'), t = document.getElementById(id);
    if (t && id !== st.page) t.scrollIntoView();
    else if (nav && window.scrollY > nav.offsetTop) window.scrollTo(0, nav.offsetTop);
  }
  window.addEventListener('hashchange', () => showPage(location.hash.slice(1)));

  function selectionHTML(sel) {
    if (!sel) return '<h2 id="layout-h">Selected layout</h2><p class="muted">Nothing reaches a feasible layout on this box with these knobs. Try another box, or turn on a What if upgrade.</p>';
    const { ctx, ev } = sel;
    const ts = timeSplitHTML(sel);
    let bits = 0; for (let e = 0; e < sel.lay.type.length; e++) bits += v(data.types[ctx.types[sel.lay.type[e]]].bpw);
    const avg = bits / sel.lay.type.length;
    let h = `<h2 id="layout-h">Selected layout</h2>${layoutPicker()}<div class="kpi" style="margin:8px 0">`;
    h += `<div><b>${f1(ev.tokps)}</b>tok/s projected</div><div><b>${f1(ev.quality)} %</b>top-1 agreement${chip(data.assume.quality_a)}</div><div><b>${ev.kld.toFixed(3)}</b>KLD (routing-weighted)</div><div><b>${f2(avg)}</b>expert bpw (mean)</div></div>`;
    if (avg < v(data.assume.cliff_bpw)) h += `<div class="warnline">Mean expert bits ${f2(avg)} is below ${v(data.assume.cliff_bpw)}: ${esc(data.assume.cliff_bpw.src)}</div>`;
    h += ts.svg + `<div class="small muted">${ts.text}</div>`;
    const needs = [];
    if (ctx.perExpert || sel.needsCode) needs.push('per-expert types need the split-expert ik build');
    ctx.tiers.forEach((t, i) => { if (t.gpu && t.gpu.needs_code && t.gpu.needs_code[ctx.eng] && sel.ev.cnt[i] > 0) needs.push(t.gpu.needs_code[ctx.eng]); });
    if (ctx.eng === 'sycl' && ctx.mk === 'v41') needs.push('V4.1 support in mainline llama.cpp SYCL is UNKNOWN');
    if (needs.length) h += `<div class="warnline">Needs: ${needs.map(esc).join('; ')}.</div>`;
    if (ctx.notes.length) h += `<div class="small muted">${ctx.notes.map(esc).join(' ')}</div>`;
    h += checksHTML(FM.checksFor(ctx, sel.lay, ev));
    h += `<div class="row" style="margin-top:8px"><button id="c-export">download expert map (JSON)</button><span class="small muted">tier + type for every routed expert, for a loader or a --custom-q recipe</span></div>`;
    h += `<p class="nextlink"><a href="#bytes">Where every byte of this layout sits</a> · <a href="#trust">how far to trust these numbers</a></p>`;
    return h;
  }
  // every named layout on this box (best at 20, best at 25, fastest per curve, and the owner layout) as one select;
  // a point picked on the chart shows as its own option
  function layoutPicker() {
    const cur = selCode(st.sel), opts = [];
    R.curves.forEach(c => {
      const nm = data.models[c.mk].short + ' ' + engShort(c.eng);
      if (c.b20) opts.push([selCode({ mk: c.mk, eng: c.eng, kind: 't20' }), nm + ': best at 20 tok/s']);
      if (c.b25) opts.push([selCode({ mk: c.mk, eng: c.eng, kind: 't25' }), nm + ': best at 25 tok/s']);
      opts.push([selCode({ mk: c.mk, eng: c.eng, kind: 'fast' }), nm + ': fastest']);
    });
    if (R.owner) opts.push(['owner', 'our per-expert plan (V4.1, per-expert types)']);
    if (st.sel && !opts.some(o => o[0] === cur)) opts.push([cur, data.models[st.sel.mk].short + ' ' + engShort(st.sel.eng) + ': point at ' + f1(st.sel.tokps) + ' tok/s']);
    return `<label class="small">Layout <select id="c-layout">${opts.map(o => `<option value="${esc(o[0])}" ${o[0] === cur ? 'selected' : ''}>${esc(o[1])}</option>`).join('')}</select></label>`;
  }
  const CHKCOL = { PASS: 'var(--good)', FAIL: 'var(--crit)', UNKNOWN: 'var(--neutral)' };
  // status as text: the same hues, dark enough for small text on the card (AA)
  const CHKTXT = { PASS: 'var(--good-text)', FAIL: 'var(--crit-text)', UNKNOWN: 'var(--neutral-text)' };
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
    const tr = trustRows();
    h += `<details><summary>Predictions for ${tr.length} recorded runs, ${tr.filter(x => !x.ev.length).length} not modelled</summary>`;
    tr.forEach(({ r, ev, why }) => {
      h += `<div style="margin:6px 0"><div><b>${esc(r.box)}</b> ${esc(r.label || r.recipe.id)}</div>` + (ev.length ? ev.map(rollupHTML).join('') : `<div class="small muted">UNMODELLED: ${esc(why || 'no prediction')}</div>`) + '</div>';
    });
    h += `</details><details><summary>Bench-fitted terms: ${Object.keys(B.gpu || {}).length} GPU backends, ${Object.keys(B.numa || {}).reduce((a, b) => a + Object.keys(B.numa[b]).length, 0)} NUMA placements (runs/bench.jsonl)</summary><div class="scroll"><table><tr><th>backend</th><th>cards</th><th class="n">reach of nameplate</th><th class="n">host ms/token</th><th class="n">MoE ms per layer</th></tr>`;
    Object.keys(B.gpu || {}).forEach(k => { const g = B.gpu[k]; h += `<tr><td>${esc(k)}</td><td>${esc(g.cards.join(', ') || 'borrowed from ' + g.borrowed_from)}</td><td class="n">${f2(v(g.eta))}${chip(g.eta)}</td><td class="n">${f2(v(g.host_ms))}${chip(g.host_ms)}</td><td class="n">${v(g.moe_layer_ms) === null ? '-' : v(g.moe_layer_ms).toFixed(3)}${chip(g.moe_layer_ms)}</td></tr>`; });
    h += '</table></div><div class="scroll"><table><tr><th>box</th><th>NUMA placement (policy, weight nodes, thread nodes, sockets, threads)</th><th class="n">decode vs one local node</th></tr>';
    Object.keys(B.numa || {}).forEach(b => Object.keys(B.numa[b]).forEach(k => { h += `<tr><td>${esc(b)}</td><td>${esc(k)}</td><td class="n">${f2(v(B.numa[b][k]))}x${chip(B.numa[b][k])}</td></tr>`; }));
    const all = S.by_class_placement.reduce((a, g) => a + g.n, 0);
    h += `</table></div></details><details><summary>External check: ${all} community rows, never a fitted term</summary><div class="scroll">`;
    h += '<table><tr><th>class / placement</th><th class="n">rows</th><th class="n">median |error|</th><th class="n">bias</th></tr>';
    S.by_class_placement.forEach(g => { h += `<tr><td>${esc(g.key)}</td><td class="n">${g.n}</td><td class="n">${g.median_abs_pct.toFixed(0)} %</td><td class="n">${(g.bias_pct >= 0 ? '+' : '') + g.bias_pct.toFixed(0)} %</td></tr>`; });
    h += `</table></div><div class="small muted">${S.unmodelled.length} rows UNMODELLED: ${esc(S.unmodelled.map(u => u.id + ' (' + u.why + ')').join('; '))}</div></details>`;
    return h + '</div>';
  }
  function checksCardHTML() {
    let h = '<div class="card"><h2>Hard checks on every recorded run</h2><div class="small muted">Each recipe is checked for: storage tier of on-file tensors (never a rotational disk), VRAM peak (post-warm-up + transient margin, not the warm-up reading), the -rtr prefill penalty on split models, DRAM headroom per NUMA node (node MemFree after experts + non-expert interleave + host buffers + page cache must stay >= 3 GiB), the box P0 settings (numa_balancing, demotion, swap), and the 2-bit read-share quality gate. Recorded windows show what a run cost.</div>';
    const recs = FM.checkRecipes(data), failed = recs.filter(r => r.checks.some(c => c.status === 'FAIL')).length;
    h += `<p class="small" style="margin:6px 0">${recs.length} recorded runs checked; ${failed} fail at least one check.</p><details><summary>Every run, check by check</summary>`;
    h += '<div class="scroll"><table><tr><th>when</th><th>run</th><th>outcome</th><th>storage</th><th>VRAM peak</th><th>NUMA free</th><th>P0</th><th>-rtr</th><th>quality</th><th>cost</th></tr>';
    recs.forEach(r => {
      const by = Object.fromEntries(r.checks.map(c => [c.id, c]));
      const cell = id => { const c = by[id]; return c ? `<td title="${esc(c.detail || '')}" style="color:${CHKTXT[c.status]};font-weight:${c.status === 'FAIL' ? 700 : 400}">${esc(c.status)}</td>` : '<td>-</td>'; };
      h += `<tr><td>${esc(r.when)}</td><td>${esc(r.box + ' ' + r.label)}</td><td class="small">${esc(r.outcome)}</td>${cell('storage')}${cell('vram_peak')}${cell('numa_free')}${cell('system')}${cell('rtr')}${cell('quality')}<td class="small muted">${esc(r.wasted || '')}</td></tr>`;
    });
    h += `</table></div></details><details><summary>Measured levers (10-17): ${(data.levers || []).length}</summary><div class="scroll">`;
    h += '<table><tr><th>box</th><th>lever</th><th>before</th><th>after</th><th>source</th></tr>';
    (data.levers || []).forEach(l => { h += `<tr><td>${esc(l.box + ' ' + l.model)}</td><td>${esc(l.lever)}</td><td>${esc(l.before)}</td><td>${esc(l.after)}${tagChip(l.tag, l.src)}</td><td class="small muted">${esc(l.src)}</td></tr>`; });
    return h + '</table></div></details></div>';
  }
  function runTip(a) {
    return `<b>${esc(a.label)}</b><br>measured ${f2(a.meas)} tok/s, ${armText(a)} (${esc(a.phase || '')}) · ` +
      (a.pred ? `model ${f2(a.pred)} for this run's settings, ${(a.gap >= 0 ? '+' : '') + (100 * a.gap).toFixed(1)} %` : `not modelled: ${esc(a.why || '')}`) +
      (a.waiver ? `<br><span class="muted">part of this run is outside the model: ${esc(a.waiver)}</span>` : '') +
      `<br>height: projected top-1 of its stored types (${esc(a.qhow)})` + (a.top1 ? `; measured top-1 ${f1(100 * a.top1)} % against its own reference` : '') +
      `<br><span class="muted">run ${esc(a.id)}${a.holdout ? ' (held out of the fit)' : ''}</span>`;
  }
  function frontierTable() {
    let h = '<div class="scroll"><table><caption class="sr-only">Quality vs speed: every point on each curve, the best layouts at 20 and 25 tok/s, and the measured runs</caption><tr><th>curve</th><th class="n">tok/s</th><th class="n">top-1 %</th><th class="n">KLD</th><th>kind</th></tr>';
    R.curves.forEach(c => {
      const nm = esc(data.models[c.mk].short + ' ' + engShort(c.eng));
      envelope(c.pts).forEach(p => { h += `<tr><td>${nm}</td><td class="n">${f2(p.tokps)}</td><td class="n">${f2(p.quality)}</td><td class="n">${p.kld.toFixed(4)}</td><td>sweep</td></tr>`; });
      if (c.b20) h += `<tr><td>${nm}</td><td class="n">${f2(c.b20.ev.tokps)}</td><td class="n">${f2(c.b20.ev.quality)}</td><td class="n">${c.b20.ev.kld.toFixed(4)}</td><td>best at 20</td></tr>`;
      if (c.b25) h += `<tr><td>${nm}</td><td class="n">${f2(c.b25.ev.tokps)}</td><td class="n">${f2(c.b25.ev.quality)}</td><td class="n">${c.b25.ev.kld.toFixed(4)}</td><td>best at 25</td></tr>`;
    });
    measuredHere().forEach(a => { h += `<tr><td>${esc(a.label)}</td><td class="n">${f2(a.meas)} measured / ${a.pred ? f2(a.pred) + ' model' : 'not modelled'}</td><td class="n">${f2(a.quality)}</td><td></td><td>run ${esc(a.id)}, ${armText(a)}${a.bad ? ', went bad (not on the chart)' : a.guess ? ', per-prompt or per-pass phases only, no single served decode (not on the chart)' : ''}${a.pred ? ', ' + (a.gap >= 0 ? '+' : '') + (100 * a.gap).toFixed(1) + ' %' : ': ' + esc(a.why || '')}${a.waiver ? '; partly outside the model' : ''}</td></tr>`; });
    return h + '</table></div>';
  }
  function calHTML() {
    let h = '<div class="card"><h2>Calibration: measured runs vs the model</h2><div class="small muted">Fit per box: a CPU efficiency factor f (shared by both models), a host overhead per model family, and on droopy the Optane read rate inside its measured band. Draft rows use the measured speed-up against their no-draft twin. "check" and "ref" rows are not fitted.</div>';
    h += '<div class="scroll"><table><tr><th>box</th><th>CPU factor f</th><th>host ms/token</th><th>draft speed-up</th><th>Optane rate</th><th class="n">rms log error</th></tr>';
    Object.keys(data.cal.boxes).forEach(bk => {
      const c = data.cal.boxes[bk];
      h += `<tr><td>${esc(bk)}</td><td>${f2(v(c.f))}${chip(c.f)}</td><td>${Object.keys(c.host).filter(m => MODELS.indexOf(m) >= 0 || m === 'v40').map(m => esc(m) + ' ' + f1(v(c.host[m])) + chip(c.host[m])).join(' ')}${c.split_ms ? ' · split GGUF +' + f1(v(c.split_ms)) + chip(c.split_ms) : ''}</td><td>${Object.keys(c.draft).map(m => esc(m) + ' x' + f2(v(c.draft[m])) + chip(c.draft[m])).join(' ') || '-'}</td><td>${c.pmem_bw ? f1(v(c.pmem_bw)) + ' GB/s' + chip(c.pmem_bw) : '-'}</td><td class="n">${c.rms !== undefined ? c.rms.toFixed(3) : '-'}</td></tr>`;
    });
    h += `</table></div><details><summary>Every measured run against the model (${data.cal.rows.length} rows)</summary><div class="scroll"><table><tr><th>run</th><th>use</th><th class="n">measured tok/s</th>`;
    h += '<th class="n">model</th><th class="n">residual</th><th>source</th></tr>';
    data.cal.rows.forEach(r => { h += `<tr><td>${esc(r.label)}</td><td>${esc(r.fit)}</td><td class="n">${f2(r.meas)}${tagChip(r.tag, r.src)}</td><td class="n">${f2(r.pred)}</td><td class="n">${(100 * r.resid).toFixed(1)} %</td><td class="small muted">${esc(r.src)}${r.mult_src ? ' · speed-up ' + f2(r.mult) + ' = ' + esc(r.mult_src) : ''}</td></tr>`; });
    return h + '</table></div></details></div>';
  }
  function tablesHTML() {
    const tk = Object.keys(data.types);
    let h = '<div class="card"><h2>Type table (editable)</h2><div class="small muted">CPU efficiency = share of the memory ceiling the CPU expert kernel reaches with this type (odie AVX2 smoke). KLD = full-model KL divergence if every routed expert used this type (community priors where a row exists). Edits recalibrate and recompute; they stay in this browser.</div>';
    h += '<div class="scroll"><table><tr><th>type</th><th class="n">bpw</th><th class="n">CPU efficiency</th><th class="n">KLD V4.1</th><th class="n">KLD GLM</th><th class="n">KLD Flash-Next</th><th>engines</th><th>note</th></tr>';
    tk.forEach(q => {
      const t = data.types[q];
      const kap = data.overrides.kappa[q] !== undefined ? data.overrides.kappa[q] : v(t.kappa);
      const kv = fam => { const o = data.overrides.kld[fam] && data.overrides.kld[fam][q]; return o !== undefined ? o : v(t.kld[fam]); };
      const cell = (fam) => t.kld[fam] && v(t.kld[fam]) !== null ? `<input type="number" step="0.001" min="0" id="kld-${fam}-${q}" aria-label="KLD ${fam} for ${q}" data-kld="${fam}" data-q="${q}" value="${kv(fam)}">${chip(t.kld[fam])}` : '-';
      h += `<tr><td>${esc(q)}</td><td class="n">${v(t.bpw).toFixed(3)}${chip(t.bpw)}</td><td class="n">${kap === null || kap === undefined ? '-' : `<input type="number" step="0.01" min="0.05" max="2" id="kap-${q}" aria-label="CPU efficiency for ${q}" data-kappa="${q}" value="${(+kap).toFixed(3)}">`}${chip(t.kappa)}</td><td class="n">${cell('v41')}</td><td class="n">${cell('glm')}</td><td class="n">${cell('fnx')}</td><td>${esc((t.engines || []).join(', '))}</td><td class="small muted">${esc(t.note || '')}</td></tr>`;
    });
    h += `</table></div><button id="c-reset-ov">reset edits</button> <span class="small muted">Quality curve: top-1 % = 100 - ${v(data.assume.quality_a).toFixed(1)} x KLD^${v(data.assume.quality_b).toFixed(3)}${chip(data.assume.quality_a)}</span></div>`;
    return h;
  }
  function ideaHTML() {
    return `<div class="card"><h2>My idea</h2><div class="small muted">Write the idea; the download carries it with every knob, the box and the selected layout's numbers. It stays in this browser too.</div><textarea id="c-idea" aria-label="My idea" placeholder="e.g. put the 1080 Ti in droopy beside the RTX 6000 and give it the next 200 hottest experts">${esc(load('idea', ''))}</textarea><button id="c-idea-dl">download idea (JSON)</button></div>`;
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
  function showTip(e, html) { tip.innerHTML = html; tip.style.display = 'block'; const x = Math.max(4, Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8)), y = Math.min(e.clientY + 14, window.innerHeight - tip.offsetHeight - 8); tip.style.left = x + 'px'; tip.style.top = y + 'px'; }
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
    on('c-layout', 'change', e => { st.sel = parseSel(e.target.value); loadSelection(); render(); });
    [['intro', o => save('intro', o)], ['heatd', o => save('heatd', o)], ['whatifd', o => save('whatifd', o)], ['chart-table', o => { st.table = o; }]].forEach(([id, fn]) => on(id, 'toggle', e => fn(e.target.open)));
    bindKeys();
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
        else if (k === 'anc') { const a = anc[+n.dataset.a]; showTip(e, runTip(a)); }
        else if (k === 'owner') { const o = R.owner.ev; showTip(e, `<b>Our per-expert plan (droopy)</b><br>hottest coding experts MXFP4 in VRAM, other coding experts q3_K in DRAM, the rest MXFP4 in Optane<br>${f2(o.tokps)} tok/s · top-1 ${f2(o.quality)} %<br><span class="muted">needs per-expert types (split-expert build); click to load</span>`); }
        else if (k === 'seg') { const rows = tierRows(R.selected), g = rows[+n.dataset.tier].segs[+n.dataset.seg]; showTip(e, `<b>${esc(g.label)}</b><br>${f2(g.gib)} GiB${g.tag ? ' ' + tagChip(g.tag, g.src || '') : ''}${g.src ? '<br><span class="muted">' + esc(g.src) + '</span>' : ''}`); }
        else if (k === 'time') { const p = timeSplitHTML(R.selected).parts[+n.dataset.p]; showTip(e, `<b>${esc(p[0])}</b><br>${f1(p[1])} ms per token`); }
        else if (k === 'bh') { const t = R.selected.ctx.tiers[+n.dataset.t]; showTip(e, `<b>${esc(t.name)}</b><br>heat bin ${+n.dataset.b + 1}: ${n.dataset.n} experts at ${(+n.dataset.bpw).toFixed(2)} bpw on average`); }
        else if (k === 'thr') { showTip(e, n.dataset.u === '1' ? 'threads running the expert kernel' : 'idle threads: ' + esc(data.boxes[st.box].threads_note || '')); }
      });
      n.addEventListener('mouseleave', hideTip);
      if (n.getAttribute('role') === 'listitem') { n.addEventListener('focus', () => { n.dispatchEvent(new MouseEvent('mousemove', tipAt(n))); }); n.addEventListener('blur', hideTip); }
      if (k === 'pt' || k === 't20' || k === 't25' || k === 'owner') n.addEventListener('click', () => {
        hideTip();
        if (k === 'owner') st.sel = { kind: 'owner', mk: 'v41', eng: 'ik' };
        else { const c = R.curves[+n.dataset.c]; st.sel = k === 'pt' ? { mk: c.mk, eng: c.eng, kind: 'pt', tokps: +n.dataset.x } : { mk: c.mk, eng: c.eng, kind: k }; }
        loadSelection(); render();
      });
    });
  }
  // a tooltip for keyboard focus: the event coordinates of the element's lower left corner
  const tipAt = el => { const r = el.getBoundingClientRect(); return { clientX: r.left, clientY: r.bottom, bubbles: false }; };
  const live = document.getElementById('live');
  const say = t => { if (live) live.textContent = t; };
  // chart cursor: st.cur = {g: group (curve, or the owner layout), i: point}; moves without a re-render, Enter loads the layout
  function curPoint() {
    const nav = R.nav || [];
    if (!nav.length) return null;
    const si = st.sel ? nav.findIndex(n => st.sel.kind === 'owner' ? n.owner : !n.owner && R.curves[n.ci].mk === st.sel.mk && R.curves[n.ci].eng === st.sel.eng) : -1;
    const c = st.cur || (si >= 0 ? { g: si, i: nav[si].pts.reduce((b, q, i) => R.selected && Math.abs(q.tokps - R.selected.ev.tokps) < Math.abs(nav[si].pts[b].tokps - R.selected.ev.tokps) ? i : b, 0) } : {});
    const g = Math.max(0, Math.min(nav.length - 1, c.g || 0)), pts = nav[g].pts;
    if (!pts.length) return null;
    return { g, i: Math.max(0, Math.min(pts.length - 1, c.i === undefined ? pts.length - 1 : c.i)), grp: nav[g] };
  }
  function showCursor() {
    const cur = curPoint(), ring = document.getElementById('fcur');
    if (!cur || !ring) return;
    st.cur = { g: cur.g, i: cur.i };
    const p = cur.grp.pts[cur.i];
    ring.setAttribute('cx', p.x); ring.setAttribute('cy', p.y); ring.setAttribute('visibility', 'visible');
    const txt = `${cur.grp.name}: ${f1(p.tokps)} tok/s, top-1 ${f1(p.quality)} %, KLD ${p.kld.toFixed(4)}. Point ${cur.i + 1} of ${cur.grp.pts.length}. Enter loads it.`;
    const r = ring.getBoundingClientRect();
    showTip({ clientX: r.right, clientY: r.bottom }, `<b>${esc(cur.grp.name)}</b><br>${f2(p.tokps)} tok/s · top-1 ${f2(p.quality)} % · KLD ${p.kld.toFixed(4)}<br><span class="muted">Enter loads this layout</span>`);
    say(txt);
  }
  function bindKeys() {
    const plot = document.getElementById('plot');
    if (plot) {
      plot.addEventListener('focus', showCursor);
      plot.addEventListener('blur', () => { const r = document.getElementById('fcur'); if (r) r.setAttribute('visibility', 'hidden'); hideTip(); });
      plot.addEventListener('keydown', e => {
        const cur = curPoint();
        if (!cur) return;
        const nav = R.nav, p = cur.grp.pts[cur.i];
        const near = g => nav[g].pts.reduce((b, q, i) => Math.abs(q.tokps - p.tokps) < Math.abs(nav[g].pts[b].tokps - p.tokps) ? i : b, 0);
        const step = { ArrowRight: [0, 1], ArrowLeft: [0, -1], ArrowUp: [-1, 0], ArrowDown: [1, 0] }[e.key];
        if (step && step[0]) { let g = cur.g; do { g = (g + step[0] + nav.length) % nav.length; } while (!nav[g].pts.length && g !== cur.g); st.cur = { g, i: near(g) }; }
        else if (step) st.cur = { g: cur.g, i: cur.i + step[1] };
        else if (e.key === 'Home' || e.key === 'End') st.cur = { g: cur.g, i: e.key === 'Home' ? 0 : cur.grp.pts.length - 1 };
        else if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          st.sel = cur.grp.owner ? { kind: 'owner', mk: 'v41', eng: 'ik' } : { mk: R.curves[cur.grp.ci].mk, eng: R.curves[cur.grp.ci].eng, kind: 'pt', tokps: p.tokps };
          loadSelection(); render();
          say('Loaded ' + (R.selected ? R.selected.label : 'nothing') + '. The layout card and the Where it lives tab now show it.');
          return;
        } else if (e.key === 'Escape') { plot.blur(); return; }
        else return;
        e.preventDefault();
        showCursor();
      });
    }
    app.querySelectorAll('.hbar > [role=listitem]:not([data-k])').forEach(n => {
      const t = () => '<b>' + esc(n.getAttribute('aria-label')) + '</b>';
      n.addEventListener('mousemove', e => showTip(e, t())); n.addEventListener('mouseleave', hideTip);
      n.addEventListener('focus', () => showTip(tipAt(n), t())); n.addEventListener('blur', hideTip);
    });
    // byte bars: one tab stop each, arrow keys move between segments
    app.querySelectorAll('.hbar').forEach(bar => bar.addEventListener('keydown', e => {
      const d = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
      if (!d) { if (e.key === 'Escape') hideTip(); return; }
      const items = [...bar.children], i = items.indexOf(document.activeElement), n = items[Math.max(0, Math.min(items.length - 1, i + d))];
      e.preventDefault();
      items.forEach(x => x.setAttribute('tabindex', x === n ? '0' : '-1'));
      n.focus();
    }));
  }
  document.addEventListener('keydown', e => { if (e.key === 'Escape') hideTip(); });
  const stamp = () => new Date().toISOString().replace(/[-:]/g, '').slice(0, 13);
  function download(name, obj) {
    const blob = new Blob([JSON.stringify(obj, null, 1)], { type: 'application/json' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = name; document.body.appendChild(a); a.click();
    setTimeout(() => { URL.revokeObjectURL(a.href); a.remove(); }, 500);
  }

  let rz = null, lastW = 0;
  window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { if (Math.abs(app.clientWidth - lastW) > 40) { lastW = app.clientWidth; render(); } else fitBarLabels(); }, 200); });
  try { compute(); render(); lastW = app.clientWidth; const t = location.hash.slice(1) !== st.page && document.getElementById(location.hash.slice(1)); if (t) t.scrollIntoView(); } catch (e) { app.innerHTML = '<pre>' + esc(e.stack || e) + '</pre>'; throw e; }
  window.FIT_APP = { st, get R() { return R; }, compute, render };
})();
