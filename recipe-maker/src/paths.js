// Summit paths (data.paths, src/paths.py): each target's lineage drawn on its per-agent against total speed chart, the
// node table and the lever pairs the target never composed. Shared by index.html (browser) and report.js (node).
(function (root) {
  'use strict';
  const esc = s => String(s === undefined || s === null ? '' : s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
  const mid = x => { const v = x ? x.v : null; return v === null || v === undefined ? null : Array.isArray(v) ? (v[0] + v[1]) / 2 : v; };
  const lo = x => { const v = x ? x.v : null; return v === null || v === undefined ? null : Array.isArray(v) ? v[0] : v; };
  const hi = x => { const v = x ? x.v : null; return v === null || v === undefined ? null : Array.isArray(v) ? v[1] : v; };
  const fv = x => { if (!x || x.v === null || x.v === undefined) return '-'; const r = y => (Math.abs(y) >= 100 ? y.toFixed(0) : y.toFixed(1)); return Array.isArray(x.v) ? r(x.v[0]) + '-' + r(x.v[1]) : r(x.v); };
  const tagShort = x => !x ? '' : x.tag === 'MEASURED' ? 'measured' : x.tag === 'PREDICTED' ? 'predicted' : 'unknown';
  const STATE = { kept: 'kept', dropped: 'dropped', open: 'open' };
  const point = (nd, n) => nd.points.find(p => p.agents === n);
  const headOf = nd => nd.head === null || nd.head === undefined ? null : point(nd, nd.head);
  const nodeOf = (t, id) => t.nodes.find(n => n.id === id);
  const leverLabel = (cat, id) => { const l = cat.levers.find(x => x.id === id); return l ? l.label : id; };

  // the chart: x = tok/s each agent gets, y = tok/s over all agents; a node's points (one per agent count) join as its load
  // curve; its head point carries its marker; the lineage edges join parent heads to child heads (the path); the goal is
  // the region right of per_agent_min between the rays total = n x per-agent for the goal's agent band
  function chart(t, opts) {
    opts = opts || {};
    const W = opts.width || 640, H = opts.height || 400, L = 52, Rm = 16, T = 14, B = 42;
    const pts = t.nodes.flatMap(nd => nd.points.filter(p => mid(p.per_agent) !== null && mid(p.aggregate) !== null));
    const g = t.goal || {};
    const xmax = Math.max(10, ...pts.map(p => hi(p.per_agent)), g.per_agent_min ? g.per_agent_min * 1.6 : 0) * 1.08;
    const ymax = Math.max(10, ...pts.map(p => hi(p.aggregate)), g.per_agent_min && g.agents ? g.per_agent_min * g.agents[0] * 1.3 : 0) * 1.08;
    const X = x => L + (W - L - Rm) * Math.min(x, xmax) / xmax, Y = y => T + (H - T - B) * (1 - Math.min(y, ymax) / ymax);
    const step = m => { const r = m / 6, e = Math.pow(10, Math.floor(Math.log10(r))); return [1, 2, 5, 10].map(k => k * e).find(k => k >= r); };
    let s = `<svg class="pathchart" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="width:${W}px;max-width:100%" role="img" aria-label="${esc(t.label)}: tok/s each agent against tok/s in total, one line per lineage">`;
    const xs = step(xmax), ys = step(ymax);
    for (let x = 0; x <= xmax; x += xs) s += `<line x1="${X(x)}" x2="${X(x)}" y1="${T}" y2="${H - B}" stroke="var(--grid, #ddd)"/><text x="${X(x)}" y="${H - B + 14}" text-anchor="middle">${+x.toFixed(2)}</text>`;
    for (let y = 0; y <= ymax; y += ys) s += `<line x1="${L}" x2="${W - Rm}" y1="${Y(y)}" y2="${Y(y)}" stroke="var(--grid, #ddd)"/><text x="${L - 6}" y="${Y(y) + 4}" text-anchor="end">${+y.toFixed(2)}</text>`;
    s += `<text x="${(L + W - Rm) / 2}" y="${H - 8}" text-anchor="middle">${esc(t.metric)} each agent gets</text>`;
    s += `<text transform="translate(14 ${(T + H - B) / 2}) rotate(-90)" text-anchor="middle">${esc(t.metric)} in total</text>`;
    // the goal region and the agent-count rays that bound it
    if (g.per_agent_min && g.agents) {
      const [a0, a1] = g.agents, x0 = g.per_agent_min, xe = xmax;
      // x >= x0 between the rays total = a0 x and total = a1 x, clipped to the plot
      const poly = [[x0, a0 * x0], [x0, Math.min(ymax, a1 * x0)]];
      if (a1 * xe > ymax) poly.push([ymax / a1, ymax], [xe, ymax]); else poly.push([xe, a1 * xe]);
      if (a0 * xe > ymax) poly.push([ymax / a0, ymax]); else poly.push([xe, a0 * xe]);
      const band = a0 === a1 ? `${a0} agent${a0 > 1 ? 's' : ''}` : `${a0}-${a1} agents`;
      // one agent count: the region has no area, so the goal is a thick stretch of that ray
      if (a0 === a1) { const xe1 = Math.min(xe, ymax / a0); s += `<line class="goal" x1="${X(x0).toFixed(1)}" y1="${Y(a0 * x0).toFixed(1)}" x2="${X(xe1).toFixed(1)}" y2="${Y(a0 * xe1).toFixed(1)}" stroke="var(--good-bg, #bfe5bf)" stroke-width="14" stroke-linecap="round" opacity="0.8"><title>goal: ${band} at ${x0}+ ${esc(t.metric)} each</title></line>`; }
      else s += `<polygon class="goal" points="${poly.map(p => X(p[0]).toFixed(1) + ',' + Y(p[1]).toFixed(1)).join(' ')}" fill="var(--good-bg, #e3f4e3)" opacity="0.7"><title>goal: ${band} at ${x0}+ ${esc(t.metric)} each</title></polygon>`;
      s += `<text x="${X(x0) + 4}" y="${a0 === a1 ? Y(a0 * x0) + 22 : (Y(a1 * x0) - 4 > T + 10 ? Y(a1 * x0) - 4 : T + 12)}" class="ink">goal: ${band} at ${x0}+ each</text>`;
    }
    [1, 4, 8, 16, 32].forEach(n => {
      const xe = Math.min(xmax, ymax / n);
      if (xe < xmax * 0.03) return;
      s += `<line x1="${X(0)}" y1="${Y(0)}" x2="${X(xe)}" y2="${Y(n * xe)}" stroke="var(--text-secondary, #999)" stroke-dasharray="2 4" opacity="0.6"/><text x="${X(xe) - 2}" y="${Y(n * xe) + (n * xe >= ymax * 0.98 ? 12 : -3)}" text-anchor="end" class="muted" font-size="10">${n} agent${n > 1 ? 's' : ''}</text>`;
    });
    // each node's load curve and its predicted ranges
    t.nodes.forEach(nd => {
      const ps = nd.points.filter(p => mid(p.per_agent) !== null && mid(p.aggregate) !== null).sort((a, b) => a.agents - b.agents);
      ps.forEach(p => {
        if ((p.per_agent.tag === 'PREDICTED' || p.aggregate.tag === 'PREDICTED') && (Array.isArray(p.per_agent.v) || Array.isArray(p.aggregate.v))) {
          const x0 = X(lo(p.per_agent)), x1 = X(hi(p.per_agent)), y0 = Y(hi(p.aggregate)), y1 = Y(lo(p.aggregate));
          s += `<rect class="range" data-node="${esc(nd.id)}" x="${x0.toFixed(1)}" y="${y0.toFixed(1)}" width="${Math.max(1, x1 - x0).toFixed(1)}" height="${Math.max(1, y1 - y0).toFixed(1)}" fill="var(--s3, #c9b8e8)" opacity="0.25"/>`;
        }
      });
      if (ps.length > 1) s += `<polyline class="load" data-node="${esc(nd.id)}" fill="none" stroke="var(--text-secondary, #999)" stroke-width="1" points="${ps.map(p => X(mid(p.per_agent)).toFixed(1) + ',' + Y(mid(p.aggregate)).toFixed(1)).join(' ')}"/>`;
      ps.forEach(p => {
        const meas = p.per_agent.tag === 'MEASURED' && p.aggregate.tag === 'MEASURED';
        s += `<circle class="pt" cx="${X(mid(p.per_agent)).toFixed(1)}" cy="${Y(mid(p.aggregate)).toFixed(1)}" r="2.5" fill="${meas ? 'var(--text-secondary, #777)' : 'var(--surface-1, #fff)'}" stroke="var(--text-secondary, #777)"><title>${esc(nd.id)} at ${p.agents} agent${p.agents > 1 ? 's' : ''}: ${fv(p.per_agent)} each (${tagShort(p.per_agent)}), ${fv(p.aggregate)} total (${tagShort(p.aggregate)})</title></circle>`;
      });
    });
    // the lineage: parent head to child head
    (t.edges || []).forEach(([a, b]) => {
      const pa = headOf(nodeOf(t, a)), pb = headOf(nodeOf(t, b));
      if (!pa || !pb) return;
      const pred = [pa, pb].some(p => p.aggregate.tag !== 'MEASURED');
      s += `<line class="edge" data-edge="${esc(a)} ${esc(b)}" x1="${X(mid(pa.per_agent)).toFixed(1)}" y1="${Y(mid(pa.aggregate)).toFixed(1)}" x2="${X(mid(pb.per_agent)).toFixed(1)}" y2="${Y(mid(pb.aggregate)).toFixed(1)}" stroke="var(--s1, #2a6fdb)" stroke-width="2"${pred ? ' stroke-dasharray="6 4"' : ''}/>`;
    });
    // head markers and labels last, on top; a label that would overlap one already placed moves down a line
    const placed = [];
    t.nodes.forEach(nd => {
      const p = headOf(nd);
      if (!p) return;
      const x = X(mid(p.per_agent)), y = Y(mid(p.aggregate)), st = nd.verdict.state;
      const fill = st === 'kept' ? 'var(--s1, #2a6fdb)' : st === 'open' ? 'var(--surface-1, #fff)' : 'var(--neutral-2, #bbb)';
      s += `<circle class="head" data-node="${esc(nd.id)}" data-state="${st}" cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="6" fill="${fill}" stroke="var(--s1, #2a6fdb)" stroke-width="2"><title>${esc(nd.id)}: ${esc(nd.label)}. ${p.agents} agent${p.agents > 1 ? 's' : ''}: ${fv(p.per_agent)} each, ${fv(p.aggregate)} total (${tagShort(p.aggregate)}). ${esc(STATE[st])}${nd.verdict.why ? ': ' + esc(nd.verdict.why) : ''}</title></circle>`;
      if (st === 'dropped') s += `<path d="M${x - 4} ${y - 4}L${x + 4} ${y + 4}M${x - 4} ${y + 4}L${x + 4} ${y - 4}" stroke="var(--text-primary, #222)" stroke-width="1.5" pointer-events="none"/>`;
      const lw = nd.id.length * 6.2, lx = x + 9 + lw > W - Rm ? x - 9 - lw : x + 9;   // flip left at the right edge
      let ly = y - 7;
      while (placed.some(q => Math.abs(q[1] - ly) < 12 && lx < q[0] + q[2] && q[0] < lx + lw)) ly += 13;
      placed.push([lx, ly, lw]);
      s += `<text class="label ink" x="${lx.toFixed(1)}" y="${ly.toFixed(1)}">${esc(nd.id)}</text>`;
    });
    return s + '</svg>';
  }

  function nodeTable(t, cat) {
    let h = `<div style="overflow-x:auto"><table class="small pathnodes"><caption>${esc(t.label)}: every node, its levers and its points</caption><tr><th>node</th><th>parents</th><th>levers composed</th><th>points: agents, each / total ${esc(t.metric)}, TTFT s</th><th>recipe</th><th>verdict</th></tr>`;
    t.nodes.forEach(nd => {
      const pts = nd.points.map(p => `${p.agents}: ${fv(p.per_agent)} / ${fv(p.aggregate)}${p.ttft_s && p.ttft_s.v !== null ? ', ' + fv(p.ttft_s) + ' s' : ''} <span class="muted">(${tagShort(p.aggregate)})</span>`).join('<br>');
      h += `<tr data-node="${esc(nd.id)}"><td><b>${esc(nd.id)}</b><br><span class="muted">${esc(nd.label)}</span></td><td>${esc((nd.parents || []).join(', ') || '-')}</td><td>${nd.levers.map(x => esc(leverLabel(cat, x))).join(', ')}</td><td>${pts}</td><td>${esc(nd.recipe)}</td><td>${esc(STATE[nd.verdict.state])}${nd.verdict.why ? ': ' + esc(nd.verdict.why) : ''}</td></tr>`;
    });
    return h + '</table></div>';
  }

  function pairsHTML(t, cat) {
    const ps = t.never_composed || [];
    if (!ps.length) return '<p class="small">Every pair of levers this target tried has been composed in some node.</p>';
    const both = ps.filter(p => p.tried[0] && p.tried[1]);
    let h = `<details class="pairs"><summary>${ps.length} lever pair${ps.length > 1 ? 's' : ''} never composed on this target (${both.length} of them both tried alone)</summary><ul class="small">`;
    ps.forEach(p => { h += `<li data-pair="${esc(p.a)}+${esc(p.b)}">${esc(leverLabel(cat, p.a))} + ${esc(leverLabel(cat, p.b))}${p.tried[0] && p.tried[1] ? '' : ' <span class="muted">(' + [p.tried[0] ? '' : leverLabel(cat, p.a), p.tried[1] ? '' : leverLabel(cat, p.b)].filter(Boolean).map(esc).join(', ') + ' never tried)</span>'}</li>`; });
    return h + '</ul></details>';
  }

  function html(data, opts) {
    const P = data.paths;
    if (!P || !P.targets.length) return '<h2 id="paths-h">Summit paths</h2><p>No summit paths in this build (data/paths/).</p>';
    let h = '<h2 id="paths-h">Summit paths</h2><p class="small">One chart per target: tok/s each agent gets (right) against tok/s over all agents (up). Each node is one layout we ran or predicted; its points at several agent counts join as a thin line, and its large marker sits at its best total inside the goal band. The thick lines are the lineage, parent to child: solid where both ends were measured, dashed where one is predicted. Filled = kept, hollow = open, crossed = dropped; shaded boxes are predicted ranges; the green region is the goal.</p>';
    P.targets.forEach(t => {
      h += `<section class="card" id="path-${esc(t.id)}"><h3>${esc(t.label)}</h3><p class="small muted">${esc(t.model)}. ${esc(t.about)}</p>`;
      h += `<div class="plot">${chart(t, opts)}</div>` + nodeTable(t, P.catalog) + pairsHTML(t, P.catalog) + '</section>';
    });
    return h;
  }

  // report.js --paths: plain text
  function text(data) {
    const P = data.paths;
    if (!P) return 'no summit paths in this build (data/paths/)';
    return P.targets.map(t => {
      const lines = [`${t.label} (${t.file}): ${t.nodes.length} nodes, ${(t.edges || []).length} edges, ${(t.never_composed || []).length} lever pairs never composed`];
      t.nodes.forEach(nd => {
        const p = headOf(nd);
        lines.push(`  ${nd.id} [${nd.verdict.state}]${(nd.parents || []).length ? ' <- ' + nd.parents.join(', ') : ''}: ${nd.levers.join(' + ')}` +
          (p ? `; head ${p.agents} agents: ${fv(p.per_agent)} each, ${fv(p.aggregate)} total (${tagShort(p.aggregate)})` : ''));
      });
      return lines.join('\n');
    }).join('\n\n');
  }

  const api = { chart, nodeTable, pairsHTML, html, text, mid, headOf };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.FitPaths = api;
})(this);
