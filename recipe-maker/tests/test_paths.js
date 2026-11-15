#!/usr/bin/env node
// Summit paths view (src/paths.js): each seeded target draws its lineage on the per-agent against aggregate chart (one head
// marker per node, one edge per parent, predicted edges dashed, the goal region), the card lists the never-composed pairs,
// report.js --paths prints every node. Run: node tests/test_paths.js (needs data.json: python3 build.py)
'use strict';
const assert = require('assert');
const fs = require('fs'), path = require('path'), cp = require('child_process');
const ROOT = path.join(__dirname, '..');
const FP = require(path.join(ROOT, 'src', 'paths.js'));
const data = JSON.parse(fs.readFileSync(path.join(ROOT, 'data.json'), 'utf8'));
let n = 0;
const t = (name, fn) => { fn(); n++; console.log('ok ' + name); };
const target = id => data.paths.targets.find(x => x.id === id);
const count = (s, re) => (s.match(re) || []).length;

t('build.py loads the three seeded targets with the serving-lever catalog', () => {
  assert.ok(data.paths && data.paths.catalog, 'data.paths (build.py from data/paths/)');
  assert.deepStrictEqual(data.paths.targets.map(x => x.id).sort(), ['ds41-one-agent', 'glm53-batched', 'qfn-odie-agents']);
  assert.deepStrictEqual(data.paths.catalog.exclusive_kinds, ['engine', 'quant']);
});

t('GLM: every node with a total gets a head marker, every lineage edge is drawn, d3 sits in the goal at 8 agents', () => {
  const g = target('glm53-batched'), s = FP.chart(g, { width: 720, height: 420 });
  assert.strictEqual(count(s, /<circle class="head"/g), g.nodes.length);
  assert.strictEqual(count(s, /<line class="edge"/g), g.edges.length);
  assert.ok(g.edges.length >= 4);
  assert.match(s, /<polygon class="goal"/);
  assert.match(s, /data-edge="d2 d3"[^>]*stroke-dasharray/, 'a predicted edge is dashed');
  assert.match(s, /data-edge="glmlite-snoopy glmlite-b70-tier"(?![^>]*stroke-dasharray)/, 'a measured edge is solid');
  assert.match(s, /data-node="d3" data-state="open"/);
  assert.match(s, /data-node="odie-sycl" data-state="dropped"/);
  assert.strictEqual(FP.headOf(g.nodes.find(x => x.id === 'd3')).agents, 8);
  assert.ok(count(s, /<rect class="range"/g) >= 5, 'predicted ranges draw their boxes (a single predicted number draws none)');
});

t('the head markers land where their numbers say: d3 right of and above glmlite-b70-tier', () => {
  const s = FP.chart(target('glm53-batched'), { width: 720, height: 420 });
  const at = id => { const m = s.match(new RegExp(`class="head" data-node="${id}"[^>]*cx="([\\d.]+)" cy="([\\d.]+)"`)); return m.slice(1).map(Number); };
  const [x3, y3] = at('d3'), [xb, yb] = at('glmlite-b70-tier');
  assert.ok(x3 > xb && y3 < yb, `d3 ${x3},${y3} vs b70-tier ${xb},${yb}`);
});

t('a one-agent goal draws a band along the 1-agent ray, and labels that would collide move apart', () => {
  const d = target('ds41-one-agent'), s = FP.chart(d, { width: 720, height: 420 });
  assert.match(s, /<line class="goal"/);
  assert.match(s, /goal: 1 agent at 25\+ each/);
  const ys = id => Number(s.match(new RegExp(`<text class="label ink" x="[\\d.]+" y="([\\d.]+)">${id}<`))[1]);
  assert.ok(Math.abs(ys('lassie-q8') - ys('droopy-optane')) >= 12, 'lassie 13.2 and droopy 13.1 labels apart');
  assert.ok(Math.abs(ys('cpu-control') - ys('ced')) >= 12);
});

t('a head label at the right edge flips left of its marker and stays inside the plot', () => {
  const s = FP.chart(target('qfn-odie-agents'), { width: 720, height: 420 });
  const m = s.match(/<text class="label ink" x="([\d.]+)" y="[\d.]+">sycl-tensor-graphs</);
  assert.ok(m, 'sycl-tensor-graphs is labelled');
  assert.ok(Number(m[1]) + 'sycl-tensor-graphs'.length * 6.2 <= 720, 'label ends inside the chart: x ' + m[1]);
});

t('the card: one section per target with its chart, node table and never-composed pairs', () => {
  const h = FP.html(data, { width: 600 });
  data.paths.targets.forEach(x => {
    assert.match(h, new RegExp(`id="path-${x.id}"`));
    assert.ok(x.never_composed.length > 0, x.id);
  });
  assert.strictEqual(count(h, /<svg class="pathchart"/g), data.paths.targets.length);
  assert.match(h, /never composed/i);
  assert.match(h, /<tr data-node="conv-door">/);
});

t('report.js --paths prints every node and --json round-trips data.paths', () => {
  const out = cp.execFileSync('node', [path.join(ROOT, 'report.js'), '--paths'], { cwd: ROOT, encoding: 'utf8' });
  data.paths.targets.forEach(x => x.nodes.forEach(nd => assert.ok(out.includes('  ' + nd.id + ' ['), nd.id)));
  assert.match(out, /d3 \[open\] <- d2/);
  const j = JSON.parse(cp.execFileSync('node', [path.join(ROOT, 'report.js'), '--paths', '--json'], { cwd: ROOT, encoding: 'utf8' }));
  assert.deepStrictEqual(j.targets.map(x => x.id), data.paths.targets.map(x => x.id));
});

t('no data/paths: html and text say so instead of throwing', () => {
  const empty = Object.assign({}, data, { paths: null });
  assert.strictEqual(typeof FP.html(empty, {}), 'string');
  assert.strictEqual(typeof FP.text(empty), 'string');
});

console.log(n + ' passed');
