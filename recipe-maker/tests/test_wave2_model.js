'use strict';
const assert = require('assert');
const fs = require('fs');
const path = require('path');
const FM = require('../src/model.js');
const ADV = require('../src/advisor.js');

const D = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'data.json'), 'utf8'));
let passed = 0;

function test(name, fn) {
  fn();
  passed++;
  console.log('ok ' + name);
}

test('calibrate does not mutate caller data in place while applyCal applies', () => {
  const clone = JSON.parse(JSON.stringify(D));
  const beforeBw = Object.fromEntries(Object.entries(clone.boxes).map(([k, b]) => [k, b.pmem && b.pmem.bw && b.pmem.bw.v]));
  const beforeKappa = Object.fromEntries(Object.entries(clone.types).map(([k, t]) => [k, t.kappa && t.kappa.v]));
  const cal = FM.calibrate(clone);
  assert.ok(cal && cal.boxes);
  for (const [k, v] of Object.entries(beforeBw)) {
    assert.strictEqual(clone.boxes[k].pmem && clone.boxes[k].pmem.bw && clone.boxes[k].pmem.bw.v, v, `mutated box ${k}`);
  }
  for (const [k, v] of Object.entries(beforeKappa)) {
    assert.strictEqual(clone.types[k].kappa && clone.types[k].kappa.v, v, `mutated type ${k}`);
  }
  const cal2 = FM.applyCal(clone);
  assert.ok(cal2 && cal2.boxes);
  for (const [k, row] of Object.entries(cal2.boxes)) {
    if (row.fit !== undefined) assert.strictEqual(clone.boxes[k].pmem.bw.v, row.fit);
  }
});

test('unlockState parses ISO instants with timezone offsets rather than raw strings', () => {
  const g = { firmware_unlock: { v: 'unlocked', since: '2026-11-10T00:00:00Z' } };
  assert.strictEqual(FM.unlockState(g, { at: '2026-11-10T01:00:00+02:00' }), 'stock');
  assert.strictEqual(FM.unlockState(g, { at: '2026-11-10T03:00:00+02:00' }), 'unlocked');
});

test('solveLambda handles non-finite shadow prices without null bsel crash', () => {
  const ctx = FM.prepare(
    FM.context(D, 'lassie', 'orn35', 'ik', D.knobs),
    D.models.orn35.heat.plan_default,
    D.models.orn35.heat.score_default,
  );
  for (const lam of [0, 1e30, Infinity, NaN]) {
    const res = FM.solveLambda(ctx, lam);
    assert.ok(res && res.out && res.out.length === ctx.groups.length);
  }
});

test('rooflineAttribute produces finite attribution terms on valid prefill input', () => {
  const fit = FM.rooflineFit(D, 'tqllm');
  assert.strictEqual(fit.status, 'FIT');
  const q = { box: 'snoopy', model: 'fnx', engine: 'tqllm', store: 'gsq_iq3s', unlock: 'stock', tokens: 1000 };
  const pred = FM.prefillRoof(D, q, fit.eta);
  assert.strictEqual(pred.status, 'OK', pred.why);
  const out = FM.rooflineAttribute(D, q, 100, fit.eta);
  assert.strictEqual(out.status, 'OK');
  assert.ok(Array.isArray(out.terms) && out.terms.length > 0);
  for (const t of out.terms) {
    assert.ok(t.k === null || Number.isFinite(t.k));
  }
});

test('bestPassing and advisor.all memoize klEquivKey across runs consistently', () => {
  const runs = ADV.ledger(path.join(__dirname, '..', 'runs', 'runs.jsonl'));
  assert.ok(runs.length > 0);
  const bp1 = ADV.bestPassing(D, runs);
  const allOut = ADV.all(D, runs);
  assert.ok(bp1.best.length > 0);
  assert.ok(allOut && allOut.upgrades && allOut.upgrades.boxes);
  const keys1 = runs.slice(0, 20).map(r => ADV.klEquivKey(r));
  const keys2 = runs.slice(0, 20).map(r => ADV.klEquivKey(r));
  assert.deepStrictEqual(keys1, keys2);
  const rq = ADV.runQuality(D, bp1.best[0].run, runs);
  assert.strictEqual(rq.status, 'PASS');
});

console.log(passed + ' tests passed');
