'use strict';
// the generic MoE path's planner side (src/model.js addModels, calibrate's host_borrow list): a harvested MoE model with
// no family joins data.models for one check (report.js --check --models), borrows on each box the median host of the
// listed families that were fitted there, and leaves every known family's calibration as it was.
const assert = require('assert'), fs = require('fs'), path = require('path');
const FM = require('../src/model.js');
const ROOT = path.join(__dirname, '..');
let passed = 0;
const test = (name, fn) => { fn(); passed++; console.log('ok ' + name); };
const load = () => JSON.parse(fs.readFileSync(path.join(ROOT, 'data.json'), 'utf8'));
const v = x => (x && typeof x === 'object' && 'v' in x) ? x.v : x;
const median = xs => { xs = xs.slice().sort((a, b) => a - b); const h = xs.length; return h % 2 ? xs[(h - 1) / 2] : (xs[h / 2 - 1] + xs[h / 2]) / 2; };

// a stand-in generic entry: the Ornith-35B family's shape under a new key, as tools/harvest.py moe_model builds one
function generic(data, key, borrow) {
  const m = Object.assign({}, data.models.orn35, { name: 'generic test', short: 'generic', show: false, generic: true, kld_family: key, host_borrow: borrow });
  delete m.draft;
  const kld = {};
  Object.keys(data.types).forEach(q => { const k = (data.types[q].kld || {}).orn35; if (k) kld[q] = k; });
  return { models: { [key]: m }, kld: { [key]: kld } };
}

test('a list borrow takes the median of the fitted hosts on each box; known families keep their calibration', () => {
  const base = load(); FM.applyCal(base);
  const data = load(), borrow = ['v41', 'v40', 'glm', 'orn122'];
  assert.deepStrictEqual(FM.addModels(data, generic(data, 'hx_test', borrow)), ['hx_test']);
  FM.applyCal(data);   // report.js --check: --models joins before the calibration
  Object.keys(data.boxes).forEach(bk => {
    const H = base.cal.boxes[bk].host, have = borrow.filter(k => H[k] !== undefined && v(H[k]) !== null);
    const fit = have.filter(k => H[k].tag !== 'ESTIMATE'), use = fit.length ? fit : have;
    const got = data.cal.boxes[bk].host.hx_test;
    assert(got && got.tag === 'ESTIMATE', bk + ': borrowed host is an ESTIMATE');
    assert.strictEqual(v(got), median(use.map(k => v(H[k]))), bk);
    assert(got.src.includes(fit.length ? 'fitted' : 'estimated'), got.src);
    Object.keys(base.models).forEach(mk => assert.deepStrictEqual(data.cal.boxes[bk].host[mk], H[mk], bk + ' ' + mk + ' host moved'));
    assert.deepStrictEqual(data.cal.boxes[bk].f, base.cal.boxes[bk].f, bk + ' CPU factor moved');
  });
  Object.keys(data.types).forEach(q => assert.ok(data.types[q].kld.hx_test, q + ' has a KLD entry (an analog or UNKNOWN)'));
});

test('a generic entry plans and times on the same solver as its family, and a known key is refused', () => {
  const data = load();
  FM.addModels(data, generic(data, 'hx_plan', ['orn35']));
  FM.applyCal(data);
  const r = { box: 'lassie', engine: 'ik', store: { split: false, types: ['q4_K'] }, rtr: false, draft: false, banks_gib: 0, threads: 64 };
  const a = FM.predictRecipe(data, Object.assign({}, r, { model: 'hx_plan' })), b = FM.predictRecipe(data, Object.assign({}, r, { model: 'orn35' }));
  assert.strictEqual(a.status, 'PREDICTED', a.why);
  // the same shape and heat, and a list borrow of the family alone takes its run-ledger host (6.9 ms on lassie, fitted
  // after calibrate), not the 15 ms estimate calibrate has before that fit: the same decode
  assert(Math.abs(a.decode_tps / b.decode_tps - 1) < 1e-9, a.decode_tps + ' vs ' + b.decode_tps);
  assert(/fitted host overheads of orn35/.test(data.cal.boxes.lassie.host.hx_plan.src), data.cal.boxes.lassie.host.hx_plan.src);
  assert.throws(() => FM.addModels(data, { models: { orn35: {} } }), /already a planner model/);
  assert.deepStrictEqual(FM.addModels(data, {}), []);
});

console.log(passed + ' passed');
