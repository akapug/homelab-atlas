// The run ledger on disk (node): sealed plain segments runs/runs.NNNN.jsonl (the older ones runs.NNNN.jsonl.gz)
// (oldest first), then the open file
// runs/runs.jsonl. Same layout and order as src/ledger.py, which also packs it.
'use strict';
const fs = require('fs'), path = require('path'), zlib = require('zlib');

const base = file => file.endsWith('.jsonl') ? file.slice(0, -'.jsonl'.length) : file;

function segments(file) {
  const dir = path.dirname(file), stem = path.basename(base(file));
  const rx = new RegExp('^' + stem.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\.\\d{4}\\.jsonl(\\.gz)?$');
  return fs.existsSync(dir) ? fs.readdirSync(dir).filter(f => rx.test(f)).sort().map(f => path.join(dir, f)) : [];
}

const files = file => segments(file).concat(fs.existsSync(file) ? [file] : []);

function text(f) {
  const b = fs.readFileSync(f);
  return (f.endsWith('.gz') ? zlib.gunzipSync(b) : b).toString('utf8');
}

// beside the ledger: dated caveats and corrections per run id (src/ledger.py overlay; the rows stay as recorded)
const CAVEATS = 'run_caveats.json';
function caveats(file) {
  const p = path.join(path.dirname(file), CAVEATS);
  if (!fs.existsSync(p)) return {};
  const cv = JSON.parse(fs.readFileSync(p, 'utf8'));
  delete cv._doc;
  return cv;
}
function overlay(row, cv) {
  const c = cv[row.id];
  if (!c) return row;
  const m = Object.assign({}, row.measured), src = Object.assign({}, m.src);
  Object.keys(c.correct || {}).forEach(k => { m[k] = c.correct[k].v; src[k] = c.correct[k].src; });
  if (c.kind) {
    m.caveats = (m.caveats || []).concat([['kind', 'why', 'superseded_by'].reduce((a, k) => (k in c ? Object.assign(a, { [k]: c[k] }) : a), {})]);
    src.caveats = c.src;
  }
  return Object.assign({}, row, { measured: Object.assign(m, { src }) });
}
// the ledger's rows in order; unless raw, with the caveats file beside it applied
function rows(file, raw) {
  const out = files(file).flatMap(f => text(f).split('\n').filter(l => l.trim()).map(l => JSON.parse(l)));
  const cv = raw ? {} : caveats(file);
  return Object.keys(cv).length ? out.map(r => overlay(r, cv)) : out;
}
// the caveat kinds that keep a run out of best-run picks, fits and speed scores
const caveated = row => ((row.measured || {}).caveats || []).filter(c => c.kind === 'confounded').map(c => c.kind);

module.exports = { segments, files, rows, overlay, caveats, caveated };
