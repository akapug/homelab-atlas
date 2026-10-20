// The run ledger on disk (node): sealed segments runs/runs.NNNN.jsonl.gz, or plain runs.NNNN.jsonl in a public copy
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

const rows = file => files(file).flatMap(f => text(f).split('\n').filter(l => l.trim()).map(l => JSON.parse(l)));

module.exports = { segments, files, rows };
