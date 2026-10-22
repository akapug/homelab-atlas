"""The run ledger on disk: sealed segments plus one open file that runs are appended to.

A ledger named runs/runs.jsonl is read as runs/runs.0001.jsonl, runs/runs.0002.jsonl, ... (oldest first), then
runs/runs.jsonl. pack() seals plain-text segments, each at most MAX_OPEN_BYTES: no archive enters git (the never-track
guard refuses one), so the ledger stays diffable here and in a public copy. The gzip segments sealed before 2026-10-21
(runs.0001.jsonl.gz ... runs.0008.jsonl.gz) are still read in their place; history is never rewritten. Writers append to the open file only; after each append tools/record_run.py calls pack(), which seals
the rows older than KEEP_DAYS into the next segment once the open file passes MAX_OPEN_BYTES, so a checkout never
holds a ledger file over GIT_FILE_LIMIT. Sealed segments are never rewritten. src/ledger.js reads the same layout for
node.
  python3 src/ledger.py pack [LEDGER] [--max-bytes N] [--keep-days D] [--all]           seal rows (see pack())
  python3 src/ledger.py count [LEDGER]                                         rows per file
"""
import datetime
import glob
import gzip
import itertools
import json
import os
import re
import sys

GIT_FILE_LIMIT = 1 << 20          # no tracked file is larger than this (1 MiB)
MAX_OPEN_BYTES = 512 << 10        # pack seals rows once the open file passes this
KEEP_DAYS = 1                     # rows younger than this stay in the open file (they may still be corrected)
SEGMENT_DIGITS = 4


def base(path):
    return path[:-len('.jsonl')] if path.endswith('.jsonl') else path


def segments(path):
    rx = re.compile(re.escape(os.path.basename(base(path))) + r'\.\d{%d}\.jsonl(?:\.gz)?$' % SEGMENT_DIGITS)
    return sorted(p for p in glob.glob(glob.escape(base(path)) + '.*.jsonl*') if rx.match(os.path.basename(p)))


def files(path):
    """the ledger's files in read order: segments oldest first, then the open file when it exists"""
    return segments(path) + ([path] if os.path.exists(path) else [])


def exists(path):
    return bool(files(path))


def lines(path):
    for f in files(path):
        with (gzip.open(f, 'rt', encoding='utf-8') if f.endswith('.gz') else open(f, encoding='utf-8')) as h:
            yield from (x for x in h if x.strip())


CAVEATS = 'run_caveats.json'   # beside the ledger: dated caveats and corrections per run id (the rows stay as recorded)


def caveats(path):
    p = os.path.join(os.path.dirname(path), CAVEATS)
    return {k: v for k, v in json.load(open(p, encoding='utf-8')).items() if k != '_doc'} if os.path.exists(p) else {}


def overlay(row, cv):
    """the row as read: cv[id].correct sets measured fields (value + src) that the recorder got wrong; cv[id].kind adds
    measured.caveats [{kind, why, superseded_by}] (src in measured.src.caveats). A 'confounded' run is no clean measure of its
    recipe: no best-run pick, fit or speed score takes it; it stays in every run list with its caveat"""
    c = cv.get(row.get('id'))
    if not c:
        return row
    m = dict(row.get('measured') or {})
    src = dict(m.get('src') or {})
    for k, x in (c.get('correct') or {}).items():
        m[k], src[k] = x['v'], x['src']
    if c.get('kind'):
        m['caveats'] = list(m.get('caveats') or []) + [{k: c[k] for k in ('kind', 'why', 'superseded_by') if k in c}]
        src['caveats'] = c['src']
    return dict(row, measured=dict(m, src=src))


def rows(path, raw=False):
    """the ledger's rows in order; unless raw, with the caveats file beside it applied (overlay)"""
    out = [json.loads(x) for x in lines(path)]
    cv = {} if raw else caveats(path)
    return [overlay(r, cv) for r in out] if cv else out


def caveated(row):
    """the caveat kinds that keep a run out of best-run picks, fits and speed scores"""
    return [c['kind'] for c in (row.get('measured') or {}).get('caveats') or [] if c.get('kind') == 'confounded']


def when(row):
    """the row's run time as an aware datetime, None when it has none"""
    s = str(row.get('when') or '')
    m = re.match(r'(\d{4}-\d\d-\d\d)(?:T(\d\d):(\d\d)(?::(\d\d))?)?', s)
    if not m:
        return None
    d = datetime.datetime.strptime(m.group(1), '%Y-%m-%d').replace(tzinfo=datetime.timezone.utc)
    return d + datetime.timedelta(hours=int(m.group(2) or 0), minutes=int(m.group(3) or 0), seconds=int(m.group(4) or 0))


def chunks(lines, cap):
    """consecutive groups of lines, each at most cap bytes (a single longer line is a group of its own)"""
    out, n = [[]], 0
    for x in lines:
        if out[-1] and n + len(x) > cap:
            out.append([])
            n = 0
        out[-1].append(x)
        n += len(x)
    return out


def pack(path, max_bytes=None, keep_days=None, everything=False, now=None, limit=None):
    """Seal the open file's oldest rows into the next segment. Without everything: when the open file is larger than
    max_bytes, seal only the leading rows older than keep_days (append order is kept, so packing stops at the first younger
    row). Independently, if the rows left would still pass limit (GIT_FILE_LIMIT), the fewest further leading rows are sealed,
    young ones too, so the open file always fits in git. Sealed rows go into plain-text segments of at most MAX_OPEN_BYTES each, as many as the rows need. Returns
    (rows sealed, the last segment written or None)."""
    max_bytes = MAX_OPEN_BYTES if max_bytes is None else max_bytes
    keep_days = KEEP_DAYS if keep_days is None else keep_days
    limit = GIT_FILE_LIMIT if limit is None else limit
    if not os.path.exists(path) or (not everything and os.path.getsize(path) <= min(max_bytes, limit)):
        return 0, None
    with open(path, encoding='utf-8') as h:
        raw = [x for x in h if x.strip()]
    cut = (now or datetime.datetime.now(datetime.timezone.utc)) - datetime.timedelta(days=keep_days)
    old = lambda x: (when(json.loads(x)) or cut) < cut   # a row without a time stays open
    k = len(raw) if everything else next((i for i, x in enumerate(raw) if not old(x)), len(raw))
    left = list(itertools.accumulate((len(x.encode('utf-8')) for x in reversed(raw)), initial=0))[::-1]   # bytes from row i on
    k = max(k, next(i for i, n in enumerate(left) if n <= limit))
    if not k:
        return 0, None
    sealed = [(x if x.endswith('\n') else x + '\n').encode('utf-8') for x in raw[:k]]
    bodies = [b''.join(g) for g in chunks(sealed, MAX_OPEN_BYTES)]
    for body in bodies:
        if len(body) > GIT_FILE_LIMIT:
            raise SystemExit('ledger pack: a row of %d bytes is over %d' % (len(body), GIT_FILE_LIMIT))
    first = len(segments(path)) + 1
    for i, body in enumerate(bodies):
        seg = '%s.%0*d.jsonl' % (base(path), SEGMENT_DIGITS, first + i)
        with open(seg, 'wb') as h:
            h.write(body)
    tmp = path + '.pack-tmp'
    with open(tmp, 'w', encoding='utf-8') as h:
        h.writelines(raw[k:])
    os.replace(tmp, path)
    return k, seg


def main(argv):
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('cmd', choices=('pack', 'count'))
    ap.add_argument('ledger', nargs='?', default=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'runs', 'runs.jsonl'))
    ap.add_argument('--max-bytes', type=int, default=MAX_OPEN_BYTES)
    ap.add_argument('--keep-days', type=float, default=KEEP_DAYS)
    ap.add_argument('--all', action='store_true', help='seal every row now (a published snapshot)')
    a = ap.parse_args(argv)
    if a.cmd == 'pack':
        k, seg = pack(a.ledger, a.max_bytes, a.keep_days, a.all)
        print('ledger pack: %s' % ('sealed %d rows into %s' % (k, seg) if k else 'nothing to seal'))
    for f in files(a.ledger):
        print('%s: %d bytes' % (f, os.path.getsize(f)))
    print('rows: %d' % len(rows(a.ledger)))


if __name__ == '__main__':
    main(sys.argv[1:])
