"""Merge probed / intake JSON (data/boxes/*.json, data/models/*.json) over the hand-typed tables in build.py.

Rules (the same for boxes and models):
  * A tagged value {"v", "tag", "src"} replaces the hand value of the same key when its tag ranks at least as high
    (MEASURED > READ > FIT > anything else; UNKNOWN never replaces). The replaced hand value is kept under "hand".
  * A record (a dict with "tag" but no "v", e.g. a NUMA node or a mount) carries raw fields; they replace the hand
    record's fields under the same rank rule, except names/labels. Replaced fields are kept under record["hand"].
  * A plain value (raw int/str) only fills a key the hand table lacks; a different hand value is a conflict, logged
    and left alone (structural values like a model's layer count must agree with the hand data built on them).
  * A hand None (a tier deliberately absent, e.g. no PMem) is never replaced by a dict; the probe goes to "<key>_probed".
  * Lists: GPUs match by PCI address (hand "pci" key or the short address in the hand name), then in order; DRAM NUMA
    nodes by position; disks by mount. Unmatched probed GPUs go to "gpus_unlisted". Other lists only fill.
Everything replaced, added or in conflict is logged; build.py stores the log in the entry's "overlay" key.
"""
import copy
import json
import os
from pathlib import Path

RANK = {'MEASURED': 3, 'READ': 2, 'FIT': 1}
KEEP = {'name', 'label', 'id', 'desc', 'what', 'note'}


def is_n(x):
    return isinstance(x, dict) and 'tag' in x and 'v' in x


def rank(tag):
    return -1 if tag in (None, 'UNKNOWN') else RANK.get(str(tag).split()[0], 0)


def load_dir(path):
    out = {}
    if os.path.isdir(path):
        for f in sorted(os.listdir(path)):
            if f.endswith('.json'):
                out[f[:-5]] = json.loads(Path(os.path.join(path, f)).read_text())
    return out


def _short(x):
    s = json.dumps(x.get('v') if is_n(x) else x, default=str)
    return s if len(s) <= 80 else s[:77] + '...'


def _gpu_match(hand, probed):
    out, used = {}, set()
    pool = [g for g in probed if g.get('discrete') is not False]
    for i, h in enumerate(hand):
        for g in pool:
            short = g.get('pci', '').split(':', 1)[-1]
            if g.get('pci') not in used and (h.get('pci') == g.get('pci') or (short and short in str(h.get('name', '')).lower())):
                out[i] = g
                used.add(g.get('pci'))
                break
    rest = [g for g in pool if g.get('pci') not in used]
    for i in range(len(hand)):
        if i not in out and rest:
            g = rest.pop(0)
            out[i] = g
            used.add(g.get('pci'))
    return out, [g for g in pool if g.get('pci') not in used]


def merge(hand, over, path='', log=None):
    """merge `over` into a deep copy of `hand` -> (merged, log)"""
    log = [] if log is None else log
    h = copy.deepcopy(hand)
    _merge_dict(h, over, path, log)
    return h, log


def _merge_dict(h, o, path, log):
    rec_rank = rank(o.get('tag')) if ('tag' in o and 'v' not in o) else None
    for k, ov in o.items():
        p = (path + '.' + k) if path else k
        if rec_rank is not None and k in ('tag', 'src'):
            continue  # a record's provenance moves with its fields (below)
        if k not in h:
            h[k] = copy.deepcopy(ov)
            log.append(('add', p, None, _short(ov)))
            continue
        hv = h[k]
        if is_n(ov):
            if rank(ov['tag']) < 0:
                continue
            if is_n(hv) and rank(ov['tag']) < rank(hv['tag']):
                log.append(('kept', p, _short(hv), _short(ov) + ' (%s < hand %s)' % (ov['tag'], hv['tag'])))
                continue
            if is_n(hv) and hv.get('v') == ov.get('v'):
                continue
            h[k] = dict(ov, hand=hv)
            log.append(('replace', p, _short(hv), _short(ov)))
        elif isinstance(ov, dict):
            if hv is None:
                h[k + '_probed'] = copy.deepcopy(ov)
                log.append(('conflict', p, 'None (hand: tier absent)', 'probe found one -> ' + k + '_probed'))
            elif isinstance(hv, dict):
                _merge_dict(hv, ov, p, log)
        elif isinstance(ov, list):
            if k == 'gpus' and isinstance(hv, list):
                mt, rest = _gpu_match(hv, ov)
                for i, g in mt.items():
                    _merge_dict(hv[i], g, '%s[%s]' % (p, hv[i].get('id', i)), log)
                if rest:
                    h['gpus_unlisted'] = rest
                    log.append(('conflict', p, '%d hand GPUs' % len(hv), '%d probed GPUs unmatched -> gpus_unlisted' % len(rest)))
            elif k == 'nodes' and isinstance(hv, list):
                for i, (a, b) in enumerate(zip(hv, ov)):
                    _merge_dict(a, b, '%s[%d]' % (p, i), log)
                if len(hv) != len(ov):
                    log.append(('conflict', p, '%d nodes' % len(hv), '%d nodes' % len(ov)))
            elif k == 'disks' and isinstance(hv, list):
                by = {d.get('mount'): d for d in hv}
                for d in ov:
                    if d.get('mount') in by:
                        _merge_dict(by[d['mount']], d, '%s[%s]' % (p, d['mount']), log)
                    else:
                        hv.append(copy.deepcopy(d))
                        log.append(('add', '%s[%s]' % (p, d.get('mount')), None, d.get('kind')))
            elif hv != ov:
                log.append(('conflict', p, _short(hv), _short(ov)))
        else:
            if hv == ov:
                continue
            if rec_rank is not None and k not in KEEP and not isinstance(hv, dict) and rec_rank >= rank(h.get('tag')):
                h.setdefault('hand', {})[k] = hv
                h[k] = ov
                log.append(('replace', p, _short(hv), _short(ov)))
            elif k not in KEEP and not k.endswith('_src'):
                log.append(('conflict', p, _short(hv), _short(ov)))
    if rec_rank is not None:
        for k in ('tag', 'src'):
            if k in o and h.get(k) != o[k]:
                if k in h:
                    h.setdefault('hand', {})[k] = h[k]
                h[k] = o[k]


def apply(table, over_dir, kind):
    """merge every <over_dir>/<key>.json into table[key]; -> (skipped keys with no hand entry, {key: log})"""
    logs, skipped = {}, []
    for key, over in load_dir(over_dir).items():
        if key not in table:
            skipped.append(key)
            continue
        table[key], log = merge(table[key], over)
        table[key]['overlay'] = {'file': os.path.join(os.path.basename(os.path.dirname(over_dir.rstrip('/'))), os.path.basename(over_dir.rstrip('/')), key + '.json'),
                                 'kind': kind, 'changes': [list(x) for x in log]}
        logs[key] = log
    return skipped, logs
