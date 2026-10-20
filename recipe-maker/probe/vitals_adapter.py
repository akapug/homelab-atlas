"""Optional adapter: cross-check a probed box against a fleet "vitals" sidecar, when one exists and is fresh.

Some fleets already run a collector that writes per-node disk inventories (here: the homelab vitals.json contract,
schema 1: nodes[].disks[] with mount, kind in {NVMe, SSD, HDD, USB HDD, USB SSD, image}, model). When that file is
present and younger than MAX_AGE_S, every mount both sides list is compared; a disagreement is reported, never
silently resolved (the probe's own reading stays in the box). With no file, or a stale one, this does nothing:
probe_box.py is complete on its own.

Default path: $FITVIZ_VITALS, else ~/.cache/pack-vitals/vitals.json.
"""
import datetime
import json
import os
from pathlib import Path

MAX_AGE_S = 600  # the contract: older than 10 min = stale
KIND = {'NVMe': 'nvme', 'SSD': 'sata-ssd', 'HDD': 'hdd', 'USB HDD': 'hdd', 'USB SSD': 'usb-ssd'}


def default_path():
    return os.environ.get('FITVIZ_VITALS') or os.path.expanduser('~/.cache/pack-vitals/vitals.json')


def load(path=None, now=None):
    path = path or default_path()
    if not os.path.exists(path):
        return None, 'no vitals file at ' + path
    try:
        d = json.loads(Path(path).read_text())
    except (OSError, ValueError) as ex:
        return None, 'vitals file unreadable: %s' % ex
    gen = d.get('generated')
    try:
        t = datetime.datetime.strptime(gen, '%Y-%m-%dT%H:%M:%SZ').replace(tzinfo=datetime.timezone.utc)
    except (TypeError, ValueError):
        return None, 'vitals file has no generated time'
    age = ((now or datetime.datetime.now(datetime.timezone.utc)) - t).total_seconds()
    if age > MAX_AGE_S:
        return None, 'vitals file stale (%.0f s old)' % age
    return d, 'vitals %s (%.0f s old)' % (gen, age)


def cross_check(box, host, path=None, now=None):
    d, why = load(path, now)
    if d is None:
        return None
    node = next((x for x in d.get('nodes') or [] if host in (x.get('host'), x.get('ssh_host')) or host in (x.get('aliases') or [])), None)
    if not node:
        return {'source': why, 'note': 'host %s not in the vitals file' % host}
    theirs = {x.get('mount'): x for x in node.get('disks') or []}
    agree, disagree = [], []
    for dk in box.get('disks') or []:
        t = theirs.get(dk['mount'])
        if not t or not t.get('kind'):
            continue
        tk = 'image' if t['kind'] == 'image' else KIND.get(t['kind'], t['kind'])
        if tk == 'image' or tk == dk['kind']:
            agree.append(dk['mount'])
        else:
            disagree.append({'mount': dk['mount'], 'probe': dk['kind'], 'vitals': t['kind'], 'vitals_model': t.get('model')})
    return {'source': why, 'agree': agree, 'disagree': disagree,
            'note': 'vitals classifies by the top disk only; the probe also resolves btrfs members and loop-image backing files'}
