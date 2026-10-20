#!/usr/bin/env python3
"""Read-only hardware probe of a Linux box for the fit planner.

    probe_box.py [--ssh HOST] [--name NAME] [--write data/boxes/NAME.json] [--raw-out FILE] [--from-raw FILE]
                 [--diff data.json] [--vitals PATH]

Runs probe/collect.sh locally, or as ONE `ssh HOST bash -s` call (nothing is installed or written on the target), and
prints a box entry in the shape build.py's `boxes['<name>']` entries use. Every probed number is tagged
{"v": value, "tag": "MEASURED", "src": "probe <host> <UTC time>"}; anything the probe cannot read is
{"v": null, "tag": "UNKNOWN", "src": "<reason>"} and is listed under probe.unknown. Nothing is guessed.

What it reads: CPU model, sockets, cores, threads (online and allowed for a user process), ISA flags; NUMA nodes
(CPUs, MemTotal, MemFree); memory-only NUMA nodes as PMem/CXL tier candidates (Optane identified from the NVDIMM
bus in sysfs, no root needed); DIMM type/speed/count (`sudo -n dmidecode`, else UNKNOWN); GPUs (nvidia-smi, xpu-smi,
amdgpu sysfs; PCIe link at the root port); mounted filesystems with the kind of disk beneath (btrfs members, dm/md
parents and loop-image backing files resolved; a mount is rotational if ANY member is); swap, swappiness,
numa_balancing, NUMA demotion, THP.

Fields a probe cannot measure (fitted bandwidths, usable budgets, notes) are never emitted: the hand table keeps them.
--diff compares the probe with the boxes of a built data.json and prints every disagreement.
"""
import argparse
import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
COLLECT = os.path.join(HERE, 'collect.sh')
M, U, E = 'MEASURED', 'UNKNOWN', 'ESTIMATE'
GiB = 2 ** 30
KiB = 1024

# ISA extensions that change which CPU expert kernels a box can run
ISA_FLAGS = ('avx', 'avx2', 'fma', 'f16c', 'avx512f', 'avx512bw', 'avx512vl', 'avx512_vnni', 'avx512_bf16', 'avx512_fp16',
             'avx_vnni', 'amx_tile', 'amx_int8', 'amx_bf16')
# mounts never worth listing for model placement
SKIP_MOUNT = re.compile(r'^/(proc|sys|run|snap|boot)(/|$)|^/dev(/(?!shm$)|$)|^/var/lib/(docker|containers)/')
NETWORK_FS = {'nfs', 'nfs4', 'cifs', 'smb3', '9p', 'fuse.sshfs', 'ceph', 'glusterfs', 'fuse.rclone'}
PCI_VENDOR = {'0x10de': 'nvidia', '0x8086': 'intel', '0x1002': 'amd'}
NFIT_VENDOR = {'0x8089': 'Intel'}


def n(v, tag, src):
    return {'v': v, 'tag': tag, 'src': src}


def r2(x):
    return round(x, 2)


# ---------------------------------------------------------------- collection
def collect(host=None, timeout=120):
    script = Path(COLLECT).read_text()
    cmd = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=10', host, 'bash', '-s'] if host else ['bash', '-s']
    p = subprocess.run(cmd, input=script, capture_output=True, text=True, timeout=timeout)
    if '@@meta' not in p.stdout:
        raise SystemExit('probe failed on %s (rc %d): %s' % (host or 'localhost', p.returncode, p.stderr.strip()[:400]))
    return p.stdout


def sections(text):
    """'@@name' starts a section; '@@!name reason' marks it unreadable. '@@xpu_dev 1' keeps its argument."""
    out, err, key = {}, {}, None
    for line in text.splitlines():
        if line.startswith('@@!'):
            name, _, reason = line[3:].partition(' ')
            err[name] = reason.strip()
            continue
        if line.startswith('@@'):
            key = line[2:].strip()
            out.setdefault(key, [])
            continue
        if key is not None:
            out[key].append(line)
    return out, err


def kv_line(line):
    """'a=1 b=two c=' -> dict (values never contain spaces: collect.sh replaces them)."""
    return {k: v for k, v in re.findall(r'(\w+)=(\S*)', line)}


def cpulist_count(s):
    total = 0
    for part in (s or '').split(','):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition('-')
        total += (int(b) - int(a) + 1) if b else 1
    return total


# ---------------------------------------------------------------- parsers (each takes the section's lines)
def parse_cpu(lines):
    procs, cur, model, flags, online, allowed = [], {}, None, set(), None, None
    for line in lines:
        k, _, v = line.partition(':')
        k, v = k.strip(), v.strip()
        if k == 'processor':
            cur = {'id': int(v)}
            procs.append(cur)
        elif k == 'physical id':
            cur['pkg'] = int(v)
        elif k == 'core id':
            cur['core'] = int(v)
        elif k == 'model name':
            model = v
        elif k == 'flags':
            flags = set(v.split())
        elif k == 'online':
            online = v
        elif k == 'Cpus_allowed_list':
            allowed = v
    pkgs = {p.get('pkg', 0) for p in procs}
    cores = {(p.get('pkg', 0), p.get('core', p['id'])) for p in procs}
    return {'model': model, 'sockets': len(pkgs) or None, 'phys': len(cores) or None, 'threads_online': cpulist_count(online) if online else len(procs),
            'threads_allowed': cpulist_count(allowed) if allowed else None, 'allowed_list': allowed,
            'isa': {f: (f in flags) for f in ISA_FLAGS} if flags else None}


def parse_nodes(lines):
    nodes, cur = [], None
    for line in lines:
        if line.startswith('node='):
            d = kv_line(line)
            cur = {'node': int(d['node']), 'cpus': d.get('cpus', ''), 'total_kb': None, 'free_kb': None}
            nodes.append(cur)
        elif cur is not None:
            m = re.search(r'(MemTotal|MemFree):\s+(\d+) kB', line)
            if m:
                cur['total_kb' if m.group(1) == 'MemTotal' else 'free_kb'] = int(m.group(2))
    return nodes


def parse_meminfo(lines):
    return {m.group(1): int(m.group(2)) for m in (re.match(r'(\w+):\s+(\d+) kB', l) for l in lines) if m}


def parse_sysctl(lines):
    out = {}
    for line in lines:
        k, _, v = line.partition('=')
        out[k] = None if v.startswith('!unreadable') else v.strip()
    return out


def thp_mode(s):
    m = re.search(r'\[(\w[\w+]*)\]', s or '')
    return m.group(1) if m else s


def parse_swaps(lines):
    out = []
    for line in lines[1:]:
        f = line.split()
        if len(f) >= 5:
            out.append({'device': f[0], 'type': f[1], 'size_gib': r2(int(f[2]) * KiB / GiB), 'used_gib': r2(int(f[3]) * KiB / GiB), 'priority': int(f[4])})
    return out


def parse_dax(lines):
    return [kv_line(l) for l in lines if l.startswith('dev=')]


def parse_nd(lines):
    regions = [kv_line(l) for l in lines if l.startswith('region=')]
    nmems = [kv_line(l) for l in lines if l.startswith('nmem=')]
    cxl = [l.split('=', 1)[1].split() for l in lines if l.startswith('cxl_devices=')]
    return {'regions': regions, 'nmems': nmems, 'cxl': cxl[0] if cxl else []}


def parse_dmidecode(lines):
    """Memory Device records -> populated DIMMs (DRAM and persistent memory kept apart)."""
    devs, cur = [], None
    for line in lines:
        if line.startswith('Memory Device'):
            cur = {}
            devs.append(cur)
        elif line.startswith('Handle'):
            cur = None
        elif cur is not None and ':' in line:
            k, _, v = line.strip().partition(':')
            cur[k.strip()] = v.strip()
    dram, pmem = [], []
    for d in devs:
        size = d.get('Size', '')
        m = re.match(r'(\d+)\s*(MB|GB|TB)', size)
        if not m:
            continue  # "No Module Installed"
        gib = int(m.group(1)) * {'MB': 1 / 1024, 'GB': 1, 'TB': 1024}[m.group(2)]
        speed = re.match(r'(\d+)', d.get('Configured Memory Speed', '') or d.get('Configured Clock Speed', ''))
        rated = re.match(r'(\d+)', d.get('Speed', ''))
        rec = {'gib': gib, 'type': d.get('Type'), 'locator': d.get('Locator'), 'bank': d.get('Bank Locator'),
               'mts': int(speed.group(1)) if speed else None, 'rated_mts': int(rated.group(1)) if rated else None}
        tech = d.get('Memory Technology', '')
        if 'non-volatile' in (d.get('Type') or '').lower() or 'persistent' in tech.lower() or 'optane' in tech.lower():
            pmem.append(rec)
        else:
            dram.append(rec)
    return {'slots': len(devs), 'dram': dram, 'pmem': pmem}


def parse_nvidia(lines):
    out = []
    for line in lines:
        f = [x.strip() for x in line.split(',')]
        if len(f) < 10 or not f[0].isdigit():
            continue
        num = lambda x: int(x) if x.isdigit() else None
        bus = f[2].lower()
        bus = bus[4:] if re.match(r'^[0-9a-f]{8}:', bus) else bus  # 00000000:2d:00.0 -> 0000:2d:00.0
        out.append({'index': int(f[0]), 'name': f[1], 'pci': bus, 'total_mib': num(f[3]), 'used_mib': num(f[4]), 'free_mib': num(f[5]),
                    'gen': num(f[6]), 'max_gen': num(f[7]), 'width': num(f[8]), 'max_width': num(f[9]), 'driver': f[10] if len(f) > 10 else None})
    return out


def parse_xpu(secs):
    out = []
    for key, lines in secs.items():
        if not key.startswith('xpu_dev '):
            continue
        try:
            d = json.loads('\n'.join(lines))
        except ValueError:
            continue
        out.append({'index': int(key.split()[1]), 'name': d.get('device_name'), 'pci': (d.get('pci_bdf_address') or '').lower(),
                    'total_bytes': d.get('memory_physical_size_byte'), 'free_bytes': int(d['memory_free_size_byte']) if str(d.get('memory_free_size_byte', '')).isdigit() else None,
                    'driver': d.get('driver_version'), 'device_type': d.get('device_type')})
    return out


def link_gen(speed):
    """'16.0_GT/s_PCIe' -> PCIe generation."""
    m = re.match(r'([\d.]+)', speed or '')
    if not m:
        return None
    return {2.5: 1, 5.0: 2, 8.0: 3, 16.0: 4, 32.0: 5, 64.0: 6}.get(float(m.group(1)))


def parse_drm(lines):
    out = []
    for line in lines:
        if not line.startswith('card='):
            continue
        d = kv_line(line)
        if not d.get('class', '').startswith('0x03'):
            continue
        path = d.get('path', '')
        # devices between the root port and the GPU: >1 means an on-board switch (its endpoint link is virtual)
        m = re.search(r'/pci[0-9a-f]+:[0-9a-f]+/(.*)$', path)
        depth = len(m.group(1).split('/')) - 1 if m else None
        num = lambda k: int(d[k]) if d.get(k, '').isdigit() else None
        out.append({'card': d['card'], 'pci': d.get('pci', '').lower(), 'vendor': PCI_VENDOR.get(d.get('vendor'), d.get('vendor')), 'device': d.get('device'),
                    'driver': d.get('driver') or None, 'boot_vga': d.get('boot_vga') == '1', 'vram_total': num('vram_total'), 'vram_used': num('vram_used'),
                    'vram_vendor': d.get('vram_vendor') or None,
                    'behind_switch': depth is not None and depth > 1,
                    'ep': {'gen': link_gen(d.get('ep_speed')), 'width': num('ep_width'), 'max_gen': link_gen(d.get('ep_max_speed')), 'max_width': num('ep_max_width')},
                    'rp': {'gen': link_gen(d.get('rp_speed')), 'width': num('rp_width'), 'max_gen': link_gen(d.get('rp_max_speed')), 'max_width': num('rp_max_width')}})
    return out


def unescape_findmnt(s):
    return re.sub(r'\\x([0-9a-fA-F]{2})', lambda m: chr(int(m.group(1), 16)), s)


def parse_mounts(lines):
    out = []
    for line in lines:
        f = line.split(' ')
        if len(f) < 5:
            continue
        tgt, src, fs = unescape_findmnt(f[0]), unescape_findmnt(f[1]), f[2]
        size = int(f[3]) if f[3].isdigit() else None
        avail = int(f[4]) if f[4].isdigit() else None
        out.append({'mount': tgt, 'source': src, 'fstype': fs, 'size': size, 'avail': avail, 'uuid': f[5] if len(f) > 5 and f[5] else None})
    return out


def parse_lsblk(lines):
    try:
        tree = json.loads('\n'.join(lines)).get('blockdevices') or []
    except ValueError:
        return {}
    rota = lambda x: str(x).lower() in ('1', 'true')
    devs = {}  # kname -> {type, tops: {top kname}, path}; a dm/md under several disks collects all of them

    def walk(dev, top):
        k = dev.get('kname') or dev.get('name')
        e = devs.setdefault(k, {'kname': k, 'path': dev.get('path') or '/dev/' + k, 'type': dev.get('type'), 'tops': {}})
        e['tops'][top.get('kname') or top.get('name')] = {'rota': rota(top.get('rota')), 'tran': top.get('tran'), 'model': (top.get('model') or '').strip(),
                                                         'type': top.get('type')}
        for kid in dev.get('children') or []:
            walk(kid, top)
    for dev in tree:
        walk(dev, dev)
    return devs


def parse_btrfs(lines):
    return {m.group(1): m.group(2).split() for m in (re.match(r'uuid=(\S+) members=(.*)$', l) for l in lines) if m}


def parse_loop(lines):
    return {m.group(1): m.group(2) for m in (re.match(r'dev=(\S+) backing=(.*)$', l) for l in lines) if m}


# ---------------------------------------------------------------- disks: kind per mount
def disk_kind(tops):
    """Kind of the storage beneath a filesystem from its top-level disks: rotational if ANY member is (a btrfs RAID or
    a dm/md set is as slow as its slowest member for random reads). Vocabulary: nvme, sata-ssd, usb-ssd, hdd."""
    if not tops:
        return None
    if any(t['rota'] for t in tops.values()):
        return 'hdd'
    if all(t.get('tran') == 'nvme' or k.startswith('nvme') for k, t in tops.items()):
        return 'nvme'
    if any(t.get('tran') == 'usb' for t in tops.values()):
        return 'usb-ssd'
    return 'sata-ssd'


def mount_disks(mounts, devs, btrfs, loops):
    by_path = {d['path']: d for d in devs.values()}
    by_mount = {m['mount']: m for m in mounts}

    def kname_of(source):
        src = source.split('[')[0]
        if src in by_path:
            return by_path[src]['kname']
        real = os.path.basename(src)
        return real if real in devs else None

    def resolve(m, depth=0):
        """-> (kind, members {kname: top-info}, how)"""
        fs, src = m['fstype'], m['source']
        if fs in NETWORK_FS or re.match(r'^[^/]+:/', src) or src.startswith('//'):
            return 'network', {}, 'network filesystem ' + fs
        if fs == 'tmpfs':
            return 'memory', {}, 'tmpfs (RAM)'
        k = kname_of(src)
        if k is None:
            return None, {}, 'source ' + src + ' is not a block device lsblk lists'
        members = [k]
        if fs == 'btrfs' and m.get('uuid') in btrfs:
            members = btrfs[m['uuid']]
        tops, how = {}, []
        for mk in members:
            d = devs.get(mk)
            if not d:
                how.append(mk + ' not in lsblk')
                continue
            if d['type'] == 'loop' or mk.startswith('loop'):
                back = loops.get(mk)
                if not back or depth > 3:
                    return None, {}, 'loop device ' + mk + ' with no readable backing file'
                host = max((x for x in mounts if back == x['mount'] or back.startswith(x['mount'].rstrip('/') + '/')), key=lambda x: len(x['mount']), default=None)
                if not host:
                    return None, {}, 'loop image ' + back + ' on no listed mount'
                kind, t2, how2 = resolve(host, depth + 1)
                return kind, t2, 'image file ' + back + ' on ' + host['mount'] + ' (' + how2 + ')'
            tops.update(d['tops'])
        kinds = ', '.join('%s %s%s%s' % (k2, 'rotational' if t['rota'] else 'non-rotational', ' ' + t['tran'] if t.get('tran') else '',
                                          ' "' + t['model'] + '"' if t.get('model') else '') for k2, t in sorted(tops.items()))
        what = ('btrfs members ' + ' '.join(members) + ' -> ' if len(members) > 1 else '') + kinds
        return disk_kind(tops), tops, what + ('; ' + '; '.join(how) if how else '')

    out = []
    for m in mounts:
        if SKIP_MOUNT.match(m['mount']) and m['mount'] != '/dev/shm':
            continue
        fs = m['fstype']
        real = m['source'].startswith('/dev/') or fs in NETWORK_FS or re.match(r'^[^/]+:/', m['source'])
        ram = fs == 'tmpfs' and (m['mount'] in ('/tmp', '/dev/shm') or m['mount'].startswith(('/mnt/', '/srv/')))
        if not (real or ram):
            continue
        kind, tops, how = resolve(m)
        src = re.sub(r'^[^/]+:/', '<server>:/', m['source']) if kind == 'network' else m['source']  # no LAN host names in shared files
        out.append({'mount': m['mount'], 'kind': kind or 'unknown', 'fstype': fs, 'source': src,
                    'size_gib': r2(m['size'] / GiB) if m['size'] is not None else None, 'free_gib': r2(m['avail'] / GiB) if m['avail'] is not None else None,
                    'rotational': any(t['rota'] for t in tops.values()) if tops else None, 'members': sorted(tops), 'how': how})
    return out


# ---------------------------------------------------------------- assemble the box entry
def build_box(text, name=None, host=None):
    secs, err = sections(text)
    meta = dict(l.split('=', 1) for l in secs.get('meta', []) if '=' in l)
    host = host or meta.get('hostname') or 'localhost'
    name = name or host
    src = 'probe %s %s' % (host, meta.get('time', '?'))
    unknown = []

    def unk(field, reason):
        unknown.append({'field': field, 'reason': reason})
        return n(None, U, reason)

    cpu = parse_cpu(secs.get('cpuinfo', []))
    thr = cpu['threads_allowed'] or cpu['threads_online']
    box = {'name': name, 'cpu': {
        'model': n(cpu['model'], M, src + ' /proc/cpuinfo') if cpu['model'] else unk('cpu.model', 'no model name in /proc/cpuinfo'),
        'sockets': n(cpu['sockets'], M, src + ' /proc/cpuinfo physical id'),
        'phys': n(cpu['phys'], M, src + ' /proc/cpuinfo (physical id, core id) pairs'),
        'threads': n(thr, M, src + (' Cpus_allowed_list %s of %d online (a user process gets these)' % (cpu['allowed_list'], cpu['threads_online'])
                                    if cpu['threads_allowed'] and cpu['threads_allowed'] != cpu['threads_online'] else ' online CPUs')),
        'threads_online': n(cpu['threads_online'], M, src + ' /sys/devices/system/cpu/online'),
        'isa': n(cpu['isa'], M, src + ' /proc/cpuinfo flags') if cpu['isa'] is not None else unk('cpu.isa', 'no flags line in /proc/cpuinfo'),
    }}
    if cpu['phys'] and thr:
        box['cpu']['smt'] = n(cpu['threads_online'] > cpu['phys'], M, src + ' online CPUs vs physical cores')

    # NUMA: nodes with CPUs are DRAM; memory-only nodes are tier candidates (PMem via kmem, CXL, HBM)
    nodes = parse_nodes(secs.get('nodes', []))
    mi = parse_meminfo(secs.get('meminfo', []))
    dax, nd = parse_dax(secs.get('dax', [])), parse_nd(secs.get('nd', []))
    nsrc = src + ' /sys/devices/system/node/node*/meminfo'
    cpu_nodes = [x for x in nodes if x['cpus']]
    mem_nodes = [x for x in nodes if not x['cpus'] and (x['total_kb'] or 0) > 0]
    dram = {}
    if cpu_nodes:
        dram['total_gib'] = n(r2(sum(x['total_kb'] for x in cpu_nodes) * KiB / GiB), M, nsrc + ' (MemTotal of nodes with CPUs)')
        dram['free_gib'] = n(r2(sum(x['free_kb'] for x in cpu_nodes) * KiB / GiB), M, nsrc + ' (MemFree now, not a plan figure)')
        dram['nodes'] = [{'node': x['node'], 'name': 'node %d' % x['node'], 'cpus': x['cpus'], 'gib': r2(x['total_kb'] * KiB / GiB),
                          'free_gib': r2(x['free_kb'] * KiB / GiB), 'tag': M, 'src': nsrc} for x in cpu_nodes]
    elif 'MemTotal' in mi:
        dram['total_gib'] = n(r2(mi['MemTotal'] * KiB / GiB), M, src + ' /proc/meminfo (no NUMA sysfs)')
        dram['free_gib'] = n(r2(mi['MemFree'] * KiB / GiB), M, src + ' /proc/meminfo')
        unknown.append({'field': 'dram.nodes', 'reason': err.get('nodes', 'no NUMA nodes in sysfs')})
    if 'MemAvailable' in mi:
        dram['available_gib'] = n(r2(mi['MemAvailable'] * KiB / GiB), M, src + ' /proc/meminfo MemAvailable (whole box, incl. memory-only nodes)')

    dmi = None if 'dmidecode' in err else parse_dmidecode(secs.get('dmidecode', []))
    if dmi and dmi['dram']:
        dd = dmi['dram']
        mts = sorted({d['mts'] for d in dd if d['mts']})
        types_ = sorted({d['type'] for d in dd if d['type']})
        dsrc = src + ' sudo -n dmidecode -t memory'
        dram['dimms'] = n({'populated': len(dd), 'slots': dmi['slots'] - len(dmi['pmem']), 'gib_each': sorted({round(d['gib'], 1) for d in dd}), 'type': '/'.join(types_)},
                          M, dsrc + ' (slots = memory devices listed, PMem modules excluded)')
        dram['speed_mts'] = n(mts[0] if len(mts) == 1 else mts, M, dsrc + ' Configured Memory Speed') if mts else unk('dram.speed_mts', 'dmidecode lists no configured speed')
        # channels: only when the bank locators name them (e.g. "P0 CHANNEL A"); many boards name only the socket
        ch = {(re.search(r'P(\d+)', d['bank'] or '') or [None, '0'])[1] + ':' + mch.group(1) for d in dd for mch in [re.search(r'CHANNEL\s*([A-Z0-9]+)', d['bank'] or '', re.I)] if mch}
        if ch and all(re.search(r'CHANNEL', d['bank'] or '', re.I) for d in dd):
            dram['channels'] = n(len(ch), M, dsrc + ' distinct populated channels in Bank Locator')
            if len(mts) == 1:
                dram['bw_peak_gbs'] = n(r2(len(ch) * mts[0] * 8 / 1000), E, 'nameplate arithmetic: %d channels x %d MT/s x 8 B (not a measured read rate)' % (len(ch), mts[0]))
        else:
            sockets = {}
            for d in dd:
                key = (re.match(r'(CPU\d+|P\d+)', d['locator'] or '') or re.match(r'(CPU\d+|P\d+)', d['bank'] or '') or [None, '?'])[1]
                sockets[key] = sockets.get(key, 0) + 1
            dram['channels'] = unk('dram.channels', 'dmidecode bank locators name no channels (%s); populated DIMMs per socket %s' % (
                ', '.join(sorted({d['bank'] or '?' for d in dd})), json.dumps(sockets, sort_keys=True)))
    else:
        reason = err.get('dmidecode', 'dmidecode listed no DRAM modules')
        dram['dimms'] = unk('dram.dimms', reason)
        dram['speed_mts'] = unk('dram.speed_mts', reason)
        dram['channels'] = unk('dram.channels', reason)
    box['dram'] = dram

    # memory-only nodes: identify the technology behind each
    if mem_nodes:
        regions = {r.get('region'): r for r in nd['regions']}
        intel_nfit = [m for m in nd['nmems'] if NFIT_VENDOR.get(m.get('vendor'))]
        tiers = []
        for x in mem_nodes:
            dv = [d for d in dax if d.get('target_node') == str(x['node'])]
            how, kind = [], 'unknown'
            for d in dv:
                path = d.get('path', '')
                reg = re.search(r'/(region\d+)/', path)
                if reg and regions.get(reg.group(1), {}).get('devtype') == 'nd_pmem':
                    r = regions[reg.group(1)]
                    vend = ', '.join(sorted({NFIT_VENDOR[m['vendor']] for m in intel_nfit})) or 'unknown vendor'
                    kind = 'optane' if intel_nfit else 'nvdimm'
                    how.append('%s (kmem: %s) on NVDIMM %s, %s modules, NFIT vendor %s%s' % (d['dev'], d.get('driver') or '?', reg.group(1), r.get('mappings') or '?', vend,
                                                                                          ' = Intel Optane PMem' if intel_nfit else ''))
                elif '/cxl' in path or nd['cxl']:
                    kind = 'cxl'
                    how.append('%s on a CXL region (%s)' % (d['dev'], path))
                else:
                    how.append('%s (%s)' % (d['dev'], path))
            if not dv:
                how.append('memory-only node with no dax device in /sys/bus/dax (HBM, CXL without dax, or firmware-onlined memory): technology UNKNOWN')
            tiers.append({'node': x['node'], 'gib': r2(x['total_kb'] * KiB / GiB), 'free_gib': r2(x['free_kb'] * KiB / GiB), 'kind': kind, 'how': '; '.join(how),
                          'tag': M, 'src': nsrc + ' + /sys/bus/dax + /sys/bus/nd'})
            if kind == 'unknown':
                unknown.append({'field': 'pmem.nodes[%d].kind' % x['node'], 'reason': '; '.join(how)})
        pm = {'total_gib': n(r2(sum(t['gib'] for t in tiers)), M, nsrc + ' (MemTotal of memory-only nodes)'),
              'free_gib': n(r2(sum(t['free_gib'] for t in tiers)), M, nsrc + ' (MemFree now)'), 'nodes': tiers}
        if dmi and dmi['pmem']:
            pm['modules'] = n({'count': len(dmi['pmem']), 'gib_each': sorted({round(d['gib'], 1) for d in dmi['pmem']}), 'mts': sorted({d['mts'] for d in dmi['pmem'] if d['mts']})},
                              M, src + ' sudo -n dmidecode -t memory (non-volatile modules)')
        box['pmem'] = pm
    else:
        box['pmem'] = None

    # GPUs: nvidia-smi / xpu-smi / sysfs, joined on the PCI address
    drm = {g['pci']: g for g in parse_drm(secs.get('drm', []))}
    gpus = []
    for g in parse_nvidia(secs.get('nvidia', [])):
        gpus.append({'pci': g['pci'], 'vendor': 'nvidia', 'model': g['name'], 'discrete': True, 'total': g['total_mib'] / 1024 if g['total_mib'] else None,
                     'free': g['free_mib'] / 1024 if g['free_mib'] is not None else None, 'via': 'nvidia-smi memory.total/free (MiB)', 'driver': 'nvidia ' + (g['driver'] or '')})
    for g in parse_xpu(secs):
        gpus.append({'pci': g['pci'], 'vendor': 'intel', 'model': g['name'], 'discrete': g['device_type'] == 'Discrete GPU' if g['device_type'] else None, 'total': g['total_bytes'] / GiB if g['total_bytes'] else None,
                     'free': g['free_bytes'] / GiB if g['free_bytes'] is not None else None, 'via': 'xpu-smi discovery memory_physical_size_byte / memory_free_size_byte',
                     'driver': (drm.get(g['pci']) or {}).get('driver')})
    seen = {g['pci'] for g in gpus}
    for pci, d in drm.items():
        if pci in seen:
            continue
        # amdgpu lists a VRAM vendor only for real VRAM; an APU's carve-out has none. An Intel GPU on bus 00 is the iGPU.
        if d['vendor'] not in PCI_VENDOR.values():  # BMC / server VGA (ASPEED, Matrox): a display controller, not a compute GPU
            continue
        disc = True if d['vram_vendor'] else (False if d['driver'] == 'amdgpu' or (d['vendor'] == 'intel' and pci.startswith('0000:00:')) else None)
        if d['vram_total']:
            gpus.append({'pci': pci, 'vendor': d['vendor'], 'discrete': disc, 'model': '%s %s' % (d['vendor'], d['device']), 'total': d['vram_total'] / GiB,
                         'free': (d['vram_total'] - d['vram_used']) / GiB if d['vram_used'] is not None else None, 'via': 'sysfs mem_info_vram_total/used', 'driver': d['driver']})
        else:
            tool = {'nvidia': 'nvidia-smi', 'intel': 'xpu-smi'}.get(d['vendor'], 'a vendor tool')
            gpus.append({'pci': pci, 'vendor': d['vendor'], 'discrete': disc, 'model': '%s %s' % (d['vendor'], d['device']), 'total': None, 'free': None,
                         'via': 'no VRAM reading: %s %s' % (tool, err.get({'nvidia-smi': 'nvidia', 'xpu-smi': 'xpu'}.get(tool, ''), 'gave none')), 'driver': d['driver']})
    out_g = []
    for g in sorted(gpus, key=lambda x: x['pci']):
        d = drm.get(g['pci']) or {}
        gsrc = src + ' ' + g['via']
        e = {'pci': g['pci'], 'name': g['model'], 'model': n(g['model'], M, gsrc), 'vendor': g['vendor'], 'driver': g['driver'],
             'vram_gib': n(round(g['total'], 4), M, gsrc) if g['total'] else unk('gpus[%s].vram_gib' % g['pci'], g['via']),
             'vram_free_gib': n(r2(g['free']), M, gsrc + ' (now)') if g['free'] is not None else n(None, U, g['via'])}
        e['discrete'] = g['discrete']
        if d:
            rp, ep = d['rp'], d['ep']
            link = rp if d['behind_switch'] else {'gen': ep['gen'] or rp['gen'], 'width': ep['width'] or rp['width'],
                                                  'max_gen': min(x for x in (ep['max_gen'], rp['max_gen']) if x) if (ep['max_gen'] or rp['max_gen']) else None,
                                                  'max_width': min(x for x in (ep['max_width'], rp['max_width']) if x) if (ep['max_width'] or rp['max_width']) else None}
            e['pcie'] = n(link, M, src + (' root-port link (the card sits behind an on-board PCIe switch; its endpoint reports a virtual gen%s x%s link)' % (ep['gen'], ep['width'])
                                          if d['behind_switch'] else ' sysfs current/max link (current gen can idle down)'))
        else:
            e['pcie'] = unk('gpus[%s].pcie' % g['pci'], 'no DRM card for this PCI address in sysfs')
        out_g.append(e)
    box['gpus'] = out_g
    if not out_g:
        unknown.append({'field': 'gpus', 'reason': 'no display-class PCI device under /sys/class/drm (%s)' % '; '.join('%s: %s' % kv for kv in sorted(err.items()) if kv[0] in ('nvidia', 'xpu'))})

    # disks
    mounts = parse_mounts(secs.get('mounts', []))
    devs = parse_lsblk(secs.get('lsblk', []))
    disks = mount_disks(mounts, devs, parse_btrfs(secs.get('btrfs', [])), parse_loop(secs.get('loop', [])))
    for d in disks:
        d['tag'] = M
        d['src'] = src + ' findmnt + lsblk ROTA/TRAN: ' + d.pop('how')
        if d['kind'] == 'unknown':
            unknown.append({'field': 'disks[%s].kind' % d['mount'], 'reason': d['src']})
    box['disks'] = disks
    if 'home_mount' in meta:  # collectors before 2026-10-17 21Z did not print it
        hm = meta['home_mount'].strip()
        if hm:
            box['home_mount'] = n(hm, M, src + ' findmnt -T $HOME (the mount "~/" recipe paths resolve to)')
        else:
            unknown.append({'field': 'home_mount', 'reason': 'findmnt -T $HOME printed nothing'})

    # P0 system settings
    sc = parse_sysctl(secs.get('sysctl', []))
    swaps = parse_swaps(secs.get('swaps', []))
    ssrc = src + ' '

    def scn(key, conv, path):
        val = sc.get(key)
        return n(conv(val), M, ssrc + path) if val is not None else unk('system.' + key, path + ' unreadable')
    box['system'] = {
        'numa_balancing': scn('numa_balancing', int, '/proc/sys/kernel/numa_balancing'),
        'demotion': scn('demotion_enabled', lambda x: x.strip().lower() in ('true', '1'), '/sys/kernel/mm/numa/demotion_enabled'),
        'thp': scn('thp_enabled', thp_mode, '/sys/kernel/mm/transparent_hugepage/enabled'),
        'thp_defrag': scn('thp_defrag', thp_mode, '/sys/kernel/mm/transparent_hugepage/defrag'),
        'swappiness': scn('swappiness', int, '/proc/sys/vm/swappiness'),
        'swap': n('on' if swaps else 'off', M, ssrc + '/proc/swaps'),
        'swap_devices': n(swaps, M, ssrc + '/proc/swaps'),
    }
    box['probe'] = {'host': host, 'time': meta.get('time'), 'kernel': meta.get('kernel'), 'sudo': meta.get('sudo') == 'yes', 'src': src,
                    'unknown': unknown, 'unreadable_sections': err}
    return box


# ---------------------------------------------------------------- diff against a hand table
def v(x):
    return x.get('v') if isinstance(x, dict) and 'tag' in x and 'v' in x else x


def close(a, b, rel=0.01, abs_=0.06):
    try:
        return abs(float(a) - float(b)) <= max(abs_, rel * max(abs(float(a)), abs(float(b))))
    except (TypeError, ValueError):
        return a == b


def match_gpus(hand, probed):
    """hand GPU index -> probed GPU (by 'pci' key, by the short PCI address in the hand name, then by order)."""
    pool = [g for g in probed if g.get('discrete') is not False]
    out, used = {}, set()
    for i, h in enumerate(hand):
        for g in pool:
            short = g['pci'].split(':', 1)[-1]  # 0000:4c:00.0 -> 4c:00.0
            if g['pci'] not in used and (h.get('pci') == g['pci'] or short in str(h.get('name', '')).lower()):
                out[i] = g
                used.add(g['pci'])
                break
    rest = [g for g in pool if g['pci'] not in used]
    for i, h in enumerate(hand):
        if i not in out and rest:
            out[i] = rest.pop(0)
    return out, rest


def diff_box(hand, box):
    """-> list of (field, hand value, probed value, note). Only fields both sides describe, plus presence mismatches."""
    rows = []
    add = lambda f, a, b, note='': rows.append((f, a, b, note))
    hc, pc = hand.get('cpu', {}), box['cpu']
    for k in ('threads', 'phys'):
        if k in hc and not close(v(hc[k]), v(pc[k]), 0, 0):
            add('cpu.' + k, v(hc[k]), v(pc[k]), pc[k]['src'])
    desc = hc.get('desc', '')
    model = v(pc['model']) or ''
    sock = v(pc['sockets'])
    if sock and sock > 1 and not desc.startswith('%dx' % sock):
        add('cpu.desc sockets', desc, '%d sockets' % sock)
    toks = [t for t in re.findall(r'\b(\d{4,5}[A-Z0-9]*)\b', model)]
    if toks and not any(t in desc for t in toks):
        add('cpu.desc model', desc, model)
    isa = v(pc['isa']) or {}
    for flag, word in (('avx512f', 'AVX-512'), ('avx512_vnni', 'VNNI'), ('amx_tile', 'AMX')):
        if word.lower() in desc.lower() and not isa.get(flag):
            add('cpu.isa', desc, flag + ' absent')
        if isa.get(flag) and word == 'AVX-512' and 'avx-512' not in desc.lower():
            add('cpu.isa', desc, 'has ' + flag + ' (desc does not say AVX-512)')
    hd, pd = hand.get('dram', {}), box['dram']
    if 'total_gib' in hd and 'total_gib' in pd and not close(v(hd['total_gib']), v(pd['total_gib'])):
        add('dram.total_gib', v(hd['total_gib']), v(pd['total_gib']), pd['total_gib']['src'])
    hn, pn = hd.get('nodes') or [], pd.get('nodes') or []
    if len(hn) != len(pn) and pn:
        add('dram.nodes count', len(hn), len(pn), 'nodes with CPUs: ' + ', '.join('%d (%s)' % (x['node'], x['cpus']) for x in pn))
    for i, (a, b) in enumerate(zip(hn, pn)):
        if not close(a.get('gib'), b['gib']):
            add('dram.nodes[%d].gib' % i, a.get('gib'), b['gib'])
        if a.get('avail_gib') is not None:
            add('dram.nodes[%d].avail_gib vs MemFree now' % i, v(a['avail_gib']), b['free_gib'], 'not a discrepancy by itself: avail_gib is a plan-time figure, MemFree is now')
    hp, pp = hand.get('pmem'), box.get('pmem')
    if bool(hp) != bool(pp):
        add('pmem present', bool(hp), bool(pp), (pp or {}).get('nodes') and pp['nodes'][0]['how'] or '')
    elif hp and pp:
        add('pmem.gib (hand usable) vs total_gib', v(hp['gib']), v(pp['total_gib']), 'hand keeps a usable figure; probe reads MemTotal of the memory-only nodes')
        kinds = {t['kind'] for t in pp['nodes']}
        if kinds != {'optane'}:
            add('pmem kind', 'Optane (hand)', ', '.join(sorted(kinds)))
    hg, pg = hand.get('gpus') or [], box['gpus']
    mt, rest = match_gpus(hg, pg)
    for i, h in enumerate(hg):
        g = mt.get(i)
        if not g:
            add('gpus[%s]' % h.get('id'), h.get('name'), None, 'no probed GPU matches')
            continue
        if not close(v(h.get('vram_gib')), v(g['vram_gib']), 0.005, 0.01):
            add('gpus[%s].vram_gib' % h.get('id'), v(h.get('vram_gib')), v(g['vram_gib']), g['vram_gib']['src'])
        lk = v(g.get('pcie')) or {}
        m = re.search(r'PCIe (\d) x(\d+)', h.get('name', ''))
        if m and lk and (int(m.group(1)) != lk.get('max_gen') or int(m.group(2)) != lk.get('max_width')):
            add('gpus[%s] PCIe link' % h.get('id'), 'gen%s x%s (name)' % m.groups(), 'gen%s x%s now, max gen%s x%s' % (lk.get('gen'), lk.get('width'), lk.get('max_gen'), lk.get('max_width')), g['pcie']['src'])
        for o in h.get('other') or []:
            if g.get('vram_free_gib', {}).get('v') is not None and v(o.get('gib')):
                used = r2(v(g['vram_gib']) - v(g['vram_free_gib']))
                add('gpus[%s] co-tenant "%s"' % (h.get('id'), o.get('what')), v(o['gib']), used, 'VRAM in use now (all processes)')
    for g in rest:
        add('gpus (unlisted)', None, '%s %s %s GiB' % (g['pci'], v(g['model']), v(g['vram_gib'])), 'probed GPU with no hand entry')
    for g in pg:
        if g.get('discrete') is False:
            add('gpus (integrated, ignored)', None, '%s %s %s GiB' % (g['pci'], v(g['model']), v(g['vram_gib'])))
    hdisk = {d['mount']: d for d in hand.get('disks') or []}
    pdisk = {d['mount']: d for d in box['disks']}
    for mnt, h in hdisk.items():
        p = pdisk.get(mnt)
        if not p:
            add('disks[%s]' % mnt, h['kind'], None, 'not mounted now')
        elif p['kind'] != h['kind']:
            add('disks[%s].kind' % mnt, h['kind'], p['kind'], p['src'])
    for mnt, p in pdisk.items():
        if mnt not in hdisk:
            add('disks[%s]' % mnt, None, '%s %s, %s GiB free' % (p['kind'], p['fstype'], p['free_gib']), 'mount not in the hand table')
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--ssh', help='probe this host over ssh (one bash script on stdin, read-only)')
    ap.add_argument('--name', help='box key (default: the ssh host or the local hostname)')
    ap.add_argument('--write', help='write the box JSON here (e.g. data/boxes/<name>.json)')
    ap.add_argument('--raw-out', help='also save the collector output (for fixtures)')
    ap.add_argument('--from-raw', help='parse a saved collector output instead of probing')
    ap.add_argument('--diff', help='compare with boxes[<name>] of this data.json (a build without overlays: FITVIZ_OVERLAYS=0)')
    ap.add_argument('--vitals', default=None, help='optional homelab vitals.json to cross-check disk kinds (adapter; ignored when absent or stale)')
    a = ap.parse_args(argv)
    text = Path(a.from_raw).read_text() if a.from_raw else collect(a.ssh)
    if a.raw_out:
        open(a.raw_out, 'w').write(text)
    box = build_box(text, name=a.name or a.ssh, host=a.ssh)
    try:
        sys.path.insert(0, HERE)
        import vitals_adapter
        vx = vitals_adapter.cross_check(box, a.ssh or box['probe']['host'], a.vitals)
        if vx:
            box['probe']['vitals'] = vx
    except ImportError:
        pass
    js = json.dumps(box, indent=1, sort_keys=False)
    if a.write:
        os.makedirs(os.path.dirname(os.path.abspath(a.write)), exist_ok=True)
        open(a.write, 'w').write(js + '\n')
    if a.diff:
        hand = json.loads(Path(a.diff).read_text())['boxes'].get(box['name'])
        if not hand:
            print('no boxes[%s] in %s: nothing to diff' % (box['name'], a.diff))
        else:
            rows = diff_box(hand, box)
            print('== %s: %d discrepancies (hand -> probe)' % (box['name'], len(rows)))
            for f, x, y, note in rows:
                print('  %-44s %s -> %s%s' % (f, json.dumps(x), json.dumps(y), ('   [' + note + ']') if note else ''))
        for u in box['probe']['unknown']:
            print('  UNKNOWN %-36s %s' % (u['field'], u['reason']))
    elif not a.write:
        print(js)
    return 0


if __name__ == '__main__':
    sys.exit(main())
