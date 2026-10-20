"""Parser tests for probe/ (box probe, GGUF reader, model intake, overlay merge) on captured fixtures.

Run: python3 -m unittest discover -s tests -v      (from the fit-viz directory; no network, no ssh)
Fixtures: tests/fixtures/collect-<box>.txt = probe/collect.sh output captured 2026-10-17 (LAN addresses replaced);
gguf-*.hdr.json = probe/gguf.py output (tokenizer and file paths dropped); hf-*.json = config.json + safetensors
header rows (hf-v41-subset keeps experts 0-1 of every layer).
"""
import datetime
import io
import json
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
FX = os.path.join(HERE, 'fixtures')
sys.path.insert(0, os.path.join(os.path.dirname(HERE), 'probe'))
import gguf  # noqa: E402
import model_intake as mi  # noqa: E402
import overlay  # noqa: E402
import probe_box as pb  # noqa: E402
import vitals_adapter  # noqa: E402


def box(name):
    return pb.build_box(Path(os.path.join(FX, 'collect-%s.txt' % name)).read_text(), name=name)


def disk(b, mount):
    return next(d for d in b['disks'] if d['mount'] == mount)


class ProbeDroopy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = box('droopy')

    def test_cpu(self):
        c = self.b['cpu']
        self.assertEqual((c['sockets']['v'], c['phys']['v'], c['threads']['v']), (2, 20, 40))
        self.assertEqual(c['threads']['tag'], 'MEASURED')
        self.assertTrue(c['isa']['v']['avx512_vnni'])
        self.assertFalse(c['isa']['v']['amx_tile'])
        self.assertIn('Gold 5215', c['model']['v'])

    def test_numa_and_pmem(self):
        d = self.b['dram']
        self.assertEqual([x['node'] for x in d['nodes']], [0, 1])
        self.assertEqual(d['nodes'][0]['cpus'], '0-9,20-29')
        self.assertAlmostEqual(d['total_gib']['v'], 183.35, places=1)
        p = self.b['pmem']
        self.assertEqual(p['nodes'][0]['node'], 2)
        self.assertEqual(p['nodes'][0]['kind'], 'optane')
        self.assertIn('Intel Optane PMem', p['nodes'][0]['how'])
        self.assertEqual(p['modules']['v']['count'], 2)

    def test_dimms_channels_unknown(self):
        d = self.b['dram']
        self.assertEqual(d['dimms']['v']['populated'], 12)
        self.assertEqual(d['speed_mts']['v'], 2666)
        self.assertEqual(d['channels']['tag'], 'UNKNOWN')
        self.assertIn('dram.channels', [u['field'] for u in self.b['probe']['unknown']])

    def test_gpu(self):
        g = self.b['gpus'][0]
        self.assertEqual(g['pci'], '0000:2d:00.0')
        self.assertAlmostEqual(g['vram_gib']['v'], 22.5)
        self.assertEqual((g['pcie']['v']['max_gen'], g['pcie']['v']['max_width']), (3, 16))

    def test_disks(self):
        self.assertEqual(disk(self.b, '/mnt/datadrive1')['kind'], 'hdd')            # NTFS via FUSE on a SATA HDD
        self.assertEqual(disk(self.b, '/mnt/datadrive1/fab-hot')['kind'], 'hdd')    # loop image living on that HDD
        self.assertIn('image file', disk(self.b, '/mnt/datadrive1/fab-hot')['src'])
        self.assertEqual(disk(self.b, '/mnt/nvme-p4')['kind'], 'nvme')
        net = disk(self.b, '/mnt/odie-cache')
        self.assertEqual(net['kind'], 'network')
        self.assertTrue(net['source'].startswith('<server>:'))
        self.assertEqual(disk(self.b, '/dev/shm')['kind'], 'memory')

    def test_system(self):
        s = self.b['system']
        self.assertEqual(s['numa_balancing']['v'], 0)
        self.assertIs(s['demotion']['v'], False)
        self.assertEqual(s['thp']['v'], 'madvise')
        self.assertEqual(s['swap']['v'], 'on')


class ProbeHomeMount(unittest.TestCase):
    def test_home_mount_recorded_without_the_path(self):
        raw = Path(os.path.join(FX, 'collect-droopy.txt')).read_text().replace('sudo=yes\n', 'sudo=yes\nhome_mount=/\n', 1)
        b = pb.build_box(raw, name='droopy')
        self.assertEqual(b['home_mount']['v'], '/')
        self.assertEqual(b['home_mount']['tag'], 'MEASURED')

    def test_old_collector_has_no_home_mount(self):
        self.assertNotIn('home_mount', box('droopy'))

    def test_empty_home_mount_is_unknown(self):
        raw = Path(os.path.join(FX, 'collect-droopy.txt')).read_text().replace('sudo=yes\n', 'sudo=yes\nhome_mount=\n', 1)
        b = pb.build_box(raw, name='droopy')
        self.assertNotIn('home_mount', b)
        self.assertIn('home_mount', [u['field'] for u in b['probe']['unknown']])


class ProbeOdie(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.b = box('odie')

    def test_allowed_cpus(self):
        c = self.b['cpu']
        self.assertEqual(c['threads']['v'], 44)        # user.slice cpuset 0-21,24-45
        self.assertEqual(c['threads_online']['v'], 48)
        self.assertFalse(c['isa']['v']['avx512f'])

    def test_b70_behind_switch(self):
        gs = {g['pci']: g for g in self.b['gpus']}
        g = gs['0000:4c:00.0']
        self.assertAlmostEqual(g['vram_gib']['v'], 31.89, places=2)
        self.assertTrue(g['discrete'])
        self.assertEqual((g['pcie']['v']['gen'], g['pcie']['v']['width']), (4, 16))  # root port, not the virtual x1
        self.assertIn('root-port', g['pcie']['src'])

    def test_btrfs_raid_members(self):
        d = disk(self.b, '/mnt/bulk')
        self.assertEqual(d['kind'], 'hdd')
        self.assertEqual(d['members'], ['sda', 'sdb'])
        self.assertEqual(disk(self.b, '/mnt/models-cache')['kind'], 'hdd')  # USB HDD

    def test_channels(self):
        self.assertEqual(self.b['dram']['channels']['v'], 4)
        self.assertEqual(self.b['dram']['bw_peak_gbs']['tag'], 'ESTIMATE')
        self.assertIsNone(self.b['pmem'])


class ProbeSnoopyLassie(unittest.TestCase):
    def test_snoopy_igpu_ignored(self):
        b = box('snoopy')
        disc = {g['pci']: g['discrete'] for g in b['gpus']}
        self.assertEqual(disc, {'0000:01:00.0': True, '0000:79:00.0': False})
        self.assertEqual(disk(b, '/scratch')['kind'], 'nvme')

    def test_lassie_bmc_vga_skipped(self):
        b = box('lassie')
        self.assertEqual([g['pci'] for g in b['gpus']], ['0000:21:00.0'])
        self.assertEqual(len(b['dram']['nodes']), 8)
        self.assertEqual(disk(b, '/')['kind'], 'sata-ssd')  # LVM on a SATA SSD


class ProbeUnreadable(unittest.TestCase):
    def test_sections_and_unknown(self):
        txt = '@@meta\nhostname=x\ntime=T\n@@cpuinfo\nprocessor : 0\n@@dmidecode\n@@!dmidecode needs root: no passwordless sudo\n@@nvidia\n@@!nvidia nvidia-smi not installed\n@@sysctl\nnuma_balancing=!unreadable /proc/sys/kernel/numa_balancing\n'
        secs, err = pb.sections(txt)
        self.assertEqual(err['dmidecode'], 'needs root: no passwordless sudo')
        b = pb.build_box(txt)
        self.assertEqual(b['dram']['speed_mts']['tag'], 'UNKNOWN')
        self.assertIn('no passwordless sudo', b['dram']['speed_mts']['src'])
        self.assertEqual(b['system']['numa_balancing']['tag'], 'UNKNOWN')

    def test_disk_kind(self):
        t = lambda rota, tran: {'rota': rota, 'tran': tran}
        self.assertEqual(pb.disk_kind({'nvme0n1': t(False, 'nvme')}), 'nvme')
        self.assertEqual(pb.disk_kind({'nvme0n1': t(False, 'nvme'), 'sda': t(True, 'sata')}), 'hdd')  # any rotational member
        self.assertEqual(pb.disk_kind({'sdb': t(False, 'usb')}), 'usb-ssd')
        self.assertEqual(pb.disk_kind({'sda': t(False, 'sata')}), 'sata-ssd')
        self.assertIsNone(pb.disk_kind({}))

    def test_cpulist(self):
        self.assertEqual(pb.cpulist_count('0-21,24-45'), 44)
        self.assertEqual(pb.cpulist_count('3'), 1)


def write_gguf(path, kv, tensors, align=32, split=None):
    """minimal GGUF v3 writer for tests: kv {key: int|str|list[int]}, tensors [(name, dims, type, nbytes)]"""
    b = io.BytesIO()
    s = lambda x: b.write(struct.pack('<Q', len(x.encode())) + x.encode())
    items = list(kv.items()) + ([('split.no', split[0]), ('split.count', split[1])] if split else [])
    b.write(b'GGUF' + struct.pack('<IQQ', 3, len(tensors), len(items)))
    for k, v in items:
        s(k)
        if isinstance(v, str):
            b.write(struct.pack('<I', 8)); s(v)
        elif isinstance(v, list):
            b.write(struct.pack('<IIQ', 9, 4, len(v)) + struct.pack('<%dI' % len(v), *v))
        else:
            b.write(struct.pack('<II', 4, v))
    off = 0
    for name, dims, t, nb in tensors:
        s(name)
        b.write(struct.pack('<I', len(dims)) + struct.pack('<%dQ' % len(dims), *dims) + struct.pack('<IQ', t, off))
        off += (nb + align - 1) // align * align
    pad = (-b.tell()) % align
    b.write(b'\0' * pad)
    data = b''.join(b'\x11' * ((nb + align - 1) // align * align) for *_, nb in tensors)
    with open(path, 'wb') as f:
        f.write(b.getvalue() + data)


class Gguf(unittest.TestCase):
    def test_split_and_sizes(self):
        with tempfile.TemporaryDirectory() as d:
            p1, p2 = (os.path.join(d, 'm-%05d-of-00002.gguf' % i) for i in (1, 2))
            kv = {'general.architecture': 'toy', 'toy.block_count': 2, 'toy.expert_count': 4, 'toy.expert_used_count': 2,
                  'toy.embedding_length': 64, 'toy.expert_feed_forward_length': 32, 'toy.layer_ids': [1, 2, 3]}
            write_gguf(p1, kv, [('token_embd.weight', [64, 100], 8, 6800), ('blk.0.ffn_up_exps.weight', [64, 32, 4], 39, 4352)], split=(0, 2))
            write_gguf(p2, {'general.architecture': 'toy'}, [('blk.1.ffn_up_exps.weight', [64, 32, 4], 10, 2688)], split=(1, 2))
            h = gguf.read_files([p1])
        self.assertEqual([f[0] for f in h['files']], ['m-00001-of-00002.gguf', 'm-00002-of-00002.gguf'])
        self.assertEqual(h['kv']['toy.layer_ids'], [1, 2, 3])
        by = {t[0]: t for t in h['tensors']}
        self.assertEqual(by['token_embd.weight'][3], 6816)          # 6800 padded to 32
        self.assertEqual(by['blk.0.ffn_up_exps.weight'][3], 4352)   # last tensor: to end of file
        self.assertEqual(gguf.type_name(39), 'mxfp4')
        self.assertEqual(gguf.type_name(999), 'type999')

    def test_chunked_fetch_reads_header_only(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'x.gguf')
            write_gguf(p, {'general.architecture': 'toy', 'general.name': 'x' * 5000}, [('a', [8], 0, 1 << 20)])
            reads = []
            with open(p, 'rb') as f:
                def fetch(off, k):
                    reads.append((off, k))
                    f.seek(off)
                    return f.read(k)
                h = gguf.read_header(fetch, os.path.getsize(p), chunk=1024)
        self.assertEqual(h['kv']['general.name'], 'x' * 5000)
        self.assertLess(sum(k for _, k in reads), 64 << 10)  # 1 MiB of tensor data never read

    def test_shard_paths(self):
        self.assertEqual(gguf.shard_paths('/a/M-00001-of-00003.gguf')[-1], '/a/M-00003-of-00003.gguf')
        self.assertEqual(gguf.shard_paths('/a/M.gguf'), ['/a/M.gguf'])


class Intake(unittest.TestCase):
    def gguf_model(self, f):
        hdr = json.loads(Path(os.path.join(FX, f)).read_text())
        rows, shape = mi.from_gguf(hdr, 'fixture')
        return mi.summarise(rows, shape, 'fixture')

    def test_v41_split_gguf(self):
        m = self.gguf_model('gguf-v41-split.hdr.json')
        self.assertEqual((m['L'], m['E'], m['k'], m['P']), (40, 384, 6, 5120 * 2304 * 3))
        it = m['intake']
        self.assertAlmostEqual(it['gib_by_class']['engram_table'], 194.555, places=2)
        self.assertAlmostEqual(it['dense_gib'], 8.33, places=2)          # = hand dense (Q8_0 + BF16 output)
        self.assertEqual(set(it['expert_types']['expert_down']), {'mxfp4', 'q2_K', 'q3_K'})
        self.assertEqual(m['kv_bytes_per_token']['tag'], 'UNKNOWN')
        self.assertFalse(m['mtp']['v']['in_this_source'])
        self.assertIn('blk.{L}.ffn_up_exps_hot.weight', m['tensor_names'])
        self.assertEqual(it['unclassified'], [])

    def test_qwen_gguf(self):
        m = self.gguf_model('gguf-qfn-q8.hdr.json')
        self.assertEqual((m['L'], m['E'], m['k']), (48, 512, 10))
        self.assertEqual(len(m['layers']['v']['linear_attention']), 36)
        self.assertEqual(m['kv_bytes_per_token']['v'], 12 * 2 * 512 * 2)
        self.assertEqual(m['kv_bytes_per_token']['tag'], 'ESTIMATE')
        self.assertAlmostEqual(m['intake']['gib_by_class']['ngram_table'], 50.66, places=1)
        self.assertAlmostEqual(m['intake']['expert_bpw'], 8.5, places=2)
        self.assertEqual(m['intake']['unclassified'], [])

    def test_hf(self):
        d = json.loads(Path(os.path.join(FX, 'hf-qfn.json')).read_text())
        rows, shape = mi.from_hf(d['config'], d['rows'], 'fixture', 131)
        m = mi.summarise(rows, shape, 'fixture')
        self.assertEqual((m['L'], m['E'], m['k'], m['P']), (48, 512, 10, 3 * 2560 * 640))
        self.assertAlmostEqual(m['intake']['gib_by_class']['expert_gate_up'], 150.0, places=1)
        self.assertEqual(m['mtp']['v']['layers_config'], 1)
        self.assertTrue(m['mtp']['v']['in_this_source'])
        self.assertEqual(m['intake']['unclassified'], [])
        d = json.loads(Path(os.path.join(FX, 'hf-v41-subset.json')).read_text())
        rows, shape = mi.from_hf(d['config'], d['rows'], 'fixture', 48)
        m = mi.summarise(rows, shape, 'fixture')
        self.assertEqual((m['L'], m['E'], m['k']), (40, 384, 6))
        self.assertEqual(m['mtp']['v']['layers_config'], 3)
        self.assertIn('DSpark', m['draft_heads']['v'])
        self.assertGreater(m['intake']['gib_by_class']['vision'], 0.5)
        self.assertEqual(m['intake']['unclassified'], [])

    def test_classify(self):
        c = mi.classify
        self.assertEqual(c('layers.3.ffn.experts.7.w2.scale'), 'expert_down')
        self.assertEqual(c('layers.3.ffn.shared_experts.w1.weight'), 'shared_expert')
        self.assertEqual(c('layers.3.ffn.gate.bias'), 'router')
        self.assertEqual(c('model.language_model.layers.0.mlp.shared_expert_gate.weight'), 'router')
        self.assertEqual(c('blk.4.hc_attn_fn.weight'), 'hyper_connection')
        self.assertEqual(c('blk.0.attn_qkv.weight', 0, None, {0}), 'linear_attention')
        self.assertEqual(c('blk.45.attn_q.weight', 45, 45), 'mtp')


class Overlay(unittest.TestCase):
    def test_rules(self):
        n = lambda v, t: {'v': v, 'tag': t, 'src': 's'}
        hand = {'cpu': {'threads': n(40, 'MEASURED'), 'desc': 'hand'}, 'pmem': None, 'L': 40,
                'gpus': [{'id': 'a', 'name': 'card (4c:00.0)', 'vram_gib': n(29.8, 'READ')}, {'id': 'b', 'name': 'card (23:00.0)', 'vram_gib': n(29.8, 'READ')}],
                'dram': {'bw': n(148, 'MEASURED'), 'nodes': [{'name': 'node 0 (socket 0)', 'gib': 92.9, 'avail_gib': n(86, 'MEASURED')}]},
                'disks': [{'mount': '/', 'kind': 'unknown', 'src': 'inventory'}], 'kv': n(5.4, 'MEASURED')}
        over = {'cpu': {'threads': n(44, 'MEASURED'), 'model': n('X', 'MEASURED')}, 'pmem': {'total_gib': n(246, 'MEASURED')}, 'L': 43,
                'gpus': [{'pci': '0000:23:00.0', 'vram_gib': n(31.89, 'MEASURED')}, {'pci': '0000:4c:00.0', 'vram_gib': n(31.0, 'MEASURED')}],
                'dram': {'nodes': [{'name': 'node 0', 'gib': 92.91, 'free_gib': 0.5, 'tag': 'MEASURED', 'src': 'p'}]},
                'disks': [{'mount': '/', 'kind': 'nvme', 'tag': 'MEASURED', 'src': 'p'}, {'mount': '/x', 'kind': 'hdd', 'tag': 'MEASURED', 'src': 'p'}],
                'kv': n(5.0, 'READ'), 'new': n(None, 'UNKNOWN')}
        m, log = overlay.merge(hand, over)
        self.assertEqual(m['cpu']['threads']['v'], 44)
        self.assertEqual(m['cpu']['threads']['hand']['v'], 40)
        self.assertEqual(m['cpu']['desc'], 'hand')
        self.assertIsNone(m['pmem'])
        self.assertIn('pmem_probed', m)
        self.assertEqual(m['L'], 40)                                         # raw structural value: conflict, hand kept
        self.assertIn(('conflict', 'L', '40', '43'), log)
        self.assertEqual(m['gpus'][0]['vram_gib']['v'], 31.0)               # matched by PCI address, not by order
        self.assertEqual(m['gpus'][1]['vram_gib']['v'], 31.89)
        node = m['dram']['nodes'][0]
        self.assertEqual((node['name'], node['gib'], node['avail_gib']['v'], node['free_gib']), ('node 0 (socket 0)', 92.91, 86, 0.5))
        self.assertEqual(m['dram']['bw']['v'], 148)                         # hand-only field stays
        self.assertEqual(m['disks'][0]['kind'], 'nvme')
        self.assertEqual(m['disks'][0]['hand']['src'], 'inventory')
        self.assertEqual(m['disks'][1]['mount'], '/x')
        self.assertEqual(m['kv']['v'], 5.4)                                  # READ does not replace MEASURED
        self.assertEqual(m['new']['tag'], 'UNKNOWN')                         # added (no hand value)

    def test_vitals_adapter(self):
        b = {'disks': [{'mount': '/', 'kind': 'nvme'}, {'mount': '/b', 'kind': 'hdd'}]}
        now = datetime.datetime(2026, 10, 17, 20, 0, tzinfo=datetime.timezone.utc)
        doc = {'generated': '2026-10-17T19:58:00Z', 'nodes': [{'host': 'h', 'disks': [{'mount': '/', 'kind': 'NVMe'}, {'mount': '/b', 'kind': 'SSD'}]}]}
        with tempfile.NamedTemporaryFile('w', suffix='.json', delete=False) as f:
            json.dump(doc, f)
        try:
            r = vitals_adapter.cross_check(b, 'h', f.name, now)
            self.assertEqual(r['agree'], ['/'])
            self.assertEqual(r['disagree'][0]['mount'], '/b')
            self.assertIsNone(vitals_adapter.cross_check(b, 'h', f.name, now + datetime.timedelta(hours=1)))  # stale -> nothing
        finally:
            os.unlink(f.name)


if __name__ == '__main__':
    unittest.main()
