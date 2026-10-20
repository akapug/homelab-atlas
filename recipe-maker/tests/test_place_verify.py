"""tools/place_verify.py on real droopy placement logs (10-18/10-19 server.log lines)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
TOOL = os.path.join(os.path.dirname(HERE), 'tools', 'place_verify.py')
sys.path.insert(0, os.path.dirname(TOOL))
import place_verify as P  # noqa: E402

A = 'llama_expert_numa_apply: '
Q4Z = [A + '/srv/kit/optane-place/glm-be1q4d.placement.tsv: 129 expert tensors placed in 12009 ranges (0 skipped), 11880 straddle pages, 0.50 MiB left at tensor edges',
       A + '  node(s) 0,1      interleave   145403.8 MiB', A + '  node(s) 1        bind          12285.3 MiB', A + '  node(s) 2        bind          23372.4 MiB',
       'llama_expert_numa_report: verified (1 page in 64): node0=30849.8MiB node1=120952.8MiB node2=29288.8MiB; 3.26% of sampled pages are off their target']
R1 = [A + '/srv/kit/optane-place/glm-be1r1.placement.tsv: 129 expert tensors placed in 63945 ranges (0 skipped), 63816 straddle pages, 0.50 MiB left at tensor edges',
      A + '  node(s) 0        bind          72701.7 MiB', A + '  node(s) 1        bind          84987.3 MiB', A + '  node(s) 2        bind          23372.4 MiB',
      'llama_expert_numa_report: verified (1 page in 64): node0=72867.2MiB node1=85151.2MiB node2=23383.8MiB; 0.00% of sampled pages are off their target']
HEAL = [A + '/srv/kit/optane-place/v41-1017-split-dramfill.placement.tsv: 240 expert tensors placed in 11079 ranges (0 skipped), 10839 straddle pages, 0.94 MiB left at tensor edges',
        A + '  node(s) 0,1      interleave   155644.9 MiB', A + '  node(s) 2        bind          63061.5 MiB', A + '  node(s) 1        bind           3062.4 MiB',
        'llama_expert_numa_report: verified (1 page in 64): node0=86353.0MiB node1=68496.5MiB node2=68037.0MiB; 1.95% of sampled pages are off their target',
        A + '/srv/kit/optane-place/v41-1017-split-dramfill.placement.tsv: 0 expert tensors placed in 0 ranges (0 skipped), 0 straddle pages, 0.00 MiB left at tensor edges',
        A + '  node(s) 0,1      interleave        0.0 MiB', A + '  node(s) 2        bind              0.0 MiB']
SMOKE_IL = [A + '/srv/kit/optane-bench/srv/smoke-rows/il.tsv: 120 expert tensors placed in 120 ranges (0 skipped), 0 straddle pages, 0.47 MiB left at tensor edges',
            A + '  node(s) 0,1      interleave    18725.5 MiB',
            'llama_expert_numa_report: verified (1 page in 8): node0=18726.0MiB; 0.00% of sampled pages are off their target']


V41R1 = [A + '/srv/kit/optane-place/v41-1017-split-rows.placement.tsv: 240 expert tensors placed in 63102 ranges (0 skipped), 62088 straddle pages, 0.94 MiB left at tensor edges',
         A + '  node(s) 0        bind          80896.5 MiB', A + '  node(s) 1        bind          77810.7 MiB', A + '  node(s) 2        bind          63061.5 MiB',
         'llama_expert_numa_report: verified (1 page in 64): node0=89298.0MiB node1=56473.2MiB node2=79264.2MiB; 11.37% of sampled pages are off their target']
MAPS = ['7f0000000000 bind:1 anon=%d dirty=%d N1=%d kernelpagesize_kB=4' % (3 * 2**18, 3 * 2**18, 3 * 2**18),
        '7f1000000000 bind:0 anon=1 N0=%d N2=%d kernelpagesize_kB=4' % (2**18, 2**18),
        '7f2000000000 interleave:0-1 anon=1 N0=%d N1=%d N2=%d kernelpagesize_kB=4' % (2**18, 2**18, 2**18),
        '7f3000000000 default anon=1 N0=999999 kernelpagesize_kB=4', '7f4000000000 bind:2 file=/x N2=512 kernelpagesize_kB=2048']


class PlaceVerify(unittest.TestCase):
    def test_live_numa_maps_counts_bind_and_interleave_pages(self):
        got, off = P.live(MAPS)
        self.assertEqual(got, {1: 4096.0, 0: 2048.0, 2: 3072.0})   # default ranges are not the placement's
        self.assertAlmostEqual(off, 100.0 * 2048 / 9216, places=2)   # bind:0 pages on N2 + interleave 0-1 pages on N2

    def test_v41r1_rows_fail_from_the_log_and_the_live_maps_decide_after_a_heal(self):
        r = P.verify(V41R1)
        self.assertEqual(r['status'], 'FAIL')
        self.assertTrue(any(x.startswith('node1 -20.8 GiB') for x in r['reasons']), r['reasons'])
        healed = ['a bind:0 N0=%d kernelpagesize_kB=4' % int(80896.5 * 256), 'b bind:1 N1=%d kernelpagesize_kB=4' % int(77810.7 * 256),
                  'c bind:2 N2=%d kernelpagesize_kB=4' % int(63061.5 * 256)]
        h = P.verify(V41R1, numa_maps=healed)
        self.assertEqual(h['status'], 'PASS', h['reasons'])
        self.assertEqual(h['off_pct'], 0.0)

    def test_rows_on_target_passes(self):
        r = P.verify(R1)
        self.assertEqual(r['status'], 'PASS', r)
        self.assertEqual(r['expected_mib'], {'0': 72701.7, '1': 84987.3, '2': 23372.4})

    def test_skewed_interleave_fails_although_the_report_calls_it_on_target(self):
        r = P.verify(SMOKE_IL)
        self.assertEqual(r['status'], 'FAIL')
        self.assertEqual(r['off_pct'], 0.0)
        self.assertTrue(any(x.startswith('node1 -9.1 GiB') for x in r['reasons']), r['reasons'])
        q = P.verify(Q4Z)
        self.assertEqual(q['status'], 'FAIL')
        self.assertTrue(any(x.startswith('node0 -40.9 GiB') for x in q['reasons']) and any('3.26 % of sampled pages' in x for x in q['reasons']), q['reasons'])

    def test_last_block_with_tensors_is_checked_not_the_draft_echo(self):
        r = P.verify(HEAL)
        self.assertEqual(r['status'], 'FAIL')
        for k, v in {'0': 77822.45, '1': 80884.85, '2': 63061.5}.items():
            self.assertAlmostEqual(r['expected_mib'][k], v, delta=0.11)
        self.assertAlmostEqual(r['off_mib']['0'], 8530.55, delta=0.11)

    def test_no_block_is_none_and_bars_are_settings(self):
        self.assertEqual(P.verify(['llama_model_loader: loaded'])['status'], 'NONE')
        self.assertEqual(P.verify(HEAL, tol_gib=20, off_max=2.0)['status'], 'PASS')

    def test_cli_exit_codes_and_json(self):
        with tempfile.TemporaryDirectory() as d:
            for name, lines, rc in (('r1', R1, 0), ('q4z', Q4Z, 3), ('none', ['x'], 2)):
                log, js = os.path.join(d, name + '.log'), os.path.join(d, name + '.json')
                with open(log, 'w') as f:
                    f.write('\n'.join(lines) + '\n')
                p = subprocess.run([sys.executable, '-I', TOOL, log, '--json', js], capture_output=True, text=True)
                self.assertEqual(p.returncode, rc, p.stdout + p.stderr)
                self.assertTrue(p.stdout.startswith('== PLACEMENT '), p.stdout)
                with open(js) as f:
                    self.assertEqual(json.load(f)['status'], {0: 'PASS', 3: 'FAIL', 2: 'NONE'}[rc])


if __name__ == '__main__':
    unittest.main()
