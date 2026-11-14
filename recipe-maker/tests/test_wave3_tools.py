import math
import json
import subprocess
import tempfile
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from heat_shape import synth_heat
from priors_adapter import _num

class Wave3Tools(unittest.TestCase):
    def test_missing_heat_never_fabricates_expert_ids(self):
        self.assertIsNone(synth_heat(3, 128))
        self.assertIsNone(synth_heat(3, 128, .01, seed=29))

    def test_heat_invalid_inputs(self):
        for args in [(0, 128), (3, 0), (True, 5), (1, 2, 0), (1, 2, float('nan')), (1, 2, 2)]:
            with self.assertRaises(ValueError):
                synth_heat(*args)

    def test_priors_are_finite_nonnegative(self):
        for x in [True, float('nan'), float('inf'), -1]:
            self.assertFalse(_num(x))
        self.assertTrue(_num(0))

    def test_screen_writers_do_not_lose_concurrent_rows(self):
        tools = Path(__file__).resolve().parents[1] / 'tools'
        with tempfile.TemporaryDirectory() as d:
            path = str(Path(d) / 'screen.jsonl')
            script = "import sys; sys.path.insert(0,sys.argv[1]); import record_screen; record_screen.write([{'id':sys.argv[3]}],sys.argv[2],False)"
            procs = [subprocess.Popen([sys.executable, '-c', script, str(tools), path, str(i)]) for i in range(8)]
            self.assertEqual([p.wait(timeout=10) for p in procs], [0] * 8)
            rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
            self.assertEqual({r['id'] for r in rows}, {str(i) for i in range(8)})

    def test_patch_source_checks_survive_optimized_python(self):
        script = Path(__file__).resolve().parents[2] / 'b70/vllm-xpu/length-finish.py'
        with tempfile.TemporaryDirectory() as d:
            target = Path(d) / 'serving.py'
            target.write_text('# upstream changed\n')
            p = subprocess.run([sys.executable, '-O', str(script), str(target)],capture_output=True,text=True)
            self.assertNotEqual(p.returncode,0)
            self.assertEqual(target.read_text(),'# upstream changed\n')
