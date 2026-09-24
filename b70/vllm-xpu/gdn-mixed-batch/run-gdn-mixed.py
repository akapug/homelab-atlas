"""Run the mixed spec/non-spec GDN differential from vllm-project/vllm-xpu-kernels#552
(tests/gdn_attn/test_gdn_attn_mixed.py at 84990fd, by CySpiegel; adapted copy beside this file)
against an installed vllm_xpu_kernels, without pytest (not in the llm-scaler image). Every case
runs, in both token layouts when the test supports them; prints PASS/FAIL per case and a summary.
Usage: python3 run-gdn-mixed.py test_gdn_attn_mixed_adapted.py"""
import importlib.util
import itertools
import sys
import traceback
import types

import torch

fake = types.ModuleType("pytest")
fake.mark = types.SimpleNamespace(parametrize=lambda *a, **k: (lambda f: f))
fake.skip = lambda msg: (_ for _ in ()).throw(RuntimeError("skip: " + msg))
sys.modules["pytest"] = fake

spec = importlib.util.spec_from_file_location("t", sys.argv[1])
t = importlib.util.module_from_spec(spec)
spec.loader.exec_module(t)

fails = 0
layouts = ["shuffled", "per-request"] if "layout" in t.test_gdn_attention_mixed_batch.__code__.co_varnames else ["shuffled"]
cases = list(itertools.product([(1, 2), (2, 0), (0, 3)], [(2, 2), (1, 3)], [True, False], layouts))
for (npf, nd), (nsd, nst), reorder, layout in cases:
    name = f"layout={layout} prefills={npf} decodes={nd} spec_decodes={nsd} spec_tokens={nst} reorder={reorder}"
    try:
        kw = {"layout": layout} if len(layouts) > 1 else {}
        t.test_gdn_attention_mixed_batch(npf, nd, nsd, nst, torch.bfloat16, reorder, **kw)
        print("PASS", name, flush=True)
    except AssertionError as e:
        fails += 1
        print("FAIL", name, "--", e, flush=True)
    except Exception as e:
        fails += 1
        print("ERROR", name, "--", type(e).__name__, str(e)[:300], flush=True)
        traceback.print_exc(limit=2)
print(f"{len(cases) - fails} of {len(cases)} cases bitwise identical", flush=True)
