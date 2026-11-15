"""Tests of tail_trainer.py and tail_arch.py on synthetic data (a tiny random qwen3moe GGUF written here).

    python3 -m unittest -q test_tail_trainer        (needs numpy, gguf and torch; CPU torch is enough)

The torch tail is checked against a token-by-token reference written separately below, from the same llama.cpp
arithmetic (tail_arch.py docstring), so a wrong GQA mapping, mask, rope or routing shows as a mismatch.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import tail_trainer as tt  # noqa: E402

try:
    import torch
    import tail_arch as ta
except ImportError:          # the GGUF and dump parts still run
    torch = ta = None
try:
    import gguf
except ImportError:
    gguf = None

E, H, HK, D, NE, NU, FF, V, NL = 64, 4, 2, 16, 8, 2, 32, 50, 3


def tiny_gguf(path, arch="qwen3moe", seed=0, expert_count=NE, tied=False):
    g = np.random.default_rng(seed)
    w = gguf.GGUFWriter(path, arch=arch)
    w.add_block_count(NL)
    w.add_embedding_length(E)
    w.add_head_count(H)
    w.add_head_count_kv(HK)
    w.add_key_length(D)
    w.add_value_length(D)
    w.add_rope_dimension_count(D)
    w.add_rope_freq_base(10000.0)
    w.add_layer_norm_rms_eps(1e-6)
    w.add_expert_count(expert_count)
    w.add_expert_used_count(NU)
    w.add_token_list([f"t{i}" for i in range(V)])
    r = lambda *s: (g.standard_normal(s) * 0.3).astype(np.float32)
    w.add_tensor("token_embd.weight", r(V, E))
    w.add_tensor("output_norm.weight", (1 + 0.1 * g.standard_normal(E)).astype(np.float32))
    if not tied:
        w.add_tensor("output.weight", r(V, E))
    for il in range(NL):
        p = f"blk.{il}."
        w.add_tensor(p + "attn_norm.weight", (1 + 0.1 * g.standard_normal(E)).astype(np.float32))
        w.add_tensor(p + "attn_q.weight", r(H * D, E))
        w.add_tensor(p + "attn_k.weight", r(HK * D, E))
        w.add_tensor(p + "attn_v.weight", r(HK * D, E))
        w.add_tensor(p + "attn_output.weight", r(E, H * D))
        w.add_tensor(p + "attn_q_norm.weight", (1 + 0.1 * g.standard_normal(D)).astype(np.float32))
        w.add_tensor(p + "attn_k_norm.weight", (1 + 0.1 * g.standard_normal(D)).astype(np.float32))
        w.add_tensor(p + "ffn_norm.weight", (1 + 0.1 * g.standard_normal(E)).astype(np.float32))
        w.add_tensor(p + "ffn_gate_inp.weight", r(expert_count, E))
        w.add_tensor(p + "ffn_gate_exps.weight", r(expert_count, FF, E))
        w.add_tensor(p + "ffn_up_exps.weight", r(expert_count, FF, E))
        w.add_tensor(p + "ffn_down_exps.weight", r(expert_count, E, FF))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def tiny_dump(d, first, n_seq=3, T=40, seed=1):
    """A tail-dump directory with random residuals and engine logits at 4 positions per window."""
    g = np.random.default_rng(seed)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "index.tsv"), "w") as idx:
        idx.write("seq\tsource\tstart\ttokens\n")
        for i in range(n_seq):
            s = f"seq-{i:05d}"
            os.makedirs(os.path.join(d, s), exist_ok=True)
            g.integers(0, V, T).astype(np.int32).tofile(os.path.join(d, s, "tokens.i32"))
            g.standard_normal((T, E)).astype(np.float32).tofile(os.path.join(d, s, f"l_out-{first - 1}.f32"))
            pos = np.array([T // 2, T // 2 + 5, T - 2, T - 1], np.int32)
            pos.tofile(os.path.join(d, s, "logit_pos.i32"))
            g.standard_normal((4, V)).astype(np.float32).tofile(os.path.join(d, s, "logits.f32"))
            with open(os.path.join(d, s, "meta.tsv"), "w") as m:
                m.write(f"tokens\t{T}\nlogit_rows\t4\nn_vocab\t{V}\nadapters\t0\nl_out-{first - 1}\t{T}\t{E}\n")
            idx.write(f"{s}\tx.txt\t{i * T}\t{T}\n")


def reference(W, spec, x, first):
    """Token by token, f64, straight from the llama.cpp arithmetic (no batching, no shared code with tail_arch)."""
    t = lambda n: torch.from_numpy(np.asarray(W[n])).double()
    T = x.shape[0]
    x = x.double()
    rms = lambda v, w: v / torch.sqrt((v * v).mean(-1, keepdim=True) + spec["eps"]) * w
    inv = spec["rope_base"] ** (-torch.arange(0, D, 2, dtype=torch.float64) / D)

    def rope(v, p):            # v [heads, D], NEOX
        a = p * inv
        c, s = a.cos(), a.sin()
        h = D // 2
        return torch.cat([v[:, :h] * c - v[:, h:] * s, v[:, :h] * s + v[:, h:] * c], -1)

    for il in range(first, NL):
        p = f"blk.{il}."
        hs = [rms(x[i], t(p + "attn_norm.weight")) for i in range(T)]
        ks = [rope(rms((t(p + "attn_k.weight") @ hs[j]).reshape(HK, D), t(p + "attn_k_norm.weight")), j) for j in range(T)]
        vs = [(t(p + "attn_v.weight") @ hs[j]).reshape(HK, D) for j in range(T)]
        new = []
        for i in range(T):
            q = rope(rms((t(p + "attn_q.weight") @ hs[i]).reshape(H, D), t(p + "attn_q_norm.weight")), i)
            o = []
            for h in range(H):
                kh = h // (H // HK)
                sc = torch.stack([q[h] @ ks[j][kh] for j in range(i + 1)]) / math.sqrt(D)
                pr = torch.softmax(sc, 0)
                o.append(sum(pr[j] * vs[j][kh] for j in range(i + 1)))
            xa = x[i] + t(p + "attn_output.weight") @ torch.cat(o)
            hf = rms(xa, t(p + "ffn_norm.weight"))
            pr = torch.softmax(t(p + "ffn_gate_inp.weight") @ hf, 0)
            top = torch.topk(pr, NU).indices
            wt = pr[top] / pr[top].sum()
            moe = 0
            for e, wgt in zip(top.tolist(), wt):
                g = t(p + "ffn_gate_exps.weight")[e] @ hf
                u = t(p + "ffn_up_exps.weight")[e] @ hf
                moe = moe + wgt * (t(p + "ffn_down_exps.weight")[e] @ (torch.nn.functional.silu(g) * u))
            new.append(xa + moe)
        x = torch.stack(new)
    hn = torch.stack([rms(x[i], t("output_norm.weight")) for i in range(T)])
    head = t("output.weight") if "output.weight" in W else t("token_embd.weight")
    return torch.log_softmax(hn @ head.t(), -1)


@unittest.skipIf(gguf is None, "gguf-py not installed")
class Gguf(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.path = os.path.join(self.d, "m.gguf")
        tiny_gguf(self.path)

    def test_spec_reads_the_metadata(self):
        s = tt.spec_of(gguf.GGUFReader(self.path), 1)
        self.assertEqual((s["arch"], s["n_layer"], s["n_embd"], s["n_head"], s["n_head_kv"], s["head_dim"]),
                         ("qwen3moe", NL, E, H, HK, D))
        self.assertEqual((s["n_expert"], s["n_expert_used"], s["rope"], s["norm_w"], s["n_vocab"]), (NE, NU, "neox", True, V))

    def test_spec_refuses_a_dense_model_an_unknown_arch_and_a_bad_first(self):
        p = os.path.join(self.d, "dense.gguf")
        tiny_gguf(p, expert_count=0)
        with self.assertRaisesRegex(SystemExit, "dense"):
            tt.spec_of(gguf.GGUFReader(p), 1)
        p = os.path.join(self.d, "x.gguf")
        tiny_gguf(p, arch="mamba")
        with self.assertRaisesRegex(SystemExit, "not implemented"):
            tt.spec_of(gguf.GGUFReader(p), 1)
        with self.assertRaisesRegex(SystemExit, "--first"):
            tt.spec_of(gguf.GGUFReader(self.path), 0)

    def test_weights_writes_the_tail_only_in_numpy_order(self):
        out = os.path.join(self.d, "w")
        tt.main(["weights", self.path, "--first", "1", "--out", out])
        idx = json.load(open(os.path.join(out, "index.json")))
        self.assertNotIn("blk.0.attn_q.weight", idx)
        self.assertNotIn("token_embd.weight", idx)
        self.assertEqual(idx["blk.2.ffn_down_exps.weight"]["shape"], [NE, E, FF])
        self.assertEqual(np.load(os.path.join(out, "blk.1.attn_q.weight.npy")).shape, (H * D, E))
        ref = gguf.GGUFReader(self.path)
        t = next(t for t in ref.tensors if t.name == "blk.1.attn_q.weight")
        np.testing.assert_array_equal(np.load(os.path.join(out, "blk.1.attn_q.weight.npy")), np.asarray(t.data))
        self.assertEqual(json.load(open(os.path.join(out, "spec.json")))["first"], 1)

    def test_a_tied_head_reads_token_embd(self):
        p = os.path.join(self.d, "tied.gguf")
        tiny_gguf(p, tied=True)
        out = os.path.join(self.d, "wt")
        tt.main(["weights", p, "--first", "2", "--out", out])
        self.assertIn("token_embd.weight", json.load(open(os.path.join(out, "index.json"))))

    def test_adapter_round_trip_and_llama_cpp_shapes(self):
        g = np.random.default_rng(3)
        st = {"blk.2.attn_output.weight": (g.standard_normal((4, H * D)).astype(np.float32),
                                           g.standard_normal((E, 4)).astype(np.float32))}
        p = os.path.join(self.d, "a.gguf")
        tt.write_adapter(st, p, "qwen3moe")
        r = gguf.GGUFReader(p)
        self.assertEqual(tt.field(r, "general.type"), "adapter")
        self.assertEqual(tt.field(r, "adapter.type"), "lora")
        self.assertEqual(tt.field(r, "general.architecture"), "qwen3moe")
        ne = {t.name: [int(x) for x in t.shape] for t in r.tensors}
        # llama.cpp: lora_a ne = (n_in, r), lora_b ne = (r, n_out)  (llama-adapter.cpp shape checks)
        self.assertEqual(ne["blk.2.attn_output.weight.lora_a"], [H * D, 4])
        self.assertEqual(ne["blk.2.attn_output.weight.lora_b"], [4, E])
        back = tt.read_adapter(p)
        np.testing.assert_array_equal(back["blk.2.attn_output.weight"][0], st["blk.2.attn_output.weight"][0])
        np.testing.assert_array_equal(back["blk.2.attn_output.weight"][1], st["blk.2.attn_output.weight"][1])


class DumpFiles(unittest.TestCase):
    def test_dump_reads_windows_residuals_and_logit_rows(self):
        d = tempfile.mkdtemp()
        tiny_dump(d, 2, n_seq=2, T=30)
        dm = tt.Dump(d)
        self.assertEqual(dm.seqs, ["seq-00000", "seq-00001"])
        self.assertEqual(dm.resid("seq-00001", 2).shape, (30, E))
        lg, pos = dm.logits("seq-00000")
        self.assertEqual((lg.shape, pos.tolist()), ((4, V), [15, 20, 28, 29]))
        with self.assertRaisesRegex(SystemExit, "--layers 1"):
            dm.resid("seq-00000", 1)

    def test_compare_is_zero_for_equal_rows_and_counts_top1(self):
        a = tt.log_softmax_np(np.random.default_rng(0).standard_normal((5, 30)))
        m = tt.compare(a, a)
        self.assertEqual((m["top1"], m["kl_mean"], m["top20_abs_mean"]), (1.0, 0.0, 0.0))
        b = a.copy()
        b[0] = a[0][::-1]
        self.assertLess(tt.compare(a, b)["top1"], 1.0)

    def test_rocm_is_cuda(self):
        self.assertEqual(tt.norm_device("rocm"), "cuda")
        self.assertEqual(tt.norm_device("rocm:1"), "cuda:1")
        self.assertEqual(tt.norm_device("xpu"), "xpu")


@unittest.skipIf(torch is None or gguf is None, "torch or gguf-py not installed")
class TorchTail(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = tempfile.mkdtemp()
        cls.path = os.path.join(cls.d, "m.gguf")
        tiny_gguf(cls.path)
        cls.w = os.path.join(cls.d, "w")
        tt.main(["weights", cls.path, "--first", "1", "--out", cls.w])
        cls.spec = json.load(open(os.path.join(cls.w, "spec.json")))
        idx = json.load(open(os.path.join(cls.w, "index.json")))
        cls.W = {k: np.load(os.path.join(cls.w, k + ".npy")) for k in idx}
        cls.types = {k: v["type"] for k, v in idx.items()}

    def tail(self, **kw):
        return ta.Tail(ta.Spec.from_json(self.spec), self.W, self.types, device="cpu", store=torch.float32, **kw)

    def test_forward_matches_the_token_by_token_reference(self):
        x = torch.from_numpy(np.random.default_rng(5).standard_normal((12, E)).astype(np.float32))
        rows = [3, 7, 11]
        with torch.no_grad():
            got = self.tail()(x, rows).double()
        ref = reference(self.W, self.spec, x, 1)[rows]
        self.assertLess(float((got - ref).abs().max()), 1e-4)

    def test_lora_with_b_zero_is_the_base_and_a_frozen_adapter_adds(self):
        x = torch.from_numpy(np.random.default_rng(6).standard_normal((9, E)).astype(np.float32))
        base = self.tail()
        t = ta.default_targets(base.spec, set(self.W), "attn")
        self.assertEqual(t[:4], ["blk.1.attn_q.weight", "blk.1.attn_k.weight", "blk.1.attn_v.weight",
                                 "blk.1.attn_output.weight"])
        tr = self.tail(lora=t)
        with torch.no_grad():
            self.assertTrue(torch.equal(base(x, [8]), tr(x, [8])))
            g = torch.Generator().manual_seed(1)
            for k, p in tr.lora.items():
                if k.endswith("_B"):
                    p.copy_(torch.randn(p.shape, generator=g) * 0.05)
            edited = tr(x, [4, 8])
            st = {n: (tr.lora[ta.key(n) + "_A"].numpy(), tr.lora[ta.key(n) + "_B"].numpy()) for n in t}
            fz = self.tail()
            fz.add_adapter(st)
            self.assertLess(float((fz(x, [4, 8]) - edited).abs().max()), 1e-5)
            self.assertGreater(float((edited - base(x, [4, 8])).abs().max()), 1e-3)

    def test_q8_rounding_matches_the_engine_formula(self):
        x = torch.tensor([[0.5, -1.27, 0.0, 0.3] * 8])
        y = ta.q8_blocks(x, 32, True)
        d = torch.tensor(1.27 / 127).half().float()
        self.assertTrue(torch.allclose(y, torch.round(x / (1.27 / 127)) * d))
        z = torch.zeros(1, 256)
        self.assertTrue(torch.equal(ta.q8_blocks(z, 256, False), z))
        self.assertEqual(ta.round_like_engine(x, "F32", "q8") is x, True)

    def test_rope_norm_rotates_adjacent_pairs(self):
        s = ta.Spec.from_json(dict(self.spec, rope="norm"))
        t = ta.Tail(s, self.W, self.types, store=torch.float32)
        v = torch.zeros(1, 1, D)
        v[0, 0, 0] = 1.0
        out = t.rope(v, torch.tensor([1]))
        self.assertAlmostEqual(float(out[0, 0, 0]), math.cos(1.0), places=6)
        self.assertAlmostEqual(float(out[0, 0, 1]), math.sin(1.0), places=6)
        n = ta.Spec.from_json(self.spec)
        out = ta.Tail(n, self.W, self.types, store=torch.float32).rope(v, torch.tensor([1]))
        self.assertAlmostEqual(float(out[0, 0, D // 2]), math.sin(1.0), places=6)

    def test_checkpointed_training_runs_and_gate_eval_read_the_adapter(self):
        dump = os.path.join(self.d, "dump")
        tiny_dump(dump, 1)
        run = os.path.join(self.d, "run")
        rc = tt.main(["train", "--dump", dump, "--weights", self.w, "--out", run, "--device", "cpu", "--steps", "6",
                      "--lr", "1e-2", "--match", "none", "--blocks", "2-2"])
        self.assertEqual(rc, 0)
        r = json.load(open(os.path.join(run, "train.json")))
        self.assertTrue(all(n.startswith("blk.2.") for n in r["targets"]))
        self.assertEqual(r["steps"], 6)
        st = tt.read_adapter(os.path.join(run, "adapter.gguf"))
        self.assertEqual(sorted(st), sorted(r["targets"]))
        self.assertGreater(max(float(np.abs(b).max()) for _, b in st.values()), 0)
        e0, e1 = os.path.join(self.d, "e0.json"), os.path.join(self.d, "e1.json")
        tt.main(["eval", "--dump", dump, "--weights", self.w, "--out", e0, "--device", "cpu", "--match", "none"])
        tt.main(["eval", "--dump", dump, "--weights", self.w, "--out", e1, "--device", "cpu", "--match", "none",
                 "--lora", os.path.join(run, "adapter.gguf")])
        a, b = json.load(open(e0)), json.load(open(e1))
        self.assertLess(b["torch_ce_all_positions"], a["torch_ce_all_positions"])   # trained on these windows
        self.assertEqual(a["engine_ce_logit_rows"], b["engine_ce_logit_rows"])      # the dump's own logits
        g = os.path.join(self.d, "g.json")
        rc = tt.main(["gate", "--dump", dump, "--weights", self.w, "--out", g, "--device", "cpu", "--seqs", "2"])
        self.assertEqual(rc, 3)                         # random engine logits: the gate must fail
        self.assertFalse(json.load(open(g))["pass"])


class Source(unittest.TestCase):
    def test_gate_thresholds_are_the_preregistered_ones(self):
        self.assertEqual(tt.GATE, {"top1": 0.97, "kl": 0.02})

    def test_cli_help_runs_without_torch_imports(self):
        out = subprocess.run([sys.executable, os.path.join(HERE, "tail_trainer.py"), "--help"], capture_output=True,
                             text=True)
        self.assertEqual(out.returncode, 0)
        self.assertIn("gate", out.stdout)

    def test_dump_tool_uses_only_the_public_api(self):
        src = open(os.path.join(HERE, "dump", "tail-dump.cpp")).read()
        self.assertNotIn('#include "common.h"', src)
        self.assertIn("llama_set_adapters_lora", src)


if __name__ == "__main__":
    unittest.main()
