"""The last blocks of a MoE transformer GGUF plus its output head, in torch, built from the GGUF's own metadata.

The arithmetic follows llama.cpp (src/models/<arch>.cpp, llm_graph_context::build_moe_ffn, build_norm, build_rope,
build_attn), read from its source, not guessed:

  block       x = l_out of the block before (the residual stream, [T, n_embd])
              h = RMSNorm(x) * attn_norm
              q, k, v = h Wq (+ bq), h Wk (+ bk), h Wv (+ bv)        biases when the GGUF has them
              q/k RMSNorm: per head (qwen3moe: attn_q_norm [head_dim]) or over the whole projection before the head
                  split (olmoe: attn_q_norm [n_head * head_dim]); none when the GGUF has no attn_q_norm
              rope on the first n_rot dims of each head: NEOX (rotate halves; qwen2moe, qwen3moe, olmoe) or NORM
                  (adjacent pairs; llama / Mixtral), theta_i = pos * freq_base^(-2i / n_rot)
              causal softmax attention, GQA (q head h reads kv head h // (n_head / n_head_kv)), scale 1/sqrt(head_dim)
              x = x + attn Wo
              h = RMSNorm(x) * ffn_norm
              router: softmax over n_expert logits h Wr; top n_expert_used by probability; weights = those
                  probabilities, divided by their sum (clamped at 6.1e-5) when the arch normalizes (qwen3moe, llama;
                  not qwen2moe, olmoe), times expert_weights_scale when it is set
              each expert: down(silu(gate h) * up h); summed with the weights
              shared expert (qwen2moe): down(silu(gate h) * up h) * sigmoid(h . w_gate_inp_shexp)
              x = x + moe
  head        logits = RMSNorm(x) * output_norm @ output (token_embd when the model ties them)

Kernel arithmetic (match="q8"): llama.cpp's CPU backend multiplies a quantized weight with its activation rounded to
8 bits: blocks of 32 with an f16 scale for Q4_0..Q8_0 / IQ4_NL / MXFP4 weights (q8_0), blocks of 256 with an f32 scale
for the K-quants and the other IQ types (q8_K); an F16 weight takes an f16 activation, a BF16 weight a bf16 one;
attention reads f16 q/k/v (the f16 KV cache). In training these roundings pass the gradient straight through.
match="none" is plain f32. The gate (tail_trainer.py gate) says which one matches your engine.

LoRA: W x + B (A x) per target tensor (GGUF name), A [r, n_in], B [n_out, r]; the adapter GGUF holds NAME.lora_a = A
and NAME.lora_b = B with no alpha, which llama.cpp applies at --lora scale 1 exactly so. A target sees the raw
activation (llama.cpp's build_lora_mm: only the base matmul reads the rounded one).
"""
import math

import torch
import torch.nn.functional as F
import torch.utils.checkpoint

Q8_0_ACT = {"Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q8_1", "IQ4_NL", "MXFP4"}
EXACT = {"F32", "F16", "BF16"}


class Spec:
    """Everything the torch tail needs from the GGUF metadata (tail_trainer.py weights writes it as spec.json)."""

    KEYS = ("arch", "n_layer", "n_embd", "n_head", "n_head_kv", "head_dim", "n_rot", "rope_base", "rope", "eps",
            "n_expert", "n_expert_used", "norm_w", "w_scale", "n_vocab", "first")

    def __init__(self, **kw):
        for k in self.KEYS:
            setattr(self, k, kw[k])

    def to_json(self):
        return {k: getattr(self, k) for k in self.KEYS}

    @classmethod
    def from_json(cls, d):
        return cls(**{k: d[k] for k in cls.KEYS})


def ste(x, y):
    """y forward, gradient of x."""
    return x + (y - x).detach()


def q8_blocks(x, block, f16_scale):
    """llama.cpp's 8-bit view of an activation row: per block of `block` values, d = amax / 127, q = round(x / d).
    q8_0 stores d as f16 and rounds half away from zero; q8_K keeps d in f32 and rounds half to even."""
    shp = x.shape
    xb = x.reshape(*shp[:-1], shp[-1] // block, block)
    amax = xb.abs().amax(-1, keepdim=True)
    d = amax / 127
    safe = torch.where(amax > 0, d, torch.ones_like(d))
    if f16_scale:
        q = torch.sign(xb) * torch.floor(xb.abs() / safe + 0.5)
        d = d.half().float()
    else:
        q = torch.round(xb / safe)
    q = torch.where(amax > 0, q, torch.zeros_like(q))
    return (q * d).reshape(shp)


def round_like_engine(x, wtype, match):
    """The activation a CPU matmul with a `wtype` weight reads (match="q8"), or x itself (match="none")."""
    if match == "none" or wtype == "F32":
        return x
    if wtype == "F16":
        return ste(x, x.half().float())
    if wtype == "BF16":
        return ste(x, x.to(torch.bfloat16).float())
    if x.shape[-1] % 32:
        return x
    if wtype in Q8_0_ACT:
        return ste(x, q8_blocks(x, 32, True))
    if x.shape[-1] % 256:
        return x
    return ste(x, q8_blocks(x, 256, False))


class _Frozen(torch.autograd.Function):
    """x @ W^T for a frozen low-precision W, in f32. Autograd keeps W itself (no f32 copy) and rebuilds W.float() in
    the backward."""

    @staticmethod
    def forward(ctx, x, W):
        ctx.save_for_backward(W)
        return x @ W.to(x.device, non_blocking=True).float().t()

    @staticmethod
    def backward(ctx, g):
        (W,) = ctx.saved_tensors
        return g @ W.to(g.device, non_blocking=True).float(), None


def frozen_mm(x, W):
    """W may live in pinned host memory (offloaded experts): it is copied to x's device for each use only."""
    if torch.is_grad_enabled() and x.requires_grad:
        return _Frozen.apply(x, W)
    return x @ W.to(x.device, non_blocking=True).float().t()


def has_fp64(device):
    """Some Intel GPUs have no native f64: there the rope angles are computed on the CPU."""
    if torch.device(device).type == "xpu":
        return bool(getattr(torch.xpu.get_device_properties(torch.device(device)), "has_fp64", False))
    return True


def pin(t, notes):
    """Page-locked host memory for an offloaded expert stack (fast async copies); pageable when the backend cannot."""
    try:
        return t.pin_memory()
    except (RuntimeError, NotImplementedError) as e:
        notes.setdefault("pin", f"pageable host memory: {type(e).__name__}: {str(e)[:120]}")
        return t


class Tail(torch.nn.Module):
    """Blocks spec.first..n_layer-1 and the head. W: {GGUF tensor name: numpy array} (numpy order: [n_out, n_in],
    experts [n_expert, n_out, n_in]); types: {name: GGUF type name}. lora: target names to train (rank `rank`);
    frozen adapters are added with add_adapter. offload_experts: the routed expert stacks stay in host memory."""

    def __init__(self, spec, W, types, device="cpu", match="none", lora=(), rank=8, offload_experts=False,
                 store=torch.float16, seed=20261115, loaded=None):
        super().__init__()
        self.spec, self.device, self.match, self.types = spec, device, match, types
        self.notes = {}
        self.routes = None
        self.f64 = has_fp64(device)
        self.w = {}
        for k, v in W.items():
            t = torch.from_numpy(v)
            if t.dtype == torch.float32 and t.numel() > 1 << 20 and k not in ("output.weight", "token_embd.weight"):
                t = t.to(torch.bfloat16) if types.get(k) == "BF16" else t.to(store)
            if offload_experts and "_exps." in k and torch.device(device).type != "cpu":
                self.w[k] = pin(t.contiguous(), self.notes)
            else:
                self.w[k] = t.to(device)
            if loaded is not None:
                loaded(k)
        self.head_name = "output.weight" if "output.weight" in self.w else "token_embd.weight"
        if self.w[self.head_name].dtype != torch.float32:
            self.w[self.head_name] = self.w[self.head_name].float()   # the head is read per chunk: keep one f32 copy
        self.lora = torch.nn.ParameterDict()
        g = torch.Generator().manual_seed(seed)
        for name in lora:
            n_out, n_in = W[name].shape
            self.lora[key(name) + "_A"] = torch.nn.Parameter((torch.randn(rank, n_in, generator=g) / math.sqrt(n_in)).to(device))
            self.lora[key(name) + "_B"] = torch.nn.Parameter(torch.zeros(n_out, rank, device=device))
        self.frozen = {}          # name -> [(A, B)] of loaded adapters, summed
        base = spec.rope_base
        self.inv_freq = base ** (-torch.arange(0, spec.n_rot, 2, dtype=torch.float64,
                                               device=device if self.f64 else "cpu") / spec.n_rot)

    def add_adapter(self, state):
        """state: {name: (A [r, n_in], B [n_out, r])} (numpy or torch), applied at scale 1 beside any others."""
        for name, (A, B) in state.items():
            if name not in self.w:
                raise ValueError(f"adapter tensor {name} is not in this tail")
            A, B = (torch.as_tensor(x).float().to(self.device) for x in (A, B))
            self.frozen.setdefault(name, []).append((A, B))

    # ------------------------------------------------------------------------------------------------ arithmetic
    def has(self, name):
        return name in self.w

    def mm(self, name, x):
        y = frozen_mm(round_like_engine(x, self.types.get(name, "F32"), self.match), self.w[name])
        b = name[:-len("weight")] + "bias"
        if b in self.w:
            y = y + self.w[b].float()
        k = key(name)
        if k + "_A" in self.lora:
            y = y + (x @ self.lora[k + "_A"].t()) @ self.lora[k + "_B"].t()
        for A, B in self.frozen.get(name, ()):
            y = y + (x @ A.t()) @ B.t()
        return y

    def vec(self, name):
        return self.w[name].float()

    def rms(self, x, w=None):
        y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.spec.eps)
        return y * w if w is not None else y

    def f16(self, x):
        return ste(x, x.half().float()) if self.match != "none" else x

    def rope(self, x, pos):
        """x [T, H, D]: rotate the first n_rot dims, NEOX (halves) or NORM (adjacent pairs)."""
        n_rot = self.spec.n_rot
        ang = (pos if self.f64 else pos.cpu()).double()[:, None] * self.inv_freq[None, :]
        cos = ang.cos().float().to(x.device)[:, None, :]
        sin = ang.sin().float().to(x.device)[:, None, :]
        rot, rest = x[..., :n_rot], x[..., n_rot:]
        if self.spec.rope == "neox":
            h = n_rot // 2
            x0, x1 = rot[..., :h], rot[..., h:]
            out = torch.cat([x0 * cos - x1 * sin, x0 * sin + x1 * cos], -1)
        else:
            x0, x1 = rot[..., 0::2], rot[..., 1::2]
            out = torch.stack([x0 * cos - x1 * sin, x0 * sin + x1 * cos], -1).flatten(-2)
        return torch.cat([out, rest], -1)

    # ------------------------------------------------------------------------------------------------ pieces
    def attn(self, il, x, pos, q_rows, chunk=256):
        """x [T, n_embd] (normed block input) -> attention output rows q_rows, through Wo: [len(q_rows), n_embd]."""
        s, p = self.spec, f"blk.{il}."
        T, H, Hk, D = x.shape[0], s.n_head, s.n_head_kv, s.head_dim
        k = self.mm(p + "attn_k.weight", x)
        v = self.mm(p + "attn_v.weight", x)
        k = self.qk_norm(p + "attn_k_norm.weight", k, Hk)
        k = self.f16(self.rope(k, pos))
        v = self.f16(v.reshape(T, Hk, D))
        kk = k.repeat_interleave(H // Hk, dim=1)
        vv = v.repeat_interleave(H // Hk, dim=1)
        outs = []
        for a in range(0, len(q_rows), chunk):
            r = q_rows[a:a + chunk]
            q = self.mm(p + "attn_q.weight", x[r])
            q = self.f16(self.rope(self.qk_norm(p + "attn_q_norm.weight", q, H), pos[r]))
            sc = torch.einsum("rhd,thd->hrt", q, kk) / math.sqrt(D)
            causal = pos[None, :] <= pos[r][:, None]                      # [R, T], a comparison (no mask indexing)
            sc = sc.masked_fill(~causal[None], float("-inf"))
            outs.append(torch.einsum("hrt,thd->rhd", torch.softmax(sc, -1), vv).reshape(len(r), H * D))
        return self.mm(p + "attn_output.weight", torch.cat(outs))

    def qk_norm(self, name, y, n_heads):
        """y [R, n_heads * D] -> [R, n_heads, D], RMS-normed per head or over the whole row, as the GGUF's norm says."""
        D = self.spec.head_dim
        if name not in self.w:
            return y.reshape(-1, n_heads, D)
        w = self.vec(name)
        if w.numel() == D:
            return self.rms(y.reshape(-1, n_heads, D), w)
        return self.rms(y, w).reshape(-1, n_heads, D)

    def moe(self, il, h):
        s, p = self.spec, f"blk.{il}."
        probs = torch.softmax(self.mm(p + "ffn_gate_inp.weight", h), -1)
        # a checkpoint's recompute routes as its forward did: atomic sums (rocBLAS split-K, oneDNN, index_add_) can
        # differ in the last bits between the two passes and flip a near-tied expert
        ex = self.routes.get(il) if self.routes is not None else None
        if ex is None:
            ex = probs.detach().topk(s.n_expert_used, dim=-1).indices
            if self.routes is not None:
                self.routes[il] = ex
        wt = probs.gather(-1, ex)
        if s.norm_w:
            wt = wt / wt.sum(-1, keepdim=True).clamp(min=6.103515625e-5)
        if s.w_scale not in (0.0, 1.0):
            wt = wt * s.w_scale
        Wg, Wu, Wd = (self.w[p + f"ffn_{n}_exps.weight"] for n in ("gate", "up", "down"))
        xg = round_like_engine(h, self.types.get(p + "ffn_gate_exps.weight", "F32"), self.match)
        flat = ex.reshape(-1)
        order = torch.argsort(flat)
        counts = torch.bincount(flat, minlength=Wg.shape[0]).tolist()
        tok = order // s.n_expert_used
        slot_w = wt.reshape(-1)[order]
        out = torch.zeros_like(h)
        at = 0
        dtype = self.types.get(p + "ffn_down_exps.weight", "F32")
        for e, n in enumerate(counts):
            if n == 0:
                continue
            t = tok[at:at + n]
            xe = xg[t]
            a = F.silu(frozen_mm(xe, Wg[e])) * frozen_mm(xe, Wu[e])
            a = round_like_engine(a, dtype, self.match)
            out = out.index_add(0, t, frozen_mm(a, Wd[e]) * slot_w[at:at + n, None])
            at += n
        if self.has(p + "ffn_up_shexp.weight"):
            sh = self.mm(p + "ffn_down_shexp.weight",
                         F.silu(self.mm(p + "ffn_gate_shexp.weight", h)) * self.mm(p + "ffn_up_shexp.weight", h))
            if self.has(p + "ffn_gate_inp_shexp.weight"):
                gw = self.vec(p + "ffn_gate_inp_shexp.weight").reshape(-1)
                sh = sh * torch.sigmoid(h @ gw)[:, None]
            out = out + sh
        return out

    def block(self, il, x, pos, rows=None):
        """One block on x [T, n_embd]. rows: the output rows to keep (the last block only needs those)."""
        p = f"blk.{il}."
        q_rows = rows if rows is not None else torch.arange(x.shape[0], device=x.device)
        a = self.attn(il, self.rms(x, self.vec(p + "attn_norm.weight")), pos, q_rows)
        x = (x[rows] if rows is not None else x) + a
        return x + self.moe(il, self.rms(x, self.vec(p + "ffn_norm.weight")))

    def hidden(self, x, rows=None, checkpoint=False):
        """x [T, n_embd], the input of block spec.first -> the normed head input at rows (all rows when None)."""
        x = x.to(self.device, torch.float32)
        self.routes = {} if checkpoint else None
        pos = torch.arange(x.shape[0], device=self.device)
        rows = None if rows is None else torch.as_tensor(rows, device=self.device)
        last = self.spec.n_layer - 1
        for il in range(self.spec.first, self.spec.n_layer):
            r = rows if il == last else None
            if checkpoint and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(self.block, il, x, pos, r, use_reentrant=False)
            else:
                x = self.block(il, x, pos, r)
        return self.rms(x, self.vec("output_norm.weight"))

    def logits(self, hn):
        return self.mm(self.head_name, hn)

    def forward(self, x, rows):
        """Log-probabilities at rows [len(rows), n_vocab]."""
        return torch.log_softmax(self.logits(self.hidden(x, rows)), -1)


def key(name):
    return name.replace(".", "_")


def default_targets(spec, names, which="attn"):
    """LoRA targets of the tail: attn_q/k/v/output of each block (which="attn"), plus the shared expert when the
    model has one (which="attn+shexp"), or only attn_output (which="o")."""
    want = {"attn": ("attn_q", "attn_k", "attn_v", "attn_output"), "o": ("attn_output",),
            "attn+shexp": ("attn_q", "attn_k", "attn_v", "attn_output", "ffn_gate_shexp", "ffn_up_shexp",
                           "ffn_down_shexp")}[which]
    out = []
    for il in range(spec.first, spec.n_layer):
        for n in want:
            t = f"blk.{il}.{n}.weight"
            if t in names:
                out.append(t)
    return out
