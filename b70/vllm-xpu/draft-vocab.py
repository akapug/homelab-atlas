"""A reduced vocabulary for the Qwen3.5-family MTP drafter (qwen3_5_mtp.py in llm-scaler 0.26.0-b2).

Each draft token runs the drafter's lm_head over the whole vocabulary, and the MTP head shares the
target's unquantized lm_head: for Qwen3.8-27B that is 248,320 x 5,120 fp16, 2.5 GB read per draft
token, ~4.3 ms of the ~5.5 ms each extra draft costs on a B70 (../bench/spec-step-cost.sh:
61.6 ms a step at 5 drafts against 50.7 at 3). With VLLM_DRAFT_VOCAB naming a file of token ids
(one per line), the drafter scores only those rows (a copy of them is taken on first use) and gives
every other token -inf, so it can only propose tokens in the list. The target still verifies with
its full lm_head, so the output is unchanged; a token outside the list is a rejected draft, which
costs acceptance, not correctness. Unset, nothing changes (the image is shared with other models).
Writes a .orig backup once and always patches from it.
  python3 draft-vocab.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/qwen3_5_mtp.py
(inside the image), then serve with VLLM_DRAFT_VOCAB=<list file> in the environment."""
import os
import shutil
import sys

p = sys.argv[1]
if not os.path.exists(p + ".orig"):
    shutil.copy2(p, p + ".orig")
s = open(p + ".orig").read()
old = """    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)
"""
new = """    def compute_logits(
        self,
        hidden_states: torch.Tensor,
        spec_step_idx: int = 0,
    ) -> torch.Tensor | None:
        # draft-vocab patch (homelab-atlas b70/vllm-xpu/draft-vocab.py): score only VLLM_DRAFT_VOCAB's ids
        ids = self._draft_vocab_ids(hidden_states.device)
        if ids is None:
            return self.logits_processor(self.lm_head, hidden_states)
        w = getattr(self, "_draft_vocab_w", None)
        if w is None:
            w = self._draft_vocab_w = self.lm_head.weight.index_select(0, ids).contiguous()
        sub = hidden_states.to(w.dtype) @ w.t()
        out = sub.new_full((sub.shape[0], self.config.vocab_size), float("-inf"))
        out[:, ids] = sub
        return out

    def get_top_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if self._draft_vocab_ids(hidden_states.device) is None:
            return super().get_top_tokens(hidden_states)
        return self.compute_logits(hidden_states).argmax(dim=-1)

    def _draft_vocab_ids(self, device):
        if not hasattr(self, "_draft_vocab"):
            f = os.environ.get("VLLM_DRAFT_VOCAB")
            self._draft_vocab = None
            if f:
                ids = sorted({int(x) for x in open(f).read().split()})
                self._draft_vocab = torch.tensor(ids, dtype=torch.long, device=device)
                logger.info("draft vocab: %d of %d tokens from %s", len(ids), self.config.vocab_size, f)
        return self._draft_vocab
"""
assert s.count(old) == 1, "compute_logits not found as expected"
s = s.replace(old, new)
assert s.count("\nfrom collections.abc import Iterable\n") == 1 and "\nlogger = init_logger(__name__)\n" in s
s = s.replace("\nfrom collections.abc import Iterable\n", "\nimport os\nfrom collections.abc import Iterable\n", 1)
open(p, "w").write(s)
print("patched", p)
