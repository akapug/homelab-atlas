"""Build a draft-vocabulary list for draft-vocab.py from token counts over a text corpus.

The drafter can only propose tokens in the list, so the list should hold every token the model is
likely to write. It is the union of: every token seen in the corpus (source trees, docs, logs), ids
below K (BPE merges are roughly in frequency order, so low ids are the common ones), and the
tokenizer's added (special) tokens. Coverage is measured on held-out text the model itself wrote
(spec-differential.py outputs, as token ids), since a token missing from the list is a lost draft.
Run it with the image's Python, which has `tokenizers`:
  draft-vocab-build.py <tokenizer.json> <K> <out.txt> <heldout.json ...> -- <corpus dir or file ...>
Prints, for several K, the list size and the held-out share it covers; writes the list for <K>."""
import json
import os
import sys

from tokenizers import Tokenizer

EXT = (".py", ".sh", ".md", ".txt", ".json", ".jsonl", ".toml", ".yaml", ".yml", ".rs", ".ts", ".js",
       ".go", ".c", ".h", ".patch", ".diff", ".log", ".service", ".jinja", ".cfg", ".ini", ".html", ".css")
tok_path, k_out, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
cut = sys.argv.index("--")
heldout = [i for f in sys.argv[4:cut] for r in json.load(open(f))["rows"] for i in r["ids"]]
tok = Tokenizer.from_file(tok_path)
spec = {a["id"] for a in json.load(open(tok_path)).get("added_tokens", [])}


def files(paths):
    for p in paths:
        if os.path.isfile(p):
            yield p
        for root, dirs, names in os.walk(p):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("node_modules", "__pycache__", "target")]
            yield from (os.path.join(root, n) for n in names if n.endswith(EXT))


seen, n_files, n_tok = set(), 0, 0
batch = []
for f in files(sys.argv[cut + 1:]):
    try:
        batch.append(open(f, errors="replace").read(200_000))
    except OSError:
        continue
    if len(batch) == 256:
        for e in tok.encode_batch(batch, add_special_tokens=False):
            seen.update(e.ids); n_tok += len(e.ids)
        n_files += len(batch); batch = []
for e in tok.encode_batch(batch, add_special_tokens=False):
    seen.update(e.ids); n_tok += len(e.ids)
n_files += len(batch)
print(f"corpus: {n_files} files, {n_tok} tokens, {len(seen)} distinct; held out: {len(heldout)} tokens")
for k in sorted({0, 8192, 16384, 32768, 65536, k_out}):
    s = seen | set(range(k)) | spec
    print(f"K={k:6d}: {len(s):6d} ids, held-out coverage {sum(i in s for i in heldout) / len(heldout):.4f}")
s = sorted(seen | set(range(k_out)) | spec)
open(out, "w").write("\n".join(map(str, s)) + "\n")
print(f"wrote {len(s)} ids (K={k_out}) to {out}")
