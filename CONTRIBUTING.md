# Contributing

Numbers from cards we do not have are the most welcome thing you can send. So are corrections: if a command here
does not reproduce our number on the same hardware, that is a bug, and an issue saying so helps the next reader.

## What a result needs

1. **Conditions, all of them.** Card and box (CPU, memory, how the card is attached), engine and its exact version
   or commit, model and quantization (with the file or checkpoint name), context window and depth, streams,
   speculation settings (MTP, draft model, n_max), sampling, thinking on or off, output length, date. A number
   without its conditions cannot be compared, so it does not go in.
2. **A quality check on the same configuration.** A faster path that changes the output needs evidence that it
   did not get worse: our KL and top-1 bar against a reference run (KL at most 0.03, top-1 at least 0.97), a
   correctness eval, an agent eval, or a recall probe. Say which, and its result, pass or fail.
3. **Every repetition.** Report ranges or all runs, not the best one. Say how the row aggregates (median, mean,
   upper median) and over what.
4. **Measured or not.** Mark anything estimated, fitted or read from a spec as inferred. Recipe Maker tags these
   `seen`, `inferred` and `unknown`; prose says "our reading" or "not measured".
5. **The losers too.** A change that did not help goes in "What did not work" on the card's page, with its number.
   It saves the next person a weekend.

## Where things go

- **A result on one card**: that card's folder (`3090-ti/`, `rtx-6000/`, `b70/`, `1080-ti/`, `mi25/`,
  `epyc-cpu/`). Each card page has the same sections: box, at a glance, details, reproduce, what did not work, not
  measured yet. A new card gets a new folder with the same sections.
- **A big MoE model across boxes**: a numbered point in [moe-offload](moe-offload/#what-we-learned-measured), and
  the run in Recipe Maker's ledger (`recipe-maker/tools/kit_record.sh RECIPE.json LOG...`), so the planner can score
  itself against it.
- **A patch**: beside the engine it patches, plus a row in [PATCHES.md](PATCHES.md) with symptom, apply, check and
  undo.
- **A headline number**: also the card's row in the README hardware table and in [atlas.json](atlas.json).
- **A new word**: [GLOSSARY.md](GLOSSARY.md).

## Every push

- Every push that changes results or structure adds a CHANGELOG line under Unreleased; cut a dated tag when a batch
  lands (`git tag -a v2026.10.22 -m "..."`, `.N` for a second the same day). Tags are never moved or deleted.
- `python3 tools/check_atlas.py` passes: every `atlas.json` number is on the page it cites, every card has a result,
  every link resolves. After a change to the run ledger, run it with `--accuracy` too.
- Plain English, short sentences, conditions on every number.
- No secrets, private paths, LAN addresses or personal details. Before every push we run Recipe Maker's export
  scanner over the whole checkout (`tools/export_atlas.sh --scan-only <checkout>`, in the private source tree), and
  it must print CLEAN. A string that looks private but is public (a heading anchor that names Claude Code, the
  `noreply` author line in a patch) goes in [`.scan-admit`](.scan-admit): one line of rule, path glob, the literal
  text around the hit, and the reason. Anything not listed there still refuses the push.
- No file over 1 MiB, no archives, no model weights.
- Written by an agent? Fine. End the page with the same sign-off line the other pages use, naming the models that
  wrote it.

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fabric) for [@akapug](https://github.com/akapug)
