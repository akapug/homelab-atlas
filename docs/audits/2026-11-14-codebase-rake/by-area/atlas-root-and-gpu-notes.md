# Atlas Root Build, Checkers & GPU Kernel Notes (`build.py`, `tools/`, `b70/`, `rtx-6000/`)

Part of the [2026-11-14 Full-Codebase Audit Rake](../README.md) for `homelab-atlas`.

- **Shards in chapter**: `2`
- **File slices covered**: `28` (`1,540` lines)
- **Findings**: `22` total (`0` CRITICAL, `3` HIGH, `12` MEDIUM, `2` LOW; `3` COSPLAY-tagged)
- **Module-verdict removable LoC estimate**: `255`

## Chapter Shard Index

| Shard ID | Scope / Domain | Files | Lines | Findings (`C/H/M/L`) | Removable LoC |
| :--- | :--- | ---: | ---: | :--- | ---: |
| [`s177_homelab-atlas`](#s177_homelab-atlas) | `.` | 21 | 1,124 | `0/1/6/0` (11) | 145 |
| [`s186_homelab-atlas`](#s186_homelab-atlas) | `.` | 7 | 416 | `0/2/6/2` (11) | 110 |

---

## s177_homelab-atlas

- **Scope**: `.` | **Model**: `glm-5.3` | **Status**: `ok` | **Files**: `21` | **Lines**: `1,124`
- **Covered slices**: `b70/agent-eval/run.py` (`L1-L124` of `124`), `b70/agent-eval/sandbox-agent.sh` (`L1-L16` of `16`), `b70/bench/constrained-probe.py` (`L1-L46` of `46`), `b70/bench/cpu-contention-probe.sh` (`L1-L46` of `46`), `b70/bench/decode-streams.py` (`L1-L65` of `65`), `b70/bench/decode-vllm.py` (`L1-L93` of `93`), `b70/bench/serve-ab.py` (`L1-L132` of `132`), `b70/bench/spec-step-cost.sh` (`L1-L23` of `23`), `b70/llama.cpp-sycl/build.sh` (`L1-L28` of `28`), `b70/llama.cpp-sycl-bonsai/build.sh` (`L1-L46` of `46`), `b70/llama.cpp-sycl-fa-sinks/build.sh` (`L1-L16` of `16`), `b70/llama.cpp-sycl-fa-sinks/paths.py` (`L1-L31` of `31`), `b70/llama.cpp-sycl-fa-sinks/tbo.sh` (`L1-L18` of `18`), `b70/vllm-xpu/draft-vocab-build.py` (`L1-L56` of `56`), `b70/vllm-xpu/draft-vocab.py` (`L1-L66` of `66`), `b70/vllm-xpu/effort-template.py` (`L1-L25` of `25`), `b70/vllm-xpu/length-finish.py` (`L1-L44` of `44`), `b70/vllm-xpu/gdn-mixed-batch/run-gdn-mixed.py` (`L1-L39` of `39`), `b70/vllm-xpu/serving-cpu-guard.sh` (`L1-L48` of `48`), `b70/vllm-xpu/strict-tools.py` (`L1-L102` of `102`), `mi25/whole-layer-banks.py` (`L1-L60` of `60`)

### Summary

A homelab GPU/LLM benchmarking + monkey-patch toolkit: benchmark probes and pinned-source build scripts are genuinely rigorous (tree-hash verification, contamination flags, steady-state decode math), while the vllm-xpu side is a tower of four independent exact-string patchers over installed site-packages that will all break silently-or-loudly on every image upgrade. Overall health: good measurement engineering, fragile delivery mechanism.

### Module & Bloat Verdicts

| Module / File | Verdict | Removable LoC | Rationale & Salvage Notes |
| :--- | :--- | ---: | :--- |
| `b70/agent-eval/run.py` | `FIX_IN_PLACE` |  | Sound harness: fresh git fixture per task, hidden check.sh, verify mode with reference patch. Real isolation-by-construction, not theater. |
| `b70/agent-eval/sandbox-agent.sh` | `FIX_IN_PLACE` |  | Does isolate the filesystem (only $QWEN_CWD mounted, throwaway HOME) but --network=host plus skipped permissions makes the network boundary nonexistent; fine for a trusted homelab, misnamed as a sandbox. |
| `b70/bench/constrained-probe.py` | `FIX_IN_PLACE` |  | Small live probe, real assertions on real server responses; only flaw is it cannot be used as a gate because it always exits 0. |
| `b70/bench/cpu-contention-probe.sh` | `FIX_IN_PLACE` |  | Careful phase design, cpuset restore on EXIT trap, guard against serving-cpu-guard interference. stress() launch failure is unchecked. |
| `b70/bench/decode-streams.py` | `FIX_IN_PLACE` |  | Correct decode-only methodology (prefill each slot, then timed pass); uncached-token leak indicator is honest measurement hygiene. |
| `homelatlas/b70/bench/decode-vllm.py".replace("homelatlas` | `` |  | The steady-overlap-window computation (lines 83-87) is genuinely good benchmarking; but it is a second implementation of decode-streams.py with a different config surface (env vars vs argparse) and crashes on servers that omit usage. |
| `b70/bench/serve-ab.py` | `FIX_IN_PLACE` |  | Solid short measurement set with real usage validation and shared-server idle gate; module-level execution and model-specific TOK_PER_WORD constant are the only warts. |
| `b70/bench/spec-step-cost.sh` | `FIX_IN_PLACE` |  | The counter-delta contamination check (line 23) is real diligence, but the pipeline is glued by sed-regex over another script's prose output. |
| `b70/llama.cpp-sycl/build.sh` | `FIX_IN_PLACE` |  | Pinned base commit + patch series + git tree-hash verification is exactly how benchmark binary provenance should be done. |
| `b70/llama.cpp-sycl-bonsai/build.sh` | `SALVAGE_THEN_RIP` | 30 | ~90 percent duplicate of llama.cpp-sycl/build.sh with only BASE/TREE/FETCH/targets differing; two builders will drift. — Salvage the case-style variant table (lines 18-30: BASE/FETCH/TREE/DIR per build flavor) into one parameterized build.sh taking a variant name; delete the copy. |
| `b70/llama.cpp-sycl-fa-sinks/build.sh` | `FIX_IN_PLACE` |  | Digest-pinned container, sane log-on-fail; small and honest. |
| `b70/llama.cpp-sycl-fa-sinks/paths.py` | `FIX_IN_PLACE` |  | Appropriate regex parse of a debug trace format that is itself line-oriented; correct diff and counter rollup. |
| `b70/llama.cpp-sycl-fa-sinks/tbo.sh` | `FIX_IN_PLACE` |  | Device-by-PCI-path selection is the right OS boundary; unquoted $OUT/$CR/$RR paths are the only bug. |
| `b70/vllm-xpu/draft-vocab-build.py` | `FIX_IN_PLACE` |  | Clean offline vocabulary builder with honest held-out coverage reporting across K values. |
| `b70/vllm-xpu/draft-vocab.py` | `SALVAGE_THEN_RIP` | 40 | The idea (score only draft-vocab rows, -inf elsewhere, target verifies full head so correctness is preserved) is excellent; the delivery vehicle (exact-string replace into site-packages) is an upgrade landmine. — Keep the generated compute_logits/get_top_tokens/_draft_vocab_ids block (lines 29-59) as a patch file in a git fork or a sitecustomize import hook applied at container build; delete the self-rewriting patcher pattern. |
| `b70/vllm-xpu/effort-template.py` | `SALVAGE_THEN_RIP` | 15 | One-line jinja mapping fixing a real 400-failure class; guarded by count check. Still a prose-string patcher that dies silently when the template changes. — Keep the effort-word mapping (high/max->xhigh, minimal/none->low, null->default) as a checked-in .jinja overlay in the image build; delete the runtime patcher. |
| `b70/vllm-xpu/length-finish.py` | `SALVAGE_THEN_RIP` | 25 | Correct correctness-critical fix (length stop must not masquerade as tool_calls) with clever upstream-fix detection, but delivered as a second independent string-patcher against the same site-packages tree. — Keep the (old,new,upstream) hunk triple with upstream-supersession detection (lines 25-42) and carry it as a quilt/git patch series in the image Dockerfile; delete the bespoke patcher. |
| `b70/vllm-xpu/gdn-mixed-batch/run-gdn-mixed.py` | `FIX_IN_PLACE` |  | Pragmatic pytest-free differential runner; fake pytest skip raising RuntimeError is hacky but honest. |
| `b70/vllm-xpu/serving-cpu-guard.sh` | `FIX_IN_PLACE` |  | Correct mechanism (systemd slice AllowedCPUs + container cpuset, restore-on-stop) at the right OS boundary; polling loop treats fetch failure as idle. |
| `b70/vllm-xpu/strict-tools.py` | `SALVAGE_THEN_RIP` | 35 | The schema-validation fallback ladder (constrain name only, then no constraint, never a 500) is real defensive engineering; the appended code references unasserted module globals and joins the same patcher tower. — Keep _strict_tools_tag with its per-schema cache and fail-soft ladder (lines 59-99) as a proper upstream PR or fork patch; the reference to ChatCompletionToolsParam/FunctionTool/get_model_structural_tag must be asserted like the other anchors. |
| `mi25/whole-layer-banks.py` | `FIX_IN_PLACE` |  | Clear first-fit banking planner with honest CPU-leftover reporting; t.shape[2] expert-count assumption is brittle against 2D tensors. |

### Findings (11)

1. **[HIGH | `ARCH_DEAD_END` | `REPLACE_BOUNDARY` | `-60 LoC`]** `b70/vllm-xpu/strict-tools.py:22-65` — **Four independent exact-string monkey-patchers over one site-packages tree**
   - **Mechanism**: draft-vocab.py, effort-template.py, length-finish.py and strict-tools.py each read installed vLLM/jinja sources, assert on a hardcoded prose snippet, and rewrite the file in place with a .orig backup. Any image upgrade shifts the anchor text and every patcher asserts out at deploy time, or worse applies over a partially changed file; there is no inventory of which patches a running image carries and no ordering guarantee between them.
   - **Fix / Action**: Replace all four with one git fork of the pinned llm-scaler tree carrying the hunks as a numbered patch series applied by the image Dockerfile (the same pattern llama.cpp-sycl/build.sh already proves with tree-hash verification); emit the patched tree hash into the server log for provenance.
2. **[COSPLAY | `COSPLAY_THEATER` | `FIX_IN_PLACE`]** `b70/bench/constrained-probe.py:25-46` — **Probe prints OK/WRONG verdicts but always exits 0**
   - **Mechanism**: All four checks report correctness only via stdout strings; any wrapper (CI, cron, health gate) sees exit 0 whether the server's constrained decoding is broken or not, so the probe cannot actually gate anything.
   - **Fix / Action**: Accumulate failures in a list and sys.exit(1 if failures else 0) at the end; also catch json/KeyError per check so one malformed response reports WRONG instead of crashing before the other checks run.
3. **[MEDIUM | `FLAW_SECURITY_OVERSIGHT` | `FIX_IN_PLACE`]** `b70/agent-eval/sandbox-agent.sh:11-16` — **Sandbox with --network=host and skipped permissions**
   - **Mechanism**: The script claims the working copy is the only host path the agent sees, but --network=host plus --dangerously-skip-permissions lets the evaluated agent reach every host-local service (including the QWEN_PROXY backend and anything on localhost) and exec freely; a prompt-injected fixture repo can exfiltrate or pivot.
   - **Fix / Action**: Drop --network=host, add -p 127.0.0.1::8399 style port mapping or --network with only the proxy reachable, and keep permissions skipped only inside a user-namespaced container (--security-opt no-new-privileges, read-only rootfs).
4. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `b70/bench/decode-vllm.py:65-66` — **Unguarded usage indexing crashes the whole benchmark run**
   - **Mechanism**: If any stream's final usage chunk is missing or usage stays None (proxy hiccup, server restart, timeout mid-stream), usage.get(...) raises TypeError / usage[completion_tokens] raises KeyError, killing all threads and discarding every completed row of a multi-minute run.
   - **Fix / Action**: Wrap the return-dict construction: if usage is None or lacks completion_tokens, return a sentinel row and have the caller report and skip that stream instead of propagating.
5. **[MEDIUM | `ARCH_DEAD_END` | `REPLACE_BOUNDARY` | `-10 LoC`]** `b70/bench/spec-step-cost.sh:15-21` — **Benchmark pipeline glued by sed over another script's prose output**
   - **Mechanism**: spec-step-cost.sh extracts the decode rate by sed-matching the literal phrase `decode aggregate <num> tok/s` from decode-vllm.py's formatted print line; any wording or format change in decode-vllm.py silently yields an empty rate, and awk then divides by zero-ish r and prints garbage step-ms numbers with no error.
   - **Fix / Action**: Make decode-vllm.py emit a machine-readable line (jsonl to stderr or a --json flag) and have spec-step-cost.sh consume that; or have decode-vllm.py itself read the spec-decode counters and print steps/ms, deleting the wrapper entirely.
6. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `b70/vllm-xpu/serving-cpu-guard.sh:26-27` — **Metrics fetch failure is indistinguishable from idle**
   - **Mechanism**: busy() pipes curl output into awk; if every curl fails (server container restarting, port down, metrics endpoint hung beyond -m 1), awk sums nothing and exits 0 = idle, so the guard opens the CPUs to user load while the server is actually mid-crash-loop or rebooting into a serving state; conversely a hung poll never escalates.
   - **Fix / Action**: Track per-port fetch success: if all curls returned empty/non-200, keep the current split state and log a warning instead of decoding it as idle; alert after N consecutive failures.
7. **[MEDIUM | `FLAW_SECURITY_OVERSIGHT` | `FIX_IN_PLACE`]** `b70/vllm-xpu/strict-tools.py:67-73` — **Fail-open constraint drop on any xgrammar exception**
   - **Mechanism**: valid() catches bare Exception and returns False, and _STRICT_TAG_OK False means the request proceeds with no grammar at all — exactly the unconstrained mode that once emitted a tool named Bash-backtick. A malformed or adversarial tool schema in a request therefore silently disables the guard the patch exists to provide.
   - **Fix / Action**: Make the combined-tag failure configurable: fail closed (reject request) when VLLM_STRICT_TOOLS_MODE=strict, and always log the dropped schema hash so silent constraint loss is observable in metrics.
8. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `b70/vllm-xpu/strict-tools.py:53-56` — **Appended patch code references unasserted module globals**
   - **Mechanism**: The appended block uses ChatCompletionToolsParam, FunctionTool and get_model_structural_tag at call time; unlike the anchor assertions above, their presence is never checked, so an upstream rename turns every auto tool request into a NameError 500 — the exact failure mode the patch documents avoiding.
   - **Fix / Action**: Add assert lines for each referenced symbol in the host source before appending, same as the existing s.count checks, so a rename fails at patch time not at request time.
9. **[LOW_PLACEHOLDER | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `b70/agent-eval/run.py:64-96` — **Append-only results and last-stdout-line JSON make reruns ambiguous**
   - **Mechanism**: run() opens the out jsonl in append mode, so re-running a label mixes stale rows from earlier code/models into the same file with no way to distinguish epochs; agent telemetry comes from json.loads of the final stdout line, which silently becomes {} (nulls in every column) whenever the agent prints a trailing non-JSON line.
   - **Fix / Action**: Write to a timestamped file or truncate per (label, run-id) recorded in each row; on JSON parse failure record raw[-300:] in the row so the failure is inspectable instead of silently null.
10. **[LOW_PLACEHOLDER | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `b70/bench/cpu-contention-probe.sh:32-36` — **stress() never checks the transient unit actually started**
   - **Mechanism**: systemd-run failure (unit name collision from $$-$RANDOM, stress-ng missing, scope limit) leaves STRESS set but nothing running; the phase then measures idle CPU while labeling it busy/pinned, producing confidently wrong contention numbers — the worst outcome for a measurement script.
   - **Fix / Action**: After systemd-run, poll systemctl --user is-active $STRESS (up to ~5s) and abort the phase with an error if the unit is not active; also verify taskset/stress-ng exist upfront.
11. **[LOW_PLACEHOLDER | `ARCH_DEAD_END` | `REPLACE_BOUNDARY` | `-15 LoC`]** `b70/bench/decode-vllm.py:31-36` — **Dual config surfaces for twin benchmark scripts**
   - **Mechanism**: decode-streams.py takes argparse flags (--streams, --chars, --tokens) while decode-vllm.py, its declared mirror, reads env vars STREAMS/CHARS/TOKENS and positional argv with no argparse; users must remember two incompatible invocations for the same experiment matrix.
   - **Fix / Action**: Extract the shared prompt-construction/matrix-iteration into one module, give both frontends (or one frontend with --backend llama|vllm) the same argparse interface.

---

## s186_homelab-atlas

- **Scope**: `.` | **Model**: `glm-5.3` | **Status**: `ok` | **Files**: `7` | **Lines**: `416`
- **Covered slices**: `rtx-6000/v41-recipe/bench.py` (`L1-L49` of `49`), `rtx-6000/v41-recipe/build-draft.sh` (`L1-L45` of `45`), `rtx-6000/v41-recipe/build-engine.sh` (`L1-L45` of `45`), `rtx-6000/v41-recipe/build-store.sh` (`L1-L76` of `76`), `rtx-6000/v41-recipe/launch.sh` (`L1-L34` of `34`), `rtx-6000/v41-recipe/patch-imatrix-keep-partial.py` (`L1-L32` of `32`), `tools/check_atlas.py` (`L1-L135` of `135`)

### Summary

A reproducible DeepSeek-V4.1-on-RTX-6000 recipe: build scripts pin revisions and verify sha256s genuinely, bench.py measures a live llama-server, and launch.sh is a real tuned launcher — the build/run half is unusually honest. The one rotten tooth is tools/check_atlas.py, a 135-line regex tower that 'verifies' benchmark claims by substring-matching formatted numbers against Markdown prose, which is circular self-confirmation, not verification.

### Module & Bloat Verdicts

| Module / File | Verdict | Removable LoC | Rationale & Salvage Notes |
| :--- | :--- | ---: | :--- |
| `rtx-6000/v41-recipe/bench.py` | `FIX_IN_PLACE` |  | Genuine live benchmark client: real HTTP against llama-server, real timings, real draft counters. Only defects are all-or-nothing result persistence and no error handling. |
| `rtx-6000/v41-recipe/build-draft.sh` | `FIX_IN_PLACE` |  | Pinned HF revision + sha256 verification + pinned converter commit; structurally sound. Only unpinned pip deps mar reproducibility. |
| `rtx-6000/v41-recipe/build-engine.sh` | `FIX_IN_PLACE` |  | Dirty-tree guard, SOURCE_SHA pin, packaged libs with hashes — real reproducibility engineering. Unpinned apt cmake is the only drift hole. |
| `rtx-6000/v41-recipe/build-store.sh` | `FIX_IN_PLACE` |  | Real 800GB pipeline with genuine imatrix binary-format validation and sampled-byte verification. Final hash check for shards 8-10 is displayed but not enforced. |
| `rtx-6000/v41-recipe/launch.sh` | `FIX_IN_PLACE` |  | Real tuned launcher, but it mutates global kernel state (numa_balancing, demotion, swap) with no restore path despite its own comment promising one. |
| `rtx-6000/v41-recipe/patch-imatrix-keep-partial.py` | `FIX_IN_PLACE` |  | String-patching C++ by exact anchors is brittle, but it hard-fails on anchor mismatch and is pinned to one tested commit; should ship as a real .patch file instead. |
| `tools/check_atlas.py` | `REPLACE_BOUNDARY` | 110 | The whole verification model is wrong: it confirms atlas.json numbers by regex-substring-matching their formatted string inside Markdown sections, i.e. prose agrees with prose. The --accuracy mode is a triple-regex tower over report.js stdout and table cells. Replace with one machine-readable boundary: generate atlas.json and tables from bench.json/data.json so there is nothing to cross-check. — section() lines 31-44 (heading-scoped fenced-aware section extraction) is a useful 14-line utility if any Markdown-scoped checking survives; belongs in a small mdutil module. |

### Findings (11)

1. **[HIGH | `COSPLAY_THEATER` | `REPLACE_BOUNDARY` | `-60 LoC`]** `tools/check_atlas.py:69-73` — **Number 'verification' is substring-matching claims against prose that restates them**
   - **Mechanism**: A decode_tps value of 12.69 passes if the literal string '12.69' appears anywhere in the cited section — any coincidental number (a different metric, a hash, a date) satisfies it, and the cited page is itself written by the same pipeline being checked. The docstring admits it cannot catch drift onto another number the section prints. This rubber-stamps self-consistency, not correctness of measurements.
   - **Fix / Action**: Single source of truth: emit atlas.json result entries directly from bench.json (bench.py output) and recipe-maker data.json via a generator; the checker then compares two structured files field-by-field, and Markdown tables are rendered output, never an oracle.
2. **[HIGH | `ARCH_DEAD_END` | `REPLACE_BOUNDARY` | `-40 LoC`]** `tools/check_atlas.py:94-132` — **--accuracy is a three-regex tower over stdout prose and Markdown table cells**
   - **Mechanism**: Parses report.js human-readable output by line-prefix filtering, then regex-matches table cells 'P % (N runs, bias B %)' back against those lines with a fourth dynamically-built regex, plus a range-row reconciliation pass. Any formatting change in report.js or the README table silently breaks coverage in both directions, and line 128 `line.match(l).groups() if line.match(l) else (None,0,0)` masks unparseable lines as covered-by-default (b=None never matches, so uncovered — but also never errors on malformed input).
   - **Fix / Action**: Make report.js --accuracy emit JSON (box, n, median_err, bias); the checker compares JSON to a JSON view of the table parsed once by a real Markdown-table parser (or the table itself is generated from the JSON).
3. **[COSPLAY | `COSPLAY_THEATER` | `FIX_IN_PLACE`]** `rtx-6000/v41-recipe/build-store.sh:63-76` — **Final hash verification for shards 8-10 is displayed, not enforced**
   - **Mechanism**: Comment claims shard 10's sha256 equals the source's, but line 76 merely greps SHA256SUMS and prints 'ours:' — no cmp or sha256sum -c runs against the produced store's shards 1-7 or the symlinked 8-10. A corrupted or truncated earlier step passes the script.
   - **Fix / Action**: Run sha256sum -c against the coldq2k lines of SHA256SUMS over the 7 produced shards and `cmp` shard 10 against its source; exit non-zero on mismatch (document that shards 1-7 only match with the same imatrix).
4. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `rtx-6000/v41-recipe/bench.py:31-49` — **All-or-nothing result persistence across multi-hour runs**
   - **Mechanism**: Results are accumulated in `res` and dumped to bench.json only after every phase completes; with timeout=7200 per request, a single urllib error or Ctrl-C at phase n4b discards hours of measurements. Non-atomic single write can also leave a truncated bench.json.
   - **Fix / Action**: Write/rewrite bench.json (to a temp file + os.replace) after each request or at least each phase, and wrap post() with a try/except that dumps partial results before re-raising.
5. **[MEDIUM | `FLAW_SECURITY_OVERSIGHT` | `FIX_IN_PLACE`]** `rtx-6000/v41-recipe/build-draft.sh:36-41` — **Unpinned pip install into a persistent shared target dir**
   - **Mechanism**: requirements-convert_hf_to_gguf.txt is fetched live and installed unpinned into $OUT/py, which every subsequent run reuses without revalidating; a yanked/changed torch or transformers release silently changes the converter's output GGUF, and the sha-pinned inputs no longer guarantee a pinned artifact.
   - **Fix / Action**: Hash-pin the requirements (pip freeze to a lockfile committed next to SHA256SUMS) and install to a versioned dir; fail if $OUT/py exists but does not match the lock.
6. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `rtx-6000/v41-recipe/build-engine.sh:22-24` — **Pinned git SHA, unpinned build toolchain**
   - **Mechanism**: Meticulous commit pinning is defeated by `apt-get install cmake python3` from the rolling ubuntu24.04 repo: a cmake point release with different default flags or a GCC update changes the binary, so SOURCE_SHA no longer implies a reproducible package.
   - **Fix / Action**: Pin the container digest (IMG@sha256:...) and install a fixed cmake version (e.g. kitware repo or a copied binary), recording tool versions next to SOURCE_SHA.
7. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `rtx-6000/v41-recipe/launch.sh:13-17` — **Irreversible global kernel-state mutation with no restore trap**
   - **Mechanism**: Disables numa_balancing, memory demotion, and all swap as root, then `exec`s the server — the comment says 'Restore them afterwards' but the process that knew the prior values is gone. A crashed/killed server leaves the host with swap off and NUMA balancing disabled indefinitely.
   - **Fix / Action**: Read and save prior values to a state file, install an EXIT/INT/TERM trap in a wrapper that restores them, and exec llama-server from that wrapper.
8. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `tools/check_atlas.py:26-28` — **Fence stripping and slug() both wrong on ordinary Markdown**
   - **Mechanism**: ```.*?``` non-greedy strip merges across indented/tilded fences and breaks when a fence opens and the closer differs, so headings inside or after odd fences are dropped or invented; slug() does not implement GitHub slugging (unicode kept, duplicate headings get -1 suffixes), so anchors like '#results-1' never resolve and get reported as false misses or false hits on the first duplicate.
   - **Fix / Action**: Track fence state line-by-line (as section() already does) for anchors() too, and use a GitHub-compatible slugger with duplicate counters.
9. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `tools/check_atlas.py:75-89` — **Link scan regex misses titled links and false-positives on code blocks**
   - **Mechanism**: \]\(([^)\s]+)\) skips links with titles `](a.md 'x')` (broken link never caught) and the scan runs on raw text including fenced code blocks, so example links in code are flagged as broken. Same fence bug as anchors(), fixed there but not here.
   - **Fix / Action**: Strip fences once (shared helper) before scanning; allow `\]\(([^)\s]+)(?:\s+"[^"]*")?\)`.
10. **[LOW | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `rtx-6000/v41-recipe/patch-imatrix-keep-partial.py:13-31` — **Source patching by exact-string anchors instead of a patch file**
   - **Mechanism**: Whitespace-exact anchor matching breaks on any upstream reformat even when the logic is unchanged, and the patch silently becomes a no-op failure at the worst time (imatrix build). It does hard-fail on ambiguous anchors, which keeps it from being theater.
   - **Fix / Action**: Ship a real unified .patch applied with `patch --fuzz=0 -p1` (fails loudly on drift), or better, upstream the env-var guard into the tier-split branch.
11. **[LOW | `FLAW_SECURITY_OVERSIGHT` | `FIX_IN_PLACE`]** `tools/check_atlas.py:97-99` — **Checker executes repo code (build.py, node report.js) as a side effect**
   - **Mechanism**: A script whose job is static validation silently builds data.json by running build.py and then arbitrary node code from recipe-maker; in CI this means a 'check' step is really arbitrary code execution from the tree being checked, blurring the verifier/verified boundary.
   - **Fix / Action**: Require data.json to exist (fail with a clear message otherwise) and treat report.js output as data consumed via a file, keeping check_atlas.py purely read-and-compare.

---
