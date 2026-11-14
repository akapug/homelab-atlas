# Recipe Maker Probes & Ledger Tooling (`recipe-maker/probe/`, `recipe-maker/tools/`)

Part of the [2026-11-14 Full-Codebase Audit Rake](../README.md) for `homelab-atlas`.

- **Shards in chapter**: `2`
- **File slices covered**: `19` (`4,295` lines)
- **Findings**: `20` total (`0` CRITICAL, `5` HIGH, `14` MEDIUM, `0` LOW; `2` COSPLAY-tagged)
- **Module-verdict removable LoC estimate**: `317`

## Chapter Shard Index

| Shard ID | Scope / Domain | Files | Lines | Findings (`C/H/M/L`) | Removable LoC |
| :--- | :--- | ---: | ---: | :--- | ---: |
| [`s180_homelab-atlas__recipe-maker__probe`](#s180_homelab-atlas__recipe-maker__probe) | `recipe-maker/probe` | 7 | 1,813 | `0/2/7/0` (10) | 282 |
| [`s185_homelab-atlas__recipe-maker__tools`](#s185_homelab-atlas__recipe-maker__tools) | `recipe-maker/tools` | 12 | 2,482 | `0/3/7/0` (10) | 35 |

---

## s180_homelab-atlas__recipe-maker__probe

- **Scope**: `recipe-maker/probe` | **Model**: `glm-5.3` | **Status**: `ok` | **Files**: `7` | **Lines**: `1,813`
- **Covered slices**: `recipe-maker/probe/collect.sh` (`L1-L126` of `126`), `recipe-maker/probe/gguf.py` (`L1-L173` of `173`), `recipe-maker/probe/model_intake.py` (`L1-L429` of `429`), `recipe-maker/probe/overlay.py` (`L1-L151` of `151`), `recipe-maker/probe/probe_box.py` (`L1-L738` of `738`), `recipe-maker/probe/vitals_adapter.py` (`L1-L62` of `62`), `recipe-maker/probe/window_inputs.py` (`L1-L134` of `134`)

### Summary

A read-only Linux hardware/model-header probing suite that mostly does what it claims: tagged READ/MEASURED values with provenance, honest UNKNOWNs, header-only model intake. It is unusually healthy, but suffers duplicated GPU-matching/diff logic across overlay.py and probe_box.py, a regex tower over prose benchmark logs in window_inputs.py, silent GPU drops in the xpu parser, non-atomic state writes, and an optional vitals_adapter that adds near-zero value.

### Module & Bloat Verdicts

| Module / File | Verdict | Removable LoC | Rationale & Salvage Notes |
| :--- | :--- | ---: | :--- |
| `recipe-maker/probe/collect.sh` | `FIX_IN_PLACE` |  | Genuinely read-only, careful collector: unset of shadowed shell functions, sudo -n only, @@!section failure markers, space-quoting via tr. Structurally sound; only minor fragilities in its ad-hoc KV line format. |
| `recipe-maker/probe/gguf.py` | `FIX_IN_PLACE` |  | Clean stdlib GGUF header reader with a sound chunked Reader, offset-derived tensor sizes (type-independent), shard expansion. Minor: GGML_TYPES table is dead weight since sizes never depend on names. |
| `recipe-maker/probe/model_intake.py` | `FIX_IN_PLACE` | 30 | Real range-request header intake from HF/safetensors/GGUF/ssh with tag discipline. Needs surgical fixes: HF-token redirect handling, atomic --write, and the giant CLASSES regex ladder is a maintenance hazard but earns its keep for now. |
| `recipe-maker/probe/overlay.py` | `REPLACE_BOUNDARY` | 60 | Rank-based merge over untyped dicts is a hand-rolled fold with a duplicate GPU matcher (vs probe_box.match_gpus), a 'hand' key namespace that collides with real keys, and silent positional GPU fallback. Replace the merge with a single typed schema + one canonical matcher. — Keep the RANK (MEASURED>READ>FIT>UNKNOWN) tag lattice and the audit log of add/replace/conflict; put them in one merge function parameterized by a typed field spec. |
| `recipe-maker/probe/probe_box.py` | `FIX_IN_PLACE` | 40 | The core probe is honest (tagged values, unknown registry, resolver for btrfs/loop/dm chains). Fix: silent xpu JSON drops, non-atomic --write, brittle PCI-substring GPU matching, and the duplicated match_gpus/_gpu_match/diff logic. |
| `recipe-maker/probe/vitals_adapter.py` | `RIP_OUT` | 62 | An optional, stale-prone, annotate-only cross-check that can never change a value, duplicates the disk-kind taxonomy with a weaker one ('image' auto-agrees with everything), and is wired in via except ImportError: pass. Pure optional weight; the probe is self-sufficient per its own docstring. |
| `recipe-maker/probe/window_inputs.py` | `REPLACE_BOUNDARY` | 90 | Regex tower over human-formatted STAGE ROLL-UP prose (column counts guessed as `cols[2:] if x.isdigit()`, table end detected by a heuristic break, kind attribution by name-substring tiebreak). Any whitespace change in the upstream tool silently mis-parses numbers. — Keep parse_spec (lines 24-40): a clean 15-line dims/layer line format. Replace parse_ubench by making the XeStrata ubench tool emit JSON (its own --json flag) — one wire boundary instead of 90 lines of prose scraping. |

### Findings (10)

1. **[HIGH | `ARCH_DEAD_END` | `REPLACE_BOUNDARY` | `-17 LoC`]** `recipe-maker/probe/overlay.py:46-62` — **Duplicate GPU matcher with silent positional fallback cross-assignment**
   - **Mechanism**: _gpu_match is a near-copy of probe_box.match_gpus (lines 600-615); both fall back to assigning leftover probed GPUs to unmatched hand slots by list order, so on a rebussed box a hand entry silently absorbs a different physical card's VRAM/driver data with only a generic 'replace' log line.
   - **Fix / Action**: Delete both copies; one match_by_pci(hand, probed) that matches only on exact PCI (hand 'pci' key or parsed short address) and reports unmatched probed GPUs as gpus_unlisted without positional backfill.
2. **[HIGH | `UNDERENGINEERED` | `REPLACE_BOUNDARY` | `-90 LoC`]** `recipe-maker/probe/window_inputs.py:50-98` — **Microbenchmark cost model fed by prose-regex table scraping**
   - **Mechanism**: parse_ubench reconstructs a numeric table from whitespace-split tokens: the row grid is guessed as `cols[2:] if x.isdigit()`, ntail is derived from that guess, table termination depends on `if classes and (totals or unmeasured): break`, and class-to-kind attribution falls back to a case-insensitive name-substring (class_kind lines 101-112). A renamed column or an extra footnote line shifts every parsed float into the wrong field with no error.
   - **Fix / Action**: Make batch_window_ubench emit a machine-readable summary (JSON or CSV with a header contract); this file then reduces to parse_spec plus a json.load of that output.
3. **[COSPLAY | `COSPLAY_THEATER` | `RIP_OUT` | `-62 LoC`]** `recipe-maker/probe/vitals_adapter.py:43-62` — **Cross-check that can never disagree usefully**
   - **Mechanism**: 'image' kinds auto-agree with any probe kind (line 57), mismatches are only recorded in box['probe']['vitals'] and never surfaced in the diff output, and the adapter is wired via `except ImportError: pass` in probe_box (lines 709-716) so its absence or a syntax error is indistinguishable from 'no vitals file'. It verifies nothing and gates nothing.
   - **Fix / Action**: Delete the module and the probe_box import block; if a vitals disagreement ever matters, print it from diff_box like every other discrepancy.
4. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/probe/collect.sh:89-93` — **xpu-smi discovery IDs scraped with grep -oE instead of the JSON it just fetched**
   - **Mechanism**: The script already receives `xpu-smi discovery -j` (machine-readable) but re-extracts device ids with two chained `grep -oE` over the JSON text, then issues one `discovery -d $i -j` per device; a key formatted with different spacing or a new field containing 'device_id' yields wrong or duplicated ids, and N sequential subprocesses (with 20s timeouts each) balloon the probe wall time.
   - **Fix / Action**: Emit the single discovery -j blob verbatim under its own @@xpu_json section and parse it once in probe_box.py (which already json.loads per-device blobs at line 228); ids come from the parsed structure, one xpu-smi invocation total.
5. **[MEDIUM | `FLAW_SECURITY_OVERSIGHT` | `FIX_IN_PLACE`]** `recipe-maker/probe/model_intake.py:111-119` — **HF_TOKEN sent via urllib redirects outside huggingface.co**
   - **Mechanism**: The Authorization header is set only for huggingface.co hostnames, but resolve URLs 302/302 to CDN hosts (cas-bridge.xethub.hf.co, cdn-lfs); urllib.request follows redirects and re-sends request headers, leaking the bearer token to third-party CDN hosts on older Pythons and to any host a repo can redirect to.
   - **Fix / Action**: Use a custom HTTPRedirectHandler that strips the Authorization header on any cross-host redirect (or open with an opener that pins credentials per-URL), and assert the final URL host is *.hf.co / *.huggingface.co before sending the token.
6. **[MEDIUM | `FLAW_SECURITY_OVERSIGHT` | `FIX_IN_PLACE`]** `recipe-maker/probe/model_intake.py:398-407` — **Non-atomic --write truncates data files on crash**
   - **Mechanism**: `open(a.write, 'w').write(js)` truncates data/models/KEY.json in place; a crash, full disk, or Ctrl-C mid-write destroys the previous good entry (the same pattern at probe_box.py:718-720 and header_out at :399).
   - **Fix / Action**: Write to a temp file in the same directory and os.replace() it; one 3-line helper shared by all three call sites.
7. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/probe/model_intake.py:47-91` — **First-match-wins regex ladder over tensor names as the classification engine**
   - **Mechanism**: CLASSES is an ordered 17-regex chain applied to arbitrary upstream naming (GGUF + every HF repo); any new architecture whose tensor names happen to hit 'attn' or 'norm' early (e.g. an expert tensor named 'attn_experts') is silently misclassified into the wrong byte bucket, and the only signal is the 20-name 'unclassified' list which cannot detect misclassification, only non-classification.
   - **Fix / Action**: Keep the ladder but assert per-model that classified bytes sum to total_gib within epsilon and that expert byte totals reconcile with E x P x bpw sanity (bpw in 1..8); fail loudly (exit nonzero) on unreconciled mass instead of emitting a plausible-looking entry.
8. **[MEDIUM | `FLAW_SECURITY_OVERSIGHT` | `FIX_IN_PLACE`]** `recipe-maker/probe/overlay.py:123-137` — **Record merge overwrites untagged hand fields; 'hand' namespace collision**
   - **Mechanism**: Line 126 compares rec_rank >= rank(h.get('tag')); when the hand dict carries no tag, rank(None) = -1, so ANY probe record overwrites every non-KEEP hand field silently (only a log line). Worse, h.setdefault('hand', {}) collides with an overlay record that legitimately contains a 'hand' key, letting probe data clobber preserved hand values (and lines 132-137 do the same for 'tag'/'src').
   - **Fix / Action**: Use a reserved key like '__hand__' for preserved values, and require rec_rank > 0 (record must carry READ/MEASURED) before raw-field replacement; refuse to merge overlay keys matching the reserved namespace.
9. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/probe/probe_box.py:222-234` — **parse_xpu silently drops GPUs on JSON decode failure**
   - **Mechanism**: `except ValueError: continue` discards an xpu-smi device record wholesale; a truncated or error-laced discovery blob makes an Intel GPU vanish from box['gpus'] with no UNKNOWN entry and no probe.unknown entry, so the planner treats the box as GPU-less.
   - **Fix / Action**: On decode failure append {'index': int(key.split()[1]), 'error': 'unparseable xpu-smi JSON', ...} or at minimum register it in the unknown list, matching the honest-UNKNOWN contract the file advertises.
10. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/probe/probe_box.py:82-84` — **kv_line regex loses empty and '='-containing values**
   - **Mechanism**: `re.findall(r'(\w+)=(\S*)', line)` pairs every '=' occurrence left-to-right; a value containing '=' (sysfs path= values, driver names) is split into two bogus keys, and an empty value plus an adjacent key merges tokens. Every parser downstream (parse_dax, parse_nd, parse_drm) inherits the corruption with no validation.
   - **Fix / Action**: Split on first '=' per whitespace-token: `{p.split('=',1)[0]: p.split('=',1)[1] for p in line.split() if '=' in p}`, and have collect.sh quote or hex-escape '=' in values as it already does for spaces.

---

## s185_homelab-atlas__recipe-maker__tools

- **Scope**: `recipe-maker/tools` | **Model**: `glm-5.3` | **Status**: `ok` | **Files**: `12` | **Lines**: `2,482`
- **Covered slices**: `recipe-maker/tools/bench_gpu.sh` (`L1-L20` of `20`), `recipe-maker/tools/bench_numa.sh` (`L1-L36` of `36`), `recipe-maker/tools/heatmap.py` (`L1-L544` of `544`), `recipe-maker/tools/kit_record.sh` (`L1-L29` of `29`), `recipe-maker/tools/heat-capture/moe-trace.cpp` (`L1-L145` of `145`), `recipe-maker/tools/heat-capture/patch-imatrix-expert-counts.py` (`L1-L22` of `22`), `recipe-maker/tools/place_verify.py` (`L1-L128` of `128`), `recipe-maker/tools/record_bench.py` (`L1-L132` of `132`), `recipe-maker/tools/record_dllm.py` (`L1-L254` of `254`), `recipe-maker/tools/record_run.py` (`L1-L835` of `835`), `recipe-maker/tools/record_screen.py` (`L1-L253` of `253`), `recipe-maker/tools/heat-capture/route_stats.py` (`L1-L84` of `84`)

### Summary

A genuinely-working homelab measurement pipeline: shell bench sweeps, a canonical hot-expert map format with digests and real validation, an in-engine MoE router tracer, and ledger recorders that cite sources per number. Weak points are cosmetic-but-real: shell pipelines that mask bench exit codes, one whole-file non-atomic ledger rewrite, and a recipe-cloning path that edits serialized flag strings with regex instead of structured state.

### Module & Bloat Verdicts

| Module / File | Verdict | Removable LoC | Rationale & Salvage Notes |
| :--- | :--- | ---: | :--- |
| `recipe-maker/tools/bench_gpu.sh` | `FIX_IN_PLACE` |  | Minimal, honest sweep harness; only defect is exit-status masking from the un-pipefailed pipeline. |
| `recipe-maker/tools/bench_numa.sh` | `FIX_IN_PLACE` |  | Correct background/wait/rc handling; the fixed-time numastat sample is the only soft spot. |
| `recipe-maker/tools/heatmap.py` | `FIX_IN_PLACE` |  | Real canonical format: atomic writes, digest over core fields, validate() recomputes summaries and refuses mismatches. Only edge-case crashes in converters. |
| `recipe-maker/tools/kit_record.sh` | `FIX_IN_PLACE` |  | Thin, correct wrapper; gate-output discovery and explicit --no-check post-hoc marking are honest. |
| `recipe-maker/tools/heat-capture/moe-trace.cpp` | `FIX_IN_PLACE` |  | Real cb_eval instrumentation writing a binary format the Python side parses with magic checks; only fwrite results are unchecked. |
| `recipe-maker/tools/heat-capture/patch-imatrix-expert-counts.py` | `FIX_IN_PLACE` |  | 22-line one-shot vendored patcher that fails loudly on anchor drift (assert count==1, idempotence check). Does its job. |
| `recipe-maker/tools/place_verify.py` | `FIX_IN_PLACE` |  | The --pid /proc/numa_maps path is a real OS-boundary check; the log-regex path is a fallback that its own docstring admits the engine report can game. Keep, but bias to live mode. |
| `recipe-maker/tools/record_bench.py` | `FIX_IN_PLACE` |  | Solid ingest with per-model header caching and duplicate-id refusal; reads real GGUF headers over ssh instead of guessing. |
| `recipe-maker/tools/record_dllm.py` | `FIX_IN_PLACE` |  | Honest post-hoc ingest that marks gaps as stopped/post-hoc rather than fabricating; a few unguarded next()/division paths will crash instead of erroring cleanly. |
| `recipe-maker/tools/record_run.py` | `FIX_IN_PLACE` | 35 | The 835-line core is mostly sound (per-number source citations, launch vs post-hoc prediction basis, fallback-receipt identity binding), but recipe_from() is a regex surgery tower over serialized flag strings that belongs behind a structured representation. — recipe_from flag-editing block (lines 660-674): the intent (arm env/flags override base recipe) is right; the mechanism must move to a dict of engine flags serialized once at ledger-write time. |
| `recipe-maker/tools/record_screen.py` | `FIX_IN_PLACE` |  | Good failure-shape capture with sourced settings; the ledger rewrite is non-atomic and the retry_only definition ignores loop/leak on the retry attempt. |
| `recipe-maker/tools/heat-capture/route_stats.py` | `FIX_IN_PLACE` |  | Real monkeypatch instrumentation counting actual router picks pre-masking; only the prefill/decode classification is a size heuristic. |

### Findings (10)

1. **[HIGH | `FLAW_SECURITY_OVERSIGHT` | `FIX_IN_PLACE`]** `recipe-maker/tools/bench_gpu.sh:15-17` — **Pipeline exit status masked: r captures tail, not the bench**
   - **Mechanism**: `out=$(BENCH ... | tail -n 1); r=$?` records tail exit (always 0) because pipefail is off; a bench that fails after emitting at least one jsonl row is recorded with rc:0 and a stale row, and downstream record_bench trusts r['rc'] as the arm outcome.
   - **Fix / Action**: Add `set -o pipefail` at the top, or capture via a temp file like bench_numa.sh already does (bench to file, then tail), so r is the bench status.
2. **[HIGH | `ARCH_DEAD_END` | `REPLACE_BOUNDARY` | `-35 LoC`]** `recipe-maker/tools/record_run.py:660-674` — **recipe_from edits serialized engine_flags prose with stacked regex subs**
   - **Mechanism**: Engine configuration lives as a formatted string (engine_flags with ;-delimited sections) and arm overrides are applied via re.sub(r'max_tokens \d+'), r'MAX_CONTEXT=\d+', flag-prefix lookahead subs, and label regex surgery (line 654). A base recipe whose label lacks 'at N agents' or whose flag wording differs silently no-ops the substitution, producing a ledger row that claims settings the run never used.
   - **Fix / Action**: Represent engine flags as an ordered dict in the recipe; apply arm overrides as key assignment; serialize to the display string once at write time. Labels: build from fields, never re.sub.
3. **[HIGH | `FLAW_SECURITY_OVERSIGHT` | `FIX_IN_PLACE`]** `recipe-maker/tools/record_screen.py:211-221` — **Non-atomic whole-file rewrite of the screening ledger**
   - **Mechanism**: write() opens the ledger with mode 'w' and rewrites old+new rows in place; a crash, full disk, or SIGINT mid-loop truncates runs/screen.jsonl and silently destroys all prior rounds, unlike record_run.append (append-only) and heatmap.write (tmp + os.replace).
   - **Fix / Action**: Write keep+rows to path+'.tmp', then os.replace onto path; the replace pattern already exists in heatmap.py:71-73.
4. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/tools/bench_numa.sh:24-28` — **Fixed 12s numastat sample misses short arms and dead processes**
   - **Mechanism**: sleep $SAMPLE_S then `numastat -p $pid 2>/dev/null`: if the bench finished (or not yet allocated weights) at t=12s, the sample is empty or pre-load; first-touch arms then feed record_bench an empty node_mib, whose NUMA_FT_SHARE inference (record_bench.py:73) yields an empty mem_nodes list, mislabeling the socket span of the row.
   - **Fix / Action**: Poll: wait until the bench's RSS exceeds a floor (or 2s ticks up to SAMPLE_S), then sample numastat; verify node_mib non-empty before printing the row or emit a null with an explicit reason field.
5. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/tools/heat-capture/moe-trace.cpp:52-54` — **Unchecked fwrite on the trace file silently loses records**
   - **Mechanism**: fwrite return values are ignored; on ENOSPC the tracer keeps 'succeeding' (recs++ counts phantom records) and the heatmap converter then hits bad magic / truncation at the first lost record — all records after the short write are discarded (the converter raises at that byte), losing the tail of every later prompt despite the run completing.
   - **Fix / Action**: Check both fwrite returns; on failure log LOG_ERR with tr->recs and abort the workload loop (the per-prompt GEN jsonl already exists to salvage completed prompts) — or buffer and write per-prompt atomically after fflush.
6. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/tools/heat-capture/route_stats.py:45-49` — **Prefill/decode phase classified by tensor-size threshold**
   - **Mechanism**: `m = 'prefill' if NV.pool.stage_n and s.numel() >= NV.pool.stage_min else 'decode'` — any decode batch at or above stage_min tokens (large speculative windows, batched concurrent requests) is counted as prefill, and a short staged forward as decode; the resulting per-phase counts and meta token totals in route_stats.json are systematically skewed with no flag in the output that classification was heuristic.
   - **Fix / Action**: Pass the true phase from the caller (run() knows whether it enqueued a prefill chunk or decode step), or at minimum emit the misclassification risk and the count of borderline calls (numel within 2x of stage_min) into meta so downstream maps can carry the caveat.
7. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/tools/heatmap.py:344-350` — **conv_moe_trace unpacks past EOF without bounds check; empty-slice crash in build_map**
   - **Mechanism**: struct.unpack_from('<7i', data, off) and the ids read assume a well-formed tail; a truncated trace raises raw struct.error (not MapError with path context), and build_map line 394 (`next(iter(slices.values()))['counts'][0]`) IndexErrors when a converter legitimately produced zero layers (e.g. imatrix with no *_exps rows), both escaping as tracebacks instead of the file's own error vocabulary.
   - **Fix / Action**: Check `off + 28 + 4*n*k <= len(data)` before each unpack and raise MapError('%s: truncated record at byte %d'); guard the experts inference with `if slices and next(iter(slices.values()))['counts']` else raise MapError('no MoE rows found in source').
8. **[MEDIUM | `COSPLAY_THEATER` | `FIX_IN_PLACE`]** `recipe-maker/tools/place_verify.py:34-57` — **Log-mode verification trusts regex-scraped engine prose the engine itself games**
   - **Mechanism**: The default mode verifies placement against llama_expert_numa_report lines, whose off-target share the file's own docstring (lines 7-9) admits a skewed interleave passes (q4z: 3.26% off yet 121/146 GiB on one node); only the derived per-node even-split comparison catches it, and that assumes exact even interleave. APPLY/POLICY/REPORT regexes are also one log-format tweak from matching nothing, which degrades to status NONE (exit 2), not failure.
   - **Fix / Action**: Make --pid live numa_maps the default verification path (it already exists at lines 60-78 and measures reality), keeping the log parse only as a NONE-status fallback; fail closed on regex drift by alerting when an apply line appears but no policy/report matches.
9. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/tools/record_dllm.py:204-217` — **cli_runs ingest: bare next() StopIteration and unguarded tpf division**
   - **Mechanism**: `i, x = next((...) for ... if CLI_RE.search(s))` raises an uncaught StopIteration when the catalog log lacks the timing line, aborting the whole ingest mid-host; `steps = e['gen_tokens'] / e['tpf']` (line 217) ZeroDivision-errors on a catalog entry with tpf 0. Both crash after partial rows/raw files were already written.
   - **Fix / Action**: Use next(..., None) and skip the entry with a printed reason; assert e['tpf'] > 0 at catalog load and reject the row with a message.
10. **[MEDIUM | `UNDERENGINEERED` | `FIX_IN_PLACE`]** `recipe-maker/tools/record_run.py:235-240` — **Kit arm JSON fallback can attribute the previous arm's numbers**
   - **Mechanism**: `js = [j for j in js if j.get('arm') == r.group(1)] or js[-1:]` falls back to the last JSON line seen regardless of which arm emitted it; when an arm prints no JSON of its own, its measured decode_tps/vram come from a different arm's line, poisoning the ledger with cross-attributed numbers and no warning.
   - **Fix / Action**: Drop the `or js[-1:]` fallback; when no matching JSON exists, record outcome crash with note 'no JSON arm line' instead of borrowing another arm's measurements.

---
