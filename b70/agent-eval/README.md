# Agent eval: can a local model finish real repository work?

Short-function benchmarks could not tell our two local models apart, and the work we want from
them is finishing changes in a repository. A coding agent that runs is not yet one that finishes.
This is a small suite of repo-scale tasks with hidden checks, and what two models on Arc Pro B70s
scored driving [Claude Code](https://github.com/anthropics/claude-code).

## The tasks

Each task is a small repository (Python stdlib and bash, under ~400 lines), a request as a busy
developer would type it, a hidden check the agent never sees, and a reference fix. `run.py --verify`
proves every check fails on the fixture and passes with the reference fix, and each task's author
also confirmed that plausible wrong fixes fail (editing the failing test, fixing one of two call
sites, `>=` for `>`, and so on).

| task | difficulty | what it takes |
|---|---|---|
| `a1-semver-sort` | easy | a failing test exposes string comparison of version parts |
| `a2-retry-backoff` | medium | exponential backoff across a config loader, env overrides, the client and its tests |
| `a3-export-ties` | hard | an export cursor that loses rows sharing a timestamp; only the symptom is given |
| `b1-archive-spaces` | easy | a bash job that splits file names with spaces and hides failures behind `tee` |
| `b2-ledger-report` | medium | a new CLI subcommand from a short spec, with tests |
| `b3-rename-money-fmt` | hard | a keyword-only rename where one untested module calls the old name positionally |
| `c1-log-bom-crlf` | easy | a log parser that misreads a file with a UTF-8 BOM and CRLF line endings |
| `c2-joborder-perf` | medium | a quadratic scheduler and loader that must get fast without changing any output |
| `c3-quote-cache-alias` | hard | a crash whose cause is a shared cached object that three modules change in place |
| `d1-build-green-on-error` | easy | a build script that reports success through two separately masked failures |
| `d2-config-v2-migration` | medium | a config format v1 to v2 across the loader, the writer, two readers and the tests |
| `d3-stream-glued-records` | hard | a streaming decoder that merges records when an escape is split across reads |

The six tasks added in round 2 were also checked with an alternative correct fix placed elsewhere in the code,
which passes, so the checks test behavior and not the reference patch.

## Results (2026-09-24)

Claude Code 2.1.281, each model served by vLLM on its own B70 (see [`../`](../) and
[`../qwen36-35b-a3b.md`](../qwen36-35b-a3b.md)), reached through an Anthropic-to-OpenAI proxy.
"Capped" is vLLM's `thinking_token_budget` 8192; "off" is `enable_thinking: false`. 20-minute limit
per task. Rows: [`results.csv`](results.csv).

**The two leading configs on all 12 tasks, 2 passes each (24 runs per config): round 1's six tasks plus the six
added in round 2, as the `round` column of `results.csv` splits them.**

| model and thinking | passed | 95 % CI (Wilson) | median time per task |
|---|---|---|---|
| Qwen3.8-27B (dense), off | 22 / 24 | 74-98 % | 168 s |
| Qwen3.6-35B-A3B (MoE, ~3B active), capped | 15 / 24 | 43-79 % | 124 s |

Fisher two-sided p = 0.036. On the harder tasks the MoE's speed lead mostly went away, because it
spends longer when it is wrong: one run hit the 20-minute limit and another took 19 minutes. Over
all 12 tasks it was only ~1.35x faster per task.

**Round 1: four configs on the first 6 tasks, 2 passes each (12 runs per config).**

| model and thinking | passed | 95 % CI | median time per task |
|---|---|---|---|
| Qwen3.8-27B, capped | 12 / 12 | 76-100 % | 278 s |
| Qwen3.8-27B, off | 11 / 12 | 65-99 % | 143 s |
| Qwen3.6-35B-A3B, capped | 9 / 12 | 47-91 % | 63 s |
| Qwen3.6-35B-A3B, off | 8 / 12 | 39-86 % | 47 s |

Thinking made no detectable difference for either model (Fisher p = 1.0) and doubled the 27B's
time, so round 2 kept the faster 27B arm and the stronger MoE arm. A single pass would have misled:
the MoE went 6/6 then 3/6 with thinking capped.

- **How the MoE failed:** one-turn breakdowns with thinking off (a tool call written out as text the
  server's parser did not accept, then echoed harness text); wrong fixes (the backoff precedence, a
  bash failure path, the rename's untested caller, the export cursor, the shared cached object, the
  config migration, the build's masked failures); a scheduler still too slow; and the streaming
  decoder not fixed within 20 minutes, twice.
- **The 27B's misses:** the backoff precedence once, the config migration once.

Twelve small tasks is still a small suite, but the gap held when the suite doubled.

## Later results (2026-10-05 to 2026-10-18)

The same 12 tasks and hidden checks, Claude Code 2.1.291 in the same sandbox, 1,200 s per task unless a row says
otherwise. These rows are not in `results.csv`; each links the entry with its conditions.

| model, where | passed | conditions | entry |
|---|---|---|---|
| Qwen3.8 Flash-Next, Q8_0 on an RTX 3090 Ti + DDR5 (Strata) | 23 / 24 (21 / 24 inside the limit) | 2 passes, the model card's thinking sampling, MTP 4, 2026-10-18 | [3090 Ti](../../3090-ti/#qwen38-flash-next-agent-work-from-ram) |
| Qwen3.8 Flash-Next, IQ3_S on one B70 (XeStrata) | 11 / 12 | 1 pass, median 410 s a task, 2026-10-18 | [moe-offload point 22](../../moe-offload/#what-we-learned-measured) |
| Ornith-1.5-35B-A3B on one B70 (vLLM, MTP 3) | 17 / 24 | 2 passes, 2026-10-05 | [Ornith on the B70](../ornith-35b-a3b.md) |
| DeepSeek V4.1 Flash, hosted | 22 / 24 | 2 passes, the provider's own serving settings | reference only |
| GLM-5.3-Flash, hosted | 23 / 24 | 2 passes, the provider's settings; a second free provider also 23 / 24 with a 3,600 s limit | reference only |

The hosted rows are a ceiling to aim at, not local results. Qwen3.8-27B's 22 of 24 from 2026-09-24 above is still
the best local dense score; Flash-Next on the 3090 Ti box matched the hosted GLM's 23 of 24 when
runs past the limit count.

## Running it

The agent runs in a container that sees only the task's working copy ([`Dockerfile`](Dockerfile),
[`sandbox-agent.sh`](sandbox-agent.sh)), because it runs with permissions bypassed. Build the image,
point the environment at an Anthropic-compatible endpoint for your model, and run:

```bash
docker build -t agent-eval:claude .
python3 run.py --verify
QWEN_PROXY=http://127.0.0.1:8399 QWEN_MODEL=my-model QWEN_BACKEND=http://127.0.0.1:8000 \
QWEN_KEY_FILE=~/.my-proxy-key python3 run.py --label my-model -- bash $PWD/sandbox-agent.sh
```

`run.py --verify` checks every task's hidden tests fail on the broken code and pass with the reference fix; it
should print OK for all 12. If your box sends test runs through a wrapper that refuses `python3 -m unittest`, ten
of the twelve fail both ways, which looks like broken fixtures and is not: run each task's check through the wrapper.

`QWEN_BACKEND` is the OpenAI-compatible server itself; the script reads the context window from its
`/v1/models`. Any agent command works after `--`: the runner appends `-p <prompt> --output-format
json` and runs it in the task's working copy.

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [arc](https://github.com/akapug/arc) + [fab](https://github.com/akapug/fabric) for [@akapug](https://github.com/akapug)
