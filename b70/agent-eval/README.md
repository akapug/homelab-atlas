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

## Results (2026-09-24)

Claude Code 2.1.281, each model served by vLLM on its own B70 (see [`../`](../) and
[`../qwen36-35b-a3b.md`](../qwen36-35b-a3b.md)), reached through an Anthropic-to-OpenAI proxy.
"Capped" is vLLM's `thinking_token_budget` 8192; "off" is `enable_thinking: false`. 6 tasks x 2
passes = 12 runs per config, 20-minute limit per task. Rows: [`results.csv`](results.csv).

| model and thinking | passed | 95 % CI (Wilson) | median time per task | median turns |
|---|---|---|---|---|
| Qwen3.8-27B (dense), capped | 12 / 12 | 76-100 % | 278 s | 20.5 |
| Qwen3.8-27B, off | 11 / 12 | 65-99 % | 143 s | 23 |
| Qwen3.6-35B-A3B (MoE, ~3B active), capped | 9 / 12 | 47-91 % | 63 s | 23.5 |
| Qwen3.6-35B-A3B, off | 8 / 12 | 39-86 % | 47 s | 20 |

- **The dense 27B finished more:** 23 of 24 runs against the MoE's 17 of 24 (Fisher two-sided
  p = 0.048). The MoE was 3-4.5x faster per task.
- **Thinking made no detectable difference** at this size (p = 1.0 for either model), and on the
  27B it doubled the time. A single pass would have said otherwise: the MoE went 6/6 then 3/6 with
  thinking, 3/6 then 5/6 without.
- **How the MoE failed:** twice in one turn, both with thinking off on the rename task (a tool call
  written out as text the server's parser did not accept, then echoed harness text), and five
  times with a wrong fix: the backoff precedence, a bash failure path (twice), the rename's
  untested positional caller, the export cursor. The 27B's one miss was the backoff
  precedence.

Six small tasks is a small suite. The model gap is borderline and would need more tasks, not more
passes of these, to firm up.

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

`QWEN_BACKEND` is the OpenAI-compatible server itself; the script reads the context window from its
`/v1/models`. Any agent command works after `--`: the runner appends `-p <prompt> --output-format
json` and runs it in the task's working copy.

-- Claude Opus 5.5, working in [helm](https://github.com/akapug/helm) for @akapug
