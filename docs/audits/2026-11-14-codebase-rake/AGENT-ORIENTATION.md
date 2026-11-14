# Orientation for `@akapug`'s Agents: Closing Seams Without Literalist Amplification

This briefing is addressed **agent-to-agent** to every Claude, Codex, Gemini, and local Qwen/DeepSeek session operating in `@akapug`'s repositories (`builders-dev`, `fleet-gen3`, `homelab-atlas`, `homelab`).

## 1. Obey the Physics Already Proven in This Estate

Across `homelab-atlas` and `b70-rig`, empirical measurements established five hard constraints that govern every agent session here:

1. **Per-Model Comprehension Depth vs. VRAM Capacity**:
   - On `Qwen3.8-27B int4`, planted-defect recall dropped from **`9/9` (`100%`) at `2–12k` tokens to `3/9` at `~23k` (`10/36` across padded depths)**, with `12–23k` and other model families still unmeasured (treat unmeasured models as `unknown`, not hard-failed or assumed safe).
   - Every rule file, per-turn hook whisper, and un-trimmed tool schema directly taxes reasoning accuracy and busts prefix KV caches.
2. **Pre-Turn-1 Token Discipline (`request-census` Standard)**:
   - Unmodified harnesses previously burned `33,338` tokens before Turn 1 (`19,579` from 25 built-in harness tool schemas + `11,632` from injected rules); trimming unused built-in tools via a disallow file plus static rules cut startup to `8,734` tokens.
   - Keep project-controlled startup (`AGENTS.md` + active slice brief) `<= 2,500` tokens, default MCP `tools/list` `<= 2,500` tokens, and per-turn dynamic store injection `0` tokens by default (`0` on `bg-result` / `peer-wake`).
3. **Local Seats Are Bounded Authors Under an Executed Judge, Not Diff Reviewers**:
   - Measured panels show local models succeed as **authors on bounded Cards when graded by executed hidden tests** (`22/24` on 27B int4, `11/12` on Flash-Next), while performing poorly as **diff reviewers** (`0–3 of 29` flawed changes caught, with false alarms on clean controls).
   - Admit local lanes via **executed outcome probes** (or distance from that model family's own perturbation floor — never an unattainable global `EARf >= 0.98` hard-block), and reserve automated diff critique for top-ranked frontier families bound to `(base_sha, head_sha)`.
4. **Bounded Slot Admission (`door32`) Over Quota-Burn Motion**:
   - Over-subscribing concurrent agents past KV/CPU slot capacity collapses throughput (`79.1 -> 8.0 tok/s`), whereas bounded slot admission restores `85.3 tok/s`.
   - **Idle is a valid, healthy state.** Never invent synthetic tasks, split one logical change into multi-card chains, or re-stage work merely because subscription quota is under 90% burn. When the ledger hits zero, stop.
5. **Zero Silent Fallbacks, Honest Mechanism, & Runnable Contracts**:
   - Follow `homelab-atlas`'s standard: print `FALSIFIED` or return typed `{ status: "unavailable" | "failed", reason }` when a dependency, model, or check fails. Never emit stubbed deliberations, placeholder terminal output, or fabricated metrics (`branch: "main"` on 500, `"connected"` without a live socket) on live paths.
   - Every command or file named in `AGENTS.md` must actually exist and run cleanly in a fresh clone.

## 2. Five Anti-Patterns Previous Agent Generations Must Stop Repeating

`@akapug` steers product direction and taste (`"filter through /decision-spirit"`). Previous agent sessions repeatedly damaged `builders-dev` and `fleet-gen3` by compiling casual chat remarks into rigid global machinery:

1. **Stop Compiling Casual Remarks Into Global AST Scanners, Stop-Refusal Loops, or Input Rewrites**:
   - Fix concrete seams with targeted regression tests. Do **not** build multi-thousand-line AST push refusers for house style, regexes on an agent's final sentence, Stop hooks that force the model to manage message-delivery plumbing (`fleet chat wait`), or silent Bash input mutations (`renice`, forced `run_in_background=True`).
2. **Stop Hoarding Superseded Code or Neutering by Config Switch Instead of Deleting**:
   - Protecting undiscoverable real features from naive 3-day usage counters does **not** make dead code, duplicate shells, redirect graveyards, or superseded prototypes immortal (`11-12` owner clarification: *"if they are present in new fleet or superseded or whatever of course they can be cut, why wouldnt you?"*).
   - **Deletion and consolidation are free** whenever convergence checks stay green; **net-new** Workers, Durable Objects, top-level routes, D1 tables, and hooks require explicit owner sign-off.
3. **Interpret `(h + a)^n` Symmetry at the Domain Capability Layer, Never Page-Per-RPC**:
   - Humans and agents operate on the same **six canonical nouns** (`Team · Project · Card · Run · Room · Review`, plus `Inbox` as a personal filter) anchored at `/p/:project`. Internal plumbing RPCs do not get dedicated human pages or startup MCP schemas.
4. **Separate TCB Containment From Inner-Loop Construction Clocks**:
   - Hard `PreToolUse` blocks (`exit 2`) belong strictly on **protected TCB/host paths** (`judge.json`, `.judge/**`, workflows, `~/.git-hooks`, `~/.claude`, `~/.codex`) and **destructive git commands**.
   - Slice `owns:` globs are **advisory overlap leases**, not hard edit ceilings — never duplicate shared types locally just to avoid touching `packages/shared-types`.
   - Keep inner-loop checks fast (`<5s` scoped `tsc` / single-file unit test); run full E2E + behavioral `#tooth` verification at **wave/candidate convergence** on prospective merge tree `T_M`, reusing HMAC-authenticated cached receipts when a check's input cone is unchanged.
5. **Keep Mechanism vs. Claim (`attested` vs. `observed` / `decided`) Strictly Separate**:
   - Snapshotting a run's working tree to a run-scoped ref on exit is **mechanism**; opening a PR, naming a feature branch, or advancing card status is a **claim/policy** that requires clean exit + observed verification.
   - Commit prefixes (`feat():`), env vars (`FAB_PUSH_TESTED`), agent Stop events, checklist boxes, and LLM diff critiques are **`attested`**. Only authenticated out-of-lane `JudgeReceipt`s and verified repository/webhook events are **`observed` / `decided`**.

## 3. What Product You Are Actually Building Together

Keep every slice aligned with the core product emerging across these repos:
- **One coherent Studio loop at `/p/:project`**: `Intend (Card on Tufte-clean Project Map) -> Run (real Run object with one run_id, isolated worktree + repo-scoped auth in cloud Sandbox or attached local/fabd runner behind slot admission) -> See (backed Diff, server-recorded Checks, unified Event-Spine Activity, diff-bound Review, and Context Gauge) -> Land (authenticated T_M merge door)`.
- Close vertical end-to-end seams completely before proposing any horizontal breadth; quarantine playful cultures, experimental evals, and optional chat bridges into zero-overhead extensions outside the core runtime and startup token budget.
