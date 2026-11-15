# Changing a local coding model's habits: a null and a baseline

Speed is half of a local coding model; the other half is what it does with a task. We test "behaviour levers" on our
coding model: a system prompt, a small weight edit (LoRA), a control vector, a neuron edit. Each lever is fit on one
model file and judged against the prompt alone, with its predictions written down before any answer is graded.
Recipe Maker's `--check` carries the rules ([behaviour levers](../recipe-maker/README.md#behaviour-levers-levers)).

Two results from 2026-11-14, both on Qwen3.8 Flash-Next Coder (GSQ-RCO quants). Negative and early results go in,
because they save the next person the run. The task sets and their graded answers are private (a public task set
stops measuring anything once models train on it), so these pages give the design and the grades, not the items.

## 1. A trained LoRA on the last five blocks does not change when the model stops: NULL

**The habit.** On a code-review task with a planted defect, the model often keeps hunting until its token budget runs
out instead of stating a verdict. A closed-form edit of the same blocks was already NULL (16 of 16 cells under the
floor below), so the preregistered fallback was a trained edit.

**The run.** A rank-8 LoRA on 7 targets of the last five blocks (two attention outputs, five shared-expert down
projections), trained with [the tail trainer](../tail-portable/) on an RTX 3090 Ti to match a teacher's next-token
distribution (the same model with demonstrations in context) at the decision points: 72 hunts (24 items x 3 seeds),
3,581 decision rows (2,093 clean), AdamW lr 1e-3, 4 epochs, leave-one-family-out over 12 folds (about 23 minutes each).

**The result.** Closure = 1 - KL(teacher || edited) / KL(teacher || unedited) on the held-out family's clean rows, with
a floor of 0.25 set before the run:

| | held-out closure | train closure |
|---|---|---|
| mean of 12 folds | **-0.017** (pooled -0.026) | 0.137 |
| best fold | 0.141 | 0.310 |
| folds at or above 0.25 | **0 of 12** | |

The edit fits its own training rows a little and transfers nothing to a held-out family; planted rows stay put (KL to
the base 0.002-0.007). It stopped at the floor, so the preregistered screen and test were never asked.

- **Overfit smoke, also NULL** (run after the preregistration, beside it): the best fold's adapter and a norm-matched
  random twin, on 4 items the adapter was trained on, 6,144-token budget. Short prompt: 0 verdicts, 4 of 4 answers ran
  to the budget. Short prompt plus the edit: the same. The 30-minute wall stopped it after 8 of 18 planned answers.
- **Why even a perfect fit would not help** (our reading, from training rows only): the first token of the path that
  states a verdict has P = 1.25 % unedited and 1.37 % under the teacher. Over a whole hunt, the chance that this token is ever drawn
  moves from 0.49 to 0.53 (an upper bound: the rest of the path must follow). The teacher itself barely changes when the model stops, so matching it
  cannot either. The demonstrations change what the model looks at, not when it stops.
- **Flags for anyone reusing the method:** small headroom (the unedited model starts at about 0.015 nats per row from
  the teacher, and four epochs lower that by about 20 %); AdamW at lr 1e-3 was unstable (11 of 12 folds ended above
  their epoch-1 or epoch-2 training loss).

Next: a lever has to change the teacher, not the fit: a stronger demonstration set, or a budget or stop rule outside
the weights.

## 2. With no file list, the base model patches the symptom: a habit target at last

**The test.** Small repositories where the honest fix sits away from where the bug shows. Four habits, each with
treatment items and controls (where the near move is right): H1 DUPLICATE (writes new code where an existing helper
already does most of it), H2 SYMPTOM (patches the file the report names when the cause is upstream), H3 GREEN (calls a
vacuous green test suite done), H4 ONE-CALLER (changes a contract and misses its other callers). Version 3 of the set: 24 families (36 items), the prompt lists no
files, and the far files, their folders and functions carry opaque names. Two arms: the base model, and the same with
a short system prompt that names the habits ("fix the cause, not where it shows"). The model: Flash-Next Coder
GSQ-RCO IQ1_M on a GTX 1080 Ti. One answer per item per arm, 72 answers, graded by hidden tests; predictions
registered before any answer.

| habit | way | base: myopic | predicted | short prompt: myopic | predicted |
|---|---|---|---|---|---|
| H1 DUPLICATE | repo | 0/4 | 3/4 (2-4) | 0/4 | 2/4 (1-3) |
| H1 | prompt | 0/4 | 0/4 (0-1) | 0/4 | 0/4 (0-1) |
| H2 SYMPTOM | repo | **4/4** (pass 0/4, Wilson 95 % [0, 49]) | 1/4 (0-2) | **0/4** (pass 4/4, [51, 100]) | 0/4 (0-1) |
| H3 GREEN | repo | 0/4 | 0/4 (0-1) | 0/4 | 0/4 (0-1) |
| H4 ONE-CALLER | repo | 1/4 | 2/4 (1-4) | 0/4 | 1/4 (0-2) |
| H4 | prompt | 0/4 | 0/4 (0-1) | 0/4 | 0/4 (0-1) |
| controls (12) | | 12/12 pass | at least 10/12 | 12/12 pass | |

"Myopic" = a treatment item that fails; "way repo" hides the far code in the repository, "way prompt" shows it in the
prompt.

- **H2 is the habit this version exposes.** With no file list, the base model trusted the file the report names and
  patched it there, 4 of 4 (11-12 steps each). The short prompt changed the upstream cause and added a test, 4 of 4,
  at about 6 more steps. n = 1 per item, so 0/4 against 4/4 is a direction (Fisher's exact p = 0.029), not a rate.
- **H1 refutes our retrieval reading.** We predicted that opaque names and no file list would hide the existing helper
  (+3 items for "repo" over "prompt"). Observed: +0. The model found every far helper by content search. So the cost of
  far code is not search.
- The model searched less than we predicted (median 12 steps base, 15 short, against 15-22 and 17-24).

Next: replicates first (the 4 H2 items x 4 seeds x both arms), then a lever has three questions to answer: does it
reach the prompt's result without the prompt, does it cut the prompt's extra steps, and does it hold where the prompt
alone fails (which needs harder H2 items).

---

-- Claude Opus 5.5 and GPT-6.1 Sol (Codex), working in [fleet](https://github.com/akapug/fleet) for [@akapug](https://github.com/akapug)
