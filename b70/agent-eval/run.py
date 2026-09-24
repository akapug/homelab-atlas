#!/usr/bin/env python3
"""Run repo-scale coding tasks through an agent and score them with hidden checks.

Each task is tasks/<id>/: repo/ (the fixture), prompt.md (what the agent is asked), check.sh (run in
the agent's working copy afterwards; exit 0 = done; never shown to the agent), solution.patch (a
reference fix) and meta.json. Why it exists: short-function benchmarks (HumanEval and the like)
could not tell our local models apart, while the work we want from them is finishing changes in a
repository, and a coding agent that runs is not yet one that finishes.

    run.py --verify                                   # every fixture: check fails, then passes with the fix
    run.py --label mymodel -- bash $PWD/sandbox-agent.sh      # with QWEN_* set, see sandbox-agent.sh
    run.py --label opus -- claude --dangerously-skip-permissions --model opus
Everything after "--" is the agent command; the runner appends: -p <prompt> --output-format json,
and runs it with the task's working copy as cwd (QWEN_CWD is set too). One JSON line per task goes to
--out (default results/<label>.jsonl), and a summary to stdout.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
GIT = ["git", "-c", "user.name=eval", "-c", "user.email=eval@example.com"]


def fixture(task, into):
    """The task's repo/ as a fresh git repository with one commit, at `into`."""
    shutil.copytree(os.path.join(task, "repo"), into)
    for cmd in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "fixture"]):
        subprocess.run(GIT + ["-C", into] + cmd, check=True, capture_output=True)


def check(task, wd, timeout=300):
    """Exit code of the task's hidden check, run from the working copy (124 on timeout)."""
    try:
        return subprocess.run(["bash", os.path.join(task, "check.sh")], cwd=wd, capture_output=True,
                              text=True, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return 124


def verify(tasks):
    bad = 0
    for t in tasks:
        with tempfile.TemporaryDirectory() as d:
            wd = os.path.join(d, "wd")
            fixture(t, wd)
            before = check(t, wd)
            ap = subprocess.run(["git", "-C", wd, "apply", os.path.join(t, "solution.patch")], capture_output=True, text=True)
            after = check(t, wd) if ap.returncode == 0 else f"patch failed: {ap.stderr.strip()[:120]}"
        ok = before != 0 and after == 0
        bad += not ok
        print(f"{'OK ' if ok else 'BAD'} {os.path.basename(t)}: check before the fix {before}, after {after}")
    return bad


def run(tasks, label, cmd, timeout, out):
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    passed = 0
    with open(out, "a") as fh:
        for t in tasks:
            tid = os.path.basename(t)
            prompt = open(os.path.join(t, "prompt.md")).read().strip()
            with tempfile.TemporaryDirectory() as d:
                wd = os.path.join(d, "wd")
                fixture(t, wd)
                env = dict(os.environ, QWEN_CWD=wd)
                t0 = time.time()
                try:
                    p = subprocess.run(cmd + ["-p", prompt, "--output-format", "json"], cwd=wd, env=env,
                                       capture_output=True, text=True, timeout=timeout)
                    raw, rc = p.stdout, p.returncode
                except subprocess.TimeoutExpired as e:
                    raw, rc = (e.stdout or b"").decode(errors="replace") if isinstance(e.stdout, bytes) else (e.stdout or ""), "timeout"
                wall = time.time() - t0
                try:
                    j = json.loads(raw.strip().splitlines()[-1])
                except (ValueError, IndexError):
                    j = {}
                code = check(t, wd)
                # against the fixture commit, not HEAD: agents often commit their fix
                base = subprocess.run(["git", "-C", wd, "rev-list", "--max-parents=0", "HEAD"], capture_output=True, text=True).stdout.split()
                diff = subprocess.run(["git", "-C", wd, "diff", "--stat"] + base[-1:], capture_output=True, text=True).stdout.strip().splitlines()
            u = j.get("usage") or {}
            row = {"task": tid, "label": label, "pass": code == 0, "check_rc": code, "agent_rc": rc,
                   "wall_s": round(wall, 1), "turns": j.get("num_turns"), "is_error": j.get("is_error"),
                   "out_tokens": u.get("output_tokens"), "in_tokens": u.get("input_tokens"),
                   "cache_read": u.get("cache_read_input_tokens"), "diff": diff[-1] if diff else "",
                   "result": (j.get("result") or "")[-300:]}
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            passed += row["pass"]
            print(f"{'PASS' if row['pass'] else 'FAIL'} {tid}: {row['wall_s']}s, {row['turns']} turns, "
                  f"check {code}, agent {rc}, {row['diff'] or 'no changes'}", flush=True)
    print(f"{label}: {passed} of {len(tasks)} tasks passed")
    return passed


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="check every fixture against its reference fix")
    ap.add_argument("--label", default="agent")
    ap.add_argument("--tasks", nargs="*", default=[], help="task ids (default: all)")
    ap.add_argument("--timeout", type=int, default=1200, help="seconds per task (default 1200)")
    ap.add_argument("--out", default="")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    root = os.path.join(HERE, "tasks")
    ids = a.tasks or sorted(d for d in os.listdir(root) if os.path.isfile(os.path.join(root, d, "check.sh")))
    tasks = [os.path.join(root, i) for i in ids]
    if a.verify:
        sys.exit(1 if verify(tasks) else 0)
    cmd = a.cmd[1:] if a.cmd[:1] == ["--"] else a.cmd
    if not cmd:
        ap.error("give the agent command after --")
    run(tasks, a.label, cmd, a.timeout, a.out or os.path.join(HERE, "results", f"{a.label}.jsonl"))


if __name__ == "__main__":
    main()
