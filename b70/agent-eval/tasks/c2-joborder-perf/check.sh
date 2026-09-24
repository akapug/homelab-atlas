#!/usr/bin/env bash
# Hidden check for c2-joborder-perf. Run from the root of the agent's working
# copy. Exit 0 only if `jobq order` prints exactly what the original algorithm
# prints (order, tie-breaks, cycle errors) on many small job files, and handles
# a 100,000-job file, and the same file with a cycle, within a time limit.
set -u
root=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failures=0
ok()   { echo "ok   - $1"; }
fail() { echo "FAIL - $1"; failures=$((failures + 1)); }
ulimit -v 8000000 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
LIMIT=10   # seconds for one 100,000-job run; the reference fix takes about 0.4

# 1. The repo's own tests pass.
if timeout -k 2 25 python3 -m unittest discover -s tests -t . >"$work/own.txt" 2>&1; then
    ok "repo tests pass"
else
    fail "repo tests pass"; tail -30 "$work/own.txt"
fi

cat >"$work/ref.py" <<'PY'
"""Job-file generators and two references: `slow` is the original algorithm
(small inputs only), `fast` an O(E log V) equivalent for the large inputs."""
import heapq
import random


def jobs_of(text):
    jobs = []
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            f = line.split()
            jobs.append((f[0], int(f[1]), tuple(f[2].split(",")) if len(f) == 3 else ()))
    return jobs


def cli_result(names, stuck):
    """(exit code, stdout, last stderr line) as jobq prints them."""
    if stuck:
        return 1, "", "error: dependency cycle among: " + ", ".join(sorted(stuck))
    return 0, "".join(n + "\n" for n in names), ""


def slow(text):
    done, pending = [], jobs_of(text)
    while pending:
        ready = [j for j in pending if all(d in done for d in j[2])]
        if not ready:
            return cli_result(None, [j[0] for j in pending])
        best = ready[0]
        for j in ready[1:]:
            if j[1] > best[1]:
                best = j
        done.append(best[0])
        pending.remove(best)
    return cli_result(done, None)


def fast(text):
    jobs = jobs_of(text)
    index = {name: i for i, (name, _, _) in enumerate(jobs)}
    waiting = [len(set(deps)) for _, _, deps in jobs]
    dependents = [[] for _ in jobs]
    for i, (_, _, deps) in enumerate(jobs):
        for d in set(deps):
            dependents[index[d]].append(i)
    heap = [(-p, i) for i, (_, p, _) in enumerate(jobs) if not waiting[i]]
    heapq.heapify(heap)
    out, ran = [], [False] * len(jobs)
    while heap:
        _, i = heapq.heappop(heap)
        out.append(jobs[i][0])
        ran[i] = True
        for k in dependents[i]:
            waiting[k] -= 1
            if not waiting[k]:
                heapq.heappush(heap, (-jobs[k][1], k))
    stuck = [jobs[i][0] for i in range(len(jobs)) if not ran[i]]
    return cli_result(out, stuck)


POOL = ("alpha beta gamma delta zed mike kilo a b c x y z a1 a2 a10 b/web b/api "
        "db.migrate ops-deploy ml_train lint test-unit test-e2e fetch pack sign "
        "report docs cache warm cold Zulu Echo").split()


def small(rng):
    n = rng.randint(1, 28)
    names = rng.sample(POOL, n)
    rank = list(range(n))
    rng.shuffle(rank)
    lines = []
    for i, name in enumerate(names):
        before = [names[k] for k in range(n) if rank[k] < rank[i]]
        deps = rng.sample(before, min(len(before), rng.choice([0, 0, 1, 1, 2, 3])))
        if deps and rng.random() < 0.15:
            deps.append(deps[0])                    # the same dependency twice
        if rng.random() < 0.05:
            deps.append(rng.choice(names))          # may close a cycle, or be a self-dependency
        line = f"{name} {rng.randint(-2, 3)}" + (" " + ",".join(deps) if deps else "")
        lines.append(line + ("   # note" if rng.random() < 0.1 else ""))
        if rng.random() < 0.08:
            lines.append("")
    return "\n".join(lines) + "\n"


FIXED = [
    "x 1 y\ny 5\nz 1\n",             # a tie decided by declaration order, not readiness order
    "a 0\n",
    "solo 3 solo\n",
    "b 1 a\na 1 b\nc 9\nd 0 c,b\n",
    "p 2 q,q,q\nq -1\nr 2\n",
    "n3 0\nn2 0\nn1 0\nn0 0 n1,n2,n3\n",
]


def big(n, seed, cycle=False):
    """n jobs, declared in random order, each needing up to 3 jobs from the
    2,000 before it in a hidden topological order; priorities 0-4 (many ties)."""
    rng = random.Random(seed)
    groups = ["api", "web", "db", "ml", "ops", "docs", "infra", "mobile"]
    names = [f"{rng.choice(groups)}/{rng.choice(['build', 'test', 'lint', 'pack', 'ship'])}-{k:05d}"
             for k in rng.sample(range(n), n)]
    deps = []
    for r in range(n):
        lo = max(0, r - 2000)
        d = [names[rng.randrange(lo, r)] for _ in range(rng.choice([0, 1, 2, 2, 3]))] if r else []
        deps.append(d)
    if cycle:
        a, b = n // 2, n // 2 + 7
        deps[b].append(names[a])
        deps[a].append(names[b])
    order = list(range(n))
    rng.shuffle(order)
    lines = [f"{names[r]} {rng.randint(0, 4)}" + (" " + ",".join(deps[r]) if deps[r] else "")
             for r in order]
    return "\n".join(lines) + "\n"
PY

cat >"$work/hidden_test.py" <<'PY'
import os
import random
import subprocess
import sys
import unittest

sys.path.insert(0, os.environ["CHECK_WORK"])
import ref  # noqa: E402

ROOT, WORK = os.environ["CHECK_ROOT"], os.environ["CHECK_WORK"]
LIMIT = int(os.environ["CHECK_LIMIT"])


def jobq(text, name, timeout):
    path = os.path.join(WORK, name)
    with open(path, "w") as f:
        f.write(text)
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run([sys.executable, "-m", "jobq", "order", path], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=timeout)
    err = p.stderr.strip().splitlines()
    return p.returncode, p.stdout, (err[-1] if err and p.returncode else "")


class Hidden(unittest.TestCase):
    def run_big(self, cycle):
        text = ref.big(100000, 7, cycle=cycle)
        try:
            got = jobq(text, "big.jobs", LIMIT)
        except subprocess.TimeoutExpired:
            self.fail(f"jobq order took longer than {LIMIT}s on 100,000 jobs")
        want = ref.fast(text)
        self.assertEqual(got[0], want[0], got[2][:300])
        self.assertTrue(got[1] == want[1], "order differs on the 100,000-job file")
        self.assertTrue(got[2] == want[2], "cycle error differs on the 100,000-job file: " + got[2][:200])

    def test_1_small_files_match_the_original_algorithm(self):
        rng = random.Random(20260923)
        cases = ref.FIXED + [ref.small(rng) for _ in range(140)]
        for n, text in enumerate(cases):
            with self.subTest(case=n, jobs=text):
                self.assertEqual(jobq(text, "small.jobs", 10), ref.slow(text))

    def test_2_large_graph(self):
        self.run_big(False)

    def test_3_large_graph_with_cycle(self):
        self.run_big(True)


if __name__ == "__main__":
    unittest.main(verbosity=2, failfast=True)
PY
if CHECK_ROOT="$root" CHECK_WORK="$work" CHECK_LIMIT="$LIMIT" timeout -k 2 85 \
        python3 "$work/hidden_test.py" >"$work/hidden.txt" 2>&1; then
    ok "hidden order and speed tests"
else
    fail "hidden order and speed tests"; tail -40 "$work/hidden.txt" | cut -c1-400
fi

echo
[ "$failures" -eq 0 ] && { echo "CHECK PASSED"; exit 0; }
echo "CHECK FAILED: $failures"
exit 1
