#!/usr/bin/env bash
# Hidden check for c3-quote-cache-alias. Run from the root of the agent's
# working copy. Exit 0 only if every quote in a batch equals the quote for that
# order run on its own (no state leaks between orders), the batch completes,
# a genuinely bad order still stops the batch, and the repo's tests pass.
set -u
root=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
failures=0
ok()   { echo "ok   - $1"; }
fail() { echo "FAIL - $1"; failures=$((failures + 1)); }
ulimit -v 8000000 2>/dev/null || true
export PYTHONDONTWRITEBYTECODE=1
unset QUOTER_DATA

# 1. The repo's own tests pass.
if timeout -k 2 30 python3 -m unittest discover -s tests -t . >"$work/own.txt" 2>&1; then
    ok "repo tests pass"
else
    fail "repo tests pass"; tail -30 "$work/own.txt"
fi

# 2. Hidden batch tests through the CLI.
cat >"$work/hidden_test.py" <<'PY'
import json
import os
import subprocess
import sys
import unittest

ROOT, WORK = os.environ["CHECK_ROOT"], os.environ["CHECK_WORK"]

# Chosen so that state kept between orders shows up: add-on lines leaking into
# later quotes on the same plan, volume discounts compounding on a plan's seat
# price, and on a shared add-on's per-seat price across plans.
ORDERS = [
    ("H-01", "team", 12, ["sso"]),
    ("H-02", "team", 8, []),
    ("H-03", "team", 60, ["audit-log"]),
    ("H-04", "team", 10, []),
    ("H-05", "enterprise", 250, ["sso"]),
    ("H-06", "team", 5, ["sso", "sandbox"]),
    ("H-07", "starter", 55, ["priority-support"]),
    ("H-08", "starter", 4, ["priority-support", "sandbox"]),
    ("H-09", "enterprise", 20, []),
    ("H-10", "team", 60, []),
    ("H-11", "enterprise", 300, ["audit-log", "sandbox"]),
    ("H-12", "team", 1, ["priority-support", "sandbox"]),
    ("H-13", "starter", 2, []),
    ("H-14", "team", 12, ["sso"]),
    ("H-15", "enterprise", 200, ["sso", "audit-log", "sandbox"]),
    ("H-16", "team", 50, ["sso", "priority-support", "audit-log", "sandbox"]),
]

# The record for each order quoted alone, in a fresh process, by the original code.
GOLDEN = json.loads(r"""{
"H-01": {"quote": "H-01", "customer": "Customer H-01", "plan": "team", "seats": 12, "discount_pct": 0, "lines": [["base-team", 1, 9900, 9900], ["seat-team", 12, 1500, 18000], ["sso", 12, 300, 3600]], "total_cents": 31500},
"H-02": {"quote": "H-02", "customer": "Customer H-02", "plan": "team", "seats": 8, "discount_pct": 0, "lines": [["base-team", 1, 9900, 9900], ["seat-team", 8, 1500, 12000]], "total_cents": 21900},
"H-03": {"quote": "H-03", "customer": "Customer H-03", "plan": "team", "seats": 60, "discount_pct": 10, "lines": [["base-team", 1, 9900, 9900], ["seat-team", 60, 1350, 81000], ["audit-log", 1, 4900, 4900]], "total_cents": 95800},
"H-04": {"quote": "H-04", "customer": "Customer H-04", "plan": "team", "seats": 10, "discount_pct": 0, "lines": [["base-team", 1, 9900, 9900], ["seat-team", 10, 1500, 15000]], "total_cents": 24900},
"H-05": {"quote": "H-05", "customer": "Customer H-05", "plan": "enterprise", "seats": 250, "discount_pct": 20, "lines": [["base-enterprise", 1, 49900, 49900], ["seat-enterprise", 250, 2000, 500000], ["support-premium", 1, 20000, 20000], ["sso", 250, 240, 60000]], "total_cents": 629900},
"H-06": {"quote": "H-06", "customer": "Customer H-06", "plan": "team", "seats": 5, "discount_pct": 0, "lines": [["base-team", 1, 9900, 9900], ["seat-team", 5, 1500, 7500], ["sso", 5, 300, 1500], ["sandbox", 1, 2500, 2500]], "total_cents": 21400},
"H-07": {"quote": "H-07", "customer": "Customer H-07", "plan": "starter", "seats": 55, "discount_pct": 10, "lines": [["base-starter", 1, 2900, 2900], ["seat-starter", 55, 810, 44550], ["priority-support", 55, 180, 9900]], "total_cents": 57350},
"H-08": {"quote": "H-08", "customer": "Customer H-08", "plan": "starter", "seats": 4, "discount_pct": 0, "lines": [["base-starter", 1, 2900, 2900], ["seat-starter", 4, 900, 3600], ["priority-support", 4, 200, 800], ["sandbox", 1, 2500, 2500]], "total_cents": 9800},
"H-09": {"quote": "H-09", "customer": "Customer H-09", "plan": "enterprise", "seats": 20, "discount_pct": 0, "lines": [["base-enterprise", 1, 49900, 49900], ["seat-enterprise", 20, 2500, 50000], ["support-premium", 1, 20000, 20000]], "total_cents": 119900},
"H-10": {"quote": "H-10", "customer": "Customer H-10", "plan": "team", "seats": 60, "discount_pct": 10, "lines": [["base-team", 1, 9900, 9900], ["seat-team", 60, 1350, 81000]], "total_cents": 90900},
"H-11": {"quote": "H-11", "customer": "Customer H-11", "plan": "enterprise", "seats": 300, "discount_pct": 20, "lines": [["base-enterprise", 1, 49900, 49900], ["seat-enterprise", 300, 2000, 600000], ["support-premium", 1, 20000, 20000], ["audit-log", 1, 4900, 4900], ["sandbox", 1, 2500, 2500]], "total_cents": 677300},
"H-12": {"quote": "H-12", "customer": "Customer H-12", "plan": "team", "seats": 1, "discount_pct": 0, "lines": [["base-team", 1, 9900, 9900], ["seat-team", 1, 1500, 1500], ["priority-support", 1, 200, 200], ["sandbox", 1, 2500, 2500]], "total_cents": 14100},
"H-13": {"quote": "H-13", "customer": "Customer H-13", "plan": "starter", "seats": 2, "discount_pct": 0, "lines": [["base-starter", 1, 2900, 2900], ["seat-starter", 2, 900, 1800]], "total_cents": 4700},
"H-14": {"quote": "H-14", "customer": "Customer H-14", "plan": "team", "seats": 12, "discount_pct": 0, "lines": [["base-team", 1, 9900, 9900], ["seat-team", 12, 1500, 18000], ["sso", 12, 300, 3600]], "total_cents": 31500},
"H-15": {"quote": "H-15", "customer": "Customer H-15", "plan": "enterprise", "seats": 200, "discount_pct": 20, "lines": [["base-enterprise", 1, 49900, 49900], ["seat-enterprise", 200, 2000, 400000], ["support-premium", 1, 20000, 20000], ["sso", 200, 240, 48000], ["audit-log", 1, 4900, 4900], ["sandbox", 1, 2500, 2500]], "total_cents": 525300},
"H-16": {"quote": "H-16", "customer": "Customer H-16", "plan": "team", "seats": 50, "discount_pct": 10, "lines": [["base-team", 1, 9900, 9900], ["seat-team", 50, 1350, 67500], ["sso", 50, 270, 13500], ["priority-support", 50, 180, 9000], ["audit-log", 1, 4900, 4900], ["sandbox", 1, 2500, 2500]], "total_cents": 107300}
}""")


def order(oid, plan, seats, addons):
    o = {"id": oid, "customer": f"Customer {oid}", "plan": plan, "seats": seats}
    if addons:
        o["addons"] = addons
    return o


def batch(orders, name):
    path, out = os.path.join(WORK, name + ".jsonl"), os.path.join(WORK, name + ".out.jsonl")
    with open(path, "w") as f:
        f.writelines(json.dumps(o) + "\n" for o in orders)
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run([sys.executable, "-m", "quoter", "batch", path, "-o", out], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=30)
    records = []
    if os.path.exists(out):
        with open(out) as f:
            records = [json.loads(line) for line in f if line.strip()]
    return p.returncode, records, p.stderr


def expected(oid, as_id=None):
    rec = dict(GOLDEN[oid])
    rec["quote"] = as_id or oid
    rec["customer"] = f"Customer {as_id or oid}"
    return rec


class Hidden(unittest.TestCase):
    def check_batch(self, orders, name):
        code, records, err = batch(orders, name)
        self.assertEqual(code, 0, "the batch did not complete:\n" + err[-1500:])
        self.assertEqual([r.get("quote") for r in records], [o["id"] for o in orders])
        for o, rec in zip(orders, records):
            with self.subTest(order=o):
                self.assertEqual(rec, expected(o["id"].split("/")[0], o["id"]))

    def test_each_quote_in_a_batch_equals_the_quote_alone(self):
        self.check_batch([order(*o) for o in ORDERS], "forward")

    def test_same_orders_again_in_reverse_in_one_run(self):
        orders = [order(*o) for o in ORDERS]
        again = [dict(o, id=o["id"] + "/again", customer=f"Customer {o['id']}/again") for o in reversed(orders)]
        self.check_batch(orders + again, "twice")

    def test_a_bad_order_still_stops_the_batch(self):
        bad = [order("H-13", "starter", 2, []), order("X-1", "starter", 3, ["sso"]), order("H-09", "enterprise", 20, [])]
        code, records, err = batch(bad, "bad")
        self.assertNotEqual(code, 0, "an order with an add-on its plan does not sell must fail the batch")
        self.assertNotIn("X-1", [r.get("quote") for r in records])


if __name__ == "__main__":
    unittest.main(verbosity=2)
PY
if CHECK_ROOT="$root" CHECK_WORK="$work" timeout -k 2 80 python3 "$work/hidden_test.py" >"$work/hidden.txt" 2>&1; then
    ok "hidden batch tests"
else
    fail "hidden batch tests"; tail -60 "$work/hidden.txt" | cut -c1-400
fi

echo
[ "$failures" -eq 0 ] && { echo "CHECK PASSED"; exit 0; }
echo "CHECK FAILED: $failures"
exit 1
