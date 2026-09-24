#!/usr/bin/env bash
# Hidden check for b3-rename-money-fmt. Run from the root of the agent's
# working copy. Exit 0 only if the refactor is complete and behavior is
# unchanged.
set -u
root=$(pwd)
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
export PYTHONIOENCODING=utf-8 PYTHONDONTWRITEBYTECODE=1

# 1. The repo's own suite passes.
if ! python3 -m unittest discover -s tests -t . >"$work/own.txt" 2>&1; then
    echo "FAIL - repo test suite does not pass"; tail -n 30 "$work/own.txt"; exit 1
fi
echo "ok   - repo suite passes"

# 2. Hidden checks: the new API, and byte-identical output everywhere.
cat > "$work/hidden_check.py" <<'PY'
import importlib, inspect, json, os, subprocess, sys, tempfile, unittest

ROOT = os.environ["SHOP_ROOT"]
sys.path.insert(0, ROOT)

EXAMPLES = ["examples/inv-0042.json", "examples/inv-0043.json", "examples/inv-0041.json"]
EXTRA = {"number": "INV-0050", "date": "2026-10-02", "customer": "Zurich AG", "currency": "CHF",
         "lines": [{"desc": "Audit", "qty": 2, "unit": 99950}]}

GOLDEN = {
    'invoice': (
        'INVOICE INV-0042\n'
        'Date:    01 Sep 2026\n'
        'Bill to: Acme Corp\n'
        '\n'
        'Widget                  3 x       €12.50       €37.50\n'
        'Gadget, large           1 x    €1,234.56    €1,234.56\n'
        'Loyalty discount        1 x      -€25.00      -€25.00\n'
        '\n'
        'TOTAL                                       €1,247.06\n'
        '\n'
        'INVOICE INV-0043\n'
        'Date:    03 Sep 2026\n'
        'Bill to: Kaito Trading\n'
        '\n'
        'Consulting (hours)     12 x      \xa515,000     \xa5180,000\n'
        'Travel                  1 x       \xa54,280       \xa54,280\n'
        '\n'
        'TOTAL                                        \xa5184,280\n'
        '\n'
        'INVOICE INV-0041\n'
        'Date:    28 Aug 2026\n'
        'Bill to: Blue Fern LLC\n'
        '\n'
        'Support plan            1 x      $499.00      $499.00\n'
        'Refund: overcharge      1 x     -$600.00     -$600.00\n'
        '\n'
        'TOTAL                                        -$101.00\n'
    ),
    'report': (
        'INV-0041   28 Aug 2026  Blue Fern LLC          -$101.00\n'
        'INV-0042   01 Sep 2026  Acme Corp             €1,247.06\n'
        'INV-0043   03 Sep 2026  Kaito Trading          \xa5184,280\n'
    ),
    'csv': (
        'number,date,customer,currency,total\n'
        'INV-0042,2026-09-01,Acme Corp,EUR,"1,247.06"\n'
        'INV-0043,2026-09-03,Kaito Trading,JPY,"184,280"\n'
        'INV-0041,2026-08-28,Blue Fern LLC,USD,-101.00\n'
    ),
}
EXTRA_CSV = ('number,date,customer,currency,total\n'
             'INV-0050,2026-10-02,Zurich AG,CHF,"1,999.00"\n')
EXTRA_REPORT = 'INV-0050   02 Oct 2026  Zurich AG          CHF 1,999.00\n'


def shop(*argv):
    env = dict(os.environ, PYTHONPATH=ROOT)
    p = subprocess.run([sys.executable, "-m", "shop", *argv], cwd=ROOT, env=env,
                       capture_output=True, text=True, encoding="utf-8", timeout=60)
    return p.returncode, p.stdout, p.stderr


class NewApi(unittest.TestCase):
    def setUp(self):
        self.money = importlib.import_module("shop.money")

    def test_old_name_is_gone(self):
        self.assertFalse(hasattr(self.money, "fmt"), "shop.money.fmt should no longer exist")

    def test_signature(self):
        sig = inspect.signature(self.money.format_cents)
        params = list(sig.parameters.values())
        P = inspect.Parameter
        self.assertIn(params[0].kind, (P.POSITIONAL_ONLY, P.POSITIONAL_OR_KEYWORD))
        rest = {p.name: p for p in params[1:]}
        self.assertEqual(set(rest), {"currency", "show_symbol", "pad"})
        for p in rest.values():
            self.assertEqual(p.kind, P.KEYWORD_ONLY, p.name)
        self.assertEqual(rest["currency"].default, "USD")
        self.assertIs(rest["show_symbol"].default, True)
        self.assertEqual(rest["pad"].default, 0)

    def test_positional_flags_rejected(self):
        with self.assertRaises(TypeError):
            self.money.format_cents(100, "EUR")
        with self.assertRaises(TypeError):
            self.money.format_cents(100, "JPY", False, 8)

    def test_behavior(self):
        f = self.money.format_cents
        self.assertEqual(f(123456), "$1,234.56")
        self.assertEqual(f(-5, currency="EUR"), "-€0.05")
        self.assertEqual(f(1500, currency="JPY", show_symbol=False, pad=8), "   1,500")
        self.assertEqual(f(100, currency="CHF"), "CHF 1.00")
        self.assertEqual(f(-250, currency="GBP", pad=9), "   -\u00a32.50")

    def test_date_helper_untouched(self):
        import datetime
        dates = importlib.import_module("shop.dates")
        self.assertEqual(dates.fmt(datetime.date(2026, 9, 1)), "01 Sep 2026")


class SameOutput(unittest.TestCase):
    def check(self, cmd, want, files=EXAMPLES):
        code, out, err = shop(cmd, *files)
        self.assertEqual(code, 0, err)
        self.assertEqual(out, want)

    def test_invoice(self):
        self.check("invoice", GOLDEN["invoice"])

    def test_report(self):
        self.check("report", GOLDEN["report"])

    def test_csv(self):
        self.check("csv", GOLDEN["csv"])

    def test_unknown_currency(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "extra.json")
            with open(path, "w") as f:
                json.dump(EXTRA, f)
            self.check("csv", EXTRA_CSV, [path])
            self.check("report", EXTRA_REPORT, [path])


if __name__ == "__main__":
    unittest.main(verbosity=2)
PY
if SHOP_ROOT="$root" python3 "$work/hidden_check.py" >"$work/hidden.txt" 2>&1; then
    echo "ok   - hidden API and output checks"; echo "CHECK PASSED"; exit 0
fi
echo "FAIL - hidden API and output checks"; tail -n 50 "$work/hidden.txt"
exit 1
