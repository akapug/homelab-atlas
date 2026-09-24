import os
import unittest

from jobq.order import CycleError, run_order
from jobq.spec import parse

EXAMPLE = os.path.join(os.path.dirname(__file__), "..", "examples", "nightly.jobs")


def order(text):
    return run_order(parse(text))


class OrderTest(unittest.TestCase):
    def test_dependencies_first(self):
        self.assertEqual(order("c 0 b\nb 0 a\na 0\n"), ["a", "b", "c"])

    def test_higher_priority_first(self):
        self.assertEqual(order("low 1\nhigh 5\nmid 3\n"), ["high", "mid", "low"])

    def test_ties_go_in_declared_order(self):
        self.assertEqual(order("zed 2\nalpha 2\nmike 2\n"), ["zed", "alpha", "mike"])

    def test_priority_only_among_ready_jobs(self):
        self.assertEqual(order("urgent 9 slow\nslow 0\nother 1\n"), ["other", "slow", "urgent"])

    def test_cycle_names_every_stuck_job(self):
        with self.assertRaises(CycleError) as ctx:
            order("ok 0\nb 1 a\na 1 b\nafter 5 a,ok\n")
        self.assertEqual(ctx.exception.names, ["a", "after", "b"])
        self.assertEqual(str(ctx.exception), "dependency cycle among: a, after, b")

    def test_example(self):
        with open(EXAMPLE) as f:
            self.assertEqual(run_order(parse(f.read())), [
                "fetch-sources", "build-api", "migrate-staging", "unit-api", "fetch-assets",
                "build-web", "deploy-staging", "smoke-staging", "unit-web", "e2e", "lint", "report"])


if __name__ == "__main__":
    unittest.main()
