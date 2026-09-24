import unittest

from quoter import catalog, quote, render
from quoter.addons import AddonError


def order(plan, seats, *addons, id="Q-1"):
    return {"id": id, "customer": "Test Co", "plan": plan, "seats": seats, "addons": list(addons)}


class QuoteTest(unittest.TestCase):
    def setUp(self):
        catalog.clear_cache()

    def test_plan_lines(self):
        q = quote.build_quote(order("team", 10))
        self.assertEqual([(l["sku"], l["qty"], l["amount_cents"]) for l in q["lines"]],
                         [("base-team", 1, 9900), ("seat-team", 10, 15000)])
        self.assertEqual(q["total_cents"], 24900)

    def test_addons_follow_the_plan_lines(self):
        q = quote.build_quote(order("enterprise", 20, "sso", "audit-log"))
        self.assertEqual([l["sku"] for l in q["lines"]][-2:], ["sso", "audit-log"])
        self.assertEqual(q["total_cents"], 49900 + 20 * 2500 + 20000 + 20 * 300 + 4900)

    def test_volume_discount_on_per_seat_lines(self):
        q = quote.build_quote(order("team", 60, "sso", "sandbox"))
        self.assertEqual(q["discount_pct"], 10)
        units = {l["sku"]: l["unit_cents"] for l in q["lines"]}
        self.assertEqual(units, {"base-team": 9900, "seat-team": 1350, "sso": 270, "sandbox": 2500})

    def test_addon_must_be_sold_on_the_plan(self):
        with self.assertRaises(AddonError):
            quote.build_quote(order("starter", 3, "sso"))

    def test_unknown_plan(self):
        with self.assertRaises(catalog.UnknownItem):
            quote.build_quote(order("platinum", 3))


class RenderTest(unittest.TestCase):
    def setUp(self):
        catalog.clear_cache()

    def test_record(self):
        rec = render.to_record(quote.build_quote(order("starter", 2, "sandbox", id="Q-7")))
        self.assertEqual(rec, {
            "quote": "Q-7", "customer": "Test Co", "plan": "starter", "seats": 2, "discount_pct": 0,
            "lines": [["base-starter", 1, 2900, 2900], ["seat-starter", 2, 900, 1800],
                      ["sandbox", 1, 2500, 2500]],
            "total_cents": 7200})

    def test_same_addon_twice_is_refused(self):
        with self.assertRaises(render.RenderError):
            render.to_record(quote.build_quote(order("team", 5, "sandbox", "sandbox")))


if __name__ == "__main__":
    unittest.main()
