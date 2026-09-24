"""Building the quote for one order."""
from . import addons, catalog, pricing


def build_quote(order):
    """The quote for `order`, a dict with "id", "customer", "plan", "seats" and
    optionally "addons" (a list of add-on SKUs)."""
    seats = order["seats"]
    plan = catalog.get_plan(order["plan"])
    lines = plan["lines"]
    addons.attach(lines, order["plan"], order.get("addons", []))
    discount = pricing.apply_volume_discount(lines, seats)
    priced = []
    for line in lines:
        qty = seats if line["per_seat"] else 1
        priced.append({"sku": line["sku"], "desc": line["desc"], "qty": qty,
                       "unit_cents": line["unit_cents"], "amount_cents": qty * line["unit_cents"]})
    return {"id": order["id"], "customer": order["customer"], "plan": order["plan"],
            "seats": seats, "discount_pct": discount, "lines": priced,
            "total_cents": sum(p["amount_cents"] for p in priced)}
