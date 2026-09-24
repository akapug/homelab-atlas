"""Quotes -> the JSON records the billing system imports."""


class RenderError(ValueError):
    pass


def to_record(quote):
    """The billing record for `quote`.

    Billing rejects a quote with two lines for the same SKU, so we refuse to write one."""
    seen = set()
    for line in quote["lines"]:
        sku = line["sku"]
        if sku in seen:
            raise RenderError(f"quote {quote['id']}: duplicate line for {sku}")
        seen.add(sku)
    return {
        "quote": quote["id"],
        "customer": quote["customer"],
        "plan": quote["plan"],
        "seats": quote["seats"],
        "discount_pct": quote["discount_pct"],
        "lines": [[l["sku"], l["qty"], l["unit_cents"], l["amount_cents"]] for l in quote["lines"]],
        "total_cents": quote["total_cents"],
    }
