"""Invoices: loading, totals, and the printable form."""
import json

from . import dates, money


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def line_total(line):
    return line["qty"] * line["unit"]


def total(inv):
    return sum(line_total(line) for line in inv["lines"])


def render(inv):
    """The printable invoice as one string."""
    cur = inv["currency"]
    out = [
        f"INVOICE {inv['number']}",
        f"Date:    {dates.fmt(dates.parse(inv['date']))}",
        f"Bill to: {inv['customer']}",
        "",
    ]
    for line in inv["lines"]:
        unit = money.fmt(line["unit"], cur, True, 12)
        amount = money.fmt(line_total(line), cur, True, 12)
        out.append(f"{line['desc']:<20} {line['qty']:>4} x {unit} {amount}")
    out.append("")
    out.append(f"{'TOTAL':<20} {'':>4}   {'':>12} {money.fmt(total(inv), cur, pad=12)}")
    return "\n".join(out) + "\n"
