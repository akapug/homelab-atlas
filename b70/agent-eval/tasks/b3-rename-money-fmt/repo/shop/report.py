"""One-line-per-invoice summary report."""
from . import dates, invoice, money


def summary(invoices):
    rows = []
    for inv in sorted(invoices, key=lambda i: (i["date"], i["number"])):
        when = dates.fmt(dates.parse(inv["date"]))
        amount = money.fmt(invoice.total(inv), currency=inv["currency"], pad=14)
        rows.append(f"{inv['number']:<10} {when}  {inv['customer']:<16} {amount}")
    return "\n".join(rows) + "\n"
