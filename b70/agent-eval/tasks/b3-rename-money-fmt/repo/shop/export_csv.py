"""CSV export for the accountant: one row per invoice, plain numbers."""
import csv
import io

from . import invoice
from .money import fmt as _amount


def to_csv(invoices):
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(["number", "date", "customer", "currency", "total"])
    for inv in invoices:
        cur = inv["currency"]
        w.writerow([inv["number"], inv["date"], inv["customer"], cur,
                    _amount(invoice.total(inv), cur, False)])
    return buf.getvalue()
