"""Date helpers. Invoices store dates as YYYY-MM-DD strings."""
import datetime

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def parse(text):
    return datetime.date.fromisoformat(text)


def fmt(d):
    """date(2026, 9, 1) -> '01 Sep 2026' (locale-independent)."""
    return f"{d.day:02d} {MONTHS[d.month - 1]} {d.year}"
