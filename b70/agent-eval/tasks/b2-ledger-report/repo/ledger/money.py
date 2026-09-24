"""Money is kept as integer cents everywhere; convert only at the edges."""
from decimal import Decimal, InvalidOperation


def parse_amount(text):
    """'12.5' -> 1250, '-3' -> -300. Raises ValueError on junk or sub-cent input."""
    try:
        d = Decimal(text)
    except InvalidOperation:
        raise ValueError(f"not an amount: {text!r}") from None
    if not d.is_finite() or d != d.quantize(Decimal("0.01")):
        raise ValueError(f"not an amount: {text!r}")
    return int(d * 100)


def format_cents(cents):
    """1250 -> '12.50', -5 -> '-0.05'."""
    sign = "-" if cents < 0 else ""
    cents = abs(cents)
    return f"{sign}{cents // 100}.{cents % 100:02d}"
