"""Formatting of money amounts held as integer minor units."""

SYMBOLS = {"USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥"}
DECIMALS = {"JPY": 0}


def fmt(cents, currency="USD", show_symbol=True, pad=0):
    """Format an amount of minor units for display.

    fmt(123456) -> '$1,234.56'; fmt(-5, 'EUR') -> '-€0.05';
    fmt(1500, 'JPY', False) -> '1,500'. A nonzero pad right-aligns the
    result to that width. Unknown currencies use the code as a prefix.
    """
    places = DECIMALS.get(currency, 2)
    sign = "-" if cents < 0 else ""
    n = abs(cents)
    if places:
        major, minor = divmod(n, 10 ** places)
        body = f"{major:,}.{minor:0{places}d}"
    else:
        body = f"{n:,}"
    if show_symbol:
        body = SYMBOLS.get(currency, currency + " ") + body
    return (sign + body).rjust(pad)
