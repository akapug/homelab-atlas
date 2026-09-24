"""Volume discounts on per-seat prices."""

# (at least this many seats, percent off every per-seat price), largest first
TIERS = [(200, 20), (50, 10)]


def volume_discount(seats):
    """Percent off per-seat prices for an order of `seats` seats."""
    for minimum, pct in TIERS:
        if seats >= minimum:
            return pct
    return 0


def apply_volume_discount(lines, seats):
    """Reduce the unit price of every per-seat line by the volume discount; return the percent."""
    pct = volume_discount(seats)
    if pct:
        for line in lines:
            if line["per_seat"]:
                line["unit_cents"] = line["unit_cents"] * (100 - pct) // 100
    return pct
