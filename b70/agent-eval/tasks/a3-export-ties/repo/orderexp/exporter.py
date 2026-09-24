"""Write orders changed since the last run as CSV."""
import csv

from . import checkpoint

HEADER = ["id", "ref", "updated_at", "total"]


def money(cents):
    return f"{cents // 100}.{cents % 100:02d}"


def export(store, out, checkpoint_path, page_size=500, max_pages=None):
    """Write orders changed since the last run to the text file `out` as CSV, advancing the
    checkpoint after every page. Stops after `max_pages` pages if given; the next run continues
    from there. Returns the number of orders written."""
    writer = csv.writer(out)
    writer.writerow(HEADER)
    position = checkpoint.load(checkpoint_path)
    written = pages = 0
    while max_pages is None or pages < max_pages:
        rows = store.changed_since(position, page_size)
        if not rows:
            break
        for order_id, ref, updated_at, cents in rows:
            writer.writerow([order_id, ref, updated_at, money(cents)])
        written += len(rows)
        pages += 1
        position = rows[-1][2]
        checkpoint.save(checkpoint_path, position)
    return written
