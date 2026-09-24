import csv
import io
import os
import tempfile


def stamp(second):
    """An updated_at value `second` seconds after 2026-08-01T12:00:00Z (second < 3600)."""
    return f"2026-08-01T12:{second // 60:02d}:{second % 60:02d}Z"


def checkpoint_path(test):
    """A checkpoint file path in a temp dir that is removed after `test`."""
    d = tempfile.TemporaryDirectory()
    test.addCleanup(d.cleanup)
    return os.path.join(d.name, "state.json")


def run(export, store, cp, **kw):
    """One export run: (returned count, parsed CSV rows without the header)."""
    buf = io.StringIO()
    n = export(store, buf, cp, **kw)
    rows = list(csv.reader(io.StringIO(buf.getvalue())))
    return n, rows[1:]
