"""Reading log files, plain or gzip-compressed, as lines of text."""
import gzip


def open_log(path):
    """Yield the lines of a log file without their line endings.

    The file is read as bytes and decoded one line at a time, so a corrupt
    byte costs one replacement character instead of the whole file.
    """
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rb") as f:
        for raw in f:
            yield raw.decode("utf-8", errors="replace").rstrip("\n")
