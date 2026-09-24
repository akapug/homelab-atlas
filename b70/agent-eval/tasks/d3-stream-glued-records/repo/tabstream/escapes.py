"""Escaping of single fields and records (the text format of PostgreSQL's COPY)."""

ESCAPES = {"\\": "\\", "t": "\t", "n": "\n", "r": "\r"}
REVERSE = {v: "\\" + k for k, v in ESCAPES.items()}
NULL = "\\N"


class DecodeError(ValueError):
    """Text that is not a valid record."""


def escape(value):
    """One field as it goes on the wire."""
    if value is None:
        return NULL
    return "".join(REVERSE.get(c, c) for c in value)


def unescape(raw):
    """One field as it came off the wire (None for NULL)."""
    if raw == NULL:
        return None
    out = []
    i = 0
    while i < len(raw):
        c = raw[i]
        if c != "\\":
            out.append(c)
            i += 1
            continue
        if i + 1 == len(raw):
            raise DecodeError(f"dangling backslash at the end of {raw!r}")
        if raw[i + 1] not in ESCAPES:
            raise DecodeError(f"unknown escape \\{raw[i + 1]} in {raw!r}")
        out.append(ESCAPES[raw[i + 1]])
        i += 2
    return "".join(out)


def encode_record(fields):
    """One record, with its LF."""
    return "\t".join(escape(f) for f in fields) + "\n"


def decode_record(line):
    """The fields of one record, given without its LF. A TAB in a field is always escaped, so
    every bare TAB separates two fields."""
    return [unescape(raw) for raw in line.split("\t")]
