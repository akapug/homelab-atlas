"""python3 -m tabstream [FILE]: print each record of FILE (default: stdin) as a JSON array."""
import json
import sys

from .escapes import DecodeError
from .reader import read_records


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) > 1:
        print("usage: python3 -m tabstream [FILE]", file=sys.stderr)
        return 2
    try:
        stream = open(argv[0], "rb") if argv else sys.stdin.buffer
        with stream:
            for record in read_records(stream):
                print(json.dumps(record, ensure_ascii=False))
    except (OSError, DecodeError) as e:
        print(f"tabstream: {e}", file=sys.stderr)
        return 1
    return 0
