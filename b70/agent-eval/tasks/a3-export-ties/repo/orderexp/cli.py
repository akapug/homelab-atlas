"""python3 -m orderexp.cli --db FILE --checkpoint FILE --out FILE [--page-size N] [--max-pages N]"""
import argparse
import sys

from .exporter import export
from .store import Store


def main(argv=None):
    p = argparse.ArgumentParser(prog="orderexp")
    p.add_argument("--db", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--page-size", type=int, default=500)
    p.add_argument("--max-pages", type=int)
    args = p.parse_args(argv)
    store = Store(args.db)
    with open(args.out, "w", newline="") as out:
        n = export(store, out, args.checkpoint, args.page_size, args.max_pages)
    print(f"exported {n} orders to {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
