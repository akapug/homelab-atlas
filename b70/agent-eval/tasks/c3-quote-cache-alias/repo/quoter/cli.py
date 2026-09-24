"""python3 -m quoter batch ORDERS.jsonl [-o QUOTES.jsonl]

Quotes every order in ORDERS.jsonl (one JSON object per line) and writes one
billing record per line. Stops at the first order that cannot be quoted.
"""
import argparse
import json
import logging
import os
import sys
import time

from . import quote, render

log = logging.getLogger("quoter.batch")


def run_batch(orders_path, out):
    """Write a record for each order to `out`; 0 if every order was quoted, 1 if the batch stopped."""
    with open(orders_path, encoding="utf-8") as f:
        orders = [json.loads(line) for line in f if line.strip()]
    log.info("%d orders from %s", len(orders), os.path.basename(orders_path))
    for n, order in enumerate(orders, 1):
        try:
            q = quote.build_quote(order)
            out.write(json.dumps(render.to_record(q)) + "\n")
        except Exception:
            log.exception("batch aborted at order %d (%s)", n, order.get("customer"))
            return 1
        log.info("wrote quote %s (%s, %s)", q["id"], order["customer"], order["plan"])
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="quoter")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("batch", help="quote a file of orders")
    b.add_argument("orders", help="JSON lines, one order per line")
    b.add_argument("-o", "--out", help="where to write the records (default: stdout)")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, stream=sys.stderr, datefmt="%Y-%m-%dT%H:%M:%SZ",
                        format="%(asctime)s %(levelname)-5s %(name)s: %(message)s")
    logging.Formatter.converter = time.gmtime
    if not args.out:
        return run_batch(args.orders, sys.stdout)
    with open(args.out, "w", encoding="utf-8") as out:
        return run_batch(args.orders, out)
