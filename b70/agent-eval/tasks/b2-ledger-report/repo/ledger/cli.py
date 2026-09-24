"""Command-line interface: python3 -m ledger [--file PATH] COMMAND ..."""
import argparse
import datetime
import os

from . import money, store

DEFAULT_FILE = "ledger.json"


def iso_date(text):
    """argparse type: a YYYY-MM-DD date, returned as an ISO string."""
    try:
        return datetime.date.fromisoformat(text).isoformat()
    except ValueError:
        raise argparse.ArgumentTypeError(f"invalid date {text!r} (want YYYY-MM-DD)") from None


def amount(text):
    """argparse type: a money amount, returned as integer cents."""
    try:
        return money.parse_amount(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def cmd_add(args):
    entry = {
        "date": args.date or datetime.date.today().isoformat(),
        "amount": args.amount,
        "category": args.category,
        "note": args.note,
    }
    store.append(args.file, entry)
    print(f"added {money.format_cents(args.amount)} {args.category}")


def cmd_list(args):
    for e in sorted(store.load(args.file), key=lambda e: e["date"]):
        if args.category and e["category"] != args.category:
            continue
        print("\t".join([e["date"], e["category"], money.format_cents(e["amount"]), e["note"]]))


def cmd_total(args):
    total = sum(e["amount"] for e in store.load(args.file))
    print(f"TOTAL\t{money.format_cents(total)}")


def build_parser():
    p = argparse.ArgumentParser(prog="ledger", description="A tiny personal expense ledger.")
    p.add_argument("--file", default=os.environ.get("LEDGER_FILE", DEFAULT_FILE),
                   help="ledger JSON file (default: $LEDGER_FILE or ledger.json)")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("add", help="record an expense (negative for a refund)")
    a.add_argument("amount", type=amount)
    a.add_argument("category")
    a.add_argument("--date", type=iso_date, help="YYYY-MM-DD (default: today)")
    a.add_argument("--note", default="")
    a.set_defaults(func=cmd_add)

    ls = sub.add_parser("list", help="list entries, oldest first")
    ls.add_argument("--category")
    ls.set_defaults(func=cmd_list)

    t = sub.add_parser("total", help="sum of all entries")
    t.set_defaults(func=cmd_total)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0
