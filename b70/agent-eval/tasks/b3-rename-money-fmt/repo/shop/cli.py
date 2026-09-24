"""python3 -m shop {invoice,report,csv} FILE..."""
import argparse
import sys

from . import export_csv, invoice, report


def main(argv=None):
    p = argparse.ArgumentParser(prog="shop")
    p.add_argument("command", choices=["invoice", "report", "csv"])
    p.add_argument("files", nargs="+")
    args = p.parse_args(argv)
    invoices = [invoice.load(f) for f in args.files]
    if args.command == "invoice":
        sys.stdout.write("\n".join(invoice.render(i) for i in invoices))
    elif args.command == "report":
        sys.stdout.write(report.summary(invoices))
    else:
        sys.stdout.write(export_csv.to_csv(invoices))
    return 0
