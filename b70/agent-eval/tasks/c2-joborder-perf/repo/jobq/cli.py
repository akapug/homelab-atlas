"""python3 -m jobq {order,check} JOBFILE"""
import argparse
import sys

from . import order, spec


def main(argv=None):
    p = argparse.ArgumentParser(prog="jobq", description="Order jobs by dependencies and priority.")
    p.add_argument("command", choices=["order", "check"],
                   help="order: print job names in run order; check: only validate")
    p.add_argument("file", help="job file (see README)")
    args = p.parse_args(argv)
    try:
        with open(args.file, encoding="utf-8") as f:
            jobs = spec.parse(f.read())
        names = order.run_order(jobs)
    except (OSError, spec.SpecError, order.CycleError) as err:
        print(f"error: {err}", file=sys.stderr)
        return 1
    if args.command == "check":
        print(f"{len(jobs)} jobs, no cycles")
    else:
        sys.stdout.write("".join(name + "\n" for name in names))
    return 0
