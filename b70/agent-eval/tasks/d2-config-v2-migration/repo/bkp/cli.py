"""bkp plan FILE | bkp show FILE | bkp fmt FILE..."""
import argparse
import sys

from . import config
from .plan import plan


def cmd_plan(args):
    print("\n".join(plan(config.load(args.file))))


def cmd_show(args):
    job = config.load(args.file)
    n = len(job["exclude"])
    print(f"name:    {job['name']}")
    print(f"source:  {job['source_dir']} ({n} exclude{'' if n == 1 else 's'})")
    print(f"dest:    {job['dest']}")
    print(f"keep:    {job['keep']}")


def cmd_fmt(args):
    jobs = [(path, config.load(path)) for path in args.files]
    for path, job in jobs:
        config.save(job, path)


def main(argv=None):
    p = argparse.ArgumentParser(prog="bkp")
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("plan", help="print the commands a job runs")
    s.add_argument("file")
    s.set_defaults(func=cmd_plan)
    s = sub.add_parser("show", help="summarize a job")
    s.add_argument("file")
    s.set_defaults(func=cmd_show)
    s = sub.add_parser("fmt", help="rewrite job files in the canonical form")
    s.add_argument("files", nargs="+")
    s.set_defaults(func=cmd_fmt)
    args = p.parse_args(argv)
    try:
        args.func(args)
    except config.ConfigError as e:
        print(f"bkp: {e}", file=sys.stderr)
        return 1
    return 0
