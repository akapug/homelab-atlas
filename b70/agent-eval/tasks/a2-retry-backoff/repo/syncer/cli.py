"""python3 -m syncer.cli [--config FILE] show-config"""
import argparse
import sys

from .config import ConfigError, load_config


def show_config(cfg):
    return "\n".join([
        f"sync.endpoint = {cfg.endpoint}",
        f"sync.timeout = {cfg.timeout:g}",
        f"retry.max_attempts = {cfg.retry.max_attempts}",
        f"retry.delay = {cfg.retry.delay:g}",
    ])


def main(argv=None):
    parser = argparse.ArgumentParser(prog="syncer")
    parser.add_argument("--config", help="INI file with [sync] and [retry] sections")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("show-config", help="print the effective settings")
    args = parser.parse_args(argv)
    try:
        cfg = load_config(args.config)
    except ConfigError as err:
        print(f"config error: {err}", file=sys.stderr)
        return 2
    if args.cmd == "show-config":
        print(show_config(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
