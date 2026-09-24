"""python3 -m logstat [--top N] [--errors] LOGFILE..."""
import argparse
import sys

from . import parse, reader, report


def load(paths):
    """(requests, number of malformed lines) over all the files, in order."""
    requests, malformed = [], 0
    for path in paths:
        for line in reader.open_log(path):
            if not line.strip():
                continue
            req = parse.parse_line(line)
            if req is None:
                malformed += 1
            else:
                requests.append(req)
    return requests, malformed


def main(argv=None):
    p = argparse.ArgumentParser(prog="logstat", description="Summarize access logs.")
    p.add_argument("files", nargs="+", help="log files (.gz is fine)")
    p.add_argument("--top", type=int, default=5, help="how many endpoints to list (default 5)")
    p.add_argument("--errors", action="store_true", help="also list every 5xx request")
    args = p.parse_args(argv)
    requests, malformed = load(args.files)
    print(report.summarize(requests, malformed, args.top))
    if args.errors:
        print()
        print("server errors:")
        print(report.server_errors(requests) or "(none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
