#!/usr/bin/env python3
"""validate.py FILE: exit 1 with a message if a generated service config is not deployable."""
import json
import sys


def problems(conf):
    for key in ("name", "image"):
        if not isinstance(conf.get(key), str) or not conf[key]:
            yield f"{key} is required"
    port = conf.get("port")
    if not isinstance(port, int) or not 1 <= port <= 65535:
        yield f"port must be 1-65535, got {port!r}"
    replicas = conf.get("replicas")
    if not isinstance(replicas, int) or replicas < 1:
        yield f"replicas must be 1 or more, got {replicas!r}"


def main(path):
    try:
        with open(path) as f:
            conf = json.load(f)
    except (OSError, ValueError) as e:
        sys.exit(f"validate: {path}: {e}")
    found = list(problems(conf))
    for p in found:
        print(f"validate: {path}: {p}", file=sys.stderr)
    return 1 if found else 0


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: validate.py FILE")
    sys.exit(main(sys.argv[1]))
