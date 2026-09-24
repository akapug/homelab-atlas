#!/usr/bin/env python3
"""gen.py SRC_DIR DEST_DIR: write DEST_DIR/<name>.json for every SRC_DIR/<name>.conf."""
import json
import os
import sys

INT_KEYS = {"port", "replicas"}


def parse(path):
    """The key = value pairs of one .conf file; exits with a message on a bad line."""
    conf = {}
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            key, sep, value = line.partition("=")
            key, value = key.strip(), value.strip()
            if not sep or not key:
                sys.exit(f'gen: {path}:{lineno}: expected "key = value", got {line!r}')
            if key in INT_KEYS:
                try:
                    value = int(value)
                except ValueError:
                    sys.exit(f"gen: {path}:{lineno}: {key} must be a whole number, got {value!r}")
            conf[key] = value
    return conf


def main(src, dest):
    names = sorted(n for n in os.listdir(src) if n.endswith(".conf"))
    for name in names:
        conf = parse(os.path.join(src, name))
        with open(os.path.join(dest, name[:-5] + ".json"), "w") as f:
            json.dump(conf, f, indent=2, sort_keys=True)
            f.write("\n")
    print(f"gen: wrote {len(names)} configs")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("usage: gen.py SRC_DIR DEST_DIR")
    main(*sys.argv[1:])
