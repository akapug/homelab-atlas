"""python3 -m semverlite {max|sort|bump} ..."""
import sys

from . import bump, max_version, sort_versions


def main(argv):
    if len(argv) < 2 or argv[0] not in ("max", "sort", "bump"):
        print("usage: semverlite max|sort VERSION... | bump VERSION major|minor|patch", file=sys.stderr)
        return 2
    cmd, args = argv[0], argv[1:]
    if cmd == "max":
        print(max_version(args))
    elif cmd == "sort":
        print("\n".join(sort_versions(args)))
    else:
        print(bump(args[0], args[1]))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
