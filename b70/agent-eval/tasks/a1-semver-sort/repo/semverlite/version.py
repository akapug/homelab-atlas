"""Parsing and formatting of MAJOR.MINOR.PATCH[-PRERELEASE] strings."""
import re
from collections import namedtuple

Version = namedtuple("Version", "major minor patch prerelease")

_PATTERN = re.compile(r"^v?(\d+)\.(\d+)\.(\d+)(?:-([0-9A-Za-z]+(?:\.[0-9A-Za-z]+)*))?$")


class InvalidVersion(ValueError):
    pass


def parse(text):
    """Parse "1.2.3", "v1.2.3" or "1.2.3-rc.1" into a Version."""
    m = _PATTERN.match(text.strip())
    if not m:
        raise InvalidVersion(f"not a version: {text!r}")
    major, minor, patch, pre = m.groups()
    return Version(int(major), int(minor), int(patch), pre or "")


def format_version(v):
    core = f"{v.major}.{v.minor}.{v.patch}"
    return f"{core}-{v.prerelease}" if v.prerelease else core


def prerelease_key(pre):
    """Sort key for a prerelease tag. A release (no tag) sorts after any prerelease;
    numeric identifiers compare as numbers and sort before alphanumeric ones."""
    if not pre:
        return (1,)
    idents = tuple((0, int(i), "") if i.isdigit() else (1, 0, i) for i in pre.split("."))
    return (0, idents)
