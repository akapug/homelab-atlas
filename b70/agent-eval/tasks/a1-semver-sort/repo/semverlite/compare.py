"""Ordering of version strings."""
from .version import InvalidVersion, prerelease_key


def _key(text):
    # Cheap key for sorting long tag lists without building a Version per item.
    core, _, pre = text.strip().lstrip("v").partition("-")
    parts = core.split(".")
    if len(parts) != 3:
        raise InvalidVersion(f"not a version: {text!r}")
    return (tuple(parts), prerelease_key(pre))


def compare(a, b):
    """-1 if a < b, 0 if equal, 1 if a > b."""
    ka, kb = _key(a), _key(b)
    return (ka > kb) - (ka < kb)


def sort_versions(items, reverse=False):
    """The version strings in `items`, oldest first (newest first with reverse=True)."""
    return sorted(items, key=_key, reverse=reverse)


def max_version(items):
    """The newest version string in `items` (ValueError if empty)."""
    return max(items, key=_key)
