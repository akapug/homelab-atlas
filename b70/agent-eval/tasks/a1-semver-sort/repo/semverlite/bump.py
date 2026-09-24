"""Next-version arithmetic."""
from .version import Version, format_version, parse


def bump(text, part):
    """Bump the "major", "minor" or "patch" part of a version string.

    Bumping "patch" on a prerelease releases it: 1.2.3-rc.1 -> 1.2.3.
    """
    v = parse(text)
    if part == "major":
        return format_version(Version(v.major + 1, 0, 0, ""))
    if part == "minor":
        return format_version(Version(v.major, v.minor + 1, 0, ""))
    if part == "patch":
        if v.prerelease:
            return format_version(v._replace(prerelease=""))
        return format_version(Version(v.major, v.minor, v.patch + 1, ""))
    raise ValueError(f"unknown part: {part!r}")
