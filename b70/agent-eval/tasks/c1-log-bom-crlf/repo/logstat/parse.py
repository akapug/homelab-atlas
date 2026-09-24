"""One access-log line -> Request.

The format is one request per line, fields separated by single spaces:

    2026-09-14T08:15:02Z GET /api/items?page=2 200 35ms
"""
import re
from dataclasses import dataclass

LINE = re.compile(
    r"(?P<ts>\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ) (?P<method>[A-Z]+) (?P<path>\S+) "
    r"(?P<status>\d{3}) (?P<ms>\d+)ms$"
)


@dataclass(frozen=True)
class Request:
    ts: str
    method: str
    path: str
    status: int
    ms: int

    @property
    def endpoint(self):
        """The path without its query string."""
        return self.path.split("?", 1)[0]


def parse_line(line):
    """The Request on this line, or None if the line is not a well-formed request."""
    m = LINE.match(line)
    if not m:
        return None
    return Request(m["ts"], m["method"], m["path"], int(m["status"]), int(m["ms"]))
