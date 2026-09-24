# semverlite

Small helpers for `MAJOR.MINOR.PATCH[-PRERELEASE]` version strings, used by the
release scripts to pick the newest tag and to bump versions.

```python
from semverlite import compare, sort_versions, max_version, bump

compare("1.2.3", "1.2.4")          # -1
sort_versions(["1.2.0", "1.0.0"])  # ["1.0.0", "1.2.0"]
max_version(["v1.0.0", "v1.1.0"])  # "v1.1.0"
bump("1.2.3", "minor")             # "1.3.0"
```

Command line: `python3 -m semverlite max 1.0.0 1.1.0`, `python3 -m semverlite sort ...`,
`python3 -m semverlite bump 1.2.3 patch`.

Run the tests with `make test` (or `python3 -m unittest discover -s tests -v`).
