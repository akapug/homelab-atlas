# logstat

Summarizes HTTP access logs from our services: request counts, status classes,
the busiest endpoints and their latency, and optionally every 5xx request.

    python3 -m logstat tests/data/app-2026-09-13.log
    python3 -m logstat --top 3 --errors /var/log/app/access-*.log.gz

Every service writes the same line format, one request per line:

    2026-09-14T08:15:02Z GET /api/items?page=2 200 35ms

Lines that do not match it are counted as malformed and otherwise ignored.
Rotated logs are gzip-compressed; logstat reads them directly.

## Layout

- `logstat/reader.py` opening plain and .gz logs
- `logstat/parse.py`  the line format
- `logstat/report.py` the summaries
- `logstat/cli.py`    the command line

## Tests

    ./run_tests.sh
