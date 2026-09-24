# ledger

A tiny personal expense ledger. Entries live in a JSON file (default
`ledger.json`, override with `--file PATH` or `LEDGER_FILE`).

    python3 -m ledger add 12.50 food --date 2026-09-01 --note lunch
    python3 -m ledger add -3.50 food              # a refund
    python3 -m ledger list [--category food]
    python3 -m ledger total

Output is tab-separated so it pipes cleanly into `cut`/`sort`/`awk`.

## Tests

    python3 -m unittest
