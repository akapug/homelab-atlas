# orderexp

Incremental CSV export of the `orders` table for finance. Each run writes the
orders changed since the previous run and records where it stopped in a small
JSON checkpoint file, so the next run picks up from there. Runs are capped at
`--max-pages` pages so the nightly job stays short; the rest goes out the next
night.

    python3 -m orderexp.cli --db orders.db --checkpoint export-state.json \
        --out exports/orders-$(date +%F).csv --page-size 500 --max-pages 20

`updated_at` is an ISO-8601 UTC timestamp with second precision
(`2026-08-01T12:00:00Z`).

## Tests

    make test
