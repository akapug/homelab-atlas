# shop

Invoice tooling for a small shop. Invoices are JSON files; amounts are
integer minor units (cents, or whole yen for JPY).

    python3 -m shop invoice examples/inv-0042.json   # printable invoice
    python3 -m shop report examples/*.json           # one line per invoice
    python3 -m shop csv examples/*.json              # CSV for the accountant

Modules:

- `shop/money.py`   formatting money amounts
- `shop/dates.py`   parsing and formatting dates
- `shop/invoice.py` loading invoices, totals, the printable invoice
- `shop/report.py`  the summary report
- `shop/export_csv.py` the CSV export
- `shop/cli.py`     the command line

## Tests

    python3 -m unittest
