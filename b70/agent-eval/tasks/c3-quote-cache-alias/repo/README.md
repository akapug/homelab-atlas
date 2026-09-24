# quoter

Turns subscription orders into price quotes for the billing system. The
nightly job runs

    python3 -m quoter batch orders-YYYY-MM-DD.jsonl -o quotes-YYYY-MM-DD.jsonl

Each order is one JSON line:

    {"id": "Q-0922-001", "customer": "Harbor Analytics", "plan": "team", "seats": 12, "addons": ["sso"]}

and each quote one JSON line: the plan's lines, then one line per add-on, each
`[sku, qty, unit_cents, amount_cents]`, and the total. Orders of 50 or more
seats get 10% off every per-seat price, 200 or more 20%.

The price list is in `data/` (`plans/*.json`, `addons.json`).

## Layout

- `quoter/catalog.py` reading the price list
- `quoter/quote.py`   building one quote
- `quoter/addons.py`  add-on lines
- `quoter/pricing.py` volume discounts
- `quoter/render.py`  the billing record
- `quoter/cli.py`     the batch command

## Tests

    ./run_tests.sh
