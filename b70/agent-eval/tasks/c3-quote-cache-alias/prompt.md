Last night's quote batch died partway through and billing got nothing after order 36; the input was samples/orders-2026-09-22.jsonl and the log is below. Don't just make the error go away: find out why it happened and fix the cause. A quote also must not depend on what else is in the batch: each one has to come out exactly as it would if that order were run on its own.

```
2026-09-22T02:14:07Z INFO  quoter.batch: 48 orders from orders-2026-09-22.jsonl
2026-09-22T02:14:07Z INFO  quoter.batch: wrote quote Q-0922-001 (Harbor Analytics, team)
2026-09-22T02:14:07Z INFO  quoter.batch: wrote quote Q-0922-002 (Pine & Co, starter)
...
2026-09-22T02:14:08Z INFO  quoter.batch: wrote quote Q-0922-035 (Polar Seafood, team)
2026-09-22T02:14:08Z INFO  quoter.batch: wrote quote Q-0922-036 (Cedar Accounting, team)
2026-09-22T02:14:08Z ERROR quoter.batch: batch aborted at order 37 (Brightline Dental)
Traceback (most recent call last):
  File "/srv/quoter/quoter/cli.py", line 26, in run_batch
    out.write(json.dumps(render.to_record(q)) + "\n")
                         ~~~~~~~~~~~~~~~~^^^
  File "/srv/quoter/quoter/render.py", line 16, in to_record
    raise RenderError(f"quote {quote['id']}: duplicate line for {sku}")
quoter.render.RenderError: quote Q-0922-037: duplicate line for sso
```
