# syncer

Pushes local changes to the inventory API. Flaky network calls are retried
according to the `[retry]` settings.

## Configuration

Settings come from built-in defaults, then the config file given with
`--config`, then environment variables (highest priority).

| file key              | env var                   | default                     |
|-----------------------|---------------------------|-----------------------------|
| `[sync] endpoint`     | `SYNC_ENDPOINT`           | `http://localhost:8080/api` |
| `[sync] timeout`      | `SYNC_TIMEOUT`            | `10` (seconds)              |
| `[retry] max_attempts`| `SYNC_RETRY_MAX_ATTEMPTS` | `4`                         |
| `[retry] delay`       | `SYNC_RETRY_DELAY`        | `2` (seconds between tries) |

See `settings.example.ini`. Print the effective settings with:

    python3 -m syncer.cli --config settings.example.ini show-config

## Tests

    ./run_tests.sh
