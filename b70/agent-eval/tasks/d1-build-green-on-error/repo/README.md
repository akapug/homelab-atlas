# svc-bundle

Builds the release bundle of service configs that deploy tooling picks up.

    ./build.sh

Each `configs/*.conf` file (`key = value` lines, `#` comments) is turned into
JSON by `tools/gen.py`, checked by `tools/validate.py`, and the results are
packed into `dist/bundle.tar.gz` with a `dist/bundle.sha256` next to it.
Everything a step prints also goes to `dist/build.log`, which CI keeps as an
artifact. CI publishes the bundle when `build.sh` exits 0.

Required keys: `name`, `image`, `port` (1-65535) and `replicas` (1 or more).

## Tests

    tests/run.sh
