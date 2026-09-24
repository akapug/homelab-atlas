#!/usr/bin/env bash
# Run the unit tests from the repository root.
cd "$(dirname "$0")" && exec python3 -m unittest discover -s tests "$@"
