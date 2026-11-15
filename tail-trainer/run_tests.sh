#!/usr/bin/env bash
# The tool's own tests in a throwaway venv with CPU torch: bash run_tests.sh  (exit code = the tests')
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
V=${TEST_VENV:-$HERE/.tv}
[ -x "$V/bin/python" ] || python3 -m venv "$V" >/dev/null || exit 2
"$V/bin/pip" install -q torch numpy gguf --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple >/dev/null 2>&1 || { echo "pip install failed"; exit 2; }
cd "$HERE" && CUDA_VISIBLE_DEVICES= "$V/bin/python" -m unittest -v test_tail_trainer 2>&1 | tail -40
rc=${PIPESTATUS[0]}
echo "rc=$rc"
exit "$rc"
