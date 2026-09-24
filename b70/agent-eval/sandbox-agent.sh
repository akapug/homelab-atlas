#!/bin/bash
# Claude Code on a local model, in the agent-eval sandbox (Dockerfile): the task's working copy
# ($QWEN_CWD, set by run.py) is the only host path it sees. Arguments go to claude.
#   QWEN_PROXY=http://127.0.0.1:8411 QWEN_MODEL=qwen-local QWEN_BACKEND=http://127.0.0.1:8081 \
#   QWEN_KEY_FILE=... run.py --label x -- bash sandbox-agent.sh
# The window is read from the backend (vLLM's max_model_len), as local-agent.sh does.
set -u
: "${QWEN_CWD:?}" "${QWEN_PROXY:?}" "${QWEN_MODEL:?}" "${QWEN_BACKEND:?}" "${QWEN_KEY_FILE:?}"
CTX=$(curl -s -m 5 "$QWEN_BACKEND/v1/models" | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["max_model_len"])' 2>/dev/null)
case "${CTX:-}" in ''|*[!0-9]*) CTX=65536 ;; esac
exec docker run --rm -i --network=host --user "$(id -u):$(id -g)" \
  -v "$QWEN_CWD":/work -w /work -e HOME=/tmp/home -e CLAUDE_CONFIG_DIR=/tmp/home/.claude \
  -e ANTHROPIC_BASE_URL="$QWEN_PROXY" -e ANTHROPIC_AUTH_TOKEN="$(cat "$QWEN_KEY_FILE")" \
  -e CLAUDE_CODE_SUBAGENT_MODEL="$QWEN_MODEL" -e DISABLE_AUTOUPDATER=1 \
  -e CLAUDE_CODE_MAX_CONTEXT_TOKENS="$CTX" -e CLAUDE_CODE_AUTO_COMPACT_WINDOW="$CTX" -e CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=80 \
  agent-eval:claude claude --dangerously-skip-permissions --model "$QWEN_MODEL" "$@"
