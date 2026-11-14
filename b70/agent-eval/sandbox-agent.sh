#!/bin/bash
# Claude Code on a local model, in the agent-eval sandbox (Dockerfile): the task's working copy
# ($QWEN_CWD, set by run.py) is the only host path it sees. Arguments go to claude.
#   QWEN_PROXY=http://proxy:8399 QWEN_MODEL=my-model QWEN_BACKEND=http://odie:8000 \
#   QWEN_KEY_FILE=... run.py --label x -- bash sandbox-agent.sh
# The window is read from the backend (vLLM's max_model_len); 65,536 if it cannot be read.
set -euo pipefail
: "${QWEN_CWD:?}" "${QWEN_PROXY:?}" "${QWEN_MODEL:?}" "${QWEN_BACKEND:?}" "${QWEN_KEY_FILE:?}"
T=${AGENT_TIMEOUT_S:-1200}
if ! [[ "$T" =~ ^[1-9][0-9]{0,5}$ ]] || [ "$T" -gt 86400 ]; then
  echo '{"status":"failed","reason":"agent_timeout_must_be_1_to_86400_seconds"}' >&2; exit 2
fi
CTX=$(curl -s -m 5 "$QWEN_BACKEND/v1/models" | python3 -c 'import sys,json;print(json.load(sys.stdin)["data"][0]["max_model_len"])' 2>/dev/null) || true
case "${CTX:-}" in ''|*[!0-9]*) CTX=65536 ;; esac
NETWORK=${AGENT_DOCKER_NETWORK:-bridge}
[ "$NETWORK" != host ] || { echo '{"status":"failed","reason":"sandbox_host_network_forbidden"}' >&2; exit 2; }
case "$QWEN_PROXY" in http://127.*|https://127.*|http://localhost:*|https://localhost:*)
  echo '{"status":"unavailable","reason":"sandbox_proxy_requires_container_reachable_address"}' >&2; exit 2 ;;
esac
exec docker run --rm -i --network="$NETWORK" --cap-drop=ALL --security-opt=no-new-privileges --pids-limit=256 --user "$(id -u):$(id -g)" \
  -v "$QWEN_CWD":/work -w /work -e HOME=/tmp/home -e CLAUDE_CONFIG_DIR=/tmp/home/.claude \
  -e ANTHROPIC_BASE_URL="$QWEN_PROXY" -v "$QWEN_KEY_FILE":/run/secrets/model-key:ro \
  -e CLAUDE_CODE_SUBAGENT_MODEL="$QWEN_MODEL" -e DISABLE_AUTOUPDATER=1 \
  -e CLAUDE_CODE_MAX_CONTEXT_TOKENS="$CTX" -e CLAUDE_CODE_AUTO_COMPACT_WINDOW="$CTX" -e CLAUDE_AUTOCOMPACT_PCT_OVERRIDE=80 \
  agent-eval:claude sh -c '
    key=$(cat /run/secrets/model-key 2>/dev/null) && [ -n "$key" ] || {
      echo "{\"status\":\"unavailable\",\"reason\":\"model_key_unavailable\"}" >&2; exit 1;
    }
    export ANTHROPIC_AUTH_TOKEN="$key"; unset key; exec "$@"
  ' sh timeout -k 5 "$T" claude --dangerously-skip-permissions --model "$QWEN_MODEL" "$@"
