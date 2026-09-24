# The agent-eval sandbox: Claude Code with git and Python, and nothing of the host's but the task's
# working copy (mounted at /work by sandbox-agent.sh). The model servers stay reachable over the
# host network; the host's files and systemd units do not.
FROM node:22-bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends git python3 ca-certificates procps \
    && rm -rf /var/lib/apt/lists/* && npm install -g @anthropic-ai/claude-code && npm cache clean --force
