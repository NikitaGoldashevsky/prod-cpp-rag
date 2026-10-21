#!/usr/bin/env bash
# Serve one PROD_RAG instance for the whole team.
#
#   ./scripts/serve-shared.sh                  # MCP on 8765 + web UI on 8088
#   PORT_MCP=9000 PORT_WEB=9001 ./scripts/serve-shared.sh
#   PROD_RAG_LLM_BASE_URL=http://192.168.10.46:8081 ./scripts/serve-shared.sh
#
# Both bind to all interfaces. This exposes your indexed source code to anyone
# who can reach the machine, so run it on a host your team already trusts and
# firewall the ports. The web UI accepts a shared token via PROD_RAG_TOKEN.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="$HERE/.venv/bin/prod-rag"
MCP_PORT="${PORT_MCP:-8765}"
WEB_PORT="${PORT_WEB:-8088}"

[ -x "$BIN" ] || { echo "run ./install.sh first" >&2; exit 1; }

"$BIN" mcp --transport http --host 0.0.0.0 --port "$MCP_PORT" &
MCP_PID=$!
trap 'kill "$MCP_PID" 2>/dev/null || true' EXIT INT TERM

echo "MCP  http://0.0.0.0:$MCP_PORT/mcp"
echo "Web  http://0.0.0.0:$WEB_PORT/"
"$BIN" serve --host 0.0.0.0 --port "$WEB_PORT"
