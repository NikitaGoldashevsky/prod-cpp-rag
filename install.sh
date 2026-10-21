#!/usr/bin/env bash
# install.sh — set up PROD_RAG in a local virtualenv.
#
#   ./install.sh                      # creates ./.venv and installs
#   ./install.sh --venv /opt/prod-rag # install somewhere shared
#
# No sudo, no system packages, no compiler: every dependency ships a wheel.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$HERE/.venv"

while [ $# -gt 0 ]; do
  case "$1" in
    --venv) VENV="$2"; shift 2 ;;
    -h|--help) sed -n '2,8p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "error: python3 not found" >&2
  exit 1
fi

VER=$("$PY" -c 'import sys;print("%d.%d"%sys.version_info[:2])')
echo "==> python $VER ($(command -v "$PY"))"
"$PY" - <<'EOF' || { echo "error: Python 3.10+ is required" >&2; exit 1; }
import sys
sys.exit(0 if sys.version_info >= (3, 10) else 1)
EOF

echo "==> creating venv at $VENV"
"$PY" -m venv "$VENV" || {
  echo "error: could not create a venv." >&2
  echo "       On Ubuntu: sudo apt-get install -y python3-venv" >&2
  exit 1
}

"$VENV/bin/pip" install --quiet --upgrade pip
echo "==> installing dependencies"
"$VENV/bin/pip" install --quiet -r "$HERE/requirements.txt"
echo "==> installing prod-rag"
"$VENV/bin/pip" install --quiet -e "$HERE"

echo
"$VENV/bin/prod-rag" doctor || true
cat <<EOF

Done. Add this to your PATH for convenience:

    export PATH="$VENV/bin:\$PATH"

Next:

    prod-rag index add <name> <path/to/submodule>
    prod-rag index status
    prod-rag serve --port 8088            # web UI
    prod-rag mcp                          # MCP server on stdio
EOF
