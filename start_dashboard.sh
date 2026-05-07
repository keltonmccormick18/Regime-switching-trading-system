#!/usr/bin/env bash
# ─────────────────────────────────────────────
# Quant Trading System — React dashboard
# Assumes API is running on localhost:8000
# Usage: ./start_dashboard.sh [--no-ssl]
# ─────────────────────────────────────────────
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FRONTEND="$SCRIPT_DIR/dashboard/frontend"
SSL=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --no-ssl) SSL=false; shift ;;
    *)        shift ;;
  esac
done

# Generate TLS certs if requested (skipped automatically if already valid)
if $SSL; then
  bash "$SCRIPT_DIR/make_certs.sh"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Quant Trading Dashboard"
if $SSL && [[ -f "$SCRIPT_DIR/certs/server.crt" ]]; then
  echo "  https://localhost:5173"
else
  echo "  http://localhost:5173"
fi
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

cd "$FRONTEND"

if [ ! -d "node_modules" ]; then
  echo "  Installing npm dependencies..."
  npm install
fi

npm run dev
