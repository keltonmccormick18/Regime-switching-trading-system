#!/usr/bin/env bash
# ─────────────────────────────────────────────
# Quant Trading System — API launcher
# Usage: ./start_api.sh [--port 8000] [--reload] [--ssl] [--venv /path/to/venv]
# ─────────────────────────────────────────────
set -e

# Resolve venv: explicit VENV env var → .venv in project root → ~/tf_env → system Python
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -z "$VENV" ]]; then
  if   [[ -f "$SCRIPT_DIR/.venv/bin/activate" ]];  then VENV="$SCRIPT_DIR/.venv"
  elif [[ -f "$HOME/tf_env/bin/activate" ]];        then VENV="$HOME/tf_env"
  else VENV=""
  fi
fi
PORT=8000
RELOAD=""
SSL=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --port)    PORT="$2";  shift 2 ;;
    --reload)  RELOAD="--reload"; shift ;;
    --ssl)     SSL=true;   shift ;;
    --venv)    VENV="$2";  shift 2 ;;
    *)         shift ;;
  esac
done

# Activate virtual environment if found, otherwise bail with a clear message
if [[ -n "$VENV" && -f "$VENV/bin/activate" ]]; then
  source "$VENV/bin/activate"
else
  echo "  Error: no virtual environment found."
  echo "  Create one with:"
  echo "    python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt"
  echo "  Or point to an existing one:"
  echo "    VENV=/path/to/venv ./start_api.sh"
  exit 1
fi

export PYTHONPATH="$SCRIPT_DIR"

KEY="$SCRIPT_DIR/certs/server.key"
CRT="$SCRIPT_DIR/certs/server.crt"

# Auto-generate certs if --ssl requested and they don't exist yet
if $SSL; then
  bash "$SCRIPT_DIR/make_certs.sh"
  SSL_FLAGS="--ssl-keyfile \"$KEY\" --ssl-certfile \"$CRT\""
  SCHEME="https"
else
  SSL_FLAGS=""
  SCHEME="http"
fi

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Quant Trading System — API"
echo "  $SCHEME://localhost:$PORT"
echo "  Docs: $SCHEME://localhost:$PORT/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Postgres / Redis are optional."
echo "  Routes that need them return 503;"
echo "  /predict, /backtest, /regime work without them."
echo ""

# Force PyTorch to CPU on Apple Silicon — MPS backend has a known SIGSEGV
# in Metal GPU streams triggered by causal attention + LSTM ops in these models.
export PYTORCH_ENABLE_MPS_FALLBACK=1

cd "$SCRIPT_DIR"

if $SSL; then
  uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info \
    --ssl-keyfile "$KEY" \
    --ssl-certfile "$CRT" \
    $RELOAD
else
  uvicorn src.api.main:app \
    --host 0.0.0.0 \
    --port "$PORT" \
    --log-level info \
    $RELOAD
fi
