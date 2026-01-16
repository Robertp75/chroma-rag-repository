#!/usr/bin/env bash
set -e

# Start tailscaled
tailscaled --state=/tmp/tailscaled.state --socket=/tmp/tailscaled.sock &
sleep 2

if [ -z "$TAILSCALE_AUTHKEY" ]; then
  echo "TAILSCALE_AUTHKEY not set"
  exit 1
fi

# Join your tailnet
tailscale --socket=/tmp/tailscaled.sock up \
  --authkey="$TAILSCALE_AUTHKEY" \
  --hostname="render-chroma-rag-proxy" \
  --accept-dns=false

# Start API (Render sets PORT)
uvicorn main:app --host 0.0.0.0 --port "${PORT:-10000}"
