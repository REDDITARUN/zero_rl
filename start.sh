#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

GREEN='\033[0;32m'
CYAN='\033[0;36m'
RED='\033[0;31m'
DIM='\033[2m'
NC='\033[0m'

cleanup() {
    echo ""
    echo -e "${DIM}Shutting down...${NC}"
    kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    wait $BACKEND_PID $FRONTEND_PID 2>/dev/null || true
    echo -e "${GREEN}Stopped.${NC}"
}
trap cleanup EXIT INT TERM

# ── Preflight ───────────────────────────────────────────────────────────────
[ -d ".venv" ]             || { echo -e "${RED}No .venv found. Run ./setup.sh first.${NC}"; exit 1; }
[ -d "frontend/node_modules" ] || { echo -e "${RED}No node_modules. Run ./setup.sh first.${NC}"; exit 1; }

source .venv/bin/activate

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_HOST="${FRONTEND_HOST:-127.0.0.1}"
FRONTEND_PORT="${FRONTEND_PORT:-3000}"

echo ""
echo -e "${CYAN}╔══════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║          zero-rl  —  starting             ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════════╝${NC}"
echo ""

# ── Backend ─────────────────────────────────────────────────────────────────
echo -e "${DIM}Starting backend on ${BACKEND_HOST}:${BACKEND_PORT}...${NC}"
.venv/bin/uvicorn backend.main:app \
    --reload --reload-dir backend --reload-dir core \
    --host "$BACKEND_HOST" --port "$BACKEND_PORT" \
    2>&1 | sed "s/^/  ${DIM}[api]${NC} /" &
BACKEND_PID=$!

sleep 2

# ── Frontend ────────────────────────────────────────────────────────────────
echo -e "${DIM}Starting frontend on ${FRONTEND_HOST}:${FRONTEND_PORT}...${NC}"
cd frontend
npm run dev -- -H "$FRONTEND_HOST" -p "$FRONTEND_PORT" \
    2>&1 | sed "s/^/  ${DIM}[web]${NC} /" &
FRONTEND_PID=$!
cd "$ROOT_DIR"

sleep 3

echo ""
echo -e "${GREEN}╔══════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Ready!                                   ║${NC}"
echo -e "${GREEN}║                                           ║${NC}"
echo -e "${GREEN}║  Frontend: http://${FRONTEND_HOST}:${FRONTEND_PORT}       ║${NC}"
echo -e "${GREEN}║  Backend:  http://${BACKEND_HOST}:${BACKEND_PORT}       ║${NC}"
echo -e "${GREEN}║                                           ║${NC}"
echo -e "${GREEN}║  Press Ctrl+C to stop                     ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════════╝${NC}"
echo ""

wait
